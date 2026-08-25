#!/usr/bin/env python3
"""
Receptor architecture analysis -- goes beyond Claim 1's per-receptor,
per-metric univariate tests (15 receptors x 2 controllability metrics,
treated independently) to characterize the MULTIVARIATE structure of the
receptor fingerprint itself, motivated directly by the Claim 4B finding
that "7 significant receptors" was substantially one collinear cluster
(r=0.81-0.98 among 5 of them at n=8) -- i.e. univariate testing was
missing/mismodeling real shared structure in the receptor data. Three
layered analyses, each with its own FDR family, all exploratory/post hoc
(not one of the 5 pre-specified claims):

  A. FINGERPRINT SIMILARITY STRUCTURE. Treats each of the 8 hOc
     region x hemisphere units' 15-receptor profile as a vector. Computes
     inter-unit receptor-fingerprint similarity (Pearson correlation across
     the 15 receptors), hierarchically clusters the 8 units by it (does it
     recover the known V1-vs-extrastriate architectonic distinction?), and
     tests -- via a proper Mantel permutation test, not a naive pairwise
     correlation p-value, since matrix-pair correlations are not
     independent observations -- whether units with similar receptor
     fingerprints also have similar controllability profiles across
     subjects.

  B. COMPOSITE RECEPTOR AXES (PCA). Reduces the 8x15 receptor matrix to
     its principal components (rank-limited to <=7 given only 8
     observations, reported explicitly) and tests PC1/PC2 against
     controllability using Claim 1's own spatial-permutation machinery
     (imported, not reimplemented) -- resolves the Claim 4B collinearity
     problem with a principled method instead of a caveat.

  C. EXCITATION/INHIBITION BALANCE. A single composite index per unit
     (log ratio of mean excitatory to mean inhibitory receptor density),
     using a literature-rooted classification by receptor signaling
     mechanism (ionotropic channel type / G-protein coupling, cross-checked
     against the IUPHAR/BPS Guide to Pharmacology -- see EXCITATORY/
     INHIBITORY set definitions below for per-receptor basis), tested
     against controllability the same way as the PCA axes.

All three reuse run_claim1_analysis.py's UNIT_ORDER, load_region_
controllability, load_receptor_vectors, run_one_receptor_metric (partial
correlation + spatial permutation test), and bh_fdr, so results are
guaranteed methodologically consistent with Claim 1's own numbers rather
than a parallel reimplementation. Run on BOTH backbone atlases (Schaefer-
200, Glasser MMP1.0), matching the rest of the project's dual-atlas
robustness design; receptor data itself does not depend on backbone atlas
(Julich-Brain is fixed), only the controllability values being tested
against it do.

E/I classification basis (verified against IUPHAR/BPS Guide to
Pharmacology and standard receptor pharmacology, not a single
pre-packaged published "balance index" for this exact 15-receptor panel
-- flagged honestly rather than over-cited):
  Excitatory -- ionotropic cation channels: AMPA, kainate, NMDA
                (glutamatergic), alpha4beta2 (nicotinic ACh)
              -- Gq/11-coupled metabotropic: M1, M3 (muscarinic ACh),
                alpha1 (adrenergic), 5-HT2 (serotonergic)
              -- Gs-coupled metabotropic: D1 (dopaminergic)
  Inhibitory -- ionotropic Cl- channel: GABAA (+ BZ, its positive
                allosteric modulatory site)
              -- Gi/o-coupled metabotropic: GABAB, M2 (muscarinic ACh),
                alpha2 (adrenergic), 5-HT1A (serotonergic)

Usage:
    python3 receptor_architecture_analysis.py \
        --receptor-csv receptor-densities-deduped.csv \
        --region-controllability-schaefer hoc_region_controllability/schaefer200.csv \
        --region-controllability-glasser hoc_region_controllability/glasser.csv \
        --out-dir receptor_architecture/
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# This script depends on run_claim1_analysis.py, which lives in a sibling
# scripts/ subdirectory in this repository layout. Resolve that path
# relative to this file's own location rather than hardcoding an absolute
# path, so the repo works regardless of where it is cloned.
_SIBLING_DIR = Path(__file__).resolve().parent.parent / "05_receptor_architecture_vs_controllability"
if _SIBLING_DIR.is_dir():
    sys.path.insert(0, str(_SIBLING_DIR))

import run_claim1_analysis as c1  # UNIT_ORDER, load_region_controllability, load_receptor_vectors,
                                   # run_one_receptor_metric, bh_fdr, partial_corr

RECEPTOR_ORDER = ["AMPA", "kainate", "NMDA", "GABAA", "GABAB", "BZ", "M1", "M2", "M3",
                   "alpha1", "alpha2", "alpha4beta2", "5-HT1A", "5-HT2", "D1"]

EXCITATORY = {"AMPA", "kainate", "NMDA", "alpha4beta2", "M1", "M3", "alpha1", "5-HT2", "D1"}
INHIBITORY = {"GABAA", "BZ", "GABAB", "M2", "alpha2", "5-HT1A"}
assert EXCITATORY | INHIBITORY == set(RECEPTOR_ORDER) and not (EXCITATORY & INHIBITORY), \
    "E/I classification must partition the full 15-receptor set exactly"


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_fingerprint_matrix(receptor_csv, unit_order=c1.UNIT_ORDER, receptor_order=RECEPTOR_ORDER):
    """Returns an (n_units, n_receptors) matrix of mean_density, real units
    (fmol/mg) confirmed uniform across all 15 receptors before this is called
    (see fixture test / real-data check in module change notes)."""
    vecs = c1.load_receptor_vectors(receptor_csv)  # {receptor: {unit: density}}
    missing = [r for r in receptor_order if r not in vecs]
    if missing:
        raise ValueError(f"receptor CSV missing expected receptor(s): {missing}")
    M = np.zeros((len(unit_order), len(receptor_order)))
    for j, r in enumerate(receptor_order):
        for i, u in enumerate(unit_order):
            if u not in vecs[r]:
                raise ValueError(f"receptor CSV missing unit {u} for receptor {r}")
            M[i, j] = vecs[r][u]
    return M


def zscore_columns(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, ddof=0, keepdims=True)
    if np.any(sd == 0):
        raise ValueError("zero-variance receptor column -- cannot z-score")
    return (M - mu) / sd


# ---------------------------------------------------------------------------
# Part A -- fingerprint similarity, clustering, Mantel test
# ---------------------------------------------------------------------------

def fingerprint_similarity_matrix(M):
    """Pearson correlation across receptor columns between every pair of
    units (rows). M is (n_units, n_receptors)."""
    return np.corrcoef(M)


def hierarchical_cluster(sim_matrix, k=2):
    """Average-linkage hierarchical clustering on 1-similarity distance.
    Returns the linkage matrix and a k-cluster assignment."""
    dist = 1 - sim_matrix
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)  # guard against tiny negative floating-point noise
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=k, criterion="maxclust")
    return Z, clusters


def controllability_similarity_matrix(subj_metric, unit_order):
    """Builds an (n_subjects, n_units) matrix from subj_metric[subject][unit],
    keeping only subjects with all units present, then returns the
    (n_units, n_units) Pearson correlation matrix ACROSS SUBJECTS between
    every pair of units -- i.e. do unit i and unit j tend to have correlated
    controllability across the cohort?"""
    subjects = sorted(subj_metric.keys())
    rows = []
    for s in subjects:
        d = subj_metric[s]
        if all(u in d for u in unit_order):
            rows.append([d[u] for u in unit_order])
    Y = np.array(rows)  # (n_valid_subjects, n_units)
    return np.corrcoef(Y.T), Y.shape[0]


def mantel_test(mat_a, mat_b, n_permutations, rng):
    """Standard Mantel test: correlates the upper-triangle entries of two
    same-sized similarity/distance matrices, with significance from
    permuting one matrix's unit labels (not permuting the pairwise
    correlations independently, which would be invalid given they are not
    independent observations -- each unit appears in n-1 pairs)."""
    n = mat_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a_vec = mat_a[iu]
    if np.std(a_vec) == 0 or np.std(mat_b[iu]) == 0:
        return np.nan, np.nan, np.array([])
    observed = np.corrcoef(a_vec, mat_b[iu])[0, 1]
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        b_perm = mat_b[np.ix_(perm, perm)]
        null[i] = np.corrcoef(a_vec, b_perm[iu])[0, 1]
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
    return float(observed), float(p), null


# ---------------------------------------------------------------------------
# Part B -- PCA composite axes
# ---------------------------------------------------------------------------

def run_pca(M_z):
    """SVD-based PCA on an already-mean-centered/z-scored (n_units,
    n_receptors) matrix. Returns per-unit scores (n_units, n_components),
    receptor loadings (n_components, n_receptors), and explained-variance
    ratio. Rank is capped at n_units-1 (7 here) since z-scoring a 8-row
    matrix removes one degree of freedom -- reported explicitly, not
    glossed over."""
    U, S, Vt = np.linalg.svd(M_z, full_matrices=False)
    explained_var = (S ** 2) / np.sum(S ** 2)
    scores = U * S
    return scores, Vt, explained_var


# ---------------------------------------------------------------------------
# Part C -- excitation/inhibition balance
# ---------------------------------------------------------------------------

def ei_balance_index(M, receptor_order=RECEPTOR_ORDER):
    """Per-unit log(mean excitatory density / mean inhibitory density).
    Uses raw fmol/mg means (not z-scored) since this is a real physical
    ratio, valid because all 15 receptors share the same real unit
    (confirmed uniform across the CSV before this function is called)."""
    exc_idx = [i for i, r in enumerate(receptor_order) if r in EXCITATORY]
    inh_idx = [i for i, r in enumerate(receptor_order) if r in INHIBITORY]
    exc_mean = M[:, exc_idx].mean(axis=1)
    inh_mean = M[:, inh_idx].mean(axis=1)
    return np.log(exc_mean / inh_mean)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def bh_fdr_rows(results, pkey="perm_p_value", qkey="q_value_fdr_perm"):
    pvals = [r[pkey] for r in results if r.get(pkey) is not None and np.isfinite(r[pkey])]
    q = c1.bh_fdr(pvals) if pvals else []
    qi = 0
    for r in results:
        if r.get(pkey) is not None and np.isfinite(r[pkey]):
            r[qkey] = float(q[qi]); qi += 1
        else:
            r[qkey] = None
    return results


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor-csv", required=True)
    ap.add_argument("--region-controllability-schaefer", required=True)
    ap.add_argument("--region-controllability-glasser", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    M = load_fingerprint_matrix(args.receptor_csv)
    M_z = zscore_columns(M)
    print(f"Loaded {M.shape[0]}x{M.shape[1]} receptor fingerprint matrix "
          f"({c1.UNIT_ORDER}, {RECEPTOR_ORDER})")

    atlases = {
        "schaefer200": args.region_controllability_schaefer,
        "glasser": args.region_controllability_glasser,
    }

    # ---------------- Part A: fingerprint similarity + clustering + Mantel ----------------
    sim_receptor = fingerprint_similarity_matrix(M)
    Z, clusters_k2 = hierarchical_cluster(sim_receptor, k=2)
    print("\n=== Part A: receptor-fingerprint similarity ===")
    print("2-cluster split (average-linkage on 1-r distance):")
    for u, c in zip(c1.UNIT_ORDER, clusters_k2):
        print(f"  {u}: cluster {c}")

    sim_rows = []
    for i, u1 in enumerate(c1.UNIT_ORDER):
        for j, u2 in enumerate(c1.UNIT_ORDER):
            sim_rows.append({"unit_1": u1, "unit_2": u2, "receptor_fingerprint_r": sim_receptor[i, j]})
    write_csv(out_dir / "receptor_fingerprint_similarity_matrix.csv", sim_rows,
              ["unit_1", "unit_2", "receptor_fingerprint_r"])

    cluster_rows = [{"unit": u, "cluster_k2": int(c)} for u, c in zip(c1.UNIT_ORDER, clusters_k2)]
    write_csv(out_dir / "receptor_fingerprint_clusters.csv", cluster_rows, ["unit", "cluster_k2"])

    mantel_results = []
    for atlas, path in atlases.items():
        subj_avg, subj_modal, _ = c1.load_region_controllability(path)
        for metric_name, subj_c in [("average_controllability", subj_avg), ("modal_controllability", subj_modal)]:
            sim_ctrl, n_subj = controllability_similarity_matrix(subj_c, c1.UNIT_ORDER)
            observed, p, null = mantel_test(sim_receptor, sim_ctrl, args.n_permutations, rng)
            mantel_results.append({
                "atlas": atlas, "metric": metric_name, "n_subjects_used": n_subj,
                "mantel_r": observed, "perm_p_value": p, "n_permutations_used": len(null),
            })
            print(f"  Mantel [{atlas}/{metric_name}]: r={observed:.3f}, perm_p={p:.4g}, n_subj={n_subj}")
    mantel_results = bh_fdr_rows(mantel_results)
    write_csv(out_dir / "receptor_fingerprint_mantel_test.csv", mantel_results,
              ["atlas", "metric", "n_subjects_used", "mantel_r", "perm_p_value",
               "n_permutations_used", "q_value_fdr_perm"])

    # ---------------- Part B: PCA composite axes ----------------
    print("\n=== Part B: PCA composite receptor axes ===")
    scores, loadings, explained_var = run_pca(M_z)
    n_components_report = min(3, scores.shape[1])
    print(f"Explained variance ratio (components 1-{scores.shape[1]}, rank-limited to n_units-1=7): "
          f"{np.round(explained_var, 3).tolist()}")

    pca_score_rows = []
    for i, u in enumerate(c1.UNIT_ORDER):
        row = {"unit": u}
        for k in range(n_components_report):
            row[f"PC{k+1}_score"] = float(scores[i, k])
        pca_score_rows.append(row)
    write_csv(out_dir / "receptor_pca_scores.csv", pca_score_rows,
              ["unit"] + [f"PC{k+1}_score" for k in range(n_components_report)])

    loading_rows = []
    for k in range(n_components_report):
        row = {"component": f"PC{k+1}", "explained_variance_ratio": float(explained_var[k])}
        for j, r in enumerate(RECEPTOR_ORDER):
            row[r] = float(loadings[k, j])
        loading_rows.append(row)
    write_csv(out_dir / "receptor_pca_loadings.csv", loading_rows,
              ["component", "explained_variance_ratio"] + RECEPTOR_ORDER)

    pca_test_results = []
    for atlas, path in atlases.items():
        subj_avg, subj_modal, subj_size = c1.load_region_controllability(path)
        subjects = sorted(subj_avg.keys())
        for k in range(min(2, scores.shape[1])):  # test PC1, PC2 only -- own FDR family
            pc_vec = {u: float(scores[i, k]) for i, u in enumerate(c1.UNIT_ORDER)}
            for metric_name, subj_c in [("average_controllability", subj_avg), ("modal_controllability", subj_modal)]:
                stat = c1.run_one_receptor_metric(subj_c, subj_size, pc_vec, subjects,
                                                   n_permutations=args.n_permutations, rng=rng)
                row = {"atlas": atlas, "component": f"PC{k+1}", "metric": metric_name}
                row.update(stat)
                pca_test_results.append(row)
                if "perm_p_value" in stat:
                    print(f"  PC{k+1} [{atlas}/{metric_name}]: partial_r={stat.get('mean_partial_r', float('nan')):.3f}, "
                          f"perm_p={stat.get('perm_p_value', float('nan')):.4g}")
    pca_test_results = bh_fdr_rows(pca_test_results)
    write_csv(out_dir / "receptor_pca_controllability_results.csv", pca_test_results,
              ["atlas", "component", "metric", "n_subjects_valid", "mean_raw_r", "mean_partial_r",
               "mean_fisher_z", "perm_p_value", "n_permutations_used", "q_value_fdr_perm",
               "t_stat", "p_ttest", "wilcoxon_stat", "p_wilcoxon", "note"])

    # ---------------- Part C: E/I balance ----------------
    print("\n=== Part C: excitation/inhibition balance ===")
    ei = ei_balance_index(M)
    for u, v in zip(c1.UNIT_ORDER, ei):
        print(f"  {u}: E/I log-ratio = {v:.3f}")
    write_csv(out_dir / "receptor_ei_balance_index.csv",
              [{"unit": u, "ei_log_ratio": float(v)} for u, v in zip(c1.UNIT_ORDER, ei)],
              ["unit", "ei_log_ratio"])

    ei_vec = {u: float(v) for u, v in zip(c1.UNIT_ORDER, ei)}
    ei_test_results = []
    for atlas, path in atlases.items():
        subj_avg, subj_modal, subj_size = c1.load_region_controllability(path)
        subjects = sorted(subj_avg.keys())
        for metric_name, subj_c in [("average_controllability", subj_avg), ("modal_controllability", subj_modal)]:
            stat = c1.run_one_receptor_metric(subj_c, subj_size, ei_vec, subjects,
                                               n_permutations=args.n_permutations, rng=rng)
            row = {"atlas": atlas, "metric": metric_name}
            row.update(stat)
            ei_test_results.append(row)
            if "perm_p_value" in stat:
                print(f"  E/I [{atlas}/{metric_name}]: partial_r={stat.get('mean_partial_r', float('nan')):.3f}, "
                      f"perm_p={stat.get('perm_p_value', float('nan')):.4g}")
    ei_test_results = bh_fdr_rows(ei_test_results)
    write_csv(out_dir / "receptor_ei_controllability_results.csv", ei_test_results,
              ["atlas", "metric", "n_subjects_valid", "mean_raw_r", "mean_partial_r",
               "mean_fisher_z", "perm_p_value", "n_permutations_used", "q_value_fdr_perm",
               "t_stat", "p_ttest", "wilcoxon_stat", "p_wilcoxon", "note"])

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
