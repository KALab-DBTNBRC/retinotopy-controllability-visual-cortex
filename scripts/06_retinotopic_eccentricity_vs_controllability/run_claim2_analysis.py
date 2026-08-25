#!/usr/bin/env python3
"""
Claim 2 -- pRF eccentricity predicts structural controllability within visual
cortex.

Tests whether a subject's own retinotopic eccentricity representation (from
Benson et al. pRF solutions, R2>30-thresholded, aggregated per Schaefer-200
node by aggregate_prf_to_backbone.py) predicts that same node's structural
controllability, WITHIN each subject, across the Vis-network nodes.

Design differs from Claim 1 in one fundamental way, verified against real
data before building (2026-08-12): Claim 1's receptor density was a FIXED,
group-level vector (one Julich-Brain value per region, same for every
subject) -- eccentricity is NOT. It's measured per-subject from each
person's own retinotopic mapping, so both the predictor (eccentricity) and
the outcome (controllability) vary by subject here. This changes two things
versus run_claim1_analysis.py:

  1. Node set is per-subject, not a fixed 8-unit vector. The real
     all_subjects_prf_backbone.csv shows 5 of Schaefer-200's 29 Vis-network
     nodes have zero R2>30 coverage in 100% of subjects (already known,
     aggregate_prf_to_backbone.py) -- excluded entirely, leaving 24 nodes.
     But even among those 24, EVERY one of the 46 subjects is missing at
     least one node's eccentricity value (median 4 missing, one subject
     missing 23 of 24 and excluded outright) -- so each subject's own
     per-node vector length varies. A subject is included only if they have
     at least --min-valid-nodes (default 15, user-set 2026-08-12 after
     reviewing the real retention gradient: any threshold 5-15 retains the
     same 45/46 subjects, so 15 was chosen as the largest threshold with zero
     sample-size cost, giving the healthiest per-subject degrees of freedom).
     A per-node quality floor (--min-vertices-per-node, default 5) also
     excludes individual (subject, node) eccentricity estimates built from
     too few R2>30 vertices to trust (8.7% of real nonzero-coverage entries
     have under 5 valid vertices).
  2. The spatial permutation test (added to Claim 1 after a real false-
     positive was caught there, see run_claim1_analysis.py's docstring) has
     to work differently here. Claim 1 permuted ONE shared fixed vector
     (receptor density) identically for every subject. Here there's no fixed
     vector to permute -- instead, for EACH subject independently, that
     subject's own eccentricity values are reshuffled across their own valid
     node positions (their real controllability/size values keep their real
     node identity), breaking the within-subject node-to-node correspondence
     while preserving each subject's own real data distribution. This is
     still the PRIMARY significance test, same reasoning as Claim 1: with a
     modest number of spatial units per subject, a per-subject-then-t-test
     design needs an empirical null specific to the actual design, not a
     bare parametric test.

average_controllability is read and log-transformed directly per Schaefer-200
node (no cross-node aggregation needed here, unlike Claim 1's hOc-to-backbone
many-to-one mapping) -- same log-space rationale as Claim 1 (see
run_claim1_analysis.py and build_hoc_region_controllability.py docstrings):
average controllability's heavy-tailed, eigenstructure-sensitive distribution
is concentrated in visual cortex specifically in this cohort, confirmed via
diagnose_connectome_hemisphere_asymmetry.py. modal_controllability is used
raw (no asymmetry ever found for it).

Partial correlation controls for node size (n_vertices_total, fixed by the
atlas -- real range 192-527 across the 24 usable Vis nodes, a ~2.7x spread),
same size-covariate rationale as Claim 1's methodological note (average
controllability scales with region size/streamline count).

Usage:
    python3 run_claim2_analysis.py \
        --prf-backbone-csv /path/to/prf_backbone/all_subjects_prf_backbone.csv \
        --controllability-csv /path/to/controllability/all_subjects_controllability_schaefer200.csv \
        --out-csv /path/to/claim2_results_schaefer200.csv

    # Glasser rerun (Claim 5, 2026-08-13):
    python3 run_claim2_analysis.py \
        --prf-backbone-csv /path/to/prf_backbone/all_subjects_prf_backbone.csv \
        --controllability-csv /path/to/controllability/all_subjects_controllability_glasser.csv \
        --atlas glasser \
        --out-csv /path/to/claim2_results_glasser.csv
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Glasser MMP1.0's "Vis-network" equivalent (2026-08-13 decision, see todo.md
# decisions log). Glasser has no built-in functional-network label the way
# Schaefer-200's 7Networks_..._Vis_... naming does -- area groupings instead
# come from Glasser et al. 2016's own official "Cortical Division" scheme,
# sourced here from Huang & Rolls et al. 2022 (Brain Structure and Function
# 227:763-778), Table 1, which explicitly reproduces the original Glasser
# area-to-division assignments verbatim (region abbreviations match our own
# Glasser dlabel.nii/controllability node labels exactly: V1, V2, ..., V4t).
# Union of all 5 visual-system divisions (Primary Visual + Early Visual +
# Dorsal Stream Visual + Ventral Stream Visual + MT+ Complex) is used, not
# just "early visual," to match the SPIRIT of Schaefer-200's Vis network
# (which itself spans early through extrastriate/MT-like visual cortex, not
# just V1-V4) -- 26 areas/hemisphere, 52 nodes total.
GLASSER_VIS_AREAS = {
    "V1",                                              # Primary Visual
    "V2", "V3", "V4",                                  # Early Visual
    "V3A", "V3B", "V6", "V6A", "V7", "IPS1",           # Dorsal Stream Visual
    "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",  # Ventral Stream Visual
    "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t",  # MT+ Complex
}
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def is_glasser_vis_node(node_label):
    """node_label format confirmed against real data (2026-08-13):
    'L_V1_ROI' / 'R_V1_ROI' etc. -- strips the hemisphere prefix and _ROI
    suffix, checks the bare area name against GLASSER_VIS_AREAS."""
    m = GLASSER_LABEL_RE.match(node_label)
    if not m:
        return False
    return m.group(1) in GLASSER_VIS_AREAS


def load_prf_backbone(path, min_vertices_per_node, atlas):
    """Returns:
      subj_ecc[subject][node_label] = mean_eccentricity_deg
      node_size[node_label] = n_vertices_total (fixed by the atlas)
    Restricted to the given atlas and its Vis-network-equivalent node set
    (Schaefer-200: real is_vis_network column; Glasser: GLASSER_VIS_AREAS,
    see module docstring), and entries with a real (non-missing) eccentricity
    value built from >= min_vertices_per_node R2>30 vertices -- thinner
    estimates are dropped as too noisy to trust, not silently kept.
    """
    subj_ecc = defaultdict(dict)
    node_size = {}
    n_dropped_thin = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["atlas"] != atlas:
                continue
            if atlas == "schaefer200":
                is_vis = row["is_vis_network"] == "True"
            else:
                is_vis = is_glasser_vis_node(row["backbone_label"])
            if not is_vis:
                continue
            node_size[row["backbone_label"]] = int(row["n_vertices_total"])
            if row["mean_eccentricity_deg"] == "":
                continue  # zero R2>30 vertices for this (subject, node) -- real, expected, not an error
            n_valid = int(row["n_vertices_valid_R2gt30"])
            if n_valid < min_vertices_per_node:
                n_dropped_thin += 1
                continue
            subj_ecc[row["subject"]][row["backbone_label"]] = float(row["mean_eccentricity_deg"])
    print(f"Dropped {n_dropped_thin} (subject, node) entries with < {min_vertices_per_node} "
          f"R2>30 vertices (too thin to trust the mean eccentricity estimate)")
    return subj_ecc, node_size


def load_controllability(path):
    """Returns subj_avg_log[subject][node_label], subj_modal[subject][node_label].
    average_controllability is log-transformed here directly (see module
    docstring re: the visual-cortex-specific asymmetry finding)."""
    subj_avg_log = defaultdict(dict)
    subj_modal = defaultdict(dict)
    n_nonpositive = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s, n = row["subject"], row["node_label"]
            avg_c = float(row["average_controllability"])
            subj_modal[s][n] = float(row["modal_controllability"])
            if avg_c > 0:
                subj_avg_log[s][n] = math.log(avg_c)
            else:
                n_nonpositive += 1  # not expected -- Phase 6 QC confirmed all-positive on real data
    if n_nonpositive:
        print(f"WARNING: {n_nonpositive} rows had non-positive average_controllability -- "
              f"excluded from the log-space metric only, investigate if this is unexpected")
    return subj_avg_log, subj_modal


def partial_corr(x, y, z):
    """First-order partial correlation of x,y controlling for z. Returns NaN
    if any pairwise correlation is undefined (e.g. zero variance)."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return (rxy - rxz * ryz) / denom


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns q-values in the same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def pearson_rowwise(A, B):
    """Row-wise Pearson correlation between two (n, k) matrices. Verified
    against scipy.stats.pearsonr in run_claim1_analysis.py's fixture."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def subject_permuted_partial_r(x, y, z, n_permutations, rng):
    """For ONE subject's real (x, y, z) vectors (length k = however many
    valid nodes this subject has), returns an array of length n_permutations
    of that subject's partial-r under random within-subject reshuffles of x
    only (y, z keep their real node identity) -- vectorized across all
    permutations at once via pearson_rowwise, not a Python loop per
    permutation."""
    k = len(x)
    # build n_permutations independent random permutations of x's k positions
    keys = np.random.default_rng(rng.integers(0, 2**63 - 1)).random((n_permutations, k))
    perm_idx = np.argsort(keys, axis=1)
    X_perm = x[perm_idx]  # (n_permutations, k)
    Y_mat = np.tile(y, (n_permutations, 1))
    Z_mat = np.tile(z, (n_permutations, 1))
    rxy = pearson_rowwise(X_perm, Y_mat)
    rxz = pearson_rowwise(X_perm, Z_mat)
    ryz = pearson_rowwise(Y_mat, Z_mat)  # constant across permutations, but cheap and correct as-is
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rxy - rxz * ryz) / denom


def run_one_metric(subj_ecc, subj_controllability, node_size, subjects,
                    min_valid_nodes, n_permutations, rng):
    per_subject_r = []
    per_subject_partial_r = []
    used_subjects = []
    subject_arrays = []  # (x, y, z) real arrays per used subject, for the permutation step

    for s in subjects:
        ecc_vec = subj_ecc.get(s, {})
        c_vec = subj_controllability.get(s, {})
        common_nodes = sorted(set(ecc_vec) & set(c_vec) & set(node_size))
        if len(common_nodes) < min_valid_nodes:
            continue
        x = np.array([ecc_vec[n] for n in common_nodes])
        y = np.array([c_vec[n] for n in common_nodes])
        z = np.array([node_size[n] for n in common_nodes])

        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r_raw = stats.pearsonr(x, y)[0]
        r_partial = partial_corr(x, y, z)
        if not np.isfinite(r_partial):
            continue

        per_subject_r.append(r_raw)
        per_subject_partial_r.append(r_partial)
        used_subjects.append(s)
        subject_arrays.append((x, y, z))

    n_valid = len(used_subjects)
    if n_valid < 5:
        return {"n_subjects_valid": n_valid, "note": "too few valid subjects, skipped"}

    z_vals = np.arctanh(np.clip(per_subject_partial_r, -0.999999, 0.999999))
    t_stat, p_t = stats.ttest_1samp(z_vals, 0.0)
    try:
        w_stat, p_w = stats.wilcoxon(per_subject_partial_r)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    observed_mean_fisher_z = float(np.mean(z_vals))

    # PRIMARY: spatial permutation test, per-subject reshuffle (see module docstring)
    per_subject_null_z = []
    for (x, y, z) in subject_arrays:
        rp = subject_permuted_partial_r(x, y, z, n_permutations, rng)
        zz = np.arctanh(np.clip(rp, -0.999999, 0.999999))
        per_subject_null_z.append(zz)
    null_stack = np.vstack(per_subject_null_z)  # (n_subjects_valid, n_permutations)
    null_mean_z = np.nanmean(null_stack, axis=0)
    null_mean_z = null_mean_z[np.isfinite(null_mean_z)]
    if len(null_mean_z) == 0:
        perm_p = np.nan
    else:
        perm_p = (np.sum(np.abs(null_mean_z) >= abs(observed_mean_fisher_z)) + 1) / (len(null_mean_z) + 1)

    median_n_nodes = float(np.median([len(x) for (x, _, _) in subject_arrays]))

    return {
        "n_subjects_valid": n_valid,
        "median_n_nodes_used": median_n_nodes,
        "mean_raw_r": float(np.mean(per_subject_r)),
        "mean_partial_r": float(np.mean(per_subject_partial_r)),
        "mean_fisher_z": observed_mean_fisher_z,
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else None,
        "p_wilcoxon": float(p_w) if np.isfinite(p_w) else None,
        "perm_p_value": float(perm_p) if np.isfinite(perm_p) else None,
        "n_permutations_used": n_permutations,
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--atlas", default="schaefer200", choices=("schaefer200", "glasser"),
                     help="which atlas's Vis-network-equivalent node set to use (2026-08-13: "
                          "Glasser's is GLASSER_VIS_AREAS, see module docstring)")
    ap.add_argument("--min-valid-nodes", type=int, default=15,
                     help="minimum Vis-network nodes with usable data a subject needs to be "
                          "included (2026-08-12 decision: 15, see module docstring)")
    ap.add_argument("--min-vertices-per-node", type=int, default=5,
                     help="minimum R2>30 vertices behind a (subject, node) eccentricity estimate "
                          "for it to be trusted, otherwise treated as missing")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subj_ecc, node_size = load_prf_backbone(args.prf_backbone_csv, args.min_vertices_per_node, args.atlas)
    subj_avg_log, subj_modal = load_controllability(args.controllability_csv)
    subjects = sorted(subj_ecc.keys())

    print(f"{len(subjects)} subjects with pRF-backbone data")
    print(f"{len(node_size)} Vis-network nodes in the atlas (5 known always-zero-coverage nodes "
          f"will simply never appear as 'common_nodes' for any subject, not hardcoded/excluded by name)")
    print(f"min_valid_nodes={args.min_valid_nodes}, min_vertices_per_node={args.min_vertices_per_node}, "
          f"n_permutations={args.n_permutations}, seed={args.seed}")

    results = []
    for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                 ("modal_controllability", subj_modal)]:
        stat = run_one_metric(subj_ecc, subj_c, node_size, subjects,
                               args.min_valid_nodes, args.n_permutations, rng)
        row = {"metric": metric_name}
        row.update(stat)
        results.append(row)
        n_used = stat.get("n_subjects_valid", 0)
        print(f"{metric_name}: {n_used}/{len(subjects)} subjects used "
              f"(median {stat.get('median_n_nodes_used', 'n/a')} nodes/subject)")

    perm_p_values = [r["perm_p_value"] for r in results if r.get("perm_p_value") is not None]
    q_values = bh_fdr(perm_p_values) if perm_p_values else []
    qi = 0
    for r in results:
        if r.get("perm_p_value") is not None:
            r["q_value_fdr_perm"] = float(q_values[qi])
            qi += 1
        else:
            r["q_value_fdr_perm"] = None

    fieldnames = ["metric", "n_subjects_valid", "median_n_nodes_used", "mean_raw_r", "mean_partial_r",
                  "mean_fisher_z", "perm_p_value", "n_permutations_used", "q_value_fdr_perm",
                  "t_stat", "p_ttest", "wilcoxon_stat", "p_wilcoxon", "note"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n=== Results (permutation FDR q < 0.05 -- PRIMARY test) ===")
    sig = [r for r in results if r.get("q_value_fdr_perm") is not None and r["q_value_fdr_perm"] < 0.05]
    if sig:
        for r in sorted(sig, key=lambda x: x["q_value_fdr_perm"]):
            print(f"  {r['metric']}: partial_r={r['mean_partial_r']:.3f}, "
                  f"perm_p={r['perm_p_value']:.4g}, q={r['q_value_fdr_perm']:.4g}, "
                  f"(parametric p_ttest={r['p_ttest']:.4g} for comparison), n={r['n_subjects_valid']}")
    else:
        print("  (none)")

    print(f"\nFull results ({len(results)} rows) written to {out_csv}")


if __name__ == "__main__":
    main()
