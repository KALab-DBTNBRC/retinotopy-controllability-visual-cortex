#!/usr/bin/env python3
"""
Receptor architecture vs. behavior, WITHOUT routing through average/modal
controllability -- user request (2026-08-13): "Check receptor with
behavior. Don't lean towards only NCT."

The real constraint this design has to solve: receptor density (Julich-
Brain histology) is a FIXED, group-level 8-unit vector -- identical for
every subject -- while behavior (delay-discounting, Flanker) is per-
subject. There is no direct way to correlate a fixed vector against 46
different subject values; every prior receptor test in this project
(Claim 1, Claim 4B, today's receptor-architecture analysis) bridges this
gap using each subject's own controllability values across the same 8
units. To test receptor-behavior WITHOUT that NCT bridge, this script uses
a different, genuinely non-NCT per-subject regional quantity that is
already in hand: n_hoc_vertices, each subject's own real anatomical region
size (vertex count) at each of the 8 hOc units -- purely a registration/
anatomy measure, no eigenstructure computation, no nctpy involvement
anywhere in this script.

Two layers, each mirroring an existing, already-verified design:

  PART 1 (mirrors Claim 1's exact structure, receptor density vs. an
  outcome across the 8 units, per subject then group-tested) -- but Y is
  n_hoc_vertices instead of controllability, and there is no covariate
  (region size IS the outcome here, not a nuisance to control for). For
  each of the 15 receptors: per subject, raw Pearson r between the fixed
  receptor vector and that subject's own region-size vector; Fisher z;
  one-sample t-test (parametric, reported for comparison) and a spatial
  permutation test (PRIMARY, reassigns the 8 receptor values across the 8
  unit labels, same logic as Claim 1's spatial_permutation_test, just
  without the partial-correlation machinery since there's no covariate
  here) -- own 15-test FDR family. Answers: does a receptor's spatial
  density pattern predict which regions are anatomically larger/smaller
  in a typical individual?

  PART 2 (mirrors the personality "coupling strength" design -- each
  subject's own per-subject alignment score, correlated against an
  outcome across SUBJECTS, standard N=46 individual-differences test, no
  permutation needed per Claim 3's own reasoning) -- each subject's Part-1
  per-subject raw r (their own "receptor-anatomy alignment score") is
  correlated against the 3 HCP behavioral measures already used in Claim
  3, reusing run_claim3_analysis.py's BEHAVIORAL_COLUMNS/load_behavioral/
  run_one_pair (imported, not reimplemented). Run twice, two separate FDR
  families:
    2a. All 15 individual receptors x 3 behavioral measures = 45 tests.
    2b. The 3 composite receptor axes from today's receptor_architecture_
        analysis.py (PC1, PC2, E/I balance) x 3 behavioral measures = 9
        tests -- the more principled, non-fishing-expedition version,
        kept as its own family rather than pooled with 2a.

A separate, more direct non-NCT bridge -- real per-subject structural
connectivity STRENGTH (weighted node degree from the same connectomes,
standard graph theory, still not network control theory) -- would be a
stronger test than anatomical region size, but requires raw per-subject
connectome matrices not available in this sandbox; see
extract_node_strength.py (companion script, for the user to run
server-side) for that follow-up.

Usage:
    python3 receptor_behavior_direct.py \
        --receptor-csv receptor-densities-deduped.csv \
        --region-controllability-schaefer hoc_region_controllability/schaefer200.csv \
        --master-table-xlsx master-table.xlsx \
        --pca-scores-csv receptor_architecture/receptor_pca_scores.csv \
        --ei-balance-csv receptor_architecture/receptor_ei_balance_index.csv \
        --out-dir receptor_behavior_direct/
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# This script depends on run_claim1_analysis.py and run_claim3_analysis.py,
# which live in sibling scripts/ subdirectories in this repository layout.
# Resolve those paths relative to this file's own location rather than
# hardcoding an absolute path, so the repo works regardless of where it is
# cloned.
_SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
for _sibling in ("05_receptor_architecture_vs_controllability", "07_behavior_personality"):
    _sibling_dir = _SCRIPTS_ROOT / _sibling
    if _sibling_dir.is_dir():
        sys.path.insert(0, str(_sibling_dir))

import run_claim1_analysis as c1
import run_claim3_analysis as c3

RECEPTOR_ORDER = ["AMPA", "kainate", "NMDA", "GABAA", "GABAB", "BZ", "M1", "M2", "M3",
                   "alpha1", "alpha2", "alpha4beta2", "5-HT1A", "5-HT2", "D1"]


# ---------------------------------------------------------------------------
# Part 1: receptor density vs. anatomical region size, per-unit, no NCT
# ---------------------------------------------------------------------------

def per_subject_raw_r(receptor_x, subj_size, subjects):
    """Per subject: raw Pearson r between the fixed receptor vector and that
    subject's own n_hoc_vertices vector across the 8 units. Drops subjects
    missing any unit or with zero-variance size (no guessing)."""
    used, per_r, y_rows = [], [], []
    for s in subjects:
        d = subj_size.get(s, {})
        if not all(u in d for u in c1.UNIT_ORDER):
            continue
        y = np.array([d[u] for u in c1.UNIT_ORDER], dtype=float)
        if np.std(y) == 0:
            continue
        r = stats.pearsonr(receptor_x, y)[0]
        if not np.isfinite(r):
            continue
        used.append(s)
        per_r.append(r)
        y_rows.append(y)
    return used, np.array(per_r), np.array(y_rows)


def spatial_permutation_test_raw(receptor_x, Y_mat, observed_mean_z, n_permutations, rng):
    """Same logic as run_claim1_analysis.spatial_permutation_test but for a
    RAW (non-partial) correlation, since region size is the outcome here,
    not a covariate to control for."""
    n_units = len(receptor_x)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_units)
        X_mat = np.tile(receptor_x[perm], (Y_mat.shape[0], 1))
        r = c1.pearson_rowwise(X_mat, Y_mat)
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        valid = np.isfinite(z)
        null_stats[i] = np.mean(z[valid]) if np.any(valid) else np.nan
    null_stats = null_stats[np.isfinite(null_stats)]
    if len(null_stats) == 0 or not np.isfinite(observed_mean_z):
        return np.nan, null_stats
    p = (np.sum(np.abs(null_stats) >= abs(observed_mean_z)) + 1) / (len(null_stats) + 1)
    return float(p), null_stats


def test_receptor_vs_size(receptor_x, subj_size, subjects, n_permutations, rng):
    used, per_r, Y_mat = per_subject_raw_r(receptor_x, subj_size, subjects)
    if len(used) < 5:
        return {"n_subjects_valid": len(used), "note": "too few valid subjects, skipped"}, {}
    z = np.arctanh(np.clip(per_r, -0.999999, 0.999999))
    mean_z = float(np.mean(z))
    t_stat, p_t = stats.ttest_1samp(z, 0.0)
    perm_p, null = spatial_permutation_test_raw(receptor_x, Y_mat, mean_z, n_permutations, rng)
    result = {
        "n_subjects_valid": len(used), "mean_raw_r": float(np.mean(per_r)),
        "mean_fisher_z": mean_z, "t_stat": float(t_stat), "p_ttest": float(p_t),
        "perm_p_value": perm_p, "n_permutations_used": int(len(null)), "note": "",
    }
    per_subject_scores = dict(zip(used, per_r))
    return result, per_subject_scores


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


def bh_fdr_rows_pearson(results):
    pvals = [r["p_pearson"] for r in results if "p_pearson" in r]
    q = c1.bh_fdr(pvals) if pvals else []
    qi = 0
    for r in results:
        if "p_pearson" in r:
            r["q_value_fdr"] = float(q[qi]); qi += 1
        else:
            r["q_value_fdr"] = None
    return results


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_pca_scores(path):
    """Returns {component: {unit: score}} for PC1/PC2 from
    receptor_architecture_analysis.py's output."""
    rows = list(csv.DictReader(open(path)))
    out = {"PC1_score": {}, "PC2_score": {}}
    for r in rows:
        for comp in list(out.keys()):
            if comp in r and r[comp] != "":
                out[comp][r["unit"]] = float(r[comp])
    return out


def load_ei_balance(path):
    return {r["unit"]: float(r["ei_log_ratio"]) for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor-csv", required=True)
    ap.add_argument("--region-controllability-schaefer", required=True,
                     help="only used for its n_hoc_vertices column -- controllability columns unused here")
    ap.add_argument("--master-table-xlsx", required=True)
    ap.add_argument("--pca-scores-csv", required=True)
    ap.add_argument("--ei-balance-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    _, _, subj_size = c1.load_region_controllability(args.region_controllability_schaefer)
    receptor_vecs = c1.load_receptor_vectors(args.receptor_csv)
    subjects = sorted(subj_size.keys())
    behavioral = c3.load_behavioral(args.master_table_xlsx)
    pca_scores = load_pca_scores(args.pca_scores_csv)
    ei_scores = load_ei_balance(args.ei_balance_csv)

    print(f"{len(subjects)} subjects with region-size data; {len(behavioral)} with complete behavioral data")

    # ---------------- Part 1: receptor density vs. anatomical region size ----------------
    print("\n=== Part 1: receptor density vs. per-subject anatomical region size (no NCT) ===")
    part1_results = []
    alignment_scores = {}  # {receptor: {subject: r}}
    for receptor in RECEPTOR_ORDER:
        vec = receptor_vecs[receptor]
        receptor_x = np.array([vec[u] for u in c1.UNIT_ORDER])
        stat, per_subj = test_receptor_vs_size(receptor_x, subj_size, subjects, args.n_permutations, rng)
        row = {"receptor": receptor}
        row.update(stat)
        part1_results.append(row)
        alignment_scores[receptor] = per_subj
        if "perm_p_value" in stat:
            print(f"  {receptor}: mean_r={stat['mean_raw_r']:.3f}, perm_p={stat['perm_p_value']:.4g}")
    part1_results = bh_fdr_rows(part1_results)
    write_csv(out_dir / "receptor_vs_region_size_results.csv", part1_results,
              ["receptor", "n_subjects_valid", "mean_raw_r", "mean_fisher_z", "t_stat", "p_ttest",
               "perm_p_value", "n_permutations_used", "q_value_fdr_perm", "note"])

    align_rows = []
    for s in subjects:
        row = {"subject": s}
        for receptor in RECEPTOR_ORDER:
            row[receptor] = alignment_scores[receptor].get(s, "")
        align_rows.append(row)
    write_csv(out_dir / "receptor_size_alignment_scores.csv", align_rows, ["subject"] + RECEPTOR_ORDER)

    # ---------------- Part 2a: per-receptor alignment score vs. behavior ----------------
    print("\n=== Part 2a: per-subject receptor-size alignment vs. behavior (15 receptors x 3 measures) ===")
    part2a = []
    for receptor in RECEPTOR_ORDER:
        align_vals = alignment_scores[receptor]
        for b_key in c3.BEHAVIORAL_COLUMNS:
            stat = c3.run_one_pair(receptor, align_vals, b_key, behavioral, subjects)
            row = {"receptor": receptor, "behavioral_measure": b_key}
            row.update(stat)
            part2a.append(row)
    part2a = bh_fdr_rows_pearson(part2a)
    write_csv(out_dir / "receptor_alignment_vs_behavior_results.csv", part2a,
              ["receptor", "behavioral_measure", "n_subjects", "pearson_r", "p_pearson",
               "q_value_fdr", "spearman_rho", "p_spearman", "note"])
    sig2a = [r for r in part2a if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"  {len(sig2a)}/{len(part2a)} significant at FDR q<0.05" if part2a else "  no valid tests")
    for r in sorted(part2a, key=lambda x: x.get("p_pearson", 1) if "p_pearson" in x else 1)[:5]:
        if "p_pearson" in r:
            print(f"  closest: {r['receptor']} vs {r['behavioral_measure']}: r={r['pearson_r']:.3f}, "
                  f"p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}")

    # ---------------- Part 2b: composite receptor axes vs. behavior ----------------
    print("\n=== Part 2b: composite receptor axes (PC1/PC2/E-I) vs. behavior (9 tests) ===")
    composite_alignment = {}
    for comp_name, comp_vec in [("PC1", pca_scores["PC1_score"]), ("PC2", pca_scores["PC2_score"]),
                                 ("EI_balance", ei_scores)]:
        receptor_x = np.array([comp_vec[u] for u in c1.UNIT_ORDER])
        stat, per_subj = test_receptor_vs_size(receptor_x, subj_size, subjects, args.n_permutations, rng)
        composite_alignment[comp_name] = per_subj
        print(f"  [receptor-vs-size check] {comp_name}: mean_r={stat.get('mean_raw_r', float('nan')):.3f}, "
              f"perm_p={stat.get('perm_p_value', float('nan')):.4g}")

    part2b = []
    for comp_name, align_vals in composite_alignment.items():
        for b_key in c3.BEHAVIORAL_COLUMNS:
            stat = c3.run_one_pair(comp_name, align_vals, b_key, behavioral, subjects)
            row = {"composite": comp_name, "behavioral_measure": b_key}
            row.update(stat)
            part2b.append(row)
    part2b = bh_fdr_rows_pearson(part2b)
    write_csv(out_dir / "receptor_composite_alignment_vs_behavior_results.csv", part2b,
              ["composite", "behavioral_measure", "n_subjects", "pearson_r", "p_pearson",
               "q_value_fdr", "spearman_rho", "p_spearman", "note"])
    for r in part2b:
        if "p_pearson" in r:
            print(f"  {r['composite']} vs {r['behavioral_measure']}: r={r['pearson_r']:.3f}, "
                  f"p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
