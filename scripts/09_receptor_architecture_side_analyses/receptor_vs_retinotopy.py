#!/usr/bin/env python3
"""
Links the paper's two threads directly: does cytoarchitectonic receptor
density track a subject's own RETINOTOPIC organization (eccentricity,
polar angle) at the 8 hOc units -- independent of controllability
entirely? User request (2026-08-13): "let's try to link the two findings,
so that the paper doesn't read as two separate papers."

Two outcomes, both tie the threads together, just differently:
  - If receptor density DOES track retinotopic position: receptor
    architecture and controllability are both organized by the same
    underlying variable (retinotopy) -- a genuinely unifying finding.
  - If it does NOT (consistent with every other receptor-vs-function test
    so far -- controllability null, behavior null, personality null):
    strengthens the paper's actual thesis, now demonstrated end-to-end --
    retinotopic position, not cytoarchitectonic identity, organizes
    visual-cortex function, and this holds even when tested against
    retinotopy itself rather than only its downstream (controllability)
    consequence.

Reuses receptor_behavior_direct.py's test_receptor_vs_size (eccentricity
is a linear per-unit outcome, structurally identical design, no
covariate) UNMODIFIED for the eccentricity arm. Polar angle is circular
(0-360 deg wraparound), so uses the Mardia/Fisher-Lee circular-linear
correlation coefficient (same formula as run_claim2_polar_angle.py,
re-verified here against a fresh fixture) between the fixed 15-length...
per-receptor 8-length density vector (linear) and each subject's own
8-length polar-angle vector (circular), with the SAME spatial-permutation
design as everything else in this project (permute which receptor value
maps to which of the 8 units, not which subject's angle maps to which
unit, since receptor density -- not angle -- is what's fixed here).

Own FDR families throughout, kept separate from every pre-specified claim
and from today's other exploratory receptor work:
  1a. 15 receptors x eccentricity = 15 tests.
  1b. 15 receptors x polar angle = 15 tests.
  2a. 3 composite axes (PC1/PC2/E-I) x eccentricity = 3 tests.
  2b. 3 composite axes x polar angle = 3 tests.
Run on both atlases (hoc_region_retinotopy/{schaefer200,glasser}.csv).

Usage:
    python3 receptor_vs_retinotopy.py \
        --receptor-csv receptor-densities-deduped.csv \
        --retinotopy-schaefer hoc_region_retinotopy/schaefer200.csv \
        --retinotopy-glasser hoc_region_retinotopy/glasser.csv \
        --pca-scores-csv receptor_architecture/receptor_pca_scores.csv \
        --ei-balance-csv receptor_architecture/receptor_ei_balance_index.csv \
        --out-dir receptor_vs_retinotopy/
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# This script depends on run_claim1_analysis.py, which lives in a sibling
# scripts/ subdirectory in this repository layout (receptor_behavior_direct
# is in this same directory, so no extra path is needed for that import).
# Resolve the sibling path relative to this file's own location rather than
# hardcoding an absolute path, so the repo works regardless of where it is
# cloned.
_SIBLING_DIR = Path(__file__).resolve().parent.parent / "05_receptor_architecture_vs_controllability"
if _SIBLING_DIR.is_dir():
    sys.path.insert(0, str(_SIBLING_DIR))

import run_claim1_analysis as c1
from receptor_behavior_direct import test_receptor_vs_size, bh_fdr_rows, write_csv

RECEPTOR_ORDER = ["AMPA", "kainate", "NMDA", "GABAA", "GABAB", "BZ", "M1", "M2", "M3",
                   "alpha1", "alpha2", "alpha4beta2", "5-HT1A", "5-HT2", "D1"]


# ---------------------------------------------------------------------------
# Circular-linear arm (polar angle)
# ---------------------------------------------------------------------------

def circular_linear_corr(theta_deg, x):
    """Mardia/Fisher-Lee circular-linear correlation, r_cl^2 in [0,1], no
    sign. Same formula as run_claim2_polar_angle.py, re-verified below."""
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    x = np.asarray(x, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    if np.std(c) == 0 or np.std(s) == 0 or np.std(x) == 0:
        return np.nan
    r_xc = stats.pearsonr(x, c)[0]
    r_xs = stats.pearsonr(x, s)[0]
    r_cs = stats.pearsonr(c, s)[0]
    denom = 1 - r_cs ** 2
    if denom <= 0:
        return np.nan
    r2 = (r_xc ** 2 + r_xs ** 2 - 2 * r_xc * r_xs * r_cs) / denom
    return float(np.clip(r2, 0, 1))


def load_angle_data(retinotopy_csv, min_resultant_r=0.5):
    """Returns subj_angle[subject][unit] = angle_deg, restricted to units
    whose mean_resultant_r clears min_resultant_r (unreliable circular
    summaries excluded, same practice as run_claim2_polar_angle.py)."""
    subj_angle = {}
    n_dropped = 0
    with open(retinotopy_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["mean_polar_angle_deg"] in ("", None):
                continue
            r = float(row["mean_resultant_r"])
            if r < min_resultant_r:
                n_dropped += 1
                continue
            subj_angle.setdefault(row["subject"], {})[row["hoc_name"]] = float(row["mean_polar_angle_deg"])
    print(f"  (dropped {n_dropped} unit-entries with mean_resultant_r < {min_resultant_r})")
    return subj_angle


def test_receptor_vs_angle(receptor_x, subj_angle, subjects, n_permutations, rng):
    """Vectorized (same performance pattern as run_claim2_polar_angle.py's
    permuted_circular_linear_r2_vectorized -- a naive per-permutation,
    per-subject Python loop was measured too slow, see module change
    notes). Here the permutation axis is the FIXED receptor vector across
    the 8 units (not per-subject angles, since angle is subject-varying
    and receptor is fixed) -- for a given subject, cos/sin/r_cs of that
    subject's own theta are fixed across all permutations, so they're
    precomputed once per subject and only r_xc/r_xs (which depend on the
    permuted receptor vector) are recomputed per permutation, vectorized
    across all n_permutations at once via pearson_rowwise."""
    used = [s for s in subjects if all(u in subj_angle.get(s, {}) for u in c1.UNIT_ORDER)]
    if len(used) < 5:
        return {"n_subjects_valid": len(used), "note": "too few valid subjects, skipped"}

    thetas_deg = np.array([[subj_angle[s][u] for u in c1.UNIT_ORDER] for s in used])
    thetas = np.radians(thetas_deg)
    cos_all = np.cos(thetas)  # (n_subj, n_units)
    sin_all = np.sin(thetas)
    r_cs_all = np.array([np.corrcoef(cos_all[k], sin_all[k])[0, 1] for k in range(len(used))])
    denom_all = 1 - r_cs_all ** 2

    observed_r2 = np.empty(len(used))
    for k in range(len(used)):
        if denom_all[k] <= 0:
            observed_r2[k] = np.nan
            continue
        r_xc = stats.pearsonr(receptor_x, cos_all[k])[0]
        r_xs = stats.pearsonr(receptor_x, sin_all[k])[0]
        observed_r2[k] = np.clip((r_xc ** 2 + r_xs ** 2 - 2 * r_xc * r_xs * r_cs_all[k]) / denom_all[k], 0, 1)
    observed_mean_r2 = float(np.nanmean(observed_r2))

    n_units = len(receptor_x)
    keys = rng.random((n_permutations, n_units))
    perm_idx = np.argsort(keys, axis=1)
    X_perm = receptor_x[perm_idx]  # (n_permutations, n_units)

    per_subject_null_mean = np.zeros(n_permutations)
    n_valid_subj = 0
    for k in range(len(used)):
        if denom_all[k] <= 0:
            continue
        cos_tile = np.tile(cos_all[k], (n_permutations, 1))
        sin_tile = np.tile(sin_all[k], (n_permutations, 1))
        r_xc = c1.pearson_rowwise(X_perm, cos_tile)
        r_xs = c1.pearson_rowwise(X_perm, sin_tile)
        r2 = np.clip((r_xc ** 2 + r_xs ** 2 - 2 * r_xc * r_xs * r_cs_all[k]) / denom_all[k], 0, 1)
        per_subject_null_mean += r2
        n_valid_subj += 1
    if n_valid_subj == 0:
        return {"n_subjects_valid": len(used), "note": "no subject had a usable circular fit, skipped"}
    null_stats = per_subject_null_mean / n_valid_subj
    null_stats = null_stats[np.isfinite(null_stats)]

    # one-sided: r_cl^2 >= 0 by construction, larger = more relationship (matches run_claim2_polar_angle.py)
    perm_p = (np.sum(null_stats >= observed_mean_r2) + 1) / (len(null_stats) + 1)
    return {
        "n_subjects_valid": len(used), "mean_circular_linear_r2": observed_mean_r2,
        "perm_p_value": float(perm_p), "n_permutations_used": int(len(null_stats)), "note": "",
    }


def load_pca_scores(path):
    rows = list(csv.DictReader(open(path)))
    out = {"PC1_score": {}, "PC2_score": {}}
    for r in rows:
        for comp in list(out.keys()):
            if comp in r and r[comp] != "":
                out[comp][r["unit"]] = float(r[comp])
    return out


def load_ei_balance(path):
    return {r["unit"]: float(r["ei_log_ratio"]) for r in csv.DictReader(open(path))}


def load_ecc_and_size(retinotopy_csv):
    """Returns subj_ecc[subject][unit] = eccentricity_deg, shaped exactly
    like test_receptor_vs_size expects (same shape as n_hoc_vertices)."""
    subj_ecc = {}
    for row in csv.DictReader(open(retinotopy_csv)):
        if row["mean_eccentricity_deg"] in ("", None):
            continue
        subj_ecc.setdefault(row["subject"], {})[row["hoc_name"]] = float(row["mean_eccentricity_deg"])
    return subj_ecc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor-csv", required=True)
    ap.add_argument("--retinotopy-schaefer", required=True)
    ap.add_argument("--retinotopy-glasser", required=True)
    ap.add_argument("--pca-scores-csv", required=True)
    ap.add_argument("--ei-balance-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    receptor_vecs = c1.load_receptor_vectors(args.receptor_csv)
    pca_scores = load_pca_scores(args.pca_scores_csv)
    ei_scores = load_ei_balance(args.ei_balance_csv)
    predictors = {r: {u: receptor_vecs[r][u] for u in c1.UNIT_ORDER} for r in RECEPTOR_ORDER}
    predictors.update({"PC1": pca_scores["PC1_score"], "PC2": pca_scores["PC2_score"], "EI_balance": ei_scores})

    atlases = {"schaefer200": args.retinotopy_schaefer, "glasser": args.retinotopy_glasser}

    all_ecc_receptor, all_ecc_composite = [], []
    all_ang_receptor, all_ang_composite = [], []

    for atlas, retinotopy_csv in atlases.items():
        print(f"\n=== {atlas} ===")
        subj_ecc = load_ecc_and_size(retinotopy_csv)
        subjects = sorted(subj_ecc.keys())
        print(f"Eccentricity: {len(subjects)} subjects with data")
        subj_angle = load_angle_data(retinotopy_csv)

        for name, vec in predictors.items():
            receptor_x = np.array([vec[u] for u in c1.UNIT_ORDER])
            is_composite = name in ("PC1", "PC2", "EI_balance")

            ecc_stat, _ = test_receptor_vs_size(receptor_x, subj_ecc, subjects, args.n_permutations, rng)
            row = {"atlas": atlas, "predictor": name}
            row.update(ecc_stat)
            (all_ecc_composite if is_composite else all_ecc_receptor).append(row)

            ang_stat = test_receptor_vs_angle(receptor_x, subj_angle, sorted(subj_angle.keys()),
                                               args.n_permutations, rng)
            row2 = {"atlas": atlas, "predictor": name}
            row2.update(ang_stat)
            (all_ang_composite if is_composite else all_ang_receptor).append(row2)

    def fdr_and_write(rows, path, pkey="perm_p_value"):
        rows = bh_fdr_rows(rows, pkey=pkey)
        fn = ["atlas", "predictor", "n_subjects_valid"]
        extra = [k for k in rows[0].keys() if k not in fn] if rows else []
        write_csv(path, rows, fn + extra)
        return rows

    r1a = fdr_and_write(all_ecc_receptor, out_dir / "receptor_vs_eccentricity_results.csv")
    r1b = fdr_and_write(all_ang_receptor, out_dir / "receptor_vs_polarangle_results.csv")
    r2a = fdr_and_write(all_ecc_composite, out_dir / "composite_vs_eccentricity_results.csv")
    r2b = fdr_and_write(all_ang_composite, out_dir / "composite_vs_polarangle_results.csv")

    print("\n=== Summary (permutation-corrected FDR q < 0.05) ===")
    for label, rows in [("receptor vs eccentricity", r1a), ("receptor vs polar angle", r1b),
                         ("composite vs eccentricity", r2a), ("composite vs polar angle", r2b)]:
        sig = [r for r in rows if r.get("q_value_fdr_perm") is not None and r["q_value_fdr_perm"] < 0.05]
        print(f"  {label}: {len(sig)}/{len(rows)} significant")
        for r in sorted(rows, key=lambda x: x.get("perm_p_value", 1) if x.get("perm_p_value") is not None else 1)[:3]:
            if r.get("perm_p_value") is not None:
                metric_key = "mean_raw_r" if "mean_raw_r" in r else "mean_circular_linear_r2"
                print(f"    {r['atlas']}/{r['predictor']}: {metric_key}={r.get(metric_key, float('nan')):.3f}, "
                      f"perm_p={r['perm_p_value']:.4g}, q={r['q_value_fdr_perm']:.4g}")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
