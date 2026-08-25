#!/usr/bin/env python3
"""
Claim 1 -- Monte Carlo power / minimum-detectable-effect-size analysis for
the 8-spatial-unit permutation design (hOc1/hOc2/hOc3v/hOc3d x
left/right). run_claim1_analysis.py's own docstring documents why this
design's small unit count (n=8) is a real vulnerability for FALSE
positives (fixed here via spatial_permutation_test) -- but nothing in the
project has yet asked the complementary question: given only 8 spatial
units, how big would a receptor's TRUE relationship to controllability
need to be before this design could reliably detect it at all? A clean
null result (Claim 1: lowest q=0.358) is only informative to the extent
this design has power to detect real effects in the first place.

This CANNOT use the standard closed-form correlation power formula
(power_analysis_correlation.py) -- that formula assumes the classic
"n independent (x,y) pairs" design. Claim 1's design is structurally
different: n=8 FIXED spatial units (not independent draws) tested via a
permutation test whose null distribution and power both depend on the
REAL per-subject controllability data's actual structure (46 subjects x 8
units, each subject correlated with the group-level receptor vector,
Fisher-z-averaged across subjects) -- there's no textbook formula for
this. Monte Carlo simulation using the REAL, already-validated pipeline
functions (spatial_permutation_test, vectorized_mean_fisher_z, imported
from run_claim1_analysis.py, not reimplemented) is the correct approach,
same reasoning as calibrate_claim4_interaction.py's use of simulation over
a parametric formula for this project's other small-N permutation design.

Method: for each of a range of TRUE injected effect sizes r_true (target
correlation with the group-mean regional controllability pattern -- the
shared "shape" a real receptor-controllability relationship would need to
track), synthesize many candidate 8-length receptor vectors as
  receptor_x = r_true * base_pattern + sqrt(1 - r_true^2) * noise
(base_pattern = z-scored across-subject mean of the real controllability
metric; noise ~ N(0,1) iid across the 8 units) -- by construction this
targets an expected correlation of r_true with the real, typical spatial
profile of controllability across the 8 hOc units, though at n=8 units the
REALIZED correlation for any one draw varies around that target (reported
alongside, for transparency, as the empirical achieved correlation). The
REAL per-subject Y/Z matrices (from real 46-subject data, not simulated)
are used throughout, so subject-level noise is genuine, not synthetic.
For each draw, the REAL spatial permutation test is run and whether it
detects q<0.05-equivalent (perm_p<0.05, single-test threshold since this
is a power analysis, not itself part of an FDR family) is recorded.
Empirical power at each r_true level = fraction of draws detected.

Calibration checked first (r_true=0, i.e. real receptor-density
independent of controllability's pattern): false-positive rate should be
close to the nominal 5% -- reusing the exact same real Y/Z data and test
code the real Claim 1 analysis used, so if this doesn't calibrate near 5%,
something is wrong with the harness, not just an expected small-N wobble.

Usage:
    python3 run_claim1_power_analysis.py \
        --script-dir /path/to/Frontiers_project \
        --region-controllability-csv hoc_region_controllability/schaefer200.csv \
        --n-reps 150 --n-permutations 2000 \
        --out-csv claim1_power_analysis_schaefer200.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def import_c1(script_dir):
    sys.path.insert(0, script_dir)
    import run_claim1_analysis as c1
    return c1


def build_real_YZ(c1, region_controllability_csv, metric_key):
    """metric_key: 'avg' or 'modal'. Returns (subjects_used, Y_mat, Z_mat)
    using the REAL 46-subject data, restricted to subjects with all 8
    spatial units present -- identical inclusion rule to
    run_one_receptor_metric, just without a specific receptor vector yet."""
    subj_avg, subj_modal, subj_size = c1.load_region_controllability(region_controllability_csv)
    subj_c = subj_avg if metric_key == "avg" else subj_modal
    subjects = sorted(subj_c.keys())
    y_rows, z_rows, used = [], [], []
    for s in subjects:
        c_vec = subj_c.get(s, {})
        size_vec = subj_size.get(s, {})
        if not all(u in c_vec for u in c1.UNIT_ORDER) or not all(u in size_vec for u in c1.UNIT_ORDER):
            continue
        y = np.array([c_vec[u] for u in c1.UNIT_ORDER])
        if np.std(y) == 0:
            continue
        y_rows.append(y)
        z_rows.append([size_vec[u] for u in c1.UNIT_ORDER])
        used.append(s)
    return used, np.array(y_rows), np.array(z_rows)


def make_base_pattern(Y_mat):
    """z-scored across-subject mean of the real controllability metric --
    the shared/typical regional 'shape' a real receptor-controllability
    relationship would need to track, at the group level."""
    mean_profile = Y_mat.mean(axis=0)
    return (mean_profile - mean_profile.mean()) / mean_profile.std()


def simulate_power_curve(c1, Y_mat, Z_mat, r_true_levels, n_reps, n_permutations, rng, alpha=0.05):
    base_pattern = make_base_pattern(Y_mat)
    n_units = Y_mat.shape[1]
    results = []
    for r_true in r_true_levels:
        n_detected = 0
        achieved_rs = []
        for _ in range(n_reps):
            noise = rng.normal(size=n_units)
            receptor_x = r_true * base_pattern + np.sqrt(max(0.0, 1 - r_true ** 2)) * noise
            achieved_r = float(np.corrcoef(receptor_x, base_pattern)[0, 1])
            achieved_rs.append(achieved_r)

            observed_stat = c1.vectorized_mean_fisher_z(receptor_x, Y_mat, Z_mat)
            if not np.isfinite(observed_stat):
                continue
            perm_p, _ = c1.spatial_permutation_test(receptor_x, Y_mat, Z_mat, observed_stat, n_permutations, rng)
            if np.isfinite(perm_p) and perm_p < alpha:
                n_detected += 1
        power = n_detected / n_reps
        results.append({
            "r_true_target": r_true,
            "mean_achieved_r_vs_base_pattern": float(np.mean(achieved_rs)),
            "n_reps": n_reps,
            "n_permutations_per_rep": n_permutations,
            "empirical_power": power,
        })
        print(f"  r_true={r_true:.2f} (achieved mean r={np.mean(achieved_rs):.3f}): "
              f"empirical power = {power:.3f} ({n_detected}/{n_reps} detected at p<{alpha})")
    return results


def interpolate_mdes(results, target_power=0.80):
    """Linear interpolation on the (r_true, power) curve to estimate the
    minimum detectable effect size at the target power -- simple, adequate
    given the coarse grid used (this is a diagnostic estimate, not a
    precision claim)."""
    rs = [r["r_true_target"] for r in results]
    ps = [r["empirical_power"] for r in results]
    for i in range(len(ps) - 1):
        if ps[i] <= target_power <= ps[i + 1]:
            if ps[i + 1] == ps[i]:
                return rs[i]
            frac = (target_power - ps[i]) / (ps[i + 1] - ps[i])
            return rs[i] + frac * (rs[i + 1] - rs[i])
    if ps and ps[-1] < target_power:
        return None  # never reached target power in the tested range
    if ps and ps[0] >= target_power:
        return rs[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--region-controllability-csv", required=True)
    ap.add_argument("--r-true-levels", default="0.0,0.2,0.4,0.6,0.8,0.95")
    ap.add_argument("--n-reps", type=int, default=150)
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    c1 = import_c1(args.script_dir)
    rng = np.random.default_rng(args.seed)
    r_true_levels = [float(x) for x in args.r_true_levels.split(",")]

    all_rows = []
    for metric_key, metric_label in [("avg", "average_controllability_log"), ("modal", "modal_controllability")]:
        used, Y_mat, Z_mat = build_real_YZ(c1, args.region_controllability_csv, metric_key)
        print(f"\n=== {metric_label}: n_subjects={len(used)} ===")
        results = simulate_power_curve(c1, Y_mat, Z_mat, r_true_levels, args.n_reps, args.n_permutations, rng)
        mdes = interpolate_mdes(results, target_power=0.80)
        print(f"  Minimum detectable effect (interpolated, 80% power): "
              f"{mdes:.3f}" if mdes is not None else "  Minimum detectable effect (80% power): not reached in tested range")
        for r in results:
            r["metric"] = metric_label
            r["n_subjects"] = len(used)
            r["mdes_80pct_power_interpolated"] = mdes
            all_rows.append(r)

    fieldnames = ["metric", "n_subjects", "r_true_target", "mean_achieved_r_vs_base_pattern",
                  "n_reps", "n_permutations_per_rep", "empirical_power", "mdes_80pct_power_interpolated"]
    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWritten to {out_csv}")


if __name__ == "__main__":
    main()
