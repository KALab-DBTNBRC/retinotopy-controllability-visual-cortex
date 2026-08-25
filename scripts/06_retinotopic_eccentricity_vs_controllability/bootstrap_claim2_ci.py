#!/usr/bin/env python3
"""
Claim 2, Phase B4 -- bootstrap confidence interval on the mean partial-r
effect size (both metrics, both atlases), to accompany the permutation
p-values with a real interval estimate rather than just a point estimate.

Method: reuses run_claim2_analysis.py's own loading + per-subject partial-r
computation (not reimplemented, to guarantee this matches the actual
reported statistic exactly). For each metric/atlas, collects each used
subject's own partial-r (same n_subjects_valid/min_valid_nodes criteria as
the main analysis), then bootstrap-resamples SUBJECTS (not nodes) with
replacement 10,000 times, recomputing the mean Fisher-z (and back-
transformed r) each time -- subject is the real unit of replication here
(N=39-41), matching the permutation test's own subject-level design.
Percentile bootstrap CI (2.5/97.5) is reported alongside the mean.

Usage:
    python3 bootstrap_claim2_ci.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_schaefer200.csv \
        --atlas schaefer200 --min-valid-nodes 15 --min-vertices-per-node 5 \
        --n-boot 10000 --seed 42
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))


def import_claim2(script_dir):
    sys.path.insert(0, script_dir)
    import run_claim2_analysis as c2
    return c2


def per_subject_partial_r(c2, subj_ecc, subj_controllability, node_size, subjects, min_valid_nodes):
    """Reproduces run_claim2_analysis.run_one_metric's subject-selection and
    partial-r computation exactly, but returns the per-subject array instead
    of just summary stats -- needed for bootstrap resampling."""
    used_subjects, per_subject_r = [], []
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
        r_partial = c2.partial_corr(x, y, z)
        if not np.isfinite(r_partial):
            continue
        used_subjects.append(s)
        per_subject_r.append(r_partial)
    return used_subjects, np.array(per_subject_r)


def bootstrap_mean_r(per_subject_r, n_boot, rng):
    """Bootstrap-resample subjects with replacement, recompute mean Fisher-z
    -> back-transformed r each time. Returns (boot_r_array,)."""
    n = len(per_subject_r)
    z_vals = np.arctanh(np.clip(per_subject_r, -0.999999, 0.999999))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_z_means = z_vals[idx].mean(axis=1)
    boot_r = np.tanh(boot_z_means)
    return boot_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    c2 = import_claim2(args.script_dir)
    rng = np.random.default_rng(args.seed)

    subj_ecc, node_size = c2.load_prf_backbone(args.prf_backbone_csv, args.min_vertices_per_node, args.atlas)
    subj_avg_log, subj_modal = c2.load_controllability(args.controllability_csv)
    subjects = sorted(subj_ecc.keys())

    rows = []
    for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                 ("modal_controllability", subj_modal)]:
        used, per_r = per_subject_partial_r(c2, subj_ecc, subj_c, node_size, subjects, args.min_valid_nodes)
        boot_r = bootstrap_mean_r(per_r, args.n_boot, rng)
        ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
        observed_mean_r = float(np.tanh(np.mean(np.arctanh(np.clip(per_r, -0.999999, 0.999999)))))
        print(f"{args.atlas} {metric_name}: n={len(used)}, mean_partial_r={observed_mean_r:.4f}, "
              f"95% bootstrap CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rows.append({
            "atlas": args.atlas, "metric": metric_name, "n_subjects": len(used),
            "mean_partial_r": observed_mean_r, "ci_lower_2.5": float(ci_lo), "ci_upper_97.5": float(ci_hi),
            "n_boot": args.n_boot,
        })

    import csv
    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
