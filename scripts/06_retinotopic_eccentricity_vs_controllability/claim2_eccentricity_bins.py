#!/usr/bin/env python3
"""
Claim 2, Phase C3 -- eccentricity-binned check: is the eccentricity-
controllability relationship monotonic/roughly linear across the tested
range, or driven by a handful of extreme-eccentricity nodes/subjects?

Method (explicitly DESCRIPTIVE/exploratory, not a replacement for the
primary permutation test -- pools within- and between-subject variance
together, which the primary per-subject-then-group design deliberately
avoids): pools every (subject, node) pair from subjects that passed Claim
2's own min-valid-nodes threshold into one long table, size-residualizes
controllability against node size (z) via simple OLS (removing the same
size confound the primary partial correlation controls for), bins by
eccentricity quartile (computed on the pooled distribution), and reports
each bin's mean size-residualized controllability +/- SEM. Also fits a
pooled linear vs. linear+quadratic OLS (on the size-residualized values) and
compares AIC to check whether a curved relationship fits meaningfully
better than a straight line.

Usage:
    python3 claim2_eccentricity_bins.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_schaefer200.csv \
        --atlas schaefer200 --out-csv claim2_eccentricity_bins_schaefer200.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim2_analysis as c2

    subj_ecc, node_size = c2.load_prf_backbone(args.prf_backbone_csv, args.min_vertices_per_node, args.atlas)
    subj_avg_log, subj_modal = c2.load_controllability(args.controllability_csv)
    subjects = sorted(subj_ecc.keys())

    results = []
    for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                 ("modal_controllability", subj_modal)]:
        # pool (subject, node) rows for subjects meeting the real min-valid-nodes bar
        pooled_x, pooled_y, pooled_z, pooled_subj = [], [], [], []
        for s in subjects:
            ecc_vec = subj_ecc.get(s, {})
            c_vec = subj_c.get(s, {})
            common_nodes = sorted(set(ecc_vec) & set(c_vec) & set(node_size))
            if len(common_nodes) < args.min_valid_nodes:
                continue
            for n in common_nodes:
                pooled_x.append(ecc_vec[n])
                pooled_y.append(c_vec[n])
                pooled_z.append(node_size[n])
                pooled_subj.append(s)

        x = np.array(pooled_x)
        y = np.array(pooled_y)
        z = np.array(pooled_z)
        n_rows = len(x)
        n_subj = len(set(pooled_subj))

        # size-residualize y against z (simple OLS, same size confound as the primary partial corr)
        Z = np.column_stack([np.ones_like(z), z])
        coef_z, *_ = np.linalg.lstsq(Z, y, rcond=None)
        y_resid = y - Z @ coef_z

        # pooled linear vs linear+quadratic fit on (x -> y_resid), compare AIC
        def fit_aic(X, y_):
            n = len(y_)
            k = X.shape[1]
            coef, *_ = np.linalg.lstsq(X, y_, rcond=None)
            resid = y_ - X @ coef
            rss = np.sum(resid ** 2)
            aic = n * np.log(rss / n) + 2 * k
            return aic, coef

        X_lin = np.column_stack([np.ones_like(x), x])
        X_quad = np.column_stack([np.ones_like(x), x, x ** 2])
        aic_lin, coef_lin = fit_aic(X_lin, y_resid)
        aic_quad, coef_quad = fit_aic(X_quad, y_resid)

        # eccentricity quartile bins on the pooled distribution
        quantiles = np.percentile(x, np.linspace(0, 100, args.n_bins + 1))
        bin_idx = np.clip(np.digitize(x, quantiles[1:-1]), 0, args.n_bins - 1)

        print(f"\n=== {args.atlas} {metric_name}: {n_rows} pooled (subject,node) rows, {n_subj} subjects ===")
        print(f"  AIC linear={aic_lin:.2f}, AIC linear+quadratic={aic_quad:.2f} "
              f"(lower is better; quadratic 'wins' by more than ~2 AIC points would suggest real curvature)")
        for b in range(args.n_bins):
            mask = bin_idx == b
            bin_mean_ecc = float(np.mean(x[mask]))
            bin_mean_resid = float(np.mean(y_resid[mask]))
            bin_sem = float(np.std(y_resid[mask], ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else float("nan")
            print(f"  bin {b+1}/{args.n_bins}: mean_ecc={bin_mean_ecc:.2f} deg, n={mask.sum()}, "
                  f"mean_size_resid_controllability={bin_mean_resid:.4f} +/- {bin_sem:.4f} SEM")
            results.append({
                "atlas": args.atlas, "metric": metric_name, "bin": b + 1, "n_bins": args.n_bins,
                "bin_mean_eccentricity_deg": bin_mean_ecc, "n_rows_in_bin": int(mask.sum()),
                "mean_size_residualized_controllability": bin_mean_resid, "sem": bin_sem,
                "aic_linear": aic_lin, "aic_linear_plus_quadratic": aic_quad,
                "n_pooled_rows": n_rows, "n_subjects": n_subj,
            })

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWritten to {out_csv}")


if __name__ == "__main__":
    main()
