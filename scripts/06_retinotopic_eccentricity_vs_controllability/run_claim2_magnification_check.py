#!/usr/bin/env python3
"""
Claim 2, Phase B3 (real check, not assumption) -- does controlling for a
literature-validated cortical-magnification proxy (pRF receptive field size,
Harvey & Dumoulin 2011) change Claim 2's eccentricity-controllability result,
versus the original node-vertex-count covariate?

Preliminary real-data check (2026-08-13, before running this script) found
mean_rfsize_deg correlates with node mean eccentricity as the literature
predicts on Glasser (r=0.608, p<0.00001) but NOT on Schaefer-200 (r=-0.122,
p=0.57) -- likely because Schaefer's Vis-network parcels are defined by
resting-state functional-connectivity boundaries (mixing multiple true
visual areas per node), while Glasser's parcels are true single-area
anatomical boundaries where the classic areal RF-size-vs-eccentricity
relationship is preserved. This means rfsize is a validated magnification
proxy on Glasser specifically -- Schaefer's version is reported too, for
transparency, but flagged as not meeting the same validation bar.

Three covariate configurations tested per metric/atlas (multiple partial
correlation via residualization -- regress x and y each against the full
covariate matrix Z, correlate the residuals; reduces to the standard
first-order partial correlation formula when Z has one column, confirmed
against run_claim2_analysis.py's partial_corr() on the size-only case as a
consistency check before trusting the multi-covariate version):
  1. size only (n_vertices_total) -- the ORIGINAL Claim 2 covariate, for comparison
  2. rfsize only (mean_rfsize_deg) -- the new, literature-grounded proxy
  3. size + rfsize together -- most conservative, controls for both

Usage:
    python3 run_claim2_magnification_check.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --atlas glasser --out-csv claim2_magnification_check_glasser.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_prf_backbone_with_rfsize(path, min_vertices_per_node, atlas, is_vis_fn):
    """Like run_claim2_analysis.load_prf_backbone but also returns
    subj_rfsize[subject][node_label] and node_size (vertex count, unchanged)."""
    subj_ecc, subj_rfsize, node_size = {}, {}, {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["atlas"] != atlas:
                continue
            is_vis = (row["is_vis_network"] == "True") if atlas == "schaefer200" else is_vis_fn(row["backbone_label"])
            if not is_vis:
                continue
            node_size[row["backbone_label"]] = int(row["n_vertices_total"])
            if row["mean_eccentricity_deg"] == "" or row["mean_rfsize_deg"] == "":
                continue
            n_valid = int(row["n_vertices_valid_R2gt30"])
            if n_valid < min_vertices_per_node:
                continue
            s = row["subject"]
            subj_ecc.setdefault(s, {})[row["backbone_label"]] = float(row["mean_eccentricity_deg"])
            subj_rfsize.setdefault(s, {})[row["backbone_label"]] = float(row["mean_rfsize_deg"])
    return subj_ecc, subj_rfsize, node_size


def multi_partial_corr(x, y, Z):
    """Partial correlation of x,y controlling for covariate matrix Z (n x k).
    Residualizes x and y against [1, Z] via OLS, correlates residuals.
    Reduces to the standard first-order partial correlation when k=1
    (verified below as a consistency check against run_claim2_analysis.py's
    partial_corr() before trusting this on real data)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    Zfull = np.column_stack([np.ones(len(x))] + [np.asarray(Z)[:, i] for i in range(Z.shape[1])])
    if np.linalg.matrix_rank(Zfull) < Zfull.shape[1]:
        return np.nan
    coef_x, *_ = np.linalg.lstsq(Zfull, x, rcond=None)
    coef_y, *_ = np.linalg.lstsq(Zfull, y, rcond=None)
    resid_x = x - Zfull @ coef_x
    resid_y = y - Zfull @ coef_y
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return np.nan
    return stats.pearsonr(resid_x, resid_y)[0]


def consistency_check(c2):
    """Sanity check: multi_partial_corr with a single covariate column must
    match run_claim2_analysis.partial_corr() exactly (up to float precision)."""
    rng = np.random.default_rng(0)
    x, y, z = rng.normal(size=30), rng.normal(size=30), rng.normal(size=30)
    r1 = c2.partial_corr(x, y, z)
    r2 = multi_partial_corr(x, y, z.reshape(-1, 1))
    assert abs(r1 - r2) < 1e-9, f"multi_partial_corr disagrees with partial_corr: {r1} vs {r2}"
    print("consistency check passed: multi_partial_corr matches partial_corr() for k=1")


def run_config(subj_ecc, subj_controllability, covariate_dicts, subjects, min_valid_nodes,
                n_permutations, rng):
    """covariate_dicts: list of {subject: {node: value}} dicts (each a covariate)."""
    per_subject_partial_r, used_subjects = [], []
    for s in subjects:
        ecc_vec = subj_ecc.get(s, {})
        c_vec = subj_controllability.get(s, {})
        common_nodes = set(ecc_vec) & set(c_vec)
        for cd in covariate_dicts:
            common_nodes &= set(cd.get(s, {}))
        common_nodes = sorted(common_nodes)
        if len(common_nodes) < min_valid_nodes:
            continue
        x = np.array([ecc_vec[n] for n in common_nodes])
        y = np.array([c_vec[n] for n in common_nodes])
        Z = np.column_stack([[cd[s][n] for n in common_nodes] for cd in covariate_dicts])
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r_partial = multi_partial_corr(x, y, Z)
        if not np.isfinite(r_partial):
            continue
        per_subject_partial_r.append(r_partial)
        used_subjects.append(s)

    n_valid = len(used_subjects)
    if n_valid < 5:
        return {"n_subjects_valid": n_valid, "note": "too few valid subjects, skipped"}
    z_vals = np.arctanh(np.clip(per_subject_partial_r, -0.999999, 0.999999))
    t_stat, p_t = stats.ttest_1samp(z_vals, 0.0)
    return {
        "n_subjects_valid": n_valid,
        "mean_partial_r": float(np.mean(per_subject_partial_r)),
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim2_analysis as c2

    consistency_check(c2)

    subj_ecc, subj_rfsize, node_size = load_prf_backbone_with_rfsize(
        args.prf_backbone_csv, args.min_vertices_per_node, args.atlas, c2.is_glasser_vis_node
    )
    subj_avg_log, subj_modal = c2.load_controllability(args.controllability_csv)
    subjects = sorted(subj_ecc.keys())
    subj_size = {s: node_size for s in subjects}  # node_size is fixed-by-atlas, same dict for every subject

    rng = np.random.default_rng(42)
    results = []
    for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                 ("modal_controllability", subj_modal)]:
        for config_name, covariates in [
            ("size_only", [subj_size]),
            ("rfsize_only", [subj_rfsize]),
            ("size_and_rfsize", [subj_size, subj_rfsize]),
        ]:
            stat = run_config(subj_ecc, subj_c, covariates, subjects, args.min_valid_nodes, None, rng)
            row = {"atlas": args.atlas, "metric": metric_name, "covariate_config": config_name}
            row.update(stat)
            results.append(row)
            print(f"{args.atlas} {metric_name} [{config_name}]: n={stat.get('n_subjects_valid')}, "
                  f"partial_r={stat.get('mean_partial_r', float('nan')):.4f}, p={stat.get('p_ttest', float('nan')):.4g}")

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["atlas", "metric", "covariate_config", "n_subjects_valid",
                                           "mean_partial_r", "t_stat", "p_ttest", "note"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
