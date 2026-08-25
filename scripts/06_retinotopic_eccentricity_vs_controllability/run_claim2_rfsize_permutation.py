#!/usr/bin/env python3
"""
Claim 2, Phase B3 permutation cross-check -- run_claim2_magnification_check.py's
parametric-only comparison showed controlling for rfsize (instead of node
vertex-count) makes the eccentricity-controllability effect noticeably
STRONGER on both atlases, dramatically so on Glasser. Given this project's
documented history of parametric tests overstating significance in these
per-subject spatial-correlation designs, this script reruns the size_only
vs. rfsize_only comparison through the SAME permutation-test machinery
run_claim2_analysis.py uses as its primary test (subject_permuted_partial_r,
per-subject-independent reshuffles), not just the parametric t-test, before
this dramatic strengthening is treated as a real result.

Usage:
    python3 run_claim2_rfsize_permutation.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --atlas glasser --out-csv claim2_rfsize_permutation_glasser.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_prf_backbone_with_rfsize(path, min_vertices_per_node, atlas, is_vis_fn):
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


def run_one_metric_persubject_z(c2, subj_ecc, subj_controllability, subj_z, subjects,
                                 min_valid_nodes, n_permutations, rng):
    """Same design as run_claim2_analysis.run_one_metric, but z comes from a
    PER-SUBJECT dict (subj_z[subject][node]) instead of a fixed node_size
    dict -- needed because rfsize, unlike vertex count, varies by subject."""
    per_subject_r, per_subject_partial_r, used_subjects, subject_arrays = [], [], [], []
    for s in subjects:
        ecc_vec = subj_ecc.get(s, {})
        c_vec = subj_controllability.get(s, {})
        z_vec = subj_z.get(s, {})
        common_nodes = sorted(set(ecc_vec) & set(c_vec) & set(z_vec))
        if len(common_nodes) < min_valid_nodes:
            continue
        x = np.array([ecc_vec[n] for n in common_nodes])
        y = np.array([c_vec[n] for n in common_nodes])
        z = np.array([z_vec[n] for n in common_nodes])
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r_raw = stats.pearsonr(x, y)[0]
        r_partial = c2.partial_corr(x, y, z)
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
    observed_mean_fisher_z = float(np.mean(z_vals))

    per_subject_null_z = []
    for (x, y, z) in subject_arrays:
        rp = c2.subject_permuted_partial_r(x, y, z, n_permutations, rng)
        zz = np.arctanh(np.clip(rp, -0.999999, 0.999999))
        per_subject_null_z.append(zz)
    null_stack = np.vstack(per_subject_null_z)
    null_mean_z = np.nanmean(null_stack, axis=0)
    null_mean_z = null_mean_z[np.isfinite(null_mean_z)]
    perm_p = (np.sum(np.abs(null_mean_z) >= abs(observed_mean_fisher_z)) + 1) / (len(null_mean_z) + 1) \
        if len(null_mean_z) > 0 else np.nan

    return {
        "n_subjects_valid": n_valid,
        "mean_raw_r": float(np.mean(per_subject_r)),
        "mean_partial_r": float(np.mean(per_subject_partial_r)),
        "mean_fisher_z": observed_mean_fisher_z,
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "perm_p_value": float(perm_p) if np.isfinite(perm_p) else None,
        "n_permutations_used": n_permutations,
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
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim2_analysis as c2

    subj_ecc, subj_rfsize, node_size = load_prf_backbone_with_rfsize(
        args.prf_backbone_csv, args.min_vertices_per_node, args.atlas, c2.is_glasser_vis_node
    )
    subj_avg_log, subj_modal = c2.load_controllability(args.controllability_csv)
    subjects = sorted(subj_ecc.keys())
    subj_size = {s: node_size for s in subjects}

    rng = np.random.default_rng(42)
    results = []
    for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                 ("modal_controllability", subj_modal)]:
        for config_name, subj_z in [("size_only", subj_size), ("rfsize_only", subj_rfsize)]:
            stat = run_one_metric_persubject_z(c2, subj_ecc, subj_c, subj_z, subjects,
                                                args.min_valid_nodes, args.n_permutations, rng)
            row = {"atlas": args.atlas, "metric": metric_name, "covariate_config": config_name}
            row.update(stat)
            results.append(row)
            print(f"{args.atlas} {metric_name} [{config_name}]: n={stat.get('n_subjects_valid')}, "
                  f"partial_r={stat.get('mean_partial_r', float('nan')):.4f}, "
                  f"perm_p={stat.get('perm_p_value')}, parametric_p={stat.get('p_ttest', float('nan')):.4g}")

    out_csv = Path(args.out_csv)
    fieldnames = ["atlas", "metric", "covariate_config", "n_subjects_valid", "mean_raw_r", "mean_partial_r",
                  "mean_fisher_z", "perm_p_value", "n_permutations_used", "t_stat", "p_ttest", "note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
