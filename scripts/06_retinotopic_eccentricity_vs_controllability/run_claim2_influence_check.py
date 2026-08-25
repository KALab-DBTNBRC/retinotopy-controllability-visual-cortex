#!/usr/bin/env python3
"""
Claim 2 -- outlier/influence check (leave-one-subject-out, LOSO).

Motivation: Claim 2's per-subject spatial permutation design (see
run_claim1_analysis.py / run_claim2_analysis.py docstrings) makes the
SUBJECT the real unit of replication (n=39-45, not the within-subject node
count). Nothing built so far checks whether the reported mean partial-r is
being driven by a small number of unusually high-leverage subjects rather
than a broad, consistent effect. This script answers that directly for
every metric x atlas x covariate-config combination reported in the paper:

  - covariate_config="size": the original node-vertex-count covariate
    (Claim 2's original headline numbers, Schaefer r=0.248/-0.234,
    Glasser r=0.138/-0.067).
  - covariate_config="rfsize": the pRF receptive-field-size covariate
    (Phase B3's magnification-corrected, strengthened numbers, Schaefer
    r=0.348/-0.341, Glasser r=?/-0.434).

Method, reusing already-validated machinery rather than reimplementing it
(run_claim2_analysis.py's partial_corr/subject_permuted_partial_r; the
rfsize-covariate data loading mirrors run_claim2_rfsize_permutation.py's
load_prf_backbone_with_rfsize):

  1. Build each used subject's own partial-r (same subject-inclusion
     criteria as the main analysis: >= min_valid_nodes common nodes,
     nonzero variance).
  2. For each subject, recompute the mean Fisher-z (-> r) with that subject
     excluded (a closed-form jackknife, not a re-permutation -- cheap) and
     record the shift from the full-sample mean. This ranks subjects by
     influence on the point estimate.
  3. Take the single most-influential subject (largest |shift|) and
     actually drop them, then rerun the REAL permutation significance test
     (subject_permuted_partial_r, the project's own PRIMARY significance
     test) on the remaining n-1 subjects, to check whether the conclusion
     -- not just the point estimate -- survives losing that one subject.

Usage:
    python3 run_claim2_influence_check.py \
        --script-dir /path/to/Frontiers_project \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv-schaefer all_subjects_controllability_schaefer200.csv \
        --controllability-csv-glasser all_subjects_controllability_glasser.csv \
        --out-csv-summary claim2_influence_summary.csv \
        --out-csv-detail claim2_influence_detail.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def import_modules(script_dir):
    sys.path.insert(0, script_dir)
    import run_claim2_analysis as c2
    import run_claim2_rfsize_permutation as c2r
    return c2, c2r


def per_subject_arrays(partial_corr_fn, subj_x, subj_y, subj_z, subjects, min_valid_nodes):
    """Generic over whether z is a FIXED dict (node_size, broadcast the same
    to every subject) or a PER-SUBJECT dict (rfsize) -- both are passed in
    as subj_z[subject][node] here, caller is responsible for broadcasting
    the fixed-size case. Returns used_subjects, list of (x,y,z) arrays,
    and the per-subject partial-r array -- mirrors run_claim2_analysis.py's
    run_one_metric inner loop exactly (same inclusion criteria) so this is
    a genuine post-hoc diagnostic on the actual reported statistic, not an
    approximation of it."""
    used, arrays, per_r = [], [], []
    for s in subjects:
        xv, yv, zv = subj_x.get(s, {}), subj_y.get(s, {}), subj_z.get(s, {})
        common = sorted(set(xv) & set(yv) & set(zv))
        if len(common) < min_valid_nodes:
            continue
        x = np.array([xv[n] for n in common])
        y = np.array([yv[n] for n in common])
        z = np.array([zv[n] for n in common])
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rp = partial_corr_fn(x, y, z)
        if not np.isfinite(rp):
            continue
        used.append(s)
        arrays.append((x, y, z))
        per_r.append(rp)
    return used, arrays, np.array(per_r)


def loso_shifts(per_r):
    """Closed-form jackknife: for each subject, the mean Fisher-z excluding
    that subject, converted back to r, and the shift from the full-sample
    mean r. Vectorized (no loop over permutations, just arithmetic)."""
    z = np.arctanh(np.clip(per_r, -0.999999, 0.999999))
    n = len(z)
    full_mean_z = np.mean(z)
    full_mean_r = float(np.tanh(full_mean_z))
    total = np.sum(z)
    loso_mean_z = (total - z) / (n - 1)  # vectorized: one value per excluded subject
    loso_mean_r = np.tanh(loso_mean_z)
    shifts = full_mean_r - loso_mean_r
    return full_mean_r, loso_mean_r, shifts


def rerun_permutation(partial_corr_fn, subject_permuted_partial_r_fn, arrays, n_permutations, rng):
    """Reruns the real subject-level spatial permutation test (the
    project's PRIMARY significance test) on an arbitrary subset of
    subjects' (x,y,z) arrays -- used both for the full-sample baseline and
    for the n-1 (most-influential-subject-excluded) check, so both numbers
    come from the identical code path."""
    per_r = np.array([partial_corr_fn(x, y, z) for (x, y, z) in arrays])
    z_vals = np.arctanh(np.clip(per_r, -0.999999, 0.999999))
    observed_mean_z = np.mean(z_vals)
    null_z = []
    for (x, y, z) in arrays:
        rp = subject_permuted_partial_r_fn(x, y, z, n_permutations, rng)
        null_z.append(np.arctanh(np.clip(rp, -0.999999, 0.999999)))
    null_stack = np.vstack(null_z)
    null_mean_z = np.nanmean(null_stack, axis=0)
    null_mean_z = null_mean_z[np.isfinite(null_mean_z)]
    perm_p = (np.sum(np.abs(null_mean_z) >= abs(observed_mean_z)) + 1) / (len(null_mean_z) + 1) \
        if len(null_mean_z) > 0 else np.nan
    return float(np.tanh(observed_mean_z)), float(perm_p) if np.isfinite(perm_p) else None


def run_one_combo(c2, c2r, atlas, metric_name, subj_c, subj_x, subj_z, subjects,
                   min_valid_nodes, n_permutations, rng):
    used, arrays, per_r = per_subject_arrays(c2.partial_corr, subj_x, subj_c, subj_z, subjects, min_valid_nodes)
    n = len(used)
    if n < 5:
        return None, []

    full_mean_r, loso_mean_r, shifts = loso_shifts(per_r)
    order = np.argsort(-np.abs(shifts))  # descending |shift|

    detail_rows = [{
        "atlas": atlas, "metric": metric_name, "subject": used[i],
        "own_partial_r": float(per_r[i]), "loso_mean_r_excluding_self": float(loso_mean_r[i]),
        "shift_in_group_mean_r": float(shifts[i]),
    } for i in range(n)]

    baseline_mean_r, baseline_perm_p = rerun_permutation(
        c2.partial_corr, c2.subject_permuted_partial_r, arrays, n_permutations, rng)

    top_idx = int(order[0])
    top_subject = used[top_idx]
    top_shift = float(shifts[top_idx])
    reduced_arrays = [arrays[i] for i in range(n) if i != top_idx]
    reduced_mean_r, reduced_perm_p = rerun_permutation(
        c2.partial_corr, c2.subject_permuted_partial_r, reduced_arrays, n_permutations, rng)

    summary = {
        "atlas": atlas, "metric": metric_name, "n_subjects": n,
        "full_sample_mean_partial_r": full_mean_r,
        "baseline_perm_p_rechecked": baseline_perm_p,
        "most_influential_subject": top_subject,
        "max_abs_shift_in_mean_r": abs(top_shift),
        "mean_partial_r_excluding_most_influential": reduced_mean_r,
        "perm_p_excluding_most_influential": reduced_perm_p,
        "conclusion_survives_exclusion": (
            (baseline_perm_p is not None and reduced_perm_p is not None) and
            ((baseline_perm_p < 0.05) == (reduced_perm_p < 0.05))
        ),
    }
    return summary, detail_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv-schaefer", required=True)
    ap.add_argument("--controllability-csv-glasser", required=True)
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv-summary", required=True)
    ap.add_argument("--out-csv-detail", required=True)
    args = ap.parse_args()

    c2, c2r = import_modules(args.script_dir)
    rng = np.random.default_rng(args.seed)

    all_summary, all_detail = [], []

    controllability_csvs = {"schaefer200": args.controllability_csv_schaefer, "glasser": args.controllability_csv_glasser}

    for atlas, cpath in controllability_csvs.items():
        subj_avg_log, subj_modal = c2.load_controllability(cpath)
        subj_ecc, node_size = c2.load_prf_backbone(args.prf_backbone_csv, args.min_vertices_per_node, atlas)
        subjects = sorted(subj_ecc.keys())
        subj_size_broadcast = {s: node_size for s in subjects}

        is_vis_fn = c2.is_glasser_vis_node if atlas == "glasser" else None
        subj_ecc_rf, subj_rfsize, node_size_rf = c2r.load_prf_backbone_with_rfsize(
            args.prf_backbone_csv, args.min_vertices_per_node, atlas, is_vis_fn)

        for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                     ("modal_controllability", subj_modal)]:
            for config_name, xdata, zdata in [
                ("size", subj_ecc, subj_size_broadcast),
                ("rfsize", subj_ecc_rf, subj_rfsize),
            ]:
                summary, detail = run_one_combo(
                    c2, c2r, atlas, metric_name, subj_c, xdata, zdata,
                    sorted(xdata.keys()), args.min_valid_nodes, args.n_permutations, rng)
                if summary is None:
                    print(f"{atlas} {metric_name} [{config_name}]: too few subjects, skipped")
                    continue
                summary["covariate_config"] = config_name
                for d in detail:
                    d["covariate_config"] = config_name
                all_summary.append(summary)
                all_detail.extend(detail)
                print(f"{atlas} {metric_name} [{config_name}]: n={summary['n_subjects']}, "
                      f"full_mean_r={summary['full_sample_mean_partial_r']:.4f}, "
                      f"baseline_perm_p={summary['baseline_perm_p_rechecked']:.4g}, "
                      f"most_influential={summary['most_influential_subject']} "
                      f"(shift={summary['max_abs_shift_in_mean_r']:.4f}), "
                      f"perm_p_excl={summary['perm_p_excluding_most_influential']:.4g}, "
                      f"conclusion_survives={summary['conclusion_survives_exclusion']}")

    summary_fields = ["atlas", "metric", "covariate_config", "n_subjects", "full_sample_mean_partial_r",
                       "baseline_perm_p_rechecked", "most_influential_subject", "max_abs_shift_in_mean_r",
                       "mean_partial_r_excluding_most_influential", "perm_p_excluding_most_influential",
                       "conclusion_survives_exclusion"]
    with open(args.out_csv_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in all_summary:
            w.writerow({k: r.get(k, "") for k in summary_fields})

    detail_fields = ["atlas", "metric", "covariate_config", "subject", "own_partial_r",
                      "loso_mean_r_excluding_self", "shift_in_group_mean_r"]
    with open(args.out_csv_detail, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for r in all_detail:
            w.writerow({k: r.get(k, "") for k in detail_fields})

    print(f"\nSummary ({len(all_summary)} combos) written to {args.out_csv_summary}")
    print(f"Detail ({len(all_detail)} rows) written to {args.out_csv_detail}")


if __name__ == "__main__":
    main()
