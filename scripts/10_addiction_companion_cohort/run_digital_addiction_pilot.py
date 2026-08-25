#!/usr/bin/env python3
"""
PILOT / EXPLORATORY (2026-08-16). Purpose: honest preliminary evidence for
the rationale of a NEW planned study on addictive digital behaviours
(problematic internet/gaming/smartphone use), which this project's HCP
restricted data cannot test directly -- HCP_RETINOTOPY_FULLCOHORT.csv's
full 801 columns were checked and contain NO screen-time, internet-use,
gaming, smartphone-use, or digital-media variable of any kind (confirmed by
exhaustive keyword search, 2026-08-16).

What HCP DOES have that the addiction literature treats as trans-diagnostic
behavioral proxies relevant to digital/behavioral addictions specifically
(delay discounting is the standard impulsivity/reward-devaluation measure
used across IGD, gambling disorder, and problematic-smartphone-use studies;
the Gambling Task is a direct reward/punishment-sensitivity behavioral
probe): Delay Discounting AUC (200-point and 40k-point conditions) and the
Gambling Task's reward/punishment percent-larger choices. Neither has been
tested against controllability anywhere else in this project (confirmed via
grep across all existing scripts/outputs) -- this is a genuinely new test,
not a re-run.

Design: 5 behavioral variables x 8 controllability summaries (hoc/vis x
schaefer200/glasser x avg_log/modal, this project's standard set) = 40
tests, run SEPARATELY for the main cohort (N=46) and the addiction-relevant
cohort (N=20) as two independent FDR families (not pooled -- these are two
different samples answering two different questions: does the effect exist
in the general/main sample, and does it look any different in the
substance-use-enriched sample).

This is explicitly PILOT/RATIONALE evidence for a study that does not yet
exist -- not a claim about digital behavioral addiction itself, which this
dataset cannot speak to. Reported honestly regardless of outcome, same
"no sketchy stuff" discipline as every other analysis in this project.

Usage:
    python3 run_digital_addiction_pilot.py \
        --script-dir /path/to/project \
        --full-cohort-csv "Received files/HCP_RETINOTOPY_FULLCOHORT.csv" \
        --main-cohort-xlsx cohort-selection.xlsx \
        --addiction-cohort-xlsx addiction-cohort-selection.xlsx \
        --main-hoc-schaefer hoc_region_controllability/schaefer200.csv \
        --main-hoc-glasser hoc_region_controllability/glasser.csv \
        --main-vis-schaefer "Received files/all_subjects_controllability_schaefer200.csv" \
        --main-vis-glasser "Received files/all_subjects_controllability_glasser.csv" \
        --add-hoc-schaefer Addiction_Cohort_Results/hoc_region_controllability_schaefer200_addiction20.csv \
        --add-hoc-glasser Addiction_Cohort_Results/hoc_region_controllability_glasser_addiction20.csv \
        --add-vis-schaefer Addiction_Cohort_Results/all_subjects_controllability_schaefer200_addiction20.csv \
        --add-vis-glasser Addiction_Cohort_Results/all_subjects_controllability_glasser_addiction20.csv \
        --out-main-csv digital_addiction_pilot_main46.csv \
        --out-addiction-csv digital_addiction_pilot_addiction20.csv
"""
import argparse
import csv
import sys

import numpy as np
import pandas as pd
from scipy import stats

BEHAVIOR_COLUMNS = {
    "delay_discounting_auc_200": "DDisc_AUC_200",
    "delay_discounting_auc_40k": "DDisc_AUC_40K",
    "gambling_perc_larger_overall": "Gambling_Task_Perc_Larger",
    "gambling_reward_perc_larger": "Gambling_Task_Reward_Perc_Larger",
    "gambling_punish_perc_larger": "Gambling_Task_Punish_Perc_Larger",
}


def bh_fdr(pvals):
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


def load_behavior(full_csv, cohort_xlsx):
    full = pd.read_csv(full_csv)
    full["Subject"] = full["Subject"].astype(int)
    cohort = pd.read_excel(cohort_xlsx, sheet_name="Cohort")
    ids = cohort["Subject ID"].astype(int).tolist()
    sub = full[full["Subject"].isin(ids)]
    out = {k: {} for k in BEHAVIOR_COLUMNS}
    for _, row in sub.iterrows():
        s = str(int(row["Subject"]))
        for key, col in BEHAVIOR_COLUMNS.items():
            v = row[col]
            if pd.notna(v):
                out[key][s] = float(v)
    return out, [str(i) for i in ids]


def run_one_pair(c_vals, b_vals, subjects):
    x, y = [], []
    for s in subjects:
        cv = c_vals.get(s)
        bv = b_vals.get(s)
        if cv is None or bv is None:
            continue
        x.append(cv)
        y.append(bv)
    n = len(x)
    if n < 5:
        return {"n_subjects": n, "note": "too few subjects, skipped"}
    x, y = np.array(x), np.array(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return {"n_subjects": n, "note": "zero variance, skipped"}
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return {"n_subjects": n, "pearson_r": float(r_p), "p_pearson": float(p_p),
            "spearman_rho": float(r_s), "p_spearman": float(p_s), "note": ""}


def run_family(behavior, subjects, controllability_series, label, out_csv):
    results = []
    for c_key, c_vals in controllability_series.items():
        for b_key in BEHAVIOR_COLUMNS:
            b_vals = behavior[b_key]
            stat = run_one_pair(c_vals, b_vals, subjects)
            row = {"controllability_summary": c_key, "behavior_variable": b_key}
            row.update(stat)
            results.append(row)

    p_values = [r["p_pearson"] for r in results if "p_pearson" in r]
    q_values = bh_fdr(p_values) if p_values else []
    qi = 0
    for r in results:
        if "p_pearson" in r:
            r["q_value_fdr"] = float(q_values[qi]); qi += 1
        else:
            r["q_value_fdr"] = None

    fieldnames = ["controllability_summary", "behavior_variable", "n_subjects", "pearson_r", "p_pearson",
                  "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    sig = [r for r in results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== {label}: {len(results)} tests, 1 FDR family. Significant at q<0.05: {len(sig)} ===")
    for r in sorted(results, key=lambda x: x.get("p_pearson", 1.0) if "p_pearson" in x else 1.0)[:10]:
        if "p_pearson" in r:
            print(f"  {r['controllability_summary']} vs {r['behavior_variable']}: "
                  f"r={r['pearson_r']:.3f}, p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, n={r['n_subjects']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", default="/path/to/project")
    ap.add_argument("--full-cohort-csv", required=True)
    ap.add_argument("--main-cohort-xlsx", required=True)
    ap.add_argument("--addiction-cohort-xlsx", required=True)
    ap.add_argument("--main-hoc-schaefer", required=True)
    ap.add_argument("--main-hoc-glasser", required=True)
    ap.add_argument("--main-vis-schaefer", required=True)
    ap.add_argument("--main-vis-glasser", required=True)
    ap.add_argument("--add-hoc-schaefer", required=True)
    ap.add_argument("--add-hoc-glasser", required=True)
    ap.add_argument("--add-vis-schaefer", required=True)
    ap.add_argument("--add-vis-glasser", required=True)
    ap.add_argument("--out-main-csv", required=True)
    ap.add_argument("--out-addiction-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim3_analysis as c3

    def load_all_series(hoc_schaefer, hoc_glasser, vis_schaefer, vis_glasser):
        hs = c3.load_hoc_summary(hoc_schaefer)
        hg = c3.load_hoc_summary(hoc_glasser)
        vs = c3.load_vis_summary(vis_schaefer, "schaefer200")
        vg = c3.load_vis_summary(vis_glasser, "glasser")
        return {
            "avg_controllability_log__hoc__schaefer200": {s: hs[s][0] for s in hs if hs[s][0] is not None},
            "modal_controllability__hoc__schaefer200": {s: hs[s][1] for s in hs if hs[s][1] is not None},
            "avg_controllability_log__hoc__glasser": {s: hg[s][0] for s in hg if hg[s][0] is not None},
            "modal_controllability__hoc__glasser": {s: hg[s][1] for s in hg if hg[s][1] is not None},
            "avg_controllability_log__vis__schaefer200": {s: vs[s][0] for s in vs if vs[s][0] is not None},
            "modal_controllability__vis__schaefer200": {s: vs[s][1] for s in vs if vs[s][1] is not None},
            "avg_controllability_log__vis__glasser": {s: vg[s][0] for s in vg if vg[s][0] is not None},
            "modal_controllability__vis__glasser": {s: vg[s][1] for s in vg if vg[s][1] is not None},
        }

    main_behavior, main_subjects = load_behavior(args.full_cohort_csv, args.main_cohort_xlsx)
    add_behavior, add_subjects = load_behavior(args.full_cohort_csv, args.addiction_cohort_xlsx)
    print(f"Main cohort N={len(main_subjects)}, addiction cohort N={len(add_subjects)}")

    main_series = load_all_series(args.main_hoc_schaefer, args.main_hoc_glasser,
                                   args.main_vis_schaefer, args.main_vis_glasser)
    add_series = load_all_series(args.add_hoc_schaefer, args.add_hoc_glasser,
                                  args.add_vis_schaefer, args.add_vis_glasser)

    run_family(main_behavior, main_subjects, main_series, "MAIN cohort (N=46)", args.out_main_csv)
    run_family(add_behavior, add_subjects, add_series, "ADDICTION cohort (N=20)", args.out_addiction_csv)

    print(f"\nWritten: {args.out_main_csv}, {args.out_addiction_csv}")


if __name__ == "__main__":
    main()
