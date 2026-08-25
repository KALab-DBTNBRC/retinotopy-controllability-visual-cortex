#!/usr/bin/env python3
"""
EXPLORATORY / POST-HOC (2026-08-16), addiction-relevant cohort (N=20) only.
User asked to dig further into the full SSAGA alcohol + tobacco variable set
(beyond the 5 curated severity measures already tested in
run_addiction_group_analysis.py's dimensional arm) for any relationship to
controllability. Explicit instruction: "no sketchy stuff" -- meaning this
runs as ONE full FDR family across every variable tested, nothing dropped
or re-run under a different family to dodge correction, and any nominally
significant hit gets checked for collinearity with other tested variables
before being described as an independent finding (same discipline already
applied to the receptor-cluster results elsewhere in this project).

Variables tested here are drawn from addiction-master-table.xlsx's full
801-column restricted-data record (not the curated 23-column
addiction-cohort-selection.xlsx), EXCLUDING the 4 alcohol/tobacco variables
already tested in run_addiction_group_analysis.py's dimensional arm
(SSAGA_Alc_D4_Ab_Sx, SSAGA_Alc_D4_Dp_Sx, Total_Drinks_7days,
SSAGA_TB_Smoking_History) to avoid re-testing the same variable twice under
a different family name.

13 alcohol variables (SSAGA past-12-month and heaviest-period drinking
quantity/frequency measures, age of first use, weekday/weekend drink
averages) + 14 tobacco variables (FTND/HSI nicotine dependence scores, DSM
nicotine dependence criteria, cigarettes-per-day measures, age of first
cigarette, smoking duration, current-tobacco-use totals) = 27 variables x
8 controllability summaries (hoc/vis x schaefer200/glasser x avg_log/modal,
identical design used throughout this project) = 216 tests, ONE FDR family.

Tobacco DSM/FTND/HSI variables have n=11 (only subjects who completed the
SSAGA tobacco module, i.e. ever-smokers) -- reported with that n visible in
every row, not hidden.

Usage:
    python3 run_addiction_ssaga_deepdive.py \
        --script-dir /path/to/project \
        --master-table-xlsx addiction-master-table.xlsx \
        --add-hoc-schaefer Addiction_Cohort_Results/hoc_region_controllability_schaefer200_addiction20.csv \
        --add-hoc-glasser Addiction_Cohort_Results/hoc_region_controllability_glasser_addiction20.csv \
        --add-vis-schaefer Addiction_Cohort_Results/all_subjects_controllability_schaefer200_addiction20.csv \
        --add-vis-glasser Addiction_Cohort_Results/all_subjects_controllability_glasser_addiction20.csv \
        --out-csv Addiction_Cohort_Results/addiction_ssaga_deepdive.csv \
        --out-descriptives Addiction_Cohort_Results/addiction_ssaga_deepdive_descriptives.csv
"""
import argparse
import csv
import sys

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

ALCOHOL_COLUMNS = [
    "SSAGA_Alc_12_Drinks_Per_Day", "SSAGA_Alc_12_Frq", "SSAGA_Alc_12_Frq_5plus",
    "SSAGA_Alc_12_Frq_Drk", "SSAGA_Alc_12_Max_Drinks", "SSAGA_Alc_Age_1st_Use",
    "SSAGA_Alc_Hvy_Drinks_Per_Day", "SSAGA_Alc_Hvy_Frq", "SSAGA_Alc_Hvy_Frq_5plus",
    "SSAGA_Alc_Hvy_Frq_Drk", "SSAGA_Alc_Hvy_Max_Drinks",
    "Avg_Weekday_Drinks_7days", "Avg_Weekend_Drinks_7days",
]
TOBACCO_COLUMNS = [
    "SSAGA_FTND_Score", "SSAGA_HSI_Score", "SSAGA_TB_Age_1st_Cig",
    "SSAGA_TB_DSM_Difficulty_Quitting", "SSAGA_TB_DSM_Tolerance", "SSAGA_TB_DSM_Withdrawal",
    "SSAGA_TB_Hvy_CPD", "SSAGA_TB_Max_Cigs", "SSAGA_TB_Reg_CPD",
    "SSAGA_TB_Still_Smoking", "SSAGA_TB_Yrs_Since_Quit", "SSAGA_TB_Yrs_Smoked",
    "Total_Any_Tobacco_7days", "Total_Cigarettes_7days",
]
ALREADY_TESTED_ELSEWHERE = {
    "SSAGA_Alc_D4_Ab_Sx": "alc_abuse_symptoms (run_addiction_group_analysis.py dimensional arm)",
    "SSAGA_Alc_D4_Dp_Sx": "alc_dependence_symptoms (same)",
    "Total_Drinks_7days": "total_drinks_7days (same)",
    "SSAGA_TB_Smoking_History": "smoking_history_class (same)",
}


def load_master_table(xlsx_path, columns):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Subjects"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    idx = {h: i for i, h in enumerate(headers)}
    subj_col = idx["Subject"]
    out = {c: {} for c in columns}
    for r in range(2, ws.max_row + 1):
        row = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
        subject = str(int(row[subj_col]))
        for c in columns:
            v = row[idx[c]]
            if v is not None:
                try:
                    out[c][subject] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def run_one_pair(c_vals, s_vals, subjects):
    x, y = [], []
    for s in subjects:
        cv = c_vals.get(s)
        sv = s_vals.get(s)
        if cv is None or sv is None:
            continue
        x.append(cv)
        y.append(sv)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", default="/path/to/project")
    ap.add_argument("--master-table-xlsx", required=True)
    ap.add_argument("--add-hoc-schaefer", required=True)
    ap.add_argument("--add-hoc-glasser", required=True)
    ap.add_argument("--add-vis-schaefer", required=True)
    ap.add_argument("--add-vis-glasser", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-descriptives", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim3_analysis as c3

    all_columns = ALCOHOL_COLUMNS + TOBACCO_COLUMNS
    var_series = load_master_table(args.master_table_xlsx, all_columns)
    subjects = sorted(set().union(*[set(v.keys()) for v in var_series.values()]))
    print(f"{len(subjects)} subjects with at least one variable present")
    print(f"Testing {len(all_columns)} alcohol/tobacco SSAGA variables "
          f"({len(ALCOHOL_COLUMNS)} alcohol, {len(TOBACCO_COLUMNS)} tobacco), "
          f"excluding {len(ALREADY_TESTED_ELSEWHERE)} already tested elsewhere: "
          f"{list(ALREADY_TESTED_ELSEWHERE)}")

    hs = c3.load_hoc_summary(args.add_hoc_schaefer)
    hg = c3.load_hoc_summary(args.add_hoc_glasser)
    vs = c3.load_vis_summary(args.add_vis_schaefer, "schaefer200")
    vg = c3.load_vis_summary(args.add_vis_glasser, "glasser")

    controllability_series = {
        "avg_controllability_log__hoc__schaefer200": {s: hs[s][0] for s in hs if hs[s][0] is not None},
        "modal_controllability__hoc__schaefer200": {s: hs[s][1] for s in hs if hs[s][1] is not None},
        "avg_controllability_log__hoc__glasser": {s: hg[s][0] for s in hg if hg[s][0] is not None},
        "modal_controllability__hoc__glasser": {s: hg[s][1] for s in hg if hg[s][1] is not None},
        "avg_controllability_log__vis__schaefer200": {s: vs[s][0] for s in vs if vs[s][0] is not None},
        "modal_controllability__vis__schaefer200": {s: vs[s][1] for s in vs if vs[s][1] is not None},
        "avg_controllability_log__vis__glasser": {s: vg[s][0] for s in vg if vg[s][0] is not None},
        "modal_controllability__vis__glasser": {s: vg[s][1] for s in vg if vg[s][1] is not None},
    }

    results = []
    for c_key, c_vals in controllability_series.items():
        for v_key in all_columns:
            v_vals = var_series[v_key]
            stat = run_one_pair(c_vals, v_vals, subjects)
            category = "alcohol" if v_key in ALCOHOL_COLUMNS else "tobacco"
            row = {"controllability_summary": c_key, "variable": v_key, "category": category}
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

    fieldnames = ["controllability_summary", "variable", "category", "n_subjects", "pearson_r", "p_pearson",
                  "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    sig = [r for r in results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== {len(results)} tests, ONE exploratory FDR family. Significant at q<0.05: {len(sig)} ===")
    for r in sorted(results, key=lambda x: x.get("p_pearson", 1.0) if "p_pearson" in x else 1.0)[:15]:
        if "p_pearson" in r:
            print(f"  {r['controllability_summary']} vs {r['variable']} ({r['category']}): "
                  f"r={r['pearson_r']:.3f}, p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, n={r['n_subjects']}")

    # descriptives
    desc_rows = []
    for c in all_columns:
        vals = np.array(list(var_series[c].values()))
        desc_rows.append({
            "column": c, "category": "alcohol" if c in ALCOHOL_COLUMNS else "tobacco",
            "n_available": len(vals),
            "n_missing": len(subjects) - len(vals),
            "mean": round(float(vals.mean()), 3) if len(vals) else None,
            "std": round(float(vals.std()), 3) if len(vals) > 1 else 0.0,
            "min": float(vals.min()) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
        })
    pd.DataFrame(desc_rows).to_csv(args.out_descriptives, index=False)
    print(f"\nWritten: {args.out_csv}, {args.out_descriptives}")


if __name__ == "__main__":
    main()
