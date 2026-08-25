#!/usr/bin/env python3
"""
EXPLORATORY / POST-HOC (2026-08-14) -- NOT one of the 5 pre-specified claims.
PI requested an "addiction angle" given the lab's addiction focus. Real check
of the newly-supplied HCP restricted-data master table (all 800 columns)
against the final 46-subject cohort.

IMPORTANT METHODOLOGICAL NOTE: cohort-selection.xlsx's own exclusion criteria
(Methodology & Limitations sheet) explicitly excluded any subject with a
lifetime DSM-IV alcohol abuse diagnosis (SSAGA_Alc_D4_Ab_Dx), alcohol
dependence diagnosis (SSAGA_Alc_D4_Dp_Dx), or marijuana abuse/dependence
(SSAGA_Mj_Ab_Dep). This script confirms that by design those three columns
have zero variance in the 46-subject cohort (all "No" / 0). Clinical
diagnostic "addiction" cannot be analyzed in this cohort -- that variance was
deliberately engineered out at cohort-selection time.

What DOES have real variance and is analyzed here: graded/continuous
substance-use measures that were never part of the exclusion criteria --
current weekly alcohol intake, lifetime smoking history classification,
marijuana use frequency, subclinical alcohol-dependence symptom count, and
family history of drug/alcohol problems.

Design mirrors run_personality_analysis.py exactly: reuses
run_claim3_analysis.py's load_hoc_summary/load_vis_summary unmodified, same
8 controllability series (hoc/vis x schaefer200/glasser x avg_log/modal),
simple cross-subject Pearson/Spearman (N~46, no permutation test -- not a
small-spatial-unit design), one exploratory FDR family, clearly labeled
post-hoc.

Usage:
    python3 run_substance_analysis.py \
        --master-csv HCP_YA_subjects_2026_08_14_09_17_24.csv \
        --cohort-xlsx cohort-selection.xlsx \
        --hoc-schaefer hoc_region_controllability/schaefer200.csv \
        --hoc-glasser hoc_region_controllability/glasser.csv \
        --vis-schaefer all_subjects_controllability_schaefer200.csv \
        --vis-glasser all_subjects_controllability_glasser.csv \
        --out-csv substance_use_controllability.csv \
        --out-descriptives substance_use_descriptives.csv
"""
import argparse
import csv
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Variables with real, usable variance in the 46-subject cohort (identified
# by inspecting n_missing/n_unique/std across all ~177 substance-related
# columns in the master table). Sparse/near-constant columns (FTND/HSI score,
# illicit-drug times-used, toxicology positives) are reported descriptively
# only -- see out-descriptives -- not correlation-tested (too few non-zero
# cases at N=46 for a meaningful test).
SUBSTANCE_COLUMNS = {
    "total_drinks_7days": "Total_Drinks_7days",
    "avg_weekend_drinks_7days": "Avg_Weekend_Drinks_7days",
    "alc_12mo_drinks_per_day": "SSAGA_Alc_12_Drinks_Per_Day",
    "alc_heaviest_drinks_per_day": "SSAGA_Alc_Hvy_Drinks_Per_Day",
    "alc_subclinical_dep_symptoms": "SSAGA_Alc_D4_Dp_Sx",
    "smoking_history_class": "SSAGA_TB_Smoking_History",
    "marijuana_ever_used": "SSAGA_Mj_Use",
    "marijuana_times_used": "SSAGA_Mj_Times_Used",
    "famhist_father_drugalc": "FamHist_Fath_DrgAlc",
}

# Reported descriptively only (too sparse / near-constant at N=46)
DESCRIPTIVE_ONLY_COLUMNS = [
    "SSAGA_Alc_D4_Ab_Dx", "SSAGA_Alc_D4_Dp_Dx", "SSAGA_Mj_Ab_Dep",  # exclusion criteria -- expected zero variance
    "Cocaine", "THC", "Opiates", "Amphetamines", "MethAmphetamine", "Oxycontin",
    "Breathalyzer_Over_05", "Breathalyzer_Over_08",
    "SSAGA_FTND_Score", "SSAGA_HSI_Score",
    "SSAGA_Times_Used_Illicits", "SSAGA_Times_Used_Cocaine",
    "SSAGA_Times_Used_Hallucinogens", "SSAGA_Times_Used_Opiates",
    "SSAGA_Times_Used_Sedatives", "SSAGA_Times_Used_Stimulants",
    "FamHist_Moth_DrgAlc",
]


def load_substance_vars(master_csv, cohort_ids):
    df = pd.read_csv(master_csv)
    df = df[df.Subject.astype(int).isin(cohort_ids)].copy()
    df["subject_str"] = df.Subject.astype(int).astype(str)

    series = {}
    for key, col in SUBSTANCE_COLUMNS.items():
        d = {}
        for _, row in df.iterrows():
            v = row[col]
            if pd.isna(v):
                continue
            d[row["subject_str"]] = float(v)
        series[key] = d
    return series, df


def run_one_pair(c_vals, p_vals, subjects):
    x, y, used = [], [], []
    for s in subjects:
        cv = c_vals.get(s)
        pv = p_vals.get(s)
        if cv is None or pv is None:
            continue
        x.append(cv)
        y.append(pv)
        used.append(s)
    n = len(used)
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
    ap.add_argument("--master-csv", required=True)
    ap.add_argument("--cohort-xlsx", required=True)
    ap.add_argument("--hoc-schaefer", required=True)
    ap.add_argument("--hoc-glasser", required=True)
    ap.add_argument("--vis-schaefer", required=True)
    ap.add_argument("--vis-glasser", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-descriptives", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim3_analysis as c3

    cohort = pd.read_excel(args.cohort_xlsx, sheet_name="Cohort")
    cohort_ids = cohort["Subject ID"].astype(int).tolist()
    print(f"{len(cohort_ids)} subjects in final analysis cohort")

    substance_series, df = load_substance_vars(args.master_csv, cohort_ids)
    subjects = df["subject_str"].tolist()

    hoc_schaefer = c3.load_hoc_summary(args.hoc_schaefer)
    hoc_glasser = c3.load_hoc_summary(args.hoc_glasser)
    vis_schaefer = c3.load_vis_summary(args.vis_schaefer, "schaefer200")
    vis_glasser = c3.load_vis_summary(args.vis_glasser, "glasser")

    controllability_series = {
        "avg_controllability_log__hoc__schaefer200": {s: hoc_schaefer[s][0] for s in hoc_schaefer if hoc_schaefer[s][0] is not None},
        "modal_controllability__hoc__schaefer200": {s: hoc_schaefer[s][1] for s in hoc_schaefer if hoc_schaefer[s][1] is not None},
        "avg_controllability_log__hoc__glasser": {s: hoc_glasser[s][0] for s in hoc_glasser if hoc_glasser[s][0] is not None},
        "modal_controllability__hoc__glasser": {s: hoc_glasser[s][1] for s in hoc_glasser if hoc_glasser[s][1] is not None},
        "avg_controllability_log__vis__schaefer200": {s: vis_schaefer[s][0] for s in vis_schaefer if vis_schaefer[s][0] is not None},
        "modal_controllability__vis__schaefer200": {s: vis_schaefer[s][1] for s in vis_schaefer if vis_schaefer[s][1] is not None},
        "avg_controllability_log__vis__glasser": {s: vis_glasser[s][0] for s in vis_glasser if vis_glasser[s][0] is not None},
        "modal_controllability__vis__glasser": {s: vis_glasser[s][1] for s in vis_glasser if vis_glasser[s][1] is not None},
    }

    results = []
    for c_key, c_vals in controllability_series.items():
        for s_key in SUBSTANCE_COLUMNS:
            s_vals = substance_series[s_key]
            stat = run_one_pair(c_vals, s_vals, subjects)
            row = {"controllability_summary": c_key, "substance_variable": s_key}
            row.update(stat)
            results.append(row)

    p_values = [r["p_pearson"] for r in results if "p_pearson" in r]
    q_values = bh_fdr(p_values) if p_values else []
    qi = 0
    for r in results:
        if "p_pearson" in r:
            r["q_value_fdr"] = float(q_values[qi])
            qi += 1
        else:
            r["q_value_fdr"] = None

    fieldnames = ["controllability_summary", "substance_variable", "n_subjects", "pearson_r", "p_pearson",
                  "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    sig = [r for r in results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== {len(results)} tests, own exploratory FDR family. Significant at q<0.05: {len(sig)} ===")
    for r in sorted(sig, key=lambda x: x["q_value_fdr"]):
        print(f"  {r['controllability_summary']} vs {r['substance_variable']}: r={r['pearson_r']:.3f}, "
              f"p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, n={r['n_subjects']}")
    print(f"Written to {args.out_csv}")

    # descriptives for ALL substance columns (tested + descriptive-only)
    all_cols = list(SUBSTANCE_COLUMNS.values()) + DESCRIPTIVE_ONLY_COLUMNS
    desc_rows = []
    for col in all_cols:
        s = df[col]
        nonnull = s.dropna()
        desc_rows.append({
            "column": col,
            "tested": col in SUBSTANCE_COLUMNS.values(),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(nonnull.nunique()),
            "mean": round(float(nonnull.mean()), 3) if len(nonnull) else None,
            "std": round(float(nonnull.std()), 3) if nonnull.nunique() > 1 else 0.0,
            "min": float(nonnull.min()) if len(nonnull) else None,
            "max": float(nonnull.max()) if len(nonnull) else None,
        })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(args.out_descriptives, index=False)
    print(f"Descriptives written to {args.out_descriptives}")


if __name__ == "__main__":
    main()
