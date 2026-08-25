#!/usr/bin/env python3
"""
Addiction-relevant cohort (N=20) vs. main cohort (N=46): the first genuine
test of whether visual-cortex controllability differs by substance-use
group -- the specific comparison cohort-selection.xlsx's own exclusion
criteria were built to make impossible within the main cohort alone.

Three sub-analyses, each its own FDR family, all reusing already-verified
project machinery rather than reimplementing:

  A) GROUP COMPARISON. 8 controllability summaries (hoc/vis x
     schaefer200/glasser x avg_log/modal -- identical design to
     run_personality_analysis.py / run_substance_analysis.py), addiction
     N=20 vs. main N=46, age+sex covaried via multiple regression, group
     coefficient's significance assessed by permutation (shuffle group
     labels, refit, 10000 draws) since the two groups are unequal-N and
     age-range-mismatched (see caveats in Methodology sheet), not assumed
     Normal.

  B) DIMENSIONAL. Within the addiction-relevant N=20 only: does substance-
     use SEVERITY (not just membership) predict controllability? Same 4
     severity variables addiction-cohort-selection.xlsx already carries
     (alcohol dependence symptom count, weekly drinks, marijuana
     times-used, smoking history), same 8 controllability summaries,
     simple Pearson/Spearman (N=20, no permutation test needed -- not a
     small-spatial-unit design, same reasoning as run_substance_analysis.py).

  C) RETINOTOPY REPLICATION. Does the paper's central finding
     (eccentricity organizes controllability) hold within this subgroup?
     Already run directly via run_claim2_analysis.py itself, unmodified,
     against the addiction cohort's own pRF-backbone + controllability
     files -- not reimplemented here, see
     claim2_replication_{schaefer200,glasser}_addiction20.csv.

Usage:
    python3 run_addiction_group_analysis.py \
        --script-dir /path/to/project \
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
        --out-group-csv Addiction_Cohort_Results/addiction_group_comparison.csv \
        --out-dimensional-csv Addiction_Cohort_Results/addiction_dimensional.csv
"""
import argparse
import csv
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEVERITY_COLUMNS = {
    "alc_dependence_symptoms": "Alcohol Dependence Symptom Count",
    "alc_abuse_symptoms": "Alcohol Abuse Symptom Count",
    "total_drinks_7days": "Total Drinks, Past 7 Days",
    "marijuana_times_used": "Marijuana Times-Used Category",
    "smoking_history_class": "Lifetime Smoking History Class",
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


def load_demographics(xlsx_path):
    """Returns {subject_str: (age, sex_M_is_1)}."""
    df = pd.read_excel(xlsx_path, sheet_name="Cohort")
    out = {}
    for _, r in df.iterrows():
        s = str(int(r["Subject ID"]))
        age = float(r["Age (yrs)"])
        sex1 = 1.0 if r["Sex"] == "M" else 0.0
        out[s] = (age, sex1)
    return out


def load_severity(xlsx_path):
    """Returns {key: {subject_str: value}} for the 5 severity variables."""
    df = pd.read_excel(xlsx_path, sheet_name="Cohort")
    out = {k: {} for k in SEVERITY_COLUMNS}
    for _, r in df.iterrows():
        s = str(int(r["Subject ID"]))
        for key, col in SEVERITY_COLUMNS.items():
            v = r[col]
            if pd.notna(v):
                out[key][s] = float(v)
    return out


def group_permutation_test(y, group, age, sex, n_permutations, rng):
    """Regress y ~ group + age + sex (OLS via lstsq), test the group
    coefficient's significance by permuting the group label (age/sex kept
    fixed to each subject -- only membership is shuffled), which is valid
    here because group is an assigned category (not itself a function of
    y) and we're asking whether GROUP explains variance in y beyond age/sex,
    exactly analogous to the real question (does this graded regression
    design 'unequal N, mismatched age range' study have a real group
    effect), not assuming Normal residuals given N=66 total and
    unequal group sizes (20 vs 46)."""
    n = len(y)
    X = np.column_stack([np.ones(n), group, age, sex])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    observed_coef = beta[1]

    null_coefs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_group = rng.permutation(group)
        Xp = np.column_stack([np.ones(n), perm_group, age, sex])
        beta_p, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        null_coefs[i] = beta_p[1]

    perm_p = (np.sum(np.abs(null_coefs) >= abs(observed_coef)) + 1) / (n_permutations + 1)
    return observed_coef, perm_p


def run_one_pair_dimensional(c_vals, s_vals, subjects):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", default="/path/to/project")
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
    ap.add_argument("--out-group-csv", required=True)
    ap.add_argument("--out-dimensional-csv", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim3_analysis as c3

    rng = np.random.default_rng(args.seed)

    main_demo = load_demographics(args.main_cohort_xlsx)
    add_demo = load_demographics(args.addiction_cohort_xlsx)
    print(f"Main cohort: {len(main_demo)} subjects. Addiction cohort: {len(add_demo)} subjects.")
    overlap = set(main_demo) & set(add_demo)
    assert len(overlap) == 0, f"UNEXPECTED: {len(overlap)} subjects appear in both cohorts: {overlap}"

    # ---- load all 8 controllability series for BOTH cohorts ----
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

    main_series = load_all_series(args.main_hoc_schaefer, args.main_hoc_glasser,
                                   args.main_vis_schaefer, args.main_vis_glasser)
    add_series = load_all_series(args.add_hoc_schaefer, args.add_hoc_glasser,
                                  args.add_vis_schaefer, args.add_vis_glasser)

    # ================= A) GROUP COMPARISON =================
    group_results = []
    for key in main_series:
        main_vals = main_series[key]
        add_vals = add_series[key]
        subjects, y, group, age, sex = [], [], [], [], []
        for s, (a, sx) in main_demo.items():
            if s in main_vals:
                subjects.append(s); y.append(main_vals[s]); group.append(0.0); age.append(a); sex.append(sx)
        for s, (a, sx) in add_demo.items():
            if s in add_vals:
                subjects.append(s); y.append(add_vals[s]); group.append(1.0); age.append(a); sex.append(sx)
        y, group, age, sex = np.array(y), np.array(group), np.array(age), np.array(sex)
        n_main = int((group == 0).sum())
        n_add = int((group == 1).sum())
        if n_main < 5 or n_add < 5:
            group_results.append({"controllability_summary": key, "n_main": n_main, "n_addiction": n_add,
                                   "note": "too few subjects in one group, skipped"})
            continue
        coef, perm_p = group_permutation_test(y, group, age, sex, args.n_permutations, rng)
        mean_main = float(y[group == 0].mean())
        mean_addiction = float(y[group == 1].mean())
        group_results.append({
            "controllability_summary": key, "n_main": n_main, "n_addiction": n_add,
            "mean_main": mean_main, "mean_addiction": mean_addiction,
            "group_coef_age_sex_adj": float(coef), "perm_p_value": float(perm_p),
            "n_permutations": args.n_permutations, "note": "",
        })

    p_values = [r["perm_p_value"] for r in group_results if "perm_p_value" in r]
    q_values = bh_fdr(p_values) if p_values else []
    qi = 0
    for r in group_results:
        if "perm_p_value" in r:
            r["q_value_fdr"] = float(q_values[qi]); qi += 1
        else:
            r["q_value_fdr"] = None

    fieldnames_a = ["controllability_summary", "n_main", "n_addiction", "mean_main", "mean_addiction",
                     "group_coef_age_sex_adj", "perm_p_value", "n_permutations", "q_value_fdr", "note"]
    with open(args.out_group_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_a)
        w.writeheader()
        for r in group_results:
            w.writerow({k: r.get(k, "") for k in fieldnames_a})

    sig_a = [r for r in group_results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== A) Group comparison: {len(group_results)} tests, 1 FDR family. "
          f"Significant at q<0.05: {len(sig_a)} ===")
    for r in sorted(group_results, key=lambda x: x.get("perm_p_value", 1.0)):
        if "perm_p_value" in r:
            print(f"  {r['controllability_summary']}: main={r['mean_main']:.3f} (n={r['n_main']}), "
                  f"addiction={r['mean_addiction']:.3f} (n={r['n_addiction']}), "
                  f"coef={r['group_coef_age_sex_adj']:.4f}, p={r['perm_p_value']:.4g}, q={r['q_value_fdr']:.4g}")

    # ================= B) DIMENSIONAL (within addiction N=20) =================
    severity = load_severity(args.addiction_cohort_xlsx)
    add_subjects = sorted(add_demo.keys())

    dim_results = []
    for c_key, c_vals in add_series.items():
        for s_key in SEVERITY_COLUMNS:
            s_vals = severity[s_key]
            stat = run_one_pair_dimensional(c_vals, s_vals, add_subjects)
            row = {"controllability_summary": c_key, "severity_variable": s_key}
            row.update(stat)
            dim_results.append(row)

    p_values_b = [r["p_pearson"] for r in dim_results if "p_pearson" in r]
    q_values_b = bh_fdr(p_values_b) if p_values_b else []
    qi = 0
    for r in dim_results:
        if "p_pearson" in r:
            r["q_value_fdr"] = float(q_values_b[qi]); qi += 1
        else:
            r["q_value_fdr"] = None

    fieldnames_b = ["controllability_summary", "severity_variable", "n_subjects", "pearson_r", "p_pearson",
                     "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(args.out_dimensional_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_b)
        w.writeheader()
        for r in dim_results:
            w.writerow({k: r.get(k, "") for k in fieldnames_b})

    sig_b = [r for r in dim_results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== B) Dimensional (within addiction N=20): {len(dim_results)} tests, 1 FDR family. "
          f"Significant at q<0.05: {len(sig_b)} ===")
    for r in sorted(dim_results, key=lambda x: x.get("p_pearson", 1.0) if "p_pearson" in x else 1.0)[:8]:
        if "p_pearson" in r:
            print(f"  {r['controllability_summary']} vs {r['severity_variable']}: r={r['pearson_r']:.3f}, "
                  f"p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, n={r['n_subjects']}")

    print(f"\nWritten: {args.out_group_csv}, {args.out_dimensional_csv}")


if __name__ == "__main__":
    main()
