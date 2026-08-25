#!/usr/bin/env python3
"""
EXPLORATORY / POST-HOC (2026-08-13) -- NOT one of the 5 pre-specified claims
from lit-review.docx. User asked whether other real, unused data could
support additional honest analysis before the paper draft (D/E). Real check
of master-table.xlsx found 5 NEO-FFI personality trait columns (Neuroticism,
Extraversion, Openness, Agreeableness, Conscientiousness), full variance,
zero missing, never used anywhere in this project. Lit-grounded (real
precedent for personality relating to structural/functional connectivity,
including visual-system-implicated networks specifically), but genuinely
exploratory -- own FDR family, separate from every pre-specified claim's own
family, clearly labeled as post-hoc in any writeup.

Sub-analysis A ("mean controllability"): same design as run_claim3_analysis.py
(simple cross-subject Pearson/Spearman, N~46, no permutation test needed --
not a small-spatial-unit design), just swapping the 3 behavioral columns for
5 personality columns. Reuses run_claim3_analysis.py's load_hoc_summary/
load_vis_summary unmodified (imported, not reimplemented) so results are
guaranteed consistent with Claim 3's own already-verified logic. Both hoc and
vis controllability summaries, both atlases (mirrors Claim 3/5's own
dual-atlas design) = 4 controllability summaries x 2 atlases x 5 traits = 40
tests, one FDR family.

Usage:
    python3 run_personality_analysis.py \
        --master-table-xlsx master-table.xlsx \
        --hoc-schaefer hoc_region_controllability/schaefer200.csv \
        --hoc-glasser hoc_region_controllability/glasser.csv \
        --vis-schaefer all_subjects_controllability_schaefer200.csv \
        --vis-glasser all_subjects_controllability_glasser.csv \
        --out-csv personality_mean_controllability.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import openpyxl
from scipy import stats

PERSONALITY_COLUMNS = {
    "neuroticism": "NEO-FFI Neuroticism",
    "extraversion": "NEO-FFI Extraversion",
    "openness": "NEO-FFI Openness",
    "agreeableness": "NEO-FFI Agreeableness",
    "conscientiousness": "NEO-FFI Conscientiousness",
}


def load_personality(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Subjects"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    missing = [c for c in PERSONALITY_COLUMNS.values() if c not in idx]
    if missing:
        raise ValueError(f"master-table.xlsx Subjects sheet missing expected column(s): {missing}")
    subj_col = idx["Subject ID"]
    out = {}
    for r in rows[1:]:
        if r[subj_col] is None:
            continue
        subject = str(r[subj_col])
        vals = {}
        ok = True
        for key, colname in PERSONALITY_COLUMNS.items():
            v = r[idx[colname]]
            if v is None:
                ok = False
                break
            vals[key] = float(v)
        if ok:
            out[subject] = vals
    return out


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
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--master-table-xlsx", required=True)
    ap.add_argument("--hoc-schaefer", required=True)
    ap.add_argument("--hoc-glasser", required=True)
    ap.add_argument("--vis-schaefer", required=True)
    ap.add_argument("--vis-glasser", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim3_analysis as c3

    personality = load_personality(args.master_table_xlsx)
    subjects = sorted(personality.keys())
    print(f"{len(subjects)} subjects with complete personality data")

    hoc_schaefer = c3.load_hoc_summary(args.hoc_schaefer)
    hoc_glasser = c3.load_hoc_summary(args.hoc_glasser)
    # run_claim3_analysis.load_vis_summary takes (path, atlas) after the 2026-08-13 Claim 5 extension
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
        for p_key in PERSONALITY_COLUMNS:
            p_vals = {s: personality[s][p_key] for s in personality}
            stat = run_one_pair(c_vals, p_vals, subjects)
            row = {"controllability_summary": c_key, "personality_trait": p_key}
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

    import csv
    fieldnames = ["controllability_summary", "personality_trait", "n_subjects", "pearson_r", "p_pearson",
                  "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    sig = [r for r in results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== {len(results)} tests, own exploratory FDR family. Significant at q<0.05: {len(sig)} ===")
    for r in sorted(sig, key=lambda x: x["q_value_fdr"]):
        print(f"  {r['controllability_summary']} vs {r['personality_trait']}: r={r['pearson_r']:.3f}, "
              f"p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, n={r['n_subjects']}")
    print(f"Written to {args.out_csv}")


if __name__ == "__main__":
    main()
