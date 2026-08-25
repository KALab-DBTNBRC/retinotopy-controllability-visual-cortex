#!/usr/bin/env python3
"""
EXPLORATORY / POST-HOC (2026-08-13), own FDR family -- see run_personality_analysis.py's
docstring for the full context (real NEO-FFI columns found unused in
master-table.xlsx, lit-grounded but not one of the 5 pre-specified claims).

This is the more novel of the two personality sub-analyses: rather than
correlating personality against a subject's MEAN visual-system
controllability (sub-analysis A, run_personality_analysis.py -- the same
design as the already-null Claim 3), this asks whether personality relates
to the STRENGTH of Claim 2's own structural principle within each subject --
i.e. each subject's own eccentricity-controllability partial-r (how cleanly
that individual's visual system follows the retinotopy-controllability
gradient), reusing bootstrap_claim2_ci.py's per_subject_partial_r() (same
node set / min-valid-nodes / size covariate as Claim 2's own headline
numbers, imported not reimplemented, so results are guaranteed consistent
with what's already reported). Tests whether "how strongly organized" a
person's visual-system retinotopy-controllability coupling is corresponds to
real individual differences, rather than just its average level.

4 metric/atlas combinations (average/modal x Schaefer/Glasser) x 5
personality traits = 20 tests, own FDR family (separate from sub-analysis A
and from every pre-specified claim).

Usage:
    python3 run_personality_coupling_strength.py \
        --master-table-xlsx master-table.xlsx \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-schaefer all_subjects_controllability_schaefer200.csv \
        --controllability-glasser all_subjects_controllability_glasser.csv \
        --out-csv personality_coupling_strength.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats



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
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-schaefer", required=True)
    ap.add_argument("--controllability-glasser", required=True)
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim2_analysis as c2
    from run_personality_analysis import load_personality
    from bootstrap_claim2_ci import per_subject_partial_r

    personality = load_personality(args.master_table_xlsx)
    print(f"{len(personality)} subjects with complete personality data")

    results = []
    for atlas, ccsv in [("schaefer200", args.controllability_schaefer), ("glasser", args.controllability_glasser)]:
        subj_ecc, node_size = c2.load_prf_backbone(args.prf_backbone_csv, args.min_vertices_per_node, atlas)
        subj_avg_log, subj_modal = c2.load_controllability(ccsv)
        subjects = sorted(subj_ecc.keys())

        for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                     ("modal_controllability", subj_modal)]:
            used, per_r = per_subject_partial_r(c2, subj_ecc, subj_c, node_size, subjects, args.min_valid_nodes)
            coupling = dict(zip(used, per_r))
            print(f"{atlas} {metric_name}: {len(used)} subjects have a real coupling-strength value")

            for trait in ["neuroticism", "extraversion", "openness", "agreeableness", "conscientiousness"]:
                x, y, n_used = [], [], []
                for s in used:
                    if s in personality:
                        x.append(coupling[s])
                        y.append(personality[s][trait])
                        n_used.append(s)
                row = {"atlas": atlas, "metric": metric_name, "personality_trait": trait, "n_subjects": len(n_used)}
                if len(n_used) < 5 or np.std(x) == 0 or np.std(y) == 0:
                    row["note"] = "too few subjects or zero variance, skipped"
                else:
                    r_p, p_p = stats.pearsonr(x, y)
                    r_s, p_s = stats.spearmanr(x, y)
                    row.update({"pearson_r": float(r_p), "p_pearson": float(p_p),
                                "spearman_rho": float(r_s), "p_spearman": float(p_s), "note": ""})
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

    fieldnames = ["atlas", "metric", "personality_trait", "n_subjects", "pearson_r", "p_pearson",
                  "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    sig = [r for r in results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    print(f"\n=== {len(results)} tests, own exploratory FDR family. Significant at q<0.05: {len(sig)} ===")
    for r in sorted(sig, key=lambda x: x["q_value_fdr"]):
        print(f"  {r['atlas']} {r['metric']} vs {r['personality_trait']}: r={r['pearson_r']:.3f}, "
              f"p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, n={r['n_subjects']}")
    print(f"Written to {args.out_csv}")


if __name__ == "__main__":
    main()
