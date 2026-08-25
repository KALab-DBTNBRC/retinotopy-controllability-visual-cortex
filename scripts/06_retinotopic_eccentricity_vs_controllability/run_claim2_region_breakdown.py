#!/usr/bin/env python3
"""
Claim 2, Phase C1 -- region breakdown within Glasser's 5 official visual
divisions (Huang & Rolls et al. 2022's reproduction of Glasser et al. 2016's
own area-to-division scheme, see run_claim2_analysis.py's GLASSER_VIS_AREAS
docstring for the full sourcing story). Tests whether the whole-Vis-network
eccentricity-controllability relationship (Claim 2's headline finding) holds
WITHIN each division separately, or is driven by one division dominating the
pooled 52-node result.

Real per-division retention checked before choosing thresholds (2026-08-13,
same discipline as the original min-valid-nodes=15 decision for the full
Vis-network): Primary_Visual has only 2 nodes total (V1 L/R) -- a per-subject
correlation across 2 points is degenerate (always r=+/-1, no permutation
null is meaningful) and is EXCLUDED, not tested. Early_Visual (6 nodes):
45/45 subjects have all 6 -- min_valid_nodes=6 (full retention, no cost).
Dorsal_Stream_Visual (12 nodes), Ventral_Stream_Visual (14 nodes), MT+_Complex
(18 nodes): coverage is much sparser per division than pooled across all 52
(each subject's real pRF R2>30 coverage is patchy per-area, expected -- see
aggregate_prf_to_backbone.py's docstring) -- min_valid_nodes=3 chosen per
division (retains 31-35/45 subjects for each, the best available tradeoff
given each division's own real retention gradient; note this is a much
smaller node count per subject than Claim 2's main 15-node threshold, so
these per-division results should be read as suggestive, not as strong
individually as the pooled result).

Uses run_claim2_analysis.py's own loading/statistics functions unmodified
(imported, not reimplemented) so results are guaranteed consistent with the
main analysis; only the node_size dict is restricted to one division's node
set per run.

Usage:
    python3 run_claim2_region_breakdown.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --out-csv claim2_region_breakdown_glasser.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


GLASSER_DIVISIONS = {
    "Primary_Visual": {"V1"},
    "Early_Visual": {"V2", "V3", "V4"},
    "Dorsal_Stream_Visual": {"V3A", "V3B", "V6", "V6A", "V7", "IPS1"},
    "Ventral_Stream_Visual": {"FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC"},
    "MT+_Complex": {"FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t"},
}
# real retention-gradient-driven thresholds (2026-08-13), see module docstring
DIVISION_MIN_VALID_NODES = {
    "Early_Visual": 6,
    "Dorsal_Stream_Visual": 3,
    "Ventral_Stream_Visual": 3,
    "MT+_Complex": 3,
}  # Primary_Visual deliberately absent -- excluded, see docstring
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def division_of(node_label):
    m = GLASSER_LABEL_RE.match(node_label)
    if not m:
        return None
    area = m.group(1)
    for div, areas in GLASSER_DIVISIONS.items():
        if area in areas:
            return div
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim2_analysis as c2

    rng = np.random.default_rng(args.seed)

    # load full Glasser Vis-network data once (all 52 nodes), then restrict node_size per division
    subj_ecc, node_size_all = c2.load_prf_backbone(args.prf_backbone_csv, args.min_vertices_per_node, "glasser")
    subj_avg_log, subj_modal = c2.load_controllability(args.controllability_csv)
    subjects = sorted(subj_ecc.keys())

    results = []
    for division, min_valid in DIVISION_MIN_VALID_NODES.items():
        division_nodes = {n for n in node_size_all if division_of(n) == division}
        node_size_div = {n: node_size_all[n] for n in division_nodes}
        print(f"\n=== {division}: {len(division_nodes)} nodes, min_valid_nodes={min_valid} ===")
        for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                     ("modal_controllability", subj_modal)]:
            stat = c2.run_one_metric(subj_ecc, subj_c, node_size_div, subjects,
                                      min_valid, args.n_permutations, rng)
            row = {"division": division, "n_nodes_in_division": len(division_nodes), "metric": metric_name}
            row.update(stat)
            results.append(row)
            n_used = stat.get("n_subjects_valid", 0)
            print(f"  {metric_name}: n={n_used}, partial_r={stat.get('mean_partial_r', float('nan')):.3f}, "
                  f"perm_p={stat.get('perm_p_value', float('nan'))}")

    # FDR across this whole family (4 divisions x 2 metrics = 8 tests) -- own family,
    # separate from the main Claim 2 test and from every other claim's family
    perm_p_values = [r["perm_p_value"] for r in results if r.get("perm_p_value") is not None]
    q_values = c2.bh_fdr(perm_p_values) if perm_p_values else []
    qi = 0
    for r in results:
        if r.get("perm_p_value") is not None:
            r["q_value_fdr_perm"] = float(q_values[qi])
            qi += 1
        else:
            r["q_value_fdr_perm"] = None

    fieldnames = ["division", "n_nodes_in_division", "metric", "n_subjects_valid", "median_n_nodes_used",
                  "mean_raw_r", "mean_partial_r", "mean_fisher_z", "perm_p_value", "n_permutations_used",
                  "q_value_fdr_perm", "t_stat", "p_ttest", "wilcoxon_stat", "p_wilcoxon", "note"]
    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWritten to {out_csv}")


if __name__ == "__main__":
    main()
