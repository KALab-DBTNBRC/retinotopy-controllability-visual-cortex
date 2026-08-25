#!/usr/bin/env python3
"""
Completes the "receptor vs. behavior without NCT" chain (2026-08-13) using
the real, stronger non-NCT bridge: per-subject weighted structural
connectivity STRENGTH at the 8 hOc units (compute_node_strength.py +
build_hoc_region_strength.py's real server output, uploaded 2026-08-13) --
plain weighted-degree graph theory, no nctpy anywhere in the chain that
produced this predictor.

Metric choice checked against real data before use, not assumed: raw
weighted_strength is close to symmetric in this dataset (skew 0.00-0.35
across both atlases), unlike average_controllability's documented heavy
tail -- log-transforming does not clearly help (Glasser log-skew actually
moves further from 0, to -0.73) and build_hoc_region_strength.py's own
weighted_strength_log column is available but not the primary metric
here. Raw weighted_strength is used throughout.

Same two-part structure as receptor_behavior_direct.py's region-size
version (test_receptor_vs_size imported unmodified, so results are
methodologically consistent with that analysis and with Claim 1):
  Part 1: receptor density (15 receptors + 3 composite axes from
    receptor_architecture_analysis.py) vs. per-subject connectivity
    strength across the 8 units, both atlases (strength magnitudes differ
    a lot by atlas -- Schaefer's 200-node graph vs. Glasser's 360-node
    graph -- so, unlike anatomical region size, atlas is NOT collapsed
    here). Own FDR families: 15x2=30 (receptors), 3x2=6 (composites).
  Part 2: each subject's own receptor-strength alignment score vs. the 3
    HCP behavioral measures (reuses run_claim3_analysis.py's
    BEHAVIORAL_COLUMNS/load_behavioral/run_one_pair). Own FDR families:
    15x3x2=90 (receptors), 3x3x2=18 (composites).

Usage:
    python3 receptor_vs_strength.py \
        --receptor-csv receptor-densities-deduped.csv \
        --hoc-strength-schaefer "schaefer200 (1)-501628c0.csv" \
        --hoc-strength-glasser glasser.csv \
        --master-table-xlsx master-table.xlsx \
        --pca-scores-csv receptor_architecture/receptor_pca_scores.csv \
        --ei-balance-csv receptor_architecture/receptor_ei_balance_index.csv \
        --out-dir receptor_vs_strength/
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# This script depends on run_claim1_analysis.py and run_claim3_analysis.py,
# which live in sibling scripts/ subdirectories in this repository layout
# (receptor_behavior_direct is in this same directory, so no extra path is
# needed for that import). Resolve those paths relative to this file's own
# location rather than hardcoding an absolute path, so the repo works
# regardless of where it is cloned.
_SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
for _sibling in ("05_receptor_architecture_vs_controllability", "07_behavior_personality"):
    _sibling_dir = _SCRIPTS_ROOT / _sibling
    if _sibling_dir.is_dir():
        sys.path.insert(0, str(_sibling_dir))

import run_claim1_analysis as c1
import run_claim3_analysis as c3
from receptor_behavior_direct import test_receptor_vs_size, bh_fdr_rows, bh_fdr_rows_pearson, write_csv

RECEPTOR_ORDER = ["AMPA", "kainate", "NMDA", "GABAA", "GABAB", "BZ", "M1", "M2", "M3",
                   "alpha1", "alpha2", "alpha4beta2", "5-HT1A", "5-HT2", "D1"]


def load_strength(path):
    """Returns subj_strength[subject][hoc_name] = weighted_strength (raw)."""
    out = {}
    for row in csv.DictReader(open(path)):
        out.setdefault(row["subject"], {})[row["hoc_name"]] = float(row["weighted_strength"])
    return out


def load_pca_scores(path):
    rows = list(csv.DictReader(open(path)))
    out = {"PC1_score": {}, "PC2_score": {}}
    for r in rows:
        for comp in list(out.keys()):
            if comp in r and r[comp] != "":
                out[comp][r["unit"]] = float(r[comp])
    return out


def load_ei_balance(path):
    return {r["unit"]: float(r["ei_log_ratio"]) for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor-csv", required=True)
    ap.add_argument("--hoc-strength-schaefer", required=True)
    ap.add_argument("--hoc-strength-glasser", required=True)
    ap.add_argument("--master-table-xlsx", required=True)
    ap.add_argument("--pca-scores-csv", required=True)
    ap.add_argument("--ei-balance-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    receptor_vecs = c1.load_receptor_vectors(args.receptor_csv)
    pca_scores = load_pca_scores(args.pca_scores_csv)
    ei_scores = load_ei_balance(args.ei_balance_csv)
    predictors = {r: {u: receptor_vecs[r][u] for u in c1.UNIT_ORDER} for r in RECEPTOR_ORDER}
    predictors.update({"PC1": pca_scores["PC1_score"], "PC2": pca_scores["PC2_score"], "EI_balance": ei_scores})

    behavioral = c3.load_behavioral(args.master_table_xlsx)
    atlases = {"schaefer200": args.hoc_strength_schaefer, "glasser": args.hoc_strength_glasser}

    part1_receptor, part1_composite = [], []
    part2_receptor, part2_composite = [], []

    for atlas, strength_csv in atlases.items():
        print(f"\n=== {atlas} ===")
        subj_strength = load_strength(strength_csv)
        subjects = sorted(subj_strength.keys())
        print(f"{len(subjects)} subjects with connectivity-strength data")

        for name, vec in predictors.items():
            receptor_x = np.array([vec[u] for u in c1.UNIT_ORDER])
            is_composite = name in ("PC1", "PC2", "EI_balance")

            stat, alignment = test_receptor_vs_size(receptor_x, subj_strength, subjects,
                                                      args.n_permutations, rng)
            row = {"atlas": atlas, "predictor": name}
            row.update(stat)
            (part1_composite if is_composite else part1_receptor).append(row)
            if "perm_p_value" in stat:
                print(f"  [strength] {name}: mean_r={stat['mean_raw_r']:.3f}, perm_p={stat['perm_p_value']:.4g}")

            for b_key in c3.BEHAVIORAL_COLUMNS:
                bstat = c3.run_one_pair(name, alignment, b_key, behavioral, subjects)
                row2 = {"atlas": atlas, "predictor": name, "behavioral_measure": b_key}
                row2.update(bstat)
                (part2_composite if is_composite else part2_receptor).append(row2)

    def fdr_write(rows, path, pkey, qkey):
        rows = bh_fdr_rows(rows, pkey=pkey, qkey=qkey) if pkey == "perm_p_value" else bh_fdr_rows_pearson(rows)
        fn = list(rows[0].keys()) if rows else []
        write_csv(path, rows, fn)
        return rows

    r1a = fdr_write(part1_receptor, out_dir / "receptor_vs_strength_results.csv", "perm_p_value", "q_value_fdr_perm")
    r1b = fdr_write(part1_composite, out_dir / "composite_vs_strength_results.csv", "perm_p_value", "q_value_fdr_perm")
    r2a = fdr_write(part2_receptor, out_dir / "receptor_strength_alignment_vs_behavior_results.csv", "p_pearson", "q_value_fdr")
    r2b = fdr_write(part2_composite, out_dir / "composite_strength_alignment_vs_behavior_results.csv", "p_pearson", "q_value_fdr")

    print("\n=== Summary (FDR q < 0.05) ===")
    for label, rows, qkey in [("Part 1a: receptor vs strength", r1a, "q_value_fdr_perm"),
                               ("Part 1b: composite vs strength", r1b, "q_value_fdr_perm"),
                               ("Part 2a: receptor-strength alignment vs behavior", r2a, "q_value_fdr"),
                               ("Part 2b: composite-strength alignment vs behavior", r2b, "q_value_fdr")]:
        sig = [r for r in rows if r.get(qkey) is not None and r[qkey] < 0.05]
        print(f"  {label}: {len(sig)}/{len(rows)} significant")
        for r in sorted(sig, key=lambda x: x[qkey])[:8]:
            metric_key = "mean_raw_r" if "mean_raw_r" in r else "pearson_r"
            pkey = "perm_p_value" if "perm_p_value" in r else "p_pearson"
            print(f"    {r}")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
