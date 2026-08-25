#!/usr/bin/env python3
"""
Companion to compute_node_strength.py -- aggregates per-subject NODE-level
weighted structural connectivity strength (compute_node_strength.py's
output, not controllability) into per-subject hOc-REGION-level strength,
using the same real vertex-overlap weights (map_hoc_to_backbone.py's
all_subjects_overlap_v3.csv) that build_hoc_region_controllability.py
already uses -- deliberately near-identical structure to that script (same
overlap loading, same UNASSIGNED-coverage renormalization logic, same
log-space aggregation for consistency/robustness against any heavy-tailed
distribution -- not yet confirmed present in strength the way it was for
average_controllability, but included so the downstream receptor test can
choose whichever is better-behaved once real data is in hand) so results
plug directly into the same style of downstream analysis as Claim 1 and
receptor_behavior_direct.py, just swapping node_strength for controllability.

Usage:
    python3 build_hoc_region_strength.py \
        --overlap-csv /path/to/all_subjects_overlap_v3.csv \
        --node-strength-csv /path/to/all_subjects_node_strength_schaefer200.csv \
        --atlas schaefer200 \
        --out-csv /path/to/hoc_region_strength/schaefer200.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_REGIONS = ["hOc1", "hOc2", "hOc3v", "hOc3d"]


def load_overlap(path, atlas, regions):
    groups = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["atlas"] != atlas:
                continue
            hoc_name = row["hoc_name"]
            region = hoc_name.rsplit("_", 1)[0]
            if region not in regions:
                continue
            key = (row["subject"], hoc_name)
            groups[key].append((row["backbone_label"], float(row["weight"]), int(row["n_hoc_vertices"])))
    return groups


def load_node_strength(path):
    """Returns dict: (subject, node_label) -> weighted_degree."""
    lookup = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[(row["subject"], row["node_label"])] = float(row["weighted_degree"])
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-csv", required=True)
    ap.add_argument("--node-strength-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    overlap = load_overlap(args.overlap_csv, args.atlas, args.regions)
    strength = load_node_strength(args.node_strength_csv)

    rows_out = []
    n_ok, n_no_coverage, n_missing_lookup = 0, 0, 0

    for (subject, hoc_name), entries in sorted(overlap.items()):
        region, hemi = hoc_name.rsplit("_", 1)
        n_hoc_vertices = entries[0][2]

        real_entries = [(label, w) for label, w, _ in entries if label != "UNASSIGNED"]
        total_real_weight = sum(w for _, w in real_entries)

        if total_real_weight <= 0:
            n_no_coverage += 1
            continue

        weighted_strength = 0.0
        weighted_strength_log = 0.0
        n_nodes_used = 0
        missing_any = False
        nonpositive_any = False
        for label, w in real_entries:
            key = (subject, label)
            if key not in strength:
                print(f"sub-{subject} {hoc_name}: backbone node '{label}' has NO node-strength entry "
                      f"-- overlap and strength tables disagree, investigate rather than silently drop")
                missing_any = True
                continue
            s_val = strength[key]
            weighted_strength += w * s_val
            if s_val > 0:
                weighted_strength_log += w * math.log(s_val)
            else:
                nonpositive_any = True
            n_nodes_used += 1

        if missing_any:
            n_missing_lookup += 1
            continue

        weighted_strength /= total_real_weight
        weighted_strength_log = (weighted_strength_log / total_real_weight) if not nonpositive_any else None

        rows_out.append({
            "subject": subject, "region": region, "hemisphere": hemi, "hoc_name": hoc_name,
            "atlas": args.atlas,
            "weighted_strength": weighted_strength,
            "weighted_strength_log": weighted_strength_log,
            "n_hoc_vertices": n_hoc_vertices, "coverage": total_real_weight,
            "n_backbone_nodes_used": n_nodes_used,
        })
        n_ok += 1

    fieldnames = ["subject", "region", "hemisphere", "hoc_name", "atlas",
                  "weighted_strength", "weighted_strength_log",
                  "n_hoc_vertices", "coverage", "n_backbone_nodes_used"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done. {n_ok} ok, {n_no_coverage} zero real coverage, {n_missing_lookup} missing lookup.")
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
