#!/usr/bin/env python3
"""
Phase 7, shared prerequisite for Claims 1 and 4 (and Claim 5's Glasser rerun):
converts per-subject Schaefer-200/Glasser NODE-level controllability into
per-subject hOc-REGION-level controllability, using the vertex-overlap
weights from map_hoc_to_backbone.py.

Why this is a separate step from the overlap mapping itself: map_hoc_to_backbone.py
records, for each (subject, hemisphere, hOc region, atlas), which backbone
node(s) overlap and by what vertex fraction -- it does NOT know about
controllability values at all (it only touches label.gii files). This script
is what actually combines that overlap table with compute_controllability.py's
real output to produce one controllability number per hOc region per subject,
which is the unit Claim 1/4's regression needs (to pair against receptor
density, which is defined per hOc region, not per Schaefer/Glasser node).

Handling UNASSIGNED coverage (medial-wall/parcellation-boundary vertices with
no real backbone node, real and expected per map_hoc_to_backbone.py's own
output -- up to ~24% for some subjects/regions): weights are renormalized
across only the REAL (non-UNASSIGNED) backbone nodes before averaging, so the
unassigned fraction doesn't silently deflate the result by being treated as
zero-controllability. The renormalized total weight actually used is written
out as `coverage` so a region with poor real coverage is visible, not hidden.

Average controllability is aggregated in LOG-SPACE as well as raw
(2026-08-12 fix): investigation of a real, systematic right>left asymmetry in
average_controllability (up to ~9x whole-brain, isolated via
diagnose_connectome_hemisphere_asymmetry.py to be visual-cortex-specific and
NOT present in the raw pre-controllability connectome at any level checked --
whole-brain or Vis-only) concluded this is a genuine, known property of
average controllability's heavy-tailed, eigenstructure-sensitive distribution
(field-standard practice is to log-transform it before linear statistics),
not a pipeline bug. Aggregating multiple overlapping backbone nodes' raw
average_controllability values with a plain weighted ARITHMETIC mean lets a
single extreme-valued node dominate a hOc region's whole aggregate (real
example: nodes [10, 15, 8000] with ~equal weight -> arithmetic mean ~2675,
almost entirely driven by the one outlier; log-space weighted mean,
equivalent to a weighted geometric mean when exponentiated, gives ~105 --
far more representative of the typical node). `weighted_avg_controllability`
(raw, unchanged) is kept for interpretability/other uses;
`weighted_avg_controllability_log` (weighted mean of log(avg_controllability)
across the same real, non-UNASSIGNED backbone nodes) is the column
run_claim1_analysis.py now uses for its average_controllability arm.

Usage:
    python3 build_hoc_region_controllability.py \
        --overlap-csv /path/to/project/BIDS/derivatives/hoc_backbone_overlap/all_subjects_overlap_v3.csv \
        --controllability-csv /path/to/project/BIDS/derivatives/controllability/all_subjects_controllability_schaefer200.csv \
        --atlas schaefer200 \
        --out-csv /path/to/project/BIDS/derivatives/hoc_region_controllability/schaefer200.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_REGIONS = ["hOc1", "hOc2", "hOc3v", "hOc3d"]  # receptor-covered regions (todo.md decisions log)


def load_overlap(path, atlas, regions):
    """Returns dict: (subject, hoc_name) -> list of (backbone_label, weight)."""
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


def load_controllability(path):
    """Returns dict: (subject, node_label) -> (avg_controllability, modal_controllability)."""
    lookup = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["node_label"])
            lookup[key] = (float(row["average_controllability"]), float(row["modal_controllability"]))
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    overlap = load_overlap(args.overlap_csv, args.atlas, args.regions)
    controllability = load_controllability(args.controllability_csv)

    rows_out = []
    n_ok, n_no_coverage, n_missing_lookup = 0, 0, 0

    for (subject, hoc_name), entries in sorted(overlap.items()):
        region, hemi = hoc_name.rsplit("_", 1)
        n_hoc_vertices = entries[0][2]

        real_entries = [(label, w) for label, w, _ in entries if label != "UNASSIGNED"]
        total_real_weight = sum(w for _, w in real_entries)

        if total_real_weight <= 0:
            n_no_coverage += 1
            print(f"sub-{subject} {hoc_name}: NO real backbone coverage at all -- skipped")
            continue

        weighted_avg = 0.0
        weighted_avg_log = 0.0
        weighted_modal = 0.0
        n_nodes_used = 0
        missing_any = False
        nonpositive_any = False
        for label, w in real_entries:
            key = (subject, label)
            if key not in controllability:
                print(f"sub-{subject} {hoc_name}: backbone node '{label}' (weight {w:.3f}) has NO "
                      f"controllability entry -- overlap and controllability tables disagree on the "
                      f"node set, this should never happen, investigate rather than silently drop")
                missing_any = True
                continue
            avg_c, modal_c = controllability[key]
            weighted_avg += w * avg_c
            weighted_modal += w * modal_c
            if avg_c > 0:
                weighted_avg_log += w * math.log(avg_c)
            else:
                # not expected -- Phase 6's aggregate QC confirmed all average_controllability
                # values positive across the real cohort -- surfaced, not silently skipped.
                print(f"sub-{subject} {hoc_name}: backbone node '{label}' has non-positive "
                      f"average_controllability ({avg_c}) -- excluded from the log-space aggregate, "
                      f"investigate if this appears for real data")
                nonpositive_any = True
            n_nodes_used += 1

        if missing_any:
            n_missing_lookup += 1
            continue

        weighted_avg /= total_real_weight
        weighted_modal /= total_real_weight
        weighted_avg_log = (weighted_avg_log / total_real_weight) if not nonpositive_any else None

        rows_out.append({
            "subject": subject,
            "region": region,
            "hemisphere": hemi,
            "hoc_name": hoc_name,
            "atlas": args.atlas,
            "weighted_avg_controllability": weighted_avg,
            "weighted_avg_controllability_log": weighted_avg_log,
            "weighted_modal_controllability": weighted_modal,
            "n_hoc_vertices": n_hoc_vertices,
            "coverage": total_real_weight,
            "n_backbone_nodes_used": n_nodes_used,
        })
        n_ok += 1

    fieldnames = ["subject", "region", "hemisphere", "hoc_name", "atlas",
                  "weighted_avg_controllability", "weighted_avg_controllability_log",
                  "weighted_modal_controllability",
                  "n_hoc_vertices", "coverage", "n_backbone_nodes_used"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nDone. {n_ok} ok, {n_no_coverage} with zero real coverage, "
          f"{n_missing_lookup} with a missing controllability lookup (investigate if > 0).")
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
