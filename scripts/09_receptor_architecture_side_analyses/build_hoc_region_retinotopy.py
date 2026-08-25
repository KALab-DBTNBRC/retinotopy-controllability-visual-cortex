#!/usr/bin/env python3
"""
Links the paper's two main threads directly: aggregates per-subject,
per-BACKBONE-NODE retinotopy (aggregate_prf_to_backbone.py's real output,
already canonical as all_subjects_prf_backbone.csv -- eccentricity and
polar angle, the exact variables driving Claim 2's central finding) onto
the 8 hOc region x hemisphere units Claim 1's receptor analysis uses,
using the SAME real vertex-overlap weights (all_subjects_overlap_v3.csv)
and the SAME renormalize-across-real-coverage logic as
build_hoc_region_controllability.py -- deliberately structured as a
sibling script so results plug into the exact same per-subject spatial-
alignment design already used throughout Phase 7.

Purpose: previously, receptor density was tested against controllability
(Claim 1, null), behavior (Claim 3 + personality, null), and anatomical
region size (2026-08-13, real positive but likely just cytoarchitectonic
area-identity, non-functional) -- but never against retinotopic position
itself, the actual variable driving the paper's one robust finding. This
closes that gap: does receptor architecture track a subject's own
eccentricity/polar-angle organization at these regions, independent of
controllability entirely? Answering this either way ties the receptor
thread and the retinotopy thread into one throughline rather than two
side-by-side stories:
  - If YES: receptor architecture and controllability are BOTH indexed by
    retinotopic position -- a unifying claim (retinotopy is the deep
    organizing variable of visual cortex, expressed in multiple readouts).
  - If NO (consistent with every other receptor-vs-function/behavior test
    so far): strengthens the paper's central contrast -- retinotopic
    position organizes controllability specifically, while cytoarchitecture
    remains disconnected from it and from everything else tested, which is
    itself the point (a structural-vs-molecular dissociation, not just an
    absence of significant receptor findings).

Eccentricity aggregated as a real-coverage-weighted arithmetic mean (same
choice aggregate_prf_to_backbone.py itself uses at the vertex level, no
heavy-tail issue documented for eccentricity the way there was for
average_controllability). Polar angle aggregated as a real-coverage-
weighted CIRCULAR mean (0/360 wraparound, same risk flagged and fixture-
tested in aggregate_prf_to_backbone.py -- a naive weighted arithmetic mean
would be wrong near the wraparound), with the resultant vector length
(mean_resultant_r) kept as a per-unit reliability index, same spirit as
the per-vertex circular_concentration_r already in the source data.

Usage:
    python3 build_hoc_region_retinotopy.py \
        --overlap-csv "Received files/all_subjects_overlap_v3.csv" \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --atlas schaefer200 \
        --out-csv hoc_region_retinotopy/schaefer200.csv
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


def load_retinotopy(path, atlas):
    """Returns dict: (subject, backbone_label) -> (eccentricity_deg or None, polar_angle_deg or None)."""
    lookup = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["atlas"] != atlas:
                continue
            ecc = float(row["mean_eccentricity_deg"]) if row["mean_eccentricity_deg"] not in ("", None) else None
            ang = float(row["mean_polar_angle_deg"]) if row["mean_polar_angle_deg"] not in ("", None) else None
            lookup[(row["subject"], row["backbone_label"])] = (ecc, ang)
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-csv", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    overlap = load_overlap(args.overlap_csv, args.atlas, args.regions)
    retinotopy = load_retinotopy(args.prf_backbone_csv, args.atlas)

    rows_out = []
    n_ok, n_no_coverage, n_no_retinotopy = 0, 0, 0

    for (subject, hoc_name), entries in sorted(overlap.items()):
        region, hemi = hoc_name.rsplit("_", 1)
        n_hoc_vertices = entries[0][2]

        real_entries = [(label, w) for label, w, _ in entries if label != "UNASSIGNED"]

        # Restrict further to backbone nodes that actually have real retinotopy data
        # (aggregate_prf_to_backbone.py's own zero-coverage nodes -- expected, not a bug,
        # renormalize across whichever real entries DO have data rather than treat missing as zero).
        usable = []
        for label, w in real_entries:
            ecc, ang = retinotopy.get((subject, label), (None, None))
            if ecc is not None:
                usable.append((label, w, ecc, ang))

        total_weight = sum(w for _, w, _, _ in usable)
        if total_weight <= 0:
            n_no_retinotopy += 1
            continue

        mean_ecc = sum(w * ecc for _, w, ecc, _ in usable) / total_weight

        # weighted circular mean for polar angle (only over entries that also have angle data)
        angle_usable = [(w, ang) for _, w, _, ang in usable if ang is not None]
        angle_weight = sum(w for w, _ in angle_usable)
        if angle_weight > 0:
            sin_sum = sum(w * math.sin(math.radians(ang)) for w, ang in angle_usable) / angle_weight
            cos_sum = sum(w * math.cos(math.radians(ang)) for w, ang in angle_usable) / angle_weight
            mean_resultant_r = math.hypot(sin_sum, cos_sum)
            mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        else:
            mean_angle, mean_resultant_r = None, None

        rows_out.append({
            "subject": subject, "region": region, "hemisphere": hemi, "hoc_name": hoc_name,
            "atlas": args.atlas,
            "mean_eccentricity_deg": mean_ecc,
            "mean_polar_angle_deg": mean_angle,
            "mean_resultant_r": mean_resultant_r,
            "n_hoc_vertices": n_hoc_vertices,
            "coverage": total_weight,
            "n_backbone_nodes_used": len(usable),
        })
        n_ok += 1

    fieldnames = ["subject", "region", "hemisphere", "hoc_name", "atlas",
                  "mean_eccentricity_deg", "mean_polar_angle_deg", "mean_resultant_r",
                  "n_hoc_vertices", "coverage", "n_backbone_nodes_used"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done. {n_ok} ok, {n_no_retinotopy} with no usable retinotopy coverage.")
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
