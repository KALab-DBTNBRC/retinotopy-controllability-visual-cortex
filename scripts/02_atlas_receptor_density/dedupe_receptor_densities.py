#!/usr/bin/env python3
"""
Average duplicate receptor-fingerprint rows in siibra_pull.py's real output
(receptor_densities.csv). hOc2 returned two full fingerprint entries in the
current siibra release (both hemispheres) -- already a settled project
decision (todo.md decisions log: "hOc2 duplicate receptor fingerprint:
averaged the two matches") -- but the raw CSV from the 2026-08-08 real pull
still has the duplicate rows, unaveraged. This script applies that decision
for real, and is written generally (group + average by every duplicate key)
rather than hardcoded to hOc2 specifically, so it's safe to re-run if any
other region/receptor combo ever comes back duplicated in a future siibra
release.

What "averaging" means here: for each (region, hemisphere, region_type,
receptor) group with more than one row, the output mean_density is the plain
mean of the input mean_density values (not sample-size-weighted -- siibra's
ReceptorDensityFingerprint doesn't expose the underlying N per study, so a
weighted average isn't available; this is disclosed in the output log and in
a dedicated column so it's never silently hidden). matched_region_name is
kept from the first row (should be identical across duplicates for the same
region/hemisphere). std_density is propagated via error propagation for an
average of independent measurements: sqrt(sum(std_i^2)) / n, not just
averaged directly (averaging two independent SDs directly would understate
the combined uncertainty).

Usage:
    python3 dedupe_receptor_densities.py \
        --input /path/to/receptor_densities.csv \
        --output /path/to/receptor_densities_deduped.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    groups = defaultdict(list)
    fieldnames = None
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            key = (row["region"], row["hemisphere"], row["region_type"], row["receptor"])
            groups[key].append(row)

    out_rows = []
    n_deduped_groups = 0
    for key, rows in groups.items():
        region, hemisphere, region_type, receptor = key
        if len(rows) == 1:
            out_rows.append(rows[0])
            continue

        n_deduped_groups += 1
        means = [float(r["mean_density"]) for r in rows]
        stds = [float(r["std_density"]) for r in rows]
        avg_mean = sum(means) / len(means)
        # error propagation for the SD of an average of independent measurements
        combined_std = math.sqrt(sum(s ** 2 for s in stds)) / len(stds)

        out_rows.append({
            "region": region,
            "hemisphere": hemisphere,
            "region_type": region_type,
            "matched_region_name": rows[0]["matched_region_name"],
            "receptor": receptor,
            "unit": rows[0]["unit"],
            "mean_density": avg_mean,
            "std_density": combined_std,
        })
        print(f"Averaged {len(rows)} entries for {region} ({hemisphere}) / {receptor}: "
              f"means={means} -> {avg_mean:.2f}, stds={stds} -> {combined_std:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. {len(groups)} unique (region, hemisphere, receptor) combos, "
          f"{n_deduped_groups} had duplicates and were averaged.")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
