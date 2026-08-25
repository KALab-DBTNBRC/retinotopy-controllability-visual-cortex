#!/usr/bin/env python3
"""
Read-only preview of what `hcp_bids_map.py --delete-verified-zips-dir` would
delete, WITHOUT deleting anything or re-running the mapper. Reads the existing
bids_manifest.csv (built by hcp_bids_map.py) and checks, for every row with
status == "ok" and mapped_files > 0, whether the matching zip/.md5 pair
actually exists in the zips directory -- using the exact same filename
pattern the real deletion function uses (zips_dir / f"{subject}_{package_raw}.zip"),
so this preview can't drift from what the real deletion would do.

Run this, review the list, and only then run:
    python3 hcp_bids_map.py --extracted-dir ... --output-dir ... --force \\
        --delete-verified-zips-dir <zips_dir>

(--force is required alongside --delete-verified-zips-dir -- see hcp_bids_map.py's
delete_verified_zips(), which only acts on packages marked "ok" in the CURRENT
run's results, not on historical manifest rows from a prior run.)

IMPORTANT: bids_manifest.csv is append-only -- hcp_bids_map.py never overwrites
it, only adds rows each time it runs. If the mapper has been run with --force
more than once (as this project has, once with a since-fixed undercounting bug
and once after the fix), the manifest contains multiple historical rows for the
same (subject, package_raw) pair. This script deduplicates by keeping only the
LAST row per pair (file order == chronological order for an append-only CSV),
so the preview reflects current/final state, not a sum across every past run.

Usage:
    python3 preview_zip_deletion.py \\
        --manifest /path/to/BIDS/logs/bids_manifest.csv \\
        --zips-dir /path/to/project
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--zips-dir", required=True, type=Path)
    args = ap.parse_args()

    # Manifest is append-only across every past run -- keep only the LAST row
    # per (subject, package_raw) pair (file order == chronological order for
    # an append-only CSV), so a package re-verified by a later --force run
    # isn't double-counted alongside its earlier, possibly-stale row.
    latest_rows: dict[tuple[str, str], dict] = {}
    n_raw_rows = 0
    with open(args.manifest, newline="") as f:
        for row in csv.DictReader(f):
            n_raw_rows += 1
            latest_rows[(row["subject"], row["package_raw"])] = row
    n_dupes_collapsed = n_raw_rows - len(latest_rows)
    if n_dupes_collapsed:
        print(f"Manifest has {n_raw_rows} total rows for {len(latest_rows)} unique "
              f"(subject, package) pairs -- collapsed {n_dupes_collapsed} historical "
              f"duplicate row(s) from prior runs, keeping only the latest per pair.\n")

    would_delete = []
    ok_no_zip_found = []
    not_verified = []

    for (subject, package_raw), row in latest_rows.items():
        status = row["status"]
        mapped_files = int(row["mapped_files"]) if row["mapped_files"] else 0

        if status != "ok" or mapped_files == 0:
            not_verified.append((subject, package_raw, status, mapped_files))
            continue

        zip_path = args.zips_dir / f"{subject}_{package_raw}.zip"
        md5_path = zip_path.with_name(zip_path.name + ".md5")
        if zip_path.exists():
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            would_delete.append((zip_path, md5_path.exists(), size_mb))
        else:
            ok_no_zip_found.append((subject, package_raw))

    total_mb = sum(sz for _, _, sz in would_delete)
    print(f"Manifest rows NOT verified ok/mapped (would be left untouched): {len(not_verified)}")
    for subject, package_raw, status, mapped_files in not_verified:
        print(f"  sub-{subject} {package_raw}: status={status} mapped_files={mapped_files}")

    print(f"\nVerified ok rows with no matching zip on disk (already deleted, or path mismatch): {len(ok_no_zip_found)}")
    for subject, package_raw in ok_no_zip_found:
        print(f"  sub-{subject} {package_raw}")

    print(f"\n=== Would delete {len(would_delete)} zip(s), {total_mb:,.1f} MB total ===")
    for zip_path, has_md5, size_mb in would_delete:
        print(f"  {zip_path.name}  ({size_mb:,.1f} MB){' + .md5' if has_md5 else ' (no .md5 found)'}")


if __name__ == "__main__":
    main()
