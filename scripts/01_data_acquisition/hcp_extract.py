#!/usr/bin/env python3
"""
hcp_extract.py
==============

Stage 1 of 2 for turning downloaded HCP zip packages into BIDS: verify each
zip against its .md5, extract it in full, as-is (native HCP internal layout,
nothing reorganized or dropped), into a persistent per-subject/per-package
folder. Zips and .md5 files are NEVER deleted by this script — that only
happens later, in the BIDS-mapping stage (hcp_to_bids.py, coming next), and
only after a package's contents have also been successfully mapped there.

Why split extraction from BIDS mapping
---------------------------------------
The first pass at this (hcp2bids.py) did checksum + extract + BIDS-map + zip
delete as one atomic per-zip step, using assumed internal folder names. One
of those assumptions was wrong (the real 7T retinotopy "Recommended" package
names its combined-runs folder tfMRI_7T_RETCCW_AP_..., not tfMRI_RET...) and
it silently fell through to a leftover bucket instead of erroring loudly.
Splitting the two stages means the BIDS-mapping rules get written and tested
against real, already-extracted directory trees — not assumptions — before
anything gets reorganized or any zip is deleted.

Layout produced by this script
-------------------------------
    <out>/sub-<ID>/<package_raw>/...        <- exact internal zip contents,
                                                untouched (e.g. sub-123456/
                                                Diffusion3TRecommended/123456/
                                                T1w/Diffusion/data.nii.gz)
    <out>/logs/extract_manifest.csv         <- one row per zip processed
    <out>/logs/extract_run_<timestamp>.jsonl

Zip naming
----------
Matches ConnectomeDB's actual "Recommended" package names, e.g.:
    123456_StructuralRecommended.zip
    123456_Diffusion3TRecommended.zip
    123456_Retinotopy7TRecommended.zip
    <same base name>.zip.md5  (or <same base name>.md5)

Usage
-----
    python3 hcp_extract.py --input-dir /data/hcp_downloads --output-dir /data/extracted

    # Re-run later (e.g. once more subjects finish downloading) -- packages
    # already extracted are skipped by default:
    python3 hcp_extract.py --input-dir /data/hcp_downloads --output-dir /data/extracted

    # Force re-extraction of everything, even packages already done:
    python3 hcp_extract.py --input-dir /data/hcp_downloads --output-dir /data/extracted --force

Only the Python standard library is used — nothing to pip install.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Package discovery (same rules as hcp2bids.py, kept in sync deliberately)
# --------------------------------------------------------------------------

ZIP_NAME_RE = re.compile(r"^(?P<subject>\d{5,7})_(?P<rest>[A-Za-z0-9]+)\.zip$")
PACKAGE_TYPE_RE = re.compile(r"(Structural|Diffusion|Retinotopy)", re.IGNORECASE)
FIELD_RE = re.compile(r"(3T|7T)", re.IGNORECASE)
DEFAULT_FIELD_BY_PACKAGE = {"structural": "3T", "diffusion": "3T", "retinotopy": "7T"}
MD5_HEX_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")


@dataclass
class PackageJob:
    subject: str
    field_strength: str
    package_raw: str
    zip_path: Path
    md5_expected: Optional[str]


@dataclass
class JobResult:
    subject: str
    field_strength: str
    package_raw: str
    zip_path: str
    extract_dir: str = ""
    md5_expected: Optional[str] = None
    md5_actual: Optional[str] = None
    checksum_ok: Optional[bool] = None
    extracted: bool = False
    n_files: int = 0
    skipped_existing: bool = False
    status: str = "pending"      # pending | ok | checksum_failed | error | skipped
    error: Optional[str] = None
    seconds: float = 0.0


def infer_field_strength(rest: str) -> str:
    m = FIELD_RE.search(rest)
    if m:
        return m.group(1).upper()
    pkg_m = PACKAGE_TYPE_RE.search(rest)
    if pkg_m:
        return DEFAULT_FIELD_BY_PACKAGE.get(pkg_m.group(1).lower(), "UNK")
    return "UNK"


def read_expected_md5(md5_path: Path) -> Optional[str]:
    try:
        text = md5_path.read_text(errors="ignore")
    except OSError:
        return None
    m = MD5_HEX_RE.search(text)
    return m.group(0).lower() if m else None


def find_md5_for_zip(zip_path: Path, manifest_hashes: dict[str, str]) -> Optional[str]:
    candidate = zip_path.with_name(zip_path.name + ".md5")
    if candidate.exists():
        h = read_expected_md5(candidate)
        if h:
            return h
    candidate = zip_path.with_suffix(".md5")
    if candidate.exists():
        h = read_expected_md5(candidate)
        if h:
            return h
    return manifest_hashes.get(zip_path.name)


def load_shared_manifests(input_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in ("*.md5", "*checksum*", "*md5sum*", "*.sha"):
        for f in input_dir.glob(pattern):
            try:
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 2 and MD5_HEX_RE.fullmatch(parts[0] or ""):
                    out[parts[-1].lstrip("*")] = parts[0].lower()
    return out


def discover_packages(input_dir: Path) -> list[PackageJob]:
    manifest_hashes = load_shared_manifests(input_dir)
    jobs: list[PackageJob] = []
    skipped = []
    for zip_path in sorted(input_dir.glob("*.zip")):
        m = ZIP_NAME_RE.match(zip_path.name)
        if not m:
            skipped.append(zip_path.name)
            continue
        subject = m.group("subject")
        rest = m.group("rest")
        md5_expected = find_md5_for_zip(zip_path, manifest_hashes)
        jobs.append(PackageJob(
            subject=subject,
            field_strength=infer_field_strength(rest),
            package_raw=rest,
            zip_path=zip_path,
            md5_expected=md5_expected,
        ))
    if skipped:
        logging.warning(
            "%d .zip file(s) did not match the expected '<subject>_<Package...>.zip' "
            "pattern and were skipped: %s",
            len(skipped), ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else "")
        )
    return jobs


# --------------------------------------------------------------------------
# Checksum + extraction
# --------------------------------------------------------------------------

def compute_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extracts in full, preserving internal structure. Returns file count."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    unzip_bin = shutil.which("unzip")
    if unzip_bin:
        result = subprocess.run(
            [unzip_bin, "-o", "-q", str(zip_path), "-d", str(dest_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unzip failed (code {result.returncode}): {result.stderr.strip()}")
    else:
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    return sum(1 for p in dest_dir.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# Per-package worker (runs in a subprocess)
# --------------------------------------------------------------------------

def process_one_package(job: PackageJob, out_dir: Path, force: bool) -> JobResult:
    t0 = time.time()
    extract_dir = out_dir / f"sub-{job.subject}" / job.package_raw
    res = JobResult(
        subject=job.subject, field_strength=job.field_strength,
        package_raw=job.package_raw, zip_path=str(job.zip_path),
        extract_dir=str(extract_dir), md5_expected=job.md5_expected,
    )
    try:
        if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
            res.status = "skipped"
            res.skipped_existing = True
            logging.info("%s: already extracted at %s — skipping (use --force to redo).",
                         job.zip_path.name, extract_dir)
            return res

        # 1) verify checksum BEFORE extracting anything
        actual = compute_md5(job.zip_path)
        res.md5_actual = actual
        if job.md5_expected is None:
            res.checksum_ok = None
            logging.warning("No .md5 found for %s — proceeding without verification.", job.zip_path.name)
        else:
            res.checksum_ok = (actual == job.md5_expected)
            if not res.checksum_ok:
                res.status = "checksum_failed"
                res.error = f"expected {job.md5_expected}, got {actual}"
                logging.error("CHECKSUM FAILED: %s (%s)", job.zip_path.name, res.error)
                return res  # never extract a file that failed verification

        # 2) extract in full, as-is -- no reorganization, nothing dropped
        if force and extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        n_files = extract_zip(job.zip_path, extract_dir)
        res.extracted = True
        res.n_files = n_files
        res.status = "ok"

        # zip and .md5 are intentionally left in place -- deletion happens
        # later, in the BIDS-mapping stage, only after mapping also succeeds.
    except Exception as e:  # noqa: BLE001 - a worker must never crash the pool
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
        logging.exception("Error processing %s", job.zip_path)
    finally:
        res.seconds = round(time.time() - t0, 2)
    return res


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder containing downloaded .zip + .md5 files")
    ap.add_argument("--output-dir", required=True, type=Path, help="Root folder to extract into (created if missing)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                     help="Parallel worker processes. Default is min(32, cpu_count). This step is "
                          "disk-I/O heavy (checksum + unzip), not CPU-bound -- on an 80-core box, "
                          "pushing past ~32-48 concurrent extractions rarely helps once storage "
                          "bandwidth saturates. Benchmark on a subset before maxing it out.")
    ap.add_argument("--force", action="store_true",
                     help="Re-extract packages even if their destination folder already has content.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"extract_run_{run_id}.log"),
        ],
    )

    jobs = discover_packages(args.input_dir)
    if not jobs:
        logging.error("No matching '<subject>_<Package...>.zip' files found in %s", args.input_dir)
        sys.exit(1)

    logging.info("Discovered %d package(s) across %d subject(s). workers=%d force=%s",
                 len(jobs), len({j.subject for j in jobs}), args.workers, args.force)

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_package, job, args.output_dir, args.force): job
            for job in jobs
        }
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = JobResult(subject=job.subject, field_strength=job.field_strength,
                                 package_raw=job.package_raw, zip_path=str(job.zip_path),
                                 status="error", error=str(e))
            results.append(res)
            done += 1
            logging.info("[%d/%d] sub-%s %s (%s) -> %s%s",
                          done, len(jobs), res.subject, res.package_raw, res.field_strength,
                          res.status, f" [{res.error}]" if res.error else "")

    # ---- write manifest + jsonl log ----
    manifest_path = log_dir / "extract_manifest.csv"
    jsonl_path = log_dir / f"extract_run_{run_id}.jsonl"
    fieldnames = list(asdict(results[0]).keys())
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    with open(jsonl_path, "w") as jf:
        for r in results:
            jf.write(json.dumps(asdict(r)) + "\n")

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed_checksum = sum(1 for r in results if r.status == "checksum_failed")
    errored = sum(1 for r in results if r.status == "error")

    logging.info("=" * 70)
    logging.info("DONE. %d extracted, %d skipped (already done), %d checksum failures, %d errors.",
                  ok, skipped, failed_checksum, errored)
    if failed_checksum:
        logging.warning("Checksum failures were NOT extracted — re-download those packages. "
                         "Their zips were left untouched.")
    logging.info("Manifest: %s", manifest_path)
    logging.info("Full log: %s", jsonl_path)
    logging.info("Next step: send me a few `tree -L 4 %s/sub-<ID>` examples so the BIDS-mapping "
                  "script can be built against the real extracted structure.", args.output_dir)


if __name__ == "__main__":
    main()
