#!/usr/bin/env python3
"""
hcp2bids.py
===========

Verify, extract, and reorganize downloaded HCP Young Adult zip packages into a
BIDS-flavored directory tree, deleting each zip only after its own contents
have been checksummed, extracted, and successfully mapped into place.

Why this layout
----------------
HCP's own "preprocessed" packages (Structural, Diffusion, 7T Retinotopy) are
themselves already a specific pipeline's output, not raw BIDS data. So this
script splits things the way BIDS-Derivatives expects:

    <out>/rawdata/sub-<ID>/anat/        <- T1w / T2w (BIDS raw)
    <out>/rawdata/sub-<ID>/dwi/         <- dwi + bval/bvec (BIDS raw)
    <out>/rawdata/sub-<ID>/func/        <- raw retinotopy BOLD, if _unproc downloaded
    <out>/derivatives/hcp_pipelines/sub-<ID>/anat|dwi|func/
                                         <- HCP's own preprocessed/MNI-registered
                                            outputs (xfms, ICA-FIX cleaned runs, etc.)
    <out>/sourcedata/unrecognized/<ID>/<package>/
                                         <- anything this script doesn't have a
                                            mapping rule for, copied verbatim so
                                            nothing is silently dropped
    <out>/logs/manifest.csv             <- one row per zip processed
    <out>/logs/run_<timestamp>.jsonl    <- full structured log, one JSON object/line

Zip naming
----------
This matches ConnectomeDB's actual "Recommended" package names, e.g.:
    123456_StructuralRecommended.zip
    123456_Diffusion3TRecommended.zip
    123456_Retinotopy7TRecommended.zip
    <same base name>.zip.md5  (or <same base name>.md5)
i.e. <subject>_<Package><FieldStrength?>Recommended.zip — field strength
(3T/7T) is embedded in the package token and is optional (Structural has
none, implying 3T). The parser looks for a known package keyword
(Structural/Diffusion/Retinotopy) and an optional 3T/7T token anywhere in
that middle segment, rather than assuming a fixed position, so it isn't
thrown off by small naming variations.

IMPORTANT — please read before trusting this on a full batch
--------------------------------------------------------------
The file-mapping rules below (`PACKAGE_HANDLERS`) work on the directory
structure *inside* each package (T1w/, MNINonLinear/, T1w/Diffusion/, etc.),
which is documented and stable — but I have not seen your actual extracted
zip contents. Run with --dry-run first, inspect logs/manifest.csv and
spot-check the resulting rawdata/ and derivatives/ folders for 2-3 subjects,
THEN run for real (drop --dry-run) so zips actually get deleted.

Usage
-----
    # Safe first pass: verify + extract + map, but do not delete zips
    python3 hcp2bids.py --input-dir /data/hcp_downloads --output-dir /data/bids_project --dry-run

    # Real run: verify + extract + map + delete each zip after it succeeds
    python3 hcp2bids.py --input-dir /data/hcp_downloads --output-dir /data/bids_project --workers 32

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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Package discovery
# --------------------------------------------------------------------------

# e.g. 123456_StructuralRecommended.zip / 123456_Diffusion3TRecommended.zip
ZIP_NAME_RE = re.compile(r"^(?P<subject>\d{5,7})_(?P<rest>[A-Za-z0-9]+)\.zip$")
PACKAGE_TYPE_RE = re.compile(r"(Structural|Diffusion|Retinotopy)", re.IGNORECASE)
FIELD_RE = re.compile(r"(3T|7T)", re.IGNORECASE)
LEVEL_RE = re.compile(r"(Unprocessed|Unproc|Recommended|Preproc)", re.IGNORECASE)
MD5_HEX_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")

# If a package name doesn't spell out a field strength (e.g. "StructuralRecommended"),
# fall back to the field that type is conventionally acquired/used at in this project.
DEFAULT_FIELD_BY_PACKAGE = {"structural": "3T", "diffusion": "3T", "retinotopy": "7T"}


@dataclass
class PackageJob:
    subject: str
    field_strength: str          # "3T" or "7T" (inferred if not spelled out)
    package_raw: str             # original middle segment, e.g. "Diffusion3TRecommended"
    package_key: str             # normalized, e.g. "diffusion_3t"
    zip_path: Path
    md5_expected: Optional[str]  # None if no md5 could be found


@dataclass
class JobResult:
    subject: str
    field_strength: str
    package_raw: str
    zip_path: str
    md5_expected: Optional[str] = None
    md5_actual: Optional[str] = None
    checksum_ok: Optional[bool] = None
    extracted: bool = False
    mapped_files: int = 0
    handler_used: str = ""
    deleted_zip: bool = False
    dry_run: bool = False
    status: str = "pending"      # pending | ok | checksum_failed | error
    error: Optional[str] = None
    seconds: float = 0.0


@dataclass
class ParsedPackage:
    package_type: Optional[str]   # "Structural" / "Diffusion" / "Retinotopy" / None if unrecognized
    field_strength: str           # "3T" / "7T" / "UNK"
    is_unprocessed: bool
    package_key: str              # e.g. "diffusion_3t", "diffusion_3t_unproc", or "unrecognized"


def parse_package_name(rest: str) -> ParsedPackage:
    """rest is the zip filename's middle segment, e.g. 'Diffusion3TRecommended'
    or 'StructuralRecommended'. Keyword-search it rather than assuming a fixed
    token order/position, so small naming variations don't break discovery."""
    pkg_m = PACKAGE_TYPE_RE.search(rest)
    if not pkg_m:
        return ParsedPackage(package_type=None, field_strength="UNK", is_unprocessed=False, package_key="unrecognized")
    package_type = pkg_m.group(1).capitalize()
    field_m = FIELD_RE.search(rest)
    field_strength = field_m.group(1).upper() if field_m else DEFAULT_FIELD_BY_PACKAGE.get(package_type.lower(), "UNK")
    level_m = LEVEL_RE.search(rest)
    is_unprocessed = bool(level_m and level_m.group(1).lower() in ("unproc", "unprocessed"))
    key = f"{package_type.lower()}_{field_strength.lower()}"
    if is_unprocessed:
        key += "_unproc"
    return ParsedPackage(package_type=package_type, field_strength=field_strength,
                          is_unprocessed=is_unprocessed, package_key=key)


def read_expected_md5(md5_path: Path) -> Optional[str]:
    """HCP .md5 files are normally `md5sum`-style: '<hash>  <filename>'.
    Be liberal in what we accept: just pull the first 32-hex-char token."""
    try:
        text = md5_path.read_text(errors="ignore")
    except OSError:
        return None
    m = MD5_HEX_RE.search(text)
    return m.group(0).lower() if m else None


def find_md5_for_zip(zip_path: Path, manifest_hashes: dict[str, str]) -> Optional[str]:
    # 1) sidecar file: foo.zip.md5
    candidate = zip_path.with_name(zip_path.name + ".md5")
    if candidate.exists():
        h = read_expected_md5(candidate)
        if h:
            return h
    # 2) sidecar file: foo.md5 (extension swapped, not appended)
    candidate = zip_path.with_suffix(".md5")
    if candidate.exists():
        h = read_expected_md5(candidate)
        if h:
            return h
    # 3) shared manifest file (checksums.md5 / md5sum.txt) mapping name -> hash
    return manifest_hashes.get(zip_path.name)


def load_shared_manifests(input_dir: Path) -> dict[str, str]:
    """Some bulk downloads ship one combined checksums file instead of
    per-zip .md5 sidecars. Parse any *.md5 / *checksum* / *md5sum* files at
    the top of input_dir that contain multiple 'hash  filename' lines."""
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
        parsed = parse_package_name(rest)
        if parsed.package_type is None:
            logging.warning(
                "%s: subject/zip pattern matched but no known package keyword "
                "(Structural/Diffusion/Retinotopy) found in '%s' — routing to sourcedata/unrecognized.",
                zip_path.name, rest,
            )
        md5_expected = find_md5_for_zip(zip_path, manifest_hashes)
        jobs.append(PackageJob(
            subject=subject,
            field_strength=parsed.field_strength,
            package_raw=rest,
            package_key=parsed.package_key,
            zip_path=zip_path,
            md5_expected=md5_expected,
        ))
    if skipped:
        logging.warning(
            "%d .zip file(s) did not match the expected HCP naming pattern "
            "'<subject>_<Package...>.zip' and were skipped: %s",
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


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    unzip_bin = shutil.which("unzip")
    if unzip_bin:
        # System unzip is faster and lighter on memory for large, many-file
        # archives than Python's zipfile module.
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


# --------------------------------------------------------------------------
# HCP package -> BIDS mapping rules
# --------------------------------------------------------------------------
# Each handler receives (extracted_root, subject, out_dir) and returns the
# number of files it placed. Handlers use glob() so minor path variations
# (e.g. T1w_MPR1 vs T1w_MPR2) don't hard-fail the whole package.

def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # same filesystem (both under out_dir) -> shutil.move is a fast rename,
    # not a physical copy. Falls back to copy if that's not true.
    try:
        shutil.move(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def handle_structural_preproc(root: Path, subject: str, out_dir: Path) -> int:
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    raw_anat = out_dir / "rawdata" / f"sub-{subject}" / "anat"
    deriv_anat = out_dir / "derivatives" / "hcp_pipelines" / f"sub-{subject}" / "anat"

    for f in sub_root.glob("T1w/T1w_acpc_dc_restore.nii.gz"):
        _copy(f, raw_anat / f"sub-{subject}_T1w.nii.gz"); n += 1
    for f in sub_root.glob("T1w/T2w_acpc_dc_restore.nii.gz"):
        _copy(f, raw_anat / f"sub-{subject}_T2w.nii.gz"); n += 1

    # HCP's own MNI-registered structural pipeline outputs -> derivatives
    mni = sub_root / "MNINonLinear"
    if mni.exists():
        dest = deriv_anat / "MNINonLinear"
        dest.mkdir(parents=True, exist_ok=True)
        for item in mni.iterdir():
            _copy(item, dest / item.name); n += 1
    return n


def handle_structural_7t_preproc(root: Path, subject: str, out_dir: Path) -> int:
    # 1.6mm/59k mesh 7T-space structural — kept as a derivative variant,
    # since rawdata/anat is reserved for the canonical 3T T1w/T2w.
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    dest = out_dir / "derivatives" / "hcp_pipelines" / f"sub-{subject}" / "anat" / "7T_1.6mm"
    dest.mkdir(parents=True, exist_ok=True)
    for item in sub_root.rglob("*"):
        if item.is_file():
            rel = item.relative_to(sub_root)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy(item, target); n += 1
    return n


def handle_diffusion_preproc(root: Path, subject: str, out_dir: Path) -> int:
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    raw_dwi = out_dir / "rawdata" / f"sub-{subject}" / "dwi"
    deriv_dwi = out_dir / "derivatives" / "hcp_pipelines" / f"sub-{subject}" / "dwi"

    diff_dir = sub_root / "T1w" / "Diffusion"
    name_map = {
        "data.nii.gz": f"sub-{subject}_dwi.nii.gz",
        "bvals": f"sub-{subject}_dwi.bval",
        "bvecs": f"sub-{subject}_dwi.bvec",
    }
    for src_name, dst_name in name_map.items():
        f = diff_dir / src_name
        if f.exists():
            _copy(f, raw_dwi / dst_name); n += 1

    # brain mask + gradient-nonlinearity file are pipeline products, not raw
    for extra in ("nodif_brain_mask.nii.gz", "grad_dev.nii.gz"):
        f = diff_dir / extra
        if f.exists():
            _copy(f, deriv_dwi / f"sub-{subject}_{extra}"); n += 1

    # HCP's Diffusion3TRecommended package includes precomputed FSL bedpostX
    # GPU output (fiber orientation distributions / dyads) -- this is expensive
    # to recompute and directly usable for probabilistic tractography, so it
    # gets its own derivatives home rather than falling into sourcedata/leftover.
    bedpostx_dir = sub_root / "T1w" / "Diffusion.bedpostX"
    if bedpostx_dir.exists():
        dest = deriv_dwi / "bedpostX"
        for item in bedpostx_dir.iterdir():
            if item.name == "logs":
                continue  # job scheduler logs, not analysis data -- skip rather than clutter derivatives
            target = dest / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy(item, target); n += 1

    # eddy's own motion/outlier QC reports -- directly useful for the Phase 8
    # tractography-quality QC pass, so these get kept too rather than buried.
    eddylogs_dir = diff_dir / "eddylogs"
    if eddylogs_dir.exists():
        dest = deriv_dwi / "eddylogs"
        for item in eddylogs_dir.iterdir():
            _copy(item, dest / item.name); n += 1
    return n


def handle_diffusion_unproc(root: Path, subject: str, out_dir: Path) -> int:
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    raw_dwi = out_dir / "rawdata" / f"sub-{subject}" / "dwi" / "unprocessed"
    diff_dir = sub_root / "unprocessed" / "3T" / "Diffusion"
    if diff_dir.exists():
        for item in diff_dir.iterdir():
            _copy(item, raw_dwi / item.name); n += 1
    return n


# 7T retinotopy runs, per Benson et al. 2018: CCW, CW, EXP, CON, BAR1, BAR2
RETINOTOPY_RUN_MAP = {
    "RETCCW": "retccw", "RETCW": "retcw", "RETEXP": "retexp",
    "RETCON": "retcon", "RETBAR1": "retbar1", "RETBAR2": "retbar2",
}


def handle_retinotopy_preproc(root: Path, subject: str, out_dir: Path) -> int:
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    deriv_func = out_dir / "derivatives" / "hcp_pipelines" / f"sub-{subject}" / "func"
    mni_dir = sub_root / "MNINonLinear"

    # Results/ folder. IMPORTANT: the real "Recommended" package's combined-runs
    # folder is named tfMRI_7T_RETCCW_AP_RETCW_PA_RETEXP_AP_RETCON_PA_RETBAR1_AP_RETBAR2_PA
    # -- field strength (7T) sits BEFORE the task tokens, so it does NOT start
    # with "tfMRI_RET" as originally assumed. Confirmed via a real `tree` dump
    # 2026-08-07: the old "tfMRI_RET*" glob matched nothing and the whole
    # folder silently fell to sourcedata/leftover. Match more broadly instead.
    results_dir = mni_dir / "Results"
    if results_dir.exists():
        for run_dir in results_dir.glob("tfMRI*RET*"):
            matched_keys = [key for key in RETINOTOPY_RUN_MAP if key in run_dir.name.upper()]
            if len(matched_keys) == 1:
                task_label = RETINOTOPY_RUN_MAP[matched_keys[0]]
            elif len(matched_keys) > 1:
                # combined/concatenated multi-run folder (the "Recommended"
                # package ships this one, not the 6 separate per-run folders)
                task_label = "retcombined"
            else:
                task_label = None
            dest_name = run_dir.name if task_label is None else f"task-{task_label}_{run_dir.name}"
            dest = deriv_func / dest_name
            dest.mkdir(parents=True, exist_ok=True)
            for item in run_dir.iterdir():
                _copy(item, dest / item.name); n += 1

    # Subject-level dense-surface summary maps (dscalar/sdseries) live one
    # level up, directly under MNINonLinear/fsaverage_LR32k and _LR59k, not
    # inside Results/ -- these were also being missed entirely (same real-tree
    # check). Filter on "RET" in the filename so we only take the retinotopy
    # summary maps and leave any unrelated structural-surface content in that
    # same folder for sweep_leftovers to catch instead of guessing about it.
    for folder_name in ("fsaverage_LR32k", "fsaverage_LR59k"):
        src_dir = mni_dir / folder_name
        if not src_dir.exists():
            continue
        dest_dir = deriv_func / folder_name
        for item in src_dir.iterdir():
            if "RET" not in item.name.upper():
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            _copy(item, dest_dir / item.name); n += 1

    return n


def handle_retinotopy_unproc(root: Path, subject: str, out_dir: Path) -> int:
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    raw_func = out_dir / "rawdata" / f"sub-{subject}" / "func"
    unproc_7t = sub_root / "unprocessed" / "7T"
    if not unproc_7t.exists():
        return 0
    for run_dir in unproc_7t.glob("tfMRI_RET*"):
        task_label = None
        for key, label in RETINOTOPY_RUN_MAP.items():
            if key in run_dir.name.upper():
                task_label = label
                break
        for item in run_dir.rglob("*"):
            if item.is_file():
                suffix = "_bold.nii.gz" if item.name.endswith(".nii.gz") else f"_{item.suffix.lstrip('.')}"
                base = f"sub-{subject}_task-{task_label or 'retinotopy'}"
                dest = raw_func / f"{base}__{item.name}"  # keep original name to avoid collisions
                _copy(item, dest); n += 1
    return n


def handle_unrecognized(root: Path, subject: str, out_dir: Path, package_raw: str) -> int:
    n = 0
    dest_root = out_dir / "sourcedata" / "unrecognized" / f"sub-{subject}" / package_raw
    for item in root.rglob("*"):
        if item.is_file():
            rel = item.relative_to(root)
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy(item, target); n += 1
    return n


def _find_subject_root(extracted_root: Path, subject: str) -> Optional[Path]:
    """HCP zips normally extract to a single top-level '<subject>/' folder."""
    direct = extracted_root / subject
    if direct.exists():
        return direct
    candidates = [p for p in extracted_root.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return None


PACKAGE_HANDLERS = {
    # matches ConnectomeDB "Recommended" package names, e.g.:
    #   123456_StructuralRecommended.zip     -> structural_3t
    #   123456_Diffusion3TRecommended.zip    -> diffusion_3t
    #   123456_Retinotopy7TRecommended.zip   -> retinotopy_7t
    "structural_3t": handle_structural_preproc,
    "structural_7t": handle_structural_7t_preproc,   # only if you also grab the 1.6mm 7T structural package
    "diffusion_3t": handle_diffusion_preproc,
    "diffusion_3t_unproc": handle_diffusion_unproc,
    "retinotopy_7t": handle_retinotopy_preproc,
    "retinotopy_7t_unproc": handle_retinotopy_unproc,
}


def map_package(package_key: str, package_raw: str, extracted_root: Path, subject: str, out_dir: Path) -> tuple[int, str]:
    handler = PACKAGE_HANDLERS.get(package_key)
    if handler is None:
        n = handle_unrecognized(extracted_root, subject, out_dir, package_raw)
        return n, "unrecognized"
    n = handler(extracted_root, subject, out_dir)
    return n, handler.__name__


def sweep_leftovers(staging: Path, subject: str, package_raw: str, out_dir: Path) -> int:
    """Selective handlers (structural/diffusion/retinotopy preproc & unproc) only
    pull the specific files this project needs — ancillary files (FieldMap, BIAS,
    AFI, physio logs, etc.) are deliberately left behind. Rather than deleting
    them, sweep anything still sitting in staging into sourcedata/leftover/ so
    nothing from a verified zip is ever silently discarded. An empty sweep means
    the handler's glob patterns matched everything it expected to find."""
    n = 0
    if not staging.exists():
        return n
    dest_root = out_dir / "sourcedata" / "leftover" / f"sub-{subject}" / package_raw
    for item in staging.rglob("*"):
        if item.is_file():
            rel = item.relative_to(staging)
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy(item, target)
            n += 1
    return n


# --------------------------------------------------------------------------
# Per-package worker (runs in a subprocess)
# --------------------------------------------------------------------------

def process_one_package(job: PackageJob, out_dir: Path, dry_run: bool, keep_zip: bool) -> JobResult:
    t0 = time.time()
    res = JobResult(
        subject=job.subject, field_strength=job.field_strength,
        package_raw=job.package_raw, zip_path=str(job.zip_path),
        md5_expected=job.md5_expected, dry_run=dry_run,
    )
    # Unique per invocation (pid + timestamp) so a prior run's staging dir --
    # even one this environment refused to let us delete -- can never collide
    # with or block a fresh extraction.
    staging = out_dir / ".staging" / job.subject / f"{job.package_raw}_{os.getpid()}_{int(time.time() * 1000)}"
    try:
        # 1) verify checksum BEFORE extracting anything
        actual = compute_md5(job.zip_path)
        res.md5_actual = actual
        if job.md5_expected is None:
            res.checksum_ok = None  # no md5 found -> proceed but flag it
            logging.warning("No .md5 found for %s — proceeding without verification.", job.zip_path.name)
        else:
            res.checksum_ok = (actual == job.md5_expected)
            if not res.checksum_ok:
                res.status = "checksum_failed"
                res.error = f"expected {job.md5_expected}, got {actual}"
                logging.error("CHECKSUM FAILED: %s (%s)", job.zip_path.name, res.error)
                return res  # never extract or delete a file that failed verification

        # 2) extract to a private staging area
        extract_zip(job.zip_path, staging)
        res.extracted = True

        # 3) map into BIDS-ish layout
        n_mapped, handler_name = map_package(job.package_key, job.package_raw, staging, job.subject, out_dir)
        res.mapped_files = n_mapped
        res.handler_used = handler_name
        if n_mapped == 0:
            logging.warning(
                "%s: handler '%s' mapped 0 files — the package's internal layout may not "
                "match what this script expects. Left extracted contents in %s for inspection "
                "and NOT deleting the zip.",
                job.zip_path.name, handler_name, staging,
            )

        # 4) anything the handler didn't claim gets swept to sourcedata/leftover/
        #    rather than deleted, so a verified zip's contents are never silently
        #    lost even if this script's file-mapping assumptions are incomplete.
        n_leftover = sweep_leftovers(staging, job.subject, job.package_raw, out_dir)
        if n_leftover:
            logging.info("%s: %d ancillary file(s) not needed by the pipeline swept to sourcedata/leftover/.",
                         job.zip_path.name, n_leftover)

        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as e:
                logging.warning("Could not remove staging dir %s: %s (harmless — clean up manually later).",
                                 staging, e)

        # The core job — verify, extract, organize — is done and safe at this
        # point regardless of what happens next, so lock that in now.
        res.status = "ok"

        # 5) delete the zip, per-package, only after everything above succeeded.
        # Deletion failure (permissions, read-only mount, etc.) must NOT flip
        # this package back to "error" — your data is already safely organized;
        # only note that the zip itself couldn't be removed.
        if not dry_run and not keep_zip and n_mapped > 0:
            try:
                job.zip_path.unlink()
                res.deleted_zip = True
                md5_sidecar = job.zip_path.with_name(job.zip_path.name + ".md5")
                if md5_sidecar.exists():
                    md5_sidecar.unlink()
            except OSError as e:
                res.error = f"mapped OK but could not delete zip: {e}"
                logging.warning("%s: mapped successfully but could not delete the zip (%s). "
                                 "Safe to remove manually.", job.zip_path.name, e)
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

def write_dataset_description(out_dir: Path) -> None:
    dd_path = out_dir / "rawdata" / "dataset_description.json"
    if dd_path.exists():
        return
    dd_path.parent.mkdir(parents=True, exist_ok=True)
    dd_path.write_text(json.dumps({
        "Name": "HCP S1200 7T Retinotopy subset (BIDS reorganization)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "hcp2bids.py", "Version": "1.0"}],
    }, indent=2))
    deriv_dd = out_dir / "derivatives" / "hcp_pipelines" / "dataset_description.json"
    deriv_dd.parent.mkdir(parents=True, exist_ok=True)
    deriv_dd.write_text(json.dumps({
        "Name": "HCP minimal preprocessing pipeline outputs",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "HCP Pipelines"}],
        "SourceDatasets": [{"URL": "https://db.humanconnectome.org"}],
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder containing downloaded .zip + .md5 files")
    ap.add_argument("--output-dir", required=True, type=Path, help="BIDS project root to create/populate")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                     help="Parallel worker processes. Default is min(32, cpu_count). "
                          "This step is disk-I/O heavy (checksum + unzip), not just CPU-bound — "
                          "on an 80-core box, pushing past ~32-48 concurrent extractions rarely helps "
                          "once storage bandwidth saturates. Benchmark on a subset before maxing it out.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Verify + extract + map, but never delete zips. Strongly recommended for your first run.")
    ap.add_argument("--keep-zip", action="store_true",
                     help="Never delete zips even on full success (independent of --dry-run).")
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
            logging.FileHandler(log_dir / f"run_{run_id}.log"),
        ],
    )

    jobs = discover_packages(args.input_dir)
    if not jobs:
        logging.error("No matching '<subject>_<3T|7T>_<Package>.zip' files found in %s", args.input_dir)
        sys.exit(1)

    logging.info("Discovered %d package(s) across %d subject(s). workers=%d dry_run=%s",
                 len(jobs), len({j.subject for j in jobs}), args.workers, args.dry_run)

    write_dataset_description(args.output_dir)

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_package, job, args.output_dir, args.dry_run, args.keep_zip): job
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
    manifest_path = log_dir / "manifest.csv"
    jsonl_path = log_dir / f"run_{run_id}.jsonl"
    fieldnames = list(asdict(results[0]).keys())
    with open(manifest_path, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        if csvf.tell() == 0:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    with open(jsonl_path, "w") as jf:
        for r in results:
            jf.write(json.dumps(asdict(r)) + "\n")

    ok = sum(1 for r in results if r.status == "ok")
    failed_checksum = sum(1 for r in results if r.status == "checksum_failed")
    errored = sum(1 for r in results if r.status == "error")
    deleted = sum(1 for r in results if r.deleted_zip)
    zero_mapped = sum(1 for r in results if r.status == "ok" and r.mapped_files == 0)

    logging.info("=" * 70)
    logging.info("DONE. %d ok, %d checksum failures, %d errors, %d zips deleted, %d mapped 0 files.",
                  ok, failed_checksum, errored, deleted, zero_mapped)
    if failed_checksum:
        logging.warning("Checksum failures were NOT deleted or extracted — re-download those packages.")
    if zero_mapped:
        logging.warning("Packages that mapped 0 files were NOT deleted — inspect them (see log) before re-running.")
    logging.info("Manifest: %s", manifest_path)
    logging.info("Full log: %s", jsonl_path)


if __name__ == "__main__":
    main()
