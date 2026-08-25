#!/usr/bin/env python3
"""
hcp_bids_map.py
================

Stage 2 of 2: maps an already-extracted HCP tree (produced by hcp_extract.py)
into a BIDS-flavored layout. This is COPY-only -- it never touches the
extracted source tree, so it can be re-run as many times as needed (e.g. as
new subjects finish extracting, or as mapping rules get refined) without
re-extracting anything. Zip deletion is a separate, explicit, opt-in step
(--delete-verified-zips-dir) run only after you've inspected the output.

Built against real `tree` output for two representative cohort subjects (not
assumptions) -- see project decisions log for what that caught.

Layout produced
----------------
    <out>/rawdata/sub-<ID>/anat/        <- T1w / T2w (BIDS raw, native space)
    <out>/rawdata/sub-<ID>/dwi/         <- dwi + bval/bvec (BIDS raw)
    <out>/derivatives/hcp_pipelines/sub-<ID>/anat/
                                         <- native wmparc (for 5ttgen) +
                                            entire MNINonLinear/ tree (atlas-
                                            space surfaces, xfms/, QC, etc.)
    <out>/derivatives/hcp_pipelines/sub-<ID>/dwi/
                                         <- brain mask, grad_dev, bedpostX/
                                            (GPU fiber orientation output,
                                            SGE scheduler logs excluded),
                                            eddylogs/ (QC reports)
    <out>/derivatives/hcp_pipelines/sub-<ID>/func/
                                         <- retinotopy Results/ (combined
                                            multi-run folder AND the 6
                                            individual per-direction runs,
                                            each labeled by task) + RET-
                                            filtered fsaverage_LR32k/59k
                                            dense summary maps
    <out>/sourcedata/leftover/sub-<ID>/<package_raw>/
                                         <- anything not claimed above,
                                            copied (not moved) so the
                                            extracted source is untouched
    <out>/logs/bids_manifest.csv
    <out>/logs/bids_run_<timestamp>.jsonl

Usage
-----
    python3 hcp_bids_map.py --extracted-dir /data/extracted --output-dir /data/BIDS

    # Re-run later as more subjects finish extracting -- already-mapped
    # subject/package pairs are skipped by default:
    python3 hcp_bids_map.py --extracted-dir /data/extracted --output-dir /data/BIDS

    # Once you've inspected the output and are satisfied, delete the zips
    # (and .md5s) for packages that mapped successfully:
    python3 hcp_bids_map.py --extracted-dir /data/extracted --output-dir /data/BIDS \\
        --delete-verified-zips-dir /path/to/verified_zips

Only the Python standard library is used -- nothing to pip install.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Package discovery (matches hcp_extract.py's <out>/sub-<ID>/<package_raw>/ layout)
# --------------------------------------------------------------------------

PACKAGE_TYPE_RE = re.compile(r"(Structural|Diffusion|Retinotopy)", re.IGNORECASE)
FIELD_RE = re.compile(r"(3T|7T)", re.IGNORECASE)
LEVEL_RE = re.compile(r"(Unprocessed|Unproc|Recommended|Preproc)", re.IGNORECASE)
DEFAULT_FIELD_BY_PACKAGE = {"structural": "3T", "diffusion": "3T", "retinotopy": "7T"}


@dataclass
class ParsedPackage:
    package_type: Optional[str]
    field_strength: str
    is_unprocessed: bool
    package_key: str


def parse_package_name(rest: str) -> ParsedPackage:
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


@dataclass
class MapJob:
    subject: str
    package_raw: str
    package_key: str
    field_strength: str
    package_extracted_root: Path


@dataclass
class JobResult:
    subject: str
    package_raw: str
    field_strength: str
    mapped_files: int = 0
    leftover_files: int = 0
    handler_used: str = ""
    skipped_existing: bool = False
    status: str = "pending"      # pending | ok | error
    error: Optional[str] = None
    seconds: float = 0.0


def discover_jobs(extracted_dir: Path) -> list[MapJob]:
    jobs: list[MapJob] = []
    for sub_dir in sorted(extracted_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject = sub_dir.name[len("sub-"):]
        for pkg_dir in sorted(sub_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            parsed = parse_package_name(pkg_dir.name)
            if parsed.package_type is None:
                logging.warning("%s: package folder name '%s' doesn't contain a recognized "
                                 "keyword (Structural/Diffusion/Retinotopy) -- will route to "
                                 "sourcedata/unrecognized.", sub_dir.name, pkg_dir.name)
            jobs.append(MapJob(
                subject=subject, package_raw=pkg_dir.name, package_key=parsed.package_key,
                field_strength=parsed.field_strength, package_extracted_root=pkg_dir,
            ))
    return jobs


# --------------------------------------------------------------------------
# HCP package -> BIDS mapping rules (copy-only; source tree is never modified)
# --------------------------------------------------------------------------

def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _copy_and_count(src: Path, dst: Path) -> int:
    """Like _copy, but returns the actual number of files placed -- for a
    directory this recursively counts every file the copytree produced,
    instead of counting the directory as "1 item" (that undercount is what
    caused mapped_files to read 64 when 182 files were really on disk for
    StructuralRecommended -- caught via qc_bids_completeness.py's manifest
    cross-check, 2026-08-07: the data was always complete, only this count
    was wrong). Use this instead of bare _copy()+n+=1 whenever the item being
    copied might be a directory."""
    _copy(src, dst)
    if src.is_dir():
        return sum(1 for p in dst.rglob("*") if p.is_file())
    return 1


def _find_subject_root(extracted_root: Path, subject: str) -> Optional[Path]:
    """HCP zips normally extract to a single top-level '<subject>/' folder."""
    direct = extracted_root / subject
    if direct.exists():
        return direct
    candidates = [p for p in extracted_root.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return None


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

    # Native-space (T1w-space) white-matter parcellation -- same space as the
    # diffusion data, so this is what 5ttgen needs for anatomically-constrained
    # tractography, NOT the MNI-space wmparc.nii.gz under MNINonLinear/. Real
    # tree check (2026-08-07) showed this was falling to leftover.
    for f in sub_root.glob("T1w/wmparc.nii.gz"):
        _copy(f, deriv_anat / f"sub-{subject}_wmparc_native.nii.gz"); n += 1

    # Native-space (T1w-space) surfaces -- captured wholesale, same pattern as
    # the MNINonLinear/ capture below. Added 2026-08-08 after a real bug: every
    # surface this pipeline had been using (register_julich_native.py) came
    # from MNINonLinear/Native/ and MNINonLinear/fsaverage_LR32k/, which are in
    # MNI-nonlinear space despite "native" appearing in the filename -- that
    # word refers to mesh density (subject-specific vertex count), not volume
    # space. Confirmed via real per-vertex coordinate comparison (see todo.md
    # decisions log, 2026-08-08): mean 4.11mm / max 12.31mm displacement
    # between MNINonLinear/Native and T1w/Native for the same subject/surface,
    # the signature of a genuine MNI warp, not noise. Any volume built in true
    # native T1w/ACPC space (e.g. the Julich MPM from applywarp --ref=<native
    # T1w>) needs T1w/Native or T1w/fsaverage_LR32k surfaces, not the
    # MNINonLinear/ equivalents. Captured wholesale (not just white/pial/
    # midthickness/sphere) so a future need for inflated/flat/etc. doesn't
    # require a third round of "oh, we also needed X" -- same reasoning as
    # capturing MNINonLinear/ wholesale below.
    t1w_native = sub_root / "T1w" / "Native"
    if t1w_native.exists():
        dest = deriv_anat / "T1w" / "Native"
        for item in t1w_native.iterdir():
            n += _copy_and_count(item, dest / item.name)
    t1w_32k = sub_root / "T1w" / "fsaverage_LR32k"
    if t1w_32k.exists():
        dest = deriv_anat / "T1w" / "fsaverage_LR32k"
        for item in t1w_32k.iterdir():
            n += _copy_and_count(item, dest / item.name)

    # HCP's own MNI-registered structural pipeline outputs -> derivatives,
    # captured wholesale (includes xfms/acpc_dc2standard.nii.gz and
    # standard2acpc_dc.nii.gz -- the nonlinear warp fields the atlas-
    # registration script needs -- plus fsaverage_LR32k/59k, Native/, QC).
    mni = sub_root / "MNINonLinear"
    if mni.exists():
        dest = deriv_anat / "MNINonLinear"
        for item in mni.iterdir():
            n += _copy_and_count(item, dest / item.name)
    return n


def handle_structural_7t_preproc(root: Path, subject: str, out_dir: Path) -> int:
    n = 0
    sub_root = _find_subject_root(root, subject)
    if sub_root is None:
        return 0
    dest = out_dir / "derivatives" / "hcp_pipelines" / f"sub-{subject}" / "anat" / "7T_1.6mm"
    for item in sub_root.rglob("*"):
        if item.is_file():
            rel = item.relative_to(sub_root)
            target = dest / rel
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

    for extra in ("nodif_brain_mask.nii.gz", "grad_dev.nii.gz"):
        f = diff_dir / extra
        if f.exists():
            _copy(f, deriv_dwi / f"sub-{subject}_{extra}"); n += 1

    bedpostx_dir = sub_root / "T1w" / "Diffusion.bedpostX"
    if bedpostx_dir.exists():
        dest = deriv_dwi / "bedpostX"
        for item in bedpostx_dir.iterdir():
            if item.name == "logs":
                continue  # SGE job-scheduler logs, not analysis data
            n += _copy_and_count(item, dest / item.name)

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

    # Results/ contains BOTH the 6 individual per-direction run folders
    # (tfMRI_RETBAR1_7T_AP, etc. -- these start with "tfMRI_RET", confirmed
    # real 2026-08-07) AND the one combined/concatenated folder (starts with
    # "tfMRI_7T_RET...", field strength before the task tokens). Match both.
    results_dir = mni_dir / "Results"
    if results_dir.exists():
        for run_dir in results_dir.glob("tfMRI*RET*"):
            matched_keys = [key for key in RETINOTOPY_RUN_MAP if key in run_dir.name.upper()]
            if len(matched_keys) == 1:
                task_label = RETINOTOPY_RUN_MAP[matched_keys[0]]
            elif len(matched_keys) > 1:
                task_label = "retcombined"
            else:
                task_label = None
            dest_name = run_dir.name if task_label is None else f"task-{task_label}_{run_dir.name}"
            dest = deriv_func / dest_name
            for item in run_dir.iterdir():
                n += _copy_and_count(item, dest / item.name)

    # Subject-level dense-surface summary maps (dscalar/sdseries), one level
    # up from Results/. Filter on "RET" so unrelated content in that same
    # folder (there isn't any for this package, but be defensive) stays out.
    for folder_name in ("fsaverage_LR32k", "fsaverage_LR59k"):
        src_dir = mni_dir / folder_name
        if not src_dir.exists():
            continue
        dest_dir = deriv_func / folder_name
        for item in src_dir.iterdir():
            if "RET" not in item.name.upper():
                continue
            n += _copy_and_count(item, dest_dir / item.name)

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
                base = f"sub-{subject}_task-{task_label or 'retinotopy'}"
                dest = raw_func / f"{base}__{item.name}"
                _copy(item, dest); n += 1
    return n


def handle_unrecognized(root: Path, subject: str, out_dir: Path, package_raw: str) -> int:
    n = 0
    dest_root = out_dir / "sourcedata" / "unrecognized" / f"sub-{subject}" / package_raw
    for item in root.rglob("*"):
        if item.is_file():
            rel = item.relative_to(root)
            _copy(item, dest_root / rel); n += 1
    return n


PACKAGE_HANDLERS = {
    "structural_3t": handle_structural_preproc,
    "structural_7t": handle_structural_7t_preproc,
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


# Leftover detection uses an explicit per-package-type set of "already
# handled" globs to exclude, and copies everything else in that package's
# extracted tree that isn't excluded -- robust to handler internals without
# needing the handlers to emit their own claimed-file manifest.
LEFTOVER_EXCLUDE_GLOBS = {
    "structural_3t": ["*/T1w/T1w_acpc_dc_restore.nii.gz", "*/T1w/T2w_acpc_dc_restore.nii.gz",
                       "*/T1w/wmparc.nii.gz", "*/T1w/Native/**", "*/T1w/fsaverage_LR32k/**",
                       "*/MNINonLinear/**"],
    "diffusion_3t": ["*/T1w/Diffusion/data.nii.gz", "*/T1w/Diffusion/bvals", "*/T1w/Diffusion/bvecs",
                      "*/T1w/Diffusion/nodif_brain_mask.nii.gz", "*/T1w/Diffusion/grad_dev.nii.gz",
                      "*/T1w/Diffusion/eddylogs/**",
                      # bedpostX: everything the handler actually copies (all top-level items
                      # except "logs" -- see handle_diffusion_preproc). Deliberately does NOT
                      # exclude */Diffusion.bedpostX/logs/** so the SGE scheduler files still
                      # fall through to the leftover sweep below instead of being silently
                      # dropped (that was a real bug here, caught 2026-08-07: the old
                      # "*/T1w/Diffusion.bedpostX/**" pattern excluded the logs subtree too,
                      # so it never got mapped AND never got swept -- just vanished).
                      "*/T1w/Diffusion.bedpostX/bvals", "*/T1w/Diffusion.bedpostX/bvecs",
                      "*/T1w/Diffusion.bedpostX/commands.txt", "*/T1w/Diffusion.bedpostX/dyads*",
                      "*/T1w/Diffusion.bedpostX/mean_*", "*/T1w/Diffusion.bedpostX/merged_*",
                      "*/T1w/Diffusion.bedpostX/nodif_brain_mask.nii.gz",
                      "*/T1w/Diffusion.bedpostX/xfms/**"],
    "retinotopy_7t": ["*/MNINonLinear/Results/**", "*/MNINonLinear/fsaverage_LR32k/*RET*",
                       "*/MNINonLinear/fsaverage_LR59k/*RET*"],
}


def sweep_leftovers_v2(extracted_root: Path, package_key: str, subject: str, package_raw: str, out_dir: Path) -> int:
    n = 0
    excludes = LEFTOVER_EXCLUDE_GLOBS.get(package_key, [])
    excluded_paths: set[Path] = set()
    for pattern in excludes:
        excluded_paths.update(extracted_root.glob(pattern))
    # also exclude anything inside an excluded directory
    excluded_dirs = [p for p in excluded_paths if p.is_dir()]

    def is_excluded(p: Path) -> bool:
        if p in excluded_paths:
            return True
        return any(d in p.parents for d in excluded_dirs)

    dest_root = out_dir / "sourcedata" / "leftover" / f"sub-{subject}" / package_raw
    for item in extracted_root.rglob("*"):
        if item.is_file() and not is_excluded(item):
            # bedpostX/logs/logs_gpu specifically is per-node GPU-job chatter
            # (dozens of tiny part_NNNN-subpart_NNNN files with no analytic
            # value) -- deliberately discarded rather than swept. Everything
            # else under bedpostX/logs/ (the top-level .o/.e SGE files,
            # postproc_ID, monitor/) still gets preserved below.
            if "Diffusion.bedpostX" in item.parts and "logs_gpu" in item.parts:
                continue
            rel = item.relative_to(extracted_root)
            _copy(item, dest_root / rel); n += 1
    return n


# --------------------------------------------------------------------------
# Per-package worker
# --------------------------------------------------------------------------

def process_one_job(job: MapJob, out_dir: Path, force: bool) -> JobResult:
    t0 = time.time()
    res = JobResult(subject=job.subject, package_raw=job.package_raw, field_strength=job.field_strength)
    marker = out_dir / "logs" / ".mapped" / f"{job.subject}__{job.package_raw}"
    try:
        if marker.exists() and not force:
            res.status = "skipped"
            res.skipped_existing = True
            return res

        n_mapped, handler_name = map_package(job.package_key, job.package_raw,
                                              job.package_extracted_root, job.subject, out_dir)
        res.mapped_files = n_mapped
        res.handler_used = handler_name
        if n_mapped == 0:
            logging.warning("%s/%s: handler '%s' mapped 0 files -- inspect %s.",
                             job.subject, job.package_raw, handler_name, job.package_extracted_root)

        n_leftover = sweep_leftovers_v2(job.package_extracted_root, job.package_key,
                                         job.subject, job.package_raw, out_dir)
        res.leftover_files = n_leftover

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"mapped={n_mapped} leftover={n_leftover}\n")
        res.status = "ok"
    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
        logging.exception("Error mapping %s/%s", job.subject, job.package_raw)
    finally:
        res.seconds = round(time.time() - t0, 2)
    return res


# --------------------------------------------------------------------------
# Optional zip cleanup (explicit, separate step)
# --------------------------------------------------------------------------

def delete_verified_zips(results: list[JobResult], zips_dir: Path) -> int:
    deleted = 0
    for r in results:
        if r.status != "ok" or r.mapped_files == 0:
            continue
        zip_path = zips_dir / f"{r.subject}_{r.package_raw}.zip"
        if not zip_path.exists():
            continue
        try:
            zip_path.unlink()
            md5_sidecar = zip_path.with_name(zip_path.name + ".md5")
            if md5_sidecar.exists():
                md5_sidecar.unlink()
            deleted += 1
            logging.info("Deleted verified zip: %s", zip_path.name)
        except OSError as e:
            logging.warning("Could not delete %s: %s", zip_path.name, e)
    return deleted


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def write_dataset_description(out_dir: Path) -> None:
    dd_path = out_dir / "rawdata" / "dataset_description.json"
    if not dd_path.exists():
        dd_path.parent.mkdir(parents=True, exist_ok=True)
        dd_path.write_text(json.dumps({
            "Name": "HCP S1200 7T Retinotopy subset (BIDS reorganization)",
            "BIDSVersion": "1.9.0", "DatasetType": "raw",
            "GeneratedBy": [{"Name": "hcp_bids_map.py", "Version": "1.0"}],
        }, indent=2))
    deriv_dd = out_dir / "derivatives" / "hcp_pipelines" / "dataset_description.json"
    if not deriv_dd.exists():
        deriv_dd.parent.mkdir(parents=True, exist_ok=True)
        deriv_dd.write_text(json.dumps({
            "Name": "HCP minimal preprocessing pipeline outputs",
            "BIDSVersion": "1.9.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "HCP Pipelines"}],
            "SourceDatasets": [{"URL": "https://db.humanconnectome.org"}],
        }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracted-dir", required=True, type=Path, help="Output of hcp_extract.py")
    ap.add_argument("--output-dir", required=True, type=Path, help="BIDS project root to create/populate")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8),
                     help="Parallel worker processes. This step copies already-extracted files "
                          "(no unzip/checksum), so it's lighter than hcp_extract.py -- default is "
                          "lower to avoid saturating disk I/O when both extraction and mapping "
                          "might run around the same time.")
    ap.add_argument("--force", action="store_true", help="Re-map subject/package pairs even if already mapped.")
    ap.add_argument("--delete-verified-zips-dir", type=Path, default=None,
                     help="If given, after mapping, delete the .zip and .md5 for every package that "
                          "mapped successfully (>0 files, no errors). Run this only after inspecting "
                          "the output -- it is NOT the default behavior.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"bids_run_{run_id}.log")],
    )

    jobs = discover_jobs(args.extracted_dir)
    if not jobs:
        logging.error("No 'sub-*/<package>' folders found under %s -- did hcp_extract.py run yet?", args.extracted_dir)
        sys.exit(1)

    logging.info("Discovered %d package(s) across %d subject(s). workers=%d force=%s",
                 len(jobs), len({j.subject for j in jobs}), args.workers, args.force)

    write_dataset_description(args.output_dir)

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_job, job, args.output_dir, args.force): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = JobResult(subject=job.subject, package_raw=job.package_raw,
                                 field_strength=job.field_strength, status="error", error=str(e))
            results.append(res)
            done += 1
            logging.info("[%d/%d] sub-%s %s -> %s%s", done, len(jobs), res.subject, res.package_raw,
                         res.status, f" [{res.error}]" if res.error else "")

    manifest_path = log_dir / "bids_manifest.csv"
    jsonl_path = log_dir / f"bids_run_{run_id}.jsonl"
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
    errored = sum(1 for r in results if r.status == "error")
    zero_mapped = sum(1 for r in results if r.status == "ok" and r.mapped_files == 0)

    logging.info("=" * 70)
    logging.info("DONE. %d mapped, %d skipped (already done), %d errors, %d mapped 0 files.",
                  ok, skipped, errored, zero_mapped)
    if zero_mapped:
        logging.warning("Packages that mapped 0 files -- inspect before trusting this run.")

    if args.delete_verified_zips_dir:
        n_deleted = delete_verified_zips(results, args.delete_verified_zips_dir)
        logging.info("Deleted %d verified zip(s) from %s.", n_deleted, args.delete_verified_zips_dir)

    logging.info("Manifest: %s", manifest_path)
    logging.info("Full log: %s", jsonl_path)


if __name__ == "__main__":
    main()
