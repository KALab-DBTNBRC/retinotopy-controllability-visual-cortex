#!/usr/bin/env python3
"""
QC completeness pass on the BIDS-mapped output, before any further Phase 3/5/6
work resumes (2026-08-07 decision: process/clean the data first).

Checks per subject, using the real directory structure verified against the
actual server output for a representative cohort subject (not assumptions):

  1. Required top-level files/dirs are present and non-empty --
     rawdata/anat (T1w, T2w), rawdata/dwi (dwi.nii.gz, bval, bvec),
     derivatives/anat (wmparc_native, MNINonLinear/, MNINonLinear/Native/,
     MNINonLinear/xfms/{acpc_dc2standard,standard2acpc_dc}.nii.gz --
     the last two are hard requirements for register_julich_native.py),
     derivatives/dwi (nodif_brain_mask, grad_dev, bedpostX/, eddylogs/),
     derivatives/func (all 7 task-ret* run folders, fsaverage_LR32k/59k).
  2. On-disk file counts per package (structural/diffusion/retinotopy) match
     bids_manifest.csv's mapped_files column -- catches silent partial copies
     or later accidental deletions that wouldn't show up as a "missing file"
     check alone.
  3. Scans derivatives/func filenames for pRF-solution-like keywords (eccen,
     polarangle, prfsize/sigma, R2/varexp) to empirically answer the open
     question of whether population receptive field parameter maps are
     actually present in what we extracted, or whether they need a separate
     pull from Benson et al.'s OSF/BALSA release -- don't guess, check.

Does not touch or modify any files. Read-only QC pass.

Usage:
    python3 qc_bids_completeness.py \
        --bids-root /path/to/bids_output \
        --manifest /path/to/logs/bids_manifest.csv \
        --out-dir /path/to/qc_reports
        [--subjects 123456 234567 ...]

--bids-root must contain rawdata/, derivatives/hcp_pipelines/, sourcedata/
(the layout hcp_bids_map.py produces).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

PRF_KEYWORDS = ["eccen", "polarangle", "polar_angle", "prfsize", "sigma", "varexp",
                "angle", "meanbold", "r2", "gain"]
# NOTE: "angle"/"r2"/"gain" are broad and may false-positive on unrelated files;
# reported as a secondary signal, not the primary yes/no call -- see report.
PRF_KEYWORDS_STRONG = ["eccen", "polarangle", "polar_angle", "prfsize", "varexp"]

REQUIRED = [
    # (label, relative path from sub-<ID> root, "file" or "dir", package)
    ("T1w (rawdata)", "rawdata/anat/{sub}_T1w.nii.gz", "file", "StructuralRecommended"),
    ("T2w (rawdata)", "rawdata/anat/{sub}_T2w.nii.gz", "file", "StructuralRecommended"),
    ("wmparc native (derivatives)", "derivatives/anat/{sub}_wmparc_native.nii.gz", "file", "StructuralRecommended"),
    ("MNINonLinear/", "derivatives/anat/MNINonLinear", "dir", "StructuralRecommended"),
    ("MNINonLinear/Native/", "derivatives/anat/MNINonLinear/Native", "dir", "StructuralRecommended"),
    ("xfms/acpc_dc2standard.nii.gz", "derivatives/anat/MNINonLinear/xfms/acpc_dc2standard.nii.gz", "file", "StructuralRecommended"),
    ("xfms/standard2acpc_dc.nii.gz", "derivatives/anat/MNINonLinear/xfms/standard2acpc_dc.nii.gz", "file", "StructuralRecommended"),

    ("dwi.nii.gz (rawdata)", "rawdata/dwi/{sub}_dwi.nii.gz", "file", "Diffusion3TRecommended"),
    ("dwi.bval (rawdata)", "rawdata/dwi/{sub}_dwi.bval", "file", "Diffusion3TRecommended"),
    ("dwi.bvec (rawdata)", "rawdata/dwi/{sub}_dwi.bvec", "file", "Diffusion3TRecommended"),
    ("nodif_brain_mask (derivatives)", "derivatives/dwi/{sub}_nodif_brain_mask.nii.gz", "file", "Diffusion3TRecommended"),
    ("grad_dev (derivatives)", "derivatives/dwi/{sub}_grad_dev.nii.gz", "file", "Diffusion3TRecommended"),
    ("bedpostX/", "derivatives/dwi/bedpostX", "dir", "Diffusion3TRecommended"),
    ("eddylogs/", "derivatives/dwi/eddylogs", "dir", "Diffusion3TRecommended"),

    ("fsaverage_LR32k/ (func)", "derivatives/func/fsaverage_LR32k", "dir", "Retinotopy7TRecommended"),
    ("fsaverage_LR59k/ (func)", "derivatives/func/fsaverage_LR59k", "dir", "Retinotopy7TRecommended"),
]

RETINOTOPY_TASKS = ["retbar1", "retbar2", "retccw", "retcw", "retcon", "retexp", "retcombined"]

PACKAGE_SUBDIRS = {
    # which on-disk subtrees count toward each HCP package, for the
    # manifest file-count cross-check (step 2)
    "StructuralRecommended": ["rawdata/anat", "derivatives/anat"],
    "Diffusion3TRecommended": ["rawdata/dwi", "derivatives/dwi"],
    "Retinotopy7TRecommended": ["derivatives/func"],
}


def count_files(root: Path, rel_dirs):
    total = 0
    for rel in rel_dirs:
        d = root / rel
        if d.exists():
            total += sum(1 for p in d.rglob("*") if p.is_file())
    return total


def load_manifest(manifest_path: Path):
    """subject -> {package_raw: mapped_files}"""
    expected = defaultdict(dict)
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            expected[row["subject"]][row["package_raw"]] = int(row["mapped_files"])
    return expected


def check_required_files(sub_root: Path, sub: str):
    missing = []
    for label, rel_template, kind, package in REQUIRED:
        rel = rel_template.format(sub=sub)
        path = sub_root / rel
        if kind == "file":
            ok = path.is_file() and path.stat().st_size > 0
        else:
            ok = path.is_dir() and any(path.iterdir())
        if not ok:
            missing.append((label, package, str(path)))
    return missing


def check_retinotopy_tasks(sub_root: Path):
    func_dir = sub_root / "derivatives/func"
    missing_tasks = []
    if not func_dir.is_dir():
        return RETINOTOPY_TASKS[:]  # all missing
    found_dirs = [p.name for p in func_dir.iterdir() if p.is_dir()]
    for task in RETINOTOPY_TASKS:
        prefix = f"task-{task}_"
        matches = [d for d in found_dirs if d.startswith(prefix)]
        if not matches:
            missing_tasks.append(task)
            continue
        task_dir = func_dir / matches[0]
        if not any(task_dir.iterdir()):
            missing_tasks.append(f"{task} (empty)")
    return missing_tasks


def check_file_counts(sub_root: Path, sub: str, expected_by_package: dict):
    mismatches = []
    for package, rel_dirs in PACKAGE_SUBDIRS.items():
        expected = expected_by_package.get(package)
        if expected is None:
            mismatches.append((package, "no manifest entry (subject/package missing from manifest)", None, None))
            continue
        actual = count_files(sub_root, rel_dirs)
        if actual != expected:
            mismatches.append((package, "file count mismatch vs manifest", expected, actual))
    return mismatches


def scan_prf_keywords(sub_root: Path):
    func_dir = sub_root / "derivatives/func"
    if not func_dir.is_dir():
        return [], []
    all_names = [p.name.lower() for p in func_dir.rglob("*") if p.is_file()]
    strong_hits = sorted({n for n in all_names if any(k in n for k in PRF_KEYWORDS_STRONG)})
    broad_hits = sorted({n for n in all_names if any(k in n for k in PRF_KEYWORDS)})
    return strong_hits, broad_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    bids_root = Path(args.bids_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = load_manifest(Path(args.manifest))
    subjects = args.subjects if args.subjects else sorted(expected.keys())

    report_rows = []
    clean_subjects, flagged_subjects = [], []
    any_strong_prf_hits = []

    for sub_id in subjects:
        sub = f"sub-{sub_id}"
        sub_root = bids_root  # required-file templates already include rawdata/derivatives
        # required files are relative to bids_root but need per-subject substitution
        sub_root_view = _SubjectView(bids_root, sub_id)

        missing = check_required_files(sub_root_view, sub)
        missing_tasks = check_retinotopy_tasks(sub_root_view)
        count_mismatches = check_file_counts(sub_root_view, sub, expected.get(sub_id, {}))
        strong_hits, broad_hits = scan_prf_keywords(sub_root_view)

        if strong_hits:
            any_strong_prf_hits.append(sub_id)

        problems = []
        for label, package, path in missing:
            problems.append(f"MISSING {label} ({package}): {path}")
            report_rows.append([sub_id, "missing_required", label, package, path])
        for task in missing_tasks:
            problems.append(f"MISSING retinotopy run: {task}")
            report_rows.append([sub_id, "missing_retinotopy_run", task, "Retinotopy7TRecommended", ""])
        for package, reason, exp, act in count_mismatches:
            problems.append(f"{package}: {reason} (expected {exp}, found {act})")
            report_rows.append([sub_id, "count_mismatch", reason, package, f"expected={exp} actual={act}"])

        if problems:
            flagged_subjects.append(sub_id)
            print(f"sub-{sub_id}: FLAGGED")
            for p in problems:
                print(f"    {p}")
        else:
            clean_subjects.append(sub_id)
            print(f"sub-{sub_id}: clean")

    report_path = out_dir / "qc_completeness_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "check_type", "detail", "package", "extra"])
        writer.writerows(report_rows)

    print(f"\n=== Summary ===")
    print(f"{len(clean_subjects)}/{len(subjects)} subjects clean, {len(flagged_subjects)} flagged")
    if flagged_subjects:
        print(f"Flagged: {flagged_subjects}")
    print(f"Report: {report_path}")

    print(f"\n=== pRF solution data status ===")
    if any_strong_prf_hits:
        print(f"Strong pRF-keyword filename matches found for {len(any_strong_prf_hits)} subjects "
              f"(e.g. {any_strong_prf_hits[:5]}) -- pRF solution maps may already be present. "
              f"Inspect actual filenames before trusting this -- keyword match, not a format check.")
    else:
        print("No strong pRF-keyword filename matches found in any subject's derivatives/func. "
              "This suggests the extracted Retinotopy7TRecommended package contains preprocessed "
              "BOLD timeseries only, NOT fitted pRF parameter maps (eccentricity/polar angle/size/R2). "
              "If so, Benson et al.'s pRF solutions need to be pulled separately from OSF/BALSA "
              "before Claim 2 or the hOc1 boundary QC can proceed.")


class _SubjectView:
    """Resolves the {sub}-templated REQUIRED paths and the fixed package subdirs
    against bids_root/{rawdata,derivatives/hcp_pipelines}/sub-<ID>/, without
    duplicating path-join logic in every check function."""
    def __init__(self, bids_root: Path, subject_id: str):
        self.bids_root = bids_root
        self.subject_id = subject_id
        self.sub = f"sub-{subject_id}"

    def __truediv__(self, rel: str) -> Path:
        rel = str(rel)
        if rel.startswith("rawdata/"):
            return self.bids_root / "rawdata" / self.sub / rel[len("rawdata/"):]
        if rel.startswith("derivatives/"):
            return self.bids_root / "derivatives" / "hcp_pipelines" / self.sub / rel[len("derivatives/"):]
        raise ValueError(f"Unexpected relative path root: {rel}")


if __name__ == "__main__":
    main()
