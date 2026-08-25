#!/usr/bin/env python3
"""
Extract Benson et al. (2018) HCP 7T Retinotopy pRF solutions from prfresults.mat
(OSF: https://osf.io/bw9ec, DOI 10.17605/OSF.IO/BW9EC) into per-subject CIFTI
dscalar files, one per cohort subject.

Why this file and not prfresultsnii.zip / prfresultsmgz.zip: prfresultsnii.zip
is subcortex-only (confirmed 2026-08-07 via real `unzip -l` listing -- useful
later for the Benson subcortical ROI masks, not for cortical pRF maps).
prfresultsmgz.zip has the cortical maps but in FreeSurfer-native per-hemisphere
MGZ format, which would need resampling to fs_LR32k ourselves. prfresults.mat
already stores everything in the standard 91282-grayordinate dense CIFTI space
(confirmed via real shape inspection: allresults.shape == (3, 184, 6, 91282)),
matching the space our Schaefer-200/Glasser CIFTI parcellations are already in
-- no resampling needed.

Verified structure (2026-08-07, against the real file on the server):
  allresults        (3 fits, 184 subjects, 6 quantities, 91282 grayordinates), float32
  quants            6 entries, decoded: ['ang','ecc','gain','meanvol','R2','rfsize']
  subjectids        184 entries (181 real HCP subjects + 3 group-average pseudo-subjects)
  All 46 of this project's cohort subjects confirmed present (0 missing).

Fit index: this script defaults to fit index 0. NOTE -- this is an inference,
not something labeled inside the .mat file itself (no 'fitnames' key exists).
It's based on the external OSF/BALSA documentation's naming convention
(fit1=full dataset/all runs, fit2=first-half-of-each-run, fit3=second-half,
matching the fit1/fit2/fit3 naming also seen in prfresultsmgz.zip's filenames)
and the fact that index 0 is the natural "primary" position. Flagged here so
this assumption is visible and revisitable, not buried.

R2 threshold (2026-08-08): the "R2" quantity in this file is NOT on a [0,1]
scale -- confirmed via diagnose_prf_r2_scale.py (whole-brain) and
diagnose_prf_r2_scale_roi.py (restricted to real Wang et al. 2015 V1/V1-hV4
ROI masks from atlas.mat, same OSF release). Real per-subject ROI-restricted
runs (123456, 234567, 345678; both V1-only and the broader V1-hV4 proxy)
showed: at R2>0.1 or R2>1, 91-99% of the ROI "passes" but eccentricity still
pins at the 16-degree aperture boundary at p99/max -- these thresholds are
functionally no-ops, not real filtering. At R2>10, p95 eccentricity looks
plausible (~7.9-8.0 deg) but the tail is still contaminated -- p99/max reach
13-16 deg for 2 of 3 subjects, meaning real garbage fits still get through.
At R2>30, eccentricity converges to a plausible, consistent range for every
subject (p95 ~7.0-7.9 deg, p99 ~7.9-8.0 deg, max ~8.0-8.07 deg, one 10.97
outlier) while retaining 45-75% of the ROI. At R2>50, coverage collapses and
becomes wildly inconsistent across subjects (0.5% for one subject, ~50% for
another) -- unusable as a fixed cutoff. At R2>70, 2 of 3 subjects lose the
ROI entirely. R2>30 was chosen as the cutoff: it's the point where the
eccentricity distribution stabilizes to the same plausible range across all
subjects, and the last threshold before coverage becomes subject-inconsistent.
This is reflected below via R2_THRESHOLD and the QC column name/values.

Usage:
    python3 extract_prf_solutions.py \
        --mat-file /path/to/prfresults/prfresults.mat \
        --template-cifti /path/to/any/91282-grayordinate/*.dscalar.nii \
        --subjects 123456 234567 ... \
        --out-dir /path/to/derivatives/prf \
        [--fit-index 0]

--template-cifti: any real HCP CIFTI dscalar/dtseries file already in this
project's derivatives (e.g. one of the subject-level RET fsaverage_LR32k
dscalar files hcp_bids_map.py already captured) -- its BrainModelAxis (the
grayordinate structure/ordering) is copied onto the new pRF output so the
result is a standards-conformant CIFTI file, not a bare numpy array.
"""
import argparse
import csv
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

EXPECTED_QUANTS = ["ang", "ecc", "gain", "meanvol", "R2", "rfsize"]

# R2 is not on a [0,1] scale in this file -- see module docstring "R2 threshold"
# section for the real ROI-restricted evidence behind this cutoff (2026-08-08).
R2_THRESHOLD = 30.0


def decode_matlab_string(f: h5py.File, ref) -> str:
    data = f[ref][()]
    return "".join(chr(int(c)) for c in data.flatten())


def load_quants(f: h5py.File):
    quants = [decode_matlab_string(f, r[0]) for r in f["quants"][:]]
    if quants != EXPECTED_QUANTS:
        raise ValueError(
            f"quants order in this .mat file ({quants}) does not match the "
            f"expected/verified order ({EXPECTED_QUANTS}) -- do not proceed blindly, "
            f"the quant->index mapping below would silently mislabel data."
        )
    return quants


def find_subject_index(sids: np.ndarray, subject_id: str):
    target = int(subject_id)
    matches = np.where(sids.astype(int) == target)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def extract_subject(f: h5py.File, subj_idx: int, fit_index: int) -> np.ndarray:
    """Returns (6, n_grayordinates) float32 array: rows in EXPECTED_QUANTS order."""
    return f["allresults"][fit_index, subj_idx, :, :]


def build_dscalar(data: np.ndarray, template_path: Path, quant_names) -> nib.Cifti2Image:
    template = nib.load(str(template_path))
    brain_model_axis = template.header.get_axis(1)
    if brain_model_axis.size != data.shape[1]:
        raise ValueError(
            f"template CIFTI has {brain_model_axis.size} grayordinates, "
            f"but extracted pRF data has {data.shape[1]} -- template mismatch, "
            f"do not proceed (this would silently misalign the data)."
        )
    scalar_axis = nib.cifti2.ScalarAxis([f"prf_{q}" for q in quant_names])
    header = nib.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    return nib.Cifti2Image(data.astype(np.float32), header=header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-file", required=True)
    ap.add_argument("--template-cifti", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fit-index", type=int, default=0,
                     help="0=fit1 (full data, default), 1=fit2 (first-half-of-run), "
                          "2=fit3 (second-half-of-run). See module docstring re: this "
                          "being an inferred mapping, not one labeled in the file itself.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template_path = Path(args.template_cifti)

    qc_rows = []
    n_ok, n_missing = 0, 0

    with h5py.File(args.mat_file, "r") as f:
        quants = load_quants(f)
        sids = f["subjectids"][:].flatten()

        for subject_id in args.subjects:
            idx = find_subject_index(sids, subject_id)
            if idx is None:
                print(f"sub-{subject_id}: NOT FOUND in {args.mat_file}")
                n_missing += 1
                continue

            data = extract_subject(f, idx, args.fit_index)  # (6, n_grayordinates)
            img = build_dscalar(data, template_path, quants)

            out_path = out_dir / f"sub-{subject_id}_prf.dscalar.nii"
            nib.save(img, str(out_path))

            for qi, qname in enumerate(quants):
                vals = data[qi, :]
                finite = vals[np.isfinite(vals)]
                row = [subject_id, qname, float(np.min(finite)) if finite.size else "",
                       float(np.max(finite)) if finite.size else "",
                       float(np.mean(finite)) if finite.size else ""]
                if qname == "R2":
                    pct_good = float(np.mean(finite > R2_THRESHOLD)) if finite.size else 0.0
                    row.append(pct_good)
                else:
                    row.append("")
                qc_rows.append(row)

            print(f"sub-{subject_id}: OK -> {out_path.name}")
            n_ok += 1

    qc_path = out_dir / "prf_extraction_qc.csv"
    with open(qc_path, "w", newline="") as qcf:
        writer = csv.writer(qcf)
        writer.writerow(["subject", "quant", "min", "max", "mean", f"pct_grayordinates_R2_gt_{R2_THRESHOLD:g}"])
        writer.writerows(qc_rows)

    print(f"\nDone. {n_ok} subjects extracted, {n_missing} not found.")
    print(f"QC summary: {qc_path}")


if __name__ == "__main__":
    main()
