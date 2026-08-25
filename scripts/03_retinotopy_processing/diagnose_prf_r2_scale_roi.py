#!/usr/bin/env python3
"""
ROI-restricted version of diagnose_prf_r2_scale.py: instead of sweeping R2
thresholds across all 91282 grayordinates (which includes non-visual cortex
and subcortex, where near-zero R2 is expected and correct), restrict to a
real retinotopic ROI mask first, then check what fraction of THAT region
survives each candidate threshold with a plausible eccentricity range.

Uses the Wang et al. (2015) probabilistic visual topography atlas bundled in
Benson's own OSF release (atlas.mat, wang2015/wang2015labels), which is
already in the same 91282-grayordinate space as prfresults.mat -- verified
2026-08-07 via direct h5py inspection (wang2015.shape == (1, 91282)), no
resampling needed. This is a closer proxy for what matters (does our actual
target region -- roughly the hOc1-hOc5 territory -- survive thresholding)
than a whole-brain percentage, which mixes in a huge amount of cortex/
subcortex that was never expected to show retinotopic signal.

Usage:
    python3 diagnose_prf_r2_scale_roi.py \
        --mat-file /path/to/prfresults.mat \
        --atlas-mat /path/to/atlas.mat \
        --subjects 123456 234567 345678 \
        --roi V1v V1d V2v V2d V3v V3d hV4 \
        [--thresholds 0.1 1 10 30 50 70]

--roi: names must match wang2015labels exactly (case-sensitive). Default ROI
is V1 only (V1v + V1d) -- pass a broader list for the early-visual-cortex
proxy closer to hOc1-hOc5's span.
"""
import argparse

import h5py
import numpy as np

EXPECTED_QUANTS = ["ang", "ecc", "gain", "meanvol", "R2", "rfsize"]


def decode_matlab_string(f: h5py.File, ref) -> str:
    data = f[ref][()]
    return "".join(chr(int(c)) for c in data.flatten())


def load_quants(f: h5py.File):
    quants = [decode_matlab_string(f, r[0]) for r in f["quants"][:]]
    if quants != EXPECTED_QUANTS:
        raise ValueError(f"quants order ({quants}) != expected ({EXPECTED_QUANTS}), aborting.")
    return quants


def load_wang_roi_mask(atlas_path: str, roi_names):
    with h5py.File(atlas_path, "r") as af:
        labels_ds = af["wang2015labels"]
        label_names = [decode_matlab_string(af, r[0]) for r in labels_ds[:]]
        wang = af["wang2015"][:].flatten()  # (91282,) float64, integer-valued labels

    name_to_idx = {name: i for i, name in enumerate(label_names)}
    missing = [r for r in roi_names if r not in name_to_idx]
    if missing:
        raise ValueError(f"ROI name(s) not found in wang2015labels: {missing}. "
                          f"Available: {label_names}")
    roi_indices = {name_to_idx[r] for r in roi_names}
    mask = np.isin(wang.astype(int), list(roi_indices))
    return mask, label_names


def find_subject_index(sids: np.ndarray, subject_id: str):
    matches = np.where(sids.astype(int) == int(subject_id))[0]
    return int(matches[0]) if len(matches) else None


def diagnose_subject_roi(f: h5py.File, subj_idx: int, fit_index: int, thresholds, roi_mask: np.ndarray):
    data = f["allresults"][fit_index, subj_idx, :, :]  # (6, n_grayordinates)
    ecc = data[EXPECTED_QUANTS.index("ecc"), :][roi_mask]
    r2 = data[EXPECTED_QUANTS.index("R2"), :][roi_mask]

    finite = np.isfinite(ecc) & np.isfinite(r2)
    ecc, r2 = ecc[finite], r2[finite]
    n_roi = len(ecc)

    rows = []
    for thresh in thresholds:
        passing = r2 > thresh
        n_pass = int(passing.sum())
        pct_of_roi = n_pass / n_roi if n_roi else 0.0
        if n_pass == 0:
            rows.append((thresh, n_pass, pct_of_roi, None, None, None, None))
            continue
        ecc_pass = ecc[passing]
        rows.append((
            thresh, n_pass, pct_of_roi,
            float(np.mean(ecc_pass)), float(np.percentile(ecc_pass, 95)),
            float(np.percentile(ecc_pass, 99)), float(np.max(ecc_pass)),
        ))
    return n_roi, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-file", required=True)
    ap.add_argument("--atlas-mat", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--roi", nargs="+", default=["V1v", "V1d"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 1, 10, 30, 50, 70])
    ap.add_argument("--fit-index", type=int, default=0)
    args = ap.parse_args()

    roi_mask, all_labels = load_wang_roi_mask(args.atlas_mat, args.roi)
    print(f"ROI = {args.roi} -> {roi_mask.sum()} grayordinates total "
          f"(out of {roi_mask.size}, {roi_mask.mean():.2%} of all grayordinates)")
    print(f"(all available wang2015 labels: {all_labels})\n")

    header = f"{'threshold':>10} {'n_pass':>8} {'%_of_ROI':>9} {'ecc_mean':>9} {'ecc_p95':>8} {'ecc_p99':>8} {'ecc_max':>8}"

    with h5py.File(args.mat_file, "r") as f:
        load_quants(f)
        sids = f["subjectids"][:].flatten()

        for subject_id in args.subjects:
            idx = find_subject_index(sids, subject_id)
            if idx is None:
                print(f"sub-{subject_id}: NOT FOUND")
                continue
            n_roi, rows = diagnose_subject_roi(f, idx, args.fit_index, args.thresholds, roi_mask)
            print(f"=== sub-{subject_id} (ROI has {n_roi} grayordinates) ===")
            print(header)
            for thresh, n_pass, pct_roi, mean_e, p95, p99, mx in rows:
                if mean_e is None:
                    print(f"{thresh:>10} {n_pass:>8} {pct_roi:>9.1%} {'--':>9} {'--':>8} {'--':>8} {'--':>8}")
                else:
                    print(f"{thresh:>10} {n_pass:>8} {pct_roi:>9.1%} {mean_e:>9.2f} {p95:>8.2f} {p99:>8.2f} {mx:>8.2f}")
            print()

    print("%_of_ROI is now relative to the ROI's own grayordinate count, not the whole "
          "brain -- this is the number that actually matters for whether we'll have "
          "enough coverage within hOc1-hOc5-equivalent territory at a given threshold.")


if __name__ == "__main__":
    main()
