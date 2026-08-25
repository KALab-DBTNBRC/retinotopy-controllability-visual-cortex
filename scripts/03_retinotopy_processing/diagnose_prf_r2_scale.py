#!/usr/bin/env python3
"""
Diagnose the actual scale of the "R2" quantity in Benson et al.'s prfresults.mat
and find the eccentricity-plausibility-recovering threshold.

Background (2026-08-07): extract_prf_solutions.py's QC pass showed per-subject
R2 max values of 31.8-82.3 and means of 0.6-3.6 -- inconsistent with a standard
[0,1] R^2 fraction. Max eccentricity values of 17.1-20.1 degrees, roughly double
the ~8 degree stimulus radius already established from Benson et al.'s 16-degree
aperture, are consistent with R2>0.1 (assuming a [0,1] scale) filtering almost
nothing, letting poorly-fit grayordinates with unconstrained eccentricity
estimates through.

Method: for each candidate R2 threshold, compute the eccentricity distribution
among grayordinates passing that threshold. If R2 is really on a 0-100 scale,
low thresholds (0.1, 1) should barely filter anything and eccentricity should
stay implausibly wide; somewhere around the true "10% variance explained"
equivalent, eccentricity should snap into the ~0-8 degree range as garbage fits
get excluded. That inflection point identifies both the real scale and the
correct cutoff to use in all downstream analysis (Claim 2, hOc1 boundary QC).

Usage:
    python3 diagnose_prf_r2_scale.py \
        --mat-file /path/to/prfresults.mat \
        --subjects 123456 234567 345678 \
        [--thresholds 0.1 1 10 30 50 70]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

EXPECTED_QUANTS = ["ang", "ecc", "gain", "meanvol", "R2", "rfsize"]


def decode_matlab_string(f: h5py.File, ref) -> str:
    data = f[ref][()]
    return "".join(chr(int(c)) for c in data.flatten())


def load_quants(f: h5py.File):
    quants = [decode_matlab_string(f, r[0]) for r in f["quants"][:]]
    if quants != EXPECTED_QUANTS:
        raise ValueError(
            f"quants order ({quants}) doesn't match expected ({EXPECTED_QUANTS}) -- "
            f"index mapping below would be wrong, aborting."
        )
    return quants


def find_subject_index(sids: np.ndarray, subject_id: str):
    matches = np.where(sids.astype(int) == int(subject_id))[0]
    return int(matches[0]) if len(matches) else None


def diagnose_subject(f: h5py.File, subj_idx: int, fit_index: int, thresholds):
    data = f["allresults"][fit_index, subj_idx, :, :]  # (6, n_grayordinates)
    ecc = data[EXPECTED_QUANTS.index("ecc"), :]
    r2 = data[EXPECTED_QUANTS.index("R2"), :]

    finite = np.isfinite(ecc) & np.isfinite(r2)
    ecc, r2 = ecc[finite], r2[finite]

    rows = []
    for thresh in thresholds:
        passing = r2 > thresh
        n_pass = int(passing.sum())
        pct_pass = n_pass / len(r2) if len(r2) else 0.0
        if n_pass == 0:
            rows.append((thresh, n_pass, pct_pass, None, None, None, None))
            continue
        ecc_pass = ecc[passing]
        rows.append((
            thresh, n_pass, pct_pass,
            float(np.mean(ecc_pass)),
            float(np.percentile(ecc_pass, 95)),
            float(np.percentile(ecc_pass, 99)),
            float(np.max(ecc_pass)),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-file", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 1, 10, 30, 50, 70])
    ap.add_argument("--fit-index", type=int, default=0)
    args = ap.parse_args()

    header = f"{'threshold':>10} {'n_pass':>8} {'%pass':>7} {'ecc_mean':>9} {'ecc_p95':>8} {'ecc_p99':>8} {'ecc_max':>8}"

    with h5py.File(args.mat_file, "r") as f:
        load_quants(f)  # validates order, raises if mismatched
        sids = f["subjectids"][:].flatten()

        for subject_id in args.subjects:
            idx = find_subject_index(sids, subject_id)
            if idx is None:
                print(f"sub-{subject_id}: NOT FOUND")
                continue
            print(f"\n=== sub-{subject_id} ===")
            print(header)
            rows = diagnose_subject(f, idx, args.fit_index, args.thresholds)
            for thresh, n_pass, pct_pass, mean_e, p95, p99, mx in rows:
                if mean_e is None:
                    print(f"{thresh:>10} {n_pass:>8} {pct_pass:>7.1%} {'--':>9} {'--':>8} {'--':>8} {'--':>8}")
                else:
                    print(f"{thresh:>10} {n_pass:>8} {pct_pass:>7.1%} {mean_e:>9.2f} {p95:>8.2f} {p99:>8.2f} {mx:>8.2f}")

    print("\nLook for the threshold where ecc_p95/ecc_p99/ecc_max drop to roughly the "
          "~8-degree stimulus radius (some tolerance expected near the aperture edge) "
          "and stay there as the threshold rises further -- that's the real cutoff.")


if __name__ == "__main__":
    main()
