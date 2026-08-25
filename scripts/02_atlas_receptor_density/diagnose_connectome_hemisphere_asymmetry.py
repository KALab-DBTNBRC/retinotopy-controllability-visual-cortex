#!/usr/bin/env python3
"""
Diagnostic for the real, systematic right>left average_controllability
asymmetry found in all_subjects_controllability_schaefer200.csv (2026-08-11):
whole-brain, 46/46 subjects, mean RH/LH ratio 9.4x. A first pass (this same
script, whole-brain only) found the RAW pre-controllability connectome
balanced overall (degree ratio R/L: mean 0.966, range 0.878-1.041) -- ruling
out a whole-brain tck2connectome.py/tractography problem and pointing at
compute_controllability.py instead.

BUT a follow-up check on the real controllability output (log-space/geometric-
mean, appropriate given average controllability's heavy-tailed distribution)
found the asymmetry is NOT uniform across the brain: every non-visual 7Networks
network (Cont, SomMot, Default, DorsAttn, SalVentAttn, Limbic) shows only a
modest 1.0-1.5x geometric-mean R/L ratio (plausible/expected), while the Vis
network specifically shows 8.6x (geometric mean, up to 74x) -- and within one
subject (123456), 13 of 15 RH Vis nodes exceed EVERY LH Vis node's value, not
a couple of outliers. Since compute_controllability.py has no network-specific
code path (every node goes through the identical normalization/ave_control
call), it cannot explain a Vis-only effect -- so this points back to the
connectome again, but LOCALLY at occipital cortex, which a whole-brain-
averaged degree check would have masked.

This version adds --network-filter (substring match against node_label, e.g.
"_Vis_") to restrict the SAME raw-connectome degree/intra-hemisphere-weight
check to just that subset of nodes, so the localized-effect hypothesis can be
tested directly instead of assumed. Whole-brain numbers are still reported
unconditionally for reference.

Usage:
    # whole-brain (as before):
    python3 diagnose_connectome_hemisphere_asymmetry.py \
        --connectome-dir /path/to/project/BIDS/derivatives/connectomes \
        --atlas schaefer200 \
        --labels-file /path/to/project/Atlas/schaefer200_labels.txt \
        --out-csv /path/to/project/BIDS/derivatives/hemisphere_asymmetry_diagnostic_schaefer200.csv

    # restricted to the Vis network only (the follow-up check):
    python3 diagnose_connectome_hemisphere_asymmetry.py \
        --connectome-dir /path/to/project/BIDS/derivatives/connectomes \
        --atlas schaefer200 \
        --labels-file /path/to/project/Atlas/schaefer200_labels.txt \
        --network-filter _Vis_ \
        --out-csv /path/to/project/BIDS/derivatives/hemisphere_asymmetry_diagnostic_schaefer200_vis.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        A = np.load(path)
    else:
        A = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{path.name}: expected square matrix, got shape {A.shape}")
    return A


def load_labels(path: Path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def hemisphere_of(label: str):
    """Real per-node label text, e.g. '7Networks_LH_Vis_1' (Schaefer-200) or
    'L_V1_ROI' / 'R_V1_ROI' (Glasser) -- NOT index position, see module
    docstring. Returns 'L', 'R', or None if the label doesn't clearly say."""
    if "_LH_" in label or label.startswith("L_") or "_L_" in label:
        return "L"
    if "_RH_" in label or label.startswith("R_") or "_R_" in label:
        return "R"
    return None


def find_subject_matrix(connectome_dir: Path, subject: str, atlas: str):
    candidate = connectome_dir / f"sub-{subject}" / f"connectome_{atlas}.csv"
    return candidate if candidate.exists() else None


def discover_subjects(connectome_dir: Path):
    return sorted(
        p.name.replace("sub-", "")
        for p in connectome_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def diagnose_subject(A, labels, network_filter=None):
    n = A.shape[0]
    hemi = [hemisphere_of(l) for l in labels]
    unlabeled = [labels[i] for i, h in enumerate(hemi) if h is None]

    if network_filter:
        keep = [i for i, l in enumerate(labels) if network_filter in l]
    else:
        keep = list(range(n))
    keep_set = set(keep)

    l_idx = [i for i, h in enumerate(hemi) if h == "L" and i in keep_set]
    r_idx = [i for i, h in enumerate(hemi) if h == "R" and i in keep_set]

    # degree computed against the FULL matrix (a node's real total weighted
    # connectivity to the whole brain), just restricted to WHICH nodes we
    # average over -- not the matrix itself, so this stays a fair "does this
    # subset of nodes have balanced real connectivity" check.
    degree = A.sum(axis=1)
    l_degree_mean = float(np.mean(degree[l_idx])) if l_idx else float("nan")
    r_degree_mean = float(np.mean(degree[r_idx])) if r_idx else float("nan")

    intra_l = float(A[np.ix_(l_idx, l_idx)].sum()) if l_idx else float("nan")
    intra_r = float(A[np.ix_(r_idx, r_idx)].sum()) if r_idx else float("nan")
    inter = float(A[np.ix_(l_idx, r_idx)].sum()) if l_idx and r_idx else float("nan")

    return {
        "n_nodes": n,
        "n_left": len(l_idx),
        "n_right": len(r_idx),
        "n_unlabeled": len(unlabeled),
        "mean_degree_left": l_degree_mean,
        "mean_degree_right": r_degree_mean,
        "degree_ratio_R_over_L": r_degree_mean / l_degree_mean if l_degree_mean else float("nan"),
        "intra_hemi_weight_left": intra_l,
        "intra_hemi_weight_right": intra_r,
        "intra_ratio_R_over_L": intra_r / intra_l if intra_l else float("nan"),
        "inter_hemi_weight": inter,
        "unlabeled_examples": ";".join(unlabeled[:5]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome-dir", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--labels-file", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--network-filter", default=None,
                     help="substring to restrict which nodes count toward the L/R stats, e.g. '_Vis_' "
                          "-- degree is still each kept node's TOTAL connectivity to the whole brain, "
                          "only the set of nodes being averaged/summed is restricted")
    args = ap.parse_args()

    connectome_dir = Path(args.connectome_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels_file))

    hemi_check = [hemisphere_of(l) for l in labels]
    print(f"Labels file: {len(labels)} nodes -- "
          f"{sum(1 for h in hemi_check if h=='L')} L, "
          f"{sum(1 for h in hemi_check if h=='R')} R, "
          f"{sum(1 for h in hemi_check if h is None)} unlabeled/ambiguous")
    if any(h is None for h in hemi_check):
        unlabeled = [l for l, h in zip(labels, hemi_check) if h is None]
        print(f"WARNING: could not determine hemisphere for {len(unlabeled)} labels "
              f"(first 10: {unlabeled[:10]}) -- these are excluded from L/R sums below, "
              f"not silently included in either side.")
    if args.network_filter:
        n_kept = sum(1 for l in labels if args.network_filter in l)
        print(f"--network-filter '{args.network_filter}': {n_kept}/{len(labels)} nodes match")

    subjects = args.subjects if args.subjects else discover_subjects(connectome_dir)
    if not subjects:
        print(f"No sub-<ID> directories found in {connectome_dir}")
        sys.exit(1)

    rows = []
    n_ok, n_failed = 0, 0
    for subject in subjects:
        mat_path = find_subject_matrix(connectome_dir, subject, args.atlas)
        if mat_path is None:
            print(f"sub-{subject}: SKIPPED -- no connectome_{args.atlas}.csv found")
            n_failed += 1
            continue
        try:
            A = load_matrix(mat_path)
            if A.shape[0] != len(labels):
                raise ValueError(f"matrix has {A.shape[0]} nodes, labels file has {len(labels)}")
            stats = diagnose_subject(A, labels, network_filter=args.network_filter)
        except Exception as e:
            print(f"sub-{subject}: FAILED -- {type(e).__name__}: {e}")
            n_failed += 1
            continue
        stats["subject"] = subject
        rows.append(stats)
        n_ok += 1

    fieldnames = ["subject", "n_nodes", "n_left", "n_right", "n_unlabeled",
                  "mean_degree_left", "mean_degree_right", "degree_ratio_R_over_L",
                  "intra_hemi_weight_left", "intra_hemi_weight_right", "intra_ratio_R_over_L",
                  "inter_hemi_weight", "unlabeled_examples"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        degree_ratios = [r["degree_ratio_R_over_L"] for r in rows if np.isfinite(r["degree_ratio_R_over_L"])]
        intra_ratios = [r["intra_ratio_R_over_L"] for r in rows if np.isfinite(r["intra_ratio_R_over_L"])]
        print(f"\n=== Summary across {len(rows)} subjects (RAW connectome, pre-controllability"
              f"{f', filtered to '+args.network_filter if args.network_filter else ''}) ===")
        if degree_ratios:
            print(f"Weighted degree R/L ratio: mean={np.mean(degree_ratios):.3f} "
                  f"median={np.median(degree_ratios):.3f} min={np.min(degree_ratios):.3f} "
                  f"max={np.max(degree_ratios):.3f}")
        if intra_ratios:
            print(f"Intra-hemisphere total edge weight R/L ratio: mean={np.mean(intra_ratios):.3f} "
                  f"median={np.median(intra_ratios):.3f} min={np.min(intra_ratios):.3f} "
                  f"max={np.max(intra_ratios):.3f}")
        print("\nIf these ratios are close to 1.0, the raw connectome (for this node subset) is "
              "balanced and the asymmetry is being introduced downstream. If these ratios already "
              "show a strong right skew, the problem is upstream (tck2connectome.py or the "
              "tractography itself), localized to whatever --network-filter subset was checked.")

    print(f"\nDone. {n_ok} subjects ok, {n_failed} failed/skipped. Written to {out_csv}")


if __name__ == "__main__":
    main()
