#!/usr/bin/env python3
"""
SERVER-SIDE SCRIPT -- run on the lab data server (needs the real per-subject
connectome files under --connectome-dir; not available in the sandbox).

Follow-up to the 2026-08-13 "receptor vs. behavior, don't lean on NCT"
request: computes each subject's raw, per-node WEIGHTED STRUCTURAL
CONNECTIVITY STRENGTH (weighted degree -- the row-sum of the SIFT2-
weighted structural connectivity matrix at each node) instead of average/
modal controllability. This is standard graph theory (weighted degree
centrality), not network control theory -- no nctpy import anywhere in
this script, no eigenstructure/normalization step. It is the more direct,
literature-standard non-NCT bridge between fixed receptor density and
per-subject behavior that receptor_behavior_direct.py's anatomical
region-size bridge was a lighter-weight stand-in for (see that script's
docstring): "does a region actually carrying more/less of a given
receptor tend to be more/less strongly wired into the rest of the brain,
in THIS individual, and does that in turn relate to behavior."

Deliberately mirrors compute_controllability.py's I/O structure exactly
(same --connectome-dir/--atlas/--labels-file/--out-dir arguments, same
per-subject file discovery via find_subject_matrix/discover_subjects,
same output layout) so it's a drop-in sibling script, not a new pipeline
to learn -- only the per-subject metric computation differs.

Usage (same real connectome-dir layout as compute_controllability.py):
    python3 compute_node_strength.py \
        --connectome-dir /path/to/derivatives/connectomes \
        --atlas schaefer200 \
        --labels-file /path/to/schaefer200_labels.txt \
        --out-dir /path/to/derivatives/node_strength
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
        labels = [line.strip() for line in f if line.strip()]
    return labels


def find_subject_matrix(connectome_dir: Path, subject: str, atlas: str):
    """Same real layout as compute_controllability.py (confirmed 2026-08-11):
    <connectome_dir>/sub-<ID>/connectome_<atlas>.csv"""
    candidate = connectome_dir / f"sub-{subject}" / f"connectome_{atlas}.csv"
    return candidate if candidate.exists() else None


def discover_subjects(connectome_dir: Path):
    return sorted(
        p.name.replace("sub-", "")
        for p in connectome_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def compute_strength_for_subject(A: np.ndarray, labels):
    """Weighted degree (row-sum) per node -- the total SIFT2-weighted
    streamline connectivity of each node to the rest of the brain. Zero-
    diagonal is assumed (matches tck2connectome.py's real output, already
    QC'd for all 46 subjects in Phase 5), so this is a pure sum of a
    node's real edges, no self-connection contamination."""
    n = A.shape[0]
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels file has {len(labels)} entries, matrix has {n} nodes")

    warnings = []
    if not np.allclose(A, A.T, atol=1e-6):
        warnings.append("matrix is not symmetric within tolerance")
    if np.any(np.diag(A) != 0):
        warnings.append("nonzero diagonal (self-connections) present -- would inflate strength")

    strength = A.sum(axis=1)
    return strength, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome-dir", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--labels-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    connectome_dir = Path(args.connectome_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels_file))

    subjects = args.subjects if args.subjects else discover_subjects(connectome_dir)
    if not subjects:
        print(f"No sub-<ID> directories found in {connectome_dir}")
        sys.exit(1)

    combined_rows = []
    n_ok, n_failed = 0, 0

    for subject in subjects:
        mat_path = find_subject_matrix(connectome_dir, subject, args.atlas)
        if mat_path is None:
            print(f"sub-{subject}: SKIPPED -- no connectome_{args.atlas}.csv found")
            n_failed += 1
            continue
        try:
            A = load_matrix(mat_path)
            strength, warnings = compute_strength_for_subject(A, labels)
        except Exception as e:
            print(f"sub-{subject}: FAILED -- {type(e).__name__}: {e}")
            n_failed += 1
            continue

        for w in warnings:
            print(f"sub-{subject}: WARNING -- {w}")

        sub_out_dir = out_dir / f"sub-{subject}"
        sub_out_dir.mkdir(parents=True, exist_ok=True)
        sub_out = sub_out_dir / f"node_strength_{args.atlas}.csv"
        with open(sub_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_label", "weighted_degree"])
            for label, s_val in zip(labels, strength):
                writer.writerow([label, s_val])
                combined_rows.append([subject, label, s_val])

        n_ok += 1
        print(f"sub-{subject}: OK -- {len(labels)} nodes -> {sub_out}")

    combined_path = out_dir / f"all_subjects_node_strength_{args.atlas}.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "node_label", "weighted_degree"])
        writer.writerows(combined_rows)

    print(f"\nDone. {n_ok} subjects ok, {n_failed} failed/skipped.")
    print(f"Combined table: {combined_path} ({len(combined_rows)} rows)")


if __name__ == "__main__":
    main()
