#!/usr/bin/env python3
"""
Compute average and modal controllability per node per subject from structural
connectivity matrices, using nctpy (Parkes, Kim et al., Nature Protocols 2024)
-- the field-standard NCT toolbox (decisions log, 2026-08-07).

Normalization: discrete-time system (nctpy.utils.matrix_normalization with
system="discrete"), the standard choice in the NCT literature this project is
following (Gu et al. 2015; Parkes/Kim et al. 2024). Modal controllability in
nctpy is defined only for discrete systems -- using "continuous" here would
silently break modal_control, so --system is deliberately restricted to
"discrete" as the only supported value for now.

Expects one connectivity matrix per subject per atlas (from Phase 5 --
tck2connectome.py), square, symmetric, node order matching --labels-file
(built by extract_atlas_labels.py, in ascending-label-key order -- confirmed
2026-08-11 to match tck2connectome's node indexing). Cortical nodes only for
now (200 for Schaefer-200, 360 for Glasser MMP1.0) -- subcortical (LGN/SC/
pulvinar) is a separate, not-yet-built follow-up, not included in these
matrices.

Real output layout (confirmed 2026-08-11, not the flat
sub-<ID>_connectome.<ext> layout this script originally assumed):
    <connectome-dir>/sub-<ID>/connectome_<atlas>.csv
where <connectome-dir> is the connectomes root (contains all subjects, both
atlases) and --atlas selects which file to read per subject.

Usage:
    python3 compute_controllability.py \
        --connectome-dir /path/to/derivatives/connectomes \
        --atlas schaefer200 \
        --labels-file /path/to/schaefer200_labels.txt \
        --out-dir /path/to/derivatives/controllability \
        [--system discrete] [--norm-c 1]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from nctpy import metrics, utils

SUPPORTED_SYSTEMS = ("discrete",)  # modal_control in nctpy is discrete-only


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
    """Real layout (confirmed 2026-08-11): <connectome_dir>/sub-<ID>/connectome_<atlas>.csv"""
    candidate = connectome_dir / f"sub-{subject}" / f"connectome_{atlas}.csv"
    return candidate if candidate.exists() else None


def discover_subjects(connectome_dir: Path):
    return sorted(
        p.name.replace("sub-", "")
        for p in connectome_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def compute_for_subject(A: np.ndarray, labels, system: str, norm_c: float):
    n = A.shape[0]
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels file has {len(labels)} entries, matrix has {n} nodes")

    # Symmetry / basic sanity checks -- surfaced, not silently ignored,
    # consistent with this project's "never silently drop/alter data" principle.
    warnings = []
    if not np.allclose(A, A.T, atol=1e-6):
        warnings.append("matrix is not symmetric within tolerance")
    if np.any(np.diag(A) != 0):
        warnings.append("nonzero diagonal (self-connections) present")
    zero_rows = np.where(A.sum(axis=1) == 0)[0]
    if len(zero_rows) > 0:
        node_desc = [labels[i] if labels else str(i) for i in zero_rows]
        warnings.append(f"{len(zero_rows)} disconnected node(s): {node_desc}")

    A_norm = utils.matrix_normalization(A, system=system, c=norm_c)
    ave = metrics.ave_control(A_norm, system=system)
    modal = metrics.modal_control(A_norm)

    return ave, modal, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome-dir", required=True,
                     help="connectomes root, containing sub-<ID>/connectome_<atlas>.csv per subject")
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--labels-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="subject IDs to process; default = all sub-<ID> dirs found in --connectome-dir")
    ap.add_argument("--system", default="discrete", choices=SUPPORTED_SYSTEMS)
    ap.add_argument("--norm-c", type=float, default=1.0)
    args = ap.parse_args()

    connectome_dir = Path(args.connectome_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels_file))

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(connectome_dir)

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
            ave, modal, warnings = compute_for_subject(A, labels, args.system, args.norm_c)
        except Exception as e:
            print(f"sub-{subject}: FAILED -- {type(e).__name__}: {e}")
            n_failed += 1
            continue

        for w in warnings:
            print(f"sub-{subject}: WARNING -- {w}")

        sub_out_dir = out_dir / f"sub-{subject}"
        sub_out_dir.mkdir(parents=True, exist_ok=True)
        sub_out = sub_out_dir / f"controllability_{args.atlas}.csv"
        with open(sub_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_label", "average_controllability", "modal_controllability"])
            for label, a_val, m_val in zip(labels, ave, modal):
                writer.writerow([label, a_val, m_val])
                combined_rows.append([subject, label, a_val, m_val])

        n_ok += 1
        print(f"sub-{subject}: OK -- {len(labels)} nodes -> {sub_out}")

    combined_path = out_dir / f"all_subjects_controllability_{args.atlas}.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "node_label", "average_controllability", "modal_controllability"])
        writer.writerows(combined_rows)

    print(f"\nDone. {n_ok} subjects ok, {n_failed} failed/skipped.")
    print(f"Combined table: {combined_path} ({len(combined_rows)} rows)")


if __name__ == "__main__":
    main()
