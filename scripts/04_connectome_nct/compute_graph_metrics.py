#!/usr/bin/env python3
"""
SERVER-SIDE SCRIPT -- run on the analysis server (needs the real per-subject
connectome files under --connectome-dir; not available in the sandbox).

WHY THIS EXISTS
---------------
The manuscript's Introduction claims network control theory asks something
descriptive graph metrics cannot. The analysis then tests exactly ONE
descriptive alternative -- weighted degree -- and shows the eccentricity
effect survives adjustment for it.

Colleagues who read the draft correctly pointed out that this is not enough to
support the NCT-specific claim. If eccentricity predicts eigenvector
centrality, communicability and participation coefficient just as strongly as
it predicts average controllability, then the finding is about spectral/
centrality structure in general, not about controllability, and the NCT
framing should be weakened. If average controllability is selectively
predicted, the framing is earned.

This script computes those additional per-node graph metrics from the SAME
SIFT2-weighted matrices already used for controllability and node strength.
No network control theory is involved: no nctpy import, no Gramian, no
discrete-time normalization.

METRICS
-------
weighted_degree            row sum (recomputed here as an internal consistency
                           check against compute_node_strength.py's output)
eigenvector_centrality     leading eigenvector of A, sign-fixed non-negative
communicability            row sum of expm(A_norm), a walk-based centrality
                           that weights short walks more heavily
participation_coefficient  Guimera-Amaral, computed against the atlas's own
                           module assignment supplied via --modules-file

NORMALIZATION
-------------
Communicability uses expm, which overflows on raw SIFT2 weights. The matrix is
therefore divided by its largest eigenvalue first -- the SAME stabilization
already applied before controllability is computed, so the two are on
comparable footing. This is recorded per subject in the output so the choice
is auditable rather than silent.

I/O deliberately mirrors compute_node_strength.py and compute_controllability.py
exactly (same --connectome-dir/--atlas/--labels-file/--out-dir arguments, same
per-subject discovery) so this is a drop-in sibling, not a new pipeline.

Usage:
    python3 compute_graph_metrics.py \
        --connectome-dir /path/to/derivatives/connectomes \
        --atlas glasser \
        --labels-file /path/to/glasser_labels.txt \
        --out-dir /path/to/derivatives/graph_metrics
    python3 compute_graph_metrics.py --self-test
"""
import argparse, csv, sys
from pathlib import Path

import re

import numpy as np
from scipy.linalg import expm

SCHAEFER_NETWORK_RE = re.compile(r"^7Networks_(?:LH|RH)_([A-Za-z]+)_")


def modules_from_labels(labels):
    """Derive a module id per node from the label itself.

    Schaefer-200 labels carry their 7-network assignment inline
    (e.g. 7Networks_LH_Vis_1 -> 'Vis'), so the participation coefficient's
    module partition can be read off the labels rather than supplied as a
    separate file. Returns None if the labels do not follow that convention
    (e.g. Glasser), in which case the caller should fall back to
    --modules-file or skip the metric.
    """
    nets = []
    for l in labels:
        m = SCHAEFER_NETWORK_RE.match(l)
        if not m:
            return None
        nets.append(m.group(1))
    uniq = sorted(set(nets))
    lut = {n: i for i, n in enumerate(uniq)}
    print(f"  derived {len(uniq)} modules from labels: {', '.join(uniq)}")
    return np.array([lut[n] for n in nets])


def load_matrix(path: Path) -> np.ndarray:
    A = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {A.shape}")
    return A


def load_labels(path: Path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def find_subject_matrix(connectome_dir: Path, subject: str, atlas: str):
    p = connectome_dir / f"sub-{subject}" / f"connectome_{atlas}.csv"
    return p if p.exists() else None


def discover_subjects(connectome_dir: Path):
    return sorted(p.name.replace("sub-", "") for p in connectome_dir.iterdir()
                  if p.is_dir() and p.name.startswith("sub-"))


# ------------------------------------------------------------------ metrics --
def normalize_by_max_eig(A):
    """Divide by largest eigenvalue magnitude -- the same stabilization applied
    before controllability. Returns (A_norm, max_eig)."""
    w = np.linalg.eigvalsh(A) if np.allclose(A, A.T) else np.linalg.eigvals(A)
    m = float(np.max(np.abs(w)))
    if m <= 0 or not np.isfinite(m):
        raise ValueError("non-positive or non-finite largest eigenvalue")
    return A / m, m


def eigenvector_centrality(A):
    """Leading eigenvector, sign-fixed to be non-negative (Perron-Frobenius:
    for a connected non-negative matrix the leading eigenvector is
    single-signed, so flipping sign is a convention fix, not a data change)."""
    w, V = np.linalg.eigh(A) if np.allclose(A, A.T) else np.linalg.eig(A)
    idx = int(np.argmax(np.real(w)))
    v = np.real(V[:, idx])
    if v.sum() < 0:
        v = -v
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def communicability_strength(A_norm):
    """Row sums of the matrix exponential: total walk-weighted communicability
    of each node to all others."""
    return expm(A_norm).sum(axis=1)


def participation_coefficient(A, modules):
    """Guimera & Amaral participation coefficient on a weighted graph.
    modules: array of integer module ids, one per node."""
    k = A.sum(axis=1)
    pc = np.zeros(A.shape[0])
    for m in np.unique(modules):
        k_m = A[:, modules == m].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pc += np.where(k > 0, (k_m / k) ** 2, 0.0)
    return np.where(k > 0, 1.0 - pc, 0.0)


def compute_for_subject(A, labels, modules=None):
    if not np.allclose(A, A.T, atol=1e-8):
        A = (A + A.T) / 2.0          # connectomes are built symmetric; enforce
    np.fill_diagonal(A, 0.0)
    A_norm, max_eig = normalize_by_max_eig(A)
    rows = []
    wd = A.sum(axis=1)
    ec = eigenvector_centrality(A_norm)
    cm = communicability_strength(A_norm)
    pc = participation_coefficient(A, modules) if modules is not None else np.full(A.shape[0], np.nan)
    for i, lab in enumerate(labels):
        rows.append({"node_label": lab,
                     "weighted_degree": float(wd[i]),
                     "eigenvector_centrality": float(ec[i]),
                     "communicability": float(cm[i]),
                     "participation_coefficient": float(pc[i]),
                     "max_eigenvalue_used": max_eig})
    return rows


# ---------------------------------------------------------------- self-test --
def self_test():
    rng = np.random.default_rng(0)
    print("1. eigenvector centrality is highest for the most-connected node")
    A = np.array([[0.,1,1,1],[1,0,1,0],[1,1,0,0],[1,0,0,0]])
    ec = eigenvector_centrality(A)
    assert int(np.argmax(ec)) == 0, ec
    print("   ok")

    print("2. eigenvector centrality is non-negative (sign convention)")
    assert (ec >= -1e-12).all(), ec
    print("   ok")

    print("3. communicability finite after eigenvalue normalization on large weights")
    B = rng.random((30, 30)) * 1e5
    B = (B + B.T) / 2; np.fill_diagonal(B, 0)
    Bn, m = normalize_by_max_eig(B)
    cm = communicability_strength(Bn)
    assert np.all(np.isfinite(cm)) and m > 0, (m, cm[:3])
    print(f"   ok (max eig {m:.3g}, communicability finite)")

    print("4. participation coefficient is 0 for a node wired inside one module only")
    A2 = np.zeros((4, 4)); A2[0,1] = A2[1,0] = 1.0; A2[2,3] = A2[3,2] = 1.0
    pc = participation_coefficient(A2, np.array([0,0,1,1]))
    assert abs(pc[0]) < 1e-12, pc
    print("   ok")

    print("5. participation coefficient is high for a node split evenly across modules")
    A3 = np.zeros((5, 5))
    A3[0,1] = A3[1,0] = 1.0; A3[0,2] = A3[2,0] = 1.0
    A3[0,3] = A3[3,0] = 1.0; A3[0,4] = A3[4,0] = 1.0
    pc = participation_coefficient(A3, np.array([0,0,0,1,1]))
    assert pc[0] > 0.4, pc
    print(f"   ok (pc={pc[0]:.3f})")

    print("6. weighted degree here matches a direct row sum")
    rows = compute_for_subject(A.copy(), [f"n{i}" for i in range(4)], np.array([0,0,1,1]))
    assert abs(rows[0]["weighted_degree"] - A.sum(axis=1)[0]) < 1e-9
    print("   ok")
    print("\nSELF-TEST: ALL PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--connectome-dir")
    ap.add_argument("--atlas", choices=("schaefer200", "glasser"))
    ap.add_argument("--labels-file")
    ap.add_argument("--modules-file", default=None,
                    help="optional: one integer module id per node, same order as labels.")
    ap.add_argument("--modules-from-labels", action="store_true",
                    help="derive modules from the labels themselves (works for Schaefer-200, "
                         "whose labels embed the 7-network assignment). Overrides --modules-file.")
    ap.add_argument("--out-dir")
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()

    if a.self_test:
        self_test(); return
    for r in ("connectome_dir", "atlas", "labels_file", "out_dir"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required")

    cdir = Path(a.connectome_dir); odir = Path(a.out_dir); odir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(a.labels_file))
    modules = None
    if a.modules_from_labels:
        modules = modules_from_labels(labels)
        if modules is None:
            print("  WARNING: labels do not carry an inline network assignment "
                  "(expected for Glasser); participation coefficient will be NaN")
    elif a.modules_file:
        modules = np.loadtxt(a.modules_file, dtype=int)
        if len(modules) != len(labels):
            sys.exit(f"ERROR: {len(modules)} modules vs {len(labels)} labels")

    subjects = a.subjects or discover_subjects(cdir)
    if not subjects:
        sys.exit(f"No sub-<ID> directories found in {cdir}")

    out_rows, missing = [], []
    for s in subjects:
        p = find_subject_matrix(cdir, s, a.atlas)
        if p is None:
            missing.append(s); continue
        A = load_matrix(p)
        if A.shape[0] != len(labels):
            sys.exit(f"ERROR: {s} matrix is {A.shape[0]}x{A.shape[0]} but {len(labels)} labels supplied")
        for row in compute_for_subject(A, labels, modules):
            row["subject"] = s; out_rows.append(row)
        print(f"  {s}: ok")

    fields = ["subject","node_label","weighted_degree","eigenvector_centrality",
              "communicability","participation_coefficient","max_eigenvalue_used"]
    out = odir / f"all_subjects_graph_metrics_{a.atlas}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    print(f"\nWritten {len(out_rows)} rows for {len(subjects)-len(missing)} subjects to {out}")
    if missing:
        print(f"MISSING matrices for {len(missing)} subjects: {missing}")


if __name__ == "__main__":
    main()
