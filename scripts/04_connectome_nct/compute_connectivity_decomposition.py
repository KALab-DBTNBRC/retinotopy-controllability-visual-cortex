#!/usr/bin/env python3
"""
Decompose each visual node's structural connectivity into within-visual and
extra-visual components.

WHY THIS EXISTS
---------------
Controllability here is computed on the WHOLE-BRAIN network. A visual node's
average controllability therefore depends on all of its connections, both to
other visual nodes and to the rest of the brain. The manuscript reports that
eccentricity is associated with average controllability, and then bridges that
to Amorosino et al. (2026), who showed that structural connectivity WITHIN
visual cortex preferentially links regions representing the same visual-field
location. That bridge is currently asserted, not tested.

This script supplies the missing decomposition. For each visual node it splits
total weighted degree into:

    within_visual_strength  = sum of weights to other visual nodes
    extra_visual_strength   = sum of weights to every non-visual node
    total_strength          = the two added together (standard weighted degree)
    within_visual_fraction  = within / total

The companion analysis script then asks which component the retinotopic
controllability gradient actually runs through. Three outcomes are
distinguishable:

  (a) the gradient survives adjustment for extra-visual strength but not
      within-visual strength  -> it is a within-visual wiring phenomenon, and
      the Amorosino bridge is supported;
  (b) the reverse -> the gradient reflects how visual cortex couples to the
      rest of the brain, and the Amorosino bridge is the wrong explanation;
  (c) it survives both -> neither component alone accounts for it.

The script does NOT decide between these. It only produces the decomposition.

SERVER-SIDE. Needs the per-subject connectivity matrices, which are not
distributed with the repository.

USAGE
-----
    python3 compute_connectivity_decomposition.py \
        --connectome-dir <DIR> \
        --atlas schaefer200 \
        --labels-file <LABELS.txt> \
        --visual-from-labels \
        --out-dir <OUT>

    # Glasser has no network label embedded in its parcel names, so its
    # visual node set must be supplied explicitly, one label per line:
    python3 compute_connectivity_decomposition.py \
        --connectome-dir <DIR> --atlas glasser \
        --labels-file <LABELS.txt> \
        --visual-nodes-file <GLASSER_VISUAL_52.txt> \
        --out-dir <OUT>

    python3 compute_connectivity_decomposition.py --self-test
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

SCHAEFER_VIS_RE = re.compile(r"^7Networks_(?:LH|RH)_Vis_")

# Glasser MMP1.0 visual node set: the union of all five visual-system divisions
# (Primary Visual, Early Visual, Dorsal Stream, Ventral Stream, MT+ Complex),
# 26 areas per hemisphere, 52 bilateral nodes. Copied verbatim from
# run_claim2_analysis.py so this script and the published analysis cannot drift
# apart, and verified against the 52 nodes in the released per-node output.
GLASSER_VIS_AREAS = {
    "V1",                                                            # Primary Visual
    "V2", "V3", "V4",                                                # Early Visual
    "V3A", "V3B", "V6", "V6A", "V7", "IPS1",                         # Dorsal Stream
    "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",               # Ventral Stream
    "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t",    # MT+ Complex
}
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def glasser_visual_mask(labels):
    """Visual mask for Glasser MMP1.0 from the built-in area list."""
    out = []
    for l in labels:
        m = GLASSER_LABEL_RE.match(l.strip())
        out.append(bool(m) and m.group(1) in GLASSER_VIS_AREAS)
    m = np.array(out)
    return m if m.any() else None


def read_label_list(path):
    """Read node labels from a plain list OR the first column of a CSV.

    Tolerates a header row and extra columns, because the obvious file to
    reach for on the server is a centroids CSV
    (label,x,y,z,n_vertices) rather than a bare label list. Reading such a
    file line-by-line yields entries like
    'L_10d_ROI,-10.7,67.4,9.0,153', which match nothing.
    """
    rows = [l.rstrip("\n").rstrip("\r") for l in
            open(path, encoding="utf-8") if l.strip()]
    if not rows:
        return []
    first_col = [r.split(",")[0].strip().strip('"') for r in rows]
    # drop an obvious header
    if first_col and not GLASSER_LABEL_RE.match(first_col[0]) \
            and not first_col[0].startswith("7Networks"):
        head = first_col[0].lower()
        if head in ("label", "node_label", "name", "roi", "region", "backbone_label"):
            first_col = first_col[1:]
    return first_col


# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------

def load_matrix(path):
    """Read a square connectivity matrix, sniffing the delimiter.

    The connectomes in this project are comma-separated .csv with no header.
    An earlier version passed delimiter=None unconditionally, which makes
    np.loadtxt split on WHITESPACE, so every comma-separated row was read as a
    single unparseable token and all 46 subjects were skipped.

    compute_graph_metrics.py keys the delimiter off the .csv suffix. This goes
    one step further and sniffs the first data line, so a .txt file that
    happens to be comma-separated (or vice versa) still loads.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first = ""
        for line in fh:
            if line.strip():
                first = line
                break
    if "," in first:
        delim = ","
    elif "\t" in first:
        delim = "\t"
    else:
        delim = None          # any run of whitespace

    A = np.loadtxt(path, delimiter=delim)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {A.shape}")
    return A


def symmetrize(A):
    """Connectomes are undirected; tolerate upper- or lower-triangular storage."""
    if np.allclose(A, A.T, equal_nan=True):
        return A
    return np.maximum(A, A.T)


def visual_mask_from_labels(labels):
    """Schaefer only: the 7Networks parcel names carry the network."""
    m = np.array([bool(SCHAEFER_VIS_RE.match(l)) for l in labels])
    return m if m.any() else None


def decompose(A, vis_mask):
    """Split each visual node's weighted degree into within- and extra-visual.

    Self-connections are excluded. Returns arrays indexed over ALL nodes;
    entries for non-visual nodes are meaningless and are filtered by the
    caller, but are computed anyway so index alignment is never in doubt.
    """
    A = symmetrize(np.asarray(A, dtype=float)).copy()
    np.fill_diagonal(A, 0.0)
    if np.any(A < 0):
        raise ValueError("negative connectivity weights are not supported")

    total = A.sum(axis=1)
    within = A[:, vis_mask].sum(axis=1)
    # a visual node's own contribution to the within sum is zero (diagonal
    # cleared), so no self-subtraction is needed
    extra = total - within

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, within / total, np.nan)

    return total, within, extra, frac


def find_subject_matrix(connectome_dir, subject, atlas):
    """Locate one subject's connectivity matrix.

    The canonical layout, and the one compute_graph_metrics.py already relies
    on, is:

        <connectome_dir>/sub-<subject>/connectome_<atlas>.csv

    That is tried first. An earlier version of this function used
    rglob(f"*{subject}*{atlas}*"), which matches against the FILENAME only --
    and in this layout the subject lives in the directory name while the
    filename carries only the atlas, so nothing ever matched and every subject
    was silently skipped. The fallbacks below therefore match against the full
    path, not the basename.
    """
    root = Path(connectome_dir)

    # 1. canonical layout
    p = root / f"sub-{subject}" / f"connectome_{atlas}.csv"
    if p.exists():
        return p

    # 2. same, without the BIDS "sub-" prefix
    p = root / str(subject) / f"connectome_{atlas}.csv"
    if p.exists():
        return p

    # 3. full-path search: subject anywhere in the path, atlas anywhere,
    #    tabular extension. Deterministic pick, shortest path first.
    cands = [c for c in root.rglob("*")
             if c.is_file()
             and c.suffix.lower() in (".csv", ".txt", ".tsv")
             and str(subject) in str(c)
             and atlas in str(c)]
    if not cands:
        return None
    return sorted(cands, key=lambda c: (len(str(c)), str(c)))[0]


def discover_subjects(connectome_dir):
    """Subject IDs from sub-* directory names, matching the existing pipeline."""
    root = Path(connectome_dir)
    subs = sorted(p.name.replace("sub-", "") for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("sub-"))
    if subs:
        return subs
    # fallback: any six-digit token in a directory or file name
    return sorted({m.group(0) for p in root.rglob("*")
                   for m in [re.search(r"\d{6}", p.name)] if m})


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    # ---- 1. hand-computable case -------------------------------------------
    # 4 nodes, first two visual.
    #   A = 0 1 2 3
    #       1 0 4 5
    #       2 4 0 6
    #       3 5 6 0
    A = np.array([[0., 1., 2., 3.],
                  [1., 0., 4., 5.],
                  [2., 4., 0., 6.],
                  [3., 5., 6., 0.]])
    vis = np.array([True, True, False, False])
    tot, wit, ext, frac = decompose(A, vis)
    # node0: total 1+2+3=6, within (to node1) = 1, extra = 5
    # node1: total 1+4+5=10, within = 1, extra = 9
    check("node 0 total = 6", np.isclose(tot[0], 6))
    check("node 0 within = 1", np.isclose(wit[0], 1))
    check("node 0 extra = 5", np.isclose(ext[0], 5))
    check("node 1 total = 10", np.isclose(tot[1], 10))
    check("node 1 within = 1", np.isclose(wit[1], 1))
    check("within + extra == total (all nodes)", np.allclose(wit + ext, tot))
    check("node 0 fraction = 1/6", np.isclose(frac[0], 1 / 6))

    # ---- 2. total strength must match plain weighted degree ----------------
    rng = np.random.default_rng(0)
    B = rng.random((20, 20)) * 3
    B = (B + B.T) / 2
    np.fill_diagonal(B, 0)
    vm = np.zeros(20, dtype=bool); vm[:7] = True
    tot2, wit2, ext2, _ = decompose(B, vm)
    deg = B.sum(axis=1)
    check("total strength reproduces weighted degree", np.allclose(tot2, deg))
    check("decomposition is exact (no leakage)", np.allclose(wit2 + ext2, deg))

    # ---- 3. all-visual network: extra-visual must be exactly zero ----------
    allvis = np.ones(20, dtype=bool)
    t3, w3, e3, f3 = decompose(B, allvis)
    check("all-visual: extra == 0", np.allclose(e3, 0))
    check("all-visual: fraction == 1", np.allclose(f3, 1))

    # ---- 4. no-visual-neighbours case: within == 0 -------------------------
    onevis = np.zeros(20, dtype=bool); onevis[3] = True
    t4, w4, e4, f4 = decompose(B, onevis)
    check("single visual node has zero within-visual strength",
          np.isclose(w4[3], 0))
    check("single visual node: extra == total", np.isclose(e4[3], t4[3]))

    # ---- 5. NEGATIVE CONTROL ------------------------------------------------
    # A node with high total strength does not have to have a high within-
    # visual fraction. If the decomposition were secretly just rescaling
    # degree, fraction and total would be perfectly correlated. Build a case
    # where they are deliberately anti-correlated and confirm we recover that.
    n = 12
    C = np.zeros((n, n))
    vis5 = np.zeros(n, dtype=bool); vis5[:6] = True
    # visual nodes 0-2: strongly wired to each other, weakly to the rest
    for i in range(3):
        for j in range(3):
            if i != j:
                C[i, j] = 10.0
        for j in range(6, n):
            C[i, j] = C[j, i] = 0.1
    # visual nodes 3-5: weakly wired to each other, strongly outward
    for i in range(3, 6):
        for j in range(3, 6):
            if i != j:
                C[i, j] = 0.1
        for j in range(6, n):
            C[i, j] = C[j, i] = 10.0
    t5, w5, e5, f5 = decompose(C, vis5)
    lo_tot, hi_tot = t5[0], t5[3]
    lo_frac, hi_frac = f5[0], f5[3]
    check("negative control: outward-wired node has HIGHER total strength",
          hi_tot > lo_tot)
    check("negative control: outward-wired node has LOWER within fraction",
          hi_frac < lo_frac)
    check("negative control: fraction is not a monotone function of total",
          not np.isclose(np.corrcoef(t5[vis5], f5[vis5])[0, 1], 1.0))

    # ---- 6. asymmetric storage is handled ----------------------------------
    U = np.triu(B)
    t6, w6, e6, _ = decompose(U, vm)
    check("upper-triangular input symmetrized correctly", np.allclose(t6, tot2))

    # ---- 7. Schaefer label parsing ------------------------------------------
    labs = ["7Networks_LH_Vis_1", "7Networks_LH_Vis_2",
            "7Networks_RH_Cont_5", "7Networks_LH_Default_3"]
    m7 = visual_mask_from_labels(labs)
    check("label parser finds exactly the two Vis parcels",
          m7 is not None and m7.tolist() == [True, True, False, False])
    check("label parser returns None when no Vis parcels exist",
          visual_mask_from_labels(["a", "b"]) is None)

    # ---- 8. path resolver, against the real server layout -------------------
    # This is the failure that skipped all 46 subjects in the first server run:
    # the subject ID lives in the DIRECTORY name and the filename carries only
    # the atlas, so a filename-only glob could never match.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for sid in ("123456", "234567"):
            d = root / f"sub-{sid}"
            d.mkdir()
            # comma-separated, matching the real server files
            (d / "connectome_schaefer200.csv").write_text("0,1.5\n1.5,0\n")
            (d / "connectome_glasser.csv").write_text("0,2\n2,0\n")
        got = find_subject_matrix(root, "123456", "schaefer200")
        check("resolver finds sub-<ID>/connectome_<atlas>.csv",
              got is not None and got.name == "connectome_schaefer200.csv"
              and "sub-123456" in str(got))
        check("resolver keeps the two atlases apart",
              find_subject_matrix(root, "123456", "glasser").name
              == "connectome_glasser.csv")
        check("resolver returns None for an absent subject",
              find_subject_matrix(root, "345678", "schaefer200") is None)
        check("discover_subjects reads sub-* directory names",
              discover_subjects(root) == ["123456", "234567"])

        # layout without the BIDS prefix
        root2 = root / "flat"
        (root2 / "456789").mkdir(parents=True)
        (root2 / "456789" / "connectome_glasser.csv").write_text("0,1\n1,0\n")
        check("resolver handles a missing sub- prefix",
              find_subject_matrix(root2, "456789", "glasser") is not None)

        # ---- delimiter sniffing: the second server failure ------------------
        # Comma-separated .csv is what the pipeline actually writes. Reading it
        # with a whitespace delimiter skipped every subject.
        M = load_matrix(root / "sub-123456" / "connectome_schaefer200.csv")
        check("comma-separated .csv loads correctly",
              M.shape == (2, 2) and np.isclose(M[0, 1], 1.5))

        alt = root / "alt"; alt.mkdir()
        (alt / "ws.txt").write_text("0 3\n3 0\n")
        (alt / "tab.tsv").write_text("0\t4\n4\t0\n")
        (alt / "comma.txt").write_text("0,5\n5,0\n")
        check("whitespace-separated file loads",
              np.isclose(load_matrix(alt / "ws.txt")[0, 1], 3))
        check("tab-separated file loads",
              np.isclose(load_matrix(alt / "tab.tsv")[0, 1], 4))
        check("comma-separated file with a .txt suffix still loads",
              np.isclose(load_matrix(alt / "comma.txt")[0, 1], 5))

        # ---- Glasser built-in visual set, and tolerant label parsing --------
        # The obvious file to reach for on the server is a centroids CSV, which
        # lists ALL 360 nodes with a header and four extra columns. Reading it
        # naively made every "label" look like
        # 'L_10d_ROI,-10.7,67.4,9.0,153' and matched nothing.
        gl = alt / "glasser_labels.txt"
        all360 = [f"{h}_{a}_ROI" for a in ("V1", "V2", "MT", "10d", "46", "A1")
                  for h in ("L", "R")]
        gl.write_text("\n".join(all360) + "\n")
        gm = glasser_visual_mask(read_label_list(gl))
        check("built-in Glasser mask selects only visual areas",
              gm is not None and int(gm.sum()) == 6)   # V1,V2,MT bilateral
        check("built-in Glasser mask rejects non-visual areas",
              not any(gm[i] for i, l in enumerate(all360)
                      if l.split("_")[1] in ("10d", "46", "A1")))
        check("GLASSER_VIS_AREAS has 26 areas, i.e. 52 bilateral nodes",
              len(GLASSER_VIS_AREAS) == 26)

        cent = alt / "centroids.csv"
        cent.write_text("label,x,y,z,n_vertices\n"
                        "L_V1_ROI,-10.7,67.4,9.0,153\n"
                        "R_V1_ROI,10.7,67.4,9.0,151\n")
        check("label reader takes column 1 of a CSV and drops the header",
              read_label_list(cent) == ["L_V1_ROI", "R_V1_ROI"])
        check("label reader leaves a plain one-per-line list unchanged",
              read_label_list(gl) == all360)

        # a non-square matrix must be rejected, not silently used
        (alt / "bad.csv").write_text("0,1,2\n1,0,3\n")
        try:
            load_matrix(alt / "bad.csv")
            check("non-square matrix is rejected", False)
        except ValueError:
            check("non-square matrix is rejected", True)

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--connectome-dir")
    ap.add_argument("--atlas", choices=("schaefer200", "glasser"))
    ap.add_argument("--labels-file",
                    help="one node label per line, in matrix row order")
    ap.add_argument("--visual-from-labels", action="store_true",
                    help="Schaefer only: derive the visual set from 7Networks_*_Vis_* names")
    ap.add_argument("--visual-nodes-file",
                    help="explicit visual node labels. Accepts a plain one-per-line "
                         "list or the first column of a CSV with a header.")
    ap.add_argument("--visual-glasser-builtin", action="store_true",
                    help="Glasser only: use the built-in 52-node visual set "
                         "(the same definition as run_claim2_analysis.py). "
                         "No external file needed.")
    ap.add_argument("--out-dir")
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    for req in ("connectome_dir", "atlas", "labels_file", "out_dir"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_','-')} is required unless --self-test")

    labels = read_label_list(a.labels_file)

    if a.visual_glasser_builtin:
        vis_mask = glasser_visual_mask(labels)
        if vis_mask is None:
            sys.exit("--visual-glasser-builtin matched no L_*_ROI / R_*_ROI labels. "
                     "Is --labels-file really the Glasser label list?")
    elif a.visual_from_labels:
        vis_mask = visual_mask_from_labels(labels)
        if vis_mask is None:
            sys.exit("--visual-from-labels found no 7Networks_*_Vis_* parcels. "
                     "Glasser labels carry no network field; use --visual-nodes-file.")
    elif a.visual_nodes_file:
        want = set(read_label_list(a.visual_nodes_file))
        vis_mask = np.array([l in want for l in labels])
        missing = want - set(labels)
        if missing:
            frac = len(missing) / max(len(want), 1)
            msg = [f"{len(missing)} of {len(want)} visual labels not found in "
                   f"--labels-file, e.g. {sorted(missing)[:3]}"]
            if frac > 0.5:
                msg.append("More than half are missing, so this is probably the "
                           "wrong file. A centroids CSV lists ALL nodes, not just "
                           "the visual ones. For Glasser, prefer "
                           "--visual-glasser-builtin.")
            sys.exit("\n".join(msg))
    else:
        ap.error("supply --visual-from-labels, --visual-glasser-builtin, "
                 "or --visual-nodes-file")

    print(f"atlas={a.atlas}  nodes={len(labels)}  visual nodes={int(vis_mask.sum())}")

    cdir = Path(a.connectome_dir)
    if a.subjects:
        subjects = list(a.subjects)
    else:
        subjects = discover_subjects(cdir)
    if not subjects:
        sys.exit(f"no subjects found under {cdir}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f"all_subjects_connectivity_decomposition_{a.atlas}.csv"

    n_ok = n_skip = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "node_label", "total_strength",
                    "within_visual_strength", "extra_visual_strength",
                    "within_visual_fraction"])
        for s in subjects:
            mp = find_subject_matrix(cdir, s, a.atlas)
            if mp is None:
                print(f"  skip {s}: no matrix found"); n_skip += 1; continue
            try:
                A = load_matrix(mp)
            except Exception as e:
                print(f"  skip {s}: {e}"); n_skip += 1; continue
            if A.shape[0] != len(labels):
                print(f"  skip {s}: matrix is {A.shape[0]}x{A.shape[0]} but "
                      f"{len(labels)} labels supplied"); n_skip += 1; continue
            tot, wit, ext, frac = decompose(A, vis_mask)
            for i, lab in enumerate(labels):
                if not vis_mask[i]:
                    continue
                w.writerow([s, lab, f"{tot[i]:.10g}", f"{wit[i]:.10g}",
                            f"{ext[i]:.10g}", f"{frac[i]:.10g}"])
            n_ok += 1

    print(f"\nwrote {out_csv}")
    print(f"subjects processed: {n_ok}, skipped: {n_skip}")
    if n_ok == 0:
        print("\nERROR: no subject matrices were read, so the output file is "
              "empty and must not be used.", file=sys.stderr)
        print(f"Looked for: {cdir}/sub-<ID>/connectome_{a.atlas}.csv",
              file=sys.stderr)
        ex = next((p for p in cdir.rglob(f"*{a.atlas}*") if p.is_file()), None)
        if ex:
            print(f"An example matching file that DOES exist: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
