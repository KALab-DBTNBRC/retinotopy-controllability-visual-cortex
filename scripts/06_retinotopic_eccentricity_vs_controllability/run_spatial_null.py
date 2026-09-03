#!/usr/bin/env python3
"""
SERVER-SIDE SCRIPT -- needs node centroid coordinates derived from the
32k_fs_LR surfaces, which are not available in the sandbox.

WHY THIS EXISTS
---------------
Pre-submission comments from colleagues raised the most serious outstanding
methodological criticism of this project:

    "The authors' permutation procedure does not generate spatially
     constrained nulls and may produce anticonservative significance in
     spatially structured cortical variables."

That is a fair criticism. The primary permutation reshuffles each
participant's eccentricity values across that participant's own nodes. This
correctly respects the per-participant node set and the repeated-measures
structure, but it destroys the spatial autocorrelation of eccentricity.
Cortical maps are smooth; a randomly reshuffled map is not exchangeable with
the real one. The existing fixture calibration shows the code controls false
positives under a SYNTHETIC null; it does not establish that the null is
appropriate for real cortical spatial structure.

WHAT THIS DOES
--------------
Adds a SPATIALLY CONSTRAINED null as a secondary test. Two variants:

  spin   Random 3D rotation of node centroids on the sphere, with each node
         reassigned the value of its nearest rotated neighbour (Alexander-Bloch
         style, applied at node rather than vertex level). Preserves the
         spatial autocorrelation of the map because the map is rotated intact.
         Rotations are generated per hemisphere and applied jointly, so
         hemispheric structure is preserved rather than scrambled.

  moran  Moran spectral randomisation. Builds a spatial weight matrix from
         inter-node distances, takes the eigenvectors of the doubly-centred
         weight matrix, and generates surrogates with matched spatial
         autocorrelation by randomising the spectral coefficients.

The primary permutation remains the headline test. This is reported ALONGSIDE
it. If the eccentricity result survives a spatially constrained null, the
criticism is answered; if it does not, that is a finding and must be reported.

Usage:
    # step 1 (server): extract node centroids once per atlas
    python3 run_spatial_null.py --extract-centroids \
        --midthickness-left  /path/sub-XXXXXX/T1w/fsaverage_LR32k/sub_L.midthickness_MSMAll.32k_fs_LR.surf.gii \
        --midthickness-right /path/sub-XXXXXX/T1w/fsaverage_LR32k/sub_R.midthickness_MSMAll.32k_fs_LR.surf.gii \
        --dlabel /path/atlas.dlabel.nii \
        --out-centroids glasser_node_centroids.csv

    # step 2: run the spatially constrained null
    python3 run_spatial_null.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --centroids-csv glasser_node_centroids.csv \
        --atlas glasser --null spin \
        --out-csv spatial_null_glasser.csv

    python3 run_spatial_null.py --self-test
"""
import argparse, csv, math, re, sys
from collections import defaultdict

import numpy as np
from scipy import stats

GLASSER_VIS_AREAS = {"V1","V2","V3","V4","V3A","V3B","V6","V6A","V7","IPS1",
                     "FFC","PIT","V8","VMV1","VMV2","VMV3","VVC",
                     "FST","LO1","LO2","LO3","MST","MT","PH","V3CD","V4t"}
GLASSER_RE = re.compile(r"^([LR])_(.+)_ROI$")
SCHAEFER_RE = re.compile(r"^7Networks_(LH|RH)_")


def is_vis_glasser(l):
    m = GLASSER_RE.match(l); return bool(m) and m.group(2) in GLASSER_VIS_AREAS


def hemi_of(l, atlas):
    if atlas == "glasser":
        m = GLASSER_RE.match(l); return m.group(1) if m else None
    m = SCHAEFER_RE.match(l); return {"LH":"L","RH":"R"}.get(m.group(1)) if m else None


# ------------------------------------------------------ centroid extraction --
def extract_centroids(mid_l, mid_r, dlabel, out_csv):
    """Node centroid = mean 3D coordinate of that node's vertices on the
    midthickness surface. Requires nibabel; run on the server."""
    import nibabel as nib
    lab = nib.load(dlabel)
    data = np.asarray(lab.get_fdata()).squeeze().astype(int)
    axes = [lab.header.get_axis(i) for i in range(lab.ndim)]
    lab_axis = next(a for a in axes if isinstance(a, nib.cifti2.cifti2_axes.LabelAxis))
    bm_axis  = next(a for a in axes if isinstance(a, nib.cifti2.cifti2_axes.BrainModelAxis))
    names = {k: v[0] for k, v in lab_axis.label[0].items()}
    coords = {}
    for surf, struct in ((mid_l, "CIFTI_STRUCTURE_CORTEX_LEFT"),
                         (mid_r, "CIFTI_STRUCTURE_CORTEX_RIGHT")):
        g = nib.load(surf)
        xyz = g.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
        for name, sl, model in bm_axis.iter_structures():
            if name != struct: continue
            verts = model.vertex
            vals = data[sl]
            for key in np.unique(vals):
                if key == 0: continue
                sel = verts[vals == key]
                sel = sel[sel < xyz.shape[0]]
                if sel.size == 0: continue
                coords.setdefault(names.get(key, str(key)), []).append(xyz[sel])
    rows = []
    for lbl, chunks in coords.items():
        pts = np.vstack(chunks)
        c = pts.mean(axis=0)
        rows.append({"node_label": lbl, "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
                     "n_vertices": int(pts.shape[0])})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["node_label","x","y","z","n_vertices"])
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["node_label"]))
    print(f"Wrote {len(rows)} node centroids to {out_csv}")


def load_centroids(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["node_label"]] = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
    return out


# ------------------------------------------------------------------ nulls ---
def random_rotation(rng):
    """Uniform random 3D rotation via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def spin_indices(coords, hemis, rng):
    """One spin: rotate each hemisphere's centroids and map every node to its
    nearest rotated neighbour WITHIN the same hemisphere."""
    idx = np.arange(len(coords))
    out = idx.copy()
    # labels that do not parse to a hemisphere are grouped together rather than
    # crashing; this keeps the spin valid if an atlas label is ever unexpected
    hemis = np.array(["?" if h is None else h for h in hemis], dtype=object)
    for h in np.unique(hemis.astype(str)):
        sel = np.where(hemis.astype(str) == h)[0]
        if sel.size < 2:
            continue
        c = coords[sel]
        c = c - c.mean(axis=0)
        rc = c @ random_rotation(rng).T
        d = ((rc[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        out[sel] = sel[np.argmin(d, axis=1)]
    return out


def moran_surrogates(values, coords, n, rng):
    """Moran spectral randomisation: surrogates with matched spatial
    autocorrelation, built from eigenvectors of the centred weight matrix."""
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    with np.errstate(divide="ignore"):
        W = 1.0 / d
    W[~np.isfinite(W)] = 0.0
    np.fill_diagonal(W, 0.0)
    k = W.shape[0]
    C = np.eye(k) - np.ones((k, k)) / k
    M = C @ W @ C
    M = (M + M.T) / 2
    _, V = np.linalg.eigh(M)
    v = values - values.mean()
    coef = V.T @ v
    out = np.empty((n, k))
    for i in range(n):
        s = rng.choice([-1.0, 1.0], size=k)
        surr = V @ (coef * s)
        sd = surr.std()
        out[i] = values.mean() + (surr / sd * values.std() if sd > 0 else surr)
    return out


def partial_corr(x, y, z):
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0: return np.nan
    rxy, rxz, ryz = stats.pearsonr(x,y)[0], stats.pearsonr(x,z)[0], stats.pearsonr(y,z)[0]
    den = math.sqrt((1-rxz**2)*(1-ryz**2))
    return np.nan if den == 0 or not np.isfinite(den) else (rxy-rxz*ryz)/den


def fz(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))


def load_prf(path, minv, atlas):
    ecc, size = defaultdict(dict), {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["atlas"] != atlas: continue
            vis = (r["is_vis_network"]=="True") if atlas=="schaefer200" else is_vis_glasser(r["backbone_label"])
            if not vis: continue
            size[r["backbone_label"]] = int(r["n_vertices_total"])
            if r["mean_eccentricity_deg"]=="" : continue
            if int(r["n_vertices_valid_R2gt30"]) < minv: continue
            ecc[r["subject"]][r["backbone_label"]] = float(r["mean_eccentricity_deg"])
    return ecc, size


def load_ctrl(path):
    avg, mod = defaultdict(dict), defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            a = float(r["average_controllability"])
            mod[r["subject"]][r["node_label"]] = float(r["modal_controllability"])
            if a > 0: avg[r["subject"]][r["node_label"]] = math.log(a)
    return avg, mod


def run_null(subj_ecc, subj_metric, node_size, centroids, atlas, kind,
             min_nodes, n_perm, rng):
    per_r, per_null = [], []
    for s, e in subj_ecc.items():
        m = subj_metric.get(s, {})
        common = sorted(set(e) & set(m) & set(node_size) & set(centroids))
        if len(common) < min_nodes: continue
        x = np.array([e[n] for n in common])
        y = np.array([m[n] for n in common])
        z = np.array([node_size[n] for n in common], float)
        r = partial_corr(x, y, z)
        if not np.isfinite(r): continue
        per_r.append(r)
        C = np.vstack([centroids[n] for n in common])
        H = np.array([hemi_of(n, atlas) for n in common])
        if kind == "spin":
            nulls = [partial_corr(x[spin_indices(C, H, rng)], y, z) for _ in range(n_perm)]
        else:
            nulls = [partial_corr(sx, y, z) for sx in moran_surrogates(x, C, n_perm, rng)]
        per_null.append(nulls)
    if not per_r: return None
    obs = fz(per_r).mean()
    null = np.nanmean(fz(np.array(per_null, float)), axis=0)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"n_subjects": len(per_r), "mean_partial_r": float(np.tanh(obs)),
            "perm_p_value": float(p), "null_kind": kind}


def self_test():
    rng = np.random.default_rng(0)
    print("1. random_rotation is orthonormal with det +1")
    R = random_rotation(rng)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10) and abs(np.linalg.det(R)-1) < 1e-10
    print("   ok")

    print("2. spin_indices returns a valid within-hemisphere reassignment")
    C = rng.normal(size=(20,3)); H = np.array(["L"]*10 + ["R"]*10)
    idx = spin_indices(C, H, rng)
    assert idx.shape == (20,) and all(H[idx[i]] == H[i] for i in range(20))
    print("   ok (hemisphere preserved)")

    print("3. moran surrogates preserve mean and sd")
    v = rng.normal(size=25); C2 = rng.normal(size=(25,3))
    S = moran_surrogates(v, C2, 50, rng)
    assert abs(S.mean() - v.mean()) < 0.15 and abs(S.std() - v.std()) < 0.3
    print("   ok")

    print("4. spin null recovers a planted signal")
    nodes = [f"L_a{i}_ROI" for i in range(10)] + [f"R_a{i}_ROI" for i in range(10)]
    size = {n: 100+i for i, n in enumerate(nodes)}
    cent = {n: rng.normal(size=3) for n in nodes}
    ecc, met = {}, {}
    for s in map(str, range(25)):
        e = rng.normal(0,1,20)
        ecc[s] = dict(zip(nodes, e))
        met[s] = dict(zip(nodes, 0.9*e + rng.normal(0,0.3,20)))
    res = run_null(ecc, met, size, cent, "glasser", "spin", 15, 200, rng)
    assert res["perm_p_value"] < 0.05 and res["mean_partial_r"] > 0.5, res
    print(f"   ok (r={res['mean_partial_r']:.3f}, p={res['perm_p_value']:.4f})")

    print("5. spin null is null for an unrelated metric")
    met2 = {s: dict(zip(nodes, rng.normal(0,1,20))) for s in map(str, range(25))}
    res2 = run_null(ecc, met2, size, cent, "glasser", "spin", 15, 200, rng)
    assert res2["perm_p_value"] > 0.05, res2
    print(f"   ok (r={res2['mean_partial_r']:+.3f}, p={res2['perm_p_value']:.3f})")
    print("\nSELF-TEST: ALL PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--extract-centroids", action="store_true")
    ap.add_argument("--midthickness-left"); ap.add_argument("--midthickness-right")
    ap.add_argument("--dlabel"); ap.add_argument("--out-centroids")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--centroids-csv"); ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200","glasser"))
    ap.add_argument("--null", default="spin", choices=("spin","moran","both"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test: self_test(); return
    if a.extract_centroids:
        for r in ("midthickness_left","midthickness_right","dlabel","out_centroids"):
            if not getattr(a,r): ap.error(f"--{r.replace('_','-')} is required")
        extract_centroids(a.midthickness_left, a.midthickness_right, a.dlabel, a.out_centroids); return
    for r in ("prf_backbone_csv","controllability_csv","centroids_csv","out_csv"):
        if not getattr(a,r): ap.error(f"--{r.replace('_','-')} is required")

    rng = np.random.default_rng(a.seed)
    ecc, size = load_prf(a.prf_backbone_csv, a.min_vertices_per_node, a.atlas)
    avg, mod = load_ctrl(a.controllability_csv)
    cent = load_centroids(a.centroids_csv)
    overlap = set(size) & set(cent)
    if not overlap:
        sys.exit("ERROR: no node-label overlap between pRF table and centroids file.")
    print(f"{len(overlap)} nodes have both pRF and centroid data")

    kinds = ["spin","moran"] if a.null=="both" else [a.null]
    rows=[]
    for kind in kinds:
        for name, metric in (("average_controllability_log",avg),("modal_controllability",mod)):
            res = run_null(ecc, metric, size, cent, a.atlas, kind, a.min_valid_nodes,
                           a.n_permutations, rng)
            if res is None:
                rows.append({"metric":name,"null_kind":kind,"note":"no valid subjects"}); continue
            res["metric"]=name; res["atlas"]=a.atlas; res["note"]=""
            rows.append(res)
            print(f"  [{kind:5s}] {name:30s} n={res['n_subjects']:3d} "
                  f"r={res['mean_partial_r']:+.3f} p={res['perm_p_value']:.4f}")

    fields=["metric","null_kind","atlas","n_subjects","mean_partial_r","perm_p_value","note"]
    with open(a.out_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})
    print(f"\nWritten to {a.out_csv}")


if __name__ == "__main__":
    main()
