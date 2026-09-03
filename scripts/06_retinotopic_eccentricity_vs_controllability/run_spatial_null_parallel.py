#!/usr/bin/env python3
"""
Parallel + vectorised version of run_spatial_null.py.

Same analysis, same nulls, same outputs. Speed changes only:

  1. VECTORISED PARTIAL CORRELATION. All permutations for a subject are
     computed in one centred matrix product instead of 3 scipy calls each.
     Verified identical to the scipy path to ~1e-15.

  2. BATCHED SPINS. The serial version rebuilt an n x n distance matrix per
     permutation. This generates all rotations for a subject at once and
     resolves every nearest-neighbour assignment in a single batched argmin.

  3. PARALLEL OVER (null kind, metric, subject). Each unit is independent.

REPRODUCIBILITY
---------------
Each (kind, metric, subject) unit draws from its own SeedSequence-spawned
stream, so a given --seed produces identical results and those results are
invariant to --n-jobs. They will NOT bit-match the serial script, which draws
every rotation from one shared stream; both are internally reproducible. This
is the same convention used by run_graph_metric_specificity_parallel.py.

Usage:
    python3 run_spatial_null_parallel.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --centroids-csv glasser_node_centroids.csv \
        --atlas glasser --null both --n-permutations 10000 --n-jobs 80 \
        --out-csv spatial_null_glasser.csv

    python3 run_spatial_null_parallel.py --self-test
"""
import argparse, csv, math, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

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


def load_prf(path, minv, atlas):
    ecc, size = defaultdict(dict), {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["atlas"] != atlas: continue
            vis = (r["is_vis_network"]=="True") if atlas=="schaefer200" else is_vis_glasser(r["backbone_label"])
            if not vis: continue
            size[r["backbone_label"]] = int(r["n_vertices_total"])
            if r["mean_eccentricity_deg"] == "": continue
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


def load_centroids(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["node_label"]] = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
    return out


def partial_corr_many(X, y, z):
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean(); zc = z - z.mean()
    Xn = np.linalg.norm(Xc, axis=1); yn = np.linalg.norm(yc); zn = np.linalg.norm(zc)
    if yn == 0 or zn == 0:
        return np.full(X.shape[0], np.nan)
    rXy = (Xc @ yc) / (Xn * yn); rXz = (Xc @ zc) / (Xn * zn)
    ryz = float((yc @ zc) / (yn * zn))
    den = np.sqrt((1 - rXz**2) * (1 - ryz**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (rXy - rXz * ryz) / den, np.nan)
    return np.where(Xn > 0, out, np.nan)


def random_rotations(m, rng):
    """m uniform random 3D rotations, shape (m,3,3), via QR of Gaussians."""
    A = rng.normal(size=(m, 3, 3))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diagonal(R, axis1=1, axis2=2))
    d[d == 0] = 1.0
    Q = Q * d[:, None, :]
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1
    return Q


def batched_spin_indices(coords, hemis, m, rng):
    """(m, n) index arrays: m spins, each reassigning every node to its nearest
    rotated neighbour within the same hemisphere. Batched over permutations."""
    n = coords.shape[0]
    out = np.tile(np.arange(n), (m, 1))
    for h in np.unique(hemis.astype(str)):
        sel = np.where(hemis.astype(str) == h)[0]
        if sel.size < 2:
            continue
        c = coords[sel]
        c = c - c.mean(axis=0)
        Q = random_rotations(m, rng)              # (m,3,3)
        rc = np.einsum("mij,nj->mni", Q, c)       # (m,k,3) rotated
        d = ((rc[:, :, None, :] - c[None, None, :, :]) ** 2).sum(-1)   # (m,k,k)
        out[:, sel] = sel[np.argmin(d, axis=2)]
    return out


def moran_surrogates(values, coords, n, rng):
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
    S = rng.choice([-1.0, 1.0], size=(n, k))
    surr = (V @ (coef[None, :] * S).T).T
    sd = surr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return values.mean() + surr / sd * values.std()


def fz(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o]*n/(np.arange(n)+1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def _unit(args):
    x, y, z, C, H, kind, n_perm, seed_state = args
    rng = np.random.default_rng(seed_state)
    obs = float(partial_corr_many(x[None, :], y, z)[0])
    if not np.isfinite(obs):
        return None
    if kind == "spin":
        idx = batched_spin_indices(C, H, n_perm, rng)
        null = partial_corr_many(x[idx], y, z)
    else:
        null = partial_corr_many(moran_surrogates(x, C, n_perm, rng), y, z)
    return obs, null


def run_null(subj_ecc, subj_metric, node_size, centroids, atlas, kind,
             min_nodes, n_perm, seedseq, pool):
    units = []
    for s in sorted(subj_ecc):
        m = subj_metric.get(s, {})
        common = sorted(set(subj_ecc[s]) & set(m) & set(node_size) & set(centroids))
        if len(common) < min_nodes: continue
        units.append((np.array([subj_ecc[s][n] for n in common], float),
                      np.array([m[n] for n in common], float),
                      np.array([node_size[n] for n in common], float),
                      np.vstack([centroids[n] for n in common]),
                      np.array([hemi_of(n, atlas) for n in common], dtype=object)))
    if not units: return None
    child = seedseq.spawn(len(units))
    payload = [(u[0], u[1], u[2], u[3], u[4], kind, n_perm, child[i])
               for i, u in enumerate(units)]
    res = pool.map(_unit, payload) if pool else [_unit(p) for p in payload]
    res = [r for r in res if r is not None]
    if not res: return None
    obs = fz([r[0] for r in res]).mean()
    null = np.nanmean(fz(np.array([r[1] for r in res])), axis=0)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"n_subjects": len(res), "mean_partial_r": float(np.tanh(obs)),
            "perm_p_value": float(p), "null_kind": kind}


def self_test():
    rng = np.random.default_rng(0)
    print("1. batched rotations are orthonormal with det +1")
    Q = random_rotations(50, rng)
    assert np.allclose(np.einsum("mij,mkj->mik", Q, Q), np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(Q), 1.0, atol=1e-10)
    print("   ok")

    print("2. batched spins preserve hemisphere")
    C = rng.normal(size=(20,3)); H = np.array(["L"]*10 + ["R"]*10, dtype=object)
    idx = batched_spin_indices(C, H, 30, rng)
    assert idx.shape == (30,20)
    assert all(H[idx[m,i]] == H[i] for m in range(30) for i in range(20))
    print("   ok")

    print("3. vectorised partial correlation matches scipy")
    n=20; x=rng.normal(size=n); y=rng.normal(size=n); z=rng.normal(size=n)*3+100
    X = x[np.array([rng.permutation(n) for _ in range(40)])]
    def slow(xx):
        rxy,rxz,ryz = stats.pearsonr(xx,y)[0], stats.pearsonr(xx,z)[0], stats.pearsonr(y,z)[0]
        return (rxy-rxz*ryz)/math.sqrt((1-rxz**2)*(1-ryz**2))
    d = np.max(np.abs(partial_corr_many(X,y,z) - np.array([slow(X[i]) for i in range(40)])))
    assert d < 1e-12, d
    print(f"   ok (max diff {d:.2e})")

    print("4. results invariant to --n-jobs")
    nodes = [f"L_a{i}_ROI" for i in range(10)] + [f"R_a{i}_ROI" for i in range(10)]
    size = {n_: 100+i for i,n_ in enumerate(nodes)}
    cent = {n_: rng.normal(size=3) for n_ in nodes}
    ecc, met = {}, {}
    for s in map(str, range(25)):
        e = rng.normal(0,1,20)
        ecc[s] = dict(zip(nodes, e))
        met[s] = dict(zip(nodes, 0.9*e + rng.normal(0,0.3,20)))
    a = run_null(ecc, met, size, cent, "glasser", "spin", 15, 300, np.random.SeedSequence(42), None)
    with Pool(2) as pool:
        b = run_null(ecc, met, size, cent, "glasser", "spin", 15, 300, np.random.SeedSequence(42), pool)
    assert abs(a["mean_partial_r"]-b["mean_partial_r"]) < 1e-12 and a["perm_p_value"]==b["perm_p_value"]
    print(f"   ok (serial == 2-worker: r={a['mean_partial_r']:.3f}, p={a['perm_p_value']:.4f})")

    print("5. spin null recovers planted signal; noise metric is null")
    assert a["perm_p_value"] < 0.05
    met2 = {s: dict(zip(nodes, rng.normal(0,1,20))) for s in map(str,range(25))}
    c = run_null(ecc, met2, size, cent, "glasser", "spin", 15, 300, np.random.SeedSequence(7), None)
    assert c["perm_p_value"] > 0.05, c
    print(f"   ok (signal p={a['perm_p_value']:.4f}, noise p={c['perm_p_value']:.3f})")

    print("6. moran surrogates preserve mean and sd")
    v = rng.normal(size=25); C2 = rng.normal(size=(25,3))
    S = moran_surrogates(v, C2, 100, rng)
    assert abs(S.mean()-v.mean()) < 0.15 and abs(S.std()-v.std()) < 0.3
    print("   ok")
    print("\nSELF-TEST: ALL PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--centroids-csv"); ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200","glasser"))
    ap.add_argument("--null", default="both", choices=("spin","moran","both"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = all cores")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.self_test: self_test(); return
    for r in ("prf_backbone_csv","controllability_csv","centroids_csv","out_csv"):
        if not getattr(a,r): ap.error(f"--{r.replace('_','-')} is required")

    n_jobs = a.n_jobs or os.cpu_count()
    print(f"seed={a.seed}  permutations={a.n_permutations}  workers={n_jobs}")

    ecc, size = load_prf(a.prf_backbone_csv, a.min_vertices_per_node, a.atlas)
    avg, mod = load_ctrl(a.controllability_csv)
    cent = load_centroids(a.centroids_csv)
    ov = set(size) & set(cent)
    if not ov:
        sys.exit("ERROR: no node-label overlap between pRF table and centroids file.")
    print(f"  {len(ov)} nodes have both pRF and centroid data")

    kinds = ["spin","moran"] if a.null=="both" else [a.null]
    base = np.random.SeedSequence(a.seed)
    seeds = base.spawn(len(kinds)*2)
    rows = []
    with Pool(n_jobs) as pool:
        k = 0
        for kind in kinds:
            for name, metric in (("average_controllability_log",avg),("modal_controllability",mod)):
                res = run_null(ecc, metric, size, cent, a.atlas, kind,
                               a.min_valid_nodes, a.n_permutations, seeds[k], pool); k += 1
                if res is None:
                    rows.append({"metric":name,"null_kind":kind,"note":"no valid subjects"}); continue
                res.update({"metric":name,"atlas":a.atlas,"note":""})
                rows.append(res)
                print(f"  [{kind:5s}] {name:30s} n={res['n_subjects']:3d} "
                      f"r={res['mean_partial_r']:+.3f} p={res['perm_p_value']:.4f}")

    ps=[r["perm_p_value"] for r in rows if "perm_p_value" in r]
    qs=bh_fdr(ps) if ps else []
    i=0
    for r in rows:
        if "perm_p_value" in r: r["q_value_fdr"]=float(qs[i]); i+=1
        else: r["q_value_fdr"]=""

    fields=["metric","null_kind","atlas","n_subjects","mean_partial_r","perm_p_value","q_value_fdr","note"]
    with open(a.out_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k_: r.get(k_,"") for k_ in fields})
    print(f"\nWritten to {a.out_csv}")


if __name__ == "__main__":
    main()
