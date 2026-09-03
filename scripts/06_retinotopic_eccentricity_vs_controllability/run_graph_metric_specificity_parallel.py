#!/usr/bin/env python3
"""
Parallel + vectorised version of run_graph_metric_specificity.py.

Same analysis, same design, same outputs. Two changes, both purely about speed:

  1. VECTORISED NULL. The serial version called scipy.stats.pearsonr three
     times per permutation. This computes all permutations for a subject in a
     single centred matrix product. Verified identical to the scipy path to
     1.1e-15 across 200 random trials (see _vec_check.py).

  2. PARALLEL OVER (metric, subject). Each subject's permutation loop is
     independent, so the work splits cleanly. With 5 metrics x ~40 subjects
     there are ~200 independent units, which maps well onto a large core count.

REPRODUCIBILITY -- READ THIS
----------------------------
Each (metric, subject) unit draws from its OWN deterministic RNG stream, via
numpy SeedSequence spawned from --seed. That means:

  * a given --seed and --n-jobs ALWAYS produce identical results;
  * results are also invariant to --n-jobs, because each unit's stream depends
    on its index, not on scheduling order;
  * but they will NOT bit-match the serial script's output, because the serial
    script draws every permutation from one shared stream in subject order.

Both are internally reproducible. If you need to reproduce a previously
reported serial number exactly, re-run the serial script. For new results,
prefer this one and record that the parallel implementation was used.

Usage:
    python3 run_graph_metric_specificity_parallel.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --graph-metrics-csv all_subjects_graph_metrics_glasser.csv \
        --atlas glasser --n-permutations 10000 --n-jobs 80 \
        --out-csv graph_metric_specificity_glasser.csv

    python3 run_graph_metric_specificity_parallel.py --self-test
"""
import argparse, csv, math, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import stats

GLASSER_VIS_AREAS = {"V1","V2","V3","V4","V3A","V3B","V6","V6A","V7","IPS1",
                     "FFC","PIT","V8","VMV1","VMV2","VMV3","VVC",
                     "FST","LO1","LO2","LO3","MST","MT","PH","V3CD","V4t"}
GLASSER_RE = re.compile(r"^[LR]_(.+)_ROI$")
GM_COLS = ["weighted_degree","eigenvector_centrality","communicability","participation_coefficient"]


def is_vis_glasser(l):
    m = GLASSER_RE.match(l); return bool(m) and m.group(1) in GLASSER_VIS_AREAS


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


def load_graph_metrics(path):
    out = {c: defaultdict(dict) for c in GM_COLS}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for c in GM_COLS:
                v = r.get(c, "")
                if v in ("", "nan", "NaN"): continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if not math.isnan(fv):
                    out[c][r["subject"]][r["node_label"]] = fv
    return {c: v for c, v in out.items() if any(v.values())}


def partial_corr_many(X, y, z):
    """X: (m, n) permuted predictors. Vectorised partial correlation."""
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean(); zc = z - z.mean()
    Xn = np.linalg.norm(Xc, axis=1); yn = np.linalg.norm(yc); zn = np.linalg.norm(zc)
    if yn == 0 or zn == 0:
        return np.full(X.shape[0], np.nan)
    rXy = (Xc @ yc) / (Xn * yn)
    rXz = (Xc @ zc) / (Xn * zn)
    ryz = float((yc @ zc) / (yn * zn))
    den = np.sqrt((1 - rXz**2) * (1 - ryz**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (rXy - rXz * ryz) / den, np.nan)
    return np.where(Xn > 0, out, np.nan)


def fz(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o]*n/(np.arange(n)+1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


# ---- one work unit: a single (metric, subject) ------------------------------
def _unit(args):
    x, y, z, n_perm, seed_state = args
    rng = np.random.default_rng(seed_state)
    obs = float(partial_corr_many(x[None, :], y, z)[0])
    if not np.isfinite(obs):
        return None
    n = x.size
    idx = np.argsort(rng.random((n_perm, n)), axis=1)   # vectorised permutations
    null = partial_corr_many(x[idx], y, z)
    return obs, null


def run_metric(subj_ecc, subj_metric, node_size, min_nodes, n_perm, seedseq, pool):
    units, states = [], []
    for s in sorted(subj_ecc):
        m = subj_metric.get(s, {})
        common = sorted(set(subj_ecc[s]) & set(m) & set(node_size))
        if len(common) < min_nodes: continue
        units.append((np.array([subj_ecc[s][n] for n in common], float),
                      np.array([m[n] for n in common], float),
                      np.array([node_size[n] for n in common], float)))
    if not units: return None
    child = seedseq.spawn(len(units))
    payload = [(u[0], u[1], u[2], n_perm, child[i]) for i, u in enumerate(units)]
    results = pool.map(_unit, payload) if pool else [_unit(p) for p in payload]
    results = [r for r in results if r is not None]
    if not results: return None
    obs = fz([r[0] for r in results]).mean()
    null = np.nanmean(fz(np.array([r[1] for r in results])), axis=0)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"n_subjects": len(results), "mean_partial_r": float(np.tanh(obs)),
            "mean_fisher_z": float(obs), "perm_p_value": float(p)}


def self_test():
    print("1. vectorised partial correlation matches scipy")
    rng = np.random.default_rng(0)
    n = 20; x = rng.normal(size=n); y = rng.normal(size=n); z = rng.normal(size=n)*3+100
    X = x[np.array([rng.permutation(n) for _ in range(40)])]
    def slow(xx):
        rxy,rxz,ryz = stats.pearsonr(xx,y)[0], stats.pearsonr(xx,z)[0], stats.pearsonr(y,z)[0]
        return (rxy-rxz*ryz)/math.sqrt((1-rxz**2)*(1-ryz**2))
    d = np.max(np.abs(partial_corr_many(X,y,z) - np.array([slow(X[i]) for i in range(40)])))
    assert d < 1e-12, d
    print(f"   ok (max diff {d:.2e})")

    print("2. results are invariant to --n-jobs")
    size = {f"n{i}": 100+i for i in range(20)}; nodes = list(size)
    ecc, met = {}, {}
    for s in map(str, range(25)):
        e = rng.normal(0,1,20)
        ecc[s] = dict(zip(nodes, e))
        met[s] = dict(zip(nodes, 0.7*e + rng.normal(0,0.5,20)))
    ss = np.random.SeedSequence(42)
    a = run_metric(ecc, met, size, 15, 500, np.random.SeedSequence(42), None)
    with Pool(2) as pool:
        b = run_metric(ecc, met, size, 15, 500, np.random.SeedSequence(42), pool)
    assert abs(a["mean_partial_r"]-b["mean_partial_r"]) < 1e-12, (a,b)
    assert a["perm_p_value"] == b["perm_p_value"], (a,b)
    print(f"   ok (serial and 2-worker identical: r={a['mean_partial_r']:.4f}, p={a['perm_p_value']:.4f})")

    print("3. planted signal recovered")
    assert a["perm_p_value"] < 0.01 and a["mean_partial_r"] > 0.4, a
    print(f"   ok")

    print("4. pure noise returns null")
    met2 = {s: dict(zip(nodes, rng.normal(0,1,20))) for s in map(str,range(25))}
    c = run_metric(ecc, met2, size, 15, 500, np.random.SeedSequence(7), None)
    assert c["perm_p_value"] > 0.05, c
    print(f"   ok (p={c['perm_p_value']:.3f})")
    print("\nSELF-TEST: ALL PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--graph-metrics-csv"); ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200","glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = all available cores")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.self_test: self_test(); return
    for r in ("prf_backbone_csv","controllability_csv","graph_metrics_csv","out_csv"):
        if not getattr(a,r): ap.error(f"--{r.replace('_','-')} is required")

    n_jobs = a.n_jobs or os.cpu_count()
    print(f"seed={a.seed}  permutations={a.n_permutations}  workers={n_jobs}")

    subj_ecc, node_size = load_prf(a.prf_backbone_csv, a.min_vertices_per_node, a.atlas)
    avg, mod = load_ctrl(a.controllability_csv)
    gm = load_graph_metrics(a.graph_metrics_csv)
    print(f"  descriptive metrics present: {', '.join(gm) if gm else 'NONE'}")

    ctrl_nodes = set().union(*(set(v) for v in avg.values())) if avg else set()
    if not (set(node_size) & ctrl_nodes):
        sys.exit("ERROR: zero node-label overlap between pRF and controllability inputs -- mismatched atlas?")

    metrics = {"average_controllability_log": avg, "modal_controllability": mod, **gm}
    base = np.random.SeedSequence(a.seed)
    per_metric_seeds = base.spawn(len(metrics))

    rows = []
    with Pool(n_jobs) as pool:
        for (name, m), ss in zip(metrics.items(), per_metric_seeds):
            res = run_metric(subj_ecc, m, node_size, a.min_valid_nodes, a.n_permutations, ss, pool)
            if res is None:
                rows.append({"metric": name, "note": "no valid subjects"}); continue
            res.update({"metric": name, "atlas": a.atlas,
                        "family": "NCT" if "controllability" in name else "descriptive", "note": ""})
            rows.append(res)
            print(f"  done: {name}")

    ps = [r["perm_p_value"] for r in rows if "perm_p_value" in r]
    qs = bh_fdr(ps) if ps else []
    k = 0
    for r in rows:
        if "perm_p_value" in r: r["q_value_fdr"] = float(qs[k]); k += 1
        else: r["q_value_fdr"] = ""

    print(f"\n=== eccentricity vs each node metric ({a.atlas}) ===")
    for r in sorted(rows, key=lambda x: -abs(x.get("mean_partial_r", 0))):
        if "perm_p_value" not in r:
            print(f"  {r['metric']:30s} {r['note']}"); continue
        print(f"  {r['metric']:30s} [{r['family']:11s}] n={r['n_subjects']:3d} "
              f"r={r['mean_partial_r']:+.3f} p={r['perm_p_value']:.4f} q={r['q_value_fdr']:.4f}")

    fields = ["metric","family","atlas","n_subjects","mean_partial_r","mean_fisher_z",
              "perm_p_value","q_value_fdr","note"]
    with open(a.out_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in fields})
    print(f"\nWritten to {a.out_csv}")


if __name__ == "__main__":
    main()
