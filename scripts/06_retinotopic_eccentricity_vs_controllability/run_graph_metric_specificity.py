#!/usr/bin/env python3
"""
Is the eccentricity relationship SPECIFIC to controllability, or does it hold
for descriptive graph metrics generally?

Runs the project's standard per-participant spatial-permutation design
(identical to run_claim2_analysis.py: partial correlation of eccentricity
against the metric across each participant's own nodes, covarying node vertex
count; each participant's eccentricity reshuffled across their own nodes to
build the null) against SIX node metrics side by side:

    average_controllability_log    (NCT)
    modal_controllability          (NCT)
    weighted_degree                (descriptive)
    eigenvector_centrality         (descriptive)
    communicability                (descriptive)
    participation_coefficient      (descriptive)

INTERPRETATION, STATED IN ADVANCE
---------------------------------
If eccentricity predicts the descriptive metrics as strongly as it predicts
average controllability, the manuscript's NCT-specific framing is not earned
and should be weakened to a claim about network structure generally.
If average controllability stands out, the framing is earned.
Either outcome is reportable; this script is not built to produce one of them.

Requires the output of compute_graph_metrics.py (server-side) plus the
existing controllability table. Helper functions are vendored, not imported,
for the same reason as run_claim2_nodestrength_control.py.

Usage:
    python3 run_graph_metric_specificity.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv "Received files/all_subjects_controllability_glasser.csv" \
        --graph-metrics-csv all_subjects_graph_metrics_glasser.csv \
        --atlas glasser --out-csv graph_metric_specificity_glasser.csv
    python3 run_graph_metric_specificity.py --self-test
"""
import argparse, csv, math, re, sys
from collections import defaultdict

import numpy as np
from scipy import stats

GLASSER_VIS_AREAS = {"V1","V2","V3","V4","V3A","V3B","V6","V6A","V7","IPS1",
                     "FFC","PIT","V8","VMV1","VMV2","VMV3","VVC",
                     "FST","LO1","LO2","LO3","MST","MT","PH","V3CD","V4t"}
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def is_glasser_vis_node(lab):
    m = GLASSER_LABEL_RE.match(lab)
    return bool(m) and m.group(1) in GLASSER_VIS_AREAS


def load_prf_backbone(path, min_vertices_per_node, atlas):
    subj_ecc, node_size = defaultdict(dict), {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["atlas"] != atlas:
                continue
            vis = (row["is_vis_network"] == "True") if atlas == "schaefer200" \
                  else is_glasser_vis_node(row["backbone_label"])
            if not vis:
                continue
            node_size[row["backbone_label"]] = int(row["n_vertices_total"])
            if row["mean_eccentricity_deg"] == "":
                continue
            if int(row["n_vertices_valid_R2gt30"]) < min_vertices_per_node:
                continue
            subj_ecc[row["subject"]][row["backbone_label"]] = float(row["mean_eccentricity_deg"])
    return subj_ecc, node_size


def load_controllability(path):
    avg, mod = defaultdict(dict), defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s, n = row["subject"], row["node_label"]
            a = float(row["average_controllability"])
            mod[s][n] = float(row["modal_controllability"])
            if a > 0:
                avg[s][n] = math.log(a)
    return avg, mod


def load_graph_metrics(path):
    """Returns {metric_name: {subject: {node: value}}}"""
    cols = ["weighted_degree","eigenvector_centrality","communicability","participation_coefficient"]
    out = {c: defaultdict(dict) for c in cols}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in cols:
                v = row.get(c, "")
                if v not in ("", "nan"):
                    try:
                        out[c][row["subject"]][row["node_label"]] = float(v)
                    except ValueError:
                        pass
    return {c: v for c, v in out.items() if any(v.values())}


def partial_corr(x, y, z):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    rxy, rxz, ryz = stats.pearsonr(x,y)[0], stats.pearsonr(x,z)[0], stats.pearsonr(y,z)[0]
    den = math.sqrt((1-rxz**2)*(1-ryz**2))
    return np.nan if den == 0 or not np.isfinite(den) else (rxy-rxz*ryz)/den


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o]*n/(np.arange(n)+1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def fz(r): return np.arctanh(np.clip(r, -0.999999, 0.999999))


def run_one_metric(subj_ecc, subj_metric, node_size, min_valid_nodes, n_perm, rng):
    per_r, per_perm = [], []
    for s, ecc in subj_ecc.items():
        m = subj_metric.get(s, {})
        common = sorted(set(ecc) & set(m) & set(node_size))
        if len(common) < min_valid_nodes:
            continue
        x = np.array([ecc[n] for n in common])
        y = np.array([m[n] for n in common])
        z = np.array([node_size[n] for n in common], float)
        r = partial_corr(x, y, z)
        if not np.isfinite(r):
            continue
        per_r.append(r)
        idx = np.array([rng.permutation(len(x)) for _ in range(n_perm)])
        per_perm.append([partial_corr(x[i], y, z) for i in idx])
    if not per_r:
        return None
    obs = fz(per_r).mean()
    null = np.nanmean(fz(np.array(per_perm)), axis=0)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"n_subjects": len(per_r), "mean_partial_r": float(np.tanh(obs)),
            "mean_fisher_z": float(obs), "perm_p_value": float(p)}


def self_test():
    rng = np.random.default_rng(0)
    print("1. planted eccentricity signal is recovered")
    ecc, met, size = {}, {}, {f"n{i}": 100+i for i in range(20)}
    nodes = list(size)
    for s in range(25):
        e = rng.normal(0,1,20)
        ecc[str(s)] = dict(zip(nodes, e))
        met[str(s)] = dict(zip(nodes, 0.8*e + rng.normal(0,0.5,20)))
    r = run_one_metric(ecc, met, size, 15, 500, rng)
    assert r["perm_p_value"] < 0.01 and r["mean_partial_r"] > 0.5, r
    print(f"   ok (r={r['mean_partial_r']:.3f}, p={r['perm_p_value']:.4f})")

    print("2. pure-noise metric returns null")
    met2 = {str(s): dict(zip(nodes, rng.normal(0,1,20))) for s in range(25)}
    r2 = run_one_metric(ecc, met2, size, 15, 500, rng)
    assert r2["perm_p_value"] > 0.05, r2
    print(f"   ok (r={r2['mean_partial_r']:+.3f}, p={r2['perm_p_value']:.3f})")

    print("3. bh_fdr monotone and bounded")
    q = bh_fdr([0.001,0.02,0.3,0.9])
    assert all(0 <= v <= 1 for v in q) and q[0] <= q[-1]
    print("   ok")
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
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    for r in ("prf_backbone_csv","controllability_csv","graph_metrics_csv","out_csv"):
        if not getattr(a, r): ap.error(f"--{r.replace('_','-')} is required")

    rng = np.random.default_rng(a.seed)
    subj_ecc, node_size = load_prf_backbone(a.prf_backbone_csv, a.min_vertices_per_node, a.atlas)
    avg, mod = load_controllability(a.controllability_csv)
    gm = load_graph_metrics(a.graph_metrics_csv)

    metrics = {"average_controllability_log": avg, "modal_controllability": mod, **gm}
    ctrl_nodes = set().union(*(set(v) for v in avg.values())) if avg else set()
    if not (set(node_size) & ctrl_nodes):
        sys.exit("ERROR: zero node overlap between pRF and controllability inputs -- mismatched atlas?")

    rows = []
    for name, m in metrics.items():
        res = run_one_metric(subj_ecc, m, node_size, a.min_valid_nodes, a.n_permutations, rng)
        if res is None:
            rows.append({"metric": name, "note": "no valid subjects"}); continue
        res.update({"metric": name, "atlas": a.atlas,
                    "family": "NCT" if "controllability" in name else "descriptive", "note": ""})
        rows.append(res)

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
