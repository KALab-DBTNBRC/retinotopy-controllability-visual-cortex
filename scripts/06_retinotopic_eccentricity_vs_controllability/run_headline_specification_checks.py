#!/usr/bin/env python3
"""
Does the headline result depend on two specification choices it should not?

WHY THIS EXISTS
---------------
Two reviewer objections, both about the same statistic: the mean per-participant
partial correlation between node eccentricity and log average controllability.

ARM A, pRF validity. The retinotopy stimulus has a radius of roughly 8 degrees,
yet the released per-vertex eccentricity estimates reach 17 to 20 degrees. Values
beyond the stimulus aperture are not physiologically supported, and with only
about 18 to 21 nodes per participant a handful of extreme values could carry a
correlation. This arm recomputes the headline statistic while capping node
eccentricity at progressively tighter ceilings, and separately while raising the
floor on the proportion of a parcel's vertices that survive the pRF fit
threshold.

    A caveat that has to be stated rather than glossed. The aggregated backbone
    table is built with a fixed R2 > 30 vertex threshold already applied, so the
    R2 threshold itself cannot be swept without re-running the vertex-level
    aggregation on the raw CIFTI files. `pct_valid`, the proportion of a parcel's
    vertices clearing that threshold, is the strongest fit-quality proxy
    available locally, and is not the same thing.

ARM B, nonlinear node strength. The paper adjusts for weighted node degree with a
linear partial correlation. Controllability is a nonlinear spectral function of
the connectome, so linear residualisation of degree does not establish
conditional independence from degree. This arm adds a quadratic degree term, a
cubic term, and a rank transform of degree, so that the reported claim can be
"robust to degree adjustment" rather than "robust to linear degree adjustment".

Inference throughout is the same within-participant spatial permutation used for
the headline result: outcome and covariates are residualised once, the reshuffled
predictor is residualised per permutation.

USAGE
-----
    python3 run_headline_specification_checks.py \
        --prf-backbone-csv <PRF.csv> --controllability-csv <CTRL.csv> \
        --node-strength-csv <STRENGTH.csv> --atlas glasser --out-csv <OUT.csv>

    python3 run_headline_specification_checks.py --self-test
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np

GLASSER_VIS_AREAS = {
    "V1", "V2", "V3", "V4", "V3A", "V3B", "V6", "V6A", "V7", "IPS1",
    "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",
    "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t"}


def is_glasser_vis(lab):
    p = lab.split("_")
    return len(p) >= 3 and p[0] in ("L", "R") and p[1] in GLASSER_VIS_AREAS


def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def _residualize(v, Z):
    X = np.column_stack([np.ones(len(v)), Z])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_corr_multi(x, y, Z):
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if len(x) < Z.shape[1] + 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx, ry = _residualize(x, Z), _residualize(y, Z)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))


def _pearson_rowwise(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def permuted_partial_r(x, y, Z, n_perm, rng):
    x = np.asarray(x, float); k = len(x)
    ry = _residualize(np.asarray(y, float), Z)
    keys = np.random.default_rng(rng.integers(0, 2 ** 63 - 1)).random((n_perm, k))
    Xp = x[np.argsort(keys, axis=1)]
    Xd = np.column_stack([np.ones(k), np.asarray(Z, float)])
    H = Xd @ np.linalg.pinv(Xd)
    return _pearson_rowwise(Xp - Xp @ H.T, np.tile(ry, (n_perm, 1)))


def cell(nodes_by_sub, min_nodes, n_perm, rng):
    """nodes_by_sub: subject -> (x, y, Z). Returns the group statistic."""
    rs, arrays = [], []
    for s, (x, y, Z) in nodes_by_sub.items():
        if len(x) < min_nodes:
            continue
        r = partial_corr_multi(x, y, Z)
        if np.isfinite(r):
            rs.append(r); arrays.append((x, y, Z))
    if len(rs) < 5:
        return {"n_subjects": len(rs), "median_n_nodes": float("nan"),
                "mean_partial_r": float("nan"), "mean_fisher_z": float("nan"),
                "p_permutation": float("nan")}
    z = fz(rs)
    obs = float(z.mean())
    null = np.nanmean(np.vstack([fz(permuted_partial_r(x, y, Z, n_perm, rng))
                                 for (x, y, Z) in arrays]), axis=0)
    null = null[np.isfinite(null)]
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)) \
        if null.size else float("nan")
    return {"n_subjects": len(rs),
            "median_n_nodes": float(np.median([len(x) for (x, _, _) in arrays])),
            "mean_partial_r": float(np.mean(rs)), "mean_fisher_z": obs,
            "p_permutation": p}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_prf(path, atlas, minv):
    """subject -> label -> (ecc, pct_valid); plus the fixed node-size map."""
    ecc, size = defaultdict(dict), {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("atlas") != atlas:
                continue
            lab = r["backbone_label"]
            vis = (r.get("is_vis_network") == "True") if atlas == "schaefer200" \
                else is_glasser_vis(lab)
            if not vis:
                continue
            try:
                size[lab] = float(r["n_vertices_total"])
                if float(r.get("n_vertices_valid_R2gt30") or 0) < minv:
                    continue
                e = float(r["mean_eccentricity_deg"])
                pv = float(r.get("pct_valid") or "nan")
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(e):
                ecc[r["subject"]][lab] = (e, pv)
    return ecc, size


def load_ctrl(path):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                a = float(r["average_controllability"])
            except (ValueError, KeyError, TypeError):
                continue
            if a > 0:
                out[r["subject"]][r["node_label"]] = math.log(a)
    return out


def load_strength(path):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["weighted_degree"])
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(v):
                out[r["subject"]][r["node_label"]] = v
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------

def build(ecc, ctrl, size, strength, ecc_ceiling=None, pct_floor=None,
          degree_terms=0, rank_degree=False):
    """Assemble per-participant (x, y, Z) under one specification."""
    out = {}
    for s, nodes in ecc.items():
        c = ctrl.get(s, {})
        st = strength.get(s, {}) if (degree_terms or rank_degree) else None
        keys = set(nodes) & set(c) & set(size)
        if st is not None:
            keys &= set(st)
        sel = []
        for n in sorted(keys):
            e, pv = nodes[n]
            if ecc_ceiling is not None and e > ecc_ceiling:
                continue
            if pct_floor is not None and not (np.isfinite(pv) and pv >= pct_floor):
                continue
            sel.append(n)
        if len(sel) < 5:
            continue
        x = np.array([nodes[n][0] for n in sel])
        y = np.array([c[n] for n in sel])
        cols = [np.array([size[n] for n in sel], float)]
        if st is not None:
            d = np.array([st[n] for n in sel], float)
            if rank_degree:
                cols.append(np.argsort(np.argsort(d)).astype(float))
            else:
                ds = (d - d.mean()) / (d.std() if d.std() > 0 else 1.0)
                for k in range(1, degree_terms + 1):
                    cols.append(ds ** k)
        out[s] = (x, y, np.column_stack(cols))
    return out


# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    def synth(n_sub=30, n_node=30, seed0=0, eff=0.6, outlier_driven=False,
              via_degree_sq=False):
        r = np.random.default_rng(seed0)
        ecc, ctrl, stg = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        size = {f"N{j:03d}": float(r.integers(200, 900)) for j in range(n_node)}
        for i in range(n_sub):
            s = f"s{i:03d}"
            e = r.uniform(0.5, 7.0, n_node)
            e[:3] = r.uniform(15.0, 20.0, 3)        # out-of-aperture nodes
            d = r.normal(0, 1, n_node)
            if outlier_driven:
                # signal lives ONLY in the out-of-aperture nodes
                y = np.where(e > 10, 3.0 * e, 0.0) + r.normal(0, 1.0, n_node)
            elif via_degree_sq:
                y = 1.5 * ((d - d.mean()) ** 2) + r.normal(0, 0.5, n_node)
                e = 0.8 * ((d - d.mean()) ** 2) + r.normal(0, 0.4, n_node)
            else:
                y = eff * e + r.normal(0, 1.0, n_node)
            for j in range(n_node):
                lab = f"N{j:03d}"
                pv = float(np.clip(r.beta(2, 3), 0, 1))
                ecc[s][lab] = (float(e[j]), pv)
                ctrl[s][lab] = float(y[j])
                stg[s][lab] = float(d[j])
        return ecc, ctrl, size, stg

    rng = np.random.default_rng(0)
    x, y, z = (rng.normal(size=90) for _ in range(3))
    check("partial_corr_multi one covariate matches the closed form",
          abs(partial_corr_multi(x, y, z[:, None])
              - (lambda: (np.corrcoef(x - np.polyval(np.polyfit(z, x, 1), z),
                                      y - np.polyval(np.polyfit(z, y, 1), z))[0, 1]))()) < 1e-9)

    e_, c_, sz_, st_ = synth(seed0=1)
    full = cell(build(e_, c_, sz_, st_), 10, 600, np.random.default_rng(1))
    cap8 = cell(build(e_, c_, sz_, st_, ecc_ceiling=8.0), 10, 600,
                np.random.default_rng(1))
    check("arm A: a genuine effect survives the 8 degree ceiling",
          cap8["p_permutation"] < 0.01)
    check("arm A: the ceiling actually removes nodes",
          cap8["median_n_nodes"] < full["median_n_nodes"])
    check("arm A: a genuine effect is not much changed by the ceiling",
          abs(cap8["mean_partial_r"] - full["mean_partial_r"]) < 0.25)

    # NEGATIVE CONTROL: effect carried entirely by out-of-aperture nodes
    e2, c2, sz2, st2 = synth(seed0=2, outlier_driven=True)
    f2 = cell(build(e2, c2, sz2, st2), 10, 600, np.random.default_rng(2))
    k2 = cell(build(e2, c2, sz2, st2, ecc_ceiling=8.0), 10, 600,
              np.random.default_rng(2))
    check("arm A NEGATIVE CONTROL: outlier-driven effect is large uncapped",
          abs(f2["mean_partial_r"]) > 0.3)
    check("arm A NEGATIVE CONTROL: the ceiling destroys it",
          abs(k2["mean_partial_r"]) < abs(f2["mean_partial_r"]) / 2)

    # arm A, quality floor
    flo = cell(build(e_, c_, sz_, st_, pct_floor=0.30), 10, 600,
               np.random.default_rng(3))
    check("arm A: quality floor retains the effect", flo["p_permutation"] < 0.05)
    check("arm A: quality floor removes nodes",
          flo["median_n_nodes"] < full["median_n_nodes"])

    # arm B, nonlinear degree
    lin = cell(build(e_, c_, sz_, st_, degree_terms=1), 10, 600,
               np.random.default_rng(4))
    quad = cell(build(e_, c_, sz_, st_, degree_terms=2), 10, 600,
                np.random.default_rng(4))
    rk = cell(build(e_, c_, sz_, st_, rank_degree=True), 10, 600,
              np.random.default_rng(4))
    check("arm B: effect independent of degree survives every degree form",
          all(c["p_permutation"] < 0.01 for c in (lin, quad, rk)))

    # NEGATIVE CONTROL: association that runs entirely through degree SQUARED.
    # Linear degree adjustment should not remove it; the quadratic term should.
    e3, c3, sz3, st3 = synth(seed0=5, via_degree_sq=True)
    l3 = cell(build(e3, c3, sz3, st3, degree_terms=1), 10, 600,
              np.random.default_rng(5))
    q3 = cell(build(e3, c3, sz3, st3, degree_terms=2), 10, 600,
              np.random.default_rng(5))
    check("arm B NEGATIVE CONTROL: quadratic-degree artifact survives LINEAR "
          "adjustment", abs(l3["mean_partial_r"]) > 0.3)
    check("arm B NEGATIVE CONTROL: the quadratic term removes it",
          abs(q3["mean_partial_r"]) < abs(l3["mean_partial_r"]) / 2)

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--node-strength-csv"); ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for r in ("prf_backbone_csv", "controllability_csv", "node_strength_csv", "out_csv"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required unless --self-test")

    ecc, size = load_prf(a.prf_backbone_csv, a.atlas, a.min_vertices_per_node)
    ctrl = load_ctrl(a.controllability_csv)
    strength = load_strength(a.node_strength_csv)

    specs = []
    for cap in [None, 12.0, 10.0, 8.0, 7.0, 6.0]:
        specs.append((f"ecc_ceiling={'none' if cap is None else cap:>4}",
                      dict(ecc_ceiling=cap)))
    for flo in [0.10, 0.20, 0.30, 0.40]:
        specs.append((f"pct_valid_floor={flo:.2f}", dict(pct_floor=flo)))
    for name, kw in [("degree: none", {}), ("degree: linear", dict(degree_terms=1)),
                     ("degree: + quadratic", dict(degree_terms=2)),
                     ("degree: + cubic", dict(degree_terms=3)),
                     ("degree: rank", dict(rank_degree=True))]:
        specs.append((name, kw))

    rows = []
    for name, kw in specs:
        res = cell(build(ecc, ctrl, size, strength, **kw), a.min_valid_nodes,
                   a.n_permutations, np.random.default_rng(a.seed))
        res["specification"] = name
        rows.append(res)

    fields = ["specification", "n_subjects", "median_n_nodes", "mean_partial_r",
              "mean_fisher_z", "p_permutation"]
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print(f"atlas = {a.atlas}   metric = average controllability   "
          f"covariate = node size\n")
    print(f"    {'specification':24s} {'n':>3s} {'nodes':>6s} {'mean r':>8s} {'p':>8s}")
    for r in rows:
        print(f"    {r['specification']:24s} {r['n_subjects']:3d} "
              f"{r['median_n_nodes']:6.1f} {r['mean_partial_r']:+8.4f} "
              f"{r['p_permutation']:8.4f}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
