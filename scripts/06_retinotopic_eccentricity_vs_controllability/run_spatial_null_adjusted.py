#!/usr/bin/env python3
"""
Does the NODE-STRENGTH-ADJUSTED eccentricity effect survive spatially
constrained nulls?

WHY THIS EXISTS
---------------
The manuscript carries two strong robustness results that have never been
combined. `run_claim2_nodestrength_control.py` shows the effect survives
adjustment for weighted node strength. `run_spatial_null.py` shows it survives
spin and Moran spectral randomisation. But the spatial nulls were run on the
UNADJUSTED statistic, so a reviewer can point out that robustness to spatial
autocorrelation and robustness to degree have been demonstrated separately and
never simultaneously.

This closes that gap. The statistic tested is the degree-adjusted partial
correlation, and the null is the spatially constrained one.

Spin and Moran machinery is imported from `run_spatial_null.py` rather than
reimplemented, so there is exactly one implementation of each null in this
repository and no possibility of the two drifting apart.

WHAT CHANGES RELATIVE TO run_spatial_null.py
--------------------------------------------
Only the covariate set. The original conditions on node size alone; this
conditions on node size and weighted node strength together, by residualisation.
With a single covariate column the two are algebraically identical, and the
self-test verifies that numerically rather than asserting it.

USAGE
-----
    python3 run_spatial_null_adjusted.py \
        --prf-backbone-csv <PRF.csv> --controllability-csv <CTRL.csv> \
        --node-strength-csv <STRENGTH.csv> --centroids-csv <CENTROIDS.csv> \
        --atlas glasser --null both --out-csv <OUT.csv>

    python3 run_spatial_null_adjusted.py --self-test
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

import run_spatial_null as sn


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


def run_null_adjusted(subj_ecc, subj_metric, node_size, subj_strength,
                      centroids, atlas, kind, min_nodes, n_perm, rng,
                      use_strength=True):
    per_r, per_null = [], []
    for s, e in subj_ecc.items():
        m = subj_metric.get(s, {})
        st = subj_strength.get(s, {}) if use_strength else None
        common = set(e) & set(m) & set(node_size) & set(centroids)
        if use_strength:
            common &= set(st)
        common = sorted(common)
        if len(common) < min_nodes:
            continue
        x = np.array([e[n] for n in common])
        y = np.array([m[n] for n in common])
        cols = [np.array([node_size[n] for n in common], float)]
        if use_strength:
            cols.append(np.array([st[n] for n in common], float))
        Z = np.column_stack(cols)

        r = partial_corr_multi(x, y, Z)
        if not np.isfinite(r):
            continue
        per_r.append(r)
        C = np.vstack([centroids[n] for n in common])
        H = np.array([sn.hemi_of(n, atlas) for n in common])
        if kind == "spin":
            nulls = [partial_corr_multi(x[sn.spin_indices(C, H, rng)], y, Z)
                     for _ in range(n_perm)]
        else:
            nulls = [partial_corr_multi(sx, y, Z)
                     for sx in sn.moran_surrogates(x, C, n_perm, rng)]
        per_null.append(nulls)

    if not per_r:
        return None
    obs = fz(per_r).mean()
    null = np.nanmean(fz(np.array(per_null, float)), axis=0)
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"n_subjects": len(per_r), "mean_partial_r": float(np.tanh(obs)),
            "mean_fisher_z": float(obs), "perm_p_value": float(p),
            "null_kind": kind, "adjusted": bool(use_strength)}


# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    rng = np.random.default_rng(0)

    # 1. the multi-covariate estimator must reduce exactly to the imported one
    x, y, z = (rng.normal(size=80) for _ in range(3))
    check("partial_corr_multi with one covariate equals run_spatial_null's",
          abs(partial_corr_multi(x, y, z[:, None]) - sn.partial_corr(x, y, z)) < 1e-10)
    check("partial_corr_multi removes a shared covariate",
          abs(partial_corr_multi(x + 3 * z, y + 3 * z, z[:, None])) < 0.25)

    # 2. the imported null machinery is the real one, not a local copy
    check("spin machinery imported from run_spatial_null",
          sn.spin_indices.__module__ == "run_spatial_null")
    check("moran machinery imported from run_spatial_null",
          sn.moran_surrogates.__module__ == "run_spatial_null")

    def synth(n_sub=25, n_node=30, seed0=0, eff=0.7, via_strength=False):
        r = np.random.default_rng(seed0)
        coords = r.normal(size=(n_node, 3)) * 30
        cents, size = {}, {}
        labs = []
        for j in range(n_node):
            lab = ("L_" if j < n_node // 2 else "R_") + f"V{j:03d}_ROI"
            labs.append(lab)
            cents[lab] = coords[j]
            size[lab] = float(r.integers(200, 900))
        ecc = defaultdict(dict); met = defaultdict(dict); stg = defaultdict(dict)
        for i in range(n_sub):
            s = f"s{i:03d}"
            e = r.uniform(0.5, 8.0, n_node)
            strength = 0.8 * e + r.normal(0, 0.5, n_node) if via_strength \
                else r.normal(0, 1.0, n_node)
            y = (1.2 * strength if via_strength else eff * e) \
                + 0.2 * np.array([size[l] for l in labs]) / 500 \
                + r.normal(0, 0.7, n_node)
            for j, lab in enumerate(labs):
                ecc[s][lab] = float(e[j]); met[s][lab] = float(y[j])
                stg[s][lab] = float(strength[j])
        return ecc, met, size, stg, cents

    # 3. a real effect, independent of strength, must survive the adjusted null
    e_, m_, sz_, st_, c_ = synth(seed0=1, eff=0.9)
    res = run_null_adjusted(e_, m_, sz_, st_, c_, "glasser", "spin", 10, 200,
                            np.random.default_rng(1))
    check("real effect survives the strength-adjusted spin null",
          res["perm_p_value"] < 0.05 and res["mean_partial_r"] > 0.3)
    res_m = run_null_adjusted(e_, m_, sz_, st_, c_, "glasser", "moran", 10, 200,
                              np.random.default_rng(1))
    check("real effect survives the strength-adjusted Moran null",
          res_m["perm_p_value"] < 0.05)

    # 4. NEGATIVE CONTROL: an effect that exists ONLY through node strength must
    #    be destroyed once strength enters, even under the spatial null
    e2, m2, sz2, st2, c2 = synth(seed0=2, via_strength=True)
    unadj = run_null_adjusted(e2, m2, sz2, st2, c2, "glasser", "spin", 10, 200,
                              np.random.default_rng(2), use_strength=False)
    adj = run_null_adjusted(e2, m2, sz2, st2, c2, "glasser", "spin", 10, 200,
                            np.random.default_rng(2), use_strength=True)
    check("NEGATIVE CONTROL: strength-mediated effect is large unadjusted",
          abs(unadj["mean_partial_r"]) > 0.3)
    check("NEGATIVE CONTROL: adding strength destroys it",
          abs(adj["mean_partial_r"]) < abs(unadj["mean_partial_r"]) / 3)

    # 5. pure noise must be null
    e3, m3, sz3, st3, c3 = synth(seed0=3, eff=0.0)
    res_n = run_null_adjusted(e3, m3, sz3, st3, c3, "glasser", "spin", 10, 200,
                              np.random.default_rng(3))
    check(f"pure noise returns null (p={res_n['perm_p_value']:.3f})",
          res_n["perm_p_value"] > 0.05)

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--node-strength-csv"); ap.add_argument("--centroids-csv")
    ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--null", default="both", choices=("spin", "moran", "both"))
    ap.add_argument("--metric", default="average", choices=("average", "modal"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for r in ("prf_backbone_csv", "controllability_csv", "node_strength_csv",
              "centroids_csv", "out_csv"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required unless --self-test")

    ecc, size = sn.load_prf(a.prf_backbone_csv, a.min_vertices_per_node, a.atlas)
    avg, mod = sn.load_ctrl(a.controllability_csv)
    metric = avg if a.metric == "average" else mod
    strength = load_strength(a.node_strength_csv)
    cents = sn.load_centroids(a.centroids_csv)

    kinds = ["spin", "moran"] if a.null == "both" else [a.null]
    rows = []
    for kind in kinds:
        for use in (False, True):
            r = run_null_adjusted(ecc, metric, size, strength, cents, a.atlas,
                                  kind, a.min_valid_nodes, a.n_permutations,
                                  np.random.default_rng(a.seed), use_strength=use)
            if r:
                rows.append(r)

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["null_kind", "adjusted", "n_subjects",
                                           "mean_partial_r", "mean_fisher_z",
                                           "perm_p_value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"atlas = {a.atlas}   metric = {a.metric} controllability   "
          f"{a.n_permutations} surrogates\n")
    for r in rows:
        tag = "size + node strength" if r["adjusted"] else "size only"
        print(f"  {r['null_kind']:6s} null, covariates {tag:22s}  "
              f"n = {r['n_subjects']:3d}   r = {r['mean_partial_r']:+.4f}   "
              f"p = {r['perm_p_value']:.4f}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
