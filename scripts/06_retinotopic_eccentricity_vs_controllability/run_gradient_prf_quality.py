#!/usr/bin/env python3
"""
Does the retinotopic gradient depend on pRF data quality?

WHY THIS EXISTS
---------------
Every node entering the analysis is a parcel average over vertices whose pRF
model exceeded R2 > 30. Nodes differ widely in how many vertices clear that
bar, and participants differ in how many good nodes they have at all. If the
eccentricity-controllability gradient were stronger in participants with worse
pRF data, or were carried by the sparsely sampled nodes, the finding would be a
measurement artefact rather than an organizing principle.

The manuscript already reports a sensitivity sweep across minimum-valid-node
thresholds, and a missingness check showing that node validity does not depend
on eccentricity. Neither asks the question directly: is the gradient itself
related to data quality?

WHAT THIS DOES
--------------
Two arms, both on the per-participant gradient, defined exactly as in
`run_gradient_demographics.py` so the two are commensurable: the partial
correlation between node eccentricity and node controllability with the
covariate removed, Fisher-z transformed.

Arm 1, association. Three participant-level quality indices are computed over
the nodes that actually entered that participant's gradient:

    mean_pct_valid            mean percentage of parcel vertices surviving R2>30
    log_total_valid_vertices  log of the summed suprathreshold vertex count
    n_valid_nodes             how many nodes entered the gradient

Each is regressed on the gradient marginally, then all three jointly. Inference
is by permutation: the predictor rows are shuffled together, which preserves the
correlation structure among the indices and destroys only their link to the
gradient. The correlation matrix among the indices is reported too, since they
are not independent and a joint model over collinear predictors is hard to read
without it.

Arm 2, threshold escalation. The cohort mean gradient is recomputed while the
minimum-suprathreshold-vertices-per-node requirement is raised. If the gradient
is carried by poorly sampled nodes it should decay as the bar rises. The number
of surviving participants is reported at every step, because a gradient that
changes while the cohort is shrinking is a different phenomenon from one that
changes while the cohort is intact.

Neither arm can prove the gradient is quality-free. A null here means quality
does not track gradient strength across the range this cohort actually spans,
which is narrower than the range the measurement could in principle take.

USAGE
-----
    python3 run_gradient_prf_quality.py \
        --prf-backbone-csv <PRF.csv> \
        --controllability-csv <CTRL.csv> \
        --atlas glasser --out-csv <OUT.csv>

    python3 run_gradient_prf_quality.py --self-test
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# estimators, identical in definition to run_gradient_demographics.py
# ----------------------------------------------------------------------------

def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z)), np.asarray(z, float)])
    try:
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))


def ols_beta(y, X):
    """Coefficients of y on [1, X]. Returns None if rank-deficient."""
    D = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    if np.linalg.matrix_rank(D) < D.shape[1]:
        return None
    try:
        return np.linalg.lstsq(D, y, rcond=None)[0][1:]
    except np.linalg.LinAlgError:
        return None


def perm_test(y, X, n_perm, rng):
    obs = ols_beta(y, X)
    if obs is None:
        return None, None
    Xa = np.column_stack([np.asarray(c, float) for c in X])
    cnt = np.zeros(len(obs))
    for _ in range(n_perm):
        idx = rng.permutation(len(y))
        b = ols_beta(y, [Xa[idx, j] for j in range(Xa.shape[1])])
        if b is not None:
            cnt += (np.abs(b) >= np.abs(obs))
    return obs, (cnt + 1) / (n_perm + 1)


def signflip_p(v, n_perm, rng):
    """Two-sided sign-flip test that the mean of v is zero."""
    v = np.asarray(v, float)
    if v.size < 3:
        return float("nan")
    signs = rng.choice([-1.0, 1.0], size=(n_perm, v.size))
    null = (signs * v).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(v.mean())) + 1) / (n_perm + 1))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_prf(path, atlas, covariate="mean_rfsize_deg"):
    """subject -> label -> (ecc, covariate, n_valid_vertices, pct_valid).

    No vertex threshold is applied here. Unlike the other gradient scripts this
    one has to vary that threshold, so filtering is deferred to gradients().
    """
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("atlas") != atlas:
                continue
            try:
                ecc = float(row["mean_eccentricity_deg"])
                cov = float(row.get(covariate) or "nan")
                nv = float(row.get("n_vertices_valid_R2gt30") or 0)
                pct = float(row.get("pct_valid") or "nan")
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(ecc):
                out[row["subject"]][row["backbone_label"]] = (ecc, cov, nv, pct)
    return out


def load_ctrl(path, metric):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        lab = next((c for c in rd.fieldnames if "label" in c.lower()), None)
        for row in rd:
            try:
                v = float(row[metric])
            except (ValueError, KeyError, TypeError):
                continue
            if metric == "average_controllability":
                if v <= 0:
                    continue
                v = np.log(v)
            out[row["subject"]][row[lab]] = v
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

def gradients(prf, ctrl, min_nodes, min_vertices):
    """subject -> (gradient_z, quality dict), at one vertex threshold."""
    out = {}
    for s in sorted(set(prf) & set(ctrl)):
        nodes = [n for n in sorted(set(prf[s]) & set(ctrl[s]))
                 if prf[s][n][2] >= min_vertices]
        if len(nodes) < min_nodes:
            continue
        ecc = np.array([prf[s][n][0] for n in nodes])
        cov = np.array([prf[s][n][1] for n in nodes])
        nv = np.array([prf[s][n][2] for n in nodes])
        pct = np.array([prf[s][n][3] for n in nodes])
        y = np.array([ctrl[s][n] for n in nodes])
        if not np.all(np.isfinite(cov)):
            continue
        r = partial_corr(ecc, y, cov)
        if not np.isfinite(r):
            continue
        out[s] = (fz(r), {
            "mean_pct_valid": float(np.nanmean(pct)),
            "log_total_valid_vertices": float(np.log(max(nv.sum(), 1.0))),
            "n_valid_nodes": float(len(nodes)),
        })
    return out


QUALITY_TERMS = ["mean_pct_valid", "log_total_valid_vertices", "n_valid_nodes"]


def run_association(grad, n_perm, seed):
    rng = np.random.default_rng(seed)
    subs = sorted(grad)
    if len(subs) < 8:
        return None
    y = np.array([grad[s][0] for s in subs])
    Q = {t: np.array([grad[s][1][t] for s in subs]) for t in QUALITY_TERMS}

    rows = []
    for t in QUALITY_TERMS:
        b, p = perm_test(y, [Q[t]], n_perm, rng)
        if b is not None:
            rows.append({"model": t, "term": t, "n_subjects": len(subs),
                         "beta": float(b[0]), "p_permutation": float(p[0])})
    b, p = perm_test(y, [Q[t] for t in QUALITY_TERMS], n_perm, rng)
    if b is not None:
        for t, bi, pi in zip(QUALITY_TERMS, b, p):
            rows.append({"model": "joint", "term": t, "n_subjects": len(subs),
                         "beta": float(bi), "p_permutation": float(pi)})

    M = np.column_stack([Q[t] for t in QUALITY_TERMS])
    corr = np.corrcoef(M, rowvar=False)
    return {"rows": rows, "n": len(subs),
            "mean_gradient_z": float(y.mean()),
            "quality_corr": corr,
            "quality_ranges": {t: (float(Q[t].min()), float(Q[t].max()))
                               for t in QUALITY_TERMS}}


def run_escalation(prf, ctrl, min_nodes, thresholds, n_perm, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for thr in thresholds:
        g = gradients(prf, ctrl, min_nodes, thr)
        if len(g) < 3:
            rows.append({"min_vertices_per_node": thr, "n_subjects": len(g),
                         "mean_gradient_z": float("nan"),
                         "p_permutation": float("nan")})
            continue
        v = np.array([g[s][0] for s in g])
        rows.append({"min_vertices_per_node": thr, "n_subjects": len(g),
                     "mean_gradient_z": float(v.mean()),
                     "p_permutation": signflip_p(v, n_perm, rng)})
    return rows


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    def synth(n_sub=46, n_node=52, seed0=0, slope=0.35,
              quality_effect=0.0, weak_node_only=False, poor_frac=0.5):
        """Synthetic cohort with controllable pRF quality structure.

        quality_effect  gradient slope scales with participant data quality
        weak_node_only  the eccentricity signal exists ONLY on poorly sampled
                        nodes, so raising the vertex threshold must kill it
        """
        r = np.random.default_rng(seed0)
        prf, ctrl = defaultdict(dict), defaultdict(dict)
        for i in range(n_sub):
            s = f"s{i:03d}"
            qual = r.uniform(0.25, 0.95)              # participant data quality
            n_poor = int(round(n_node * poor_frac))
            nv = np.concatenate([
                r.integers(2, 9, n_poor),             # sparsely sampled nodes
                r.integers(40, 400, n_node - n_poor)  # well sampled nodes
            ]).astype(float)
            r.shuffle(nv)
            ecc = r.uniform(0.5, 8.0, n_node)
            cov = 0.4 + 0.25 * ecc + r.normal(0, 0.3, n_node)
            eff = slope * (1.0 + quality_effect * (qual - 0.6))
            if weak_node_only:
                eff = np.where(nv < 10, slope * 3.0, 0.0)
            y = eff * ecc + 0.35 * cov + r.normal(0, 1.0, n_node)
            for j in range(n_node):
                lab = f"N{j:03d}"
                # pct_valid tracks participant quality, as it does in real data
                prf[s][lab] = (ecc[j], cov[j], float(nv[j]),
                               float(100 * qual * min(1.0, nv[j] / 200.0)))
                ctrl[s][lab] = float(y[j])
        return prf, ctrl

    def get(res, model, term):
        return next(r for r in res["rows"] if r["model"] == model
                    and r["term"] == term)

    # --- estimator sanity -----------------------------------------------------
    rng = np.random.default_rng(0)
    a = rng.normal(size=200); b = rng.normal(size=200); c = rng.normal(size=200)
    check("partial_corr of independent vectors is near zero",
          abs(partial_corr(a, b, c)) < 0.2)
    check("partial_corr removes a shared covariate",
          abs(partial_corr(a + 3 * c, b + 3 * c, c)) < 0.2)
    check("fz is monotone", fz(0.1) < fz(0.5) < fz(0.9))

    # --- quality indices are computed on the right node set -------------------
    prf, ctrl = synth(n_sub=10, seed0=99)
    g_lo = gradients(prf, ctrl, min_nodes=5, min_vertices=1)
    g_hi = gradients(prf, ctrl, min_nodes=5, min_vertices=40)
    check("raising the vertex bar reduces the node count",
          all(g_hi[s][1]["n_valid_nodes"] < g_lo[s][1]["n_valid_nodes"]
              for s in g_hi))
    check("raising the vertex bar raises mean pct_valid",
          all(g_hi[s][1]["mean_pct_valid"] > g_lo[s][1]["mean_pct_valid"]
              for s in g_hi))
    check("quality indices are finite",
          all(np.isfinite(list(v[1].values())).all() for v in g_lo.values()))

    # --- NEGATIVE CONTROL: gradient independent of quality --------------------
    prf, ctrl = synth(quality_effect=0.0, seed0=1)
    g = gradients(prf, ctrl, 15, 5)
    res = run_association(g, n_perm=2000, seed=1)
    check("negative control: cohort has a real gradient to begin with",
          res["mean_gradient_z"] > 0.15)
    for t in QUALITY_TERMS:
        check(f"negative control: {t} is null", get(res, t, t)["p_permutation"] > 0.05)
    check("negative control: joint model is null too",
          all(get(res, "joint", t)["p_permutation"] > 0.05 for t in QUALITY_TERMS))

    # --- planted quality dependence must be detected --------------------------
    prf, ctrl = synth(quality_effect=2.5, seed0=2)
    g = gradients(prf, ctrl, 15, 5)
    res_q = run_association(g, n_perm=2000, seed=2)
    check("planted quality dependence is detected",
          get(res_q, "mean_pct_valid", "mean_pct_valid")["p_permutation"] < 0.05)
    check("planted quality dependence has the right sign",
          get(res_q, "mean_pct_valid", "mean_pct_valid")["beta"] > 0)

    # --- escalation: a clean cohort must hold up -------------------------------
    prf, ctrl = synth(quality_effect=0.0, seed0=3)
    esc = run_escalation(prf, ctrl, 10, [1, 5, 20, 40], n_perm=2000, seed=3)
    vals = [r["mean_gradient_z"] for r in esc]
    check("escalation: gradient survives every threshold on clean data",
          all(r["p_permutation"] < 0.01 for r in esc))
    check("escalation: gradient is stable across thresholds on clean data",
          (max(vals) - min(vals)) < 0.5 * abs(np.mean(vals)))
    check("escalation: participant count is reported and non-increasing",
          all(esc[i]["n_subjects"] >= esc[i + 1]["n_subjects"]
              for i in range(len(esc) - 1)))

    # --- escalation: an artefact carried by sparse nodes must be killed --------
    prf, ctrl = synth(weak_node_only=True, seed0=4)
    esc_w = run_escalation(prf, ctrl, 10, [1, 40], n_perm=2000, seed=4)
    check("escalation NEGATIVE CONTROL: sparse-node artefact is visible at a low bar",
          esc_w[0]["p_permutation"] < 0.05)
    check("escalation NEGATIVE CONTROL: sparse-node artefact dies at a high bar",
          abs(esc_w[1]["mean_gradient_z"]) < abs(esc_w[0]["mean_gradient_z"]) / 2)

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--metric", default="average_controllability",
                    choices=("average_controllability", "modal_controllability"))
    ap.add_argument("--covariate", default="mean_rfsize_deg",
                    choices=("mean_rfsize_deg", "n_vertices_total"))
    ap.add_argument("--out-csv")
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--thresholds", default="5,10,20,40,80")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for r in ("prf_backbone_csv", "controllability_csv", "out_csv"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required unless --self-test")

    prf = load_prf(a.prf_backbone_csv, a.atlas, a.covariate)
    ctrl = load_ctrl(a.controllability_csv, a.metric)
    grad = gradients(prf, ctrl, a.min_valid_nodes, a.min_vertices_per_node)
    res = run_association(grad, a.n_permutations, a.seed)
    if res is None:
        sys.exit("too few participants with a usable gradient")
    thr = [int(t) for t in a.thresholds.split(",")]
    esc = run_escalation(prf, ctrl, a.min_valid_nodes, thr, a.n_permutations,
                         a.seed + 1)

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "model", "term", "n_subjects", "value", "p_permutation"])
        for r in res["rows"]:
            w.writerow(["association", r["model"], r["term"], r["n_subjects"],
                        f"{r['beta']:.6f}", f"{r['p_permutation']:.6f}"])
        for r in esc:
            w.writerow(["escalation", f"min_vertices={r['min_vertices_per_node']}",
                        "mean_gradient_z", r["n_subjects"],
                        f"{r['mean_gradient_z']:.6f}", f"{r['p_permutation']:.6f}"])

    print(f"atlas = {a.atlas}   metric = {a.metric}   covariate = {a.covariate}")
    print(f"  participants {res['n']}, mean gradient (Fisher z) "
          f"{res['mean_gradient_z']:+.4f}\n")
    print("  observed quality range in this cohort")
    for t, (lo, hi) in res["quality_ranges"].items():
        print(f"    {t:26s} {lo:9.3f} to {hi:9.3f}")
    print("\n  correlations among the quality indices")
    for i, t in enumerate(QUALITY_TERMS):
        print("    " + f"{t:26s}" +
              " ".join(f"{res['quality_corr'][i, j]:+.3f}"
                       for j in range(len(QUALITY_TERMS))))
    print("\n  arm 1, gradient regressed on data quality")
    for r in res["rows"]:
        print(f"    {r['model']:24s} {r['term']:26s} beta = {r['beta']:+.5f}   "
              f"p = {r['p_permutation']:.4f}")
    print("\n  arm 2, threshold escalation")
    for r in esc:
        print(f"    min_vertices >= {r['min_vertices_per_node']:3d}   "
              f"n = {r['n_subjects']:3d}   mean z = {r['mean_gradient_z']:+.4f}   "
              f"p = {r['p_permutation']:.4f}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
