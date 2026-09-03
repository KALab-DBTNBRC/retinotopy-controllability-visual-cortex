#!/usr/bin/env python3
"""
Does the nonlinear eccentricity-controllability shape survive node-strength
adjustment, and does it hold out of sample?

WHY THIS EXISTS
---------------
`run_gradient_linearity.py` found that the eccentricity-controllability
relationship is nonlinear in Glasser: a restricted cubic spline explains more
variance than a straight line, beyond what the spline's extra flexibility buys
on matched data. That test residualised only on receptive-field size.

Two objections remain, and both are fair.

(1) The curvature might be a connectivity-strength effect. The whole paper
    rests on showing the LINEAR relationship survives adjustment for weighted
    node degree. The nonlinear shape was never put through the same test. If
    the curve is driven by degree, the biological reading changes completely.

(2) A restricted cubic spline fitted to roughly 20 nodes per participant has
    enough freedom to find curvature in noise. The matched-linear null in the
    original script prices that in for IN-SAMPLE variance explained, but it
    does not show the shape predicts data the model has not seen.

WHAT THIS DOES
--------------
For each participant, and for each of three covariate sets:

    none        eccentricity only
    size        + receptive-field size            (the original specification)
    size+degree + receptive-field size and weighted node degree

it fits a straight line and a restricted cubic spline in eccentricity, with the
covariates entering BOTH models linearly, and records:

  (a) in-sample delta R-squared, tested against a null that permutes the
      residuals of the linear model so the linear trend and the covariate
      effects are preserved and only curvature is destroyed;

  (b) leave-one-node-out cross-validated prediction error for the linear and
      spline models, so the spline has to earn its flexibility out of sample.

Reading it:
    curvature survives the degree covariate AND wins on cross-validation
        -> the nonlinear shape is a property of the topography, not of
           connection strength, and is not spline overfitting
    curvature disappears under the degree covariate
        -> the shape is a connectivity-strength effect and must be reported
           as such
    curvature survives in-sample but loses on cross-validation
        -> the spline is fitting noise; report the linear summary only

USAGE
-----
    python3 run_gradient_linearity_adjusted.py \
        --prf-backbone-csv <PRF.csv> \
        --controllability-csv <CTRL.csv> \
        --node-strength-csv <STRENGTH.csv> \
        --atlas glasser --out-csv <OUT.csv>

    python3 run_gradient_linearity_adjusted.py --self-test
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# spline basis (identical parameterisation to run_gradient_linearity.py)
# ----------------------------------------------------------------------------

def rcs_basis(x, knots):
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    if k.size < 3:
        raise ValueError("need at least 3 knots")
    denom = (k[-1] - k[0]) ** 2
    if denom <= 0:
        raise ValueError("degenerate knot span")
    cols = []
    for j in range(k.size - 2):
        def cub(t):
            return np.maximum(x - t, 0.0) ** 3
        term = (cub(k[j])
                - cub(k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                + cub(k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / denom)
    return np.column_stack(cols)


def choose_knots(x, n_knots):
    k = np.unique(np.quantile(np.asarray(x, float), np.linspace(0.05, 0.95, n_knots)))
    return k if k.size >= 3 else None


def _ols_fit(X, y):
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return b


def _r2(y, X):
    b = _ols_fit(X, y)
    if b is None:
        return np.nan
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ((y - X @ b) ** 2).sum() / ss_tot)


def design(ecc, covars, knots=None):
    """[1, covars..., ecc] and optionally the spline columns."""
    n = ecc.size
    cols = [np.ones(n)] + [np.asarray(c, float) for c in covars] + [ecc]
    if knots is not None:
        cols.append(rcs_basis(ecc, knots))
    return np.column_stack(cols)


def delta_r2_adj(ecc, y, covars, n_knots):
    kn = choose_knots(ecc, n_knots)
    if kn is None:
        return np.nan
    Xl = design(ecc, covars)
    try:
        Xs = design(ecc, covars, kn)
    except ValueError:
        return np.nan
    if np.linalg.matrix_rank(Xs) < Xs.shape[1] or Xs.shape[0] <= Xs.shape[1] + 1:
        return np.nan
    lin, spl = _r2(y, Xl), _r2(y, Xs)
    if not (np.isfinite(lin) and np.isfinite(spl)):
        return np.nan
    return float(spl - lin)


def loo_cv_error(ecc, y, covars, n_knots):
    """Leave-one-node-out prediction error for the linear and spline models.

    Returns (mse_linear, mse_spline). The spline only wins if the curvature
    generalises to nodes the fit has not seen.
    """
    kn = choose_knots(ecc, n_knots)
    if kn is None:
        return np.nan, np.nan
    try:
        Xl = design(ecc, covars)
        Xs = design(ecc, covars, kn)
    except ValueError:
        return np.nan, np.nan
    n = ecc.size
    if n <= Xs.shape[1] + 2:
        return np.nan, np.nan
    el, es = [], []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        bl = _ols_fit(Xl[m], y[m])
        bs = _ols_fit(Xs[m], y[m])
        if bl is None or bs is None:
            continue
        el.append((y[i] - Xl[i] @ bl) ** 2)
        es.append((y[i] - Xs[i] @ bs) ** 2)
    if not el:
        return np.nan, np.nan
    return float(np.mean(el)), float(np.mean(es))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_prf(path, atlas, min_vertices):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("atlas") != atlas:
                continue
            try:
                if float(row.get("n_vertices_valid_R2gt30") or 0) < min_vertices:
                    continue
                ecc = float(row["mean_eccentricity_deg"])
                size = float(row.get("mean_rfsize_deg") or "nan")
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(ecc):
                out[row["subject"]][row["backbone_label"]] = (ecc, size)
    return out


def load_ctrl(path):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        lab = next((c for c in rd.fieldnames if "label" in c.lower()), None)
        for row in rd:
            try:
                v = float(row["average_controllability"])
            except (ValueError, KeyError, TypeError):
                continue
            if v > 0:
                out[row["subject"]][row[lab]] = np.log(v)
    return out


def load_strength(path):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["subject"]][row["node_label"]] = float(row["weighted_degree"])
            except (ValueError, KeyError, TypeError):
                continue
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

COVSETS = [("none", []), ("size", ["size"]), ("size+degree", ["size", "degree"])]


def run(prf, ctrl, strength, min_nodes, n_knots, n_perm, seed):
    rng = np.random.default_rng(seed)
    acc = {k: {"obs": [], "null": [], "cvl": [], "cvs": []} for k, _ in COVSETS}

    for s in sorted(set(prf) & set(ctrl) & set(strength)):
        nodes = sorted(set(prf[s]) & set(ctrl[s]) & set(strength[s]))
        if len(nodes) < min_nodes:
            continue
        ecc = np.array([prf[s][n][0] for n in nodes])
        size = np.array([prf[s][n][1] for n in nodes])
        deg = np.array([strength[s][n] for n in nodes])
        y = np.array([ctrl[s][n] for n in nodes])
        if not np.all(np.isfinite(size)):
            continue
        if np.std(deg) > 0:
            deg = np.log(np.maximum(deg, 1e-12))
        vals = {"size": size, "degree": deg}

        for key, names in COVSETS:
            cov = [vals[c] for c in names]
            d = delta_r2_adj(ecc, y, cov, n_knots)
            if not np.isfinite(d):
                continue
            acc[key]["obs"].append(d)

            # null: keep the linear trend AND the covariate effects, destroy curvature
            kn = choose_knots(ecc, n_knots)
            Xl = design(ecc, cov)
            bl = _ols_fit(Xl, y)
            if bl is None or kn is None:
                acc[key]["null"].append(np.nan); continue
            fit = Xl @ bl
            res = y - fit
            dn = []
            for _ in range(n_perm):
                dv = delta_r2_adj(ecc, fit + rng.permutation(res), cov, n_knots)
                if np.isfinite(dv):
                    dn.append(dv)
            acc[key]["null"].append(np.mean(dn) if dn else np.nan)

            ml, ms = loo_cv_error(ecc, y, cov, n_knots)
            acc[key]["cvl"].append(ml); acc[key]["cvs"].append(ms)

    rows = []
    for key, _ in COVSETS:
        o = np.array(acc[key]["obs"], float); nl = np.array(acc[key]["null"], float)
        ok = np.isfinite(o) & np.isfinite(nl)
        o, nl = o[ok], nl[ok]
        if o.size < 3:
            continue
        diff = o - nl
        signs = rng.choice([-1.0, 1.0], size=(10000, diff.size))
        p = float((np.sum(np.abs((signs * diff).mean(axis=1)) >= abs(diff.mean())) + 1) / 10001)

        cl = np.array(acc[key]["cvl"], float); cs = np.array(acc[key]["cvs"], float)
        okc = np.isfinite(cl) & np.isfinite(cs)
        cl, cs = cl[okc], cs[okc]
        if cl.size >= 3:
            cvd = cs - cl                       # negative = spline predicts better
            sg = rng.choice([-1.0, 1.0], size=(10000, cvd.size))
            pcv = float((np.sum(np.abs((sg * cvd).mean(axis=1)) >= abs(cvd.mean())) + 1) / 10001)
            frac = float((cs < cl).mean())
        else:
            cvd, pcv, frac = np.array([np.nan]), np.nan, np.nan

        rows.append({
            "covariates": key, "n_subjects": int(o.size),
            "delta_r2_observed": float(o.mean()),
            "delta_r2_null": float(nl.mean()),
            "excess": float(diff.mean()), "p_excess": p,
            "cv_mse_linear": float(cl.mean()) if cl.size else np.nan,
            "cv_mse_spline": float(cs.mean()) if cs.size else np.nan,
            "cv_diff_spline_minus_linear": float(np.mean(cvd)),
            "p_cv": pcv,
            "frac_subjects_spline_better_oos": frac,
        })
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

    rng = np.random.default_rng(5)
    e = rng.uniform(1, 8, 400)
    check("linear data: spline adds ~nothing with no covariates",
          abs(delta_r2_adj(e, 2.5 * e + rng.normal(0, .3, 400), [], 4)) < 0.02)
    check("log data: spline adds substantial variance",
          delta_r2_adj(e, 3.0 * np.log(e) + rng.normal(0, .05, 400), [], 4) > 0.05)

    # a covariate that carries the curvature must absorb it
    c = np.log(e) + rng.normal(0, .01, 400)
    y = 3.0 * np.log(e) + rng.normal(0, .05, 400)
    d_no = delta_r2_adj(e, y, [], 4)
    d_cov = delta_r2_adj(e, y, [c], 4)
    check("a covariate carrying the curvature absorbs it", d_cov < d_no / 3)

    # a covariate unrelated to the curvature must NOT absorb it
    d_irr = delta_r2_adj(e, y, [rng.normal(size=400)], 4)
    check("an unrelated covariate leaves curvature intact", d_irr > d_no * 0.7)

    # cross-validation: spline should win out of sample on curved data,
    # and lose on linear data where it only adds free parameters
    ml, ms = loo_cv_error(e, 3.0 * np.log(e) + rng.normal(0, .05, 400), [], 4)
    check("CV: spline predicts better on genuinely curved data", ms < ml)
    ml2, ms2 = loo_cv_error(e, 2.5 * e + rng.normal(0, .3, 400), [], 4)
    check("CV: spline does NOT beat the line on linear data", ms2 >= ml2 * 0.98)

    def make(kind, n_subj=25, n_nodes=30, seed0=0, degree_drives=False):
        r = np.random.default_rng(seed0)
        prf, ctrl, stg = {}, {}, {}
        for s in range(n_subj):
            sid = f"s{s:03d}"; labs = [f"N{i}" for i in range(n_nodes)]
            ecc = r.uniform(1, 8, n_nodes)
            size = r.uniform(.5, 3, n_nodes)
            if degree_drives:
                deg = np.exp(1.2 * np.log(ecc) + r.normal(0, .05, n_nodes))
                y = 1.0 * np.log(deg) + r.normal(0, .05, n_nodes)
            else:
                deg = np.exp(r.normal(3, .4, n_nodes))
                y = (1.2 * np.log(ecc) if kind == "curved" else .35 * ecc) \
                    + r.normal(0, .10, n_nodes)
            prf[sid] = {l: (ecc[i], size[i]) for i, l in enumerate(labs)}
            ctrl[sid] = {l: y[i] for i, l in enumerate(labs)}
            stg[sid] = {l: deg[i] for i, l in enumerate(labs)}
        return prf, ctrl, stg

    def get(rows, k):
        return next(r for r in rows if r["covariates"] == k)

    # curvature independent of degree: must survive the degree covariate
    rows = run(*make("curved", seed0=21), min_nodes=10, n_knots=4, n_perm=30, seed=21)
    check("curvature independent of degree survives the degree covariate",
          get(rows, "size+degree")["excess"] > 0.02
          and get(rows, "size+degree")["p_excess"] < 0.05)
    check("and the spline wins out of sample there",
          get(rows, "size+degree")["cv_diff_spline_minus_linear"] < 0)

    # curvature that IS degree: must be absorbed by the degree covariate
    rows2 = run(*make("curved", seed0=22, degree_drives=True),
                min_nodes=10, n_knots=4, n_perm=30, seed=22)
    check("NEGATIVE CONTROL: degree-driven curvature is absorbed by the covariate",
          get(rows2, "size+degree")["excess"] < get(rows2, "size")["excess"] / 2)

    # linear cohort: no curvature under any covariate set
    rows3 = run(*make("linear", seed0=23), min_nodes=10, n_knots=4, n_perm=30, seed=23)
    check("NEGATIVE CONTROL: linear cohort shows no excess under any covariate set",
          all(abs(get(rows3, k)["excess"]) < 0.02 for k, _ in COVSETS))
    check("NEGATIVE CONTROL: spline does not win out of sample on linear data",
          get(rows3, "size")["cv_diff_spline_minus_linear"] > 0)

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--node-strength-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--out-csv")
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-knots", type=int, default=4)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for r in ("prf_backbone_csv", "controllability_csv", "node_strength_csv", "out_csv"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required unless --self-test")

    rows = run(load_prf(a.prf_backbone_csv, a.atlas, a.min_vertices_per_node),
               load_ctrl(a.controllability_csv),
               load_strength(a.node_strength_csv),
               a.min_valid_nodes, a.n_knots, a.n_permutations, a.seed)
    if not rows:
        sys.exit("too few participants with usable data")

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"atlas = {a.atlas}\n")
    for r in rows:
        better = r["frac_subjects_spline_better_oos"]
        print(f"  covariates: {r['covariates']:12s} n={r['n_subjects']}")
        print(f"     in-sample  excess dR2 = {r['excess']:+.4f}   p = {r['p_excess']:.4f}")
        print(f"     out-of-sample MSE     linear {r['cv_mse_linear']:.4f}  "
              f"spline {r['cv_mse_spline']:.4f}  diff {r['cv_diff_spline_minus_linear']:+.4f}  "
              f"p = {r['p_cv']:.4f}")
        print(f"     participants where the spline predicts better: {better*100:.0f}%\n")
    print(f"wrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
