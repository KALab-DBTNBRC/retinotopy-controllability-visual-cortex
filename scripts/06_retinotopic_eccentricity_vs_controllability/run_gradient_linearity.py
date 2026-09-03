#!/usr/bin/env python3
"""
Is the eccentricity-controllability gradient linear, or only monotonic?

WHY THIS EXISTS
---------------
The manuscript estimates the eccentricity-controllability relationship with a
partial correlation, which assumes linearity. Binned within-participant plots
show the relationship is monotonic across the eccentricity range, but monotonic
is not the same as linear. This matters in visual cortex specifically: both
cortical magnification and pRF size scale nonlinearly with eccentricity, so a
foveal plateau or a peripheral acceleration is a live possibility rather than
an idle worry.

If the relationship is meaningfully nonlinear, a linear partial correlation
UNDERSTATES it, and the shape itself is biologically informative.

WHAT THIS DOES
--------------
Per participant, after residualising on the size covariate:

  1. fits eccentricity -> average controllability with a straight line;
  2. refits with a restricted cubic spline (Harrell's natural-spline basis,
     linear beyond the boundary knots, knots at eccentricity quantiles);
  3. records the improvement in explained variance, delta R-squared.

The group-level question is whether that improvement is larger than the
improvement the extra spline parameters would buy on data with no nonlinear
structure. The null is built per participant by refitting the straight line and
permuting its residuals, which preserves the linear trend exactly while
destroying any systematic curvature. This prices in the flexibility of the
spline without assuming a chi-square approximation. (Permuting eccentricity
instead would be wrong: it destroys the linear trend as well, leaving the null
model freer than the observed fit.)

The test is two-sided, so READ THE SIGN. A positive excess means genuine
curvature. A NEGATIVE excess means the spline gains less on the real data than
on matched iid-noise data, i.e. the residuals are rougher than white noise; that
is not evidence of nonlinearity and must not be reported as such.

Also reports the mean fitted shape on a common eccentricity grid, so a
plateau or acceleration can be seen rather than inferred from a p-value.

USAGE
-----
    python3 run_gradient_linearity.py \
        --prf-backbone-csv <PRF.csv> \
        --controllability-csv <CTRL.csv> \
        --atlas glasser --metric average_controllability \
        --out-csv <OUT.csv> --out-shape-csv <SHAPE.csv>

    python3 run_gradient_linearity.py --self-test
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# spline basis
# ----------------------------------------------------------------------------

def rcs_basis(x, knots):
    """Restricted (natural) cubic spline basis, Harrell's parameterisation.

    Returns k-2 columns for k knots. The basis is linear beyond the boundary
    knots, which is what keeps the tails from flying off on sparse data --
    the usual failure mode of an unrestricted cubic fit at the extremes of
    eccentricity, where few nodes sit.
    """
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    if k.size < 3:
        raise ValueError("need at least 3 knots")
    kn = k.size
    denom = (k[-1] - k[0]) ** 2
    if denom <= 0:
        raise ValueError("degenerate knot span")
    cols = []
    for j in range(kn - 2):
        def cub(t):
            d = np.maximum(x - t, 0.0)
            return d ** 3
        term = (cub(k[j])
                - cub(k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                + cub(k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / denom)
    return np.column_stack(cols)


def choose_knots(x, n_knots):
    """Knots at quantiles, deduplicated. Returns None if too few unique values."""
    x = np.asarray(x, float)
    qs = np.linspace(0.05, 0.95, n_knots)
    k = np.unique(np.quantile(x, qs))
    if k.size < 3:
        return None
    return k


def r2_of(y, X):
    """R-squared of an OLS fit of y on [1, X]."""
    n = y.size
    Z = np.column_stack([np.ones(n), X]) if X is not None and X.size else np.ones((n, 1))
    try:
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    resid = y - Z @ b
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - (resid ** 2).sum() / ss_tot)


def residualise(v, covars):
    v = np.asarray(v, float)
    if not covars:
        return v - v.mean()
    Z = np.column_stack([np.ones(v.size)] + [np.asarray(c, float) for c in covars])
    try:
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return v - Z @ b


def delta_r2(ecc, y, n_knots):
    """Extra variance explained by the spline over the straight line."""
    kn = choose_knots(ecc, n_knots)
    if kn is None:
        return np.nan
    lin = r2_of(y, ecc.reshape(-1, 1))
    try:
        B = rcs_basis(ecc, kn)
    except ValueError:
        return np.nan
    X = np.column_stack([ecc.reshape(-1, 1), B])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.nan
    spl = r2_of(y, X)
    if not (np.isfinite(lin) and np.isfinite(spl)):
        return np.nan
    return float(spl - lin)


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

def run(prf, ctrl, min_nodes, n_knots, n_perm, seed, grid_n=25):
    rng = np.random.default_rng(seed)
    obs, nulls, shapes = [], [], []
    lo_hi = []
    all_ecc = []

    subs = sorted(set(prf) & set(ctrl))
    for s in subs:
        nodes = sorted(set(prf[s]) & set(ctrl[s]))
        if len(nodes) < min_nodes:
            continue
        ecc = np.array([prf[s][n][0] for n in nodes])
        size = np.array([prf[s][n][1] for n in nodes])
        y = np.array([ctrl[s][n] for n in nodes])
        if not np.all(np.isfinite(size)):
            size = None
        yr = residualise(y, [size] if size is not None else [])
        if yr is None:
            continue

        d = delta_r2(ecc, yr, n_knots)
        if not np.isfinite(d):
            continue
        obs.append(d)
        lo_hi.append((ecc.min(), ecc.max()))

        # Per-subject null. It must preserve the LINEAR trend and destroy only
        # nonlinear structure. Permuting eccentricity would destroy both, which
        # collapses the linear model's R-squared and lets the spline win by
        # fitting noise -- making the null MORE flexible than the observed fit
        # and driving the excess negative. Instead, refit the straight line and
        # permute its residuals, which keeps the line exactly and removes any
        # systematic curvature.
        Zl = np.column_stack([np.ones(ecc.size), ecc])
        try:
            bl, *_ = np.linalg.lstsq(Zl, yr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fit_l = Zl @ bl
        resid_l = yr - fit_l
        dn = []
        for _ in range(n_perm):
            y2 = fit_l + rng.permutation(resid_l)
            dv = delta_r2(ecc, y2, n_knots)
            if np.isfinite(dv):
                dn.append(dv)
        nulls.append(np.mean(dn) if dn else np.nan)

        # Store the fitted spline coefficients and this subject's support. The
        # shape is assembled later on a POPULATION grid: taking the
        # intersection of every subject's own range collapses to whichever
        # participant has the narrowest eccentricity coverage, which produced a
        # meaningless 4.0-4.8 deg window on real data.
        kn = choose_knots(ecc, n_knots)
        if kn is not None:
            try:
                B = rcs_basis(ecc, kn)
                X = np.column_stack([np.ones(ecc.size), ecc, B])
                b, *_ = np.linalg.lstsq(X, yr, rcond=None)
                shapes.append((kn, b, float(ecc.min()), float(ecc.max())))
            except (ValueError, np.linalg.LinAlgError):
                pass
        all_ecc.append(ecc)

    obs = np.array(obs, float)
    nulls = np.array(nulls, float)
    ok = np.isfinite(obs) & np.isfinite(nulls)
    obs, nulls = obs[ok], nulls[ok]
    if obs.size < 3:
        return None

    # group test: paired sign-flip on (observed - null) per participant
    diff = obs - nulls
    signs = rng.choice([-1.0, 1.0], size=(10000, diff.size))
    null_means = (signs * diff).mean(axis=1)
    p = float((np.sum(np.abs(null_means) >= abs(diff.mean())) + 1) / 10001)

    # Average shape on a POPULATION grid (10th-90th percentile of pooled
    # eccentricity). Each subject contributes only where the grid falls inside
    # their own observed range; elsewhere they contribute nan. Curves are
    # aligned by subtracting each subject's own fitted value at the grid
    # midpoint, so subjects with different supports are put on a common
    # reference instead of being centred over different windows.
    shape_out = None
    if shapes and all_ecc:
        pooled = np.concatenate(all_ecc)
        lo, hi = np.quantile(pooled, 0.10), np.quantile(pooled, 0.90)
        if hi > lo:
            grid = np.linspace(lo, hi, grid_n)
            mid = grid[grid_n // 2]
            curves = []
            for kn, b, e_lo, e_hi in shapes:
                try:
                    Bg = rcs_basis(grid, kn)
                    Xg = np.column_stack([np.ones(grid.size), grid, Bg])
                    fit = Xg @ b
                    Bm = rcs_basis(np.array([mid]), kn)
                    ref = float(np.column_stack([[1.0], [mid], Bm]) @ b)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                fit = fit - ref
                fit[(grid < e_lo) | (grid > e_hi)] = np.nan
                curves.append(fit)
            if curves:
                C = np.vstack(curves)
                n_ok = np.sum(np.isfinite(C), axis=0)
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(C, axis=0)
                    sd = np.nanstd(C, axis=0)
                sem = np.where(n_ok > 1, sd / np.sqrt(np.maximum(n_ok, 1)), np.nan)
                mean[n_ok < 5] = np.nan     # do not draw where almost nobody contributes
                shape_out = (grid, mean, sem)

    return {
        "n_subjects": int(obs.size),
        "mean_delta_r2_observed": float(obs.mean()),
        "mean_delta_r2_null": float(nulls.mean()),
        "mean_excess": float(diff.mean()),
        "p_permutation": p,
        "ecc_range_median": (float(np.median([a for a, _ in lo_hi])),
                             float(np.median([b for _, b in lo_hi]))),
        "shape": shape_out,
    }


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    # --- spline basis properties --------------------------------------------
    x = np.linspace(0, 10, 60)
    kn = choose_knots(x, 4)
    B = rcs_basis(x, kn)
    check("rcs basis returns k-2 columns", B.shape[1] == kn.size - 2)
    check("rcs basis is finite everywhere", np.all(np.isfinite(B)))
    # linear beyond boundary knots: second difference ~0 in the outer regions
    outer = x > kn[-1]
    if outer.sum() > 3:
        fit = B[outer, :].sum(axis=1)
        check("rcs basis is linear beyond the upper boundary knot",
              np.allclose(np.diff(fit, 2), 0, atol=1e-8))
    else:
        check("rcs basis is linear beyond the upper boundary knot (skipped)", True)

    # --- a perfectly linear relationship must give delta R2 ~ 0 -------------
    rng = np.random.default_rng(3)
    e = rng.uniform(1, 8, 300)
    y_lin = 2.5 * e + rng.normal(0, 0.3, 300)
    d_lin = delta_r2(e, y_lin - y_lin.mean(), 4)
    check("linear data: spline adds almost nothing", abs(d_lin) < 0.01)

    # --- a strongly nonlinear relationship must give a large delta R2 -------
    y_nl = 3.0 * np.log(e) + rng.normal(0, 0.05, 300)
    d_nl = delta_r2(e, y_nl - y_nl.mean(), 4)
    check("log-shaped data: spline adds substantial variance", d_nl > 0.05)
    check("nonlinear delta exceeds linear delta by a wide margin", d_nl > 20 * abs(d_lin))

    # --- end-to-end: planted linear cohort ----------------------------------
    def make(kind, n_subj=25, n_nodes=30, seed0=0):
        r = np.random.default_rng(seed0)
        prf, ctrl = {}, {}
        for s in range(n_subj):
            sid = f"s{s:03d}"
            labs = [f"N{i}" for i in range(n_nodes)]
            ecc = r.uniform(1, 8, n_nodes)
            size = r.uniform(0.5, 3, n_nodes)
            if kind == "linear":
                y = 0.35 * ecc + r.normal(0, 0.4, n_nodes)
            else:  # saturating: rises then plateaus
                y = 1.2 * np.log(ecc) + r.normal(0, 0.10, n_nodes)
            prf[sid] = {l: (ecc[i], size[i]) for i, l in enumerate(labs)}
            ctrl[sid] = {l: y[i] for i, l in enumerate(labs)}
        return prf, ctrl

    res_lin = run(*make("linear", seed0=11), min_nodes=10, n_knots=4,
                  n_perm=40, seed=11)
    check("linear cohort analysed", res_lin is not None)
    check("NEGATIVE CONTROL: linear cohort shows no excess nonlinearity",
          res_lin["p_permutation"] > 0.05)
    check("NEGATIVE CONTROL: linear excess is near zero",
          abs(res_lin["mean_excess"]) < 0.02)

    res_nl = run(*make("saturating", seed0=12), min_nodes=10, n_knots=4,
                 n_perm=40, seed=12)
    check("saturating cohort detected as nonlinear",
          res_nl["p_permutation"] < 0.01)
    check("saturating excess is clearly positive", res_nl["mean_excess"] > 0.02)
    check("the two cohorts are distinguished",
          res_nl["mean_excess"] > 5 * abs(res_lin["mean_excess"]))

    # --- the null must price in spline flexibility --------------------------
    check("null delta R2 is positive (flexibility costs variance)",
          res_lin["mean_delta_r2_null"] > 0)
    check("observed and null are comparable under linearity",
          abs(res_lin["mean_delta_r2_observed"] - res_lin["mean_delta_r2_null"]) < 0.02)

    # --- shape output --------------------------------------------------------
    check("fitted shape returned for the nonlinear cohort",
          res_nl["shape"] is not None and len(res_nl["shape"][0]) > 5)
    if res_nl["shape"] is not None:
        g, f, _ = res_nl["shape"]
        check("recovered shape is monotone increasing for a log relationship",
              np.all(np.diff(f) > -1e-6))
        # a log curve is concave: early slope steeper than late slope
        half = len(f) // 2
        check("recovered shape is concave (early slope steeper than late)",
              (f[half] - f[0]) > (f[-1] - f[half]))

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv")
    ap.add_argument("--controllability-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--metric", default="average_controllability",
                    choices=("average_controllability", "modal_controllability"))
    ap.add_argument("--out-csv")
    ap.add_argument("--out-shape-csv", default=None)
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-knots", type=int, default=4)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for req in ("prf_backbone_csv", "controllability_csv", "out_csv"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_','-')} is required unless --self-test")

    prf = load_prf(a.prf_backbone_csv, a.atlas, a.min_vertices_per_node)
    ctrl = load_ctrl(a.controllability_csv, a.metric)
    res = run(prf, ctrl, a.min_valid_nodes, a.n_knots, a.n_permutations, a.seed)
    if res is None:
        sys.exit("too few participants with usable data")

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["atlas", "metric", "n_subjects", "n_knots",
                    "mean_delta_r2_observed", "mean_delta_r2_null",
                    "mean_excess", "p_permutation", "direction"])
        w.writerow([a.atlas, a.metric, res["n_subjects"], a.n_knots,
                    f"{res['mean_delta_r2_observed']:.6f}",
                    f"{res['mean_delta_r2_null']:.6f}",
                    f"{res['mean_excess']:.6f}",
                    f"{res['p_permutation']:.6f}",
                    "nonlinear" if res["mean_excess"] > 0 else "rougher_than_noise"])

    if a.out_shape_csv and res["shape"] is not None:
        g, f, se = res["shape"]
        with open(a.out_shape_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["eccentricity_deg", "mean_centred_fit", "sem"])
            for i in range(len(g)):
                w.writerow([f"{g[i]:.4f}", f"{f[i]:.6f}", f"{se[i]:.6f}"])

    ex, p = res["mean_excess"], res["p_permutation"]
    if p >= 0.05:
        verdict = "no evidence of departure from linearity"
    elif ex > 0:
        verdict = "NONLINEAR: spline explains more than its own flexibility buys"
    else:
        verdict = ("NOT nonlinear: spline explains LESS than on matched iid-noise "
                   "data, i.e. residuals are rougher than white noise. A "
                   "significant p here does NOT indicate curvature.")
    print(f"atlas={a.atlas}  metric={a.metric}  n={res['n_subjects']}")
    print(f"  observed delta R2 = {res['mean_delta_r2_observed']:.4f}")
    print(f"  null     delta R2 = {res['mean_delta_r2_null']:.4f}   (spline flexibility)")
    print(f"  excess            = {ex:+.4f}   p = {p:.4f}  (two-sided)")
    print(f"  verdict           = {verdict}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
