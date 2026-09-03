#!/usr/bin/env python3
"""
Do the receptor data support the null, or are they simply uninformative?

WHY THIS EXISTS
---------------
The manuscript reports that no receptor survived correction against either
controllability metric, and correctly caveats that the eight-unit design has a
minimum detectable effect of r ~ 0.78. But a non-significant p-value cannot
distinguish "evidence of absence" from "absence of evidence". A Bayes factor
can, and that is the quantity the receptor section actually needs.

There is a second, sharper reason to run this. The receptor design has TWO
sample sizes, and they license very different conclusions:

  * 46 participants, each contributing one partial correlation, and
  * 8 spatial units, on which every one of those correlations is computed.

Receptor densities are group-level constants. They do not vary across
participants. So a test that looks like a 46-subject analysis is, at its core,
an 8-point spatial association measured 46 times. Computing the Bayes factor
BOTH ways puts a number on how much that distinction matters, instead of
leaving it as a caveat in prose.

WHAT THIS DOES
--------------
(A) Participant-level: a default JZS one-sample Bayes factor (Rouder et al.,
    2009) on the per-participant Fisher-z values, testing mean z != 0. This is
    the design as analysed.

(B) Spatial-unit-level: an exact default Bayes factor for a single correlation
    at n = 8 (Jeffreys' prior on rho, exact sampling density via the Gaussian
    hypergeometric function), evaluated at the observed mean correlation. This
    is the design as it actually behaves.

Reports BF01 (evidence for the null over the alternative) for both. The
expected and honest outcome is that (B) sits near 1, formally confirming the
data are uninformative at the spatial-unit level rather than supportive of the
null. That is a result worth stating precisely.

Convention used throughout:
    BF01 > 3    moderate evidence FOR the null
    1/3 to 3    inconclusive; the data do not discriminate
    BF01 < 1/3  evidence AGAINST the null

USAGE
-----
    python3 run_receptor_bayes_factors.py \
        --receptor-results-csv <RECEPTOR_RESULTS.csv> \
        --out-csv <OUT.csv>

    python3 run_receptor_bayes_factors.py --self-test
"""

import argparse
import csv
import sys

import numpy as np
from scipy import integrate, special


# ----------------------------------------------------------------------------
# (A) JZS one-sample Bayes factor
# ----------------------------------------------------------------------------

def bf10_jzs_onesample(t, n, rscale=0.707):
    """Default JZS Bayes factor for a one-sample t test (Rouder et al., 2009).

    Returns BF10. t is the one-sample t statistic, n the number of observations.
    rscale 0.707 is the standard "medium" prior width.
    """
    if not np.isfinite(t) or n < 3:
        return np.nan
    nu = n - 1

    def integrand(g):
        if g <= 0:
            return 0.0
        a = (1.0 + n * g) ** -0.5
        b = (1.0 + t * t / ((1.0 + n * g) * nu)) ** (-(nu + 1) / 2.0)
        prior = (rscale ** 2 / (2 * np.pi)) ** 0.5 * g ** -1.5 * \
            np.exp(-rscale ** 2 / (2 * g))
        return a * b * prior

    num, err = integrate.quad(integrand, 1e-10, np.inf, limit=400)
    den = (1.0 + t * t / nu) ** (-(nu + 1) / 2.0)
    if den <= 0 or not np.isfinite(num):
        return np.nan
    return float(num / den)


# ----------------------------------------------------------------------------
# (B) Exact default Bayes factor for a single correlation
# ----------------------------------------------------------------------------

def _log_r_density(r, rho, n):
    """log of the exact sampling density of Pearson r given rho and n."""
    if abs(rho) >= 1 or abs(r) >= 1 or n < 4:
        return -np.inf
    lg = (np.log(n - 2)
          + special.gammaln(n - 1)
          + ((n - 1) / 2.0) * np.log1p(-rho * rho)
          + ((n - 4) / 2.0) * np.log1p(-r * r)
          - 0.5 * np.log(2 * np.pi)
          - special.gammaln(n - 0.5)
          - (n - 1.5) * np.log1p(-rho * r))
    hyp = special.hyp2f1(0.5, 0.5, n - 0.5, (rho * r + 1) / 2.0)
    if not np.isfinite(hyp) or hyp <= 0:
        return -np.inf
    return lg + np.log(hyp)


def bf10_correlation(r, n, kappa=1.0):
    """Default Bayes factor for a correlation, Jeffreys-style prior on rho.

    Prior: stretched beta on (-1, 1) with width kappa. kappa = 1 is the
    uniform prior, i.e. Jeffreys' default. Returns BF10.
    """
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return np.nan

    def marginal(rho):
        lp = _log_r_density(r, rho, n)
        if not np.isfinite(lp):
            return 0.0
        # stretched beta(1/kappa, 1/kappa) density on (-1, 1)
        a = 1.0 / kappa
        lprior = ((a - 1) * np.log1p(rho) + (a - 1) * np.log1p(-rho)
                  - (2 * a - 1) * np.log(2.0)
                  - special.betaln(a, a))
        return np.exp(lp + lprior)

    num, _ = integrate.quad(marginal, -0.9999, 0.9999, limit=400)
    l0 = _log_r_density(r, 0.0, n)
    if not np.isfinite(l0) or num <= 0:
        return np.nan
    return float(num / np.exp(l0))


def interpret(bf01):
    if not np.isfinite(bf01):
        return "undefined"
    if bf01 > 10:
        return "strong evidence for the null"
    if bf01 > 3:
        return "moderate evidence for the null"
    if bf01 > 1 / 3:
        return "inconclusive"
    if bf01 > 1 / 10:
        return "moderate evidence against the null"
    return "strong evidence against the null"


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    # ---- JZS behaves as a Bayes factor should ------------------------------
    bf_null_big = bf10_jzs_onesample(t=0.0, n=200)
    bf_null_small = bf10_jzs_onesample(t=0.0, n=6)
    check("JZS: t=0 at large n gives strong evidence FOR the null",
          1 / bf_null_big > 10)
    check("JZS: t=0 at small n is far less conclusive",
          (1 / bf_null_small) < (1 / bf_null_big) / 3)
    check("JZS: a large t gives evidence AGAINST the null",
          bf10_jzs_onesample(t=6.0, n=46) > 100)
    check("JZS: evidence for the null grows with n at fixed t=0",
          1 / bf10_jzs_onesample(0.0, 200) > 1 / bf10_jzs_onesample(0.0, 50)
          > 1 / bf10_jzs_onesample(0.0, 15))
    check("JZS: BF is monotone increasing in |t|",
          bf10_jzs_onesample(0.5, 46) < bf10_jzs_onesample(2.0, 46)
          < bf10_jzs_onesample(4.0, 46))

    # ---- correlation BF: the key n=8 behaviour ------------------------------
    bf_r0_n8 = bf10_correlation(0.0, 8)
    bf_r0_n200 = bf10_correlation(0.0, 200)
    check("correlation BF: r=0 at n=200 gives strong evidence for the null",
          1 / bf_r0_n200 > 10)
    check("correlation BF: r=0 at n=8 is INCONCLUSIVE (this is the point)",
          1 / 3 < 1 / bf_r0_n8 < 5)
    check("correlation BF: n=8 is far less informative than n=200",
          1 / bf_r0_n8 < (1 / bf_r0_n200) / 3)
    check("correlation BF: a very large r at n=8 still moves the needle",
          bf10_correlation(0.95, 8) > 3)
    check("correlation BF: a moderate r at n=8 does not",
          1 / 3 < bf10_correlation(0.40, 8) < 3)
    check("correlation BF is symmetric in the sign of r",
          np.isclose(bf10_correlation(0.5, 20), bf10_correlation(-0.5, 20), rtol=1e-6))

    # ---- the density integrates to 1 (validates the exact likelihood) ------
    for (rho, n) in [(0.0, 10), (0.5, 12), (-0.3, 8)]:
        val, _ = integrate.quad(lambda rr: np.exp(_log_r_density(rr, rho, n)),
                                -0.99999, 0.99999, limit=400)
        check(f"exact r-density integrates to 1 (rho={rho}, n={n})",
              abs(val - 1.0) < 1e-3)

    # ---- NEGATIVE CONTROL --------------------------------------------------
    # A BF must not claim evidence for the null when the data are merely few.
    # At n=8 with r=0 the BF01 must stay well under the "moderate" threshold
    # that the same r would earn at large n.
    check("NEGATIVE CONTROL: small-n null is not mistaken for support",
          (1 / bf_r0_n8) < 3)
    # And a genuinely informative null must be recognised as such.
    check("POSITIVE CONTROL: large-n null IS recognised as support",
          (1 / bf_r0_n200) > 3)

    # ---- interpretation bands ----------------------------------------------
    check("interpretation: 20 -> strong for null", interpret(20) == "strong evidence for the null")
    check("interpretation: 1 -> inconclusive", interpret(1.0) == "inconclusive")
    check("interpretation: 0.05 -> strong against", interpret(0.05) == "strong evidence against the null")

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--receptor-results-csv",
                    help="per-receptor results with mean partial r and n")
    ap.add_argument("--r-column", default="mean_partial_r")
    ap.add_argument("--n-column", default="n_subjects")
    ap.add_argument("--n-spatial-units", type=int, default=8)
    ap.add_argument("--out-csv")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for req in ("receptor_results_csv", "out_csv"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_','-')} is required unless --self-test")

    rows_out = []
    with open(a.receptor_results_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                r = float(row[a.r_column])
                n = int(float(row[a.n_column]))
            except (ValueError, KeyError, TypeError):
                continue
            if not np.isfinite(r) or abs(r) >= 1:
                continue
            # participant-level: convert mean r to a one-sample t on Fisher z
            z = np.arctanh(np.clip(r, -0.999999, 0.999999))
            se = 1.0 / np.sqrt(max(n - 3, 1))
            t = z / se
            bf10_a = bf10_jzs_onesample(t, n)
            bf10_b = bf10_correlation(r, a.n_spatial_units)
            out = dict(row)
            out["BF01_participant_level"] = 1 / bf10_a if bf10_a else np.nan
            out["BF01_spatial_unit_level"] = 1 / bf10_b if bf10_b else np.nan
            out["interpretation_spatial_unit"] = interpret(out["BF01_spatial_unit_level"])
            rows_out.append(out)

    if not rows_out:
        sys.exit("no usable rows found; check --r-column / --n-column")

    fields = list(rows_out[0].keys())
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    b_sp = np.array([r["BF01_spatial_unit_level"] for r in rows_out], float)
    b_pt = np.array([r["BF01_participant_level"] for r in rows_out], float)
    print(f"tests evaluated: {len(rows_out)}")
    print(f"  BF01 at the participant level  : median {np.nanmedian(b_pt):.2f} "
          f"(range {np.nanmin(b_pt):.2f} to {np.nanmax(b_pt):.2f})")
    print(f"  BF01 at the spatial-unit level : median {np.nanmedian(b_sp):.2f} "
          f"(range {np.nanmin(b_sp):.2f} to {np.nanmax(b_sp):.2f})")
    inconclusive = int(np.sum((b_sp > 1 / 3) & (b_sp < 3)))
    print(f"  spatial-unit tests that are inconclusive (1/3 < BF01 < 3): "
          f"{inconclusive}/{len(b_sp)}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
