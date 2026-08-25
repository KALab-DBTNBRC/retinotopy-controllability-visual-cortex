#!/usr/bin/env python3
"""
Retrospective power / minimum-detectable-effect-size (MDES) analysis for
the project's standard subject-level Pearson-correlation designs (Claim 3,
both personality sub-analyses) -- the "is the null informative or just
underpowered" question raised for these findings.

These designs are NOT the small-spatial-unit permutation tests (Claims
1/2/4, see run_claim1_power_analysis.py for that separate Monte Carlo
approach) -- they're a standard cross-subject correlation of two scalars
per subject (e.g. mean controllability vs. a behavioral score), for which
the classical closed-form power formula for testing H0: rho=0 applies
directly (Cohen, 1988, "Statistical Power Analysis for the Behavioral
Sciences", Ch. 3) -- no simulation needed, no custom permutation machinery
to validate, this is textbook statistics:

  Under H1: true correlation = r, sample size n, two-tailed alpha:
    z_r = arctanh(r)                      (Fisher z-transform of the effect)
    se  = 1 / sqrt(n - 3)                 (standard error of z under H1)
    power = Phi(z_r/se - z_(1-alpha/2)) + Phi(-z_r/se - z_(1-alpha/2))
  (second term is the opposite-tail false-negative-avoiding term, standard
  in the full two-tailed derivation, though negligible for n/r combinations
  where the first term already dominates)

Sanity-checked in the fixture test against Cohen's own published benchmark
(r=.30, alpha=.05 two-tailed, n=84 -> power ~=.80, the commonly-cited
"medium effect, 84 subjects" rule of thumb) before trusting this on real
results.

Two things computed per test:
  1. Achieved power to detect a SMALL (r=.10), MEDIUM (r=.30), and LARGE
     (r=.50) effect (Cohen's conventional benchmarks) at the REAL n used.
  2. Minimum detectable r (MDES) at 80% power and the real n -- the
     smallest true effect this specific test could have reliably found.

Usage:
    python3 power_analysis_correlation.py \
        --claim3-schaefer claim3_results_schaefer200.csv \
        --claim3-glasser claim3_results_glasser.csv \
        --personality-mean personality_mean_controllability.csv \
        --personality-coupling personality_coupling_strength.csv \
        --out-csv power_analysis_correlation_results.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats


def power_for_r(r, n, alpha=0.05):
    """Power to detect a true correlation of magnitude r at sample size n,
    two-tailed test at the given alpha. Returns NaN if n <= 3 (Fisher z's
    standard error is undefined)."""
    if n <= 3:
        return float("nan")
    z_r = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    power = stats.norm.cdf(z_r / se - z_crit) + stats.norm.cdf(-z_r / se - z_crit)
    return float(np.clip(power, 0, 1))


def min_detectable_r(n, power=0.80, alpha=0.05):
    """Minimum true |r| detectable at the given power/alpha/n, via bisection
    on power_for_r (monotonically increasing in r for r >= 0, so bisection
    is safe and exact to the tolerance below)."""
    if n <= 3:
        return float("nan")
    lo, hi = 1e-6, 0.999999
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_for_r(mid, n, alpha) < power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def required_n(r, power=0.80, alpha=0.05, n_max=100000):
    """Minimum n to reach the given power for a true effect of magnitude r,
    via linear search from n=4 (smallest n where the formula is defined)."""
    r = abs(r)
    if r <= 1e-9:
        return float("inf")
    for n in range(4, n_max):
        if power_for_r(r, n, alpha) >= power:
            return n
    return float("inf")


def annotate_row(n, observed_r=None, alpha=0.05):
    row = {
        "n": n,
        "power_small_r0.10": power_for_r(0.10, n, alpha),
        "power_medium_r0.30": power_for_r(0.30, n, alpha),
        "power_large_r0.50": power_for_r(0.50, n, alpha),
        "min_detectable_r_at_80pct_power": min_detectable_r(n, 0.80, alpha),
    }
    if observed_r is not None:
        row["observed_r"] = observed_r
        row["achieved_power_at_observed_r"] = power_for_r(abs(observed_r), n, alpha)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim3-schaefer")
    ap.add_argument("--claim3-glasser")
    ap.add_argument("--personality-mean")
    ap.add_argument("--personality-coupling")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []

    def add_from_csv(path, source_label, n_col, r_col, extra_cols):
        if not path or not Path(path).exists():
            return
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get(n_col) or not r.get(r_col):
                    continue
                try:
                    n = int(float(r[n_col]))
                    observed_r = float(r[r_col])
                except ValueError:
                    continue
                row = {"source": source_label}
                for c in extra_cols:
                    row[c] = r.get(c, "")
                row.update(annotate_row(n, observed_r))
                rows.append(row)

    add_from_csv(args.claim3_schaefer, "claim3_schaefer200", "n_subjects", "pearson_r",
                 ["controllability_summary", "behavioral_measure"])
    add_from_csv(args.claim3_glasser, "claim3_glasser", "n_subjects", "pearson_r",
                 ["controllability_summary", "behavioral_measure"])
    add_from_csv(args.personality_mean, "personality_mean_controllability", "n_subjects", "pearson_r",
                 ["controllability_summary", "personality_trait"])
    add_from_csv(args.personality_coupling, "personality_coupling_strength", "n_subjects", "pearson_r",
                 ["atlas", "metric", "personality_trait"])

    fieldnames = ["source", "controllability_summary", "behavioral_measure", "personality_trait",
                  "atlas", "metric", "n", "observed_r", "achieved_power_at_observed_r",
                  "power_small_r0.10", "power_medium_r0.30", "power_large_r0.50",
                  "min_detectable_r_at_80pct_power"]
    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # console summary: group by distinct n (real n's used across this project's null findings)
    distinct_ns = sorted(set(r["n"] for r in rows))
    print("=== Power / MDES summary by real sample size used ===")
    for n in distinct_ns:
        summ = annotate_row(n)
        print(f"n={n}: power(small r=.10)={summ['power_small_r0.10']:.3f}, "
              f"power(medium r=.30)={summ['power_medium_r0.30']:.3f}, "
              f"power(large r=.50)={summ['power_large_r0.50']:.3f}, "
              f"min detectable r @ 80% power = {summ['min_detectable_r_at_80pct_power']:.3f}")

    print(f"\n{len(rows)} rows written to {out_csv}")


if __name__ == "__main__":
    main()
