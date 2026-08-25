#!/usr/bin/env python3
"""Calibration check for run_claim4_analysis.py's sex x receptor interaction
permutation test: under a TRUE null (no real sex-dependent slope difference
for either sex, receptor density freshly redrawn each replicate so no single
draw's chance collinearity can bias the result), what fraction of replicates
falsely reach perm_p < 0.05? Should be close to 0.05 if the test is valid.

Real run (2026-08-12, 300 replicates, N_PERM=1000): 10/300 = 3.3% false
positive rate at alpha=0.05 (expected ~5%, slightly conservative), KS test
vs Uniform(0,1) p=0.52 (no evidence of miscalibration). Kept in the project
as a standing sanity check -- rerun if run_claim4_analysis.py's interaction
test logic is ever modified.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_claim4_analysis import run_sex_receptor_interaction, UNIT_ORDER

N_REPLICATES = 300
N_SUBJECTS = 46
N_PERM = 1000  # smaller than production (10000) purely for calibration speed

rng_master = np.random.default_rng(99)
false_positives = 0
p_values = []

for rep in range(N_REPLICATES):
    rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
    subjects = [f"c{rep}_{i}" for i in range(N_SUBJECTS)]
    sex_map = {s: (1 if i % 2 == 0 else 0) for i, s in enumerate(subjects)}

    receptor_density = rng.uniform(10, 50, size=8)  # fresh draw every replicate
    size_vec = rng.uniform(700, 1100, size=8)
    # TRUE NULL: same slope (0.02) for both sexes -- no real interaction
    subject_rows = {}
    for s in subjects:
        base = 5.0 + rng.normal(0, 0.3, 8)
        y = base + 0.02 * receptor_density + rng.normal(0, 0.15, 8)
        subject_rows[s] = [(receptor_density[i], size_vec[i], y[i]) for i in range(8)]

    stat = run_sex_receptor_interaction(subject_rows, sex_map, N_PERM, rng)
    p = stat.get("perm_p_value")
    if p is not None:
        p_values.append(p)
        if p < 0.05:
            false_positives += 1

print(f"{N_REPLICATES} null replicates, {len(p_values)} valid")
print(f"False positive rate at p<0.05: {false_positives}/{len(p_values)} = {false_positives/len(p_values):.4f}")
print(f"(expected ~0.05 if well-calibrated)")

ks_stat, ks_p = stats.kstest(p_values, "uniform")
print(f"KS test vs Uniform(0,1): stat={ks_stat:.4f}, p={ks_p:.4f} (p>0.05 supports uniformity/good calibration)")
