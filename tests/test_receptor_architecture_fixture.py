#!/usr/bin/env python3
"""Fixture tests for receptor_architecture_analysis.py, run before trusting
it on real data -- same discipline as every other statistical script in
this project. Covers: Mantel test calibration (null + real-signal cases),
hierarchical clustering recovery of known block structure, PCA recovery of
a dominant shared factor, and E/I index sign correctness."""
import numpy as np
from receptor_architecture_analysis import (
    fingerprint_similarity_matrix, hierarchical_cluster, mantel_test,
    run_pca, ei_balance_index, zscore_columns, RECEPTOR_ORDER,
)

rng = np.random.default_rng(0)


def test_mantel_detects_real_signal():
    n = 8
    base = rng.normal(size=(n, n))
    mat_a = (base + base.T) / 2
    np.fill_diagonal(mat_a, 1.0)
    # mat_b built to correlate strongly with mat_a
    mat_b = mat_a + rng.normal(scale=0.2, size=(n, n))
    mat_b = (mat_b + mat_b.T) / 2
    np.fill_diagonal(mat_b, 1.0)
    r, p, null = mantel_test(mat_a, mat_b, 2000, rng)
    assert r > 0.6, f"expected strong positive Mantel r, got {r}"
    assert p < 0.01, f"expected significant p, got {p}"
    print(f"test_mantel_detects_real_signal: r={r:.3f}, p={p:.4g} -- PASS")


def test_mantel_null_calibration():
    """300 independent-matrix replicates: false positive rate at alpha=0.05
    should be near 5%, not inflated -- same calibration-check spirit as
    Claim 4's calibrate_claim4_interaction.py."""
    n = 8
    n_reps = 300
    sig_count = 0
    local_rng = np.random.default_rng(1)
    for _ in range(n_reps):
        a = local_rng.normal(size=(n, n)); a = (a + a.T) / 2; np.fill_diagonal(a, 1.0)
        b = local_rng.normal(size=(n, n)); b = (b + b.T) / 2; np.fill_diagonal(b, 1.0)
        _, p, _ = mantel_test(a, b, 500, local_rng)
        if p < 0.05:
            sig_count += 1
    rate = sig_count / n_reps
    print(f"test_mantel_null_calibration: false-positive rate = {rate:.3f} (expected ~0.05) -- "
          f"{'PASS' if 0.01 < rate < 0.12 else 'FAIL'}")
    assert 0.01 < rate < 0.12, f"Mantel test false-positive rate {rate} looks miscalibrated"


def test_clustering_recovers_two_blocks():
    # 4 units strongly resembling profile A, 4 units strongly resembling profile B
    profile_a = rng.normal(size=15)
    profile_b = rng.normal(size=15)
    M = np.vstack([
        profile_a + rng.normal(scale=0.1, size=15) for _ in range(4)
    ] + [
        profile_b + rng.normal(scale=0.1, size=15) for _ in range(4)
    ])
    sim = fingerprint_similarity_matrix(M)
    _, clusters = hierarchical_cluster(sim, k=2)
    # first 4 should all share one cluster label, last 4 the other
    first_half_consistent = len(set(clusters[:4])) == 1
    second_half_consistent = len(set(clusters[4:])) == 1
    different_clusters = clusters[0] != clusters[4]
    print(f"test_clustering_recovers_two_blocks: clusters={clusters.tolist()} -- "
          f"{'PASS' if first_half_consistent and second_half_consistent and different_clusters else 'FAIL'}")
    assert first_half_consistent and second_half_consistent and different_clusters


def test_pca_recovers_dominant_shared_factor():
    # 15 "receptors" mostly driven by one shared latent factor across 8 units, plus noise
    latent = rng.normal(size=8)
    M = np.column_stack([latent * rng.uniform(0.5, 1.5) + rng.normal(scale=0.05, size=8) for _ in range(15)])
    M_z = zscore_columns(M)
    scores, loadings, explained_var = run_pca(M_z)
    print(f"test_pca_recovers_dominant_shared_factor: PC1 explained_var={explained_var[0]:.3f} -- "
          f"{'PASS' if explained_var[0] > 0.8 else 'FAIL'}")
    assert explained_var[0] > 0.8, f"expected PC1 to dominate with a strong shared factor, got {explained_var[0]}"
    # PC1 score should correlate strongly with the true latent factor (sign-agnostic)
    r = abs(np.corrcoef(scores[:, 0], latent)[0, 1])
    assert r > 0.9, f"PC1 scores should track the true latent factor, r={r}"


def test_ei_index_sign():
    # unit 0: all excitatory receptors high, inhibitory low -> positive log ratio
    # unit 1: reverse -> negative log ratio
    M = np.zeros((2, 15))
    from receptor_architecture_analysis import EXCITATORY, INHIBITORY
    for j, r in enumerate(RECEPTOR_ORDER):
        if r in EXCITATORY:
            M[0, j] = 100.0; M[1, j] = 10.0
        else:
            M[0, j] = 10.0; M[1, j] = 100.0
    ei = ei_balance_index(M)
    print(f"test_ei_index_sign: unit0={ei[0]:.3f} (expect >0), unit1={ei[1]:.3f} (expect <0) -- "
          f"{'PASS' if ei[0] > 0 and ei[1] < 0 else 'FAIL'}")
    assert ei[0] > 0 and ei[1] < 0


if __name__ == "__main__":
    test_mantel_detects_real_signal()
    test_mantel_null_calibration()
    test_clustering_recovers_two_blocks()
    test_pca_recovers_dominant_shared_factor()
    test_ei_index_sign()
    print("\nAll fixture tests passed.")
