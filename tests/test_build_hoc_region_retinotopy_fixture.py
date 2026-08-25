#!/usr/bin/env python3
"""Fixture test for build_hoc_region_retinotopy.py before trusting it on
real data -- specifically the weighted circular mean for polar angle
(wraparound risk) and the renormalize-across-real-coverage logic (nodes
missing retinotopy data shouldn't be treated as zero)."""
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "build_hoc_region_retinotopy.py"


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_wraparound_and_missing_coverage():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        overlap_csv = d / "overlap.csv"
        prf_csv = d / "prf.csv"
        out_csv = d / "out.csv"

        # hOc1_left overlaps 3 backbone nodes for subject S1: two have real retinotopy
        # data with angles straddling the 0/360 wraparound (350, 10 -> circular mean
        # should land near 0, NOT the naive arithmetic mean of 180), the third has NO
        # retinotopy data at all (should be excluded, weight renormalized, not treated
        # as angle=0/ecc=0).
        write_csv(overlap_csv, [
            {"subject": "S1", "hemisphere": "L", "hoc_name": "hOc1_left", "atlas": "schaefer200",
             "backbone_label": "node_A", "n_hoc_vertices": 900, "n_overlap": 10, "weight": 0.4},
            {"subject": "S1", "hemisphere": "L", "hoc_name": "hOc1_left", "atlas": "schaefer200",
             "backbone_label": "node_B", "n_hoc_vertices": 900, "n_overlap": 10, "weight": 0.4},
            {"subject": "S1", "hemisphere": "L", "hoc_name": "hOc1_left", "atlas": "schaefer200",
             "backbone_label": "node_C_missing", "n_hoc_vertices": 900, "n_overlap": 5, "weight": 0.2},
        ], ["subject", "hemisphere", "hoc_name", "atlas", "backbone_label", "n_hoc_vertices", "n_overlap", "weight"])

        write_csv(prf_csv, [
            {"subject": "S1", "atlas": "schaefer200", "hemisphere": "L", "backbone_label": "node_A",
             "mean_eccentricity_deg": "5.0", "mean_polar_angle_deg": "350"},
            {"subject": "S1", "atlas": "schaefer200", "hemisphere": "L", "backbone_label": "node_B",
             "mean_eccentricity_deg": "7.0", "mean_polar_angle_deg": "10"},
            # node_C_missing intentionally absent from the retinotopy file (real zero-coverage pattern)
        ], ["subject", "atlas", "hemisphere", "backbone_label", "mean_eccentricity_deg", "mean_polar_angle_deg"])

        subprocess.run([sys.executable, str(SCRIPT), "--overlap-csv", str(overlap_csv),
                         "--prf-backbone-csv", str(prf_csv), "--atlas", "schaefer200",
                         "--out-csv", str(out_csv)], check=True, cwd=Path(__file__).parent)

        rows = list(csv.DictReader(open(out_csv)))
        assert len(rows) == 1, f"expected 1 output row, got {len(rows)}"
        r = rows[0]
        ecc = float(r["mean_eccentricity_deg"])
        ang = float(r["mean_polar_angle_deg"])
        resultant_r = float(r["mean_resultant_r"])
        n_used = int(r["n_backbone_nodes_used"])

        # eccentricity: renormalized weighted mean over node_A/node_B only (equal weights 0.4/0.4 -> 0.5/0.5)
        expected_ecc = 0.5 * 5.0 + 0.5 * 7.0
        assert abs(ecc - expected_ecc) < 1e-6, f"expected ecc={expected_ecc}, got {ecc}"

        # angle: circular mean of 350 and 10 (equal weight) should be near 0 (or 360), NOT 180
        ang_diff_from_zero = min(ang, 360 - ang)
        print(f"test_wraparound_and_missing_coverage: ecc={ecc:.2f} (expect {expected_ecc}), "
              f"angle={ang:.2f} (expect near 0, NOT 180), resultant_r={resultant_r:.3f}, n_used={n_used} -- "
              f"{'PASS' if ang_diff_from_zero < 5 and n_used == 2 else 'FAIL'}")
        assert ang_diff_from_zero < 5, f"circular mean wrong: got {ang}, expected near 0/360"
        assert n_used == 2, f"expected node_C_missing excluded, n_used={n_used}"
        assert resultant_r > 0.9, f"two close angles should give high concentration, got {resultant_r}"


if __name__ == "__main__":
    test_wraparound_and_missing_coverage()
    print("\nAll fixture tests passed.")
