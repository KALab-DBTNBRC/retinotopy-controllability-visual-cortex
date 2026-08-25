#!/usr/bin/env python3
"""
siibra_pull.py
===============

Pull Julich-Brain cytoarchitectonic probability maps and receptor-density
fingerprints for the project's visual ROIs, using siibra-python against the
live EBRAINS Knowledge Graph / data proxy.

What I verified vs. what I could not
-------------------------------------
I installed siibra-python (1.0.1-alpha.21) myself and introspected the real,
installed package: the top-level API (get_parcellation/get_region/get_map/
get_template/features.get), the ReceptorDensityFingerprint class (its `.data`
is a pandas DataFrame indexed by receptor symbol with 'mean'/'std' columns,
values in fmol/mg protein), and — importantly — siibra's built-in receptor
vocabulary (`siibra.vocabularies.RECEPTOR_SYMBOLS`), all without hitting the
network. That vocabulary is:

    5-HT1A, 5-HT2, AMPA, BZ, D1, GABAA, GABAB, M1, M2, M3, NMDA,
    alpha1, alpha2, alpha4beta2, kainate, mGluR2_3

There is NO D2 and NO opioid receptor in this list. The Julich-Brain /
Palomero-Gallagher receptor autoradiography dataset simply does not include
dopamine D2 or mu-opioid binding — only D1 for dopamine, and GABAA/GABAB/BZ
for GABA. If D2 or opioid receptor maps matter to your claims, you'd need a
different source (e.g. Hansen et al. 2022's PET-derived neurotransmitter
atlas, which does cover D2 and mu-opioid, but at much coarser spatial
resolution and from a totally different acquisition modality than Julich's
cytoarchitectonic maps — not a drop-in replacement).

What I could NOT verify: the exact Julich-Brain parcellation version string,
the exact region name strings for hOc1-hOc5 (with or without hemisphere
suffixes), and whether LGN/SC/Pulvinar exist as Julich-Brain regions at all.
My sandbox's network blocks gitlab.ebrains.eu (the siibra config source), so
I could not make a single live call to check. This script therefore
*discovers* all of this at runtime on your server (which should have normal
internet access) rather than hardcoding names I couldn't confirm, and prints
a full report of what it found/didn't find. Run --smoke-test first.

Usage
-----
    # 1. Quick check that siibra can reach EBRAINS and resolve one region
    python3 siibra_pull.py --output-dir ./atlas_pull --smoke-test

    # 2. Full pull: probability maps + receptor fingerprints for all ROIs
    python3 siibra_pull.py --output-dir ./atlas_pull

    # Override auto-detected parcellation if it picks the wrong version
    python3 siibra_pull.py --output-dir ./atlas_pull --parcellation "julich-brain cytoarchitectonic maps 3.0.3"

Only siibra + its own dependencies (nibabel, pandas, numpy) are needed —
these come with `pip install siibra`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

# Cortical visual areas (Julich-Brain "hOc" nomenclature) x hemisphere.
CORTICAL_REGIONS = ["hOc1", "hOc2", "hOc3v", "hOc3d", "hOc4v", "hOc4d", "hOc5"]
# Subcortical retinotopic nuclei -- may or may not exist as named Julich-Brain
# regions; the script checks and reports rather than assuming.
SUBCORTICAL_REGIONS = ["LGN", "superior colliculus", "pulvinar"]
HEMISPHERES = ["left", "right"]


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )


def pick_parcellation(siibra, override: str | None):
    """Resolve the Julich-Brain cytoarchitectonic parcellation. Tries the
    direct fuzzy spec first; if that's ambiguous (multiple versions
    installed), lists candidates and picks the lexicographically-last one
    (usually the newest version string) -- override with --parcellation if
    that guess is wrong for your siibra version."""
    if override:
        logging.info("Using user-specified parcellation spec: '%s'", override)
        return siibra.get_parcellation(override)

    try:
        p = siibra.get_parcellation("julich-brain cytoarchitectonic")
        logging.info("Resolved parcellation unambiguously: %s", p.name)
        return p
    except Exception as e:
        logging.warning("Direct spec was ambiguous or failed (%s) -- scanning siibra.parcellations for candidates.", e)

    candidates = []
    for p in siibra.parcellations:
        name_l = p.name.lower()
        if "julich" in name_l and "cyto" in name_l:
            candidates.append(p)
    if not candidates:
        raise RuntimeError(
            "No parcellation with 'julich' and 'cyto' in its name was found in the registry. "
            "Run with --list-parcellations to see everything available and pass the exact "
            "name via --parcellation."
        )
    candidates.sort(key=lambda p: p.name)
    logging.info("Candidates found: %s", [p.name for p in candidates])
    chosen = candidates[-1]
    logging.info("Auto-selected (lexicographically last, usually newest): %s", chosen.name)
    logging.info("If this is wrong, re-run with: --parcellation \"%s\"", candidates[0].name if len(candidates) > 1 else chosen.name)
    return chosen


def find_region(siibra, parcellation, label: str, hemisphere: str | None):
    """Try a few plausible spec phrasings for a region + hemisphere, since I
    could not confirm Julich-Brain's exact naming convention live. Returns
    (region_or_None, spec_that_worked_or_None)."""
    tried = []
    specs = []
    if hemisphere:
        specs += [f"{label} {hemisphere}", f"{label} ({hemisphere})", f"{hemisphere} {label}"]
    specs += [label]  # unlateralized fallback
    for spec in specs:
        tried.append(spec)
        try:
            region = siibra.get_region(parcellation, spec)
            return region, spec
        except Exception:
            continue
    # last resort: fuzzy search, filter by hemisphere keyword if given
    try:
        matches = siibra.find_regions(label)
        if hemisphere:
            matches = [m for m in matches if hemisphere in m.name.lower()]
        if len(matches) == 1:
            return matches[0], f"find_regions('{label}') -> {matches[0].name}"
    except Exception:
        pass
    logging.warning("No match for '%s' (%s). Tried: %s", label, hemisphere or "unlateralized", tried)
    return None, None


def fetch_probability_map(siibra, parcellation, region, space_spec: str, out_path: Path) -> bool:
    try:
        import nibabel as nib
    except ImportError:
        logging.error("nibabel not installed (should have come with siibra) -- cannot save NIfTI maps.")
        return False
    try:
        smap = siibra.get_map(parcellation, space_spec, siibra.MapType.STATISTICAL)
        img = smap.fetch(region=region)
    except Exception as e:
        logging.warning("Could not fetch statistical map for %s: %s", region.name, e)
        return False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(out_path))
        return True
    except Exception as e:
        logging.warning("Fetched map for %s but could not save it: %s", region.name, e)
        return False


def fetch_receptor_fingerprint(siibra, region):
    """Returns a list of (receptor, neurotransmitter, mean, std, unit) tuples,
    or an empty list if no fingerprint is anchored to this region."""
    rows = []
    try:
        features = siibra.features.get(region, siibra.features.molecular.ReceptorDensityFingerprint)
    except Exception as e:
        logging.warning("Receptor fingerprint query failed for %s: %s", region.name, e)
        return rows
    for feat in features:
        try:
            df = feat.data  # index=receptor, columns=['mean','std']
            unit = feat.unit
            for receptor, row in df.iterrows():
                rows.append((receptor, unit, float(row["mean"]), float(row["std"])))
        except Exception as e:
            logging.warning("Could not parse fingerprint data for %s: %s", region.name, e)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--space", default="mni152", help="Reference space to fetch probability maps in (default: mni152, matching HCP's MNINonLinear space)")
    ap.add_argument("--parcellation", default=None, help="Exact parcellation name/spec, if auto-detection picks the wrong version")
    ap.add_argument("--cache-dir", default=None, help="Directory for siibra's download cache (sets SIIBRA_CACHEDIR). Defaults to siibra's own default (~/.siibra).")
    ap.add_argument("--list-parcellations", action="store_true", help="Just list every Julich-related parcellation found and exit (for picking --parcellation).")
    ap.add_argument("--smoke-test", action="store_true", help="Resolve the parcellation and ONE region (hOc1 left) only, print the result, and exit. Run this first.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "siibra_pull.log")

    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["SIIBRA_CACHEDIR"] = str(Path(args.cache_dir).resolve())
        logging.info("SIIBRA_CACHEDIR set to %s", os.environ["SIIBRA_CACHEDIR"])

    logging.info("Importing siibra (this triggers a network call to fetch the EBRAINS config registry)...")
    try:
        import siibra
    except Exception as e:
        logging.error("Failed to import siibra: %s", e)
        sys.exit(1)

    try:
        parcellation = pick_parcellation(siibra, args.parcellation)
    except Exception as e:
        logging.error("Could not resolve a Julich-Brain parcellation: %s", e)
        logging.error("Try: python3 %s --output-dir %s --list-parcellations", sys.argv[0], args.output_dir)
        sys.exit(1)

    if args.list_parcellations:
        logging.info("Julich-related parcellations found:")
        for p in siibra.parcellations:
            if "julich" in p.name.lower():
                logging.info("  - %s", p.name)
        return

    if args.smoke_test:
        region, spec = find_region(siibra, parcellation, "hOc1", "left")
        if region is None:
            logging.error("SMOKE TEST FAILED: could not resolve hOc1 (left) in parcellation '%s'.", parcellation.name)
            logging.error("Run --list-parcellations and/or inspect the parcellation's region tree interactively:")
            logging.error("  python3 -c \"import siibra; p=siibra.get_parcellation('%s'); print(p.tree2str())\"", parcellation.name)
            sys.exit(1)
        logging.info("SMOKE TEST OK. Parcellation: %s | Region: %s | Matched via spec: %s", parcellation.name, region.name, spec)
        fingerprint_rows = fetch_receptor_fingerprint(siibra, region)
        logging.info("Receptor fingerprint rows for this region: %d", len(fingerprint_rows))
        for r in fingerprint_rows[:5]:
            logging.info("    %s", r)
        logging.info("Smoke test complete -- looks good, proceed with the full pull (drop --smoke-test).")
        return

    # ---- full discovery + pull ----
    discovery_report = {"parcellation": parcellation.name, "space": args.space, "regions": {}}
    receptor_csv_path = args.output_dir / "receptor_densities.csv"
    maps_dir = args.output_dir / "probability_maps"

    all_targets = [(lbl, hemi, "cortical") for lbl in CORTICAL_REGIONS for hemi in HEMISPHERES]
    all_targets += [(lbl, hemi, "subcortical") for lbl in SUBCORTICAL_REGIONS for hemi in HEMISPHERES]

    with open(receptor_csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["region", "hemisphere", "region_type", "matched_region_name", "receptor", "unit", "mean_density", "std_density"])

        for label, hemi, region_type in all_targets:
            key = f"{label}_{hemi}"
            region, spec_used = find_region(siibra, parcellation, label, hemi)
            entry = {"region_type": region_type, "matched": region is not None}
            if region is None:
                discovery_report["regions"][key] = entry
                continue
            entry["matched_region_name"] = region.name
            entry["spec_used"] = spec_used

            map_ok = fetch_probability_map(siibra, parcellation, region, args.space, maps_dir / f"{key}.nii.gz")
            entry["probability_map_saved"] = map_ok

            fingerprint_rows = fetch_receptor_fingerprint(siibra, region)
            entry["n_receptors_found"] = len(fingerprint_rows)
            for receptor, unit, mean, std in fingerprint_rows:
                writer.writerow([label, hemi, region_type, region.name, receptor, unit, mean, std])

            discovery_report["regions"][key] = entry
            logging.info("[%s] matched='%s' map=%s receptors=%d", key, region.name, map_ok, len(fingerprint_rows))

    report_path = args.output_dir / "discovery_report.json"
    report_path.write_text(json.dumps(discovery_report, indent=2))

    n_matched = sum(1 for v in discovery_report["regions"].values() if v.get("matched"))
    n_total = len(all_targets)
    n_with_receptors = sum(1 for v in discovery_report["regions"].values() if v.get("n_receptors_found", 0) > 0)
    logging.info("=" * 70)
    logging.info("DONE. %d/%d target regions matched. %d regions had at least one receptor fingerprint.", n_matched, n_total, n_with_receptors)
    logging.info("Probability maps: %s", maps_dir)
    logging.info("Receptor densities: %s", receptor_csv_path)
    logging.info("Full discovery report (what matched/didn't, and how): %s", report_path)
    if n_matched < n_total:
        logging.warning("%d target regions did NOT match -- check discovery_report.json, especially the subcortical "
                         "ones (LGN/SC/pulvinar), which may not exist as named Julich-Brain regions at all.",
                         n_total - n_matched)


if __name__ == "__main__":
    main()
