#!/usr/bin/env python3
"""
Bring Julich-Brain MNI-space region probability maps (hOc1-hOc5, bilateral) into
each subject's native space and onto their native cortical surface, so they can
be overlapped with the Schaefer-200 backbone parcellation to transfer receptor
values (decision log, 2026-08-07: "Julich-Brain's cytoarchitectonic/receptor
values get mapped onto whichever backbone parcels overlap hOc1-hOc5").

Pipeline per subject (mirrors HCP's own PostFreeSurferPipeline logic for moving
volumetric data onto the standard surface -- not a workaround, this is the tool
HCP itself uses):

  1. applywarp: each Julich MNI-space probability map -> subject's native T1
     volume space, using HCP's own nonlinear warp field
     (MNINonLinear/xfms/standard2acpc_dc.nii.gz). Trilinear interpolation
     (probability maps, not yet discrete labels).
  2. Combine the warped per-region probability maps into a single native-volume
     maximum-probability map (MPM): argmax across regions per voxel, background
     where max probability is below --prob-threshold. This discretization is
     scoped ONLY to hOc1-hOc5 for the purpose of surface overlap -- it is not
     used to redefine parcel boundaries anywhere else in the pipeline (that
     concern -- "probability-map thresholding into hard boundaries isn't
     standardized" -- is what ruled out full Julich-Brain as the connectome
     backbone; it doesn't apply to this narrow, 5-region overlap step).
  2b. wb_command -volume-label-import: the argmax MPM from step 2 is a plain
     integer NIfTI, not yet a real Workbench "label volume" -- this step embeds
     an actual label table (region names/colors, from a generated label-list
     text file) into the volume. Required before step 3; found missing
     2026-08-08 after a real run completed cleanly but produced a surface file
     with an empty label table (see decisions log / project changelog for the
     full story).

  IMPORTANT -- steps 2 and 2b are now HEMISPHERE-RESTRICTED (fixed 2026-08-11).
  The original version ran a single whole-brain argmax across all 14
  region x hemisphere probability maps together, then used that ONE combined
  MPM volume for both hemispheres' surface projection in step 3. Real output
  from map_hoc_to_backbone.py (built for Phase 7) surfaced a serious,
  previously-undetected bug this caused: the LEFT hemisphere's surface label
  file consistently contained substantial numbers of "_right"-named vertices
  (e.g. sub-123456: 223 vertices labeled hOc1_right inside the LEFT
  hemisphere file, vs. ~876 legitimately-labeled hOc1_left vertices -- roughly
  25% contamination), in 46/46 subjects, always in the same direction (never
  "_left" labels leaking into the RIGHT file), concentrated in hOc1/hOc2/hOc3d
  -- the regions closest to the midline/calcarine sulcus. Root cause: a
  whole-brain argmax has no hemisphere awareness -- if a "_right" probability
  map has any residual density bleeding across the midline into native-space
  voxels near the LEFT hemisphere's own surface (plausible this close to the
  interhemispheric fissure), the combined argmax can hand that voxel a
  "_right" label, and the LEFT hemisphere's ribbon-constrained surface
  sampling -- which only restricts WHERE it looks, not WHICH LABELS are
  eligible there -- picks it up. Fixed by building two fully independent MPM
  volumes, one per hemisphere, each an argmax over ONLY that hemisphere's own
  7 probability maps -- the opposite hemisphere's labels cannot appear by
  construction, not by hoping ribbon geometry keeps them apart.
  3. wb_command -volume-label-to-surface-mapping (ribbon-constrained): native
     MPM label volume -> subject's native mid-thickness surface, per hemisphere.
     NOT -volume-to-surface-mapping -- that command is for continuous/metric
     data and would numerically interpolate discrete region indices across each
     vertex's ribbon polyhedron, which is meaningless for categorical labels.
  4. wb_command -label-resample: native surface labels -> fsLR32k, using the
     subject's own native-to-fsLR32k sphere registration (already produced by
     HCP's MSMAll pipeline, captured wholesale in derivatives/anat/MNINonLinear).

Requires FSL (applywarp) and Connectome Workbench (wb_command) -- run this on
the data server, not in a network-restricted sandbox.

Usage:
    python3 register_julich_native.py \
        --subject 123456 \
        --derivatives-dir /path/to/derivatives/hcp_pipelines \
        --rawdata-dir /path/to/rawdata \
        --julich-mni-dir /path/to/Parcellations/julich_hOc_mni \
        --out-dir /path/to/derivatives/julich_native \
        [--prob-threshold 0.0] [--dry-run]

--julich-mni-dir must contain one MNI-space probability-map NIfTI per region,
named <region>.nii.gz (e.g. hOc1_left.nii.gz, hOc1_right.nii.gz, ...). Pull
these from siibra (already set up, Phase 3) before running this script.
"""
import argparse
import colorsys
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REGIONS = ["hOc1", "hOc2", "hOc3v", "hOc3d", "hOc4v", "hOc4d", "hOc5"]
HEMIS = ["left", "right"]


def run(cmd, dry_run):
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr.strip()}")
    return result.returncode


def find_region_files(julich_mni_dir: Path):
    """Map (region, hemi) -> mni-space probability map path. Missing files are
    reported, not silently skipped -- this feeds Claim 1/5, gaps must be visible."""
    found = {}
    missing = []
    for region in REGIONS:
        for hemi in HEMIS:
            candidates = list(julich_mni_dir.glob(f"{region}_{hemi}*.nii.gz")) + \
                         list(julich_mni_dir.glob(f"{region}*{hemi}*.nii.gz"))
            if candidates:
                found[(region, hemi)] = candidates[0]
            else:
                missing.append(f"{region}_{hemi}")
    return found, missing


def warp_to_native(region_files, t1w_native, warp_field, out_dir, dry_run):
    """Step 1: applywarp each MNI probability map into subject's native T1 space."""
    native_dir = out_dir / "native_prob_maps"
    native_dir.mkdir(parents=True, exist_ok=True)
    warped = {}
    for (region, hemi), mni_path in region_files.items():
        out_path = native_dir / f"{region}_{hemi}_native.nii.gz"
        cmd = [
            "applywarp",
            f"--in={mni_path}",
            f"--ref={t1w_native}",
            f"--warp={warp_field}",
            f"--out={out_path}",
            "--interp=trilinear",
        ]
        rc = run(cmd, dry_run)
        if rc == 0:
            warped[(region, hemi)] = out_path
        else:
            print(f"  WARNING: applywarp failed for {region}_{hemi}, excluded from MPM")
    return warped


def _label_color(index: int, n: int):
    """Deterministic, distinct RGBA (0-255 ints, Workbench label-list-file convention)
    for label index `index` of `n` total foreground labels. Colors are cosmetic only
    (don't affect any downstream analysis) -- just need to be valid and distinct so
    the label table is inspectable in wb_view."""
    hue = ((index - 1) / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255), 255


def build_mpm(warped, hemi, out_dir, prob_threshold, dry_run):
    """Step 2: argmax across warped probability maps -> native-volume MPM NIfTI
    (plain integer labels, no embedded Workbench label table yet -- see
    import_label_volume() for why that's a separate, required step).
    Pure numpy/nibabel, testable without FSL/Workbench installed.

    HEMISPHERE-RESTRICTED (fixed 2026-08-11 -- see module docstring for the
    real cross-hemisphere-contamination evidence this fixes): `warped` is
    filtered to ONLY this hemisphere's region maps before the argmax runs, so
    the opposite hemisphere's region labels cannot appear in the output
    volume at all, regardless of any probability-map bleed near the midline.
    Call this once per hemisphere ("left", "right") -- each call produces a
    fully independent MPM volume and label-list file.

    Also writes a Workbench-format label-list text file (2 lines per label:
    name, then "key R G B A" as 0-255 ints) -- required input for
    `wb_command -volume-label-import`, which is what actually turns this plain
    NIfTI into a real label volume. Only the foreground labels (1..N) go in
    this file -- key 0 ("unlabeled"/background) must NOT be listed explicitly;
    `-volume-label-import` defaults `-unlabeled-value` to 0 automatically and
    real wb_command output (2026-08-08, sub-123456) confirmed it hard-errors
    ("the unlabeled value must not be specified in label list file") if key 0
    is included."""
    out_path = out_dir / f"julich_hOc_MPM_native_{hemi}.nii.gz"
    label_list_path = out_dir / f"julich_hOc_MPM_native_{hemi}_labellist.txt"
    hemi_warped = {k: v for k, v in warped.items() if k[1] == hemi}

    if dry_run:
        print(f"  [dry-run] would build {hemi} MPM at {out_path} from {len(hemi_warped)} region maps")
        return out_path, label_list_path

    if not hemi_warped:
        raise RuntimeError(f"No warped region maps available for hemisphere={hemi} -- cannot build MPM")

    keys = sorted(hemi_warped.keys())
    stack = []
    ref_img = None
    for key in keys:
        img = nib.load(str(hemi_warped[key]))
        if ref_img is None:
            ref_img = img
        stack.append(img.get_fdata())
    prob_stack = np.stack(stack, axis=0)  # (n_regions, X, Y, Z) -- this hemisphere only

    max_prob = prob_stack.max(axis=0)
    argmax_idx = prob_stack.argmax(axis=0) + 1  # 1-indexed labels, 0 = background
    labels = np.where(max_prob > prob_threshold, argmax_idx, 0).astype(np.int16)

    out_img = nib.Nifti1Image(labels, ref_img.affine, ref_img.header)
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(out_path))

    lut_path = out_dir / f"julich_hOc_MPM_native_{hemi}_LUT.txt"
    with open(lut_path, "w") as f:
        f.write("# label_index\tregion\themi\n")
        for i, key in enumerate(keys, start=1):
            f.write(f"{i}\t{key[0]}\t{key[1]}\n")

    with open(label_list_path, "w") as f:
        for i, key in enumerate(keys, start=1):
            name = f"{key[0]}_{key[1]}"
            r, g, b, a = _label_color(i, len(keys))
            f.write(f"{name}\n")
            f.write(f"{i} {r} {g} {b} {a}\n")

    print(f"  built {hemi} MPM ({len(keys)} regions) -> {out_path}")
    print(f"  label lookup table -> {lut_path}")
    print(f"  Workbench label-list file -> {label_list_path}")
    return out_path, label_list_path


def import_label_volume(mpm_native_path, label_list_path, out_dir, hemi, dry_run):
    """Step 2b: wb_command -volume-label-import -- converts the plain-integer MPM
    NIfTI into a real Workbench label volume (embeds an actual label table).
    This step is REQUIRED before surface projection: a plain NIfTI with integer
    region indices is not, by itself, a "label volume" as far as Workbench is
    concerned -- it's indistinguishable from continuous scalar data without this
    import step.

    Called once per hemisphere (fixed 2026-08-11, alongside build_mpm()) --
    `hemi` only affects the output filename here (the actual hemisphere
    restriction happens upstream in build_mpm()'s filtered argmax).

    Found missing 2026-08-08: the first real run on sub-123456 completed with no
    errors and produced a well-formed 32492-vertex .label.gii file, but
    `wb_command -file-information` showed only key 0 "???" in the label table --
    the actual region assignments never made it onto the surface. Root cause was
    two compounding bugs: (1) this import step was skipped entirely, so the MPM
    was never a real label volume, and (2) project_to_surface() was calling
    `-volume-to-surface-mapping` (the continuous/metric command) instead of
    `-volume-label-to-surface-mapping` -- even with real label-table data, the
    metric command has no concept of a label table and would numerically
    interpolate/average discrete region indices across each vertex's ribbon
    polyhedron, which is meaningless for categorical labels.

    Raises on failure (non-dry-run) rather than silently continuing -- found
    the hard way 2026-08-08: the original version discarded run()'s return
    code, so a real -volume-label-import failure (bad label-list format) let
    the script cascade into confusing unrelated "File does not exist" errors
    from every downstream wb_command call instead of stopping at the real
    point of failure."""
    out_path = out_dir / f"julich_hOc_MPM_native_{hemi}_label.nii.gz"
    cmd = ["wb_command", "-volume-label-import", str(mpm_native_path), str(label_list_path), str(out_path)]
    rc = run(cmd, dry_run)
    if not dry_run and rc != 0:
        raise RuntimeError(
            f"wb_command -volume-label-import failed (see STDERR above) -- "
            f"aborting before Step 3+4, which depends on {out_path}"
        )
    return out_path


def _glob_one(directory: Path, pattern: str, dry_run: bool):
    """subprocess.run (list-args, no shell) never expands '*' -- resolve globs
    ourselves. In --dry-run mode the directory may be a synthetic fixture
    without real files, so fall back to the literal pattern for command
    preview purposes only."""
    matches = sorted(directory.glob(pattern))
    if matches:
        return matches[0]
    if dry_run:
        return directory / pattern
    raise FileNotFoundError(f"No file matching {pattern} in {directory}")


def project_to_surface(mpm_label_paths, subject_native_dir, subject_native_reg_dir,
                        subject_fslr32k_dir, subject_fslr32k_reg_dir, out_dir, dry_run):
    """Step 3+4: volume MPM label -> native surface -> resample to fsLR32k, per hemisphere.
    `mpm_label_paths` must be a dict {"left": path, "right": path}, each the
    output of import_label_volume() for that hemisphere -- a real Workbench
    label volume, NOT the plain-integer NIfTI from build_mpm() directly (that
    has no label table and would silently produce an empty "???"-only surface
    label file, caught via real wb_command -file-information output on
    sub-123456's 2026-08-08 run).

    Each hemisphere now samples from its OWN hemisphere-restricted MPM volume
    (fixed 2026-08-11) -- previously both hemispheres sampled the same
    whole-brain MPM, which let "_right"-named labels leak onto the LEFT
    surface (and vice versa, though empirically only one direction ever fired)
    wherever a probability map's density crossed the midline. See module
    docstring for the real evidence.

    Two DIFFERENT directory pairs are needed, not one, per a real fix on
    2026-08-08:
      - subject_native_dir / subject_fslr32k_dir (T1w/Native, T1w/fsaverage_LR32k):
        the ANATOMICAL ribbon surfaces (midthickness, white, pial, and the
        target-mesh midthickness_MSMAll used for -area-surfs). These carry
        real anatomical coordinates and MUST be in true native T1w/ACPC
        volume space to align with the MPM volume (built via applywarp
        --ref=<native T1w>) -- confirmed via real per-vertex coordinate
        comparison that MNINonLinear/Native is in MNI-nonlinear space
        despite "native" appearing in the name (see todo.md decisions log,
        check_surface_space.py).
      - subject_native_reg_dir / subject_fslr32k_reg_dir (MNINonLinear/Native,
        MNINonLinear/fsaverage_LR32k): the MSMAll sphere REGISTRATION files
        used by -label-resample for vertex correspondence. Confirmed via a
        real `find -iname "*sphere*"` across the ENTIRE project tree
        (2026-08-08) that no sphere file of any kind exists anywhere under a
        T1w/ path, including in the original untouched Extracted/ source --
        the MSMAll-registered sphere is only ever produced under
        MNINonLinear/. This isn't a capture gap, it's real: a sphere file
        encodes vertex-to-vertex correspondence via an abstract unit-sphere
        projection, not real-world anatomical coordinates, so there's no
        "T1w-space version" of it to begin with -- using the MNINonLinear/
        copy here is correct, not a relapse into the earlier bug (which was
        specifically about ANATOMICALLY-embedded surfaces being in the
        wrong space)."""
    results = {}
    for hemi_code, hemi_name in [("L", "left"), ("R", "right")]:
        mpm_label_path = mpm_label_paths[hemi_name]
        midthickness = _glob_one(subject_native_dir, f"*.{hemi_code}.midthickness.native.surf.gii", dry_run)
        white = _glob_one(subject_native_dir, f"*.{hemi_code}.white.native.surf.gii", dry_run)
        pial = _glob_one(subject_native_dir, f"*.{hemi_code}.pial.native.surf.gii", dry_run)
        sphere_reg = _glob_one(subject_native_reg_dir, f"*.{hemi_code}.sphere.MSMAll.native.surf.gii", dry_run)
        fslr_sphere = _glob_one(subject_fslr32k_reg_dir, f"*.{hemi_code}.sphere.32k_fs_LR.surf.gii", dry_run)
        fslr_midthickness = _glob_one(subject_fslr32k_dir, f"*.{hemi_code}.midthickness_MSMAll.32k_fs_LR.surf.gii", dry_run)

        native_label = out_dir / f"julich_hOc_MPM.{hemi_code}.native.label.gii"
        fslr_label = out_dir / f"julich_hOc_MPM.{hemi_code}.32k_fs_LR.label.gii"

        cmd1 = [
            "wb_command", "-volume-label-to-surface-mapping",
            str(mpm_label_path), str(midthickness), str(native_label),
            "-ribbon-constrained", str(white), str(pial),
        ]
        rc1 = run(cmd1, dry_run)

        cmd2 = [
            "wb_command", "-label-resample",
            str(native_label), str(sphere_reg), str(fslr_sphere),
            "ADAP_BARY_AREA", str(fslr_label),
            "-area-surfs", str(midthickness), str(fslr_midthickness),
        ]
        rc2 = run(cmd2, dry_run)

        if dry_run or (rc1 == 0 and rc2 == 0):
            results[hemi_name] = fslr_label
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--derivatives-dir", required=True, help="derivatives/hcp_pipelines")
    ap.add_argument("--rawdata-dir", required=True)
    ap.add_argument("--julich-mni-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prob-threshold", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't execute")
    args = ap.parse_args()

    sub = f"sub-{args.subject}"
    deriv_dir = Path(args.derivatives_dir) / sub
    rawdata_dir = Path(args.rawdata_dir) / sub
    julich_mni_dir = Path(args.julich_mni_dir)
    out_dir = Path(args.out_dir) / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    t1w_native = rawdata_dir / "anat" / f"{sub}_T1w.nii.gz"
    warp_field = deriv_dir / "anat" / "MNINonLinear" / "xfms" / "standard2acpc_dc.nii.gz"
    # T1w/ (NOT MNINonLinear/) -- fixed 2026-08-08 after a real, confirmed bug:
    # MNINonLinear/Native and MNINonLinear/fsaverage_LR32k are in MNI-nonlinear
    # space despite "native"/"Native" appearing in the name, which refers only
    # to mesh density. Confirmed via real per-vertex coordinate comparison
    # against T1w/Native for the same subject/surface (check_surface_space.py,
    # todo.md decisions log): mean 4.11mm / max 12.31mm displacement, the
    # signature of a genuine MNI warp. The MPM volume built in Step 1-2 is in
    # true native T1w/ACPC space (applywarp --ref=<native T1w>), so the surface
    # it's projected onto here must be in that same space -- T1w/Native and
    # T1w/fsaverage_LR32k, now captured by hcp_bids_map.py alongside the
    # MNINonLinear/ tree. Same filenames/glob patterns apply in both trees,
    # only the directory changed.
    native_surf_dir = deriv_dir / "anat" / "T1w" / "Native"
    fslr32k_dir = deriv_dir / "anat" / "T1w" / "fsaverage_LR32k"
    # Sphere REGISTRATION files only -- confirmed 2026-08-08 via a real find
    # across the whole project tree that no sphere file (MSMAll or otherwise)
    # exists anywhere under T1w/, including in the original Extracted/ source.
    # A sphere is an abstract vertex-correspondence object, not an anatomical
    # surface, so this is correctly sourced from MNINonLinear/ even though the
    # anatomical ribbon surfaces above are correctly sourced from T1w/ -- see
    # project_to_surface()'s docstring for the full reasoning.
    native_reg_dir = deriv_dir / "anat" / "MNINonLinear" / "Native"
    fslr32k_reg_dir = deriv_dir / "anat" / "MNINonLinear" / "fsaverage_LR32k"

    print(f"=== {sub} ===")
    for label, path in [("T1w native", t1w_native), ("warp field", warp_field),
                         ("Native surf dir", native_surf_dir), ("fsaverage_LR32k dir", fslr32k_dir),
                         ("Native sphere-reg dir", native_reg_dir), ("fsaverage_LR32k sphere-reg dir", fslr32k_reg_dir),
                         ("Julich MNI dir", julich_mni_dir)]:
        status = "OK" if path.exists() else "MISSING"
        print(f"  {label}: {path}  [{status}]")
        if status == "MISSING" and not args.dry_run:
            print(f"ERROR: required input missing, aborting {sub}")
            sys.exit(1)

    region_files, missing = find_region_files(julich_mni_dir)
    if missing:
        print(f"  WARNING: {len(missing)} region/hemi combos not found in --julich-mni-dir: {missing}")
    print(f"  found {len(region_files)}/{len(REGIONS) * len(HEMIS)} region probability maps")

    print("\nStep 1: applywarp (MNI -> native)")
    warped = warp_to_native(region_files, t1w_native, warp_field, out_dir, args.dry_run)

    print("\nStep 2: build native-volume MPM (per hemisphere -- fixed 2026-08-11, see module docstring)")
    mpm_paths, label_list_paths = {}, {}
    for hemi in ("left", "right"):
        mpm_paths[hemi], label_list_paths[hemi] = build_mpm(
            warped, hemi, out_dir, args.prob_threshold, args.dry_run)

    print("\nStep 2b: import as Workbench label volume (per hemisphere)")
    mpm_label_paths = {}
    for hemi in ("left", "right"):
        mpm_label_paths[hemi] = import_label_volume(
            mpm_paths[hemi], label_list_paths[hemi], out_dir, hemi, args.dry_run)

    print("\nStep 3+4: project to native surface, resample to fsLR32k")
    fslr_labels = project_to_surface(mpm_label_paths, native_surf_dir, native_reg_dir,
                                      fslr32k_dir, fslr32k_reg_dir, out_dir, args.dry_run)

    print(f"\nDone. fsLR32k label files: {fslr_labels if fslr_labels else '(dry-run, none written)'}")


if __name__ == "__main__":
    main()
