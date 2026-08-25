#!/usr/bin/env python3
"""
Phase 5, connectome construction. Rasterizes each backbone CIFTI parcellation
(Schaefer-200, Glasser MMP1.0) into a subject's native diffusion space, then
runs MRtrix3 tck2connectome to build a SIFT2-weighted structural connectome
matrix per subject per atlas.

Pipeline per subject, per atlas:
  Step 0: build a native-space reference volume grid by extracting a single
          volume from the subject's own 5tt.mif (mrconvert -coord 3 0) --
          this is the exact space tckgen/tck2connectome already operate in
          for this subject, so the rasterized parcellation is guaranteed to
          share that grid.
  Step 1: wb_command -cifti-separate -- splits the dense CIFTI dlabel (which
          covers both hemispheres on the shared 32k_fs_LR mesh) into two
          per-hemisphere GIFTI label files, since the volume-mapping command
          below is per-hemisphere.
  Step 2: wb_command -label-to-volume-mapping -ribbon-constrained, once per
          hemisphere -- rasterizes that hemisphere's GIFTI label onto the
          native-space reference grid, using the subject's own
          T1w/fsaverage_LR32k white_MSMAll / pial_MSMAll surfaces as the
          ribbon bounds and midthickness_MSMAll as the coordinate surface.
          Confirmed via real `ls` on 2026-08-10 that white/pial in
          T1w/fsaverage_LR32k/ DO carry the _MSMAll suffix, same as
          midthickness (123456.L.white_MSMAll.32k_fs_LR.surf.gii,
          123456.L.pial_MSMAll.32k_fs_LR.surf.gii) -- do not assume
          unsuffixed names. No per-subject re-registration of the
          parcellation itself is needed: every subject's cortex is already
          vertex-corresponded to the shared 32k_fs_LR mesh via MSMAll, so
          only rasterization changes per subject.
  Step 3: combine the two hemisphere volumes into one native-space
          parcellation volume, in Python via nibabel (not a wb_command/fsl
          math expression) so voxel overlap between hemispheres can be
          explicitly detected and logged rather than silently mis-combined.
          Schaefer-200 keys are 1-100 LH / 101-200 RH and Glasser MMP1.0 keys
          are 1-180 RH / 181-360 LH -- both already globally unique across
          hemispheres, so this is a straightforward union, not a relabeling.
  Step 4: MRtrix3 tck2connectome -- SIFT2-weighted, symmetric, zero-diagonal
          structural connectivity matrix from the subject's existing
          tractogram (tracks_10M.tck) and SIFT2 weights
          (sift2_weights_10M.txt) against the combined native-space
          parcellation volume. Also writes -out_assignments for QC.
  Step 5: sanity-check the output matrix is square with the atlas's expected
          node count (200 for Schaefer-200, 360 for Glasser MMP1.0) --
          raises and aborts (real run only) on mismatch, since a wrong node
          count means real corruption, not something to warn-and-continue on.

Scope: cortical nodes only, this pass. Subcortical nodes (LGN / superior
colliculus / pulvinar, from Benson et al.'s OSF pRF-defined functional masks)
are a separate, not-yet-built follow-up -- not a blocker here.

IMPORTANT -- wb_command flag syntax not yet confirmed on this project's real
installation. The -cifti-separate / -label-to-volume-mapping sequence below
follows the standard, long-stable Connectome Workbench command syntax, not
something guessed specifically for this project, but it has NOT been run for
real yet. Per this project's standing rule (verify against real structure,
don't trust assumptions -- exactly what caught five real bugs in
register_julich_native.py before it could be trusted), run this script with
--dry-run first and read every printed command before running for real on
even one subject. If wb_command rejects a flag, run `wb_command
-label-to-volume-mapping` and `wb_command -cifti-separate` with no arguments
on the server to print this installation's real usage/help text and paste it
back rather than guessing a fix.

Resumable: each subject is skipped if its manifest.json already exists
(written only on full success, same convention as run_tractography.py),
unless --force is passed. Safe for unattended re-runs / partial-failure
recovery.

Usage:
    python3 tck2connectome.py \
        --subject 123456 \
        --derivatives-dir /path/to/project/BIDS/derivatives/hcp_pipelines \
        --tractography-dir /path/to/project/BIDS/derivatives/tractography \
        --out-dir /path/to/project/BIDS/derivatives/connectomes \
        --atlas-dir /path/to/project/Atlas \
        --dry-run

    # all subjects (discovered from rawdata), both atlases:
    python3 tck2connectome.py \
        --rawdata-dir /path/to/project/BIDS/rawdata \
        --derivatives-dir /path/to/project/BIDS/derivatives/hcp_pipelines \
        --tractography-dir /path/to/project/BIDS/derivatives/tractography \
        --out-dir /path/to/project/BIDS/derivatives/connectomes \
        --atlas-dir /path/to/project/Atlas
"""
import argparse
import json
import subprocess
from pathlib import Path

ATLASES = {
    "schaefer200": {
        "filename": "Schaefer2018_200Parcels_7Networks_order.dlabel.nii",
        "n_nodes": 200,
    },
    "glasser": {
        "filename": "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
        "n_nodes": 360,
    },
}


def run(cmd, dry_run, log_fh=None):
    printable = " ".join(str(c) for c in cmd)
    print(f"    $ {printable}")
    if log_fh is not None:
        log_fh.write(printable + "\n")
        log_fh.flush()
    if dry_run:
        return
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_fh is not None:
        log_fh.write(result.stdout or "")
        log_fh.flush()
    if result.returncode != 0:
        if log_fh is not None:
            log_fh.write(f"COMMAND FAILED (exit {result.returncode})\n")
            log_fh.flush()
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {printable}\n{result.stdout}"
        )


def discover_subjects(rawdata_dir):
    subs = []
    for p in sorted(Path(rawdata_dir).iterdir()):
        if p.is_dir() and p.name.startswith("sub-"):
            subs.append(p.name[len("sub-"):])
    return subs


def find_surf(fslr32k_dir, subject, hemi, kind):
    """kind in {'white', 'pial', 'midthickness'} -- all three carry the
    _MSMAll suffix in T1w/fsaverage_LR32k/, confirmed via real `ls` 2026-08-10."""
    pattern = f"{subject}.{hemi}.{kind}_MSMAll.32k_fs_LR.surf.gii"
    matches = list(Path(fslr32k_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {fslr32k_dir}")
    return matches[0]


def build_native_ref(tracto_sub_dir, sub_out_dir, dry_run, log_fh, mrconvert_bin, nthreads, force=False):
    ref_path = sub_out_dir / "native_ref.nii.gz"
    if ref_path.exists() and not force and not dry_run:
        return ref_path
    five_tt = Path(tracto_sub_dir) / "5tt.mif"
    cmd = [mrconvert_bin, str(five_tt), "-coord", "3", "0", str(ref_path),
           "-nthreads", str(nthreads), "-force"]
    run(cmd, dry_run, log_fh)
    return ref_path


def rasterize_hemisphere(cifti_path, fslr32k_dir, subject, hemi, native_ref,
                          sub_out_dir, atlas_name, dry_run, log_fh, wb_bin):
    struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
    hemi_label_gii = sub_out_dir / f"{atlas_name}.{hemi}.label.gii"
    cmd = [wb_bin, "-cifti-separate", str(cifti_path), "COLUMN",
           "-label", struct, str(hemi_label_gii)]
    run(cmd, dry_run, log_fh)

    white = find_surf(fslr32k_dir, subject, hemi, "white")
    pial = find_surf(fslr32k_dir, subject, hemi, "pial")
    midthickness = find_surf(fslr32k_dir, subject, hemi, "midthickness")

    hemi_vol = sub_out_dir / f"{atlas_name}.{hemi}.native.nii.gz"
    cmd2 = [wb_bin, "-label-to-volume-mapping", str(hemi_label_gii),
            str(midthickness), str(native_ref), str(hemi_vol),
            "-ribbon-constrained", str(white), str(pial)]
    run(cmd2, dry_run, log_fh)
    return hemi_vol


def combine_hemispheres(left_path, right_path, out_path, dry_run, log_fh):
    if dry_run:
        msg = f"[dry-run] would combine {left_path} + {right_path} -> {out_path} (python/nibabel, checks for voxel overlap)"
        print(f"    {msg}")
        if log_fh is not None:
            log_fh.write(msg + "\n")
        return out_path

    import nibabel as nib
    import numpy as np

    l_img = nib.load(str(left_path))
    r_img = nib.load(str(right_path))
    l = np.asarray(l_img.dataobj).astype(np.int32)
    r = np.asarray(r_img.dataobj).astype(np.int32)

    overlap = (l > 0) & (r > 0)
    n_overlap = int(overlap.sum())
    if n_overlap > 0:
        msg = (f"WARNING: {n_overlap} voxels have nonzero labels from BOTH hemispheres -- "
               f"left-hemisphere label kept, right discarded at these voxels")
        print(f"    {msg}")
        if log_fh is not None:
            log_fh.write(msg + "\n")

    combined = np.where(l > 0, l, r)
    out_img = nib.Nifti1Image(combined, l_img.affine, l_img.header)
    nib.save(out_img, str(out_path))
    return out_path


def rasterize_atlas(atlas_name, atlas_cfg, atlas_dir, fslr32k_dir, subject,
                     native_ref, sub_out_dir, dry_run, log_fh, wb_bin, force=False):
    combined_path = sub_out_dir / f"{atlas_name}.native.nii.gz"
    if combined_path.exists() and not force and not dry_run:
        print(f"    {atlas_name}: native parcellation already exists, skipping rasterization")
        return combined_path

    cifti_path = Path(atlas_dir) / atlas_cfg["filename"]
    if not dry_run and not cifti_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {cifti_path}")

    left_vol = rasterize_hemisphere(cifti_path, fslr32k_dir, subject, "L", native_ref,
                                     sub_out_dir, atlas_name, dry_run, log_fh, wb_bin)
    right_vol = rasterize_hemisphere(cifti_path, fslr32k_dir, subject, "R", native_ref,
                                      sub_out_dir, atlas_name, dry_run, log_fh, wb_bin)
    combine_hemispheres(left_vol, right_vol, combined_path, dry_run, log_fh)
    return combined_path


def run_tck2connectome(tracks_path, weights_path, parcellation_vol, out_csv,
                        out_assignments, dry_run, log_fh, tck2connectome_bin, nthreads):
    cmd = [tck2connectome_bin, str(tracks_path), str(parcellation_vol), str(out_csv),
           "-tck_weights_in", str(weights_path),
           "-symmetric", "-zero_diagonal",
           "-out_assignments", str(out_assignments),
           "-nthreads", str(nthreads),
           "-force"]
    run(cmd, dry_run, log_fh)


def sanity_check_matrix(out_csv, expected_n, log_fh):
    with open(out_csv) as f:
        rows = [line.strip() for line in f if line.strip()]
    n_rows = len(rows)
    n_cols = len(rows[0].split(",")) if rows else 0
    if n_rows != expected_n or n_cols != expected_n:
        msg = (f"Connectome matrix shape mismatch: expected {expected_n}x{expected_n}, "
               f"got {n_rows}x{n_cols} in {out_csv}")
        if log_fh is not None:
            log_fh.write(msg + "\n")
        raise RuntimeError(msg)
    if log_fh is not None:
        log_fh.write(f"Matrix shape OK: {n_rows}x{n_cols} matches expected {expected_n}\n")


def process_subject(subject, args):
    sub_out_dir = Path(args.out_dir) / f"sub-{subject}"
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sub_out_dir / "manifest.json"
    log_path = sub_out_dir / "run.log"

    if manifest_path.exists() and not args.force and not args.dry_run:
        print(f"sub-{subject}: manifest.json already exists, skipping (use --force to redo)")
        return "skipped"

    fslr32k_dir = Path(args.derivatives_dir) / f"sub-{subject}" / "anat" / "T1w" / "fsaverage_LR32k"
    tracto_sub_dir = Path(args.tractography_dir) / f"sub-{subject}"
    tracks_path = tracto_sub_dir / "tracks_10M.tck"
    weights_path = tracto_sub_dir / "sift2_weights_10M.txt"

    print(f"=== sub-{subject} ===")
    mode = "a" if not args.dry_run else "w"
    log_fh = open(log_path, mode) if not args.dry_run else None
    try:
        native_ref = build_native_ref(tracto_sub_dir, sub_out_dir, args.dry_run, log_fh,
                                       args.mrconvert_bin, args.nthreads, force=args.force)

        results = {}
        for atlas_name in args.atlases:
            atlas_cfg = ATLASES[atlas_name]
            print(f"  -- atlas: {atlas_name} --")
            combined_vol = rasterize_atlas(atlas_name, atlas_cfg, args.atlas_dir, fslr32k_dir,
                                            subject, native_ref, sub_out_dir, args.dry_run,
                                            log_fh, args.wb_command_bin, force=args.force)
            out_csv = sub_out_dir / f"connectome_{atlas_name}.csv"
            out_assign = sub_out_dir / f"assignments_{atlas_name}.txt"
            run_tck2connectome(tracks_path, weights_path, combined_vol, out_csv, out_assign,
                                args.dry_run, log_fh, args.tck2connectome_bin, args.nthreads)
            if not args.dry_run:
                sanity_check_matrix(out_csv, atlas_cfg["n_nodes"], log_fh)
            results[atlas_name] = {
                "connectome_csv": str(out_csv),
                "assignments": str(out_assign),
                "n_nodes": atlas_cfg["n_nodes"],
            }

        if not args.dry_run:
            manifest = {"subject": subject, "status": "ok", "atlases": results}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"sub-{subject}: OK, manifest written")
        return "ok"
    finally:
        if log_fh is not None:
            log_fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", help="single subject ID (numeric, no sub- prefix)")
    ap.add_argument("--rawdata-dir", help="used to discover all subjects if --subject not given")
    ap.add_argument("--derivatives-dir", required=True,
                     help="hcp_pipelines derivatives root (has sub-<ID>/anat/T1w/fsaverage_LR32k)")
    ap.add_argument("--tractography-dir", required=True,
                     help="tractography derivatives root (has sub-<ID>/tracks_10M.tck etc.)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--atlas-dir", required=True,
                     help="directory containing the Schaefer-200 / Glasser MMP1.0 dlabel.nii files")
    ap.add_argument("--atlases", default="schaefer200,glasser",
                     help="comma-separated subset of: schaefer200,glasser")
    ap.add_argument("--wb-command-bin", default="wb_command")
    ap.add_argument("--mrconvert-bin", default="mrconvert")
    ap.add_argument("--tck2connectome-bin", default="tck2connectome")
    ap.add_argument("--nthreads", type=int, default=8,
                     help="threads per MRtrix command (mrconvert/tck2connectome auto-multithread "
                          "otherwise -- cap this so cohort-wide parallel runs don't oversubscribe "
                          "cores; real measurement 2026-08-10 showed a single unthrottled subject "
                          "used ~43 cores for ~112s wall-clock)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo subjects even if manifest.json exists")
    args = ap.parse_args()

    args.atlases = [a.strip() for a in args.atlases.split(",") if a.strip()]
    for a in args.atlases:
        if a not in ATLASES:
            raise SystemExit(f"Unknown atlas '{a}', choose from: {', '.join(ATLASES)}")

    if args.subject:
        subjects = [args.subject]
    elif args.rawdata_dir:
        subjects = discover_subjects(args.rawdata_dir)
        print(f"Discovered {len(subjects)} subjects from {args.rawdata_dir}")
    else:
        raise SystemExit("Must pass either --subject or --rawdata-dir")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    summary = {"ok": 0, "skipped": 0, "failed": 0}
    for subject in subjects:
        try:
            status = process_subject(subject, args)
            summary[status] += 1
        except Exception as e:
            print(f"sub-{subject}: FAILED -- {e}")
            summary["failed"] += 1

    print()
    print(f"Done. ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']}")


if __name__ == "__main__":
    main()
