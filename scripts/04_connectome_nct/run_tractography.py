#!/usr/bin/env python3
"""
Phase 5 tractography: per-subject probabilistic tractography with anatomically
constrained tractography (ACT) and SIFT2 filtering, using MRtrix3.

Why these choices (flagged for review, not silently baked in):
- Response function: dwi2response dhollander (multi-tissue, unsupervised, the
  current MRtrix3-recommended default) for multi-shell data -- HCP S1200
  diffusion is 3-shell (b=1000/2000/3000 + b0), so this branch is expected to
  fire for every subject in this cohort. A single-shell fallback
  (dwi2response tournier + dwi2fod csd, single-tissue) is included in case a
  subject's real .bval doesn't match that expectation -- auto-detected from
  the real .bval file, not assumed.
- 5TT tissue segmentation: 5ttgen freesurfer, from the native-space
  wmparc.nii.gz already captured into derivatives (todo.md decisions log,
  2026-08-07). NOT 5ttgen fsl (would discard the real FreeSurfer
  subcortical/WM labels we already have) and NOT hsvs (needs a full
  recon-all surface directory, not confirmed present in our derivatives
  tree -- only the flattened wmparc volume was captured).
- Seeding: 5tt2gmwmi -> tckgen -seed_gmwmi -act -backtrack -crop_at_gmwmi,
  the ACT-recommended seeding strategy (Smith et al. 2012), not naive
  whole-brain seeding.
- Streamline count: 10 million raw (before SIFT2) by default -- a widely
  used convention for HCP-quality whole-brain connectomes in the
  literature. This is a convention choice, not a hard requirement --
  pass --n-streamlines to change it, and revisit if runtime is prohibitive.
- Filtering: tcksift2 (Smith et al. 2015), weighting streamlines to match
  FOD-based fibre density rather than trusting raw tckgen counts.

Explicitly NOT run here: tck2connectome (needs the Schaefer-200/Glasser
backbone parcellation -- pull_parcellations.py has not been run yet as of
2026-08-08). tck2connectome is fast; it's meant to run separately once the
parcellation is in place, not gate this overnight run.

Real paths verified against the server (sub-123456, 2026-08-08):
  rawdata dwi:    BIDS/rawdata/sub-<ID>/dwi/sub-<ID>_dwi.{nii.gz,bval,bvec}
  dwi brain mask: BIDS/derivatives/hcp_pipelines/sub-<ID>/dwi/sub-<ID>_nodif_brain_mask.nii.gz
  native wmparc:  BIDS/derivatives/hcp_pipelines/sub-<ID>/anat/sub-<ID>_wmparc_native.nii.gz

NOT independently re-verified here: that wmparc_native and the dwi data
share the same scanner/world-space alignment. HCP's own pipeline registers
diffusion to T1w-native space, and the earlier decision to capture
wmparc_native was made specifically because it's "the same space as the
diffusion data" (todo.md, 2026-08-07) -- but that claim was never checked
with mrinfo against a real diffusion volume. This script logs both images'
dimensions/voxel size/transform to run.log for a human to sanity-check
(especially on the first subject) rather than silently trusting it. It does
NOT abort on a mismatch, because ACT in MRtrix3 interpolates the 5TT image
live and does not require matching voxel grids -- only world-space
alignment matters, which the log lets you verify by eye.

This is unattended, multi-hour-per-subject, resumable: each stage checks for
its expected output file before re-running, so an interrupted run can be
restarted with the same command and will skip completed stages.

Usage (single subject):
    python3 run_tractography.py --bids-root /path/to/BIDS --subject 123456 \
        --out-dir /path/to/BIDS/derivatives/tractography [--n-streamlines 10M] \
        [--nthreads 8]

Usage (whole cohort, sequential, safe to nohup/tmux and walk away from --
this will very likely still be running across multiple nights; see the
driver loop given alongside this script rather than parallelizing subjects
here, to keep total CPU load predictable):
    for SID in $(ls /path/to/BIDS/rawdata | grep '^sub-' | sed 's/sub-//'); do
        python3 run_tractography.py --bids-root /path/to/BIDS --subject $SID \
            --out-dir /path/to/BIDS/derivatives/tractography --nthreads 8
    done
"""
import argparse
import json
import subprocess
from pathlib import Path


def run(cmd, log_file):
    """Run a command, appending its output to the per-subject log file."""
    cmd = [str(c) for c in cmd]
    line = "+ " + " ".join(cmd)
    print(line, flush=True)
    with open(log_file, "a") as lf:
        lf.write(line + "\n")
        lf.flush()
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {line}")


def detect_shells(bval_path: Path, tol=100):
    """Return sorted distinct b-value shells (rounded to nearest `tol`), e.g.
    [0, 1000, 2000, 3000]. Drives the multi-shell vs single-shell branch --
    read from the real .bval file for this subject, not assumed from the
    HCP protocol description."""
    vals = [float(v) for v in bval_path.read_text().split()]
    return sorted(set(round(v / tol) * tol for v in vals))


def log_space_check(wmparc: Path, dwi: Path, log_file):
    """Log (not enforce) the voxel size/dimensions/transform of both images
    so a human can sanity-check world-space alignment on the first subject.
    Non-fatal by design -- see module docstring."""
    with open(log_file, "a") as lf:
        for label, img in [("wmparc_native", wmparc), ("dwi", dwi)]:
            info = subprocess.run(["mrinfo", str(img)], capture_output=True, text=True)
            lf.write(f"\n--- mrinfo {label} ({img}) ---\n{info.stdout}\n")
    print(f"sub alignment info logged to {log_file} -- eyeball this for the first "
          f"subject before trusting the rest of the overnight run.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-streamlines", default="10M",
                     help="Raw streamline count before SIFT2 filtering (MRtrix3 -select syntax, e.g. 10M).")
    ap.add_argument("--nthreads", type=int, default=8)
    args = ap.parse_args()

    sid = args.subject
    bids_root = Path(args.bids_root)
    out_dir = Path(args.out_dir) / f"sub-{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    dwi = bids_root / "rawdata" / f"sub-{sid}" / "dwi" / f"sub-{sid}_dwi.nii.gz"
    bval = bids_root / "rawdata" / f"sub-{sid}" / "dwi" / f"sub-{sid}_dwi.bval"
    bvec = bids_root / "rawdata" / f"sub-{sid}" / "dwi" / f"sub-{sid}_dwi.bvec"
    mask = bids_root / "derivatives" / "hcp_pipelines" / f"sub-{sid}" / "dwi" / f"sub-{sid}_nodif_brain_mask.nii.gz"
    wmparc_native = bids_root / "derivatives" / "hcp_pipelines" / f"sub-{sid}" / "anat" / f"sub-{sid}_wmparc_native.nii.gz"

    for f in (dwi, bval, bvec, mask, wmparc_native):
        if not f.exists():
            raise FileNotFoundError(f"sub-{sid}: expected input not found: {f}")

    threads = ["-nthreads", str(args.nthreads)]

    dwi_mif = out_dir / "dwi.mif"
    if not dwi_mif.exists():
        run(["mrconvert", dwi, dwi_mif, "-fslgrad", bvec, bval, "-datatype", "float32"] + threads, log_file)

    log_space_check(wmparc_native, dwi, log_file)

    shells = detect_shells(bval)
    n_nonzero_shells = len([s for s in shells if s > 0])
    print(f"sub-{sid}: detected b-value shells (rounded) = {shells} "
          f"({n_nonzero_shells} non-zero shell(s))", flush=True)

    response_wm = out_dir / "response_wm.txt"
    fod_wm = out_dir / "fod_wm.mif"

    if n_nonzero_shells >= 2:
        response_gm = out_dir / "response_gm.txt"
        response_csf = out_dir / "response_csf.txt"
        fod_gm = out_dir / "fod_gm.mif"
        fod_csf = out_dir / "fod_csf.mif"
        if not response_wm.exists():
            run(["dwi2response", "dhollander", dwi_mif, response_wm, response_gm, response_csf,
                 "-mask", mask] + threads, log_file)
        if not fod_wm.exists():
            run(["dwi2fod", "msmt_csd", dwi_mif,
                 response_wm, fod_wm, response_gm, fod_gm, response_csf, fod_csf,
                 "-mask", mask] + threads, log_file)
    else:
        print(f"sub-{sid}: WARNING -- only {n_nonzero_shells} non-zero shell(s) detected, "
              f"expected 3 for HCP S1200 diffusion. Falling back to single-shell "
              f"single-tissue CSD (dwi2response tournier + dwi2fod csd). Flag this "
              f"subject for a manual look -- an unexpected acquisition is worth "
              f"understanding before trusting its tractogram.", flush=True)
        if not response_wm.exists():
            run(["dwi2response", "tournier", dwi_mif, response_wm, "-mask", mask] + threads, log_file)
        if not fod_wm.exists():
            run(["dwi2fod", "csd", dwi_mif, response_wm, fod_wm, "-mask", mask] + threads, log_file)

    fivett = out_dir / "5tt.mif"
    if not fivett.exists():
        run(["5ttgen", "freesurfer", wmparc_native, fivett, "-force"], log_file)

    gmwmi = out_dir / "gmwmi.mif"
    if not gmwmi.exists():
        run(["5tt2gmwmi", fivett, gmwmi, "-force"], log_file)

    tracks = out_dir / f"tracks_{args.n_streamlines}.tck"
    if not tracks.exists():
        run(["tckgen", fod_wm, tracks,
             "-act", fivett, "-backtrack", "-crop_at_gmwmi",
             "-seed_gmwmi", gmwmi,
             "-select", args.n_streamlines,
             "-algorithm", "iFOD2",
             "-maxlength", "250"] + threads, log_file)

    weights = out_dir / f"sift2_weights_{args.n_streamlines}.txt"
    if not weights.exists():
        run(["tcksift2", tracks, fod_wm, weights, "-act", fivett] + threads, log_file)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "subject": sid,
        "shells_detected": shells,
        "n_streamlines_requested": args.n_streamlines,
        "tracks_file": str(tracks),
        "sift2_weights_file": str(weights),
        "status": "ok",
    }, indent=2))
    print(f"sub-{sid}: DONE -> {tracks.name}, {weights.name}", flush=True)


if __name__ == "__main__":
    main()
