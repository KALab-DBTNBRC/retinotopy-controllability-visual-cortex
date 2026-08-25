#!/usr/bin/env python3
"""
Gate check before building connectomes: verifies the two things Phase 5 and
Phase 6 depend on -- the Julich native-registration surface labels
(register_julich_native.py output) and the tractography output
(run_tractography.py output) -- are actually complete and correct for every
subject, not just "the script exited 0."

Built against the REAL output filenames/paths written by those two scripts
(register_julich_native.py, read in full 2026-08-08; run_tractography.py,
read in full 2026-08-08), not assumed -- see each check function's docstring
for the exact real path it inspects.

Why this exists: two real bugs already surfaced in this project where a
script exited cleanly (0 errors, "Done") but had actually produced silently
wrong or incomplete output -- the Julich label-table bug (real run looked
successful but the surface file had no real region labels, only found via
`wb_command -file-information`) and the `hcp_bids_map.py` mapped_files
undercount (real files were fine, the tracking number lied). This script
exists so that class of bug can't slip past a full-cohort loop unnoticed
again.

=== Julich native-registration check ===
For each subject, inspects the two final outputs:
    <julich-native-dir>/sub-<ID>/julich_hOc_MPM.{L,R}.32k_fs_LR.label.gii
via `wb_command -file-information`, and checks:
    - both hemisphere files exist
    - "Number of Vertices" == 32492 (the standard 32k_fs_LR mesh size --
      confirmed via real wb_command output on sub-123456, 2026-08-08)
    - the label table contains real region labels, not just background
      key 0 "???" (this is exactly the real bug found 2026-08-08 -- a
      "successful" run that silently produced an empty label table)
    - the set of real label names on the surface matches the set of names
      in that hemisphere's julich_hOc_MPM_native_<hemi>_LUT.txt exactly
      (catches any name mangling or truncation between native-volume
      construction and surface resampling, not just a count match) --
      hemisphere-specific LUT filename fixed 2026-08-11 to match
      register_julich_native.py's hemisphere-restricted MPM rework (was
      reading the old combined julich_hOc_MPM_native_LUT.txt, which the
      fixed script no longer writes -- a stale leftover copy of that file
      caused every subject to falsely FAIL this check against the new,
      correctly-hemisphere-restricted label files)
    - NEW 2026-08-11: no label whose name belongs to the OPPOSITE hemisphere
      (a "_right"-suffixed name on the L file, or vice versa) is present on
      the surface at all. This is a permanent regression test for a real,
      previously-undetected bug: a whole-brain (non-hemisphere-restricted)
      argmax in register_julich_native.py let "_right" labels leak onto the
      LEFT surface in 46/46 subjects, and the OLD version of this exact
      check did not catch it -- it only verified "all expected names
      present, no unexpected names" against a single combined 14-name list,
      so a leaked _right name looked like a legitimate hit. This check is
      independent of the LUT cross-reference above and fires even if no
      LUT file is available at all.

=== Tractography check ===
For each subject, inspects:
    <tractography-dir>/sub-<ID>/manifest.json      -- written ONLY on full
        successful completion (run_tractography.py's run() helper raises
        and kills the process on any nonzero exit code, so a subject with
        no manifest.json did not finish, whether still running or crashed)
    <tractography-dir>/sub-<ID>/sift2_weights_<N>.txt -- one weight per
        streamline, the most reliable completion signal available without
        needing MRtrix3 installed (pure Python, no external tool). Real
        format confirmed 2026-08-08 (wc -l/-w + head on sub-123456): a
        leading '#'-prefixed comment line, then all weights whitespace-
        separated on a single data line -- NOT one weight per line, which
        an earlier version of this script wrongly assumed and which made
        every subject look truncated. Counted as whitespace-separated
        tokens on non-comment lines, checked against n_streamlines_requested
        from the manifest.
    <tractography-dir>/sub-<ID>/run.log            -- grepped for
        "WARNING" (the single-shell-fallback warning is the only one
        run_tractography.py emits, but any other WARNING is surfaced too)
    manifest.json's shells_detected                -- flagged if it doesn't
        match the expected 3-shell HCP protocol ([0, 1000, 2000, 3000]),
        i.e. the single-shell fallback fired for that subject

Optional, off by default (--verify-with-tckinfo): cross-checks the
streamline count via `tckinfo` on the .tck file. This parser is written
from general MRtrix3 documentation, NOT verified against real tckinfo
output in this project (unlike everything else this script checks) --
treat its result as informational, not authoritative, until confirmed
against a real run. The weights-file line count above is the real,
fully-reliable check; tckinfo is a bonus cross-reference only.

Optional (--joblog): if GNU parallel's --joblog file is available, cross-
checks that every subject has an Exitval of 0 and that no subject is
missing from the joblog entirely.

Usage:
    python3 qc_pre_connectome.py \
        --rawdata-dir /path/to/BIDS/rawdata \
        --julich-native-dir /path/to/BIDS/derivatives/julich_native \
        --tractography-dir /path/to/BIDS/derivatives/tractography \
        [--joblog /path/to/joblog.tsv] \
        [--verify-with-tckinfo] \
        [--out-csv qc_pre_connectome_report.csv]
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

EXPECTED_N_VERTICES = 32492
EXPECTED_SHELLS = [0, 1000, 2000, 3000]


def discover_subjects(rawdata_dir: Path):
    """Subject list derived from real `ls rawdata`, never hand-typed --
    this project has already been bitten once by a hand-typed subject list
    going stale (extract_prf_solutions.py, 2026-08-08)."""
    subs = sorted(
        p.name.replace("sub-", "")
        for p in rawdata_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )
    if not subs:
        raise RuntimeError(f"No sub-* directories found under {rawdata_dir}")
    return subs


def count_weight_values(path: Path) -> int:
    """Count the actual number of SIFT2 weight values in a tcksift2 output
    file. Format CONFIRMED against real output (sub-123456, 2026-08-08, via
    `wc -l` / `wc -w` / `head -c 300`) -- NOT one weight per line as
    originally assumed. MRtrix3's real format: a leading '#'-prefixed
    command-history comment line, followed by all weights space-separated
    on a single data line (124 MB file, wc -l reported 2, wc -w reported
    10,000,011 -- 11 extra words came from the comment line, confirming the
    data line itself holds exactly the requested count as whitespace-
    separated tokens). This counts whitespace-separated tokens on every
    non-comment line, tolerant of the data being wrapped across more than
    one line even though the real file observed here uses just one."""
    n = 0
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            n += len(line.split())
    return n


def parse_mrtrix_count(s: str) -> int:
    """MRtrix3's -select shorthand: '10M' -> 10000000, '500K' -> 500000,
    plain digits -> int(digits). Documented MRtrix3 convention, not a guess."""
    s = s.strip().upper()
    m = re.match(r"^(\d+(?:\.\d+)?)([KM]?)$", s)
    if not m:
        raise ValueError(f"Cannot parse MRtrix3 streamline count: {s!r}")
    num, suffix = m.groups()
    mult = {"": 1, "K": 1_000, "M": 1_000_000}[suffix]
    return int(float(num) * mult)


def parse_wb_file_info(text: str):
    """Parse real `wb_command -file-information` output (format confirmed
    against real pasted output on sub-123456, 2026-08-08). Returns
    (n_vertices: int|None, labels: dict[int, str])."""
    n_vertices = None
    m = re.search(r"Number of Vertices:\s+(\d+)", text)
    if m:
        n_vertices = int(m.group(1))

    labels = {}
    if "Label table for ALL maps" in text:
        table_section = text.split("Label table for ALL maps", 1)[1]
        for line in table_section.splitlines():
            m = re.match(r"\s*(\d+)\s+(\S+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$", line)
            if m:
                key, name = int(m.group(1)), m.group(2)
                labels[key] = name
    return n_vertices, labels


def qc_julich_subject(sub_id: str, julich_native_dir: Path, wb_command_bin: str):
    """Returns dict: status ('OK'/'FAIL'), issues (list[str])."""
    sub_dir = julich_native_dir / f"sub-{sub_id}"
    issues = []

    for hemi_code, hemi_name in [("L", "left"), ("R", "right")]:
        opposite_hemi = "right" if hemi_name == "left" else "left"

        # Hemisphere-specific LUT (fixed 2026-08-11 -- register_julich_native.py
        # now writes one LUT per hemisphere, not one combined file).
        lut_path = sub_dir / f"julich_hOc_MPM_native_{hemi_name}_LUT.txt"
        expected_names = set()
        if lut_path.exists():
            for line in lut_path.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    expected_names.add(f"{parts[1]}_{parts[2]}")
        else:
            issues.append(f"MISSING {lut_path.name} -- cannot cross-check label names for {hemi_name}")

        label_gii = sub_dir / f"julich_hOc_MPM.{hemi_code}.32k_fs_LR.label.gii"
        if not label_gii.exists():
            issues.append(f"MISSING {label_gii.name}")
            continue

        try:
            result = subprocess.run(
                [wb_command_bin, "-file-information", str(label_gii)],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            issues.append(f"{wb_command_bin} not found in PATH -- cannot verify {label_gii.name}")
            continue
        except subprocess.TimeoutExpired:
            issues.append(f"{wb_command_bin} timed out on {label_gii.name}")
            continue

        if result.returncode != 0:
            issues.append(f"{label_gii.name}: wb_command -file-information failed: {result.stderr.strip()[:200]}")
            continue

        n_vertices, labels = parse_wb_file_info(result.stdout)

        if n_vertices != EXPECTED_N_VERTICES:
            issues.append(f"{label_gii.name}: vertex count {n_vertices} != expected {EXPECTED_N_VERTICES}")

        real_labels = {k: v for k, v in labels.items() if k != 0 and v != "???"}
        if not real_labels:
            issues.append(f"{label_gii.name}: NO REAL REGION LABELS FOUND -- only background/'???' present "
                           f"(this is the exact 2026-08-08 silent-failure pattern)")
        else:
            found_names = set(real_labels.values())

            # Permanent regression test for the 2026-08-11 cross-hemisphere
            # contamination bug -- independent of whether a LUT is available.
            cross_hemi_names = {n for n in found_names if n.endswith(f"_{opposite_hemi}")}
            if cross_hemi_names:
                issues.append(f"{label_gii.name}: CROSS-HEMISPHERE CONTAMINATION -- found "
                               f"{opposite_hemi}-hemisphere-named label(s) on the {hemi_name} surface: "
                               f"{sorted(cross_hemi_names)} (this is the exact 2026-08-11 whole-brain-argmax bug)")

            if expected_names:
                missing = expected_names - found_names
                extra = found_names - expected_names
                if missing:
                    issues.append(f"{label_gii.name}: label(s) in LUT but missing from surface: {sorted(missing)}")
                if extra:
                    issues.append(f"{label_gii.name}: label(s) on surface not in LUT: {sorted(extra)}")

    status = "OK" if not issues else "FAIL"
    return {"status": status, "issues": issues}


def parse_tckinfo(text: str):
    """Best-effort parse of `tckinfo` output. NOT verified against real
    tckinfo output in this project -- written from general MRtrix3
    documentation. Returns list of (raw_line, int_value) for every line
    containing 'count' followed by a number, so a human can eyeball which
    one is the real streamline count before this is trusted."""
    matches = []
    for line in text.splitlines():
        if "count" in line.lower():
            m = re.search(r"(\d+)", line)
            if m:
                matches.append((line.strip(), int(m.group(1))))
    return matches


def qc_tractography_subject(sub_id: str, tractography_dir: Path, tckinfo_bin: str, verify_with_tckinfo: bool):
    sub_dir = tractography_dir / f"sub-{sub_id}"
    issues = []
    warnings = []

    manifest_path = sub_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append("MISSING manifest.json -- subject did not reach the end of run_tractography.py "
                       "(still running, or crashed partway -- check run.log)")
        return {"status": "FAIL", "issues": issues, "warnings": warnings}

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        issues.append(f"manifest.json is not valid JSON: {e}")
        return {"status": "FAIL", "issues": issues, "warnings": warnings}

    if manifest.get("status") != "ok":
        issues.append(f"manifest.json status = {manifest.get('status')!r}, expected 'ok'")

    shells = manifest.get("shells_detected", [])
    nonzero_shells = sorted(s for s in shells if s > 0)
    if sorted(EXPECTED_SHELLS[1:]) != nonzero_shells:
        warnings.append(f"non-standard shells detected {shells} -- single-shell fallback likely fired "
                         f"for this subject, per run_tractography.py's own warning; worth a manual look")

    n_requested_raw = manifest.get("n_streamlines_requested", "10M")
    try:
        n_requested = parse_mrtrix_count(n_requested_raw)
    except ValueError:
        issues.append(f"cannot parse n_streamlines_requested={n_requested_raw!r}")
        n_requested = None

    tracks_path = Path(manifest.get("tracks_file", sub_dir / f"tracks_{n_requested_raw}.tck"))
    weights_path = Path(manifest.get("sift2_weights_file", sub_dir / f"sift2_weights_{n_requested_raw}.txt"))

    if not tracks_path.exists():
        issues.append(f"MISSING {tracks_path.name}")
    elif tracks_path.stat().st_size == 0:
        issues.append(f"{tracks_path.name} exists but is zero bytes")

    if not weights_path.exists():
        issues.append(f"MISSING {weights_path.name}")
    else:
        n_weights = count_weight_values(weights_path)
        if n_weights == 0:
            issues.append(f"{weights_path.name} exists but contains 0 weight values")
        elif n_requested is not None and n_weights != n_requested:
            issues.append(f"{weights_path.name}: {n_weights} weight values != "
                           f"{n_requested} requested streamlines -- possible truncation/corruption")

    run_log = sub_dir / "run.log"
    if run_log.exists():
        warning_lines = [l for l in run_log.read_text().splitlines() if "WARNING" in l]
        if warning_lines:
            warnings.append(f"run.log contains {len(warning_lines)} WARNING line(s): {warning_lines[0][:150]}")
    else:
        warnings.append("run.log not found -- can't check for logged warnings")

    if verify_with_tckinfo and tracks_path.exists():
        try:
            result = subprocess.run([tckinfo_bin, str(tracks_path)], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                matches = parse_tckinfo(result.stdout)
                warnings.append(f"[UNVERIFIED PARSER] tckinfo count-line matches (raw, for human review): {matches}")
            else:
                warnings.append(f"tckinfo failed: {result.stderr.strip()[:200]}")
        except FileNotFoundError:
            warnings.append(f"{tckinfo_bin} not found in PATH -- skipped tckinfo cross-check")
        except subprocess.TimeoutExpired:
            warnings.append(f"{tckinfo_bin} timed out")

    status = "OK" if not issues else "FAIL"
    return {"status": status, "issues": issues, "warnings": warnings}


def parse_joblog(joblog_path: Path):
    """GNU parallel --joblog format: tab-separated, header includes Seq,
    Host, Starttime, JobRuntime, Send, Receive, Exitval, Signal, Command.
    Returns dict: subject_id -> list of exitvals seen (append-only file,
    a re-run subject can appear more than once -- flag if any entry for
    that subject is nonzero, not just the last one)."""
    exitvals = {}
    with open(joblog_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cmd = row.get("Command", "")
            m = re.search(r"(?:--subject[= ]|sub-)(\d+)", cmd)
            if not m:
                continue
            sid = m.group(1)
            exitvals.setdefault(sid, []).append(row.get("Exitval", "?"))
    return exitvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdata-dir", required=True, type=Path)
    ap.add_argument("--julich-native-dir", required=True, type=Path)
    ap.add_argument("--tractography-dir", required=True, type=Path)
    ap.add_argument("--joblog", type=Path, default=None)
    ap.add_argument("--wb-command-bin", default="wb_command")
    ap.add_argument("--tckinfo-bin", default="tckinfo")
    ap.add_argument("--verify-with-tckinfo", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=Path("qc_pre_connectome_report.csv"))
    args = ap.parse_args()

    subjects = discover_subjects(args.rawdata_dir)
    print(f"Discovered {len(subjects)} subjects from {args.rawdata_dir}\n")

    joblog_exitvals = {}
    if args.joblog:
        if args.joblog.exists():
            joblog_exitvals = parse_joblog(args.joblog)
        else:
            print(f"WARNING: --joblog {args.joblog} does not exist, skipping joblog cross-check\n")

    rows = []
    n_julich_ok = n_tracto_ok = n_both_ok = 0

    for sid in subjects:
        j = qc_julich_subject(sid, args.julich_native_dir, args.wb_command_bin)
        t = qc_tractography_subject(sid, args.tractography_dir, args.tckinfo_bin, args.verify_with_tckinfo)

        joblog_note = ""
        if args.joblog and joblog_exitvals:
            vals = joblog_exitvals.get(sid)
            if vals is None:
                joblog_note = "not found in joblog"
            elif any(v != "0" for v in vals):
                joblog_note = f"nonzero exitval(s) in joblog: {vals}"
                if t["status"] == "OK":
                    t["status"] = "FAIL"
                    t["issues"].append(f"joblog reports {joblog_note}")

        if j["status"] == "OK":
            n_julich_ok += 1
        if t["status"] == "OK":
            n_tracto_ok += 1
        if j["status"] == "OK" and t["status"] == "OK":
            n_both_ok += 1

        rows.append({
            "subject": sid,
            "julich_status": j["status"],
            "julich_issues": " | ".join(j["issues"]),
            "tractography_status": t["status"],
            "tractography_issues": " | ".join(t["issues"]),
            "tractography_warnings": " | ".join(t["warnings"]),
        })

        if j["status"] != "OK" or t["status"] != "OK" or t["warnings"]:
            print(f"sub-{sid}: julich={j['status']}  tractography={t['status']}")
            for issue in j["issues"]:
                print(f"    [julich FAIL] {issue}")
            for issue in t["issues"]:
                print(f"    [tractography FAIL] {issue}")
            for warning in t["warnings"]:
                print(f"    [tractography WARN] {warning}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Summary ({len(subjects)} subjects) ===")
    print(f"Julich native registration: {n_julich_ok}/{len(subjects)} clean")
    print(f"Tractography:               {n_tracto_ok}/{len(subjects)} clean")
    print(f"Both clean (ready for tck2connectome): {n_both_ok}/{len(subjects)}")
    print(f"\nFull report written to {args.out_csv}")

    if n_both_ok < len(subjects):
        print(f"\n{len(subjects) - n_both_ok} subject(s) need attention before proceeding -- see issues above / CSV.")


if __name__ == "__main__":
    main()
