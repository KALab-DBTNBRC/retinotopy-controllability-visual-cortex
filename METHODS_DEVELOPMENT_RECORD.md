# Methods Development Record

*Repository documentation — NOT manuscript text. Do not paste any part of this file into the main manuscript.*

This file records errors identified and corrected during pipeline construction, together with how each was detected and verified. It exists so that the corrections behind the reported results are traceable and auditable, without placing development history in the manuscript's Materials and Methods. It accompanies the analysis code in the project repository.

Each entry states what went wrong, how it was found, what was changed, and how the fix was verified. Entries are ordered by pipeline stage, matching the order of the manuscript's Methods section.

---

## 1. Surface space: MNI-nonlinear used where anatomical was required

**Stage.** Julich-Brain registration to individual surfaces.

**What went wrong.** Surface files were initially taken from `MNINonLinear/Native/` and `MNINonLinear/fsaverage_LR32k/`. Despite the word "native" in the path, these are in MNI-nonlinear space. In HCP convention that word refers to mesh density, not anatomical space. Anatomical-space surfaces live under `T1w/`.

**Why it was dangerous.** The failure is silent. Filenames and vertex counts are identical between the two directories, so nothing errors, nothing looks wrong, and every downstream receptor value would have been computed against a misregistered surface.

**How it was found.** Identified while scoping the connectome step, then confirmed empirically rather than by reading documentation. A dedicated check computed real per-vertex displacement between the two surface sets. An initial centroid-only version of that check was inconclusive by construction and was rewritten to be per-vertex.

**Measured impact.** Mean displacement 4.11 mm, maximum 12.31 mm, on the test participant.

**Fix.** The BIDS mapping step was extended to capture `T1w/Native/` and `T1w/fsaverage_LR32k/`. The registration script was rewired to use those surfaces. Registration spheres continue to come from `MNINonLinear/`, which is the only place they exist.

**Verification.** Re-verified on the test participant (label table, vertex count, and a targeted labelled-vertex-location check), then the full 46-participant cohort was rebuilt and re-checked clean.

---

## 2. Cross-hemisphere label contamination in the maximum-probability map

**Stage.** Julich-Brain maximum-probability map construction.

**What went wrong.** The argmax across region probability maps was computed whole-brain, with no hemisphere restriction. Right-hemisphere-labelled vertices could therefore be assigned into left-hemisphere label files.

**Scale.** Present in 46/46 participants. Always one-directional: right-labelled vertices leaking into left files, never the reverse. Example magnitude on one participant: 223 vertices labelled `hOc1_right` inside the left-hemisphere file, against approximately 876 legitimately labelled `hOc1_left` vertices, roughly 25% contamination.

**Why it was dangerous.** This is independent of error 1 and would also have corrupted every receptor analysis silently, by mixing hemispheres within what was supposed to be a hemisphere-specific spatial unit. The project's design deliberately keeps hemispheres separate so that real bilateral asymmetry is not averaged away, which this error would have defeated.

**How it was found.** Surfaced by real output inspection while building the hOc-to-backbone overlap mapping, not by a targeted search for it.

**Fix.** The maximum-probability step was made hemisphere-restricted.

**Verification.** Two independent code paths, deliberately not sharing an implementation. A Workbench-based quality check (via `wb_command -file-information`), extended with a permanent cross-hemisphere regression check, returned 46/46 clean with zero contamination. An independent nibabel-based check during overlap mapping returned 0 of 17,022 cross-hemisphere mismatch rows. Note that the first re-run attempt was accidentally executed against stale pre-fix output; the clean result above is from a confirmed-fresh run.

---

## 3. Blank label table from a non-label-aware mapping command

**Stage.** Julich-Brain volume-to-surface mapping.

**What went wrong.** The maximum-probability map was mapped to the surface using the continuous `-volume-to-surface-mapping` command rather than the label-aware equivalent, and was never imported as a Workbench label volume. The resulting label table contained only the background key 0.

**Compounding error.** The first fix hit a Workbench hard error because the label-list file explicitly listed key 0, which Workbench reserves automatically. This was further masked by an error-swallowing bug in the import wrapper, which suppressed the failure rather than raising it.

**Fix.** Import as a real label volume, use the label-aware mapping command, omit key 0 from the label list, and remove the error suppression so failures surface.

**Verification.** Re-run on the test participant returned all 14 region labels present with the correct vertex count.

---

## 4. Path assumptions in the registration script

**Stage.** Julich-Brain registration, dry-run stage.

**What went wrong.** Three path assumptions were wrong, and surfaced as literal unresolved `*` wildcards in printed commands during a dry run. The native sphere registration file requires `MSMAll` in its glob pattern. The fsLR32k target sphere and midthickness are per-participant files under `fsaverage_LR32k/`, not fixed templates, and not under `Native/`.

**How it was found.** Dry-run output inspected against real server directory listings before any real execution.

**Fix.** All three touch points corrected: function signature and body, the existence-check block, and the call site.

**Verification.** Re-verified against a synthetic fixture reproducing the directory structure, then a clean real dry-run with zero unresolved wildcards.

---

## 5. Manifest double-counting in the archive-deletion preview

**Stage.** Source archive cleanup.

**What went wrong.** The read-only deletion preview reported 276 archives when only 138 existed. The manifest CSV is append-only and held rows from both the original run and a later forced re-run.

**How it was found.** Visibly repeated filenames in the preview output. This is the reason the preview step exists: no deletion was performed against the incorrect count.

**Fix.** Deduplicate on (subject, package) keeping the most recent row.

**Verification.** Corrected preview reported 138 archives, 648 GB, zero unverified, zero missing. The real deletion matched the preview exactly.

---

## 6. Stale script copy on the analysis server

**Stage.** pRF solution extraction, and again later during the node-strength control analysis.

**What went wrong.** Two separate instances. First, a pRF extraction run used a server-side copy of the script that still carried a superseded threshold and column name, combined with a hand-typed participant list containing errors, producing eight false "not found" results. Second, and much later, the node-strength control script failed on the server because the `run_claim2_analysis.py` copy sitting beside it was a pre-`--atlas` vintage whose loader signature had since changed.

**Fix.** For the first: the corrected script was copied to the server and the participant list was derived programmatically from the data directory rather than typed. For the second: the helper functions were vendored directly into the dependent script, so its results no longer depend on which vintage of the imported module happens to be adjacent. A signature check now reports explicitly when a stale copy is detected, and proceeds with the vendored implementations.

**Verification.** The pRF re-run returned 46/46 extracted with zero missing. The node-strength script was tested against three host conditions (current copy, absent copy, stale copy) and produced identical results in all three.

---

## 7. Silent atlas-scope truncation in a hardcoded label convention

**Stage.** Sex main-effect analysis, visual-network arm.

**What went wrong.** The visual-network loader was hardcoded to Schaefer's `_Vis_` label convention. Run against Glasser, it returned zero participants rather than raising. This silently truncated the FDR family from four tests to two, which in turn made one result appear significant that was not significant once evaluated against the correct family.

**How it was found.** Noticed while checking results, not caught by any automated check.

**Fix.** An atlas-aware loader, mirroring the equivalent fix already applied to the behavioural analysis.

**Verification.** Regression-checked against Schaefer first to confirm the previously reported numbers were reproduced, before trusting the Glasser output.

---

## 8. Input files from mismatched atlases accepted without complaint

**Stage.** Node-strength control analysis.

**What went wrong.** Two server runs were executed with a Glasser atlas flag but Schaefer input files. Because node labels between the two atlases share no strings, every intersection was empty, every participant was dropped, and the analysis wrote an output file reporting zero valid participants for all cells.

**Why it was dangerous.** An output file reporting zero valid participants is easily mistaken for a null finding rather than a failed run.

**Fix.** An input compatibility check now runs before any analysis. It reports participant counts, node counts, and node-label overlap for all three input files, and exits with a diagnostic naming the likely cause when any overlap is empty. It refuses to write output.

**Verification.** Tested against both the mismatched and correct invocations; the former now exits with a diagnostic, the latter proceeds normally.

---

## 9. Preliminary check contradicted by the definitive analysis

**Stage.** Node-strength control analysis.

**Not an error in the pipeline**, but recorded because the discrepancy could otherwise be rediscovered later and misread as instability in the data.

A quick preliminary check, using rank-based partial correlation with a Wilcoxon test across participants, suggested the modal-controllability arm survived node-strength adjustment at p = 0.004. The definitive analysis, using the project's own subject-level spatial permutation test, returned p = 0.108 under the same covariate specification. The difference is methodological (rank versus raw, Wilcoxon versus permutation), not a data instability. The permutation result is the one reported.

A related point worth noting for interpretation: in several cells, the parametric and rank tests and the permutation test diverge substantially, with the parametric tests being anti-conservative. This pattern is documented across multiple analyses in this project. The permutation test is the primary measure throughout.

---

## Cross-cutting practices adopted in response

Four conventions were adopted as a direct result of the above and applied from that point onward.

**Fixture-test before real data.** Every analysis script carries a self-test on synthetic data with known ground truth, including negative controls, before being run on real data.

**Regression-check every extension.** Where a script extends an earlier one, the extension must reproduce the original's numbers exactly on the overlapping configuration before its new results are trusted.

**Verify with an independent code path.** Where a correction matters, it is confirmed using a second implementation that does not share code with the first.

**Check outputs, not exit codes.** Aggregate sweeps verify the content of output files (shape, symmetry, value ranges, expected row counts, absence of NaNs) rather than relying on process exit status.
