# Retinotopic Organization Is Associated With Structural Controllability in Human Visual Cortex

[![tests](https://github.com/KALab-DBTNBRC/retinotopy-controllability-visual-cortex/actions/workflows/tests.yml/badge.svg)](https://github.com/KALab-DBTNBRC/retinotopy-controllability-visual-cortex/actions/workflows/tests.yml)
[![self-tests](https://img.shields.io/badge/self--tests-9%20fixture%20%2B%2018%20embedded-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](#dependencies)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

[![Data: HCP S1200 7T](https://img.shields.io/badge/data-HCP%20S1200%207T-0072B2)](https://db.humanconnectome.org)
[![Atlases](https://img.shields.io/badge/atlases-Schaefer--200%20%7C%20Glasser%20MMP1.0-56B4E9)](#repository-structure)
[![Cohorts](https://img.shields.io/badge/cohorts-N%3D46%20%2B%20N%3D20%20replication-009E73)](#summary-of-findings)
[![Nulls](https://img.shields.io/badge/inference-permutation%20%2B%20spin%20%2B%20Moran-E69F00)](#robustness-of-the-central-finding)
[![Reporting](https://img.shields.io/badge/reporting-COBIDAS%20MRI-lightgrey)](#statistical-approach)
[![Status](https://img.shields.io/badge/status-manuscript%20in%20preparation-inactive)](#citation)


Code and results for a network control theory (NCT) study of human visual
cortex, testing whether cytoarchitectonic receptor density or retinotopic
position (pRF eccentricity, polar angle) better explains average and modal
controllability across 46 healthy adults from the Human Connectome Project
(HCP) S1200 7-Tesla release.

Target journal: *Frontiers in Human Neuroscience* (Original Research), Brain
Imaging and Stimulation section.

Principal investigator: Dr. Khushbu Agarwal, National Brain Research Centre
(DBT-BRIC NBRC), Manesar, India.

## Summary of findings

Retinotopic position is associated with structural controllability in human
visual cortex. Receptor architecture, as measured here, shows no detectable
correspondence — but that null is bounded by the design rather than by sample
size, and is not evidence of absence. The limits are set out below.

The finding is carried by **average controllability**. Eccentricity tracks it in
both parcellations (Schaefer-200 mean partial r = +0.248, Glasser MMP1.0 +0.138),
under two different covariate specifications, and after adjustment for weighted
node degree in all four atlas-by-covariate combinations (+0.124 and +0.127 under
the vertex-count covariate; +0.180 and +0.270 under the pRF-size covariate). The
adjustment holds in linear, quadratic and rank forms; the cubic form is
inconclusive in Glasser (+0.076, p = 0.077), where a cubic polynomial alongside
node size consumes a large share of the degrees of freedom available at a median
of 21 nodes per participant. It survives nulls preserving the spatial
autocorrelation of a cortical map — spin and Moran spectral randomisation — both
before and after degree adjustment, survives restriction of the retinotopic
estimates to the stimulus-supported eccentricity range, survives adjustment for
per-node pRF data quality, and extends to polar angle as a second retinotopic
dimension. It replicates in a non-overlapping, substance-use-enriched companion
cohort (+0.205 Schaefer-200, +0.148 Glasser).

**Specificity is atlas-dependent, and is reported as such.** In Glasser, average
controllability is the only one of five node-level metrics to survive correction,
and a direct pairwise comparison on one common node set — participant by
participant, on absolute Fisher z — finds it significantly stronger than weighted
degree (+0.067), communicability (+0.072), eigenvector centrality (+0.123) and
modal controllability (+0.114), all q < 0.001, with average controllability the
stronger metric in 72 to 82 per cent of individual participants. In Schaefer-200
every centrality measure tracks eccentricity and no pairwise comparison approaches
significance (smallest q = 0.248): there, eccentricity tracks network centrality
in general rather than controllability in particular.

**Modal controllability behaves differently, and is reported as the weaker
result it is.** Its association with eccentricity is large before node strength
is controlled, but survives node-strength adjustment in only one of four
atlas-by-covariate combinations — weighted degree correlates with modal
controllability at a within-participant median of -0.92 to -0.95 here, so a modal
finding and a degree finding are close to the same statement (cf. Tu et al.,
2018; Pasqualetti et al., 2019). Glasser's Early Visual division is the exception,
carrying the largest modal estimate in the dataset (partial r = -0.489). That
division yields the strongest evidence for the effect but cannot be shown to have
the largest effect: divisions differ in participant retention as well as in effect
size, and no paired between-division contrast survives correction. See
`results/retinotopic_eccentricity_vs_controllability/nodestrength_control/`.

Cytoarchitectonic receptor density shows no detectable correspondence with
controllability on either parcellation (0 of 30 tests in each), is not associated
with retinotopic position when tested directly, and shows no relationship to
behaviour. Its only reliable structural correlates are anatomical region size and,
on one of two parcellations and within a single collinear cluster, raw structural
connectivity strength.

The receptor null is bounded by the design, and more severely than a sample
size argument would suggest. The Julich-Brain fingerprints are **not
lateralised** — left and right carry identical values in all 60
region-by-receptor combinations — while controllability varies most between
hemispheres. The predictor therefore cannot vary along the axis the outcome
varies on, and a perfectly informed hemisphere-constant predictor reaches only
42–70% of the statistic an unconstrained one would.

Monte Carlo simulation against the real participant data reaches 80% power in
**one of four** atlas-by-metric cells (Glasser, modal controllability, at
r = 0.87 against the region-level pattern). In the other three it does not
reach 80% at any effect size. For average controllability — the metric that
carries the retinotopic finding — power is **0.17 in Glasser and 0.00 in
Schaefer-200**, even against a mirrored predictor tracking the region-level
controllability shape almost perfectly.

So the receptor nulls for average controllability are not evidence about the
strength of a receptor–controllability relationship. They are evidence that
answering the question needs a lateralised receptor resource, which these
fingerprints are not. Nothing here should be read as showing that receptor
architecture is unrelated to controllability.

No controllability measure was associated with delay discounting, inhibitory
control, or NEO-FFI personality; the largest raw correlation anywhere was
r = -0.283, against a minimum detectable effect of r = 0.40. No controllability
measure differed by substance-use group membership or severity, in level or in
per-participant gradient, though those comparisons are underpowered at N = 20 and
rule out only large effects.

The manuscript is maintained separately and is not included here.
`METHODS_DEVELOPMENT_RECORD.md` documents the pipeline errors found and
corrected during construction, and how each was verified.

## Repository structure

```
scripts/
  00_utilities/                                  Standalone verification / preview helpers
  01_data_acquisition/                           HCP download, BIDS conversion, QC
  02_atlas_receptor_density/                     Julich-Brain cytoarchitectonic atlas + receptor density
  03_retinotopy_processing/                      7T pRF solution extraction and QC
  04_connectome_nct/                             Tractography, connectome construction, controllability,
                                                 the comparison graph metrics (degree, eigenvector
                                                 centrality, communicability, participation coefficient),
                                                 and the within-/extra-visual connectivity decomposition
  05_receptor_architecture_vs_controllability/   Receptor density vs. controllability (8 hOc units)
  06_retinotopic_eccentricity_vs_controllability/ Eccentricity/polar angle vs. controllability (central
                                                 finding), incl. run_claim2_nodestrength_control.py and
                                                 the robustness suite: metric specificity, spatially
                                                 constrained nulls, per-division slope comparison,
                                                 missingness and hemisphere diagnostics
  07_behavior_personality/                       Controllability vs. behavior and personality (exploratory)
  08_sex_receptor_interaction/                   Sex differences and sex x receptor-density interaction
  09_receptor_architecture_side_analyses/        Receptor fingerprint structure; non-controllability bridges
  10_addiction_companion_cohort/                 N=20 companion cohort: construction, replication,
                                                 group comparison, SSAGA severity sweep, and the
                                                 gradient-level cohort/severity/behaviour tests
  11_power_analysis/                             Minimum-detectable-effect analyses for the null results
  12_pernode_effect_maps/                        Node-level statistics behind the effect-map panels
tests/                                           Synthetic-fixture self-tests with known ground truth
results/
  <one subfolder per analysis stage above, mirroring scripts/05-12; the
   outputs of scripts/12_pernode_effect_maps land in results/pernode_analysis>
data/
  receptor-densities-deduped.csv                 Group-level Julich receptor densities (no subject data)
  {glasser,schaefer200}_node_centroids.csv       Node centroid coordinates from the group surfaces, used
                                                 as input to the spatially constrained null tests
METHODS_DEVELOPMENT_RECORD.md                    Pipeline errors found, fixed, and verified during
                                                 construction, with the verification method for each
```

Each `scripts/NN_*/` folder corresponds to one stage of the analysis
pipeline, run in numeric order. Scripts in `05_*` through `09_*` import
shared functions from earlier-numbered folders (e.g.
`run_claim1_analysis.py` in `05_*`); these cross-folder imports are resolved
at runtime via a path relative to each script's own file location, so the
repository works regardless of where it is cloned — no hardcoded paths.

## Data availability and de-identification

This is a secondary analysis of Human Connectome Project (HCP) S1200 7T
data, a restricted-access dataset requiring its own HCP data use agreement
(see [ConnectomeDB](https://db.humanconnectome.org)). Raw and intermediate
imaging data (BIDS-converted DWI, structural connectomes, pRF solution
files, etc.) are **not** included in this repository; scripts in
`01_data_acquisition/` through `04_connectome_nct/` operate on that data and
are provided for methodological transparency and reproducibility by
researchers with their own HCP data access, not as a runnable pipeline
against bundled inputs.

All result tables in `results/` that include subject-level rows have had
real HCP subject IDs replaced with anonymous sequential codes (`sub-01`
through `sub-46`, assigned in ascending order of the original numeric HCP
ID). The private lookup key mapping these codes back to real subject IDs is
kept outside this repository and is not distributed with it. Group-level
result tables (receptor densities, cytoarchitectonic region definitions,
etc.) carry no subject identifiers by construction, and every shipped table
was checked programmatically for six-digit HCP identifiers before release.

The subject-level graph-metric tables produced by
`scripts/04_connectome_nct/compute_graph_metrics.py` are excluded
rather than anonymised. They carry a `subject` column of raw HCP IDs, and for
the companion-cohort variants the filename plus that column would together
disclose substance-use cohort membership. Only the node- and group-level
summaries derived from them are shipped in
`results/retinotopic_eccentricity_vs_controllability/`. Anyone with HCP
access can regenerate
the subject-level tables by running that script against their own data.

The cohort-definition spreadsheets (`cohort-selection.xlsx`,
`addiction-cohort-selection.xlsx`, `master-table.xlsx`), the HCP restricted-
access subject table, and per-subject pRF/connectome tables are deliberately
**not** included, since they contain or can reconstruct restricted subject-
level information. Cohort construction is instead documented by the scripts
in `scripts/01_data_acquisition/` and
`scripts/10_addiction_companion_cohort/`, which state every inclusion and
exclusion criterion applied.

## Robustness of the central finding

The eccentricity-controllability result is supported by a suite of checks in
`scripts/06_*`, each a standalone script with an embedded synthetic-fixture
self-test (`--self-test`) that runs without real data and validates the
estimator against known ground truth, including negative controls.

| Script | Question it answers |
|---|---|
| `run_claim2_nodestrength_control.py` | Is the effect just weighted node degree wearing another name? |
| `run_graph_metric_specificity.py` (and `_parallel`) | Is the effect specific to controllability, or does eccentricity predict every network centrality measure equally? |
| `run_spatial_null.py` (and `_parallel`) | Does it survive nulls preserving the spatial autocorrelation of a cortical map (spin test, Moran spectral randomisation)? |
| `run_division_slope_comparison.py` | Is the effect genuinely larger in Early Visual cortex, or does that division simply retain more participants? |
| `run_missingness_hemisphere.py` | Is the surviving node set non-random with respect to eccentricity, and is the effect hemispherically symmetric? |
| `run_connectivity_decomposition.py` | Does the relationship track connections within visual cortex, or connections to the rest of the brain? |
| `run_gradient_linearity.py` | Is the relationship linear, or only monotonic? |
| `run_gradient_linearity_adjusted.py` | Does any curvature survive node-strength adjustment, and does it hold up out of sample? |
| `run_gradient_demographics.py` | Does the per-participant gradient vary with age or sex, given the cohort's age-by-sex skew? |
| `run_metric_pairwise_comparison.py` | Surviving correction while other metrics do not is not the same as being stronger than them. Are the dependent correlations different when compared directly, participant by participant, on one common node set? |
| `run_spatial_null_adjusted.py` | Do degree-robustness and spatial-autocorrelation-robustness hold simultaneously, rather than only one at a time? |
| `run_gradient_prf_quality.py` | Is the gradient an artefact of pRF data quality, and is it present in the worst data the cohort contains? |
| `run_headline_specification_checks.py` | Does it survive capping eccentricity at the stimulus-supported range, and adjusting for degree in quadratic, cubic and rank forms? |
| `run_claim2_influence_check.py`, `run_claim2_magnification_check.py`, `run_claim2_rfsize_permutation.py`, `bootstrap_claim2_ci.py` | Individual-participant influence, cortical magnification, receptive-field size, interval estimates |

Four of these narrowed the paper's claims rather than supporting them.
Specificity to controllability holds in Glasser MMP1.0 but not in
Schaefer-200, where eccentricity predicts all centrality measures
undifferentiatedly. Early Visual cortex yields the strongest *evidence*
for the effect but cannot be shown to have the largest *effect*, since
divisions differ in participant retention as well as in effect size. And an
apparent departure from linearity, reported in an earlier draft, proved to be
a connectivity-strength effect: it vanished once weighted node degree entered
the model, and the spline never predicted held-out nodes better than a
straight line. Finally, the cubic degree-adjustment could not be resolved in
Glasser at this node count and is reported as inconclusive rather than as either
support or refutation. **`gradient_linearity_*.csv` reports a superseded,
unadjusted result and is retained only for comparison; the adjusted result in
`gradient_linearity_adjusted_*.csv` is the one the manuscript reports.**

The two parcellations are used for different questions, and each for the
question it can answer. Glasser's architectonic parcel boundaries let the
network metrics separate, so it is the atlas for asking whether the
relationship is specific to average controllability. Schaefer-200's
near-uniform parcel sizes let the within- and extra-visual connectivity
components separate, so it is the atlas for asking which component the
relationship tracks.

The `_parallel` variants are vectorised for permutation-heavy runs and seed
per work unit via `numpy.random.SeedSequence.spawn()`, so results are
reproducible and invariant to the number of jobs.

## Statistical approach

Permutation-based testing is used as the primary significance measure
throughout, motivated by a real, confirmed design vulnerability at this
study's small spatial-unit counts (8 cytoarchitectonic units or a few dozen
visual-network nodes per subject): a fixed spatial predictor can align with
controllability's own fixed spatial pattern by chance, inflating parametric
significance without a true relationship. In several analyses the parametric
and permutation results diverge sharply, with the parametric tests being
anti-conservative — for example, 25 of 30 receptor tests in Schaefer-200 and
29 of 30 in Glasser return an uncorrected parametric p < 0.05 while failing
permutation-corrected FDR. Where parametric or rank tests are reported
alongside, they are secondary and are not counted separately in correction.

Benjamini-Hochberg FDR correction is applied within each analysis's own test
family, never pooled across analyses. Two permutation-based designs (the
receptor-density spatial permutation test and the sex x receptor-density
interaction's subject-level permutation null) were validated against
synthetic negative-control data before being trusted on real data; the
latter's calibration was independently checked across 300 true-null
replicates (false-positive rate 3.3% at alpha = 0.05, KS test against
Uniform(0,1) p = 0.52).

Every controllability finding is additionally tested against weighted node
degree as a competing explanation, following Tu et al. (2018) — see
`scripts/06_*/run_claim2_nodestrength_control.py`, which carries five
fixture self-tests including a planted signal fully mediated by node
strength that the procedure correctly eliminates. Monte Carlo and
closed-form power analyses accompany the null results (`scripts/11_*`), and
a leave-one-subject-out influence diagnostic accompanies the central
eccentricity finding.

`METHODS_DEVELOPMENT_RECORD.md` documents every pipeline error found and
corrected during construction, together with how each was detected and
verified.

## Continuous integration

Every push and pull request runs `.github/workflows/tests.yml`, which does
three things:

1. Runs all 9 fixture tests in `tests/` and every embedded `--self-test`
   suite in `scripts/` (18 at present), on Python 3.10, 3.11 and 3.12. The
   self-test suites are discovered by grep at run time rather than listed, so a
   new script carrying one is picked up without editing the workflow. These validate the
   estimators against synthetic data with known ground truth, including
   negative controls, and need no access to HCP data. CI installs only
   `requirements-ci.txt` (numpy, scipy, nibabel, openpyxl), the minimal set
   reachable from those entry points; the full analysis stack is documented
   under Dependencies below.
2. Fails the build if any tracked file contains a six-digit number that is
   not on a short, commented allow-list of known non-identifiers. Real HCP
   subject IDs are six digits, so this is a tripwire against committing one.
3. Fails the build if any tracked file contains an absolute server path.

Checks 2 and 3 exist because both problems occurred during development and
were caught by hand; they are automated so they cannot recur silently.

## Dependencies

Python 3.10. Packages: `numpy`, `scipy`, `pandas`, `nibabel`, `h5py`,
`nctpy` (1.0.1), `siibra` (1.0.1-alpha.21), `openpyxl`, `matplotlib`.

External neuroimaging tools (required for `01_data_acquisition/` through
`04_connectome_nct/` only; not required to inspect or re-analyze the
already-computed tables in `results/`): MRtrix3 3.0.8, Connectome Workbench
(`wb_command`) 1.5.0, and FSL 6.0.7.

Random seeds are fixed and explicit in every permutation and bootstrap
analysis (default seed 42), so reported results are exactly reproducible.

Each script's own module docstring documents its exact inputs, outputs, and
usage; run any script with `--help` for its full argument list.

## Reproducing an analysis on the shipped result tables

Scripts in `05_*` through `12_*` can be re-run directly against the tables
already present in `results/` (e.g. to regenerate a specific statistical
test) without needing the raw HCP data or upstream neuroimaging tools.
Figure rendering itself is not part of this repository: the surface renders
and composed panels are produced separately, and only the node-level
statistics behind them are shipped, under `results/pernode_analysis/`. See each script's `--help` and module docstring for its
expected input file arguments.

Every analysis script carries a fixture or self-test mode that runs on
synthetic data with known ground truth, including negative controls. Run
`--self-test` where offered, or the corresponding file in `tests/`. Where a
script extends an earlier one, the extension reproduces the original's
numbers exactly on the overlapping configuration as a regression check
before its new results are trusted.

## Citation

If you use this code or these result tables, please cite the accompanying
paper:

> Roy, R. K., Kothekar, R., Jebel, A., Bassi, T., Chaudhary, S., Sarkar, S.,
> Balhara, Y. P. S., and Agarwal, K. Retinotopic Organization Is Associated
> With Structural Controllability in Human Visual Cortex. *Under review.*

Volume, pages and DOI will be added on acceptance.

## Authors

Ritam Kanti Roy<sup>1</sup>, Rahul Kothekar<sup>1</sup>,
Andrea Jebel<sup>1</sup>, Tanya Bassi<sup>1</sup>,
Shefali Chaudhary<sup>2</sup>, Siddharth Sarkar<sup>3</sup>,
Yatan Pal Singh Balhara<sup>3</sup>, Khushbu Agarwal<sup>1,\*</sup>

1. Department of Cognitive and Computational Neuroscience, National Brain
   Research Centre, NH-8, Manesar, Gurugram, Haryana, 122052, India.
2. Department of Psychiatry, Yale University School of Medicine, New Haven,
   CT, 06511, USA.
3. National Drug Dependence Treatment Centre, All India Institute of Medical
   Sciences, Sri Aurobindo Marg, Ansari Nagar, Ansari Nagar East, New Delhi,
   Delhi 110029, India.

\* Corresponding author: Khushbu Agarwal, PhD —
khushbu.agarwal@nbrc.ac.in; khushbu.agarwal@nbrc.res.in

## License

MIT License — see `LICENSE`.
