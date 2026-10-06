# Provenance

## Figures

| Figure | Script | Inputs |
|---|---|---|
| Fig. 1 | none (conceptual illustration) | not computationally generated |
| Fig. 2 | `figures/fig2.py` | simulations only (`src/partial_observation/simulations/linear_network.py`, `src/partial_observation/simulations/stuart_landau.py`) |
| Fig. 3 | `figures/fig3.py` | `reference_results/fig3/`, `reference_results/cohort/` |
| Fig. S1 | `figures/figS1_S2.py` | linear simulation, idealised regime |
| Fig. S2 | `figures/figS1_S2.py` | linear simulation, recurrent noisy regime |
| Fig. S3 | `figures/figS3_S5.py` | `supplement/example_candidates/`, `oracle_controls.csv`, `radial_diagnostic_histograms.npz` |
| Fig. S4 | `figures/figS3_S5.py` | `supplement/continuous_distribution_subjects.npz` |
| Fig. S5 | `figures/figS3_S5.py` | `supplement/robustness_effects.csv`, `residual_sensitivity.csv`, `radial_sensitivity.csv` |

## `reference_results/`

Every file is produced by a script in `scripts/ecog/` from OpenNeuro ds004080
v1.2.4. Steps are numbered as in the README.

| File | Size | Produced by | Content |
|---|---:|---|---|
| `cohort/site_counts_rank50.csv.gz` | 1.8 MB | step 2 | validated-estimate counts per site, route, window and selection setting (1503 sites) |
| `cohort/site_counts_rank30.csv.gz` | 0.7 MB | step 2 on the rank-30 fits | the same for the rank-30 sensitivity analysis |
| `cohort/site_manifest.csv.gz` | 0.1 MB | step 1 (exported by step 3) | per-site configuration of all 1788 candidate sites: candidate, QC-passed and selected trials, recording channels and the reason each excluded channel was dropped, QC outcome, windows, routes and rank |
| `cohort/site_rank_audit.csv` | 0.1 MB | step 2 | whether every fit reached rank 50; defines the primary cohort (1318 sites, 44 participants) |
| `fig3/example_site_candidates.csv` | 0.2 MB | step 3 | every stable 0-100 Hz candidate at the Fig. 3C site with its criteria |
| `fig3/delay_high8_paired_subjects.csv` | 12 KB | step 3 | validated estimates below and above 8 Hz per participant (Fig. 3D) |
| `fig3/ecog_example.npz` | 55 KB | step 5 | raw trial and electrode positions of Fig. 3A-B (the fsaverage surface is fetched by MNE at plot time and not included) |
| `supplement/example_candidates/*.npz` | 2.9 MB | step 1 | decay diagnostics of the two primary routes at the Fig. 3C site |
| `supplement/selection_flow.csv` | <1 KB | step 3 | candidates remaining after each criterion |
| `supplement/radial_diagnostic_histograms.npz` | 7 KB | step 3 | exclusion by the decay check against fitted time constant (Fig. S3G) |
| `supplement/continuous_distribution_subjects.npz` | 59 KB | step 3 | per-participant frequency x decay-time histograms (Fig. S4) |
| `supplement/robustness_effects.csv` | 3 KB | step 3 | rank sensitivity (Fig. S5A) |
| `supplement/residual_sensitivity.csv` | 1 KB | step 3 | residual-criterion grid (Fig. S5B) |
| `supplement/radial_sensitivity.csv` | 1 KB | step 3 | decay-validation settings (Fig. S5C) |
| `supplement/oracle_controls.csv` | 59 KB | step 4 | synthetic calibration of the decay check (Fig. S3E-F) |

## What is not included

Raw data are not redistributed. The per-site intermediates of step 1 are not
included either, because of their size; step 1 regenerates them:

| Intermediate (`outputs/ecog/<analysis>/`) | Size |
|---|---:|
| rank 50: `per_mode/`, `npz/`, `qc/` (fits, held-out residuals, physical modes) | 7.3 GB |
| rank 50: `decay/` (multi-step decay diagnostics, all 5 routes) | 12 GB |
| rank 30: all of the above for 2 routes | 4.3 GB |

## Verification

The pipeline in this repository was checked against the artifacts from which
the manuscript figures were made:

* Steps 2-4 applied to the original step-1 artifacts reproduce every file in
  `reference_results/` byte for byte, except
  `fig3/example_site_candidates.csv`: the frozen table came from a separate
  refit of that one site, and the two agree in every selection decision, with
  relative differences in the numeric columns below 1e-8.
* Step 1 reproduces the original per-mode tables, physical modes and decay
  arrays of a test site exactly. Step 5 reproduces the trace and electrode
  arrays of the original Fig. 3A-B example exactly; the file here omits the
  fsaverage mesh and the participant age that the original also stored.
* The complete chain from the raw data (steps 1-5 on ds004080 v1.2.4, rank 50
  and rank 30) reproduces the same files, with the same single exception.
  The original rank-30 counts were stored for fewer selection settings; on the
  42,084 rows they share with `cohort/site_counts_rank30.csv.gz` (the file
  written by step 2, included here) every count agrees. Figs. 3 and S3-S5 drawn
  from these regenerated inputs match the submitted figures pixel for pixel.
* `cohort/site_manifest.csv.gz` is identical whether exported from the original
  step-1 artifacts or from the raw-data rerun.
* With `requirements-lock.txt`, all seven generated figures match the
  submitted figure files pixel for pixel at 60 dpi (Fig. 2 differs in 0.01% of
  pixels, as does a rerun of the original script).
