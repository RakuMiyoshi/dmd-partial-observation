# Partial observation, delay embedding and perturbation

Code and compact reference results accompanying the manuscript

> *Recovering Hidden Neural Oscillatory Modes from Partial Observations through
> Delay Embedding and Perturbation.*

The paper argues that the Hankel matrix of partially observed dynamics
factorises as H = P Q, where P is the observability matrix raised by delay
embedding and Q the trajectory (excitation) matrix raised by perturbation, so
that recovering an eigenvalue needs both. Figure 2 shows this in a linear and a
nonlinear simulation, and Figure 3 in human ECoG with single-pulse electrical
stimulation (OpenNeuro ds004080).

## Repository layout

```
src/partial_observation/     the package
  config.py                    every fixed parameter of the ECoG analysis
  ccep_data.py                 BIDS loader for ds004080
  cohort.py                    protocol-based site selection and quality control
  koopman.py                   single-trial Hankel DMD and held-out evaluation
  selection.py                 validation rule (residual + multi-step decay check)
  io.py                        atomic output and provenance records
  simulations/                 linear network and Stuart-Landau simulations
  plotting/                    panel and style helpers used by figures/
scripts/ecog/                raw data -> reference_results/ (steps 1-5)
figures/                     one entry script per figure or figure group
reference_results/           per-site configuration and frozen figure inputs (6 MB)
docs/provenance.md           figure -> script -> input -> generator
requirements-lock.txt        package versions used for the paper
```

## Installation

Python 3.12 was used for the paper.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

`requirements-lock.txt` pins the versions that reproduce the manuscript figures
pixel for pixel. Other versions satisfying `pyproject.toml` run the same code;
matplotlib 3.11 renders text slightly differently, and other NumPy/SciPy builds
change simulation spectra in the last digits.

## Reproducing the figures

Each script writes PDF and PNG files to `outputs/figures/`. None needs the raw
data: the simulations use fixed seeds, and the ECoG figures read
`reference_results/`. Fig. 3 additionally fetches the fsaverage template
surface through MNE on its first run (about 0.8 GB on disk, cached in `~/mne_data`);
the surface is not redistributed here.

| Figure | Command | Time |
|---|---|---|
| Fig. 2 | `python figures/fig2.py` | ~10 s |
| Fig. 3 | `python figures/fig3.py` | ~3.5 min |
| Figs. S1-S2 | `python figures/figS1_S2.py` | ~15 s |
| Figs. S3-S5 | `python figures/figS3_S5.py` | ~10 s |

Times are for one core of an AMD EPYC 7742; most of the Fig. 3 time goes into
rendering the 3-D cortical surface.

`fig3.py` and `figS3_S5.py` take `--inputs DIR` to plot a regenerated copy of
`reference_results/` instead of the frozen one.

## Reproducing `reference_results/` from the raw data

Download OpenNeuro ds004080 (v1.2.4) and point the loader to it:

```bash
export PARTIAL_OBSERVATION_BIDS_ROOT=/path/to/ds004080
```

Then run the five steps. Each script documents its inputs and outputs in its
docstring (`--help`).

```bash
# 1. Fit every site: 10 single-trial fits x 5 delay depths x 2 windows,
#    each evaluated on the 9 held-out trials. Resumable.
python scripts/ecog/01_fit_cohort.py --n-jobs 8 --strict-expected-counts
python scripts/ecog/01_fit_cohort.py --n-jobs 8 --rank 30 --routes standard hankel_d50

# 2. Apply the validation rule and count validated estimates per site.
python scripts/ecog/02_select_and_count.py --fits outputs/ecog/primary300_rank50
python scripts/ecog/02_select_and_count.py --fits outputs/ecog/primary300_rank30

# 3. Build the figure inputs.
python scripts/ecog/03_figure_inputs.py --fits outputs/ecog/primary300_rank50 \
    --rank30-counts outputs/ecog/primary300_rank30/selection/site_counts.csv.gz

# 4. Synthetic calibration of the decay check (no data needed).
python scripts/ecog/04_decay_rule_calibration.py

# 5. The recording example of Fig. 3A-B.
python scripts/ecog/05_ecog_example.py

# Plot from the regenerated inputs.
python figures/fig3.py --inputs outputs/reference_results
python figures/figS3_S5.py --inputs outputs/reference_results
```

Step 1 is the expensive one. On a 128-core machine, with the participants split
over 8 processes (`--subject` is repeatable) of `--n-jobs 8` each, it took
40 min at rank 50 and 17 min at rank 30, and wrote 19 GB and 4.3 GB. Steps 2-5
take a few minutes together.

`01_fit_cohort.py --dry-run` reproduces the cohort enumeration (1788 sites
from 50 participants before waveform quality control) without reading
waveforms.

## Data

Figure 1 is a conceptual illustration and has no computational source. Raw
ECoG data are not redistributed. They are available from OpenNeuro,
[ds004080](https://openneuro.org/datasets/ds004080) (CC0; van Blooijs et al.,
*Nature Neuroscience* 2023, doi:10.1038/s41593-023-01272-0). See
`docs/provenance.md` for which intermediate outputs are and are not included.
