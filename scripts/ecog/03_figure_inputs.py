"""Step 3: build the compact figure inputs (the contents of ``reference_results/``).

Reads the rank-50 fits and the selection outputs of steps 1-2, plus the
rank-30 selection counts, and writes the tables and arrays that
``figures/fig3.py`` and ``figures/figS3_S5.py`` plot. Steps 1-3 therefore
reproduce ``reference_results/`` from the raw data; ``04_decay_rule_calibration.py``
adds the synthetic calibration and ``05_ecog_example.py`` the Fig. 3A-B example.

The primary cohort consists of the sites at which every fit of both primary
routes reached the requested rank (``site_rank_audit.csv``).
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from partial_observation import config
from partial_observation.io import atomic_csv
from partial_observation.selection import (
    PRIMARY_CASE, decay_gap, in_band_stable, rate_and_frequency, residual_pass,
    validated,
)

EXAMPLE_SITE = "sub-ccepAgeUMCU27__run-031551__T27-T28"   # Fig. 3C and Fig. S3A-D
WINDOWS = list(config.WINDOWS_MS)
PRIMARY = list(config.PRIMARY_ROUTES)
N_FITS = config.N_TRIALS
TAU_EDGES = np.geomspace(1, 10000, 25)          # ms
FREQUENCY_EDGES = np.linspace(0, 100, 21)       # Hz
GAP_EDGES = np.linspace(-.5, .5, 301)
BOOTSTRAP_SEED = 20260920


def load_decay(decay_dir: Path, stem: str, route: str) -> dict[str, np.ndarray]:
    with np.load(decay_dir / f"{stem}__{route}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def audit_arrays(z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Candidate-level quantities before and after each criterion."""
    rate, freq = rate_and_frequency(z["eig"], float(z["sf_hz"]))
    base = in_band_stable(rate, freq)
    residual = base & residual_pass(z["res1"])
    gap = decay_gap(z)
    radial = residual & np.isfinite(gap) & (gap <= config.DECAY_VALIDATION_TOLERANCE)
    return dict(window=z["window"], rate=rate, freq=freq, gap=gap, base=base,
                residual=residual, radial=radial)


# ----------------------------------------------------------------------
# Per-site configuration
# ----------------------------------------------------------------------
def site_manifest(fits: Path, out: Path) -> None:
    """Trials, channels and QC outcome of every candidate site (1788 rows).

    The table written by step 1 minus two columns: ``run_path`` (a local
    absolute path; subject, session and run identify the recording) and
    ``age`` (not used by any figure).
    """
    sites = pd.read_csv(fits / "sites.csv").drop(columns=["run_path", "age"])
    sites = sites.sort_values(["subject", "run", "site"], kind="stable")
    atomic_csv(sites, out / "site_manifest.csv.gz")


# ----------------------------------------------------------------------
# Fig. S3: selection flow and exclusion by fitted time constant
# ----------------------------------------------------------------------
def selection_audit(stems, decay_dir: Path, out: Path) -> None:
    gh = defaultdict(lambda: np.zeros(len(GAP_EDGES)-1))
    tn = defaultdict(lambda: np.zeros(len(TAU_EDGES)-1))
    td = defaultdict(lambda: np.zeros(len(TAU_EDGES)-1))
    counts = defaultdict(lambda: np.zeros(3, int))
    for stem in sorted(stems):
        for route in PRIMARY:
            z = audit_arrays(load_decay(decay_dir, stem, route))
            tau = -1000/z["rate"]
            for window in WINDOWS:
                cell = z["window"] == window
                key = (route, window)
                counts[key] += [int((z["base"] & cell).sum()),
                                int((z["residual"] & cell).sum()),
                                int((z["radial"] & cell).sum())]
                q = z["residual"] & cell & np.isfinite(z["gap"])
                gh[key] += np.histogram(np.clip(z["gap"][q], GAP_EDGES[0], GAP_EDGES[-1]),
                                        bins=GAP_EDGES)[0]
                den, _ = np.histogram(tau[q], bins=TAU_EDGES)
                num, _ = np.histogram(
                    tau[q & (z["gap"] > config.DECAY_VALIDATION_TOLERANCE)], bins=TAU_EDGES)
                td[key] += den
                tn[key] += num
    flow = [dict(route=route, window=window, n_in_band_stable=v[0], n_residual_pass=v[1],
                 n_radial_pass=v[2], radial_excluded=v[1]-v[2],
                 radial_excluded_pct=100*(v[1]-v[2])/v[1])
            for (route, window), v in counts.items()]
    pd.DataFrame(flow).to_csv(out / "selection_flow.csv", index=False)
    np.savez_compressed(out / "radial_diagnostic_histograms.npz", gap_edges=GAP_EDGES,
                        tau_edges=TAU_EDGES, **{f"gap_{r}_{w}": x for (r, w), x in gh.items()},
                        **{f"tau_den_{r}_{w}": x for (r, w), x in td.items()},
                        **{f"tau_num_{r}_{w}": x for (r, w), x in tn.items()})


# ----------------------------------------------------------------------
# Fig. S4: frequency x decay-time distribution of validated estimates
# ----------------------------------------------------------------------
def distributions(stems, decay_dir: Path, out: Path) -> None:
    """Per-participant 2-D histograms, normalised per site and fit."""
    subjects = sorted({x.split("__")[0] for x in stems})
    si = {s: i for i, s in enumerate(subjects)}
    ns = pd.Series([x.split("__")[0] for x in stems]).value_counts()
    h = {(r, w): np.zeros((len(subjects), len(TAU_EDGES)-1, len(FREQUENCY_EDGES)-1))
         for r in PRIMARY for w in WINDOWS}
    for stem in sorted(stems):
        subject = stem.split("__")[0]
        for route in PRIMARY:
            z = load_decay(decay_dir, stem, route)
            keep = validated(z)
            rate, freq = rate_and_frequency(z["eig"], float(z["sf_hz"]))
            tau, freq, window = -1000/rate[keep], freq[keep], z["window"][keep]
            for w in WINDOWS:
                q = window == w
                inside = q & (tau >= TAU_EDGES[0]) & (tau <= TAU_EDGES[-1])
                counts, _, _ = np.histogram2d(tau[inside], freq[inside],
                                              bins=(TAU_EDGES, FREQUENCY_EDGES))
                h[(route, w)][si[subject]] += counts/(N_FITS*ns[subject])
    np.savez_compressed(out / "continuous_distribution_subjects.npz",
                        frequency_edges=FREQUENCY_EDGES, tau_edges=TAU_EDGES,
                        subjects=subjects, **{f"{r}_{w}": v for (r, w), v in h.items()})


# ----------------------------------------------------------------------
# Fig. S5: robustness to rank, residual criterion and decay validation
# ----------------------------------------------------------------------
def effects(frame, stems=None, case=PRIMARY_CASE):
    """Participant-mean post-minus-pre change per site and fit."""
    q = frame[(frame.case == case) & frame.route.isin(PRIMARY)]
    if stems is not None:
        q = q[q.stem.isin(stems)]
    p = q.pivot(index=["subject", "stem", "route"], columns="window",
                values=["n_total", "n_fast", "n_slow"])
    out = []
    for metric in ("n_total", "n_fast", "n_slow"):
        x = ((p[(metric, "early")]-p[(metric, "pre")])/N_FITS).rename("change").reset_index()
        for route in PRIMARY:
            z = x[x.route == route].groupby("subject").change.mean()
            out.append(pd.DataFrame(dict(subject=z.index, metric=metric,
                                         contrast=route, change=z.values)))
    return pd.concat(out, ignore_index=True)


def summarize(e, subset):
    """Mean change with a participant-bootstrap 95% interval."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (m, c), q in e.groupby(["metric", "contrast"]):
        x = q.change.to_numpy()
        b = rng.choice(x, (5000, len(x)), replace=True).mean(1)
        rows.append(dict(subset=subset, metric=m, contrast=c, n_subjects=len(x),
                         mean=x.mean(), median=np.median(x),
                         ci_low=np.quantile(b, .025), ci_high=np.quantile(b, .975),
                         subjects_up=int((x > 0).sum()), subjects_down=int((x < 0).sum())))
    return pd.DataFrame(rows)


def robustness(stems, counts50, counts30, grid, out: Path) -> None:
    summary = pd.concat([
        summarize(effects(counts50, stems), "rank50_matched"),
        summarize(effects(counts50, stems, "baseline"), "rank50_no_radial"),
        summarize(effects(counts30), "rank30_all"),
        # Same sites and participants at rank 30: the rank changes alone.
        summarize(effects(counts30, stems), "rank30_primary"),
    ], ignore_index=True)
    summary.to_csv(out / "robustness_effects.csv", index=False)

    def fast_change(case):
        e = effects(counts50, stems, case)
        e = e[e.metric == "n_fast"]
        return {r: e[e.contrast == r].set_index("subject").change for r in PRIMARY}

    rows = []
    none = fast_change("baseline")
    for route in PRIMARY:
        rows.append(dict(route=route, lag_ms=0, tolerance=np.nan, mean=none[route].mean(),
                         subjects_positive=int((none[route] > 0).sum()),
                         n_subjects=len(none[route])))
    for tol in (.05, .10):
        for lag in (2, 5, 10):
            change = fast_change(f"one_{lag}ms_{int(tol*100):03d}")
            for route in PRIMARY:
                rows.append(dict(route=route, lag_ms=lag, tolerance=tol,
                                 mean=change[route].mean(),
                                 subjects_positive=int((change[route] > 0).sum()),
                                 n_subjects=len(change[route])))
    pd.DataFrame(rows).to_csv(out / "radial_sensitivity.csv", index=False)

    grid = grid[grid.stem.isin(stems)]
    rows = []
    for (eps, need), q in grid.groupby(["eps", "support"]):
        p = q.pivot(index=["subject", "stem", "route"], columns="window", values="n_fast")
        change = ((p.early - p.pre) / N_FITS).unstack("route")
        row = dict(eps=eps, support=need)
        for route, key in (("hankel_d50", "hankel"), ("standard", "standard")):
            x = change[route].groupby(level="subject").mean()
            row.update({f"{key}_mean": x.mean(), f"{key}_up": int((x > 0).sum()),
                        "n_subjects": len(x)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "residual_sensitivity.csv", index=False)


# ----------------------------------------------------------------------
# Fig. 3
# ----------------------------------------------------------------------
def delay_high_frequency(stems, decay_dir: Path, out: Path) -> None:
    """Validated estimates below and at or above 8 Hz, per participant (Fig. 3D)."""
    rows = []
    for stem in sorted(stems):
        for route in PRIMARY:
            z = load_decay(decay_dir, stem, route)
            keep = validated(z)
            _, frequency = rate_and_frequency(z["eig"], float(z["sf_hz"]))
            for window in WINDOWS:
                cell = keep & (z["window"] == window)
                rows.append(dict(
                    subject=str(z["subject"]), stem=stem, route=route, window=window,
                    n_low_per_fit=int((cell & (frequency < config.FREQUENCY_SPLIT_HZ)).sum())
                    / N_FITS,
                    n_high_per_fit=int((cell & (frequency >= config.FREQUENCY_SPLIT_HZ)).sum())
                    / N_FITS))
    site = pd.DataFrame(rows)
    subject = (site.groupby(["subject", "route", "window"], as_index=False)
               .agg(n_low_per_site_fit=("n_low_per_fit", "mean"),
                    n_high_per_site_fit=("n_high_per_fit", "mean"),
                    n_sites=("stem", "size")))
    subject.to_csv(out / "delay_high8_paired_subjects.csv", index=False)


def example_site_candidates(fits: Path, decay_dir: Path, out: Path) -> None:
    """Every stable 0-100 Hz candidate at the example site, with its criteria (Fig. 3C)."""
    modes = pd.read_csv(fits / "per_mode" / f"{EXAMPLE_SITE}.csv.gz")
    residual_columns = [f"heldout_{q}_residual" for q in range(config.N_HELD_OUT_TRIALS)]
    parts = []
    for route in PRIMARY:
        z = load_decay(decay_dir, EXAMPLE_SITE, route)
        ix = int(np.flatnonzero(z["horizon_requested_ms"]
                                == config.DECAY_VALIDATION_HORIZON_MS)[0])
        k = int(z["horizon_k"][ix])
        a = modes[modes.route == route].set_index("mode_uid").loc[z["mode_uid"]].reset_index()
        residuals = a[residual_columns].to_numpy(float)
        a["support_fraction"] = (residuals <= config.RESIDUAL_THRESHOLD).mean(axis=1)
        a["median_heldout_residual"] = np.median(residuals, axis=1)
        a["mu_abs"] = np.median(np.abs(z["mu"][:, :, ix, 0]), axis=1)
        a["claimed_ratio"] = np.abs(z["eig"]) ** k
        a["radial_gap"] = a.mu_abs - a.claimed_ratio
        a["horizon_k"] = k
        a["horizon_realised_ms"] = 1000 * k / float(z["sf_hz"])
        a["radial_tolerance"] = config.DECAY_VALIDATION_TOLERANCE
        a["selected"] = ((a.support_fraction >= config.REQUIRED_HELD_OUT_TRIALS
                          / config.N_HELD_OUT_TRIALS)
                         & (a.radial_gap <= config.DECAY_VALIDATION_TOLERANCE))
        a["radial_rule"] = "one-sided"
        keep = (np.isfinite(a.frequency_hz) & np.isfinite(a.decay_rate_s_inv)
                & (a.frequency_hz >= 0) & (a.frequency_hz <= config.MAX_FREQUENCY_HZ)
                & (a.decay_rate_s_inv < 0))
        parts.append(a[keep])
    table = pd.concat(parts)
    order = {name: i for i, name in enumerate(WINDOWS + PRIMARY)}
    table = table.sort_values(
        ["window", "route", "train_trial_position", "mode_index"], kind="stable",
        key=lambda s: s.map(order) if s.name in ("window", "route") else s)
    columns = ["subject", "run", "site", "window", "route", "hankel_d",
               "train_trial_position", "mode_index", "train_event_index", "eig_real",
               "eig_imag", "frequency_hz", "decay_rate_s_inv", "median_heldout_residual",
               "support_fraction", "median_heldout_modal_amplitude_uV", "mu_abs",
               "claimed_ratio", "radial_gap", "horizon_k", "horizon_realised_ms",
               "radial_tolerance", "selected", "radial_rule"]
    table[columns].to_csv(out / "example_site_candidates.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True,
                        help="Rank-50 analysis folder of 01_fit_cohort.py")
    parser.add_argument("--selection", type=Path, default=None,
                        help="Rank-50 output of 02_select_and_count.py "
                             "(default: <fits>/selection)")
    parser.add_argument("--rank30-counts", type=Path, required=True,
                        help="site_counts.csv.gz of the rank-30 fits")
    parser.add_argument("--decay", type=Path, default=None,
                        help="Rank-50 decay arrays (default: <fits>/decay)")
    parser.add_argument("--out", type=Path, default=Path("outputs/reference_results"))
    args = parser.parse_args()
    fits = args.fits.resolve()
    selection = (args.selection or fits / "selection").resolve()
    decay_dir = (args.decay or fits / "decay").resolve()
    out = args.out.resolve()
    for folder in ("cohort", "fig3", "supplement", "supplement/example_candidates"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(selection / "site_rank_audit.csv")
    rank_column = f"all_rank{config.REQUESTED_RANK}"
    stems = set(audit.loc[audit[rank_column], "stem"])
    print(f"primary cohort: {len(stems)} sites, "
          f"{len({s.split('__')[0] for s in stems})} participants")
    counts50 = pd.read_csv(selection / "site_counts.csv.gz")
    counts30 = pd.read_csv(args.rank30_counts)
    grid = pd.read_csv(selection / "residual_grid_counts.csv.gz")

    shutil.copyfile(selection / "site_counts.csv.gz", out / "cohort/site_counts_rank50.csv.gz")
    atomic_csv(counts30, out / "cohort/site_counts_rank30.csv.gz")
    shutil.copyfile(selection / "site_rank_audit.csv", out / "cohort/site_rank_audit.csv")
    site_manifest(fits, out / "cohort")

    example_site_candidates(fits, decay_dir, out / "fig3")
    delay_high_frequency(stems, decay_dir, out / "fig3")
    selection_audit(stems, decay_dir, out / "supplement")
    distributions(stems, decay_dir, out / "supplement")
    robustness(stems, counts50, counts30, grid, out / "supplement")
    for route in PRIMARY:
        name = f"{EXAMPLE_SITE}__{route}.npz"
        shutil.copyfile(decay_dir / name, out / "supplement/example_candidates" / name)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
