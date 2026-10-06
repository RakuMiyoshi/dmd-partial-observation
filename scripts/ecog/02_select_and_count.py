"""Step 2: apply the validation rule and count validated estimates per site.

Reads the output of ``01_fit_cohort.py`` and writes, under
``<fits>/selection/``:

* ``site_counts.csv.gz``: per site, route, window and selection case (the
  primary rule and the sensitivity settings of ``selection.CASES``), the number
  of validated estimates in total and by decay/frequency class;
* ``residual_grid_counts.csv.gz``: the same counts over a grid of one-step
  residual thresholds and required held-out trials, with the primary decay
  check held fixed (Fig. S5B);
* ``site_rank_audit.csv``: whether every fit of a site reached the requested
  rank. The primary cohort keeps the sites where it did for both primary routes.

Counts are summed over the ten single-trial fits of a window.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from partial_observation import config
from partial_observation.io import atomic_csv
from partial_observation.selection import (
    CASES, decay_gap, decisions, in_band_stable, rate_and_frequency,
)

COUNTS = ["n_total", "n_fast", "n_slow", "n_fast_low", "n_fast_high", "n_extreme"]
RESIDUAL_GRID_THRESHOLDS = (0.05, 0.10, 0.20)
RESIDUAL_GRID_REQUIRED = (3, 5, 7)


def completed_sites(fits: Path) -> list[str]:
    """Site stems with a complete fit, in cohort order."""
    runs = pd.read_csv(fits / "candidate_runs.csv")
    runs = runs[runs.selected_run]
    stems = []
    for row in runs.itertuples():
        stem = f"{row.subject}__{row.run}__{row.site}"
        manifest = fits / "manifests" / f"{stem}.json"
        if manifest.exists() and json.loads(manifest.read_text())["status"] == "complete":
            stems.append(stem)
    if not stems:
        raise RuntimeError(f"No completed sites under {fits}")
    return stems


def site_counts(stem: str, *, decay_dir: Path, routes: list[str]) -> list[dict]:
    rows = []
    for route in routes:
        with np.load(decay_dir / f"{stem}__{route}.npz", allow_pickle=False) as z:
            rate, freq, eligible, masks = decisions(z)
            window = z["window"]
            for name, h, t, rule, r in CASES:
                accepted, missing = masks[name]
                for w in config.WINDOWS_MS:
                    cell = window == w
                    keep = accepted & cell
                    fast = rate <= config.FAST_DECAY_PER_S
                    low = freq < config.FREQUENCY_SPLIT_HZ
                    rows.append(dict(
                        subject=stem.split("__")[0], stem=stem, route=route, window=w,
                        case=name, horizon_ms=h, tolerance=t, rule=rule,
                        snapshot_range=r, n_before=int((eligible & cell).sum()),
                        n_missing=int((missing & cell).sum()), n_total=int(keep.sum()),
                        n_fast=int((keep & fast).sum()), n_slow=int((keep & ~fast).sum()),
                        n_fast_low=int((keep & fast & low).sum()),
                        n_fast_high=int((keep & fast & ~low).sum()),
                        n_extreme=int((keep & (rate <= -50)).sum())))
    return rows


def residual_grid_counts(stem: str, *, decay_dir: Path) -> list[dict]:
    rows = []
    for route in config.PRIMARY_ROUTES:
        with np.load(decay_dir / f"{stem}__{route}.npz", allow_pickle=False) as z:
            rate, freq = rate_and_frequency(z["eig"], float(z["sf_hz"]))
            base = in_band_stable(rate, freq)
            gap = decay_gap(z)
            decay_ok = np.isfinite(gap) & (gap <= config.DECAY_VALIDATION_TOLERANCE)
            fast = rate <= config.FAST_DECAY_PER_S
            for eps in RESIDUAL_GRID_THRESHOLDS:
                support = (z["res1"] <= eps).sum(axis=1)
                for need in RESIDUAL_GRID_REQUIRED:
                    keep = base & (support >= need) & decay_ok
                    for window in config.WINDOWS_MS:
                        cell = keep & (z["window"] == window)
                        rows.append(dict(subject=stem.split("__")[0], stem=stem,
                                         route=route, window=window, eps=eps,
                                         support=need, n_total=int(cell.sum()),
                                         n_fast=int((cell & fast).sum())))
    return rows


def rank_audit(stem: str, *, npz_dir: Path) -> pd.DataFrame:
    with np.load(npz_dir / f"{stem}.npz", allow_pickle=False) as z:
        fits = pd.DataFrame({key: z["fit_" + key] for key in
                             ("route", "window", "actual_rank", "requested_rank")})
        fits["n_channels"] = len(z["channel_name_final"])
    fits["stem"] = stem
    fits["subject"] = stem.split("__")[0]
    return fits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True,
                        help="Analysis folder written by 01_fit_cohort.py")
    parser.add_argument("--decay", type=Path, default=None,
                        help="Folder of decay arrays (default: <fits>/decay)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder (default: <fits>/selection)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fits = args.fits.resolve()
    out = (args.out or fits / "selection").resolve()
    out.mkdir(parents=True, exist_ok=True)
    stems = completed_sites(fits)
    cli = json.loads((fits / "config.json").read_text())["cli"]
    routes = cli.get("routes", list(config.ROUTE_DELAYS))
    decay_dir = (args.decay or fits / "decay").resolve()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        counts = pd.DataFrame([row for rows in pool.map(
            partial(site_counts, decay_dir=decay_dir, routes=routes), stems, chunksize=4)
            for row in rows])
        atomic_csv(counts, out / "site_counts.csv.gz")
        if set(config.PRIMARY_ROUTES) <= set(routes):
            grid = pd.DataFrame([row for rows in pool.map(
                partial(residual_grid_counts, decay_dir=decay_dir), stems, chunksize=8)
                for row in rows])
            atomic_csv(grid, out / "residual_grid_counts.csv.gz")
            fits_table = pd.concat(pool.map(partial(rank_audit, npz_dir=fits / "npz"),
                                            stems, chunksize=8), ignore_index=True)
            primary = fits_table[fits_table.route.isin(config.PRIMARY_ROUTES)]
            rank = int(primary.requested_rank.iloc[0])
            audit = primary.groupby("stem").agg(
                n_fits=("actual_rank", "size"),
                **{f"all_rank{rank}": ("actual_rank", lambda x: bool((x == rank).all()))},
                min_rank=("actual_rank", "min"), n_channels=("n_channels", "first"),
                subject=("subject", "first"))
            expected_fits = len(config.PRIMARY_ROUTES) * len(config.WINDOWS_MS) * config.N_TRIALS
            assert (audit.n_fits == expected_fits).all()
            audit.to_csv(out / "site_rank_audit.csv")
    print(f"{len(stems)} sites, {counts.subject.nunique()} participants -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
