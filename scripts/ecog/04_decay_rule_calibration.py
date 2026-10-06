"""Step 4: synthetic calibration of multi-step decay validation (Fig. S3E-F).

Eigenfunction series with a known 20-Hz eigenvalue are generated for nine
held-out trials, either as a decaying transient or as a stationary AR(1)
process, with complex measurement noise. The rule is applied with the true
eigenvalue (``claim_factor = 1``) and with a decay claimed four times too fast
(``claim_factor = 4``). No DMD is fitted: this tests the exclusion rule, not
the estimator. No data are needed.

Writes ``oracle_controls.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from partial_observation import config

SEED = 20260911
N_SNAPSHOTS = 614            # a 300-ms window at 2048 Hz
FREQUENCY_HZ = 20.0
TAUS_MS = [2., 5., 10., 20., 50., 100.]
SETTINGS = ([("transient", s) for s in [0., .03, .1, .3]]
            + [("stationary_AR", s) for s in [0., .1, .3]])
HORIZONS_MS = [2., 5., 10.]
TOLERANCES = [.05, .10]


def simulate(reps: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sf = config.SAMPLING_FREQUENCY_HZ
    n = N_SNAPSHOTS
    n_trials = config.N_HELD_OUT_TRIALS
    rows = []
    for tau in TAUS_MS:
        lam = np.exp((-1000/tau + 2j*np.pi*FREQUENCY_HZ)/sf)
        for regime, noise in SETTINGS:
            shape = (reps, n_trials, n)

            def complex_noise():
                return (rng.normal(size=shape) + 1j*rng.normal(size=shape))/np.sqrt(2)

            if regime == "transient":
                phase = np.exp(2j*np.pi*rng.random((reps, n_trials, 1)))
                g = phase*lam**np.arange(n)[None, None, :]
            else:
                innovations = np.sqrt(1 - abs(lam)**2)*complex_noise()
                innovations[:, :, 0] = (rng.normal(size=(reps, n_trials))
                                        + 1j*rng.normal(size=(reps, n_trials)))/np.sqrt(2)
                g = lfilter([1.], [1., -lam], innovations, axis=-1)
            observed = g + noise*complex_noise()
            for centered in [False, True]:
                x = observed - observed.mean(axis=-1, keepdims=True) if centered else observed
                for claim_factor in [1., 4.]:
                    claim = np.exp((-claim_factor*1000/tau + 2j*np.pi*FREQUENCY_HZ)/sf)
                    res1 = (np.linalg.norm(x[:, :, 1:] - claim*x[:, :, :-1], axis=-1)
                            / np.linalg.norm(x[:, :, :-1], axis=-1))
                    coherent = ((res1 <= config.RESIDUAL_THRESHOLD).sum(axis=1)
                                >= config.REQUIRED_HELD_OUT_TRIALS)
                    for h in HORIZONS_MS:
                        k = round(h*sf/1000)
                        gx, gy = x[:, :, :-k], x[:, :, k:]
                        mu = np.sum(gx.conj()*gy, axis=-1)/np.sum(abs(gx)**2, axis=-1)
                        gap = np.median(abs(mu), axis=1) - abs(claim)**k
                        for tol in TOLERANCES:
                            excluded = gap > tol
                            rows.append(dict(
                                tau_ms=tau, regime=regime, measurement_noise_std=noise,
                                mean_removed=centered, claim_factor=claim_factor,
                                horizon_ms=h, tolerance=tol, n_sets=reps,
                                n_coherent=int(coherent.sum()),
                                n_radial_excluded=int(excluded.sum()),
                                n_radial_excluded_coherent=int((excluded & coherent).sum()),
                                radial_exclusion_rate=excluded.mean(),
                                coherence_pass_rate=coherent.mean(),
                                radial_exclusion_given_coherence=(
                                    excluded[coherent].mean() if coherent.any() else np.nan)))
            print(f"tau={tau:g} ms, {regime}, noise={noise}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=500, help="nine-trial sets per cell")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/reference_results/supplement"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    simulate(args.reps).to_csv(args.out / "oracle_controls.csv", index=False)
    print("wrote", args.out / "oracle_controls.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
