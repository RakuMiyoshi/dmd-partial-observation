"""Validation of eigenvalue estimates (Methods: held-out residual and decay checks).

A candidate eigenpair from a single-trial fit is *validated* when

1. it is stable with a non-negative frequency up to ``MAX_FREQUENCY_HZ``;
2. its one-step held-out residual is at most ``RESIDUAL_THRESHOLD`` in at least
   ``REQUIRED_HELD_OUT_TRIALS`` of the nine held-out trials; and
3. it passes multi-step decay validation: over a lag of k samples, the median
   held-out decay ``median_q |mu_j^(q)|`` exceeds the claimed ``|lambda_j|^k``
   by at most ``DECAY_VALIDATION_TOLERANCE`` (one-sided).

The functions take the per-route decay arrays written by
``scripts/ecog/01_fit_cohort.py`` (``decay/<site>__<route>.npz``). Besides the
primary rule, ``CASES`` lists the sensitivity settings of Fig. S5.
"""
from __future__ import annotations

import numpy as np

from . import config

# (name, horizon in ms, tolerance, rule, snapshot range). ``baseline`` omits
# the decay check; ``symmetric`` bounds |gap|; ``common_*`` evaluates the gap on
# the snapshot count shared by all delay depths (range 1 of the arrays).
CASES = [("baseline", 0., 0., "none", 0), ("symmetric_5ms_005", 5., .05, "symmetric", 0)]
CASES += [(f"one_{h}ms_{int(t*100):03d}", float(h), t, "one", 0)
          for h in [2, 5, 10] for t in [.05, .10]]
CASES += [(f"common_{h}ms_{int(t*100):03d}", float(h), t, "one", 1)
          for h in [2, 5, 10] for t in [.05, .10]]
PRIMARY_CASE = (f"one_{config.DECAY_VALIDATION_HORIZON_MS:g}ms_"
                f"{int(config.DECAY_VALIDATION_TOLERANCE*100):03d}")


def rate_and_frequency(eig, sf):
    """Continuous-time decay rate [1/s] and frequency [Hz] of discrete eigenvalues."""
    rate = np.log(np.abs(eig)) * sf
    freq = np.angle(eig) * sf / (2*np.pi)
    return rate, freq


def in_band_stable(rate, freq):
    """Stable, with 0 <= f <= MAX_FREQUENCY_HZ."""
    return (np.isfinite(rate) & np.isfinite(freq) & (rate < 0) & (freq >= 0)
            & (freq <= config.MAX_FREQUENCY_HZ))


def residual_pass(res1, threshold=config.RESIDUAL_THRESHOLD,
                  required=config.REQUIRED_HELD_OUT_TRIALS):
    """Residual criterion over the held-out trials (axis 1 of ``res1``)."""
    return (res1 <= threshold).sum(axis=1) >= required


def decay_gap(z, horizon_ms=config.DECAY_VALIDATION_HORIZON_MS, snapshot_range=0):
    """``median_q |mu_j^(q)| - |lambda_j|^k`` at the requested horizon."""
    ix = int(np.flatnonzero(z["horizon_requested_ms"] == horizon_ms)[0])
    return (np.median(np.abs(z["mu"]), axis=1)[:, ix, snapshot_range]
            - np.abs(z["eig"]) ** z["horizon_k"][ix])


def decisions(z):
    """Selection masks of every case in ``CASES`` for one site and route.

    Returns ``(rate, freq, eligible, masks)``: ``eligible`` passes criteria 1-2,
    and ``masks[name] = (validated, missing)``, where ``missing`` marks eligible
    candidates whose decay gap is undefined.
    """
    eig = z["eig"]
    rate, freq = rate_and_frequency(eig, float(z["sf_hz"]))
    eligible = in_band_stable(rate, freq) & residual_pass(z["res1"])
    med = np.median(abs(z["mu"]), axis=1)
    masks = {}
    for name, h, t, rule, r in CASES:
        if rule == "none":
            masks[name] = (eligible, np.zeros(len(eig), bool))
        else:
            ix = int(np.flatnonzero(z["horizon_requested_ms"] == h)[0])
            gap = med[:, ix, r] - abs(eig) ** z["horizon_k"][ix]
            valid = np.isfinite(gap)
            accept = gap <= t if rule == "one" else abs(gap) <= t
            masks[name] = (eligible & valid & accept, eligible & ~valid)
    return rate, freq, eligible, masks


def validated(z):
    """Primary rule: criteria 1-3 at the primary horizon and tolerance."""
    rate, freq = rate_and_frequency(z["eig"], float(z["sf_hz"]))
    gap = decay_gap(z)
    return (in_band_stable(rate, freq) & residual_pass(z["res1"]) & np.isfinite(gap)
            & (gap <= config.DECAY_VALIDATION_TOLERANCE))
