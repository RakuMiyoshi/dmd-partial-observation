"""Single-trial Hankel DMD estimator and held-out evaluation for the ECoG data.

A model is fitted to one delay-embedded trial and its fixed eigenpairs are then
scored on held-out trials. Trial arrays use the (channels, time) convention.
Parameters are passed explicitly; their values are fixed in ``config``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eig

from . import config

EPS = 1e-12


# ----------------------------------------------------------------------
# Trial slicing and quality control
# ----------------------------------------------------------------------
def slice_trial(data, on, sf, *, pre_s, post_s):
    """Cut ``[on - pre_s, on + post_s)`` from a run; return the segment and time in ms."""
    a = max(0, on - round(pre_s * sf))
    b = min(data.shape[1], on + round(post_s * sf))
    t_ms = (np.arange(a, b) - on) / sf * 1e3
    return np.asarray(data[:, a:b], float), t_ms


def center(X):
    """Remove the per-channel mean within the window."""
    return X - np.nanmean(X, axis=1, keepdims=True)


def hard_window_qc(X, *,
                   finite_fraction_min=config.FINITE_FRACTION_MIN,
                   flat_std=config.FLAT_STD_UV):
    """Reject only clearly corrupted windows containing non-finite or flat data."""
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] < 2:
        return False
    finite_fraction = np.mean(np.isfinite(X), axis=1)
    channel_std = np.nanstd(X, axis=1)
    bad = (finite_fraction < finite_fraction_min) | (channel_std < flat_std)
    return bool(not np.any(bad))


def longest_true_run_ms(mask, sf):
    """Longest contiguous True interval per row, in milliseconds."""
    mask = np.asarray(mask, bool)
    if mask.ndim == 1:
        mask = mask[None, :]
    longest = np.zeros(mask.shape[0], dtype=int)
    for row_index, row in enumerate(mask):
        transitions = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        if len(starts):
            longest[row_index] = int(np.max(stops - starts))
    return 1000.0 * longest / float(sf)


def raw_epoch_qc(X, t_ms, sf, ch_names, *, event=None, pre_ms, post_scan_ms,
                 scale_floor_uv, plateau_eps_uv, plateau_warn_z, plateau_hard_z,
                 plateau_warn_ms, plateau_hard_ms, dc_window_ms, dc_warn_z,
                 dc_hard_uv, dc_absolute_hard_uv, dc_persist_ms):
    """Amplifier clipping and persistent-offset diagnostics for one raw epoch.

    Runs on the epoch *before* any window-wise mean removal, because both failure
    modes are absolute-level effects that centring hides:

    * **plateau / clipping** — a stretch of consecutive near-zero first differences
      while the level sits far from the channel's own prestimulus median. Measured
      as a robust z against that channel's prestimulus MAD, so a quiet channel is
      not flagged for a small excursion.
    * **persistent DC step** — the late-window median shifts away from the
      prestimulus median, confirmed by a same-sign run of at least
      ``dc_persist_ms`` at half that shift, so a single spike cannot trigger it.

    Thresholds are hand-set and dataset-specific (``config.RAW_QC``): treat the
    returned frame as the audit, not the verdict. Returns one row per channel with the raw statistics and
    ``warning_flag`` / ``hard_flag``.
    """
    X = np.asarray(X, float)
    t_ms = np.asarray(t_ms, float)
    if X.ndim != 2 or X.shape[1] != len(t_ms):
        raise ValueError(f"Raw epoch shape {X.shape} does not match t_ms={len(t_ms)}.")

    pre = (t_ms >= pre_ms[0]) & (t_ms < pre_ms[1])
    post_scan = (t_ms >= post_scan_ms[0]) & (t_ms < post_scan_ms[1])
    dc_window = (t_ms >= dc_window_ms[0]) & (t_ms < dc_window_ms[1])
    if not np.any(pre) or np.sum(post_scan) < 2 or not np.any(dc_window):
        raise ValueError("Raw epoch does not cover the configured QC windows.")

    pre_median = np.median(X[:, pre], axis=1)
    pre_scale = 1.4826 * np.median(np.abs(X[:, pre] - pre_median[:, None]), axis=1)
    pre_scale = np.maximum(pre_scale, scale_floor_uv)

    # Align the sample-wise excursion test and the time mask to the right-hand
    # sample of each first difference.
    dx = np.abs(np.diff(X, axis=1))
    excursion_z = np.abs(X[:, 1:] - pre_median[:, None]) / pre_scale[:, None]
    scan_diff = post_scan[1:][None, :]
    nearly_constant = dx <= plateau_eps_uv
    plateau_warn_run = longest_true_run_ms(
        nearly_constant & (excursion_z >= plateau_warn_z) & scan_diff, sf)
    plateau_hard_run = longest_true_run_ms(
        nearly_constant & (excursion_z >= plateau_hard_z) & scan_diff, sf)
    plateau_warning = plateau_warn_run >= plateau_warn_ms
    plateau_hard = ((plateau_warn_run >= plateau_hard_ms)
                    | (plateau_hard_run >= plateau_warn_ms))

    dc_values = X[:, dc_window] - pre_median[:, None]
    dc_delta = np.median(dc_values, axis=1)
    dc_z = np.abs(dc_delta) / pre_scale
    sustained = np.sign(dc_delta)[:, None] * dc_values >= 0.5 * np.abs(dc_delta)[:, None]
    dc_persistent_ms = longest_true_run_ms(sustained, sf)
    dc_warning = dc_z > dc_warn_z
    dc_hard = (
        (dc_warning & (np.abs(dc_delta) > dc_hard_uv))
        | ((np.abs(dc_delta) > dc_absolute_hard_uv)
           & (dc_persistent_ms >= dc_persist_ms))
    )

    hard_flag = plateau_hard | dc_hard
    frame = {
        "channel": list(ch_names),
        "pre_median_uV": pre_median,
        "pre_scale_uV": pre_scale,
        "plateau_warn_run_ms": plateau_warn_run,
        "plateau_hard_run_ms": plateau_hard_run,
        "plateau_warning": plateau_warning,
        "plateau_hard": plateau_hard,
        "dc_delta_uV": dc_delta,
        "dc_z": dc_z,
        "dc_persistent_ms": dc_persistent_ms,
        "dc_warning": dc_warning,
        "dc_hard": dc_hard,
        "warning_flag": plateau_warning | dc_warning | hard_flag,
        "hard_flag": hard_flag,
    }
    if event is not None:
        frame = {"event": int(event), **frame}
    return pd.DataFrame(frame)


# ----------------------------------------------------------------------
# Hankel embedding and the DMD estimator
# ----------------------------------------------------------------------
def hankel(X, d):
    """Stack ``d`` delayed copies of ``X``: shape ``(n_channels * d, n_times - d + 1)``."""
    X = np.asarray(X, float)
    _, m = X.shape
    if d < 1 or m <= d:
        raise ValueError(f"Invalid Hankel shape: X={X.shape}, d={d}")
    return np.vstack([X[:, i:m - d + 1 + i] for i in range(d)])


def lam_to_ftau(lam, sf):
    """lambda -> (frequency [Hz], decay tau [ms], continuous decay rate [1/s])."""
    omega = np.log(np.asarray(lam, np.complex128)) * sf
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_ms = np.where(omega.real < 0, -1e3 / omega.real, np.nan)
    return omega.imag / (2 * np.pi), tau_ms, omega.real


@dataclass
class Koopman:
    eigs: np.ndarray            # lambda_j
    xi: np.ndarray              # eigenfunction coefficients, K xi = lam xi
    basis: np.ndarray           # dictionary: phi(h) = basis^H h
    K: np.ndarray               # Phi_Y_train ~= Phi_X_train @ K
    modes_phys: np.ndarray      # (n_modes, n_ch) physical-channel Koopman modes,
                                #   regressed X_phys ~= gx @ modes_phys (raw scale)
    res_train: np.ndarray
    res_test: np.ndarray        # nan if no test set
    activation: np.ndarray      # mean_t |g_j(t)|^2 on train (eigenfunction energy)
    phys_norm: np.ndarray       # ||modes_phys_j||_2 (observed-space mode magnitude)
    weight: np.ndarray          # normalised modal component energy = activation*phys_norm^2
    freq_hz: np.ndarray
    tau_ms: np.ndarray
    decay_rate: np.ndarray
    rank: int
    svals: np.ndarray
    fit_error: float            # relative Frobenius error of the K fit
    retained_energy: float      # training HX energy retained by the selected rank
    rank_uncapped: int          # rank selected before applying the condition cap
    rank_cap: int               # maximum rank allowed by the condition floor
    rank_capped: bool

    @property
    def residual(self):
        """Test residual when a test set exists, else the in-sample one."""
        return self.res_train if np.all(np.isnan(self.res_test)) else self.res_test

    @property
    def has_test(self):
        return not np.all(np.isnan(self.res_test))


def _pairs(trials, d):
    """h_t -> h_{t+1} pairs built inside each trial, then concatenated."""
    HX, HY = [], []
    for i, X in enumerate(trials):
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("Each trial must be (n_channels, n_times).")
        H = hankel(X, d)
        if H.shape[1] < 2:
            raise ValueError(f"Trial {i} is too short after Hankel embedding.")
        HX.append(H[:, :-1])
        HY.append(H[:, 1:])
    if len({h.shape[0] for h in HX}) != 1:
        raise ValueError("All trials must share the same channel count.")
    return np.hstack(HX), np.hstack(HY)


def _phi(basis, H):
    """Dictionary evaluation, samples as rows: Phi[m, :] = phi(h_m)^T."""
    return (basis.conj().T @ H).T


def _residual(Phi_X, Phi_Y, lam, xi):
    """||Phi_Y xi - lam Phi_X xi|| / ||Phi_X xi||, per eigenpair."""
    gx = Phi_X @ xi
    gy = Phi_Y @ xi
    num = np.sum(np.abs(gy - lam[None, :] * gx) ** 2, axis=0)
    den = np.sum(np.abs(gx) ** 2, axis=0)
    return np.sqrt(num / np.maximum(den, EPS)).real, gx


def _rank_selection(S, rank, cond_floor=config.RANK_CONDITION_FLOOR):
    """Truncate at ``rank``, capped where ``s_i < cond_floor * s_max``.

    Returns the selected rank, the requested rank, the cap, and the fraction of
    the squared singular values that the selected rank retains.
    """
    uncapped = int(rank)
    cap = max(1, int(np.sum(S >= cond_floor * S[0])))
    r = max(1, min(uncapped, cap, len(S)))
    retained = float(np.sum(S[:r] ** 2) / max(np.sum(S ** 2), EPS))
    return r, uncapped, cap, retained


def fit_koopman(train, sf, *, d, rank, test=None):
    """Fit the reduced operator K on ``train`` trials at delay depth ``d``.

    The Hankel snapshots are projected on their leading ``rank`` left singular
    vectors (the POD dictionary), K is the least-squares map between successive
    projected snapshots, and its eigenpairs define the estimated modes. If
    ``test`` trials are given, the one-step residual of every eigenpair is also
    evaluated on them.

    OLS gives every Hankel snapshot pair the same weight; trials of equal length
    therefore carry equal total weight.

    Mode weighting: each mode j reconstructs a rank-1 observed-space component
    g_j(t) phi_j^T, where g_j = Phi_X @ xi_j is the eigenfunction value and phi_j
    is the physical-channel Koopman mode obtained by regressing the observed
    channels onto the eigenfunction values (X_phys ~= gx @ modes_phys). This
    regression fixes the left/right eigenvector scaling gauge without relying on
    scipy's unnormalised left eigenvectors. The modal component energy
    activation_j * ||phi_j||^2 is invariant to the arbitrary scale of xi_j. It is
    a per-mode self-energy, not a variance decomposition (Koopman modes are
    non-orthogonal, so components do not add up to the total observed energy).
    """
    HX, HY = _pairs(train, d)
    n_ch = HX.shape[0] // d
    U, S, _ = np.linalg.svd(HX, full_matrices=False)
    r, uncapped, cap, retained = _rank_selection(S, rank)
    basis = U[:, :r]

    Phi_X, Phi_Y = _phi(basis, HX), _phi(basis, HY)
    K = np.linalg.lstsq(Phi_X, Phi_Y, rcond=None)[0]
    lam, xi = eig(K)

    res_train, gx = _residual(Phi_X, Phi_Y, lam, xi)
    if test:
        HX_t, HY_t = _pairs(test, d)
        res_test, _ = _residual(_phi(basis, HX_t), _phi(basis, HY_t), lam, xi)
    else:
        res_test = np.full_like(res_train, np.nan)

    # Physical-channel Koopman modes: X_phys.T ~= gx @ modes_phys  (raw scale).
    # First n_ch Hankel rows are the instantaneous physical channels x_t.
    X_phys = HX[:n_ch, :].T                        # (n_snapshots, n_ch)
    modes_phys = np.linalg.lstsq(gx, X_phys, rcond=None)[0]   # (n_modes, n_ch)

    activation = np.nan_to_num(np.mean(np.abs(gx) ** 2, axis=0))
    phys_norm = np.linalg.norm(modes_phys, axis=1)
    weight = np.nan_to_num(activation * phys_norm ** 2)
    weight = weight / (weight.sum() + EPS)

    freq, tau, decay = lam_to_ftau(lam, sf)
    fit_error = float(
        np.linalg.norm(Phi_Y - Phi_X @ K) / max(np.linalg.norm(Phi_Y), EPS)
    )
    return Koopman(
        eigs=lam, xi=xi, basis=basis, K=K, modes_phys=modes_phys,
        res_train=res_train, res_test=res_test,
        activation=activation, phys_norm=phys_norm, weight=weight,
        freq_hz=freq, tau_ms=tau, decay_rate=decay,
        rank=r, svals=S, fit_error=fit_error,
        retained_energy=retained, rank_uncapped=uncapped, rank_cap=cap,
        rank_capped=bool(r < uncapped),
    )

def evaluate_fixed_modes(model, X, *, d, sf, initial_ms=None):
    """Score an already fitted Koopman model on one held-out trajectory.

    Nothing is re-estimated: ``basis``, ``xi`` and ``eigs`` stay exactly as fitted,
    so mode ``j`` of the fit owns its residual on every trajectory it is shown and
    no cross-fit mode matching is needed.

    Returns ``residual`` (the per-eigenpair validated DMD residual of that eigenpair on
    this trajectory) plus modal amplitudes in µV of per-channel RMS. The amplitude
    is invariant to the arbitrary eigenvector scale: under ``xi_j -> c xi_j`` the
    eigenfunction values scale as ``g_j -> c g_j`` while the regressed physical mode
    scales as ``phi_j -> phi_j / c``, so their product does not move.
    ``initial_ms`` additionally reports the amplitude over the first stretch of the
    window, which is what a fast-decaying mode contributes before it dies.
    """
    X = np.asarray(X, float)
    H = hankel(X, d)
    if H.shape[1] < 2:
        raise ValueError("Trajectory is too short after Hankel embedding.")
    Phi_X, Phi_Y = _phi(model.basis, H[:, :-1]), _phi(model.basis, H[:, 1:])
    residual, gx = _residual(Phi_X, Phi_Y, model.eigs, model.xi)

    amplitude_t = np.abs(gx) * (model.phys_norm / np.sqrt(X.shape[0]))[None, :]
    out = {
        "residual": residual,
        "modal_amplitude_uV": np.median(amplitude_t, axis=0).real,
    }
    if initial_ms is not None:
        n_initial = int(np.ceil(float(initial_ms) * float(sf) / 1e3))
        n_initial = min(max(n_initial, 1), amplitude_t.shape[0])
        out["initial_modal_amplitude_uV"] = np.median(
            amplitude_t[:n_initial], axis=0
        ).real
    return out


# ----------------------------------------------------------------------
# Multi-step decay diagnostics on held-out trials
# ----------------------------------------------------------------------
def eigenfunction_series(model, X, *, d):
    """Eigenfunction values g_j(t) of every fixed eigenpair on one trajectory.

    Nothing is re-estimated. Returns ``(n_snapshots, n_modes)``.
    """
    return ((model.basis.conj().T @ hankel(X, d)).T) @ model.xi


def lagged_regression(g, k):
    """Least-squares coefficient mu_j of g_j(t + k) on g_j(t), with raw energies.

    Returns ``(mu, energy, norm_ratio)``, where ``energy`` is sum_t |g_j(t)|^2
    over the regressors and ``norm_ratio`` is ||g_j(t + k)|| / ||g_j(t)||.
    Entries without usable energy are NaN; no aggregation or selection is done.
    """
    x, y = g[:-k], g[k:]
    energy = np.sum(abs(x)**2, axis=0)
    future_energy = np.sum(abs(y)**2, axis=0)
    valid = (energy > 1e-12) & np.isfinite(energy) & np.isfinite(future_energy)
    mu = np.full(energy.shape, np.nan+1j*np.nan, dtype=complex)
    ratio = np.full(energy.shape, np.nan)
    np.divide(np.sum(x.conj()*y, axis=0), energy, out=mu, where=valid)
    np.sqrt(np.divide(future_energy, energy, out=ratio, where=valid), out=ratio)
    return mu, energy, ratio


def held_out_decay_diagnostics(model, tests, *, d, horizons_k, n_common):
    """One-step residuals and k-step coefficients of a fit on each held-out trial.

    Axis order of the returned arrays: mode, held-out trial, lag, snapshot range.
    Range 0 uses the full eigenfunction series; range 1 keeps its first
    ``n_common`` snapshots, the number available at the deepest delay, so that
    delay depths can be compared on equal series lengths.

    The multi-step check of the Methods compares ``median_q |mu_j^(q)|`` at a
    lag of k samples with ``|lambda_j|^k``.
    """
    n = len(model.eigs)
    shape = (n, len(tests), len(horizons_k), 2)
    mu = np.full(shape, np.nan+1j*np.nan, dtype=complex)
    energy = np.full(shape, np.nan)
    ratio = np.full(shape, np.nan)
    pairs = np.zeros(shape, dtype=np.int32)
    res1 = np.zeros((n, len(tests)))
    for q, X in enumerate(tests):
        g = eigenfunction_series(model, X, d=d)
        res1[:, q] = np.linalg.norm(g[1:] - g[:-1]*model.eigs, axis=0) / np.maximum(
            np.linalg.norm(g[:-1], axis=0), 1e-12)
        for r, series in enumerate((g, g[:n_common])):
            for h, k in enumerate(horizons_k):
                if len(series) <= k:
                    continue
                mu[:, q, h, r], energy[:, q, h, r], ratio[:, q, h, r] = \
                    lagged_regression(series, k)
                pairs[:, q, h, r] = len(series) - k
    return dict(mu=mu, gx_energy=energy, norm_ratio=ratio, n_pairs=pairs, res1=res1)
