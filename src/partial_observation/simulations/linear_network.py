"""Linear simulation of Fig. 2A-C and Figs. S1-S2: a 24-state network seen through 8 channels.

Twelve damped oscillators (24 latent states) are wired as a layered network;
8 states are observed. Both regimes share the same spectrum:

* ``STRUCTURAL`` (idealised, Fig. S1): a feedforward network without noise.
  The passive baseline lies exactly in the slow eigenspace, so the fast modes
  are absent from it by construction and the P/Q bottlenecks are exact rank
  deficiencies.
* ``PRACTICAL`` (recurrent, Fig. S2): the same network plus a weak reverse
  edge on every forward edge, with process and observation noise. Every site
  then has full-rank Q, and recovery is set by conditioning and SNR.

An impulse at one of three sites (i, ii, iii) excites the network; Hankel DMD at
delay depth m is fitted to the observed channels, and estimates are matched to
the true eigenvalues within a fixed tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.optimize import linear_sum_assignment

from partial_observation.plotting import palette as RCFG  

# Practical-demo noise toggle.
# False -> clean practical trajectory; True -> finite-SNR system identification.
NOISE_ON = True
PROCESS_NOISE_STD = 1e-3
OBS_NOISE_STD = 1e-5

# Same impulse amplitude for the three stimulation sites.
ALPHA = 3.0

# For the noisy practical sweeps, average across realizations.
N_REALIZATIONS = 10

ALPHA_SWEEP = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
ALPHA_SWEEP_DELAY = 6

N_OSC = 12
N = 24
FS = 500.0
DT = 1.0 / FS
SEED = 2

# Frequency is deliberately decorrelated from decay rate.
FREQ_HZ = np.array([76, 35, 54, 83, 95, 28, 61, 12, 43, 17, 9, 4], dtype=float)

# One spectrum for both regimes, so the two demos are directly comparable.
# Nothing sits exactly on the unit circle: the "persistent" group is a long but finite
# decay, which is what a real system can actually have.
PERSISTENT_RHO = 0.999
RHO = np.array([
    0.90, PERSISTENT_RHO, 0.92, 0.95, PERSISTENT_RHO, 0.89,
    0.94, 0.91, 0.96, PERSISTENT_RHO, PERSISTENT_RHO, PERSISTENT_RHO,
], dtype=float)

# Anything above this radius counts as the persistent group in the figures.
PERSISTENT_THRESHOLD = 0.98

# 8 scalar observations out of 24.
OBS_STATES = np.array([0, 1, 2, 3, 12, 18, 19, 22], dtype=int)

# Same sites in both regimes.  In the ideal DAG they have different reachable
# sets; in the recurrent model all three are full-rank reachable but with very
# different controllability conditioning.
# Names run upstream to downstream: site A is the most upstream oscillator
# (broad reach, high rank Q), site C the most downstream (narrow reach).
STIM_OSC = {
    "site A": 0,   # upstream root  (rank Q = 24)
    "site B": 3,   # intermediate   (rank Q = 18)
    "site C": 6,   # downstream     (rank Q = 14)
}

T_PRE = 200
T_POST = 350
D_SWEEP = np.arange(1, 21)
D_DELAY_STRUCTURAL = 8
D_DELAY_PRACTICAL = 30

RANK_TOL = 1e-10
EIG_MATCH_TOL_STRUCTURAL = 1e-2
EIG_MATCH_TOL_PRACTICAL = 1e-2

# Weak reverse coupling in the practical recurrent model.
FEEDBACK_SCALE = 0.01             # upper bound; halved until the system is stable
TARGET_SPECTRAL_RADIUS = 0.9995   # must sit above PERSISTENT_RHO

# Figure palette, shared with the CCEP figures (utils/config.py).
# Mode groups: slowly decaying (rho=PERSISTENT_RHO) vs fast decaying (rho<PERSISTENT_THRESHOLD).
PERSISTENT_COLOR = RCFG.GROUP_COLORS["persistent"]
FAST_COLOR = RCFG.GROUP_COLORS["fast-decay"]
SPONT_COLOR = RCFG.WINDOW_COLORS["pre"]
# Stimulation sites are categorical. Green is kept for site A (it is unused by
# the rest of the paper) and the other two are chosen to stay clear of every
# semantic colour the site markers coexist with -- the mode groups (persistent
# purple, fast-decay orange) and the window colours (passive blue, perturbed
# red) -- while remaining mutually distinct and colour-blind legible: a teal and
# a warm brown.
SITE_COLORS = {"site A": "#2ca02c", "site B": "#159090", "site C": "#8c564b"}
CONDITION_COLORS = {"spontaneous": SPONT_COLOR, **SITE_COLORS}
MARKERS = {"spontaneous": "o", "site A": "s", "site B": "^", "site C": "D"}

# (label, stimulated oscillator) in one fixed order, used by every figure.
CONDITIONS = [("spontaneous", None)] + list(STIM_OSC.items())


# =============================================================================
# Network and matrices
# =============================================================================
# Directed oscillator-level graph: source -> target.  Nontrivial DAG with
# branches, merges and shortcuts.  target > source everywhere, so the structural
# A is block lower triangular and its spectrum is exactly the block spectra.
EDGES = [
    (0, 1, 0.30), (0, 2, 0.22), (0, 4, 0.18),
    (1, 3, 0.34), (1, 4, 0.26), (1, 6, 0.14),
    (2, 5, 0.32), (2, 6, 0.24), (2, 9, 0.12),
    (3, 6, 0.30), (3, 7, 0.20),
    (4, 5, 0.28), (4, 7, 0.32), (4, 10, 0.16),
    (5, 8, 0.28), (5, 10, 0.18),
    (6, 8, 0.35), (6, 9, 0.24),
    (7, 9, 0.31), (7, 10, 0.26),
    (8, 11, 0.36), (9, 11, 0.32), (10, 11, 0.27),
]

K_TEMPLATE_1 = np.array([[1.00, 0.25], [-0.18, 0.82]])
K_TEMPLATE_2 = np.array([[0.78, -0.22], [0.31, 0.95]])


def oscillator_block(rho: float, freq_hz: float) -> np.ndarray:
    """One oscillator = one complex-conjugate eigenvalue pair rho e^{+-i theta}."""
    theta = 2.0 * np.pi * freq_hz / FS
    return rho * np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])


def build_A(rhos: np.ndarray, feedback: float = 0.0) -> np.ndarray:
    """Construct the actual 24x24 A; feedback>0 adds a weak reverse edge per edge."""
    A = block_diag(*[oscillator_block(r, f) for r, f in zip(rhos, FREQ_HZ)])
    for k, (src, dst, gain) in enumerate(EDGES):
        K_fwd, K_rev = (K_TEMPLATE_1, K_TEMPLATE_2) if k % 2 == 0 else (K_TEMPLATE_2, K_TEMPLATE_1)
        A[2*dst:2*dst+2, 2*src:2*src+2] = gain * K_fwd
        if feedback > 0:
            # Weak reverse edge: makes reachability generic but poorly conditioned.
            A[2*src:2*src+2, 2*dst:2*dst+2] = feedback * gain * K_rev
    return A


def build_recurrent_A(rhos: np.ndarray, feedback: float = FEEDBACK_SCALE):
    """Weaken the feedback until the recurrent system is stable, then return (A, scale).

    Scaling the whole matrix instead would shift every prescribed eigenvalue, so
    the two regimes would no longer share a spectrum.  Shrinking only the reverse
    edges keeps the diagonal blocks — and therefore the prescribed rho — intact.
    """
    while feedback > 1e-6:
        A = build_A(rhos, feedback)
        if np.max(np.abs(np.linalg.eigvals(A))) < TARGET_SPECTRAL_RADIUS:
            return A, feedback
        feedback *= 0.5
    raise RuntimeError("no stable feedback scale found")


C = np.eye(N)[OBS_STATES]
# Input matrix: one column per stimulation site, so figures can show B next to
# A and C on the same 24-state axis.
B_SITES = np.column_stack([np.eye(N)[:, 2*osc] for osc in STIM_OSC.values()])


def oscillator_layers() -> np.ndarray:
    """Longest-path depth of every oscillator in the DAG (src < dst, so one pass)."""
    layer = np.zeros(N_OSC, dtype=int)
    for src, dst, _ in EDGES:
        layer[dst] = max(layer[dst], layer[src] + 1)
    return layer


def oscillator_positions() -> dict[int, tuple[float, float]]:
    """Layered layout: x = DAG depth, so upstream -> downstream reads left to right."""
    # Layer spacing is wider than the vertical spread because the panel is wide
    # and short: with equal spacing the layers ended up nearly touching in x
    # while the columns had room to spare in y.
    layer = oscillator_layers()
    pos = {}
    for lay in range(int(layer.max()) + 1):
        nodes = np.where(layer == lay)[0]
        ys = np.linspace(1.0, -1.0, len(nodes) + 2)[1:-1] if len(nodes) > 1 else [0.0]
        for node, y in zip(nodes, ys):
            pos[int(node)] = (2.6 * lay, 1.5 * float(y))
    return pos


POS = oscillator_positions()


# =============================================================================
# Generic linear-system utilities
# =============================================================================
def stimulation_vector(stim_osc: int, alpha: float = 1.0) -> np.ndarray:
    b = np.zeros(N)
    b[2 * stim_osc] = alpha
    return b


def observability_matrix(A: np.ndarray, d: int) -> np.ndarray:
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(d)])


def numerical_rank(M: np.ndarray, rel_tol: float = RANK_TOL) -> int:
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > s[0] * rel_tol)) if len(s) else 0


def delay_matrix(Y: np.ndarray, d: int) -> np.ndarray:
    n_col = Y.shape[1] - d + 1
    return np.vstack([Y[:, k:k+n_col] for k in range(d)])


def delay_dmd(Y: np.ndarray, d: int, fixed_rank: int | None):
    """Projected DMD on delay-embedded snapshots.

    fixed_rank=None -> rank from a relative singular-value floor.
    fixed_rank=N    -> keep N directions.  Under noise the numerical rank is full
      by construction, so this tests identifiability rather than a separate
      rank-selection problem.
    """
    H = delay_matrix(Y, d)
    X, Xp = H[:, :-1], H[:, 1:]
    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    r = numerical_rank(X) if fixed_rank is None else fixed_rank
    r = max(1, min(r, len(s)))

    A_tilde = U[:, :r].T @ Xp @ Vh[:r, :].T @ np.diag(1.0 / s[:r])
    return np.linalg.eigvals(A_tilde), r


def match_mask(true_eigs: np.ndarray, est_eigs: np.ndarray, tol: float) -> np.ndarray:
    """Boolean mask of true eigenvalues uniquely matched within tol."""
    D = np.abs(true_eigs[:, None] - est_eigs[None, :])
    rows, cols = linear_sum_assignment(D)
    mask = np.zeros(len(true_eigs), dtype=bool)
    mask[rows] = D[rows, cols] < tol
    return mask


def is_persistent(radii: np.ndarray) -> np.ndarray:
    """Persistent (long-lived) vs transient (quickly decaying) mode group."""
    return np.asarray(radii) > PERSISTENT_THRESHOLD


# =============================================================================
# Regimes: one object carries everything that differs between the two demos
# =============================================================================
@dataclass
class Regime:
    key: str
    label: str
    rhos: np.ndarray
    recurrent: bool
    d_delay: int
    dmd_rank: int | None      # None -> numerical rank; int -> keep that many modes
    match_tol: float
    n_reps: int
    noisy: bool
    A: np.ndarray = field(init=False)
    feedback: float = field(init=False)   # reverse-edge scale actually used
    lam: np.ndarray = field(init=False)   # true eigenvalues, aligned with V / W
    V: np.ndarray = field(init=False)     # right eigenvectors, unit norm
    W: np.ndarray = field(init=False)     # rows are the left eigenvectors w_i^H

    def __post_init__(self):
        self.A, self.feedback = (build_recurrent_A(self.rhos) if self.recurrent
                                 else (build_A(self.rhos), 0.0))
        lam, V = np.linalg.eig(self.A)
        V = V / np.linalg.norm(V, axis=0)
        self.lam, self.V, self.W = lam, V, np.linalg.inv(V)
        self._sigma = None
        if self.recurrent:
            sig = solve_discrete_lyapunov(self.A, np.eye(N))
            self._sigma = 0.5 * (sig + sig.T)

    # -- noise levels ---------------------------------------------------------
    @property
    def pnoise(self) -> float:
        return PROCESS_NOISE_STD if (self.noisy and NOISE_ON) else 0.0

    @property
    def onoise(self) -> float:
        return OBS_NOISE_STD if (self.noisy and NOISE_ON) else 0.0

    @property
    def noise_note(self) -> str:
        return (f"noise on: process={PROCESS_NOISE_STD:g}, obs={OBS_NOISE_STD:g}"
                if (self.pnoise or self.onoise) else "noise off")

    # -- baseline state -------------------------------------------------------
    def initial_state(self, seed: int) -> np.ndarray:
        if not self.recurrent:
            # Exact state in the slow eigenspace: intentionally idealized, so the
            # fast modes are absent from the spontaneous baseline by construction.
            # c v + conj(c v) = 2 Re(c v), so the sum stays real throughout.
            rng = np.random.default_rng(SEED)
            x = np.zeros(N)
            for j, lam in enumerate(self.lam):
                if abs(lam) > PERSISTENT_THRESHOLD and lam.imag > 0:
                    c = rng.uniform(0.8, 1.2) * np.exp(1j * rng.uniform(0, 2*np.pi))
                    x += 2.0 * np.real(c * self.V[:, j])
            return 3.0 * x / np.linalg.norm(x)

        rng = np.random.default_rng(seed)
        x = rng.multivariate_normal(np.zeros(N), self._sigma)
        # Normalize to an O(1) observed baseline for an interpretable SNR scale.
        probe = np.column_stack([x, self.A @ x, self.A @ self.A @ x])
        return x / max(np.std(C @ probe), 1e-12)

    # -- trajectories ---------------------------------------------------------
    def trajectory(self, stim_osc: int | None, alpha: float = ALPHA, seed: int = 0):
        """Return (state at the moment of stimulation, observed post-stim window)."""
        rng = np.random.default_rng(seed)
        x = self.initial_state(seed + 10000)
        for _ in range(T_PRE):
            x = self.A @ x + self.pnoise * rng.standard_normal(N)

        if stim_osc is not None:
            x = x + stimulation_vector(stim_osc, alpha)
        x0 = x.copy()

        X = np.empty((N, T_POST))
        for k in range(T_POST):
            X[:, k] = x
            x = self.A @ x + self.pnoise * rng.standard_normal(N)

        Y = C @ X
        if self.onoise > 0:
            Y = Y + self.onoise * rng.standard_normal(Y.shape)
        return x0, Y

    def paired_timeseries(self, stim_osc: int, alpha: float = ALPHA, seed: int = 123):
        """Passive and perturbed branches sharing one noise realization.

        Each branch returns (all N latent states, the 8 measured channels), so a
        figure can show what the measurement leaves out alongside what it keeps.
        """
        rng = np.random.default_rng(seed)
        x = self.initial_state(seed + 10000)
        Xpre = np.empty((N, T_PRE))
        for k in range(T_PRE):
            Xpre[:, k] = x
            x = self.A @ x + self.pnoise * rng.standard_normal(N)

        W_post = self.pnoise * rng.standard_normal((N, T_POST))
        V_post = self.onoise * rng.standard_normal((len(OBS_STATES), T_POST))

        def branch(x0):
            X = np.empty((N, T_POST))
            xx = x0.copy()
            for k in range(T_POST):
                X[:, k] = xx
                xx = self.A @ xx + W_post[:, k]
            return (np.hstack([Xpre, X]),
                    np.hstack([C @ Xpre, C @ X + V_post]))

        t = (np.arange(T_PRE + T_POST) - T_PRE) / FS
        return t, branch(x), branch(x + stimulation_vector(stim_osc, alpha))

    # -- per-mode P / Q bookkeeping ------------------------------------------
    def mode_weights(self, x0: np.ndarray):
        """(observability ||C v_i||, excitation |w_i^H x0|) for every mode."""
        return np.linalg.norm(C @ self.V, axis=0), np.abs(self.W @ x0)

    def pair_index(self) -> np.ndarray:
        """One representative index per conjugate pair, ordered by frequency."""
        idx = np.where(self.lam.imag >= 0)[0]
        return idx[np.argsort(np.angle(self.lam[idx]))]

    def seeds(self) -> range:
        return range(self.n_reps)


STRUCTURAL = Regime(
    key="structural", label="idealised feedforward setting",
    rhos=RHO, recurrent=False,
    d_delay=D_DELAY_STRUCTURAL, dmd_rank=None,
    match_tol=EIG_MATCH_TOL_STRUCTURAL, n_reps=1, noisy=False,
)
PRACTICAL = Regime(
    key="practical", label="recurrent noisy setting",
    rhos=RHO, recurrent=True,
    d_delay=D_DELAY_PRACTICAL, dmd_rank=N,
    match_tol=EIG_MATCH_TOL_PRACTICAL, n_reps=N_REALIZATIONS, noisy=True,
)


# Recovery measurements (identical code for both regimes)
def n_recovered(reg: Regime, stim_osc, alpha: float, d: int, seed: int) -> int:
    _, Y = reg.trajectory(stim_osc, alpha, seed=seed)
    eigs, _ = delay_dmd(Y, d, reg.dmd_rank)
    return int(match_mask(reg.lam, eigs, reg.match_tol).sum())


def _sweep(reg: Regime, stim_osc, alphas, ds, seed0: int):
    """Mean and sd of the number of matched eigenvalues over (alpha, d) pairs."""
    counts = np.array([[n_recovered(reg, stim_osc, float(a), int(d), seed0 + rep)
                        for a, d in zip(alphas, ds)] for rep in reg.seeds()], dtype=float)
    return counts.mean(axis=0), counts.std(axis=0)


def recovery_curve(reg: Regime, stim_osc, alpha: float = ALPHA, ds=D_SWEEP):
    return _sweep(reg, stim_osc, np.full(len(ds), alpha), ds, seed0=100)


def alpha_curve(reg: Regime, stim_osc, d: int = ALPHA_SWEEP_DELAY, alphas=ALPHA_SWEEP):
    return _sweep(reg, stim_osc, alphas, np.full(len(alphas), d), seed0=500)


# =============================================================================
# Figures: network
# =============================================================================


# =============================================================================
# Figures: the actual matrices
# =============================================================================


# =============================================================================
# Figures: perturbation propagation
# =============================================================================


# =============================================================================
# Figures: time series
# =============================================================================
UNOBSERVED_COLOR = "0.78"
# Default cycle colour C7 is a grey that reads as "unobserved" here, so drop it.
# to_hex: matplotlib >= 3.11 returns the cycle colours as RGBA tuples.
CHANNEL_COLORS = [to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
                  if to_hex(c) not in ("#7f7f7f", "#bcbd22")]


# =============================================================================
# Figures: eigenvalue recovery on the unit circle
# =============================================================================
UNIT_CIRCLE = np.exp(1j * np.linspace(0, 2*np.pi, 500))


def _eig_plane(ax, reg: Regime, eigs: np.ndarray, est_color="black",
               est_label="DMD estimate", true_s=61, est_s=47,
               true_lw=1.5, est_lw=1.3):
    """One complex-plane panel: unit circle, true modes by group, DMD estimates.

    ``true_s`` / ``est_s`` set the marker areas; the defaults match the earlier
    figures, and a caller with a dense spectrum can pass smaller values.
    """
    slow = is_persistent(np.abs(reg.lam))
    ax.plot(UNIT_CIRCLE.real, UNIT_CIRCLE.imag, ls="--", lw=1.0, color="0.72")
    for mask, color, label in ((slow, PERSISTENT_COLOR, "true slow-decaying"),
                               (~slow, FAST_COLOR, "true fast-decaying")):
        ax.scatter(reg.lam[mask].real, reg.lam[mask].imag, s=true_s,
                   facecolors="white", edgecolors=color, linewidths=true_lw,
                   label=label)
    ax.scatter(eigs.real, eigs.imag, marker="x", s=est_s, color=est_color,
               linewidths=est_lw, label=est_label)
    ax.axhline(0.0, color="0.88", lw=0.7)
    ax.axvline(0.0, color="0.88", lw=0.7)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.07, 1.07)
    ax.set_ylim(-1.07, 1.07)
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")


# =============================================================================
# Figures: sweeps
# =============================================================================


# =============================================================================
# Console diagnostics
# =============================================================================
