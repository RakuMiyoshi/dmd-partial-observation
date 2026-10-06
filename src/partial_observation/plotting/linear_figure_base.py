"""Linear-simulation panels and constants shared by Figs. 2, S1 and S2.

Presentation layer for ``partial_observation.simulations.linear_network``,
idealised (feedforward) regime. Every number and colour comes from that module
-- the network, the trajectories, the DMD fits and the palette -- so the figures
and the simulation cannot drift apart. This module holds the eigenvalue-plane
panel, the rank-P / rank-Q helpers and the site labels.
"""
from __future__ import annotations

import numpy as np

from partial_observation.plotting import style as figstyle
from partial_observation.simulations import linear_network as sim

figstyle.use_style()
FS = figstyle


REG = sim.STRUCTURAL          # the ideal regime: P/Q limits are exact ranks
SITE = "site A"               # broad/upstream site (rank Q = 24) for the 2x2
DELAY = REG.d_delay           # delay depth at which rank(P_d) is full
ALPHA = sim.ALPHA
N_LATENT = sim.N                    # n in the manuscript notation
N_OBS = len(sim.OBS_STATES)         # p
RANK = sim.N                        # r, the DMD truncation rank
SEED = 777
KRYLOV_HORIZON = 80           # same horizon the simulation uses for rank(Q)

GREEN = "#2e7d32"             # "both factors full" accent
# Display names for the simulation's site keys. The keys in sim.STIM_OSC stay
# as they are; only what is printed on the figure comes from here.
SITE_LABELS = {"site A": "site A", "site B": "site B", "site C": "site C"}


def site_label(name: str, short: bool = False) -> str:
    """Printed name of a site; ``short`` drops the leading "site "."""
    label = SITE_LABELS[name]
    return label.replace("site ", "") if short else label


RANK_FS = FS.LABEL            # rank P / rank Q values on the axis arrows


def rank_Q(stim_osc, seed: int = SEED) -> int:
    """rank of Q = [x0, A x0, ...] for the trajectory actually fitted.

    This is the Q of H = P Q, built from the initial state the fit sees: the
    spontaneous baseline for the passive condition, baseline plus impulse for
    the perturbed one. It is not the reachable subspace of the impulse alone,
    which would ignore whatever the baseline already excites.
    """
    x0, _ = REG.trajectory(stim_osc, ALPHA, seed=seed)
    K = np.empty((N_LATENT, KRYLOV_HORIZON))
    v = x0.copy()
    for k in range(KRYLOV_HORIZON):
        K[:, k] = v
        v = REG.A @ v
    return sim.numerical_rank(K)


# =============================================================================
# One cell of the eigenvalue-recovery grid
# =============================================================================
def panel_eigs(ax, stim_osc, d: int, *, highlight: bool,
               show_x: bool = True, show_y: bool = True,
               est_color: str = "black"):
    """One cell of the 2x2 grid. Returns (recovered, rank P_d, rank Q).

    Axis labels are drawn only on the outer edge of the grid: the top row's
    Re(lambda) label would otherwise sit on the bottom row's title.
    """
    _, Y = REG.trajectory(stim_osc, ALPHA, seed=SEED)
    eigs, _ = sim.delay_dmd(Y, d, REG.dmd_rank)
    recovered = int(sim.match_mask(REG.lam, eigs, REG.match_tol).sum())
    rP = sim.numerical_rank(sim.observability_matrix(REG.A, d))
    rQ = rank_Q(stim_osc)

    if highlight:
        ax.set_facecolor("#eef6ee")
        for spine in ax.spines.values():
            spine.set_edgecolor(GREEN)
            spine.set_linewidth(1.8)

    # The 24 eigenvalues sit in a tight arc near |lambda|=1, so smaller markers
    # than the shared default keep them from merging into a blob.
    sim._eig_plane(ax, REG, eigs, est_color=est_color, est_label="Hankel-DMD",
                   true_s=30, est_s=26, true_lw=1.1, est_lw=1.0)
    if not show_x:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if not show_y:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    # Recovered-count titles are omitted; the counts are reported in the text.
    return recovered, rP, rQ
