"""Supplementary Figs. S1-S2: the linear simulation in more detail.

* Fig. S1 (``figS1_sim_ideal``): idealised feedforward network. The stimulation
  site sets an exact rank-Q ceiling that delay embedding cannot exceed.
* Fig. S2 (``figS2_sim_recurrent``): recurrent network with process and
  observation noise. Rank Q is full at every site, yet the site still orders
  recovery; recovery is also shown against impulse amplitude.

The simulations use fixed seeds and need no data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from partial_observation.plotting import style as figstyle
from partial_observation.simulations import linear_network as sim

figstyle.use_style()
FS = figstyle

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
N_LATENT = sim.N                      # n
ALPHA = sim.ALPHA
D_SWEEP = sim.D_SWEEP
KRYLOV_HORIZON = 80
# Printed names follow the main-text figure; sim keys stay "site A/B/C".
SITE_LABELS = {"site A": "site i", "site B": "site ii", "site C": "site iii",
               "spontaneous": "passive"}


def label(name: str) -> str:
    return SITE_LABELS.get(name, name)


def _ceiling(ax):
    ax.axhline(N_LATENT, color="0.55", ls=":", lw=1.0)
    ax.text(0.008, N_LATENT + 0.6, f"all {N_LATENT}",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=FS.ANNOT, color="0.4")


def _recovery_axes(ax, xlabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("matched true eigenvalues")
    ax.set_ylim(0, N_LATENT + 3.0)
    ax.set_title(title, fontsize=FS.TITLE)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=FS.LEGEND, loc="lower right", framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def rank_Q(reg, stim_osc, seed: int = 100) -> int:
    """rank of Q = [x0, A x0, ...] for the trajectory actually fitted."""
    x0, _ = reg.trajectory(stim_osc, ALPHA, seed=seed)
    K = np.empty((N_LATENT, KRYLOV_HORIZON))
    v = x0.copy()
    for k in range(KRYLOV_HORIZON):
        K[:, k] = v
        v = reg.A @ v
    return sim.numerical_rank(K)


def panel_delay_sweep(ax, reg, *, label_plateau: bool):
    plateaus = {}
    for name, osc in sim.CONDITIONS:
        mean, std = sim.recovery_curve(reg, osc, ALPHA)
        ax.errorbar(D_SWEEP, mean, yerr=std if reg.n_reps > 1 else None,
                    marker=sim.MARKERS[name], ms=4.0, lw=1.7, capsize=2.0,
                    color=sim.CONDITION_COLORS[name], label=label(name))
        plateaus[name] = float(mean[-1])

    # rank(P_m) depends on m alone, so it rides on the tick labels
    rankP = [sim.numerical_rank(sim.observability_matrix(reg.A, int(d)))
             for d in D_SWEEP]
    ax.set_xticks(D_SWEEP[::2],
                  [f"{d}\n{rp}" for d, rp in zip(D_SWEEP[::2], rankP[::2])])
    if label_plateau:
        for name, osc in sim.CONDITIONS:
            ax.annotate(f"rank$\\,Q={rank_Q(reg, osc)}$",
                        xy=(D_SWEEP[-1], plateaus[name]),
                        xytext=(-2, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=FS.SMALL,
                        color=sim.CONDITION_COLORS[name])
    return plateaus, rankP


SPECTRUM_CONDITIONS = [("spontaneous", None)] + list(sim.STIM_OSC.items())
IDEAL_DEPTH = 8          # rank P_m is full from here (Methods, main-text Fig. 3)
RECURRENT_DEPTH = 20     # deepest depth of the sweep, as quoted in the text
# A realisation of the sweep (seeds 100-109) whose counts follow the mean
# ordering (10, 14, 18, 22 against means 10, 14, 18, 21.2); seed 100 is an
# unrepresentative low draw for site i.
RECURRENT_SEED = 101


def spectrum_panel(ax, reg, stim_osc, d, *, seed, show_y):
    """True eigenvalues and Hankel-DMD estimates for one condition."""
    _, Y = reg.trajectory(stim_osc, ALPHA, seed=seed)
    eigs, _ = sim.delay_dmd(Y, d, reg.dmd_rank)
    recovered = int(sim.match_mask(reg.lam, eigs, reg.match_tol).sum())
    sim._eig_plane(ax, reg, eigs, est_label="estimate", true_s=26, est_s=22,
                   true_lw=1.0, est_lw=0.9)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    if not show_y:
        ax.set_ylabel(""); ax.set_yticklabels([])
    return recovered


def spectra_row(fig, spec, reg, d, letters, *, seed):
    """One spectrum per condition: passive, then sites i, ii, iii."""
    row = spec.subgridspec(1, len(SPECTRUM_CONDITIONS), wspace=0.12)
    axes = []
    for col, (name, osc) in enumerate(SPECTRUM_CONDITIONS):
        ax = fig.add_subplot(row[0, col])
        rec = spectrum_panel(ax, reg, osc, d, seed=seed, show_y=col == 0)
        cond = "passive" if osc is None else label(name)
        colour = sim.CONDITION_COLORS[name]
        ax.set_title(f"{cond}: rank$\\,Q={rank_Q(reg, osc)}$\nrecovered {rec}/{N_LATENT}",
                     fontsize=FS.LABEL, color=colour)
        ax.text(-0.10 if col == 0 else -0.03, 1.18, f"({next(letters)})",
                transform=ax.transAxes, fontweight="bold", fontsize=FS.LETTER,
                va="bottom", ha="right")
        axes.append(ax)
    handles, labels_ = axes[0].get_legend_handles_labels()
    return axes, handles, labels_


def figure_ideal(out: Path):
    """Idealised feedforward setting: site sets an exact rank-Q ceiling."""
    from partial_observation.plotting import linear_figure_grid as network
    reg = sim.STRUCTURAL
    fig = plt.figure(figsize=(10.0, 8.8))
    outer = fig.add_gridspec(3, 1, height_ratios=(0.95, 1.0, 1.05), hspace=0.62)
    letters = iter("ABCDEF")
    ax = fig.add_subplot(outer[0])
    network.panel_state_network(ax, reg, title="feedforward network and stimulation sites")
    ax.text(-0.01, 1.04, f"({next(letters)})", transform=ax.transAxes,
            fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
    axes, handles, labels_ = spectra_row(fig, outer[1], reg, IDEAL_DEPTH, letters, seed=100)
    fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False,
               fontsize=FS.LEGEND, bbox_to_anchor=(0.5, axes[0].get_position().y0 - 0.035))
    bottom = outer[2].subgridspec(1, 3, width_ratios=(0.25, 1.0, 0.25))
    ax = fig.add_subplot(bottom[0, 1])
    plateaus, _ = panel_delay_sweep(ax, reg, label_plateau=True)
    _ceiling(ax)
    _recovery_axes(ax, "delay depth $m$   /   rank$\\,P_m$",
                   "recovery against delay depth")
    ax.text(-0.09, 1.05, f"({next(letters)})", transform=ax.transAxes,
            fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
    # Legend below the axes, as in the recurrent figure.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
              frameon=False, fontsize=FS.LEGEND, handlelength=1.8,
              columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS1_sim_ideal.{ext}", bbox_inches="tight", dpi=200)
    print("wrote figS1_sim_ideal")
    return plateaus


def figure_recurrent(out: Path):
    """Recurrent noisy setting: rank Q is full, yet the site ordering remains."""
    from partial_observation.plotting import linear_figure_grid as network
    reg = sim.PRACTICAL
    fig = plt.figure(figsize=(10.0, 8.8))
    outer = fig.add_gridspec(3, 1, height_ratios=(0.95, 1.0, 1.05), hspace=0.62)
    letters = iter("ABCDEFG")
    ax = fig.add_subplot(outer[0])
    network.panel_state_network(
        ax, reg, title=f"network with weak reverse edges ($\\varepsilon={reg.feedback:g}$) and noise",
        reverse_examples=[(0, 2), (1, 3), (5, 8), (6, 9), (10, 11)])
    ax.text(-0.01, 1.04, f"({next(letters)})", transform=ax.transAxes,
            fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
    axes, handles, labels_ = spectra_row(fig, outer[1], reg, RECURRENT_DEPTH, letters,
                                         seed=RECURRENT_SEED)
    fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False,
               fontsize=FS.LEGEND, bbox_to_anchor=(0.5, axes[0].get_position().y0 - 0.035))
    bottom = outer[2].subgridspec(1, 2, wspace=0.25)
    ax = fig.add_subplot(bottom[0, 0])
    plateaus, _ = panel_delay_sweep(ax, reg, label_plateau=False)
    # Depth of the spectra in (B-E) and of the amplitude sweep in (G).
    ax.axvline(RECURRENT_DEPTH, color="0.35", ls=":", lw=1.1)
    ax.text(RECURRENT_DEPTH - 0.3, 1.0, f"$m={RECURRENT_DEPTH}$", ha="right",
            va="bottom", fontsize=FS.SMALL, color="0.35")
    _ceiling(ax)
    _recovery_axes(ax, "delay depth $m$   /   rank$\\,P_m$",
                   "recovery against delay depth (10 realisations)")
    ax.text(-0.12, 1.05, f"({next(letters)})", transform=ax.transAxes,
            fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
    ax2 = fig.add_subplot(bottom[0, 1])
    # Same depth as the spectra: at shallower m the site curves still cross.
    d = RECURRENT_DEPTH
    spont = float(np.mean([sim.n_recovered(reg, None, 0.0, d, 500 + rep)
                           for rep in reg.seeds()]))
    ax2.axhline(spont, color=sim.SPONT_COLOR, ls="--", lw=1.4,
                label=f"passive ({spont:.1f})")
    for name, osc in sim.STIM_OSC.items():
        mean, std = sim.alpha_curve(reg, osc, d)
        ax2.errorbar(sim.ALPHA_SWEEP, mean, yerr=std, marker=sim.MARKERS[name],
                     ms=4.0, lw=1.7, capsize=2.0, color=sim.CONDITION_COLORS[name],
                     label=label(name))
    _ceiling(ax2)
    _recovery_axes(ax2, r"impulse amplitude $\alpha$", f"recovery against impulse amplitude ($m={d}$)")
    ax2.text(-0.12, 1.05, f"({next(letters)})", transform=ax2.transAxes,
             fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
    # Legends below the axes: inside, they covered the right ends of the curves.
    for a in (ax, ax2):
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
                 frameon=False, fontsize=FS.LEGEND, handlelength=1.8,
                 columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS2_sim_recurrent.{ext}", bbox_inches="tight", dpi=200)
    print("wrote figS2_sim_recurrent")
    return {"plateaus": plateaus, "alpha_passive": spont}


def main(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = {"ideal": figure_ideal(out), "recurrent": figure_recurrent(out)}
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
