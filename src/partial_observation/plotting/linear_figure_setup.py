"""Setting panels of Fig. 2A-B: what each lever of H = P Q changes in the data.

(A) Observability / delay depth: the delay-embedded data matrix the fit sees,
H_1 (8 x 1 rows) and H_8 (8 x 8 rows), on one baseline so the heights compare
the row counts. Both are cut from the same impulse trajectory.
(B) Excitation / impulse: the 8 observed channels around t = 0 without and
with the impulse at site i, from one shared noise realisation and on one common
scale, so the difference between the panels is the impulse alone.
"""
from __future__ import annotations


import numpy as np


from partial_observation.plotting import linear_figure_base as base
from partial_observation.plotting import linear_figure_grid as grid
from partial_observation.simulations import linear_network as sim

FS = base.FS

FIG_W, FIG_H = 6.8, 4.6
LABEL_X, LABEL_Y = -0.62, 1.02   # panel letters, in first-panel axes fraction
HEAD_X = -0.30                   # rotated row label, in first-panel axes fraction
HEAD_W = 0.22                    # width of the (empty) header column, in panel widths
PANEL_WSPACE = 0.25              # gap between the columns of each row
RANK_Y = -0.10                   # rank P under each delay matrix, axes fraction
RANK_Q_Y = -0.115                # rank Q under each trace panel, just below its caption


def draw_settings(fig, spec, letters=("A", "B"), hspace: float = 0.55,
                  box_aspect: float | None = None, pad: float = 0.0):
    """The two setting rows inside ``spec``: delay matrices (first letter) and
    passive / impulse traces (second letter). ``box_aspect`` (height / width)
    pins the panels' shape when the block is given a tall, narrow slot, and
    ``pad`` (in row heights) adds an empty row below so the two rows are pushed
    to the top of the slot instead of spreading over it. Returns (rP, rQ)."""
    stim = sim.STIM_OSC[base.SITE]
    rP = {d: sim.numerical_rank(sim.observability_matrix(base.REG.A, d))
          for d in (1, base.DELAY)}
    rQ = {"passive": base.rank_Q(None), "perturbed": base.rank_Q(stim)}
    if pad > 0:
        outer = spec.subgridspec(3, 1, height_ratios=[1.0, 1.0, pad], hspace=hspace)
    else:
        outer = spec.subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=hspace)

    # ---- delay depth ----------------------------------------------------------
    # The header column is only spacing; letters and row labels are placed
    # relative to the first panel, so they follow it when the panels are
    # anchored (top row to the bottom, bottom row to the top) to close the gap.
    row = outer[0].subgridspec(1, 3, width_ratios=[HEAD_W, 1, 1], wspace=PANEL_WSPACE)
    _, Y_imp = base.REG.trajectory(stim, base.ALPHA, seed=base.SEED)
    for col, d, color in ((1, 1, "0.45"), (2, base.DELAY, base.GREEN)):
        host = fig.add_subplot(row[0, col]); host.axis("off")
        if box_aspect is not None:
            host.set_box_aspect(box_aspect)
            host.set_anchor("S")
        if col == 1:
            host.text(LABEL_X, LABEL_Y, f"({letters[0]})", transform=host.transAxes,
                      fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
            host.text(HEAD_X, 0.5,
                      r"rank of observability" + "\n" + r"matrix $P$ (delay depth)",
                      transform=host.transAxes, rotation=90, ha="center",
                      va="center", fontsize=FS.LABEL, linespacing=1.2)
        grid.draw_delay_matrix(host, [0.0, 0.0, 1.0, 1.0], d, Y_imp, color)
        # Frames stay in the column colour; the plain case's value is printed
        # in black like the grid's.
        host.text(0.5, RANK_Y, f"rank $P={rP[d]}$", transform=host.transAxes,
                  ha="center", va="top", fontsize=base.RANK_FS,
                  color=color if color == base.GREEN else grid.RANK_PLAIN_COLOR,
                  fontweight="bold")

    # ---- impulse --------------------------------------------------------------
    row = outer[1].subgridspec(1, 3, width_ratios=[HEAD_W, 1, 1], wspace=PANEL_WSPACE)
    t, (_, Y_pas), (_, Y_stim) = base.REG.paired_timeseries(stim, base.ALPHA)
    sel = (t >= grid.TRACE_WINDOW[0]) & (t <= grid.TRACE_WINDOW[1])
    amp = float(max(np.abs(Y_pas[:, sel]).max(), np.abs(Y_stim[:, sel]).max()))
    ylim = (-amp, amp)
    for col, Y, color, key, imp in ((1, Y_pas, "0.45", "passive", False),
                                    (2, Y_stim, base.GREEN, "perturbed", True)):
        host = fig.add_subplot(row[0, col]); host.axis("off")
        if box_aspect is not None:
            host.set_box_aspect(box_aspect)
            host.set_anchor("N")
        if col == 1:
            host.text(LABEL_X, LABEL_Y, f"({letters[1]})", transform=host.transAxes,
                      fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")
            host.text(HEAD_X, 0.5, r"rank of trajectory" + "\n" + r"matrix $Q$ (impulse)",
                      transform=host.transAxes, rotation=90, ha="center",
                      va="center", fontsize=FS.LABEL, linespacing=1.2)
        grid.draw_traces(host, [0.0, 0.0, 1.0, 1.0], t, Y, ylim, color, impulse=imp)
        host.text(0.5, RANK_Q_Y, f"rank $Q={rQ[key]}$", transform=host.transAxes,
                  ha="center", va="top", fontsize=base.RANK_FS,
                  color=color if color == base.GREEN else grid.RANK_PLAIN_COLOR,
                  fontweight="bold")
    return rP, rQ
