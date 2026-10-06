"""Figure 2: delay embedding and perturbation relieve complementary rank bottlenecks.

Top row, the linear 24-state network: (A) delay matrices at m = 1 and m = 8,
(B) observed signals without and with an impulse, (C) eigenvalue recovery over
rank P (delay depth) and rank Q (impulse). Bottom row, four uncoupled
Stuart-Landau oscillators: (D) relaxation onto the limit cycle, (E) observed
signals, (F) DMD spectra over delay depth and impulse.

The panels are drawn by ``partial_observation.plotting.linear_figure_*`` and
``partial_observation.simulations.stuart_landau``; only the arrangement lives
here. The simulations use fixed seeds and need no data.

Output: ``outputs/figures/fig2_simulation.{pdf,png}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from partial_observation.plotting import linear_figure_base as base
from partial_observation.plotting import linear_figure_grid as grid
from partial_observation.plotting import linear_figure_setup as setup
from partial_observation.simulations import stuart_landau as nonlinear

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"

STEM = "fig2_simulation"
FIG_W, FIG_H = 13.8, 12.8
COLS = (1.0, 1.72, 0.30)      # settings column : grid column : legend column
COL_WSPACE = 0.108
ROWS = (1.0, 1.0)             # linear row : nonlinear row
ROW_HSPACE = 0.19
GRID_SHIFT_X = 0.03          # (C), (F): whole grid moved right (figure fraction)
NL_FLOW_SHIFT = 0.025         # (D): right of the left (B) panel's edge (figure fraction)
SIDE_GAP = 0.006              # side strip of (C), (F): gap left of the Im(lambda) labels
LEGEND_GAP = 0.012            # legend left edge, right of the cells (figure fraction)
# (A,B): the settings block, two panels per row.
SETTINGS_ASPECT = 1.10        # height / width of each settings panel
SETTINGS_HSPACE = 0.38        # gap between the (A) and (B) rows
SETTINGS_PAD = 0.033          # empty row under (B), in row heights: shortens
                              # the block so the rank-Q values end level with
                              # the legend under (C)
# (D,E): the flow portrait (legend to its right) over the observed signals.
NL_TOP_ROWS = (1.75, 1.55)    # (D) keeps its size; (E) takes the room the legend left
NL_TOP_HSPACE = 0.58          # gap between (D) and (E): the Re a_1 label and the (E) title
NL_PAD = 0.06                 # empty strip under (E), as a fraction of the row: lifts
                              # the time axis of (E) level with the legend under (F)
NL_FLOW_LEGEND_SIDE = "right" # (D) legend: one column to the right of the axes
NL_FLOW_LEGEND_XY = (1.03, 0.5)  # its anchor, in (D) axes fraction
NL_FLOW_ANCHOR = (0.62, 0.5)  # (D) axes position in its slot (0 = left, 1 = right):
                              # far enough right that the centred title clears the letter
NL_OBS_LABEL_GAP = 0.02       # y_1 / y_2 labels of (E): offset left of the axes,
                              # in axes width (stuart_landau.panel_observed)
NL_OBS_WSPACE = 0.12          # gap between the passive and impulse halves of (E)
NL_OBS_T_PRE = 0.3            # (E): seconds of the passive cycle before the impulse
NL_OBS_T_MAX = 1.3            # (E): width of the window in seconds (-T_PRE .. T_MAX-T_PRE)
NL_LETTER_Y = 1.06           # (D,E) letters: y in axes fraction, x at the column edge
# (A,B) letters, in first-panel axes fraction: pulled in from the setup default
# so they line up with the (C) letter at the column's left edge.
SETTINGS_LETTER_X = -0.46
SETTINGS_HEAD_W = 0.10        # (A,B) header column and column gap, narrower than the
SETTINGS_WSPACE = 0.14        # setup default so the panels grow towards the size of (C)
SETTINGS_RANK_Q_Y = -0.16     # rank-Q values under the (B) trace panels, axes fraction
                              # (lower than the setup default so they clear the caption)
# Excitation label, arrow and rank-Q values in the side strip of (C).
GRID_SIDE_LABEL_X, GRID_SIDE_ARROW_X, GRID_SIDE_TEXT_X = 0.18, 0.40, 0.60
# Strip over the (C) cells, in cell heights, and the positions inside it
# (strip axes fraction): title, arrow, rank-P values and the delay depth m
# under each value. Taller than the grid default to fit the m line.
GRID_TOP_H = 0.32
GRID_OBS_Y, GRID_ARROW_Y, GRID_RANKP_Y, GRID_M_Y = 1.16, 0.82, 0.52, 0.12
# Empty row under each grid, in cell heights: just room for the Re(lambda)
# labels, which hang into it (the legends sit beside the grids).
GRID_PAD = 0.22
# The same strip of (F); its tick labels are wider, so the strip is wider too.
NL_SIDE_W = 0.62
# Kept well left of the cells so the passive / impulse values clear Im(lambda).
NL_SIDE_LABEL_X, NL_SIDE_ARROW_X, NL_SIDE_TEXT_X = 0.10, 0.27, 0.40
NL_YTICKS = (-200, 0, 200)    # (F): fixed, so the larger cells do not add ticks
# All text is scaled by this factor relative to the shared figstyle sizes, so
# the page is filled by larger type rather than by whitespace.
FONT_SCALE = 1.40


def scale_fonts(scale: float) -> None:
    """Multiply every size in the shared hierarchy (and the derived module
    constants that were copied from it at import time) by ``scale``."""
    for name in ("LETTER", "TITLE", "LABEL", "LEGEND", "TICK", "ANNOT", "SMALL"):
        setattr(base.FS, name, getattr(base.FS, name) * scale)
    base.FS.use_style()                 # re-apply rcParams with the new sizes
    base.RANK_FS = base.FS.LABEL
    grid.AXIS_LABEL_FS = grid.AXIS_LABEL_FS * scale


def wrap_label(label: str) -> str:
    """Move a parenthesised qualifier onto a second line so the side legend
    stays narrow; labels without one are already about as wide."""
    return label.replace(" (", "\n(", 1)


def side_legend(fig, cells, spec, handles, labels):
    """One-column legend right of the grid ``cells``, centred on its two rows,
    with each label wrapped onto two lines."""
    labels = [wrap_label(label) for label in labels]
    ax = fig.add_subplot(spec); ax.axis("off")
    pos = [a.get_position() for a in cells]
    y_mid = (min(p.y0 for p in pos) + max(p.y1 for p in pos)) / 2.0
    x = max(p.x1 for p in pos) + LEGEND_GAP
    leg = ax.legend(handles, labels, loc="center left", ncol=1, frameon=False,
                    fontsize=base.FS.LEGEND, handletextpad=0.3, labelspacing=0.9,
                    bbox_to_anchor=(x, y_mid), bbox_transform=fig.transFigure)
    for text in leg.get_texts():      # centre the wrapped (k = ...) line
        text.set_multialignment("center")


def main(out: Path = OUT) -> dict:
    scale_fonts(FONT_SCALE)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = fig.add_gridspec(2, 3, width_ratios=list(COLS), height_ratios=list(ROWS),
                             wspace=COL_WSPACE, hspace=ROW_HSPACE)

    # ---- top row: linear system, (A,B) settings left, the (C) grid right ----
    setup.LABEL_X = SETTINGS_LETTER_X
    setup.HEAD_W, setup.PANEL_WSPACE = SETTINGS_HEAD_W, SETTINGS_WSPACE
    setup.RANK_Q_Y = SETTINGS_RANK_Q_Y
    grid.GRID_PAD = GRID_PAD
    grid.TOP_H = GRID_TOP_H
    grid.OBS_Y, grid.ARROW_TOP_Y, grid.RANKP_Y, grid.M_Y = (GRID_OBS_Y, GRID_ARROW_Y,
                                                   GRID_RANKP_Y, GRID_M_Y)
    rP_s, rQ_s = setup.draw_settings(fig, outer[0, 0], letters=("A", "B"),
                                   hspace=SETTINGS_HSPACE,
                                   box_aspect=SETTINGS_ASPECT, pad=SETTINGS_PAD)
    axes_ab = list(fig.axes)
    n_axes = len(fig.axes)
    results, rP, rQ, legend_c = grid.draw_grid(fig, outer[0, 1], "C", legend=False,
                                      side_label_x=GRID_SIDE_LABEL_X,
                                      side_arrow_x=GRID_SIDE_ARROW_X,
                                      side_text_x=GRID_SIDE_TEXT_X)
    grid_c = fig.axes[n_axes:]
    cells_c = [a for a in grid_c if a.axison]   # the four cells

    # ---- bottom row: nonlinear system, (D,E) settings left, (F) grid right --
    # An empty pad row under (E) shortens the block so that the time axis of
    # (E) ends level with the legend under (F).
    # The pad is split off first with no gap, so NL_TOP_HSPACE is the whole
    # (D)-(E) gap and does not also open a gap above the pad.
    block = outer[1, 0].subgridspec(2, 1, height_ratios=[1.0 - NL_PAD, NL_PAD],
                                    hspace=0.0)
    top = block[0].subgridspec(2, 1, height_ratios=list(NL_TOP_ROWS),
                               hspace=NL_TOP_HSPACE)
    ax_flow = fig.add_subplot(top[0])
    nonlinear.FLOW_LEGEND_SIDE = NL_FLOW_LEGEND_SIDE
    nonlinear.FLOW_LEGEND_XY = NL_FLOW_LEGEND_XY
    nonlinear.FLOW_ANCHOR = NL_FLOW_ANCHOR
    nonlinear.panel_flow(ax_flow)
    for text in ax_flow.get_legend().get_texts():   # narrow (D) legend, as for (C, F)
        text.set_text(wrap_label(text.get_text()))
        text.set_multialignment("center")
    nonlinear.OBS_LABEL_GAP = NL_OBS_LABEL_GAP
    ax_obs, ax_obs_imp = nonlinear.panel_observed_pair(fig, top[1], t_max=NL_OBS_T_MAX,
                                                t_pre=NL_OBS_T_PRE,
                                                wspace=NL_OBS_WSPACE)
    # Letters at the left edge of the column (figure x), level with each
    # panel's top (axes y), so they line up with (F) below.
    col_x0 = outer[1, 0].get_position(fig).x0
    # (E) is narrowed from the left so that its y_1 / y_2 labels, which hang
    # just outside the axes, start at the column edge, under the letter.
    # The labels sit at -NL_OBS_LABEL_GAP of the axes width, so solve for the
    # left edge x0 with the right edge x1 fixed:
    #   x0 - gap * (x1 - x0) - width_of_label = col_x0
    fig.canvas.draw()
    fig_w_px = fig.bbox.width
    label_w = max(t.get_window_extent().width for t in ax_obs.texts
                  if t.get_text().startswith("$y_")) / fig_w_px
    pos = ax_obs.get_position()
    pos_i = ax_obs_imp.get_position()
    # The two halves of (E) are resized together: the passive half starts at
    # x0, the impulse half keeps its right edge, both share one width and the
    # original gap between them. The label sits at -gap of the passive width.
    gap_axes = pos_i.x0 - pos.x1
    gap = NL_OBS_LABEL_GAP
    # x0 - gap * w - label_w = col_x0, with w = (pos_i.x1 - x0 - gap_axes) / 2
    x0 = (col_x0 + label_w + gap * (pos_i.x1 - gap_axes) / 2.0) / (1.0 + gap / 2.0)
    w = (pos_i.x1 - x0 - gap_axes) / 2.0
    ax_obs.set_position([x0, pos.y0, w, pos.height])
    ax_obs_imp.set_position([x0 + w + gap_axes, pos.y0, w, pos.height])
    for ax, letter in ((ax_flow, "D"), (ax_obs, "E")):
        ax.text(col_x0, NL_LETTER_Y, f"({letter})",
                transform=blended_transform_factory(fig.transFigure, ax.transAxes),
                fontweight="bold", fontsize=base.FS.LETTER, va="bottom", ha="left")
    n_axes = len(fig.axes)
    handles_f, labels_f = nonlinear.panel_grid(fig, outer[1, 1], letter="F", legend=False,
                                        square=True,
                                        side_w=NL_SIDE_W, top_h=grid.TOP_H, pad=GRID_PAD,
                                        side_label_x=NL_SIDE_LABEL_X,
                                        side_arrow_x=NL_SIDE_ARROW_X,
                                        side_text_x=NL_SIDE_TEXT_X,
                                        value_fs=base.RANK_FS)
    grid_f = fig.axes[n_axes:]
    cells_f = [a for a in grid_f if a.axison]
    for ax in cells_f:
        ax.set_yticks(NL_YTICKS)

    # The square cells are anchored inside their slots; resolve their final
    # positions before placing anything relative to them.
    fig.canvas.draw()
    for ax in grid_c + grid_f:
        p = ax.get_position()
        ax.set_position([p.x0 + GRID_SHIFT_X, p.y0, p.width, p.height])
    # The square cells leave slack in their slots, which opens a gap between
    # the side strip (excitation arrow, row values) and the Im(lambda) labels;
    # slide the strip right to close it.
    renderer = fig.canvas.get_renderer()
    fig_w = fig.bbox.width
    for grid_axes, cells in ((grid_c, cells_c), (grid_f, cells_f)):
        side = next(a for a in grid_axes if not a.axison and any(
            t.get_rotation() == 90 for t in a.texts))
        cells_left = min(a.get_tightbbox(renderer).x0 for a in cells) / fig_w
        side_right = max(t.get_window_extent(renderer).x1 for t in side.texts
                         if t.get_text()) / fig_w     # the visible labels only
        p = side.get_position()
        side.set_position([p.x0 + cells_left - side_right - SIDE_GAP,
                           p.y0, p.width, p.height])
    side_legend(fig, cells_c, outer[0, 2], *grid.legend_entries(legend_c))
    side_legend(fig, cells_f, outer[1, 2], handles_f, labels_f)

    # Level the bottom of each settings block with the Re(lambda) labels of its
    # grid: (A,B) slides as a block; (E) keeps its top under (D) and raises its
    # bottom, so it does not run into the Re a_1 label of (D).
    renderer = fig.canvas.get_renderer()

    def bottom(axes):
        return min(a.get_tightbbox(renderer).y0 for a in axes) / fig.bbox.height

    for axes, cells, keep_top in ((axes_ab, cells_c, False),
                                  ([ax_obs, ax_obs_imp], cells_f, True)):
        dy = bottom(cells) - bottom(axes)
        for a in axes:
            p = a.get_position()
            a.set_position([p.x0, p.y0 + dy, p.width, p.height - dy * keep_top])

    # Open a gap between (A) and (B): raise the (A) row until its letter is
    # level with the (C) letter, leaving (B) where the alignment above put it.
    def letter(text):
        t = next(t for ax in fig.axes for t in ax.texts if t.get_text() == text)
        return t, t.get_window_extent(renderer).y1 / fig.bbox.height

    t_a, y_a = letter("(A)")
    dy = letter("(C)")[1] - y_a
    row_a = [a for a in axes_ab
             if a.get_position().y0 >= t_a.axes.get_position().y0 - 1e-6]
    for a in row_a:
        p = a.get_position()
        a.set_position([p.x0, p.y0 + dy, p.width, p.height])

    # Vertical lines shared by the two settings blocks: (E) takes the columns
    # of the (B) panels, (D) sits NL_FLOW_SHIFT right of the left (B) panel, and the (D, E)
    # letters start where the (A, B) letters do.
    fig.canvas.draw()
    t_b, _ = letter("(B)")
    row_b = sorted((a for a in axes_ab if a not in row_a
                    and a.get_position().height > 0.5 * t_b.axes.get_position().height),
                   key=lambda a: a.get_position().x0)
    b_left, b_right = row_b[0].get_position(), row_b[-1].get_position()
    for ax, ref in ((ax_obs, b_left), (ax_obs_imp, b_right)):
        p = ax.get_position()
        ax.set_position([ref.x0, p.y0, ref.width, p.height])
    p = ax_flow.get_position()
    ax_flow.set_position([b_left.x0 + NL_FLOW_SHIFT, p.y0, p.width, p.height])
    ax_flow.set_anchor("W")
    letter_x = t_a.get_window_extent(renderer).x0 / fig.bbox.width
    for text in ("(D)", "(E)"):
        t, _ = letter(text)
        t.set_x(letter_x)

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{STEM}.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", out / f"{STEM}.pdf")

    summary = {"rankP": rP, "rankQ": rQ, "recovered": results}
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
