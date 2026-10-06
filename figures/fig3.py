"""Figure 3: ECoG example, single-site spectra, and cohort counts.

(A) Recording coverage and (B) one raw trial at an example site. (C) Validated
eigenvalue estimates at one stimulation site: rows pre-/post-stimulus, columns
DMD (m = 1) and Hankel DMD (m = 50). (D-F) Participant-level counts: DMD versus
Hankel DMD below and at or above 8 Hz (D), and Hankel DMD pre- versus
post-stimulus for fast- and slow-decaying estimates (E) and for fast-decaying
estimates split at 8 Hz (F).

Inputs: ``<inputs>/fig3/`` and ``<inputs>/cohort/`` (default
``reference_results/``; regenerate with ``scripts/ecog/03_figure_inputs.py``
and ``05_ecog_example.py``). Output: ``outputs/figures/fig3_ccep.{pdf,png}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partial_observation import config
from partial_observation.plotting import ccep_figures as cf
from partial_observation.plotting import ecog_panel as ecog
from partial_observation.plotting import style as figstyle
from partial_observation.selection import PRIMARY_CASE

ROOT = Path(__file__).resolve().parents[1]

# Figure 3 is the widest figure in the paper, so at \linewidth its type comes
# out smaller than everywhere else. Scaling the shared hierarchy before
# use_style() keeps the ordering intact and also reaches plotting.ccep_figures
# and plotting.ecog_panel, which read figstyle.* at draw time. Each figure script runs
# as its own process, so this rescales nothing but figure 3.
FONT_SCALE = 1.55
for _size in ("LETTER", "TITLE", "LABEL", "LEGEND", "TICK", "ANNOT", "SMALL"):
    setattr(figstyle, _size, getattr(figstyle, _size) * FONT_SCALE)

figstyle.use_style()
FS = figstyle

INPUTS = ROOT / "reference_results"
OUT = ROOT / "outputs" / "figures"

EXAMPLE_SITE = "sub-ccepAgeUMCU27__run-031551__T27-T28"     # panel C
EIG_ROUTES = [("standard", 1), ("hankel_d50", 50)]
EIG_WINDOWS = [("pre", "pre-stimulus"), ("early", "post-stimulus")]
N_FITS = config.N_TRIALS

# Height / width of each (C) panel: slightly wider than tall.
BOX_ASPECT = 0.85


def site_equal_window_table(frame: pd.DataFrame, value: str,
                            route: str) -> pd.DataFrame:
    """Per-site counts pooled over the ten fits (the unit of panel C),
    averaged over sites so every subject contributes once."""
    selected = frame[frame.route == route].copy()
    selected["value"] = selected[value]
    return (selected.groupby(["subject", "window"]).value.mean().unstack()
            .reindex(columns=["pre", "early"]).dropna())


def route_pair_panel(ax, table: pd.DataFrame, value: str, hankel_colour: str,
                     title: str) -> None:
    """Draw paired subject points for DMD m=1 versus Hankel DMD m=50."""
    wide = (table.pivot(index="subject", columns="route", values=value)
            .reindex(columns=["standard", "hankel_d50"]).dropna())
    assert len(wide) == 44
    rng = np.random.default_rng(20260924)
    x0 = rng.uniform(-.07, .07, len(wide))
    x1 = 1 + rng.uniform(-.07, .07, len(wide))
    standard = wide.standard.to_numpy(float)
    hankel = wide.hankel_d50.to_numpy(float)
    ax.plot(np.vstack([x0, x1]), np.vstack([standard, hankel]),
            color=".65", lw=.45, alpha=.35, zorder=1)
    ax.scatter(x0, standard, s=20, color="#5f5f5f", alpha=.78,
               linewidths=0, zorder=3)
    ax.scatter(x1, hankel, s=20, color=hankel_colour, alpha=.78,
               linewidths=0, zorder=3)
    for x, values in ((0, standard), (1, hankel)):
        median = float(np.median(values))
        ax.plot([x-.20, x+.20], [median, median], color="black", lw=1.7,
                solid_capstyle="round", zorder=4)
    values = np.concatenate([standard, hankel])
    ax.set_ylim(0, values.max() + max(.08 * np.ptp(values), .08))
    ax.set_xticks([0, 1], ["DMD\n($m=1$)", "Hankel DMD\n($m=50$)"])
    ax.set_xlim(-.38, 1.38)
    ax.set_title(title, fontsize=FS.LABEL)
    ax.grid(alpha=.12, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def split_connector(fig, source, upper, lower, colour) -> None:
    """Guide lines showing that (F) splits the whole of (E, fast-decaying).

    A bracket spans the right edge of the source panel, so the split reads as
    applying to both windows; from its middle a line forks in the gap column
    and ends short of the tick labels of (F). Drawn in figure coordinates after
    a layout pass, so it holds at any figure size.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    inverse = fig.transFigure.inverted()

    def box(artist):
        return inverse.transform_bbox(artist.get_window_extent())

    src = box(source)
    tick_left = min(box(label).x0 for ax in (upper, lower)
                    for label in ax.get_yticklabels() if label.get_text())
    bracket_x = src.x1 + 0.006
    end_x = tick_left - 0.004
    fork_x = bracket_x + 0.40 * (end_x - bracket_x)
    y_mid = src.y0 + 0.5 * src.height
    y_up = box(upper).y0 + 0.5 * box(upper).height
    y_low = box(lower).y0 + 0.5 * box(lower).height
    style = dict(color=colour, lw=1.4, transform=fig.transFigure, zorder=5,
                 solid_capstyle="butt")
    tick = 0.006
    fig.add_artist(Line2D([bracket_x - tick, bracket_x, bracket_x, bracket_x - tick],
                          [src.y1, src.y1, src.y0, src.y0], **style))
    fig.add_artist(Line2D([bracket_x, fork_x], [y_mid, y_mid], **style))
    fig.add_artist(Line2D([fork_x, fork_x], [max(y_mid, y_up), y_low], **style))
    for y in (y_up, y_low):
        fig.add_artist(FancyArrowPatch((fork_x, y), (end_x, y), arrowstyle="-|>",
                                       mutation_scale=11, color=colour, lw=1.4,
                                       shrinkA=0, shrinkB=0,
                                       transform=fig.transFigure, zorder=5))
    fig.text(fork_x - 0.004, 0.5 * (y_mid + y_low), "split at 8 Hz", rotation=90,
             ha="right", va="center", fontsize=FS.SMALL, color=colour)


def main(inputs: Path = INPUTS, out: Path = OUT) -> dict:
    # Reapply the cohort rule to the saved, unfiltered candidate audit.
    audit = pd.read_csv(inputs / "fig3" / "example_site_candidates.csv")
    audit['selected'] = ((audit.support_fraction >= config.REQUIRED_HELD_OUT_TRIALS
                          / config.N_HELD_OUT_TRIALS)
                         & (audit.radial_gap <= config.DECAY_VALIDATION_TOLERANCE))
    modes = audit[audit.selected].copy()
    modes['speed'] = modes.decay_rate_s_inv.map(
        lambda rate: 'fast' if rate <= config.FAST_DECAY_PER_S else 'slow')
    subject, site = modes.subject.iat[0], modes.site.iat[0]
    mirrored = cf.with_conjugates(modes)
    # Zoom onto the occupied region (Re near 1, |Im| <= ~0.11) instead of the
    # near-square view_limits window, which spent most of the x-axis on empty
    # space. Aspect is left free so the thin arc fills the panel width.
    ZOOM_X, ZOOM_Y = (0.970, 1.006), (-0.15, 0.15)
    xlim, ylim = ZOOM_X, ZOOM_Y


    # (A) coverage and (B) traces full width on top; below, the (C) spectra
    # block on the left and the (D-F) cohort columns as a 2 x 3 grid on the
    # right. Explicit gridspec extents keep each block's spacing independent.
    # Taller canvas: the top row (A, B) takes the added height, while the
    # lower blocks keep their previous size in inches.
    fig = plt.figure(figsize=(15.0, 11.4))

    # The coverage axes starts at the canvas edge and gets a wider share of
    # the row, so the mesh (scaled by the axes width) sits close to (A).
    top = fig.add_gridspec(1, 3, top=.985, bottom=.66, left=.0, right=.96,
                           width_ratios=[.46, .03, 1.0], wspace=0)
    brain_ax = fig.add_subplot(top[0, 0], projection="3d", computed_zorder=False)
    trace_ax = fig.add_subplot(top[0, 2])
    ecog.draw_example(brain_ax, trace_ax,
                      ecog.load_example(inputs / "fig3" / "ecog_example.npz"))
    # Figure coordinates, so the two letters share one height: a 3D axes
    # and a 2D axes do not place axes-fraction text at the same height. The
    # 3D axes settles its position only at draw time, so draw once before
    # reading the coverage title's extent; the letters are then centred on
    # the title's own vertical centre.
    fig.canvas.draw()
    title_box = (brain_ax.title.get_window_extent(fig.canvas.get_renderer())
                 .transformed(fig.transFigure.inverted()))
    title_mid = (title_box.y0 + title_box.y1) / 2
    row_letters = {}
    for x, letter in ((.005, "(A)"), (.268, "(B)")):
        row_letters[letter] = fig.text(x, title_mid, letter,
                                       fontsize=FS.LETTER, fontweight="bold",
                                       va="center", ha="left")

    # ---- (C) one site, a single 2 x 2 block sharing one letter ------------
    # Width and height are solved together, so the cells match BOX_ASPECT
    # exactly and no axes is shrunk and re-centred inside its cell: the two
    # rows plus hspace occupy (.529 - .072) * 11.4 = 5.21 in, so each axes is
    # 5.21 / 2.20 = 2.37 in tall, 2.37 / BOX_ASPECT = 2.79 in wide, and the
    # block needs 2.79 * 2.20 = 6.13 in, i.e. .409 of the 15 in canvas.
    grid = fig.add_gridspec(2, 2, top=.529, bottom=.072, left=.045, right=.454,
                            wspace=.20, hspace=.20)
    counts = {}
    axes_eig = []
    for row, (window, window_label) in enumerate(EIG_WINDOWS):
        for col, (route, delay) in enumerate(EIG_ROUTES):
            ax = fig.add_subplot(grid[row, col])
            axes_eig.append(ax)
            cell = mirrored[(mirrored.window == window) & (mirrored.route == route)]
            listed = modes[(modes.window == window) & (modes.route == route)]
            n_total, n_fast = cf.eigenplane_panel(ax, cell, listed, xlim, ylim,
                                                  inset=False)
            for collection in list(ax.collections):
                collection.remove()
            colors = [cf.SLOW_COLOR if s == 'slow' else cf.FAST_COLOR
                      for s in cell.speed]
            # Larger markers with a thin white rim: overlapping estimates
            # stay separable and the two colours read at print size.
            ax.scatter(cell.eig_real, cell.eig_imag, s=24,
                       color=colors, alpha=.85, edgecolors='white',
                       linewidths=.4, zorder=3)
            # Free the data aspect (the zoom is far from square in data
            # units) and fix the drawn box instead.
            ax.set_aspect('auto')
            ax.set_box_aspect(BOX_ASPECT)
            ax.set_xlim(*ZOOM_X)
            ax.set_ylim(*ZOOM_Y)
            # Counts still gate the cohort cross-check below; they are no
            # longer written over each panel.
            counts[(window, route)] = (n_total, n_fast)
            if row == 0:
                route_name = "DMD" if delay == 1 else "Hankel DMD"
                ax.annotate(f"{route_name} ($m={delay}$)", xy=(.5, 1.05),
                            xycoords="axes fraction", ha="center", va="bottom",
                            fontsize=FS.TITLE, fontweight="bold")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"Re($\lambda$)", fontsize=FS.TICK)
            if col == 0:
                # The axis names stay small; the row name is the structural
                # label, so it sits outside them at title size.
                ax.set_ylabel(r"Im($\lambda$)", fontsize=FS.TICK)
                # Plain bold text: in mathtext the hyphen would become a minus.
                ax.annotate(window_label, xy=(0, .5),
                            xycoords="axes fraction", xytext=(-68, 0),
                            textcoords="offset points", rotation=90,
                            ha="center", va="center", fontsize=FS.TITLE,
                            fontweight="bold")
            else:
                ax.set_yticklabels([])

    # One letter for the whole block, level with the two column headers. Offset
    # in points so it clears the tick labels whatever the axes width.
    letter_c = axes_eig[0].annotate("(C)", xy=(0, 1), xycoords="axes fraction",
                                    xytext=(-58, 20), textcoords="offset points",
                                    fontweight="bold", fontsize=FS.LETTER,
                                    va="bottom", ha="right")
    # (A) shares its left edge with (C), so the two letters line up down the
    # left margin. (C) is offset in points from its axes, so its edge is read
    # back after a draw.
    fig.canvas.draw()
    c_left = (letter_c.get_window_extent(fig.canvas.get_renderer())
              .transformed(fig.transFigure.inverted()).x0)
    row_letters["(A)"].set_x(c_left)
    fig.legend(handles=cf.speed_legend_handles(edge="white"), loc="center",
               bbox_to_anchor=(.25, .010), ncol=2, frameon=False,
               fontsize=FS.LEGEND, handletextpad=0.35, columnspacing=2.0)

    # Cohort: actual rank 50 in both routes/windows/all fits, radial selected.
    rank_audit = pd.read_csv(inputs / "cohort" / "site_rank_audit.csv")
    stems = set(rank_audit.loc[rank_audit.all_rank50, 'stem'])
    sites = pd.read_csv(inputs / "cohort" / "site_counts_rank50.csv.gz")
    sites = sites[(sites.case == PRIMARY_CASE) & sites.stem.isin(stems)]
    assert len(stems) == 1318 and sites.subject.nunique() == 44
    representative = EXAMPLE_SITE
    assert representative in stems
    for (window, route_key), (total, fast) in counts.items():
        expected = sites[(sites.stem == representative) & (sites.window == window) & (sites.route == route_key)]
        assert len(expected) == 1
        assert (total, fast) == (int(expected.n_total.iloc[0]), int(expected.n_fast.iloc[0]))

    # Columns: delay contrast | fast/slow | fast frequency split.
    # Column 3 is an empty gap between (E) and (F) that carries the guide
    # lines showing how (F) splits the fast-decaying estimates of (E).
    cohort = fig.add_gridspec(2, 4, top=.485, bottom=.055, left=.535,
                              right=.995, wspace=.30, hspace=.72,
                              width_ratios=[1, 1, .18, 1])
    cohort_axes = [fig.add_subplot(cohort[r, c])
                   for c, r in ((0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (3, 1))]

    # D: delay-related redistribution in the pre-stimulus window.
    delay_subject = pd.read_csv(inputs / "fig3" / "delay_high8_paired_subjects.csv")
    delay_subject = delay_subject[delay_subject.window == "pre"].copy()
    # The table stores per-fit means; pool the ten fits to match panel C.
    for column in ("n_low_per_site_fit", "n_high_per_site_fit"):
        delay_subject[column] = delay_subject[column] * N_FITS
    route_pair_panel(cohort_axes[0], delay_subject, "n_low_per_site_fit",
                     "#4c78a8", "mode estimates\nbelow 8 Hz")
    route_pair_panel(cohort_axes[1], delay_subject, "n_high_per_site_fit",
                     "#4c78a8", "mode estimates\nat or above 8 Hz")

    # E/F: fast/slow perturbation comparison and the fast-mode frequency
    # split at fixed m=50, with the same per-site normalisation.
    perturbation_panels = [
        ("n_fast", cf.FAST_COLOR, "fast-decaying\nmode estimates"),
        ("n_slow", cf.SLOW_COLOR, "slow-decaying\nmode estimates"),
        ("n_fast_low", cf.FAST_COLOR, "fast-decaying estimates\nbelow 8 Hz"),
        ("n_fast_high", cf.FAST_COLOR, "fast-decaying estimates\nat or above 8 Hz"),
    ]
    for ax, (column, colour, title) in zip(cohort_axes[2:], perturbation_panels):
        cf.paired_panel(ax, site_equal_window_table(sites, column, "hankel_d50"),
                        colour, "", title, annotate_median=False)
        ax.set_title(title, fontsize=FS.LABEL)
        # The column header names the method, so the ticks give the window.
        ax.set_xticks([0, 1], ["pre-\nstimulus", "post-\nstimulus"])

    for index, ax in enumerate(cohort_axes):
        # Per site, pooled over its ten fits (as in C); see the caption.
        ax.set_ylabel("validated estimates per site\n(10 fits pooled)" if index < 2 else "")
        # Narrow columns leave little room for the two-line route/window
        # labels, so they run one step below the in-panel annotation size.
        ax.tick_params(axis="x", labelsize=FS.SMALL)
    # cohort_axes runs column-major, so 0/2/4 head the three 2 x 1 columns
    # and each column carries one letter. Only (D) has a y axis label beneath
    # it, so only (D) needs the wider clearance.
    # Column headers: what each column contrasts, above the panel titles.
    headers = ("Effect of\ndelay embedding",
               "Effect of\nperturbation",
               "Fast-decaying modes\nby frequency")
    for ax, header in zip(cohort_axes[::2], headers):
        ax.annotate(header, xy=(.5, 1), xycoords="axes fraction",
                    xytext=(0, 46), textcoords="offset points",
                    fontweight="bold", fontsize=FS.LABEL, va="bottom",
                    ha="center", linespacing=1.15)
    for ax in cohort_axes[1::2]:
        ax.title.set_fontsize(FS.LABEL)
    for ax, letter, dx in zip(cohort_axes[::2], "DEF", (-36, -6, -22)):
        ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                    xytext=(dx, 62), textcoords="offset points",
                    fontweight="bold", fontsize=FS.LETTER, va="bottom",
                    ha="right")

    split_connector(fig, cohort_axes[2], cohort_axes[4], cohort_axes[5],
                    cf.FAST_COLOR)

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_ccep.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", out / "fig3_ccep.pdf")

    summary = {"subject": subject, "site": site,
               "reliable_and_fast": {f"{w}|{r}": counts[(w, r)]
                                     for w, _ in EIG_WINDOWS
                                     for r, _ in EIG_ROUTES},
               "selection": PRIMARY_CASE}
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.inputs, args.out)
