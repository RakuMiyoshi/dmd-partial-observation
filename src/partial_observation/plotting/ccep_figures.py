"""Shared panel drawing for the ECoG figures (Fig. 3 and Figs. S3-S5).

The main figure and the supplements show the same quantities at different
scopes, so the colours, the eigenvalue-plane panel and the paired
participant panel live here rather than in each figure script. Nothing in this
module fits anything.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partial_observation import config
from partial_observation.plotting import palette as RCFG
from partial_observation.plotting import style as FS

FAST_DECAY_PER_S = config.FAST_DECAY_PER_S   # slow above the cut, fast at or below

WINDOWS = tuple(config.WINDOWS_MS)
WINDOW_LABELS = dict(RCFG.WINDOW_LABELS)
WINDOW_COLORS = dict(RCFG.WINDOW_COLORS)
GROUP_COLORS = dict(RCFG.GROUP_COLORS)
SLOW_COLOR = GROUP_COLORS["persistent"]
FAST_COLOR = GROUP_COLORS["fast-decay"]


MARKER_MIN, MARKER_MAX = 8.0, 130.0
UNIT = np.exp(1j * np.linspace(0, 2 * np.pi, 500))
JITTER = 0.085


# =============================================================================
# Per-mode tables: one site, every route and window
# =============================================================================
def with_conjugates(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror every mode with non-zero imaginary part onto the lower half-plane.

    The table keeps one member of each conjugate pair, and a spectrum drawn on
    only half the plane reads as if the modes were real.
    """
    lower = frame[frame.eig_imag.abs() > 0].copy()
    lower["eig_imag"] = -lower["eig_imag"]
    return pd.concat([frame, lower], ignore_index=True)


def marker_sizes(amplitudes) -> np.ndarray:
    """Marker area from the held-out modal amplitude, on a log scale."""
    values = np.log10(np.maximum(np.asarray(amplitudes, float), 1e-3))
    low, high = np.nanpercentile(values, [2, 98])
    span = max(high - low, 1e-9)
    return MARKER_MIN + np.clip((values - low) / span, 0.0, 1.0) * (
        MARKER_MAX - MARKER_MIN)


def eigenplane_panel(ax, mirrored, listed, xlim, ylim, *, size_scale=1.0,
                     inset=True, edge=True):
    """One complex-plane panel of reliable modes, coloured slow/fast."""
    ax.plot(UNIT.real, UNIT.imag, "--", color="0.70", lw=0.9)
    for speed, colour in (("slow", SLOW_COLOR), ("fast", FAST_COLOR)):
        part = mirrored[mirrored.speed == speed]
        if part.empty:
            continue
        ax.scatter(part.eig_real, part.eig_imag, color=colour,
                   s=size_scale * marker_sizes(
                       part.median_heldout_modal_amplitude_uV),
                   alpha=0.7,
                   edgecolors="black" if edge else "none",
                   linewidths=0.3 if edge else 0.0)
    ax.axhline(0, color="0.88", lw=0.6)
    ax.set_aspect("equal", "box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.12)

    if inset:
        axin = ax.inset_axes([0.03, 0.03, 0.28, 0.28])
        axin.plot(UNIT.real, UNIT.imag, "-", color="0.68", lw=0.6)
        axin.scatter(mirrored.eig_real, mirrored.eig_imag, s=3, linewidths=0,
                     color=[SLOW_COLOR if s == "slow" else FAST_COLOR
                            for s in mirrored.speed])
        axin.plot([xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]],
                  [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]],
                  color="0.25", lw=0.7)
        axin.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        axin.set_aspect("equal", "box")
        axin.set_xticks([])
        axin.set_yticks([])

    n_fast = int((listed.speed == "fast").sum())
    return len(listed), n_fast


def speed_legend_handles(edge="black"):
    return [plt.Line2D([], [], marker="o", ls="none", markersize=8,
                       markerfacecolor=colour, markeredgecolor=edge,
                       markeredgewidth=0.3,
                       label=f"{name} (decay {sign} $-10$ /s)")
            for name, colour, sign in (("slow", SLOW_COLOR, ">"),
                                       ("fast", FAST_COLOR, r"$\leq$"))]


# =============================================================================
# Summary tables: per-subject counts, paired across windows
# =============================================================================


def paired_panel(ax, table: pd.DataFrame, colour: str, ylabel: str, title: str,
                 *, annotate_median: bool = True, ages=None, norm=None,
                 cmap: str = "viridis"):
    """Window on the x axis, one dot per subject, the same subject joined.

    ``ages`` colours the dots by a per-row value instead of the flat group
    colour; the joining lines stay grey either way.
    """
    positions = np.arange(len(WINDOWS), dtype=float)
    values = {w: table[w].to_numpy(float) for w in WINDOWS}
    rng = np.random.default_rng(0)
    dot_x = {w: positions[i] + rng.uniform(-JITTER, JITTER, len(values[w]))
             for i, w in enumerate(WINDOWS)}

    ax.plot(np.stack([dot_x[w] for w in WINDOWS]),
            np.stack([values[w] for w in WINDOWS]),
            color="0.65", lw=0.45, alpha=0.35, zorder=1)
    mappable = None
    for i, w in enumerate(WINDOWS):
        if ages is None:
            ax.scatter(dot_x[w], values[w], s=22, color=colour, alpha=0.75,
                       linewidths=0.0, zorder=3)
        else:
            mappable = ax.scatter(dot_x[w], values[w], s=26, c=ages, cmap=cmap,
                                  norm=norm, alpha=0.9, linewidths=0.0, zorder=3)
        median = float(np.nanmedian(values[w]))
        ax.plot([positions[i] - 0.22, positions[i] + 0.22], [median] * 2,
                color="black", lw=1.8, solid_capstyle="round", zorder=4)
        if annotate_median:
            ax.annotate(f"{median:.0f}", xy=(positions[i] + 0.24, median),
                        fontsize=FS.ANNOT, va="center", ha="left", color="0.25")

    delta = values[WINDOWS[1]] - values[WINDOWS[0]]
    up = int((delta > 0).sum())
    ax.set_xticks(positions, [WINDOW_LABELS.get(w, w) for w in WINDOWS])
    ax.set_xlim(-0.45, len(WINDOWS) - 0.35)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{up}/{len(delta)} subjects up", fontsize=FS.TITLE)
    ax.grid(alpha=0.12, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return {"median_passive": float(np.median(values[WINDOWS[0]])),
            "median_perturbed": float(np.median(values[WINDOWS[1]])),
            "n_up": up, "n_subjects": int(len(delta)), "mappable": mappable}
