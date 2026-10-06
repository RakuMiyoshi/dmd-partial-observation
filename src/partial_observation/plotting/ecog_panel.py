"""Fig. 3A-B: ECoG coverage and a ridge stack of one raw, unaveraged trial.

Renders from ``fig3/ecog_example.npz`` (written by
``scripts/ecog/05_ecog_example.py``), which holds the displayed trial and the
electrode coordinates, so the raw recording is not needed. The fsaverage pial
surface is not redistributed here: it is fetched through MNE on first use
(``mne.datasets.fetch_fsaverage``, FreeSurfer licence) and cached locally.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import mne
import numpy as np
from mne.surface import read_curvature

from partial_observation.plotting import style as figstyle

RIDGE_SPACING = .36    # row spacing in units of the typical amplitude
ARTEFACT_CLIP = 2.0    # artefact excursion cap, same units

PRE_COLOR = "#4c78a8"
POST_COLOR = "#e58b45"
STIM_COLOR = "#c43c39"
TRACE_COLOR = "#30343b"
RECORDING_COLOR = "#30343b"


def outward(xyz: np.ndarray, centre: np.ndarray, distance: float) -> np.ndarray:
    direction = xyz - centre
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    return xyz + distance * direction


def brain_panel(ax, recording_xyz: np.ndarray, stim_xyz: np.ndarray,
                mesh: tuple, hemi: str) -> str:
    """Draw actual coverage on an fsaverage lateral pial surface."""
    vertices, faces, curvature = mesh
    surface = ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces, color=(0.82, 0.83, 0.84), alpha=.98,
        linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=0,
    )
    # Use the subject-independent fsaverage folding pattern as restrained
    # anatomical shading.  Unlike taking every nth triangle, this leaves a
    # continuous surface and therefore exports cleanly at journal size.
    face_curvature = np.nanmean(curvature[faces], axis=1)
    lo, hi = np.nanpercentile(face_curvature, [8, 92])
    folding = np.clip((face_curvature - lo) / max(hi - lo, 1e-12), 0, 1)
    grey = .84 - .22 * folding
    surface.set_facecolors(np.column_stack([grey, grey, grey,
                                            np.full(len(grey), .98)]))
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    recording_plot = outward(recording_xyz, centre, 3.0)
    stim_plot = outward(stim_xyz, centre, 6.0)

    ax.scatter(*recording_plot.T, s=26, color=RECORDING_COLOR,
               edgecolors="white", linewidths=.45, depthshade=False,
               zorder=3)
    ax.plot(*stim_plot.T, color=STIM_COLOR, lw=2.5, zorder=7)
    ax.scatter(*stim_plot.T, s=155, marker="o", color=STIM_COLOR,
               edgecolors="white", linewidths=1.4, depthshade=False,
               zorder=8)

    span = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    c = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    radius = span / 2
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.55)
    ax.view_init(elev=10, azim=(0 if hemi == "rh" else 180))
    ax.set_axis_off()
    ax.set_title("Lateral ECoG coverage", pad=-2)

    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=8,
                      markerfacecolor=RECORDING_COLOR, markeredgecolor="white",
                      label="recording electrodes"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=11,
                      markerfacecolor=STIM_COLOR, markeredgecolor="white",
                      label="stimulation pair"),
    ]
    # Anchored below the axes so the entries clear the temporal lobe, which
    # the zoomed mesh runs close to the bottom edge.
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.11),
              frameon=False, fontsize=figstyle.LEGEND, handletextpad=.3)
    return hemi


def _tint(colour, alpha: float) -> tuple[float, float, float]:
    """Opaque equivalent of `colour` at `alpha` over white."""
    rgb = np.asarray(mcolors.to_rgb(colour))
    return tuple(1 - alpha * (1 - rgb))


def waveform_panel(ax, time_ms: np.ndarray, traces_uv: np.ndarray,
                   channel_names: list[str]) -> float:
    """Ridge stack of raw, unaveraged traces with one common vertical gain.

    Rows are drawn back (top) to front (bottom); each row's filled underside
    hides the rows behind it, which gives the oblique "viewed from above"
    reading.  The fill carries the analysis-window tint, so the windows stay
    visible without a separate background span.
    """
    baseline = (time_ms >= -310) & (time_ms < -10)
    traces = traces_uv - np.nanmedian(traces_uv[:, baseline], axis=1)[:, None]

    visible = ((time_ms >= -310) & (time_ms < -10)) | (
        (time_ms > 10) & (time_ms <= 310))
    per_channel = np.nanpercentile(np.abs(traces[:, visible]), 98, axis=1)
    # Gain unit from the large (near-site) channels rather than the median:
    # with rows ordered by distance, a median unit let the front responses
    # run far below the axis.  Distant rows then read as nearly flat, which
    # is the physical fall-off of the evoked response.
    robust_amplitude = max(float(np.percentile(per_channel, 90)), 1.0)
    spacing = RIDGE_SPACING * robust_amplitude
    n_rows = len(channel_names)
    offsets = np.arange(n_rows - 1, -1, -1) * spacing

    # The -10 to 10 ms stimulus artefact (~5x the neural signal, excluded from
    # both analysis windows) is shown in light ink and clipped so it does not
    # spike through the whole stack.
    gap = (time_ms >= -10.0) & (time_ms <= 10.0)
    shown = traces.copy()
    shown[:, gap] = np.clip(shown[:, gap], -ARTEFACT_CLIP * robust_amplitude,
                            ARTEFACT_CLIP * robust_amplitude)

    segments = (
        ((time_ms >= -310) & (time_ms <= -10), _tint(PRE_COLOR, .16)),
        ((time_ms >= -10) & (time_ms <= 10), _tint(".5", .14)),
        ((time_ms >= 10) & (time_ms <= 310), _tint(POST_COLOR, .18)),
    )
    extent = shown[:, visible] + offsets[:, None]
    # Anything below the front row is hidden by its fill, so only the front
    # row's own trough sets the lower limit (a deeper, hidden trough on the
    # row behind it had left an empty band under the stack).
    bottom = float(np.nanmin(extent[-1])) - .05 * robust_amplitude
    for row, (trace, offset, name) in enumerate(zip(shown, offsets,
                                                    channel_names)):
        z = 2 + 2 * row
        y = trace + offset
        for mask, fill in segments:
            ax.fill_between(time_ms[mask], y[mask], bottom, color=fill,
                            lw=0, zorder=z)
        body = np.where(gap, np.nan, y)
        artefact = np.where(gap, y, np.nan)
        edge = np.flatnonzero(gap)
        artefact[[edge[0] - 1, edge[-1] + 1]] = y[[edge[0] - 1, edge[-1] + 1]]
        ax.plot(time_ms, artefact, color=".60", lw=.7, zorder=z + 1)
        ax.plot(time_ms, body, color=TRACE_COLOR, lw=.85,
                solid_joinstyle="round", zorder=z + 1)
        ax.text(-316, offset, name, ha="right", va="center",
                fontsize=figstyle.SMALL, color=TRACE_COLOR, zorder=z + 1)

    top = float(np.nanmax(extent)) + .1 * robust_amplitude
    ax.set_xlim(-310, 310)
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.set_xticks([-300, -200, -100, 0, 100, 200, 300])
    ax.set_xlabel("Time from stimulation (ms)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(".35")
    ax.spines["bottom"].set_zorder(4 + 2 * n_rows)
    ax.tick_params(axis="x", color=".35")

    # Window headers: a thin coloured rule over each window, label in ink.
    rule_y = top + .01 * (top - bottom)
    for (lo, hi), colour, label in (((-310, -10), PRE_COLOR,
                                     "pre-stimulus  −310 to −10 ms"),
                                    ((10, 310), POST_COLOR,
                                     "post-stimulus  10 to 310 ms")):
        ax.plot([lo, hi], [rule_y, rule_y], color=colour, lw=3.0,
                solid_capstyle="butt", clip_on=False, zorder=4)
        ax.text((lo + hi) / 2, rule_y + .02 * (top - bottom), label,
                ha="center", va="bottom", fontsize=figstyle.TITLE,
                color=TRACE_COLOR, clip_on=False)

    # Vertical physical scale bar (common gain) beside the front row.
    scale_uv = float(max(10, 100 * np.floor(robust_amplitude / 100)))
    xbar = 318
    ybar = offsets[-1]
    ax.plot([xbar, xbar], [ybar, ybar + scale_uv], color=TRACE_COLOR,
            lw=1.5, solid_capstyle="butt", clip_on=False)
    ax.text(xbar + 6, ybar + scale_uv / 2, f"{scale_uv:g} µV", ha="left",
            va="center", fontsize=figstyle.SMALL, color=TRACE_COLOR,
            clip_on=False)
    return robust_amplitude


def fsaverage_mesh(hemi: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fsaverage pial vertices, faces and per-vertex curvature of one hemisphere."""
    surf = Path(mne.datasets.fetch_fsaverage(verbose=False)) / "surf"
    vertices, faces = mne.read_surface(str(surf / f"{hemi}.pial"))
    curvature = read_curvature(str(surf / f"{hemi}.curv"), binary=False)
    return (vertices.astype(np.float32), faces.astype(np.int32),
            curvature.astype(np.float32))


def load_example(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        example = {key: data[key] for key in data.files}
    example["meta"] = json.loads(str(example["meta"]))
    return example


def draw_example(brain_ax, trace_ax, example: dict) -> float:
    """Draw the coverage and ridge panels on existing axes."""
    mesh = fsaverage_mesh(example["meta"]["hemisphere"])
    brain_panel(brain_ax, example["recording_xyz"], example["stim_xyz"], mesh,
                example["meta"]["hemisphere"])
    return waveform_panel(trace_ax, example["time_ms"], example["traces_uv"],
                          [str(name) for name in example["channel_names"]])
