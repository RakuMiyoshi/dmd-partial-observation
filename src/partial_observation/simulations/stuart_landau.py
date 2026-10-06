"""Nonlinear simulation of Fig. 2D-F: Stuart-Landau oscillators under partial observation.

Four uncoupled limit-cycle oscillators are observed through two linear
projections of their eight-dimensional state. Each oscillator contributes two
families of Koopman eigenvalues: a marginally stable phase mode at +-i*omega,
which the ongoing oscillation always expresses, and damped isostable modes at
-2k*delta +- i*omega (k = 1, 2, ...), which describe relaxation back onto the
limit cycle and are expressed only when the trajectory is pushed off it. The
phase modes therefore stand in for the persistent modes of the linear system
and the isostable modes for the ones that need excitation.

``panel_flow`` draws the relaxation onto the limit cycle (Fig. 2D),
``panel_observed_pair`` the observed signals (Fig. 2E), and ``panel_grid`` the
DMD spectra over delay depth and impulse (Fig. 2F).
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from pydmd import DMD, HankelDMD

from partial_observation.plotting import palette as RCFG
from partial_observation.plotting import style as figstyle

figstyle.use_style()
FS = figstyle


# ---- system ----------------------------------------------------------------
DT = 0.002
T = 4.0
F_HZ = np.array([4.0, 8.0, 15.0, 30.0])
W = 2 * np.pi * F_HZ
N = 4                                  # oscillators; 2N = 8 real state variables
P_OBS = 2                              # observed projections
BETA = np.array([0.6, -0.4, 0.8, -0.7])
DELTA = np.array([0.6, 0.8, 1.2, 2.0])
GAMMA = W + BETA * DELTA
RSTAR = np.sqrt(DELTA)

# ---- sweeps ----------------------------------------------------------------
D_MAIN_LOW, D_MAIN_HIGH = 1, 200       # the two columns of panel (a)
ALPHA_MAIN = 0.20                      # the impulse used in panel (a)
D_LIST = np.array([1, 30, 60, 120, 200])
A_LIST = np.array([0.0, 0.05, 0.10, 0.20, 0.30])
TRIALS = 5

# ---- palette (shared with the other figures) -------------------------------
C_PASSIVE = RCFG.WINDOW_COLORS["pre"]
C_IMPULSE = RCFG.WINDOW_COLORS["early"]
C_PHASE = RCFG.GROUP_COLORS["persistent"]
C_ISO = ["#084594", "#4292C6", "#9ECAE1"]     # isostable -2d, -4d, -6d
C_DMD = RCFG.GROUP_COLORS["fast-decay"]
# Open-circle outlines: the fast-decaying family in orange shades (darkest k=1),
# matching the orange fast-decaying circles of the linear grid.
C_ISO_OPEN = ["#d95f02", "#f39c4a", "#f7c89b"]
ISO_LABELS = [r"true fast-decaying ($k=1$)", r"true fast-decaying ($k=2$)",
              r"true fast-decaying ($k=3$)"]


def random_C(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((P_OBS, 2 * N))
    return obs / np.linalg.norm(obs, axis=1, keepdims=True)


def simulate_state(kind: str, alpha: float, seed: int):
    """Full state X (2N x T) of the Stuart-Landau network (analytic solution).

    ``kind`` is "passive" (unperturbed limit cycle) or "impulse" (a full-state
    kick of amplitude ``alpha`` at t = 0). Returns (t, X, amplitude), where
    ``amplitude`` is the per-oscillator complex amplitude over time.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, DT)
    amp0 = RSTAR * np.exp(1j * rng.uniform(0, 2 * np.pi, N))
    if kind == "impulse":
        kick = alpha * np.ones(2 * N)
        amp0 = amp0 + (kick[0::2] + 1j * kick[1::2])

    state = np.zeros((2 * N, len(t)))
    amps = np.zeros((N, len(t)), complex)
    for i in range(N):
        r0 = max(abs(amp0[i]), 1e-8)
        phase0 = np.angle(amp0[i])
        q = DELTA[i] / r0 ** 2 - 1.0
        r_sq = DELTA[i] / (1.0 + q * np.exp(-2 * DELTA[i] * t))
        integ = 0.5 * (np.log(np.exp(2 * DELTA[i] * t) + q) - np.log(1.0 + q))
        amp = np.sqrt(r_sq) * np.exp(1j * (phase0 + GAMMA[i] * t - BETA[i] * integ))
        state[2 * i], state[2 * i + 1] = amp.real, amp.imag
        amps[i] = amp
    return t, state, amps


def simulate(kind: str, alpha: float, obs: np.ndarray, seed: int) -> np.ndarray:
    """Observed series Y = C X of the Stuart-Landau network."""
    _, state, _ = simulate_state(kind, alpha, seed)
    return obs @ state


def dmd_eigs(Y: np.ndarray, d: int, amp_frac: float = 0.01) -> np.ndarray:
    """Continuous-time DMD eigenvalues, weak-amplitude modes dropped."""
    model = DMD(svd_rank=0, exact=True) if d <= 1 else HankelDMD(svd_rank=0, d=d,
                                                                 exact=True)
    # Passive and shallow fits are ill-conditioned by design (that is the
    # point of the figure); PyDMD warns about it on every such fit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="pydmd")
        model.fit(Y)
    ev = np.asarray(model.eigs, complex)
    if ev.size == 0:
        return np.array([], complex)
    amp = np.abs(np.asarray(model.amplitudes))
    keep = (np.abs(ev) > 1e-10) & (np.abs(ev) < 1.05) & (amp > amp_frac * amp.max())
    return np.log(ev[keep]) / DT


def isostable_targets():
    """Ground-truth isostable eigenvalues, harmonics k = 1..3."""
    return [np.array([-2 * (k + 1) * DELTA[i] + s * 1j * W[i]
                      for i in range(N) for s in (-1, 1)]) for k in range(3)]


def phase_targets():
    return np.array([s * 1j * w for w in W for s in (-1, 1)])


def hits(lam: np.ndarray, which: str) -> int:
    """How many of the N oscillators have their target mode recovered."""
    tol_im = 0.3 * np.min(np.diff(np.sort(np.concatenate([W, -W]))))
    tol_re = 0.4 * (2 * DELTA)
    count = 0
    for i in range(N):
        re_target = 0.0 if which == "phase" else -2 * DELTA[i]
        count += any(abs(l.real - re_target) < tol_re[i]
                     and abs(abs(l.imag) - W[i]) < tol_im for l in lam)
    return count


GREEN = "#2e7d32"          # "both levers sufficient" accent, as in Fig. 2
REP_OSC = 0                # representative oscillator for the phase portrait
ALPHA_CYCLE = 0.30         # impulse used in the limit-cycle / relaxation panels


PHASE_TMAX = 1.3           # window for the phase portrait, so the spiral shows
FLOW_LEGEND_Y = -0.20      # legend under panel (A), in axes fraction (below the x label)


J = REP_OSC                # oscillator shown in panel (A)
ALPHA_DISPLAY = 0.50       # panel (A) only: larger than ALPHA_MAIN so the spiral shows
FLOW_TMAX = 1.5            # seconds of the relaxation drawn in panel (A)
FLOW_LEGEND_SIDE = "below" # "below": two columns under the axes (FLOW_LEGEND_Y);
                           # "right": one column beside it (FLOW_LEGEND_XY, FLOW_ANCHOR)
FLOW_LEGEND_XY = (1.05, 0.5)   # legend anchor (axes fraction) for the "right" layout
FLOW_ANCHOR = (0.18, 0.5)      # where the square axes sits in its slot, "right" layout
OBS_LABEL_GAP = 0.02       # y_1 / y_2 labels of (B): offset left of the axes, in axes width
ALPHA_OBS = 2.0            # impulse shown in the observed-signal pair (Fig. 2E): large
                           # enough for the amplitude transient to show in y_1, y_2
OBS_T_PRE = 0.5            # seconds of the passive cycle shown before the impulse
GRID_LEGEND_NCOL = 0       # columns of the legend under the spectra grid; 0 = one row
GRID_LEGEND_Y = None       # top of that legend in pad-row axes fraction; None = sit at the bottom


def lab_field(re, im):
    """d(zeta)/dt of Eq. (stuart_landau) in the laboratory (non-rotating) frame."""
    z = re + 1j * im
    dz = (DELTA[J] + 1j * GAMMA[J]) * z - (1 + 1j * BETA[J]) * np.abs(z) ** 2 * z
    return dz.real, dz.imag


def panel_flow(ax):
    """Flow of one oscillator and a trajectory spiralling back onto its cycle.

    Drawn in the laboratory frame, so the relaxation appears as a spiral. The
    impulse is ALPHA_DISPLAY rather than ALPHA_MAIN, purely for visibility; the
    phase at t=0 is that of the fitted trial (seed 11).
    """
    lim = 1.35
    grid = np.linspace(-lim, lim, 31)
    X, Y = np.meshgrid(grid, grid)
    U, V = lab_field(X, Y)
    ax.streamplot(X, Y, U, V, color="0.72", density=1.0, linewidth=.6,
                  arrowsize=.7)
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.plot(RSTAR[J] * np.cos(phi), RSTAR[J] * np.sin(phi), color="black", lw=2.0,
            label="limit cycle (passive)", zorder=4)
    t, _, amp_p = simulate_state("passive", 0.0, seed=11)
    _, _, amp_i = simulate_state("impulse", ALPHA_DISPLAY, seed=11)
    win = t <= FLOW_TMAX
    start, kicked = amp_p[J, 0], amp_i[J, 0]
    # Shaded by time (dark at the impulse, fading onto the cycle): the spiral
    # is tightly wound, so the shading carries the direction of relaxation.
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    path = np.column_stack([amp_i[J, win].real, amp_i[J, win].imag])
    segments = np.stack([path[:-1], path[1:]], axis=1)
    fade = LinearSegmentedColormap.from_list("fade", [C_IMPULSE, "#f4c7c3"])
    ax.add_collection(LineCollection(segments, cmap=fade, array=t[win][:-1],
                                     linewidths=1.1, zorder=5))
    ax.plot([], [], color=C_IMPULSE, lw=1.1,
            label=fr"after impulse ($\alpha={ALPHA_DISPLAY:g}$)")
    ax.annotate("", xy=(kicked.real, kicked.imag), xytext=(start.real, start.imag),
                arrowprops=dict(arrowstyle="-|>", color=C_IMPULSE, lw=1.1,
                                ls="--", shrinkA=3, shrinkB=2), zorder=6)
    ax.scatter([start.real], [start.imag], s=34, color=C_PASSIVE,
               edgecolors="white", linewidths=.6, zorder=7)
    ax.scatter([0], [0], s=30, facecolor="white", edgecolor="black", zorder=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(fr"Re $a_{{{J+1}}}$")
    ax.set_ylabel(fr"Im $a_{{{J+1}}}$")
    ax.set_title("relaxation onto the limit cycle", fontsize=FS.TITLE)
    if FLOW_LEGEND_SIDE == "right":
        # Legend stacked to the right of the (square) axes; the axes is
        # anchored towards the left of its slot to leave room for it.
        ax.set_anchor(FLOW_ANCHOR)
        ax.legend(fontsize=FS.SMALL, loc="center left",
                  bbox_to_anchor=FLOW_LEGEND_XY, ncol=1, frameon=False,
                  handlelength=1.6, labelspacing=0.8)
    else:
        ax.legend(fontsize=FS.SMALL, loc="upper center",
                  bbox_to_anchor=(.5, FLOW_LEGEND_Y), ncol=2, frameon=False,
                  handlelength=1.6, columnspacing=1.0)


def observed_with_pre(kind: str, alpha: float, obs: np.ndarray, seed: int,
                      t_pre: float):
    """Observed series with ``t_pre`` seconds of the passive limit cycle
    prepended, so that time runs from -t_pre to T and the impulse (if any)
    falls at t = 0, as in the text: x(0+) = x(0-) + alpha b.

    Before t = 0 every oscillator is on its cycle, where the closed-form
    solution is a_j(t) = a_j(0) exp(i w_j t) for negative t as well.
    """
    rng = np.random.default_rng(seed)
    amp0 = RSTAR * np.exp(1j * rng.uniform(0, 2 * np.pi, N))   # as in simulate_state
    t_neg = np.arange(-t_pre, 0, DT)
    amp_neg = amp0[:, None] * np.exp(1j * W[:, None] * t_neg[None, :])
    state_neg = np.zeros((2 * N, len(t_neg)))
    state_neg[0::2], state_neg[1::2] = amp_neg.real, amp_neg.imag
    t_pos, state_pos, _ = simulate_state(kind, alpha, seed)
    t = np.concatenate([t_neg, t_pos])
    Y = obs @ np.concatenate([state_neg, state_pos], axis=1)
    return t, Y


def panel_observed(ax, t_max: float = 2.0, kind: str = "passive",
                   alpha: float = 0.0, step: float | None = None,
                   title: str | None = None, ylabels: bool = True,
                   t_pre: float = 0.0, mark_impulse: bool = False) -> float:
    """The two observed projections y_1, y_2 of one trajectory.

    Same observation matrix and seed as the spectra. ``kind`` / ``alpha``
    select the passive trajectory or the one after the impulse (t = 0 is the
    kick). ``t_pre`` seconds of the passive cycle are shown before t = 0, and
    the window is [-t_pre, t_max - t_pre] so the panel is always t_max wide.
    ``step`` is the vertical offset between y_1 and y_2; pass the value
    returned by a previous call so that two panels share the same offsets.
    Returns the offset used.
    """
    obs = random_C(11)
    t, Y = observed_with_pre(kind, alpha, obs, 11, t_pre)
    t_lo, t_hi = -t_pre, t_max - t_pre
    keep = (t >= t_lo) & (t <= t_hi)
    if step is None:
        step = 1.10 * np.ptp(Y[:, keep])
    for k, trace in enumerate(Y):
        y0 = -k * step
        ax.plot(t[keep], trace[keep] + y0, color="0.30", lw=.7)
        if ylabels:
            ax.text(-OBS_LABEL_GAP, y0, fr"$y_{{{k+1}}}$",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=FS.TICK)
    if mark_impulse:
        ax.axvline(0, color=C_IMPULSE, ls="--", lw=0.9, zorder=0)
    ax.set_xlim(t_lo, t_hi)
    tick_step = 0.5 if (t_hi - t_lo) <= 1.5 else 1.0
    ticks = np.arange(np.ceil(t_lo / tick_step) * tick_step, t_hi + 1e-9, tick_step)
    ticks[np.isclose(ticks, 0.0)] = 0.0      # not '-0'
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    if title is None:   # the impulse size is given in the caption, not here
        title = "observed signals (passive)" if kind == "passive" else "after impulse"
    ax.set_title(title, fontsize=FS.TITLE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return step


def panel_observed_pair(fig, gridspec, t_max: float = 2.0,
                        alpha: float | None = None, t_pre: float | None = None,
                        wspace: float = 0.12):
    """(E) as two side-by-side panels: passive on the left, the impulse at
    t = 0 on the right, with a common vertical offset and y-range so that the
    swelling of the amplitude after the kick can be read against the passive
    trace. The impulse is ALPHA_OBS (larger than ALPHA_MAIN: at 0.2 the
    transient is buried in the beating of the four oscillators, see the
    caption). Returns (ax_passive, ax_impulse)."""
    alpha = ALPHA_OBS if alpha is None else alpha
    t_pre = OBS_T_PRE if t_pre is None else t_pre
    inner = gridspec.subgridspec(1, 2, wspace=wspace)
    ax_p = fig.add_subplot(inner[0])
    ax_i = fig.add_subplot(inner[1], sharey=ax_p)
    # The offset is set by the larger (impulse) trace so nothing overlaps.
    step = panel_observed(ax_i, t_max, "impulse", alpha, ylabels=False,
                          t_pre=t_pre, mark_impulse=True)
    panel_observed(ax_p, t_max, "passive", 0.0, step=step, title="passive",
                   t_pre=t_pre)
    return ax_p, ax_i


def _spectrum_cell(ax, kind, alpha, d, obs, phase_gt, iso_gt, *, highlight,
                   show_x, show_y, legend):
    lam = dmd_eigs(simulate(kind, alpha, obs, seed=11), d)
    iso = hits(lam, "iso")
    if highlight:
        ax.set_facecolor("#eef6ee")
        for spine in ax.spines.values():
            spine.set_edgecolor(GREEN)
            spine.set_linewidth(1.6)
    ax.axvline(0, color="0.82", lw=0.8)
    ax.axhline(0, color="0.92", lw=0.6)
    # Open circles and black crosses, as in the linear grid (Fig. 2C).
    ax.scatter(phase_gt.real, phase_gt.imag, s=38, facecolors="white",
               edgecolors=C_PHASE, linewidths=1.3,
               label=r"true non-decaying ($\pm i\omega$)")
    for gt, colour, lab in zip(iso_gt, C_ISO_OPEN, ISO_LABELS):
        ax.scatter(gt.real, gt.imag, s=38, facecolors="white", edgecolors=colour,
                   linewidths=1.3, label=lab)
    if len(lam):
        ax.scatter(lam.real, lam.imag, s=34, marker="x", linewidths=1.3,
                   color="black", zorder=5, label="estimated")
    ax.set_xlim(-15.5, 2.5)
    ax.set_ylim(-240, 240)
    if show_x:
        ax.set_xlabel(r"Re($\lambda$)")
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel(r"Im($\lambda$)")
    else:
        ax.set_yticklabels([])
    return iso, ax.get_legend_handles_labels()


def panel_grid(fig, gridspec, letter: str = "C", legend: bool = False,
               square: bool = False, side_w: float = 0.62,
               top_h: float = 0.22, pad: float = 0.0,
               side_label_x: float = 0.06, side_arrow_x: float = 0.24,
               side_text_x: float = 0.40, value_fs: float | None = None):
    """The 2x2 spectra with observability columns and excitation rows, laid out
    and annotated exactly as the linear Figure 2. Returns the marker legend so
    it can be placed once, outside the cells (or, with ``legend``, places it
    in a pad row under the cells, as linear_figure_grid.draw_grid does).
    ``square`` pins the cells to a square box, anchored at the top, so the
    grid matches the linear one when the two sit side by side. ``value_fs``
    is the size of the m = ... and passive / impulse values (default ANNOT)."""
    if value_fs is None:
        value_fs = FS.ANNOT
    obs = random_C(11)
    phase_gt, iso_gt = phase_targets(), isostable_targets()
    # Wider left/top margins than the linear figure: the axis labels here carry
    # units, so they need room to clear the excitation arrow and the titles.
    if legend or pad > 0:
        grid = gridspec.subgridspec(4, 3, width_ratios=[side_w, 1, 1],
                                    height_ratios=[top_h, 1, 1, max(pad, 0.30)],
                                    wspace=0.14, hspace=0.10 if square else 0.44)
    else:
        grid = gridspec.subgridspec(3, 3, width_ratios=[side_w, 1, 1],
                                    height_ratios=[top_h, 1, 1],
                                    wspace=0.14, hspace=0.44)
    #        slot        kind       alpha       d            hl     sx     sy
    cells = [(grid[1, 1], "passive", 0.0,       D_MAIN_LOW,  False, False, True),
             (grid[1, 2], "passive", 0.0,       D_MAIN_HIGH, False, False, False),
             (grid[2, 1], "impulse", ALPHA_MAIN, D_MAIN_LOW,  False, True,  True),
             (grid[2, 2], "impulse", ALPHA_MAIN, D_MAIN_HIGH, True,  True,  False)]
    handles = labels = None
    # One letter for the whole 2x2 (C), on its upper-left cell; the text refers
    # to the cells by position (upper left, ..., lower right).
    positions = ("upper left", "upper right", "lower left", "lower right")
    for index, ((slot, kind, alpha, d, hl, sx, sy), where) in enumerate(
            zip(cells, positions)):
        ax = fig.add_subplot(slot)
        iso, (handles, labels) = _spectrum_cell(
            ax, kind, alpha, d, obs, phase_gt, iso_gt,
            highlight=hl, show_x=sx, show_y=sy, legend=False)
        if square:
            ax.set_box_aspect(1.0)
            ax.set_anchor("N")
        lam = dmd_eigs(simulate(kind, alpha, obs, seed=11), d)
        print(f"({letter}, {where}) {kind:7s} m={d:3d}: isostable {iso}/{N}, "
              f"phase {hits(lam, 'phase')}/{N}, {len(lam)} DMD eigenvalues")
        if index == 0 and not square:
            ax.text(-0.05, 1.05, f"({letter})", transform=ax.transAxes,
                    fontweight="bold", fontsize=FS.LETTER, va="bottom", ha="right")

    if square:
        # Letter in the corner slot, on the same line as the observability
        # title, as in the linear grid.
        corner = fig.add_subplot(grid[0, 0]); corner.axis("off")
        corner.text(0.0, 1.25, f"({letter})", transform=corner.transAxes,
                    fontweight="bold", fontsize=FS.LETTER, va="center", ha="left")

    top = fig.add_subplot(grid[0, 1:3]); top.axis("off")
    if square:
        obs_y, arrow_y, val_y = 1.25, 0.85, 0.32
    else:
        obs_y, arrow_y, val_y = 0.78, 0.35, 0.02
    top.annotate("", xy=(0.95, arrow_y), xytext=(0.05, arrow_y),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.3))
    # Same wording, sizes and value colours as the linear grid (Fig. 2C).
    top.text(0.5, obs_y, r"observability (delay depth $m$)",
             ha="center", va="center", fontsize=FS.LABEL)
    top.text(0.27, val_y, f"$m={D_MAIN_LOW}$", ha="center", va="center",
             fontsize=value_fs, color="black", fontweight="bold")
    top.text(0.75, val_y, f"$m={D_MAIN_HIGH}$", ha="center", va="center",
             fontsize=value_fs, color=GREEN, fontweight="bold")

    side = fig.add_subplot(grid[1:3, 0]); side.axis("off")
    side.annotate("", xy=(side_arrow_x, 0.05), xytext=(side_arrow_x, 0.95),
                  xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.3))
    side.text(side_label_x, 0.5, r"excitation (impulse)",
              rotation=90, ha="center", va="center", fontsize=FS.LABEL)
    side.text(side_text_x, 0.75, "passive", rotation=90, ha="center",
              va="center", fontsize=value_fs, color="black", fontweight="bold")
    side.text(side_text_x, 0.25, "impulse", rotation=90, ha="center",
              va="center", fontsize=value_fs, color=GREEN, fontweight="bold")
    if legend:
        ax_leg = fig.add_subplot(grid[3, 1:3]); ax_leg.axis("off")
        ncol = GRID_LEGEND_NCOL or len(labels)
        # matplotlib fills a legend column by column; reorder so that the
        # entries read row by row (phase, k=1, k=2 / k=3, DMD for ncol=3).
        n = len(labels)
        col_sizes = [len(c) for c in np.array_split(np.arange(n), ncol)]
        col_start = np.cumsum([0] + col_sizes[:-1])
        slots = [0] * n           # slots[k] = label shown in legend slot k
        for i in range(n):
            row, col = divmod(i, ncol)
            slots[col_start[col] + row] = i
        if GRID_LEGEND_Y is None:
            place = dict(loc="lower center")
        else:   # hang the legend from GRID_LEGEND_Y, below the Re(lambda) labels
            place = dict(loc="upper center", bbox_to_anchor=(0.5, GRID_LEGEND_Y))
        ax_leg.legend([handles[i] for i in slots], [labels[i] for i in slots],
                      ncol=ncol, fontsize=FS.LEGEND, frameon=False,
                      handletextpad=0.3, columnspacing=1.0, **place)
    return handles, labels
