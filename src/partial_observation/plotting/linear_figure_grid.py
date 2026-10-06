"""The 2x2 eigenvalue-recovery grid of Fig. 2C and the state-network panel.

``draw_grid`` lays out recovery over the two factors of H = P Q: columns are
observability (delay depth, raising rank P), rows are excitation (passive or
impulse, raising rank Q). ``panel_state_network`` draws the 24-state wiring
used in Figs. S1-S2. ``draw_delay_matrix`` and ``draw_traces`` are the
building blocks of the setting panels in ``linear_figure_setup``.
"""
from __future__ import annotations


import networkx as nx
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.patches import FancyBboxPatch


from partial_observation.plotting import linear_figure_base as base
from partial_observation.simulations import linear_network as sim

FS = base.FS
# Sites are printed as i / ii / iii here; the simulation keys are unchanged.
base.SITE_LABELS = {"site A": "site i", "site B": "site ii", "site C": "site iii"}

# Layout knobs. Width is generous so that the 2x2 grid and the three site
# spectra can all stay square without shrinking.
FIG_W, FIG_H = 13.6, 6.9
COL_RATIO = (1.0, 1.22)       # left (A) : right (B-D)
SIDE_W = 0.46                 # width of the rank-Q annotation strip, in cell widths
SIDE_LABEL_X, SIDE_ARROW_X, SIDE_TEXT_X = 0.08, 0.30, 0.50   # main label, arrow, values
TICKS = [-1, 0, 1]            # the large cells would otherwise pick 0.5 steps
AXIS_LABEL_FS = 9             # Re(lambda) / Im(lambda) labels, a step below FS.LABEL
SITES_LEGEND_Y = -0.30        # legend under the site spectra, in axes fraction

# (C) state-level network: x steps per DAG layer, oscillators spread in y, and
# the two states of an oscillator stacked +-STATE_DY around its centre.
X_STEP, Y_SPREAD, STATE_DY = 2.6, 2.0, 0.32
REVERSE_COLOR = "#c51b7d"      # reverse (recurrent) edges, practical regime only
BOX_W = 0.95                  # width of the light box drawn behind each oscillator
ARROW_Y = -0.06               # upstream -> downstream arrow, in (C) axes fraction
GRID_PAD = 0.30               # empty row under the (A) grid, in cell heights
TOP_H = 0.14                  # rank-P strip over the (A) grid, in cell heights
# Positions inside that strip (axes fraction). They sit above 1.0 on purpose:
# the strip is short and its content should line up with the (B,C) titles.
OBS_Y, ARROW_TOP_Y, RANKP_Y = 1.25, 0.85, 0.32
# Delay depth under each rank-P value ("m = 1", "m = 8"), in strip axes
# fraction. None leaves the strip as it was; a figure that wants the values
# sets this (and a taller TOP_H) before calling draw_grid.
M_Y = None


def state_positions() -> dict[int, tuple[float, float]]:
    """Layered layout at the level of the 24 states.

    Same reading direction as sim.POS (upstream left, downstream right), but
    every oscillator is a vertical pair of state nodes.
    """
    layer = sim.oscillator_layers()
    pos = {}
    for lay in range(int(layer.max()) + 1):
        oscs = np.where(layer == lay)[0]
        ys = np.linspace(1.0, -1.0, len(oscs) + 2)[1:-1] if len(oscs) > 1 else [0.0]
        for osc, y in zip(oscs, ys):
            for k, sign in ((0, +1.0), (1, -1.0)):
                pos[2 * int(osc) + k] = (X_STEP * lay,
                                         Y_SPREAD * float(y) + sign * STATE_DY)
    return pos


def panel_state_network(ax, reg=None, title="stimulation network",
                        reverse_examples=None):
    """The 24-state graph of A: one node per state, one edge per non-zero
    off-diagonal entry, so the panel is the wiring diagram of the matrix in (B).

    A light box groups the two states of each oscillator. Observed states (the
    rows of C) are ringed; the states that receive the impulse (the columns of
    B) are enlarged in the site colours and labelled.
    """
    A = (base.REG if reg is None else reg).A
    pos = state_positions()
    observed = [int(s) for s in sim.OBS_STATES]
    stim_state = {2 * osc: name for name, osc in sim.STIM_OSC.items()}

    graph = nx.DiGraph()
    graph.add_nodes_from(range(sim.N))
    for i, j in np.argwhere(np.abs(A) > 1e-12):
        if i != j:
            graph.add_edge(int(j), int(i), w=float(abs(A[i, j])))
    intra = [(u, v) for u, v in graph.edges if u // 2 == v // 2]
    # Reverse edges (downstream -> upstream) exist only in the recurrent A of
    # the practical regime; they are drawn dashed so the forward wiring reads
    # the same as in the feedforward network.
    reverse = [(u, v) for u, v in graph.edges if u // 2 > v // 2]
    inter = [(u, v) for u, v in graph.edges if u // 2 < v // 2]

    # Oscillator boxes first, so everything else sits on top of them.
    for osc in range(sim.N_OSC):
        x, y_top = pos[2 * osc]
        _, y_bot = pos[2 * osc + 1]
        pad = 0.28
        ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y_bot - pad), BOX_W,
                                    (y_top - y_bot) + 2 * pad,
                                    boxstyle="round,pad=0,rounding_size=0.25",
                                    fc="0.95", ec="0.82", lw=0.6, zorder=0))

    node_size = 95
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=inter, node_size=node_size,
                           edge_color="0.72", arrows=True, arrowsize=5,
                           width=[0.3 + 2.5 * graph[u][v]["w"] for u, v in inter],
                           connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=intra, node_size=node_size,
                           edge_color="0.45", arrows=True, arrowsize=5,
                           width=[0.3 + 1.2 * graph[u][v]["w"] for u, v in intra],
                           connectionstyle="arc3,rad=0.35")

    # Reverse edges on top of the forward wiring, in a colour no other panel
    # uses, so the recurrence stands out; the structural A has none.
    n_reverse_osc = len({(u // 2, v // 2) for u, v in reverse})
    label = "reverse edge"
    if reverse and reverse_examples is not None:
        # Only a few examples, one state-level edge (the strongest) per chosen
        # oscillator pair, so the reachability of the sites stays readable.
        picked = []
        for src, dst in reverse_examples:          # forward edge src -> dst
            cand = [(u, v) for u, v in reverse if u // 2 == dst and v // 2 == src]
            picked.append(max(cand, key=lambda e: graph[e[0]][e[1]]["w"]))
        reverse = picked
        label = (f"reverse edge: one per forward edge,\n"
                 f"{len(picked)} of {n_reverse_osc} shown")
    if reverse:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=reverse,
                               node_size=node_size, edge_color=REVERSE_COLOR,
                               style="dashed", arrows=True,
                               arrowsize=4 if reverse_examples is None else 9,
                               width=0.7 if reverse_examples is None else 1.6,
                               alpha=0.75 if reverse_examples is None else 1.0,
                               connectionstyle="arc3,rad=0.06" if reverse_examples is None
                               else "arc3,rad=0.35")
        ax.legend(handles=[Line2D([], [], color="0.72", lw=1.2, label="forward edge"),
                           Line2D([], [], color=REVERSE_COLOR, lw=1.2, ls="--",
                                  label=label)],
                  loc="upper center", bbox_to_anchor=(0.5, ARROW_Y - 0.03),
                  ncol=2, fontsize=FS.SMALL, frameon=False, handlelength=1.8)
    plain = [s for s in range(sim.N) if s not in stim_state]
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=plain, node_size=node_size,
                           node_color="white", edgecolors="0.6", linewidths=0.8)
    for state, name in stim_state.items():
        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[state], node_size=230,
                               node_color=sim.SITE_COLORS[name],
                               edgecolors="white", linewidths=1.0)
        x, y = pos[state]
        # Label to the left of the node, offset in points so that it clears the
        # marker whatever the data scale; above the node sits the other state.
        ax.annotate(base.site_label(name, short=True), xy=(x, y),
                    xytext=(-11, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=FS.LABEL,
                    fontweight="bold", color=sim.SITE_COLORS[name], zorder=10,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                              alpha=0.8) if reverse else None)
    # Observed states: a black ring, drawn last so it also shows on a stim node.
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=observed,
                           node_size=[230 if s in stim_state else node_size
                                      for s in observed],
                           node_color="none", edgecolors="black", linewidths=1.4)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.4, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_axis_off()
    # Reading direction as an arrow under the wiring, in the style of the
    # rank-P arrow over (A): upstream at the tail, downstream at the head.
    ax.annotate("", xy=(0.92, ARROW_Y), xytext=(0.08, ARROW_Y),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.3))
    ax.text(0.08, ARROW_Y + 0.03, "upstream", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS.LABEL, color="0.3")
    ax.text(0.92, ARROW_Y + 0.03, "downstream", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS.LABEL, color="0.3")
    ax.set_title(title, fontsize=FS.TITLE)


# Setting schematics (drawn by linear_figure_setup.py, not in this figure).
DELAY_SAMPLES = 100           # columns of the delay matrices
TRACE_WINDOW = (-0.10, 0.30)  # seconds around the impulse shown in the traces
N_OBS = len(sim.OBS_STATES)


def draw_delay_matrix(host, rect, d: int, Y: np.ndarray, color: str):
    """The delay-embedded data matrix H_d = [Y_t; Y_t+1; ...] as a heatmap in
    ``rect`` (host axes fraction, [x0, y0, w, h_full]). The drawn height is
    h_full * d / DELAY from the bottom of the rect, so matrices for different d
    share a baseline and their heights compare the row counts p*d."""
    H = sim.delay_matrix(Y[:, :DELAY_SAMPLES + d], d)
    x0, y0, w, h_full = rect
    height = h_full * (d / base.DELAY)          # proportional to the row count
    ax = host.inset_axes([x0, y0, w, height])
    vmax = float(np.abs(H).max())
    ax.imshow(H, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(1.0)
    ax.text(0.5, 1.0, f"${N_OBS}\\times{d}$ rows", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=FS.SMALL, color=color)
    return ax


def draw_traces(host, rect, t: np.ndarray, Y: np.ndarray,
                ylim: tuple[float, float], color: str, impulse: bool):
    """The 8 observed channels around the impulse time, as stacked traces in
    ``rect`` (host axes fraction). Passive and impulse panels share ``ylim`` so
    that the impulse is visible as the difference between them."""
    ax = host.inset_axes(list(rect))
    sel = (t >= TRACE_WINDOW[0]) & (t <= TRACE_WINDOW[1])
    offsets = np.linspace(1, -1, Y.shape[0]) * 0.5 * (ylim[1] - ylim[0])
    for row, off in zip(Y[:, sel], offsets):
        ax.plot(t[sel], row + off, color="0.25", lw=0.6)
    span = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0] - 0.5 * span, ylim[1] + 0.5 * span)
    ax.set_xlim(*TRACE_WINDOW)
    if impulse:
        ax.axvline(0.0, color=color, lw=1.2)
        ax.text(0.0, 0.97, " impulse",
                transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=FS.ANNOT, fontweight="bold", color=color)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(1.0)
    ax.text(0.5, -0.05, f"{TRACE_WINDOW[1] - TRACE_WINDOW[0]:.1f} s, "
            f"{Y.shape[0]} channels", transform=ax.transAxes,
            ha="center", va="top", fontsize=FS.SMALL, color="0.45")
    return ax


LEGEND_LABELS = {"Hankel-DMD": "estimated"}   # printed names for legend entries
RANK_PLAIN_COLOR = "black"    # rank values of the non-highlighted (not "both full") cases


def legend_entries(ax):
    """Handles and printed labels of an eigenvalue panel's legend."""
    handles, labels = ax.get_legend_handles_labels()
    return handles, [LEGEND_LABELS.get(label, label) for label in labels]


def draw_grid(fig, spec, letter: str, legend: bool = False,
              side_label_x: float = SIDE_LABEL_X,
              side_arrow_x: float = SIDE_ARROW_X,
              side_text_x: float = SIDE_TEXT_X):
    """(A) of v2: the 2x2 eigenvalue grid with its rank-P / rank-Q annotations,
    laid out inside ``spec``. With ``legend`` the eigenvalue legend goes into
    the pad row under the grid. ``side_label_x``, ``side_arrow_x`` and
    ``side_text_x`` place the excitation label, its arrow and the rank-Q values
    inside the side strip (larger = closer to the cells). Returns
    (results, rP, rQ, legend_source_axes)."""
    # A pad row at the bottom soaks up the height the square cells cannot use,
    # so the two rows sit close together instead of floating in tall slots.
    grid = spec.subgridspec(4, 3, width_ratios=[SIDE_W, 1, 1],
                            height_ratios=[TOP_H, 1, 1, GRID_PAD],
                            wspace=0.12, hspace=0.10)
    stim = sim.STIM_OSC[base.SITE]
    cells = [("passive | m=1",  grid[1, 1], None, 1,        False, False, True),
             ("passive | m=8",  grid[1, 2], None, base.DELAY, False, False, False),
             ("impulse | m=1",  grid[2, 1], stim, 1,        False, True,  True),
             ("impulse | m=8",  grid[2, 2], stim, base.DELAY, True,  True,  False)]

    results, rP, rQ = {}, {}, {}
    legend_source = None
    for key, slot, osc, d, highlight, show_x, show_y in cells:
        ax = fig.add_subplot(slot)
        rec, rp, rq = base.panel_eigs(ax, osc, d, highlight=highlight,
                                    show_x=show_x, show_y=show_y)
        # The square cell is shorter than its slot; anchor it at the top so the
        # top row sits right under the rank-P labels.
        ax.set_anchor("N")
        ax.set_xticks(TICKS); ax.set_yticks(TICKS)
        ax.xaxis.label.set_size(AXIS_LABEL_FS)
        ax.yaxis.label.set_size(AXIS_LABEL_FS)
        if not show_x:
            ax.set_xticklabels([])
        if not show_y:
            ax.set_yticklabels([])
        results[key] = rec
        rP[d] = rp
        rQ["perturbed" if osc is not None else "passive"] = rq
        if highlight:
            legend_source = ax

    corner = fig.add_subplot(grid[0, 0])
    corner.axis("off")
    corner.text(0.0, OBS_Y, f"({letter})", transform=corner.transAxes,
                fontweight="bold", fontsize=FS.LETTER, va="center", ha="left")

    # Axis annotations: rank P across the columns, rank Q down the rows.
    top = fig.add_subplot(grid[0, 1:3])
    top.axis("off")
    top.annotate("", xy=(0.95, ARROW_TOP_Y), xytext=(0.05, ARROW_TOP_Y),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.3))
    depth = "" if M_Y is None else " $m$"
    top.text(0.5, OBS_Y, r"rank of observability matrix $P$  (delay depth" + depth + ")",
             ha="center", va="center", fontsize=FS.LABEL)
    top.text(0.26, RANKP_Y, f"rank $P={rP[1]}$",
             ha="center", va="center", fontsize=base.RANK_FS, color=RANK_PLAIN_COLOR,
             fontweight="bold")
    top.text(0.76, RANKP_Y, f"rank $P={rP[base.DELAY]}$",
             ha="center", va="center", fontsize=base.RANK_FS, color=base.GREEN,
             fontweight="bold")
    if M_Y is not None:
        # The delay depth behind each rank, in the style of the m values of
        # the nonlinear grid: plain grey for m = 1, green for the deep one,
        # at the size of the rank values.
        top.text(0.26, M_Y, "$m=1$", ha="center", va="center",
                 fontsize=base.RANK_FS, color="0.45")
        top.text(0.76, M_Y, f"$m={base.DELAY}$", ha="center", va="center",
                 fontsize=base.RANK_FS, color=base.GREEN, fontweight="bold")

    side = fig.add_subplot(grid[1:3, 0])
    side.axis("off")
    side.annotate("", xy=(side_arrow_x, 0.05), xytext=(side_arrow_x, 0.95),
                  xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.3))
    side.text(side_label_x, 0.5, r"rank of trajectory matrix $Q$  (impulse)",
              rotation=90, ha="center", va="center", fontsize=FS.LABEL)
    side.text(side_text_x, 0.75, f"rank $Q={rQ['passive']}$",
              rotation=90, ha="center", va="center", fontsize=base.RANK_FS,
              color=RANK_PLAIN_COLOR, fontweight="bold")
    side.text(side_text_x, 0.25, f"rank $Q={rQ['perturbed']}$",
              rotation=90, ha="center", va="center", fontsize=base.RANK_FS,
              color=base.GREEN, fontweight="bold")
    if legend:
        handles, labels = legend_entries(legend_source)
        ax_leg = fig.add_subplot(grid[3, 1:3])
        ax_leg.axis("off")
        ax_leg.legend(handles, labels, loc="lower center", ncol=len(labels),
                      fontsize=FS.LEGEND, frameon=False,
                      handletextpad=0.35, columnspacing=1.8)
    return results, rP, rQ, legend_source
