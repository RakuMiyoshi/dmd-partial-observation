"""Supplementary ECoG figures S3-S5, drawn from the frozen figure inputs.

* Fig. S3 (``figS3_ccep_selection``): candidate verdicts at the
  example site, synthetic calibration of multi-step decay validation, and the
  exclusion fraction against fitted time constant in the cohort.
* Fig. S4 (``figS4_ccep_distribution``): change in the frequency x decay-time
  distribution of validated estimates.
* Fig. S5 (``figS5_ccep_robustness``): sensitivity to rank, the residual
  criterion, and the decay-validation settings.

No model is refitted. Inputs are read from ``<inputs>/supplement/`` (default
``reference_results/``; regenerate with ``scripts/ecog/03_figure_inputs.py``
and ``04_decay_rule_calibration.py``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partial_observation import config
from partial_observation.plotting import ccep_figures as cf
from partial_observation.plotting import style as fs
from partial_observation.selection import decay_gap, rate_and_frequency, residual_pass

fs.use_style()

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "reference_results"
OUT = ROOT / "outputs" / "figures"
EXAMPLE_SITE = "sub-ccepAgeUMCU27__run-031551__T27-T28"   # same site as Fig. 3C
PRIMARY = list(config.PRIMARY_ROUTES)
WINDOWS = list(config.WINDOW_LABELS.items())
CUT_HZ = config.FREQUENCY_SPLIT_HZ
CUT_TAU_MS = -1000 / config.FAST_DECAY_PER_S      # 100 ms
ROUTE_NAMES = {"standard": "DMD ($m=1$)", "hankel_d50": "Hankel DMD ($m=50$)"}
ROUTE_COLORS = {"standard": ".35", "hankel_d50": cf.FAST_COLOR}
REACH_COLOR = "#e6ab02"        # synthetic settings that pass the residual criterion
RADIAL_FAIL_COLOR = "#c51b7d"  # candidates removed by multi-step decay validation
VERDICT_COLORS = {"removed by residual criterion": ".78",
                  "removed by multi-step decay validation": RADIAL_FAIL_COLOR,
                  "validated": "#2a788e"}
# Wider in Re than Fig. 3C (0.970) so the spread removed by decay validation
# stays in view.
ZOOM_X, ZOOM_Y = (0.950, 1.006), (-0.15, 0.15)
RES_EPS = (0.05, 0.10, 0.20)       # one-step residual threshold
RES_SUPPORT = (3, 5, 7)            # held-out trials (of 9) that must pass it
FIGURES = ("validation", "distribution", "robustness")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def panel_radial_calibration(fig, spec, letters, inputs):
    """Synthetic calibration of the radial rule (primary 5 ms, delta = 0.05)."""
    from matplotlib.patches import Rectangle
    oc = pd.read_csv(inputs / "supplement" / "oracle_controls.csv")
    oc = oc[(oc.horizon_ms == config.DECAY_VALIDATION_HORIZON_MS)
            & (oc.tolerance == config.DECAY_VALIDATION_TOLERANCE) & oc.mean_removed]
    settings = [("transient", s) for s in (0., .03, .1, .3)] + \
               [("stationary_AR", s) for s in (0., .1, .3)]
    taus = sorted(oc.tau_ms.unique())
    bottom = spec.subgridspec(1, 3, width_ratios=[1, 1, .05], wspace=.18)
    for c, (factor, title) in enumerate(((1., "correct decay claimed"),
                                         (4., "decay claimed 4$\\times$ too fast"))):
        ax = fig.add_subplot(bottom[0, c])
        grid = np.full((len(settings), len(taus)), np.nan)
        reach = np.zeros_like(grid, bool)
        for i, (regime, noise) in enumerate(settings):
            for j, tau in enumerate(taus):
                row = oc[(oc.regime == regime) & (oc.measurement_noise_std == noise)
                         & (oc.tau_ms == tau) & (oc.claim_factor == factor)].iloc[0]
                grid[i, j] = row.radial_exclusion_rate
                reach[i, j] = row.n_coherent > 0
        im = ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for i in range(len(settings)):
            for j in range(len(taus)):
                ax.text(j, i, f"{100*grid[i, j]:.0f}", ha="center", va="center",
                        fontsize=fs.SMALL, color="white" if grid[i, j] > .55 else "black")
                if reach[i, j]:
                    # Inset so it is not clipped and shows on dark cells too.
                    ax.add_patch(Rectangle((j - .44, i - .42), .88, .84, fill=False,
                                           ec=REACH_COLOR, lw=2.0))
        ax.axhline(3.5, color="white", lw=3)
        ax.set_xticks(range(len(taus)), [f"{t:g}" for t in taus])
        ax.set_yticks(range(len(settings)),
                      [f"{'transient' if r == 'transient' else 'stationary AR'}, "
                       f"noise {s:g}" for r, s in settings] if c == 0 else [])
        ax.set(xlabel=r"true decay time constant $\tau$ [ms]", title=title)
        ax.text(-.14 if c == 0 else -.03, 1.06, f"({next(letters)})",
                transform=ax.transAxes, fontweight="bold", fontsize=fs.LETTER,
                va="bottom", ha="right")
    cbar = fig.colorbar(im, cax=fig.add_subplot(bottom[0, 2]))
    cbar.set_label("rejection probability (% in cells)")


def panel_exclusion(fig, spec, letters, te, td, tn):
    """Fraction excluded by multi-step decay validation against fitted tau."""
    ax = fig.add_subplot(spec)
    tc=np.sqrt(te[:-1]*te[1:])
    for route,color in (("standard",".35"),("hankel_d50",cf.FAST_COLOR)):
        for w,ls in (("pre","-"),("early","--")):
            frac=np.divide(tn[(route,w)],td[(route,w)],
                           out=np.full_like(tn[(route,w)],np.nan),
                           where=td[(route,w)]>0)
            ax.plot(tc,frac,color=color,ls=ls,lw=1.7,
                    label=f"{'DMD' if route=='standard' else 'Hankel'}, {dict(WINDOWS)[w]}")
    ax.set_xscale("log")
    ax.set(xlabel="fitted decay time constant [ms]",
           ylabel="fraction excluded by\nmulti-step decay validation",
           title="exclusion in the ECoG cohort",ylim=(0,1))
    ax.legend(frameon=False,fontsize=8)
    ax.text(-.15,1.04,f"({next(letters)})",transform=ax.transAxes,
            fontweight="bold",fontsize=12)


def verdict_panels(fig, spec, letters, inputs, *, legend_y=.905):
    """Every stable 0-100 Hz candidate of the ten training fits at the example
    site, on the eigenvalue plane, coloured by which criterion removed it."""
    row = spec.subgridspec(1, 4, wspace=0.12)
    unit = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
    handles = None
    col = 0
    for window, wname in WINDOWS:
        for route in ("standard", "hankel_d50"):
            path = inputs / "supplement" / "example_candidates" / f"{EXAMPLE_SITE}__{route}.npz"
            with np.load(path, allow_pickle=False) as a:
                z = {key: a[key] for key in a.files}
            eig = z["eig"]
            rate, freq = rate_and_frequency(eig, float(z["sf_hz"]))
            gap = decay_gap(z)
            # Both halves of each conjugate pair are drawn, so |f| is bounded here.
            base = np.isfinite(rate) & (rate < 0) & (np.abs(freq) <= config.MAX_FREQUENCY_HZ)
            residual_ok = residual_pass(z["res1"])
            radial_ok = np.isfinite(gap) & (gap <= config.DECAY_VALIDATION_TOLERANCE)
            cell = base & (z["window"] == window)
            verdict = np.where(~residual_ok, "removed by residual criterion",
                               np.where(~radial_ok, "removed by multi-step decay validation", "validated"))
            ax = fig.add_subplot(row[0, col])
            ax.plot(unit.real, unit.imag, "--", color=".7", lw=.8)
            for name, colour in VERDICT_COLORS.items():
                q = cell & (verdict == name)
                ax.scatter(eig[q].real, eig[q].imag, s=11, color=colour,
                           edgecolors="black" if name != "removed by residual criterion" else "none",
                           linewidths=.25, alpha=.85, zorder=3 if name == "validated" else 2,
                           label=name)
            # Counts over f >= 0 only, as in main-text Fig. 3C (a conjugate
            # pair is one estimate); the plane shows both halves.
            upper = cell & (freq >= 0)
            n = {k: int((upper & (verdict == k)).sum()) for k in VERDICT_COLORS}
            ax.set(xlim=ZOOM_X, ylim=ZOOM_Y, xlabel=r"Re($\lambda$)")
            ax.set_title(f"{ROUTE_NAMES[route]}, {wname}\n"
                         f"{n['validated']} validated of {int(upper.sum())}",
                         fontsize=fs.LABEL)
            if col == 0:
                ax.set_ylabel(r"Im($\lambda$)")
            else:
                ax.set_yticklabels([])
            ax.text(-.10 if col == 0 else -.02, 1.10, f"({next(letters)})",
                    transform=ax.transAxes, fontweight="bold", fontsize=fs.LETTER,
                    va="bottom", ha="right")
            handles = ax.get_legend_handles_labels()
            col += 1
    fig.legend(*handles, loc="lower center", bbox_to_anchor=(.5, legend_y),
               ncol=3, frameon=False, fontsize=fs.LEGEND, markerscale=1.6)


def selection_validation_figure(inputs, out):
    """Fig. S3: selection at one site (A-D), then the validation rule itself (E-G)."""
    with np.load(inputs / "supplement" / "radial_diagnostic_histograms.npz") as z:
        te = z["tau_edges"]
        td = {(r, w): z[f"tau_den_{r}_{w}"] for r in PRIMARY for w, _ in WINDOWS}
        tn = {(r, w): z[f"tau_num_{r}_{w}"] for r in PRIMARY for w, _ in WINDOWS}
    fig = plt.figure(figsize=(12.6, 8.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42,
                             left=.06, right=.99, bottom=.07, top=.88)
    letters = iter("ABCDEFG")
    verdict_panels(fig, outer[0], letters, inputs, legend_y=.935)
    lower = outer[1].subgridspec(1, 2, width_ratios=[2.25, 1], wspace=0.30)
    panel_radial_calibration(fig, lower[0], letters, inputs)
    panel_exclusion(fig, lower[1], letters, te, td, tn)
    save(fig, out, "figS3_ccep_selection")


def distribution_figure(inputs, out):
    """Fig. S4: post-minus-pre change in the frequency x decay-time distribution."""
    with np.load(inputs / "supplement" / "continuous_distribution_subjects.npz") as z:
        fe, te = z["frequency_edges"], z["tau_edges"]
        h = {(r, w): z[f"{r}_{w}"] for r in PRIMARY for w, _ in WINDOWS}
    ds={r:h[(r,"early")]-h[(r,"pre")] for r in ("standard","hankel_d50")}
    d={r:x.mean(0) for r,x in ds.items()}
    vmax=max(np.abs(x).max() for x in d.values())
    fig=plt.figure(figsize=(10.5,7.5))
    outer=fig.add_gridspec(2,1,height_ratios=[1.05,.9],hspace=.36)
    top=outer[0].subgridspec(1,2,wspace=.22)
    axes_top=[fig.add_subplot(top[0,i]) for i in range(2)]
    bottom=outer[1].subgridspec(1,2,wspace=.30)
    axes_bottom=[fig.add_subplot(bottom[0,i]) for i in range(2)]
    names={"standard":"DMD ($m=1$)","hankel_d50":"Hankel DMD ($m=50$)"}
    for ax,r in zip(axes_top,("standard","hankel_d50")):
        mesh=ax.pcolormesh(fe,te,d[r],cmap="RdBu_r",vmin=-vmax,vmax=vmax,
                          shading="auto")
        ax.set_yscale("log"); ax.set_title(names[r])
        # The two cuts of main-text Fig. 3: 8 Hz and a 100-ms time constant.
        ax.axvline(CUT_HZ,color="black",ls="--",lw=1.0)
        ax.axhline(CUT_TAU_MS,color="black",ls="--",lw=1.0)
        ax.text(CUT_HZ+1,1.3,f"{CUT_HZ:g} Hz",fontsize=fs.SMALL,va="bottom")
        ax.text(99,CUT_TAU_MS*1.08,f"{CUT_TAU_MS:g} ms",fontsize=fs.SMALL,
                ha="right",va="bottom")
        if r=="hankel_d50":
            ax.add_patch(plt.Rectangle((5,20),35,230,fill=False,ec="black",
                                       lw=.9,ls=":"))
            ax.text(42,60,"gain:\n5-40 Hz\n20-250 ms",fontsize=fs.SMALL+.5,
                    va="center",ha="left")
        ax.set_xlabel("frequency [Hz]")
        if ax is axes_top[0]: ax.set_ylabel("decay time constant [ms]")
    fig.colorbar(mesh,ax=axes_top,label="signed change\nestimates/site/fit/bin",
                 fraction=.025,pad=.02)
    fc,tc=(fe[:-1]+fe[1:])/2,np.sqrt(te[:-1]*te[1:])
    rng=np.random.default_rng(20260920)
    colors={"standard":".35","hankel_d50":cf.FAST_COLOR}
    for r in ("standard","hankel_d50"):
        for ax,curves,x in ((axes_bottom[0],ds[r].sum(1),fc),
                            (axes_bottom[1],ds[r].sum(2),tc)):
            mean=curves.mean(0)
            boot=curves[rng.integers(len(curves),size=(2000,len(curves)))].mean(1)
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            ax.plot(x,mean,label=names[r],color=colors[r],lw=1.7)
            ax.fill_between(x,lo,hi,color=colors[r],alpha=.14,linewidth=0)
    axes_bottom[0].set(xlabel="frequency [Hz]",ylabel="marginal change")
    axes_bottom[1].set_xscale("log")
    axes_bottom[1].set(xlabel="decay time constant [ms]",ylabel="marginal change")
    for ax in axes_bottom: ax.axhline(0,color=".65",lw=.7)
    axes_bottom[0].axvline(CUT_HZ,color="black",ls="--",lw=1.0)
    axes_bottom[1].axvline(CUT_TAU_MS,color="black",ls="--",lw=1.0)
    axes_bottom[0].text(CUT_HZ+1,.97,f"{CUT_HZ:g} Hz",transform=axes_bottom[0].get_xaxis_transform(),
                        fontsize=fs.SMALL,va="top")
    axes_bottom[1].text(CUT_TAU_MS*1.1,.97,f"{CUT_TAU_MS:g} ms",transform=axes_bottom[1].get_xaxis_transform(),
                        fontsize=fs.SMALL,va="top")
    axes_bottom[0].legend(frameon=False,fontsize=8)
    for letter,ax in zip("ABCD",axes_top+axes_bottom):
        ax.text(-.14,1.04,f"({letter})",transform=ax.transAxes,
                fontweight="bold",fontsize=12)
    save(fig, out, "figS4_ccep_distribution")


def robustness_figure(inputs, out):
    """Fig. S5: rank, residual-criterion and decay-validation sensitivity."""
    summary = pd.read_csv(inputs / "supplement" / "robustness_effects.csv")
    rs = pd.read_csv(inputs / "supplement" / "residual_sensitivity.csv")
    radial = pd.read_csv(inputs / "supplement" / "radial_sensitivity.csv")
    n_subjects = int(rs.n_subjects.iloc[0])

    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.0))
    order = ("standard", "hankel_d50")

    # (A) Primary fast-mode contrast at the matched rank and at common rank 30.
    ax = axs[0]
    styles = {"rank50_matched": ("o", "black", "rank 50 (44 participants)"),
              "rank30_primary": ("D", "white", "rank 30, same 44 participants"),
              "rank30_all": ("s", "white", "rank 30, all 50 participants")}
    for off, subset in zip((-.18, 0, .18), ("rank50_matched", "rank30_primary", "rank30_all")):
        marker, face, name = styles[subset]
        z = summary[(summary.metric == "n_fast") &
                    (summary.subset == subset)].set_index("contrast")
        mean = np.array([z.loc[x, "mean"] for x in order])
        lo = np.array([z.loc[x, "ci_low"] for x in order])
        hi = np.array([z.loc[x, "ci_high"] for x in order])
        ax.errorbar(mean, np.arange(2) + off, xerr=[mean-lo, hi-mean],
                    fmt=marker, color="black", mfc=face, capsize=2, label=name)
    ax.axvline(0, color=".65", lw=.8)
    ax.set_yticks(range(2), [ROUTE_NAMES[r] for r in order])
    ax.set_ylim(-.6, 1.6)
    ax.set(xlabel="post-stimulus $-$ pre-stimulus\nfast-decaying estimates/site/fit",
           title="rank sensitivity")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # (B) Residual threshold and support, for the primary Hankel result.
    g = rs.pivot(index="support", columns="eps", values="hankel_mean")
    up = rs.pivot(index="support", columns="eps", values="hankel_up")
    g, up = g.loc[list(RES_SUPPORT), list(RES_EPS)], up.loc[list(RES_SUPPORT), list(RES_EPS)]
    vmax = g.to_numpy(float).max() * 1.1
    axs[1].imshow(g.to_numpy(float), cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
    for i, need in enumerate(RES_SUPPORT):
        for j, eps in enumerate(RES_EPS):
            v = g.loc[need, eps]
            axs[1].text(j, i, f"{v:.2f}\n{int(up.loc[need, eps])}/{n_subjects} up",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if v > .55*vmax else "black",
                        fontweight="bold" if (need == config.REQUIRED_HELD_OUT_TRIALS
                                                 and eps == config.RESIDUAL_THRESHOLD)
                        else "normal")
    axs[1].set_xticks(range(len(RES_EPS)), [f"{e:g}" for e in RES_EPS])
    axs[1].set_yticks(range(len(RES_SUPPORT)), [f"{n}/9" for n in RES_SUPPORT])
    axs[1].set(xlabel="one-step residual threshold",
               ylabel="held-out trials required",
               title="residual-criterion sensitivity")

    # (C) Removing or varying the radial check.
    none = radial[radial.lag_ms == 0].set_index("route")
    for route in order:
        axs[2].scatter([0], [none.loc[route, "mean"]], marker="x", s=55,
                       color=ROUTE_COLORS[route], zorder=4)
    for tol, style in ((.05, "-"), (.10, "--")):
        vals = {route: [] for route in order}
        for lag in (2, 5, 10):
            for route in order:
                row = radial[(radial.route == route) & (radial.lag_ms == lag) &
                             np.isclose(radial.tolerance, tol)].iloc[0]
                vals[route].append(row["mean"])
        for route in order:
            axs[2].plot((2, 5, 10), vals[route], style, marker="o",
                        color=ROUTE_COLORS[route],
                        label=f"{ROUTE_NAMES[route]}, tolerance {tol:.2f}")
    axs[2].axhline(0, color=".65", lw=.8)
    axs[2].set(xlabel="validation horizon [ms]",
               ylabel="change in fast-decaying estimates",
               xticks=(0, 2, 5, 10), xticklabels=("none", "2", "5", "10"),
               title="decay-validation settings")
    axs[2].legend(frameon=False, fontsize=7.5)

    for letter, ax in zip("ABC", axs):
        ax.text(-.16, 1.04, f"({letter})", transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    fig.tight_layout(w_pad=2)
    save(fig, out, "figS5_ccep_robustness")


def main(figures=FIGURES, inputs=INPUTS, out=OUT):
    if "validation" in figures:
        selection_validation_figure(inputs, out)
    if "distribution" in figures:
        distribution_figure(inputs, out)
    if "robustness" in figures:
        robustness_figure(inputs, out)
    print("wrote", ", ".join(figures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES),
                        help="subset of figures to draw (default: all)")
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.figures, args.inputs, args.out)
