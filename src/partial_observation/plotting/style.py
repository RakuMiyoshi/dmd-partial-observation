"""Shared figure style: a Helvetica-family font and one size hierarchy.

Every paper and supplementary figure calls :func:`use_style` so that they share
one typeface and one set of font sizes. The sizes are ordered by how much
information a piece of text carries, from the panel letter down to tertiary
annotations, so the same weight always reads at the same size across figures.

Helvetica itself is used when the system has it; otherwise the metric-compatible
clones (Nimbus Sans is Helvetica, Liberation Sans is Arial) stand in, with a
universal fallback last. Maths is set in a sans-serif face so symbols such as
$\\lambda$ match the surrounding text.
"""
from __future__ import annotations

import matplotlib as mpl

_SANS = ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans", "DejaVu Sans"]

# One scale, largest = most structural. Import these instead of hard-coding
# point sizes, so a change here restyles every figure.
LETTER = 12    # panel letters (a), (b), ...
TITLE = 11     # panel titles and axis-arrow labels: a panel's headline result
LABEL = 10     # axis labels
LEGEND = 9     # legend entries
TICK = 8.5     # tick labels
ANNOT = 8      # secondary in-panel annotations
SMALL = 7      # tertiary annotations where space is tight


def _families_with_bold() -> list[str]:
    """Drop families whose bold face Matplotlib cannot see.

    macOS ships Helvetica as a .ttc collection, of which Matplotlib registers
    only the regular face; panel letters and headers would then fall back to
    regular weight without a warning. Skipping such families keeps the bold
    hierarchy and lands on Arial, which the manuscript figures already use.
    """
    from matplotlib import font_manager as fm
    weights = {}
    for entry in fm.fontManager.ttflist:
        weights.setdefault(entry.name, set()).add(entry.weight)
    kept = [name for name in _SANS
            if name not in weights or {700, "bold"} & weights[name]]
    return kept or _SANS


def use_style() -> None:
    """Apply the shared font family and size hierarchy to Matplotlib."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _families_with_bold(),
        "mathtext.fontset": "stixsans",
        "font.size": LABEL,
        "axes.titlesize": TITLE,
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "figure.titlesize": LETTER,
        "axes.titleweight": "normal",
        # Embed text as TrueType so the typeface is identical in the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
