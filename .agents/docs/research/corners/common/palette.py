"""Brand palette and matplotlib style for the corner figures (AGENTS.md rule 14).

Series lines use the brand accents in order of use, SKIPPING amber and teal: both fail the
3:1 contrast floor against a white surface for thin line marks (amber 1.97:1, teal 1.56:1,
measured with the dataviz palette validator), so they are reserved for area fills and bands
that carry a direct label. The 3-slot categorical order blue, purple, red passes every check
(worst adjacent pair blue/purple at deutan dE 8.2 with direct labels as secondary encoding,
normal-vision dE 18.0, all three above 3:1 contrast).

Colors are assigned by SERIES IDENTITY, never by position in a particular figure: the same
ablation line wears the same color on every corner chat's rendering.

Importing this module needs matplotlib (the repo's viz extra):

    uv run --extra viz python ...
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # figure scripts render to files; never require a display
from matplotlib import pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

INK = "#0a0a0a"
MUTED = "#8a8a8a"
GRID = "#ececec"
BLUE = "#0070f3"
PURPLE = "#7928ca"
AMBER = "#f5a623"
RED = "#ee0000"
TEAL = "#50e3c2"

# The fixed series-identity assignment for the shared ablation chart (and any figure that
# shows the same levers). Never reassign these by figure-local position.
SERIES_COLORS: dict[str, str] = {
    "distill-only": BLUE,
    "+routing": PURPLE,
    "+compaction": RED,
}

# Fill-only accents (direct label required, never a series line): the noise-floor band is a
# neutral gray so it cannot be misread as a series.
NOISE_BAND_COLOR = GRID
NOISE_BAND_ALPHA = 0.9


def apply_style() -> None:
    """Set the Vercel/Notion-like rcParams every corner figure shares."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def footnote(fig: Figure, text: str, *, y: float = -0.02) -> None:
    """The provenance line under a corner figure (labeling rules in common/README.md).

    Wrapped to the figure width: a long single-line provenance string inflates the tight
    bounding box and shrinks the axes to a fraction of the canvas (measured on the cost
    chat's first render). `y` drops further (e.g. -0.1) when the figure has an x-axis label
    that would otherwise collide (measured on the quality chat's lever chart).
    """
    width = int(fig.get_figwidth() * 16)
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    fig.text(0.005, y, wrapped, fontsize=7.5, color=MUTED, ha="left", va="top")


def label_point(ax: Axes, x: float, y: float, text: str) -> None:
    """Direct label beside a mark, in ink: identity never rides on color alone, and text
    wears text tokens rather than the series color."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )


def save_fig(fig: Figure, path: Path) -> None:
    """Write a figure and close it (figure scripts render many; leaks add up)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
