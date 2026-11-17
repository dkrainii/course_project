"""One look for every figure of the ML experiments (static PNGs that go into the thesis).

Colours are assigned by the job they do: a fixed colour per model, one blue ramp for magnitude,
blue ↔ red around a neutral grey for signed values. Nutri-Score grades keep the colours people
know from the label (E darkened so that D and E stay apart); the grade letter is always printed
next to the colour, so colour never carries the grade alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#6250d6"
RED = "#e34948"
NEUTRAL = "#b5b3ab"  # context series (baselines)

GRADE_COLORS = {"a": "#038141", "b": "#85BB2F", "c": "#FECB02", "d": "#EE8100", "e": "#D32A1C"}
GRADE_TEXT_ON_FILL = {"a": "#ffffff", "b": INK, "c": INK, "d": INK, "e": "#ffffff"}

MODEL_COLORS = {
    "Dummy": NEUTRAL,
    "Formula": MUTED,
    "LogReg": BLUE,
    "RandomForest": ORANGE,
    "LightGBM": AQUA,
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "kolos_blue", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "kolos_diverging", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b92f2f"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.dpi": 100,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
    })


def thousands(value: float) -> str:
    """12 345 → '12 345' (thin, locale-neutral grouping for labels)."""
    return f"{value:,.0f}".replace(",", " ")


def horizontal_grid(ax) -> None:
    """Bars that grow to the right: grid follows the value axis."""
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
