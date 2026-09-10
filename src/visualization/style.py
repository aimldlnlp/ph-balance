"""Shared minimal styling for PH-Balance figures."""

from pathlib import Path

from matplotlib import font_manager, rcParams

INK = "#172033"
BLUE = "#2563EB"
TEAL = "#0F766E"
AMBER = "#D97706"
VIOLET = "#7C3AED"
GRID = "#E5E7EB"
PAPER = "#FFFFFF"


def _register_project_fonts() -> None:
    """Register bundled fonts so figures are reproducible across machines."""

    font_dir = Path(__file__).resolve().parents[2] / "assets" / "fonts"
    for font_path in font_dir.glob("*.otf"):
        font_manager.fontManager.addfont(str(font_path))


_register_project_fonts()


def configure_style() -> str:
    """Configure presentation-ready Matplotlib defaults and return the font."""

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (
            candidate
            for candidate in (
                "Latin Modern Sans",
                "Latin Modern Roman",
                "Inter",
                "IBM Plex Sans",
                "Helvetica Neue",
                "Manrope",
            )
            if candidate in installed_fonts
        ),
        "DejaVu Sans",
    )
    rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": PAPER,
        }
    )
    return font_family
