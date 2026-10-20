"""Shared paths and helpers for the paper figure scripts.

Every script reads only the compact data under ``results/`` and writes to ``figures/``.
Spreads are sample standard deviations (ddof=1) throughout, as in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"

SD_DDOF = 1


def sd(values) -> float:
    """Sample standard deviation of a sequence (ddof=SD_DDOF)."""
    return float(np.std(np.asarray(values, dtype=float), ddof=SD_DDOF))


def sd_from_population(sd_population: float, n: int) -> float:
    """Convert a stored population SD (ddof=0) over n runs to the SD convention in use."""
    return float(sd_population) * float(np.sqrt(n / (n - SD_DDOF)))


def reset_style() -> None:
    """Start each figure from the default style, as if it were drawn in a fresh process."""
    matplotlib.rc_file_defaults()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def save(fig, stem: str, dpi: int = 240, formats=("png", "pdf")) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
