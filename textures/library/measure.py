"""Measurements of a material's maps: its mean colour, how many courses it holds, how even it is.

Shared by the library (`make.py`) and the looks step (`concepts/scripts/looks.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722])


def to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def mean_colour(picture: Path) -> np.ndarray:
    """A picture's mean colour, linear RGB."""
    return to_linear(np.asarray(Image.open(picture).convert("RGB"), np.float32) / 255.0).reshape(-1, 3).mean(0)


def courses(height: Path, rows: bool) -> int:
    """How many courses a tile holds: the strongest repeat of its height map's rows (or columns),
    between 2 and 64 to a tile. The tile repeats, so a whole number of them fits it exactly.
    klein draws about as many courses as it likes whatever it is asked, so the tile's size in
    the game comes from this count and the size one course is."""
    h = np.asarray(Image.open(height).convert("L"), np.float32)
    profile = h.mean(1 if rows else 0)
    power = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    return 2 + int(np.argmax(power[2:65]))


def unevenness(albedo: Path) -> float:
    """How much of a tile's light and dark is at a large scale: the spread of its 8 x 8 block
    means over the spread of its pixels. A bare wall or roof is about 0.1; a tile with a facade
    (windows, a timber band) or a seam drawn across it, 0.3 to 0.6."""
    a = np.asarray(Image.open(albedo).convert("L"), np.float32)
    n = a.shape[0] // 8 * 8
    blocks = a[:n, :n].reshape(8, n // 8, 8, n // 8).mean((1, 3))
    return float(blocks.std() / max(a.std(), 1e-6))
