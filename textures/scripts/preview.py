"""Lit preview sheets for material candidates.

A material is judged by how it takes light, which a sheet of albedos cannot
show. `sheet` draws one row per candidate: its four maps, then the material
tiled 2 x 2 on a flat panel under a low light, then wrapped round a cylinder.
Look for seams and repetition in the panel, and for whether the relief reads
as shape rather than as a printed picture.

    pixi run python -m textures.scripts.preview out.jpg <material dir>...

A material dir holds `albedo.png`, `normal.png` (OpenGL), `roughness.png` and
`height.png`.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from textures.scripts import tex_core as tc

CELL, GAP, HEAD, ROW_HEAD = 300, 12, 70, 36
COLUMNS = [("Colour", "albedo"), ("Normal", "surface direction"), ("Roughness", "white = matte"),
           ("Height", "white = raised"), ("Lit, flat", "tiled 2x2, low light from the left"),
           ("Lit, on a cylinder", "wrapped twice round")]


def _load(d: Path, name: str, n: int, grey: bool = False) -> np.ndarray:
    im = Image.open(d / name).convert("L" if grey else "RGB").resize((n, n), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def _shade(albedo, normal, rough, occlusion, light):
    """Lambert plus a GGX highlight, viewed straight on. `normal` is in view
    space (x right, y up, z toward the viewer)."""
    light = np.asarray(light, np.float32)
    light /= np.linalg.norm(light)
    half = light + (0, 0, 1)
    half /= np.linalg.norm(half)
    ndl = np.clip(normal @ light, 0.0, 1.0)
    ndh = np.clip(normal @ half, 0.0, 1.0)
    a2 = np.clip(rough, 0.08, 1.0) ** 4
    spec = 0.04 * a2 / (np.pi * (ndh * ndh * (a2 - 1.0) + 1.0) ** 2) * ndl
    colour = albedo ** 2.2 * (2.1 * ndl + 0.2 * occlusion)[..., None] + np.minimum(spec, 2.0)[..., None]
    return np.clip(colour, 0.0, 1.0) ** (1.0 / 2.2)


def _material(d: Path, n: int):
    normal = _load(d, "normal.png", n) * 2.0 - 1.0
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    height = _load(d, "height.png", n, grey=True)
    return _load(d, "albedo.png", n), normal, _load(d, "roughness.png", n, grey=True), 1.0 - 0.7 * tc.cavity(height, n / 60)


def _flat(albedo, normal, rough, occlusion):
    tile = lambda a: np.tile(a, (2, 2) + (1,) * (a.ndim - 2))  # noqa: E731
    return _shade(tile(albedo), tile(normal), tile(rough), tile(occlusion), (-0.8, 0.6, 0.45))


def _cylinder(albedo, normal, rough, occlusion, size: int, wraps: float = 2.0):
    n = albedo.shape[0]
    radius = 0.42 * size
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (cols - size / 2) / radius
    theta = np.arcsin(np.clip(x, -1.0, 1.0))
    u = (theta / (2 * np.pi) * wraps * n).astype(int) % n
    v = (rows / (2 * np.pi * radius / wraps) * n).astype(int) % n
    zero = np.zeros_like(theta)
    out = np.stack([np.sin(theta), zero, np.cos(theta)], -1)
    along = np.stack([np.cos(theta), zero, -np.sin(theta)], -1)
    t = normal[v, u]
    world = along * t[..., 0:1] + np.array([0, 1, 0], np.float32) * t[..., 1:2] + out * t[..., 2:3]
    image = _shade(albedo[v, u], world, rough[v, u], occlusion[v, u], (-0.7, 0.5, 0.6))
    image[np.abs(x) >= 1.0] = 0.11
    return image


def sheet(dirs: list[Path], out: Path, labels: list[str] | None = None) -> Path:
    """Write the labelled preview sheet for `dirs` (one row each) to `out`."""
    labels = labels or [d.name for d in dirs]
    font = lambda size: ImageFont.load_default(size=size)  # noqa: E731
    width = len(COLUMNS) * (CELL + GAP) + GAP
    image = Image.new("RGB", (width, HEAD + len(dirs) * (ROW_HEAD + CELL + GAP)), (28, 28, 30))
    draw = ImageDraw.Draw(image)
    for i, (title, note) in enumerate(COLUMNS):
        x = GAP + i * (CELL + GAP)
        draw.text((x, 12), title, font=font(24), fill=(240, 240, 240))
        draw.text((x, 44), note, font=font(15), fill=(160, 160, 165))
    for r, (d, label) in enumerate(zip(dirs, labels)):
        y = HEAD + r * (ROW_HEAD + CELL + GAP)
        draw.text((GAP, y + 4), label, font=font(22), fill=(255, 210, 120))
        material = _material(d, 512)
        cells = [_load(d, f, CELL) for f in ("albedo.png", "normal.png", "roughness.png", "height.png")]
        cells += [_flat(*material), _cylinder(*material, 512)]
        for i, cell in enumerate(cells):
            tile = Image.fromarray(tc.to8(cell)).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            image.paste(tile, (GAP + i * (CELL + GAP), y + ROW_HEAD))
    image.save(out, quality=90)
    return out


if __name__ == "__main__":
    print(sheet([Path(p) for p in sys.argv[2:]], Path(sys.argv[1])))
