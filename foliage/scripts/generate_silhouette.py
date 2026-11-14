"""
Grass silhouette generator — RASTER backend.

Renders the shared parametric blade model (see `blade_core.py`) to a 2D binary
silhouette PNG, used downstream as both ControlNet conditioning and the alpha
channel of the final billboard composite. This is the mid-distance foliage path.

The blade shape itself (BladeParams, ClusterParams, sampling, spine integration,
width taper) lives in `blade_core` so the 3D mesh backend produces the *same*
plant. This file only owns the 2D-raster-specific concerns: filling the outline
polygon, edge noise, seed/flower heads, and grid previews.

Wind is a runtime vertex-shader concern and stays out of the baked texture.

Usage:
    pixi run silhouettes            # regenerate every profile

Outputs land in assetgen/foliage/silhouettes/.
"""

import math
import random
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from blade_core import (
    CANVAS_W,
    CANVAS_H,
    GROUND_Y,
    BladeParams,
    ClusterParams,
    blade_width,
    integrate_blade,
    load_profiles,
    sample_blade,
)


def rasterize_blade(
    draw: ImageDraw.ImageDraw,
    positions: list[tuple[float, float]],
    base_thickness: float,
    tip_thickness: float,
    spread: float = 0.0,
    edge_noise: float = 0.0,
    edge_noise_smoothness: float = 0.8,
    rng: random.Random | None = None,
    head_radius: float = 0.0,
    head_elongation: float = 1.0,
) -> None:
    n = len(positions) - 1

    # Pre-compute smoothly-varying jitter sequences for left/right edges via an
    # AR(1) process: each sample is a weighted mix of the previous sample and a
    # fresh Gaussian draw. Prevents vertex-scale zigzag; gives a few gentle
    # bumps per edge instead. `edge_noise_smoothness` in [0, 1) controls the
    # correlation length: higher = smoother/fewer features.
    if edge_noise > 0 and rng is not None:
        phi = max(0.0, min(0.98, edge_noise_smoothness))
        gain = math.sqrt(1 - phi * phi)  # variance-preserving fresh-noise scale
        l_jitters = [0.0]
        r_jitters = [0.0]
        for _ in range(n):
            l_jitters.append(phi * l_jitters[-1] + gain * rng.gauss(0, edge_noise))
            r_jitters.append(phi * r_jitters[-1] + gain * rng.gauss(0, edge_noise))
    else:
        l_jitters = r_jitters = [0.0] * (n + 1)

    left, right = [], []
    for i, (px, py) in enumerate(positions):
        t = i / n
        # Shared width model (linear taper + optional bell bulge). bell(t) hits 0
        # at both endpoints, so the two edges meet at root and tip.
        w = blade_width(t, base_thickness, tip_thickness, spread)
        bell = 4.0 * t * (1.0 - t)
        # Local segment direction, averaged from neighbors for smoothness.
        if 0 < i < n:
            prev_x, prev_y = positions[i - 1]
            next_x, next_y = positions[i + 1]
            dx, dy = next_x - prev_x, next_y - prev_y
        elif i == 0:
            next_x, next_y = positions[i + 1]
            dx, dy = next_x - px, next_y - py
        else:
            prev_x, prev_y = positions[i - 1]
            dx, dy = px - prev_x, py - prev_y
        norm = math.hypot(dx, dy) or 1
        perp_x, perp_y = -dy / norm, dx / norm
        # Bell ensures the two edges meet at the endpoints regardless of jitter.
        l_jitter = l_jitters[i] * bell
        r_jitter = r_jitters[i] * bell
        left.append((px + perp_x * (w / 2 + l_jitter), py + perp_y * (w / 2 + l_jitter)))
        right.append((px - perp_x * (w / 2 + r_jitter), py - perp_y * (w / 2 + r_jitter)))
    draw.polygon(left + list(reversed(right)), fill=255)

    if head_radius > 0:
        _rasterize_head(draw, positions, head_radius, head_elongation)


def _rasterize_head(
    draw: ImageDraw.ImageDraw,
    positions: list[tuple[float, float]],
    radius: float,
    elongation: float,
    n_samples: int = 24,
) -> None:
    """Draw a head (seed or flower) aligned to the tip's tangent, extending beyond the tip."""
    tip_x, tip_y = positions[-1]
    prev_x, prev_y = positions[-2]
    dx, dy = tip_x - prev_x, tip_y - prev_y
    norm = math.hypot(dx, dy) or 1
    # Along-blade unit vector u, perpendicular v.
    ux, uy = dx / norm, dy / norm
    vx, vy = -uy, ux

    a = radius * elongation  # semi-axis along blade
    b = radius                # semi-axis perpendicular
    # Center offset so the head grows out from the tip rather than sitting on top of it.
    cx = tip_x + ux * a
    cy = tip_y + uy * a

    pts = []
    for i in range(n_samples):
        theta = i / n_samples * 2 * math.pi
        la, lb = math.cos(theta) * a, math.sin(theta) * b
        px = cx + ux * la + vx * lb
        py = cy + uy * la + vy * lb
        pts.append((px, py))
    draw.polygon(pts, fill=255)


def generate_silhouette(
    seed: int,
    cluster: ClusterParams,
    canvas: tuple[int, int] = (CANVAS_W, CANVAS_H),
    ground_y: float = GROUND_Y,
) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("L", canvas, 0)
    draw = ImageDraw.Draw(img)
    n = max(1, cluster.n_blades + rng.randint(-cluster.n_blades_var, cluster.n_blades_var))
    blades = [sample_blade(rng, cluster) for _ in range(n)]
    blades.sort(key=lambda b: b.length)
    for blade in blades:
        positions = integrate_blade(blade, rng, ground_y)
        rasterize_blade(
            draw,
            positions,
            blade.base_thickness,
            blade.tip_thickness,
            spread=blade.spread,
            edge_noise=blade.edge_noise,
            rng=rng,
            head_radius=blade.head_radius,
            head_elongation=blade.head_elongation,
        )
    return img


def generate_grid(
    seeds: Sequence[int],
    out_path: Path,
    cluster: ClusterParams,
    cols: int = 6,
    label: str | None = None,
) -> None:
    rows = (len(seeds) + cols - 1) // cols
    thumb_w, thumb_h = 192, 256
    grid = Image.new("L", (cols * thumb_w, rows * thumb_h), 0)
    for i, seed in enumerate(seeds):
        silhouette = generate_silhouette(seed, cluster).resize(
            (thumb_w, thumb_h), Image.Resampling.LANCZOS
        )
        r, c = divmod(i, cols)
        grid.paste(silhouette, (c * thumb_w, r * thumb_h))
    grid.save(out_path)
    print(f"wrote {out_path}" + (f"  [{label}]" if label else ""))


def main():
    root = Path(__file__).resolve().parent.parent
    profiles = load_profiles(root / "profiles")
    out_dir = root / "silhouettes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cluster in profiles.items():
        generate_grid(list(range(24)), out_dir / f"grid_{name}.png", cluster, label=name)
        for seed in [0, 3, 7]:
            path = out_dir / f"{name}_seed{seed:03d}.png"
            generate_silhouette(seed, cluster).save(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
