"""Tileable procedural texture building blocks (numpy + scipy).

Everything here is periodic over the image, so every map built from these
functions tiles without a seam. Shared by `bake_ground.py` (terrain layers),
and by the bark / leaf / water bakers.

Conventions: arrays are `(n, n)` float32, `v` (rows) first. Heights and masks
are in 0..1 unless a function says otherwise.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / max(hi - lo, 1e-9)).astype(np.float32)


def _freq(n: int):
    f = np.fft.fftfreq(n) * n  # cycles per image
    return np.meshgrid(f, f, indexing="xy")  # fx varies along columns (u)


def spectral(n: int, seed, beta: float = 2.0, lo: float = 1.0, hi: float = 1e9,
             stretch: tuple[float, float] = (1.0, 1.0), angle: float = 0.0) -> np.ndarray:
    """Filtered white noise: amplitude ~ 1/f^(beta/2) between `lo` and `hi`
    cycles per image. `stretch=(su, sv)` makes features that many times longer
    along u and v, so `(6, 1)` gives horizontal streaks. `angle` rotates the
    stretch. Returns 0..1."""
    fx, fy = _freq(n)
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        fx, fy = c * fx + s * fy, -s * fx + c * fy
    # A feature long along u has little high frequency along u: weight that axis up.
    f = np.sqrt((fx * stretch[0]) ** 2 + (fy * stretch[1]) ** 2)
    amp = np.where(f > 0, 1.0 / np.maximum(f, 1e-6) ** (beta / 2.0), 0.0)
    amp *= (f >= lo) & (f <= hi)
    white = np.fft.fft2(rng(seed).standard_normal((n, n)))
    out = np.fft.ifft2(white * amp).real
    return norm01(out)


def cracks(n: int, seed, lo: float, hi: float, width: float, beta: float = 2.2,
           stretch: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Meandering crack lines: the mid-level contour of a smooth noise field.
    Unlike cell edges these wander, branch unevenly and never form a regular
    net. `width` is the half-width in noise units (about 0.01..0.05). 0..1."""
    f = spectral(n, seed, beta=beta, lo=lo, hi=hi, stretch=stretch)
    return (1.0 - smoothstep(0.0, width, np.abs(f - np.median(f)))).astype(np.float32)


def blur(a: np.ndarray, sigma) -> np.ndarray:
    return ndimage.gaussian_filter(a, sigma, mode="wrap").astype(np.float32)


def warp(a: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Resample `a` at (u + du, v + dv), offsets in pixels, wrapping."""
    n = a.shape[0]
    v, u = np.mgrid[0:n, 0:n].astype(np.float32)
    return ndimage.map_coordinates(a, [(v + dv) % n, (u + du) % n], order=1, mode="grid-wrap").astype(np.float32)


def worley(n: int, cells: int, seed, jitter: float = 1.0):
    """Periodic cellular noise with roughly `cells` x `cells` sites.

    Returns `(f1, f2, cell_id, site_value)`: distance to the nearest and second
    nearest site in units of the cell size, the index of the nearest site, and a
    per-site random 0..1 looked up per pixel."""
    r = rng(seed)
    gy, gx = np.mgrid[0:cells, 0:cells]
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64) + 0.5
    pts += (r.random(pts.shape) - 0.5) * jitter
    value = r.random(len(pts)).astype(np.float32)
    tiled = np.concatenate([pts + (ox * cells, oy * cells) for oy in (-1, 0, 1) for ox in (-1, 0, 1)])
    tree = cKDTree(tiled)
    v, u = np.mgrid[0:n, 0:n]
    q = np.stack([u.ravel(), v.ravel()], 1) * (cells / n)
    d, idx = tree.query(q, k=2, workers=-1)
    ident = (idx[:, 0] % len(pts)).reshape(n, n)
    return (d[:, 0].reshape(n, n).astype(np.float32), d[:, 1].reshape(n, n).astype(np.float32),
            ident, value[ident])


def normal_map(height: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space normal (OpenGL, Y up = +v up the image) from a height map.
    `strength` is the height range in texels: a 0..1 height step of 1 between
    neighbouring texels tilts the normal by `atan(strength)`."""
    # Image rows run top to bottom, so +v (texture up) is -row.
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5 * strength
    dy = (np.roll(height, 1, 0) - np.roll(height, -1, 0)) * 0.5 * strength
    n = np.stack([-dx, -dy, np.ones_like(height)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def height_from_normal(normal: np.ndarray, broad: float = 8.0) -> np.ndarray:
    """Height (0..1) whose slopes match a tangent-space normal map (OpenGL, as
    `normal_map` writes; `normal` is `(n, n, 3)` in -1..1). Solved in frequency
    space, so the height tiles exactly when the normal map does.

    Shapes wider than about 1/`broad` of the image are faded out. An estimated
    normal map is only trustworthy for local relief: a slope error too small
    to see adds up across the image into hills and hollows that are not in the
    material, and anything that blends by height then follows those."""
    nz = np.maximum(normal[..., 2], 0.05)
    p, q = -normal[..., 0] / nz, normal[..., 1] / nz  # dh/du, dh/dv (v runs down the image)
    fx, fy = _freq(normal.shape[0])
    k2 = fx * fx + fy * fy
    h = np.fft.ifft2(-1j * (fx * np.fft.fft2(p) + fy * np.fft.fft2(q)) / (k2 + broad * broad)).real
    lo, hi = np.percentile(h, [0.5, 99.5])
    return np.clip((h - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


def ramp(t: np.ndarray, stops) -> np.ndarray:
    """Piecewise-linear colour ramp. `stops` is `[(t, (r, g, b)), ...]` in sRGB."""
    ts = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], -1).astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cavity(height: np.ndarray, radius: float) -> np.ndarray:
    """Cheap ambient occlusion: how far a texel sits below its neighbourhood,
    0 (open) .. 1 (deep in a crevice)."""
    return np.clip((blur(height, radius) - height) * 4.0, 0.0, 1.0)


def load_material(d, n: int):
    """A generated material dir (`pixi run material`) at `n` pixels: `(albedo,
    normal, roughness, height)` as floats, the normal a unit vector (OpenGL).
    Resampling wraps, so it still tiles."""
    from pathlib import Path

    from PIL import Image

    def load(f, mode):
        im = Image.open(Path(d) / f).convert(mode)
        if im.size != (n, n):
            # Tile 3x3, resize, keep the centre: the filter then sees wrapped neighbours.
            w, h = im.size
            big = Image.new(mode, (w * 3, h * 3))
            for j in range(3):
                for i in range(3):
                    big.paste(im, (i * w, j * h))
            im = big.resize((n * 3, n * 3), Image.LANCZOS).crop((n, n, 2 * n, 2 * n))
        return np.asarray(im, np.float32) / 255.0

    normal = load("normal.png", "RGB") * 2.0 - 1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    return load("albedo.png", "RGB"), normal, load("roughness.png", "L"), load("height.png", "L")


def recolour(albedo: np.ndarray, colour, hue: float = 0.33, contrast: float = 1.0) -> np.ndarray:
    """Give a generated albedo the game's colour: keep its pattern (brightness,
    and `hue` of its own colour variation) and make `colour` (sRGB 0..1) its
    average. The models draw good structure but their colours drift (blue
    rock, orange soil), so every material entering the game is graded here."""
    grey = albedo.mean(-1, keepdims=True)
    tone = grey / max(float(grey.mean()), 1e-6)
    tint = albedo / np.maximum(grey, 1e-6)
    return np.asarray(colour, np.float32) * tone ** contrast * (1 + hue * (tint / tint.mean((0, 1)) - 1))


def to8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
