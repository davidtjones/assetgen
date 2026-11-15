"""Procedural ground materials for the terrain shader.

    pixi run ground            # bake every layer and the packed arrays
    pixi run ground -- --size 512 --only rock

Each layer comes from a generated material (`pixi run material <name>`) named
in `GENERATED`, which also gives the colour it is graded to: the model
supplies the structure and the maps, the palette is set here
(`tex_core.recolour`). A layer whose material is not on disk falls back to its
procedural function below, which returns `(albedo sRGB, height, roughness,
normal strength)`. Per layer it
writes `out/<name>/{albedo,normal,roughness,height}.png`, and for the game it
packs every layer into two stacked-row array textures under
`../game3d/assets/textures/terrain/`:

- `albedo_height.png` - RGB albedo (sRGB), A height (for height blending)
- `normal_rough.png`  - RG tangent normal xy (OpenGL), B roughness, A cavity

The layer order here is the array index in `terrain_blend.wgsl`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common.paths import game3d_assets
from textures.scripts import tex_core as tc

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"


def grass(n, seed):
    """Ground under and between grass tufts: matted blades over dark soil.
    Greens sit between the blade root and tip colours of the foliage profiles so
    the ground carries the meadow's colour past where real blades stop."""
    # Fibres: streak noise at several angles, max-combined so strands cross.
    fibres = np.zeros((n, n), np.float32)
    for i, ang in enumerate(np.linspace(0.0, np.pi, 7, endpoint=False)):
        s = tc.spectral(n, (seed, 10 + i), beta=0.6, lo=n / 26, hi=n / 3.0, stretch=(7.0, 1.0), angle=ang + 0.2)
        fibres = np.maximum(fibres, s)
    fibres = tc.norm01(fibres)
    fibres = tc.smoothstep(0.45, 0.85, fibres)
    clump = tc.spectral(n, (seed, 1), beta=2.4, lo=2, hi=40)
    patch = tc.spectral(n, (seed, 2), beta=2.0, lo=1, hi=8)
    height = tc.norm01(0.6 * fibres + 0.4 * clump)
    # Soil shows where the thatch is thin.
    soil = tc.smoothstep(0.50, 0.25, height)
    t = np.clip(0.15 + 0.65 * fibres + 0.25 * (clump - 0.5), 0.0, 1.0)
    green = tc.ramp(t, [(0.0, (0.15, 0.23, 0.08)), (0.5, (0.30, 0.44, 0.13)), (1.0, (0.50, 0.64, 0.24))])
    dry = tc.ramp(t, [(0.0, (0.20, 0.20, 0.08)), (0.5, (0.38, 0.40, 0.15)), (1.0, (0.60, 0.60, 0.28))])
    albedo = green + (dry - green) * (tc.smoothstep(0.55, 0.8, patch) * 0.6)[..., None]
    earth = np.array([0.16, 0.12, 0.07], np.float32)
    albedo = albedo + (earth - albedo) * (soil * 0.55)[..., None]
    rough = 0.82 + 0.14 * soil
    return albedo, height, rough, 5.0


def dirt(n, seed):
    """Bare soil: lumpy earth, fine grit, scattered small stones."""
    lumps = tc.spectral(n, (seed, 1), beta=2.2, lo=3, hi=120)
    clods = tc.spectral(n, (seed, 5), beta=1.2, lo=24, hi=160)
    grit = tc.spectral(n, (seed, 2), beta=0.2, lo=n / 9, hi=n / 2.1)
    # Stones: a warped cell field, a minority of cells filled, sizes varied.
    wu = (tc.spectral(n, (seed, 6), beta=1.5, lo=8, hi=60) - 0.5) * n * 0.03
    wv = (tc.spectral(n, (seed, 7), beta=1.5, lo=8, hi=60) - 0.5) * n * 0.03
    f1, _, _, val = (tc.warp(a.astype(np.float32), wu, wv) for a in tc.worley(n, 30, (seed, 3)))
    size = 0.10 + 0.32 * ((val * 7.0) % 1.0)
    pebble = np.where(val > 0.82, tc.smoothstep(size, size * 0.35, f1), 0.0).astype(np.float32)
    crack = tc.cracks(n, (seed, 4), lo=3, hi=40, width=0.012) * tc.smoothstep(0.45, 0.7, tc.spectral(n, (seed, 8), lo=1, hi=6))
    height = tc.norm01(0.45 * lumps + 0.22 * clods + 0.10 * grit + 0.40 * pebble - 0.22 * crack)
    t = np.clip(0.1 + 0.6 * lumps + 0.35 * clods + 0.35 * (grit - 0.5), 0.0, 1.0)
    albedo = tc.ramp(t, [(0.0, (0.14, 0.10, 0.06)), (0.5, (0.27, 0.20, 0.12)), (1.0, (0.43, 0.33, 0.21))])
    stone = tc.ramp((val * 13.0) % 1.0, [(0.0, (0.34, 0.32, 0.29)), (1.0, (0.56, 0.53, 0.47))])
    albedo = albedo + (stone - albedo) * tc.smoothstep(0.15, 0.4, pebble)[..., None]
    albedo *= (1.0 - 0.45 * crack - 0.3 * tc.cavity(height, n / 140))[..., None]
    rough = 0.93 - 0.15 * tc.smoothstep(0.15, 0.4, pebble)
    return albedo, height, rough, 7.0


def _generated(name, n):
    """Load a StableMaterials candidate (`pixi run material`) at `n` pixels.
    Returns `(albedo, height, roughness)`; resampling wraps so it still tiles."""
    d = OUT / "sm" / name

    def load(f, mode):
        im = Image.open(d / f).convert(mode)
        w = im.size[0]
        # Tile 3x3, resize, keep the centre: the filter then sees wrapped neighbours.
        big = Image.new(mode, (w * 3, w * 3))
        for j in range(3):
            for i in range(3):
                big.paste(im, (i * w, j * w))
        big = big.resize((n * 3, n * 3), Image.LANCZOS).crop((n, n, 2 * n, 2 * n))
        return np.asarray(big, np.float32) / 255.0

    return load("albedo.png", "RGB"), load("height.png", "L"), load("roughness.png", "L")


def _retone(albedo, stops, keep=0.25):
    """Recolour a generated albedo: map its luminance through a colour ramp and
    keep a fraction of its own hue variation. The model's structure is good but
    its colours drift (blue rock, orange soil), so the palette is set here."""
    lum = albedo @ np.array([0.3, 0.6, 0.1], np.float32)
    lo, hi = np.percentile(lum, [1, 99])
    t = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)
    toned = tc.ramp(t, stops)
    chroma = albedo - lum[..., None]
    return toned + keep * chroma


def rock(n, seed):
    """Cliff rock for steep faces, projected from the side: bedded strata running
    along u. Structure from a generated material (`cliff_strata`), retoned to
    warm grey, with fine grain added so it holds up after the 2x upsample."""
    albedo, height, rough = _generated("cliff_strata_s2", n)
    grain = tc.spectral(n, (seed, 6), beta=0.8, lo=n / 16, hi=n / 2.1)
    height = tc.norm01(tc.blur(height, 0.6) + 0.05 * grain)
    albedo = _retone(albedo, [(0.0, (0.17, 0.16, 0.15)), (0.4, (0.36, 0.34, 0.31)),
                              (0.75, (0.52, 0.49, 0.44)), (1.0, (0.68, 0.64, 0.57))])
    stain = tc.spectral(n, (seed, 8), beta=2.2, lo=1, hi=30, stretch=(1.0, 5.0))  # runs downhill
    rust = np.array([0.45, 0.33, 0.21], np.float32)
    albedo = albedo + (rust - albedo) * (tc.smoothstep(0.55, 0.85, stain) * 0.30)[..., None]
    albedo *= (0.92 + 0.16 * grain - 0.35 * tc.cavity(height, n / 90))[..., None]
    return albedo, height, 0.6 + 0.4 * rough, 14.0


def scree(n, seed):
    """Loose stones on steep-ish ground: packed angular gravel. Structure from a
    generated material (`scree_gravel`), retoned to match the cliff rock."""
    albedo, height, rough = _generated("scree_gravel_s1", n)
    grain = tc.spectral(n, (seed, 2), beta=0.8, lo=n / 16, hi=n / 2.1)
    height = tc.norm01(tc.blur(height, 0.6) + 0.05 * grain)
    albedo = _retone(albedo, [(0.0, (0.23, 0.20, 0.17)), (0.3, (0.33, 0.30, 0.27)),
                              (0.7, (0.47, 0.44, 0.40)), (1.0, (0.64, 0.60, 0.54))], keep=0.2)
    albedo *= (0.92 + 0.16 * grain - 0.3 * tc.cavity(height, n / 120))[..., None]
    return albedo, height, 0.6 + 0.4 * rough, 12.0


def sand(n, seed):
    """Beach and shore sand: wind ripples under fine grain, the odd pebble."""
    v, u = np.mgrid[0:n, 0:n].astype(np.float32) / n
    w = (tc.spectral(n, (seed, 1), beta=2.4, lo=1, hi=7) - 0.5) * 2.4
    ripples = 0.5 + 0.5 * np.sin(2 * np.pi * (14 * v + 2 * u) + w * 5.0)
    ripples = tc.blur(ripples ** 1.6, 1.0)
    dunes = tc.spectral(n, (seed, 2), beta=2.6, lo=1, hi=10)
    grain = tc.spectral(n, (seed, 3), beta=0.0, lo=n / 6, hi=n / 2.0)
    f1, _, _, val = tc.worley(n, 30, (seed, 4))
    pebble = np.where(val > 0.975, tc.smoothstep(0.22, 0.06, f1), 0.0).astype(np.float32)
    height = tc.norm01(0.45 * dunes + 0.28 * ripples + 0.08 * grain + 0.45 * pebble)
    t = np.clip(0.25 + 0.4 * dunes + 0.25 * ripples + 0.5 * (grain - 0.5), 0, 1)
    albedo = tc.ramp(t, [(0.0, (0.50, 0.42, 0.30)), (0.5, (0.68, 0.60, 0.44)), (1.0, (0.82, 0.75, 0.58))])
    albedo = albedo + (np.array([0.40, 0.38, 0.35], np.float32) - albedo) * (pebble > 0.3)[..., None]
    rough = 0.88 - 0.08 * ripples
    return albedo, height, rough, 3.0


def cobble(n, seed):
    """Rough paving: rounded field stones of mixed size set in packed earth, the
    kind a village lays by hand. Worn smooth on top, dirt and moss in the gaps."""
    wu = (tc.spectral(n, (seed, 6), beta=1.6, lo=4, hi=40) - 0.5) * n * 0.035
    wv = (tc.spectral(n, (seed, 7), beta=1.6, lo=4, hi=40) - 0.5) * n * 0.035
    f1, f2, _, val = (tc.warp(a.astype(np.float32), wu, wv) for a in tc.worley(n, 13, (seed, 1), jitter=0.85))
    gap = f2 - f1                                             # 0 on the joint between stones
    # Each stone: a dome that flattens on top, its edge rounded into the joint.
    width = 0.07 + 0.10 * ((val * 5.0) % 1.0)                 # joint width varies stone to stone
    body = tc.smoothstep(width, width + 0.22, gap)
    dome = np.clip(1.0 - (f1 / 0.62) ** 2, 0.0, 1.0)
    level = 0.72 + 0.28 * ((val * 11.0) % 1.0)                # stones sit at uneven heights
    pits = tc.spectral(n, (seed, 2), beta=0.9, lo=n / 40, hi=n / 3.0)
    wear = tc.spectral(n, (seed, 3), beta=2.0, lo=2, hi=24)
    stone_h = body * level * (0.72 + 0.28 * dome) - 0.05 * tc.smoothstep(0.55, 0.9, pits) * body
    # A few stones are missing or sunk: earth shows through.
    sunk = (val > 0.90).astype(np.float32)
    stone_h *= 1.0 - 0.75 * sunk
    grit = tc.spectral(n, (seed, 4), beta=0.3, lo=n / 10, hi=n / 2.1)
    earth_h = 0.16 + 0.10 * grit + 0.08 * wear
    height = tc.norm01(np.maximum(stone_h, earth_h))
    on_stone = tc.smoothstep(0.02, 0.10, stone_h - earth_h)
    tone = np.clip(0.15 + 0.5 * ((val * 17.0) % 1.0) + 0.3 * (wear - 0.5) + 0.25 * (pits - 0.5), 0.0, 1.0)
    stone = tc.ramp(tone, [(0.0, (0.30, 0.29, 0.27)), (0.5, (0.45, 0.43, 0.39)), (1.0, (0.60, 0.57, 0.50))])
    warm = ((val * 29.0) % 1.0 > 0.7)[..., None]              # the odd brown or rusty stone
    stone = np.where(warm, stone * np.array([1.10, 0.96, 0.80], np.float32), stone)
    earth = tc.ramp(np.clip(0.2 + 0.6 * grit + 0.4 * (wear - 0.5), 0, 1),
                    [(0.0, (0.13, 0.10, 0.06)), (1.0, (0.33, 0.26, 0.16))])
    moss = tc.smoothstep(0.55, 0.75, tc.spectral(n, (seed, 5), beta=1.8, lo=2, hi=14))[..., None]
    earth = earth + (np.array([0.17, 0.24, 0.08], np.float32) - earth) * 0.55 * moss
    albedo = earth + (stone - earth) * on_stone[..., None]
    albedo *= (1.0 - 0.35 * tc.cavity(height, n / 110))[..., None]
    rough = 0.92 - 0.30 * on_stone * (0.5 + 0.5 * dome)
    return albedo, height, rough, 9.0


# Layer -> (material dir under `out/`, average colour sRGB 0..1). The colours
# are the ones the game's tint maps were tuned against.
GENERATED = {
    "grass": ("mat/meadow_ground_s0", (0.196, 0.226, 0.089)),
    "dirt": ("mat/packed_earth_s0", (0.295, 0.223, 0.139)),
    "rock": ("mat/cliff_strata_s0", (0.443, 0.403, 0.351)),
    "scree": ("mat/scree_gravel_s0", (0.436, 0.405, 0.380)),
    "sand": ("mat/desert_sand_s0", (0.702, 0.624, 0.464)),
    "cobble": ("mat/cobblestone_s0", (0.316, 0.286, 0.224)),
}

# Array layer order. Keep in step with `terrain_blend.wgsl`.
LAYERS = [("grass", grass, 11), ("dirt", dirt, 23), ("rock", rock, 37), ("scree", scree, 41), ("sand", sand, 53),
          ("cobble", cobble, 67)]


def bake(name, fn, seed, n):
    source, colour = GENERATED.get(name, (None, None))
    if source and (OUT / source).exists():
        albedo, normal, rough, height = tc.load_material(OUT / source, n)
        normal = normal * 0.5 + 0.5
        albedo = tc.recolour(albedo, colour)
    else:
        albedo, height, rough, strength = fn(n, seed)
        height = tc.norm01(height)
        # `strength` is authored at 1024; keep the slope the same at other sizes.
        normal = tc.normal_map(height, strength * n / 1024.0)
    cav = 1.0 - 0.8 * tc.cavity(height, n / 100)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tc.to8(albedo), "RGB").save(d / "albedo.png")
    Image.fromarray(tc.to8(normal), "RGB").save(d / "normal.png")
    Image.fromarray(tc.to8(np.broadcast_to(rough, (n, n))), "L").save(d / "roughness.png")
    Image.fromarray(tc.to8(height), "L").save(d / "height.png")
    mean = albedo.reshape(-1, 3).mean(0)
    print(f"  {name}: mean albedo sRGB {mean.round(3).tolist()}")
    ah = np.concatenate([tc.to8(albedo), tc.to8(height)[..., None]], -1)
    nr = np.concatenate([tc.to8(normal[..., :2]), tc.to8(np.broadcast_to(rough, (n, n)))[..., None],
                         tc.to8(cav)[..., None]], -1)
    return ah, nr, mean


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--only", help="bake one layer to out/ and skip packing")
    args = ap.parse_args()
    packed = []
    for name, fn, seed in LAYERS:
        if args.only and name != args.only:
            continue
        packed.append((name, *bake(name, fn, seed, args.size)))
    if args.only:
        return
    game = game3d_assets("textures", "terrain")
    game.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([p[1] for p in packed], 0), "RGBA").save(game / "albedo_height.png")
    Image.fromarray(np.concatenate([p[2] for p in packed], 0), "RGBA").save(game / "normal_rough.png")
    meta = {"size": args.size, "layers": [p[0] for p in packed],
            "mean_albedo_srgb": {p[0]: [round(float(c), 4) for c in p[3]] for p in packed}}
    (game / "layers.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"packed {len(packed)} layers -> {game}")


if __name__ == "__main__":
    main()
