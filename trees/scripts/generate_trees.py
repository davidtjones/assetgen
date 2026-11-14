"""Bake trees for the game: meshes, textures, impostors, manifest.

    pixi run trees               # bake everything into ../game3d/assets/trees/
    pixi run trees -- --preview  # also write trees/out/preview.jpg

Look at the result in the game's asset viewer (`cargo run --release --bin
viewer` in `../game3d`); it reloads as files are rewritten.

Per species (see `tree_core.SPECIES`) this writes:

- `<bark>_bark.png`, `<bark>_bark_n.png`, `<bark>_bark_orm.png` - tiling bark:
  colour, normal map, and occlusion / roughness / metalness packed as glTF
  does (red, green, blue)
- `<bark>_bark_moss.png` - for bark that can grow moss (`tree_textures.MOSSY`):
  the same bark overgrown, with the bark's height in its alpha
- `<foliage>_leaves.png`, `<foliage>_leaves_n.png` - foliage cards, colour and
  normal map. The colour is relative to the tree's leaf colour, which the game
  supplies (`tree_textures.py`).
- `<species>_<seed>.gltf` per seed variant - mesh 0 is the near tree, mesh 1
  the mid-distance one; in each, primitive 0 is the bark and primitive 1 the
  foliage (absent on a bare tree).

and for all of them together:

- `impostors.png` - every variant's mid mesh drawn from `VIEWS` directions
  round the trunk, unlit, for far trees. The game shows a far tree as one
  upright quad turned to face the camera, carrying the view drawn from nearest
  where the camera stands, so its outline is the mesh's from any side. Alpha
  is the cut-out. The colour is the tree's **foliage only**, relative to the
  leaf colour as in the cards, and black where the tree shows wood.
- `impostor_normals.png` - the same views as surface normals, so the quad
  takes the light the way the mesh does. Each normal is in its view's own
  frame: red to the viewer's right, green up, blue toward the viewer, each
  stored as `n * 0.5 + 0.5`. Alpha is the tree's **wood only**: how bright
  the bark is there (the square root of its luminance), zero where the tree
  shows foliage. The game colours it with the species' `wood` tint.

  A far tree's colour is therefore `foliage * leaf colour * 2 + wood * tint`:
  a sum of two layers, which stays right when the atlas is filtered or
  reduced and a texel becomes part leaf, part bark. One picture with a mask
  saying which texels are leaf does not: a half-leaf texel then takes half
  the leaf colour on all of itself, and thin trunks come out pale.
- `trees.txt` - the manifest the game reads:

      impostors <views> <variants per atlas row> <cell width u> <cell height v>
      species <name> <nominal height m> <bark> <bark normals> <bark orm> <leaves> <leaf normals>
      wood <r> <g> <b>
      moss <mossy bark>
      leaf <name> <r> <g> <b>
      variant <gltf> <half width m> <height m>

  `wood`, `moss`, `leaf` and `variant` lines belong to the `species` line above them.
  `moss` is optional: a second bark colour texture, the same bark overgrown,
  with the bark's height in its alpha. The game blends it over the bark low on
  the trunk and on the upper sides of limbs, by a different amount per tree.
  `wood` is the bark's average colour over its luminance, linear: what the
  impostor's wood layer is multiplied by. A `leaf` line is a colour the species' foliage can wear, sRGB 0..1; the first is the
  default. A bare species has `-` for its leaf textures and no `leaf` lines.
  Variants are numbered in file order; number `i` has its views side by side
  in the atlas, starting at cell column `(i % per_row) * views`, row
  `i // per_row`. View `k` is drawn by a viewer standing `k / views` of a turn
  round the tree from +z toward -x. Species order is the index used by `tree`
  lines in a world's `world.txt`.

Impostor views are slow to draw and are cached in `trees/out/cache/`, keyed by
the mesh and textures they were drawn from.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "common"))

import gltf_io  # noqa: E402
import tree_core  # noqa: E402
import tree_textures  # noqa: E402

GAME = HERE.parents[2] / "game3d" / "assets" / "trees"
OUT = HERE.parent / "out"
VIEWS = 16                        # impostor views round the trunk
CELL_W, CELL_H = 128, 256         # one impostor view in the atlas
SUPER = 2                         # views are drawn this much larger, then shrunk
PER_ROW = 2                       # variants side by side in the atlas
CUTOFF = 0.4                      # the game's alpha cut-off for foliage
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def render(parts, azimuth, half_width, height, w=CELL_W * SUPER, h=CELL_H * SUPER, light=None, leaf=None):
    """Orthographic software render of a tree seen from `azimuth` (radians).

    `parts` is `[(mesh, texture, is_foliage), ...]`: a mesh dict as
    `tree_core.skin` returns, and its colour texture as float RGBA 0..1.
    Returns float RGBA with straight alpha, the surface normal per pixel in the
    view's frame (right, up, toward the viewer), and the wood layer. With
    `light` (a direction) the colour is the whole tree, shaded, its foliage in
    the colour `leaf`: a preview. Without, it is an impostor's two layers:
    the colour is foliage only, unlit and relative to the leaf colour, and the
    wood layer is the bark's luminance (linear) where the tree shows wood."""
    right = np.array([np.cos(azimuth), 0.0, np.sin(azimuth)])
    fwd = np.array([-np.sin(azimuth), 0.0, np.cos(azimuth)])
    color = np.zeros((h, w, 3), np.float32)
    normal = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.float32)
    zbuf = np.full((h, w), -1e9, np.float32)
    for mesh, texture, foliage in parts:
        pos, uv, nrm = mesh["positions"].astype(np.float64), mesh["uvs"], mesh["normals"].astype(np.float64)
        # Foliage keeps its shade in red alone (`tree_core.skin`).
        col = np.repeat(mesh["colors"][:, :1], 3, 1) if foliage else mesh["colors"][:, :3]
        tri = mesh["indices"].reshape(-1, 3)
        sx = (pos @ right / half_width * 0.5 + 0.5) * w
        sy = (1.0 - pos[:, 1] / height) * h
        sz = pos @ fwd
        ah, aw = texture.shape[:2]
        vn = np.stack([nrm @ right, nrm[:, 1], nrm @ fwd], -1).astype(np.float32)
        shade = np.ones(len(pos), np.float32)
        if light is not None:
            shade = (0.45 + 0.75 * np.clip(nrm @ (light / np.linalg.norm(light)), 0, 1)).astype(np.float32)
        tone = np.ones(3, np.float32)
        if foliage and leaf is not None:
            tone = _to_linear(np.asarray(leaf, np.float32) / 255.0) * 2.0
        for a, b, c in tri:
            xs, ys = sx[[a, b, c]], sy[[a, b, c]]
            x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), w - 1))
            y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), h - 1))
            if x1 < x0 or y1 < y0:
                continue
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
            l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            z = l0 * sz[a] + l1 * sz[b] + l2 * sz[c]
            u = (l0 * uv[a, 0] + l1 * uv[b, 0] + l2 * uv[c, 0]) % 1.0
            v = (l0 * uv[a, 1] + l1 * uv[b, 1] + l2 * uv[c, 1]) % 1.0
            tex = texture[np.clip((v * ah).astype(int), 0, ah - 1), np.clip((u * aw).astype(int), 0, aw - 1)]
            tint = (l0[..., None] * col[a] + l1[..., None] * col[b] + l2[..., None] * col[c])
            lit = l0 * shade[a] + l1 * shade[b] + l2 * shade[c]
            sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
            hit = inside & (tex[..., 3] > CUTOFF) & (z > sub_z)
            sub_z[hit] = z[hit]
            # The game multiplies the texture by the vertex colour in linear
            # light; do the same here.
            linear = _to_linear(tex[..., :3]) * tint * lit[..., None] * tone
            color[y0:y1 + 1, x0:x1 + 1][hit] = linear[hit]
            facing = l0[..., None] * vn[a] + l1[..., None] * vn[b] + l2[..., None] * vn[c]
            normal[y0:y1 + 1, x0:x1 + 1][hit] = facing[hit]
            alpha[y0:y1 + 1, x0:x1 + 1][hit] = 1.0
            if light is None:
                # Split into layers: wood leaves the colour for the wood layer.
                wood = linear @ LUMA if not foliage else np.zeros_like(z)
                mask[y0:y1 + 1, x0:x1 + 1][hit] = wood[hit]
                if not foliage:
                    color[y0:y1 + 1, x0:x1 + 1][hit] = 0.0
    return np.concatenate([_to_srgb(np.clip(color, 0, 1)), alpha[..., None]], -1), normal, mask


def shrink(img, k=SUPER):
    """Box-filter an image down by `k`. Colour is weighted by the last
    channel (alpha) so empty texels do not darken the edges."""
    h, w, c = img.shape
    blocks = img.reshape(h // k, k, w // k, k, c)
    a = blocks[..., -1:]
    total = a.sum((1, 3))
    rgb = (blocks[..., :-1] * a).sum((1, 3)) / np.maximum(total, 1e-6)
    return np.concatenate([rgb, total / (k * k)], -1)


def flood(rgba, channels=3):
    """Spread values into transparent texels so filtering has no dark fringe."""
    from scipy import ndimage
    solid = rgba[..., -1] > 0.25
    if solid.any():
        idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        rgba[..., :channels] = rgba[..., :channels][idx[0], idx[1]]
    return rgba


def views(parts, half, top):
    """One variant's strip of impostor views: colour RGBA and normal + mask
    RGBA, floats 0..1, each `CELL_H` x `VIEWS * CELL_W`. Cached on disk."""
    digest = hashlib.sha1()
    for mesh, texture, foliage in parts:
        for key in ("positions", "indices", "uvs", "normals", "colors"):
            digest.update(np.ascontiguousarray(mesh[key]).tobytes())
        digest.update(np.ascontiguousarray(texture).tobytes())
    digest.update(repr((half, top, VIEWS, CELL_W, CELL_H, SUPER, CUTOFF, 3)).encode())
    cache = OUT / "cache" / f"{digest.hexdigest()}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["colour"], z["normal"]
    colour = np.zeros((CELL_H, VIEWS * CELL_W, 4), np.float32)
    normal = np.zeros((CELL_H, VIEWS * CELL_W, 4), np.float32)
    for k in range(VIEWS):
        rgba, facing, wood = render(parts, 2 * np.pi * k / VIEWS, half, top)
        # Reduce in linear light, weighted by coverage: a texel that is part
        # leaf and part bark then holds that share of each layer.
        rgba[..., :3] = _to_linear(rgba[..., :3])
        img = flood(shrink(rgba))
        img[..., :3] = _to_srgb(img[..., :3])
        extra = flood(shrink(np.concatenate([facing, wood[..., None], rgba[..., 3:]], -1)), 4)
        facing = extra[..., :3]
        facing /= np.maximum(np.linalg.norm(facing, axis=-1, keepdims=True), 1e-6)
        # A 1 px clear border keeps neighbouring views out of each other.
        img[:1, :, 3] = img[-1:, :, 3] = 0
        img[:, :1, 3] = img[:, -1:, 3] = 0
        colour[:, k * CELL_W:(k + 1) * CELL_W] = img
        normal[:, k * CELL_W:(k + 1) * CELL_W, :3] = facing * 0.5 + 0.5
        normal[:, k * CELL_W:(k + 1) * CELL_W, 3] = np.sqrt(np.clip(extra[..., 3], 0, 1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, colour=colour, normal=normal)
    return colour, normal


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    GAME.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.time()
    count = sum(len(s.seeds) for s in tree_core.SPECIES)
    sheet_w, sheet_h = PER_ROW * VIEWS * CELL_W, -(-count // PER_ROW) * CELL_H
    sheet = np.zeros((sheet_h, sheet_w, 4), np.float32)
    normals = np.zeros((sheet_h, sheet_w, 4), np.float32)
    normals[..., :3] = (0.5, 0.5, 1.0)
    lines = [f"impostors {VIEWS} {PER_ROW} {CELL_W / sheet_w:.6f} {CELL_H / sheet_h:.6f}"]
    previews = []
    slot = 0
    textures = {}

    def texture(kind, make, suffix):
        """Bake one texture set once, however many species share it. Returns
        its file names and a small float copy of the colour for rendering."""
        if (kind, suffix) not in textures:
            maps = make(kind)
            colour = maps[0]
            names = tuple(f"{kind}_{suffix}{end}.png" for end in ("", "_n", "_orm")[:len(maps)])
            for name, image in zip(names, maps):
                Image.fromarray(image, "RGBA").save(GAME / name)
            # Half size for rendering: a card covers few impostor pixels.
            small = Image.fromarray(colour, "RGBA")
            small = small.resize((small.width // 2, small.height // 2), Image.BOX)
            textures[(kind, suffix)] = (names, np.asarray(small).astype(np.float32) / 255.0)
        return textures[(kind, suffix)]

    for species in tree_core.SPECIES:
        bark_files, bark_tex = texture(species.bark, tree_textures.bark, "bark")
        leaf_files, leaf_tex = (("-", "-"), None)
        if species.foliage:
            leaf_files, leaf_tex = texture(species.foliage, tree_textures.foliage, "leaves")
        header = None
        for seed in species.seeds:
            habit, near_bark, near_leaves, mid_bark, mid_leaves = tree_core.build(species, seed)
            if header is None:
                lines.append(f"species {species.name} {habit.height:.2f} {' '.join(bark_files)} "
                             f"{leaf_files[0]} {leaf_files[1]}")
                linear = _to_linear(bark_tex[..., :3]).reshape(-1, 3).mean(0)
                wood = linear / max(float(linear @ LUMA), 1e-6)
                lines.append(f"wood {wood[0]:.4f} {wood[1]:.4f} {wood[2]:.4f}")
                if tree_textures.moss(species.bark) is not None:
                    lines.append(f"moss {texture(species.bark, tree_textures.moss, 'bark_moss')[0][0]}")
                for name, rgb in species.leaves:
                    lines.append(f"leaf {name} {rgb[0] / 255:.4f} {rgb[1] / 255:.4f} {rgb[2] / 255:.4f}")
                header = True
            file = f"{species.name}_{seed}.gltf"
            gltf_io.write_meshes(GAME / file, [
                (f"{species.name}_{seed}_near", [p for p in (near_bark, near_leaves) if p is not None]),
                (f"{species.name}_{seed}_mid", [p for p in (mid_bark, mid_leaves) if p is not None]),
            ])
            both = [near_bark["positions"]] + ([near_leaves["positions"]] if near_leaves else [])
            pos = np.concatenate(both)
            half = float(np.abs(pos[:, [0, 2]]).max()) * 1.04
            top = float(pos[:, 1].max()) * 1.02
            # The views are of the mid mesh: that is the one a tree hands over
            # from, so the outline must be its.
            parts = [(mid_bark, bark_tex, False)] + ([(mid_leaves, leaf_tex, True)] if mid_leaves else [])
            colour, normal = views(parts, half, top)
            gx, gy = (slot % PER_ROW) * VIEWS * CELL_W, (slot // PER_ROW) * CELL_H
            sheet[gy:gy + CELL_H, gx:gx + VIEWS * CELL_W] = colour
            normals[gy:gy + CELL_H, gx:gx + VIEWS * CELL_W] = normal
            lines.append(f"variant {file} {half:.3f} {top:.3f}")
            tris = lambda *ps: sum(len(p["indices"]) // 3 for p in ps if p is not None)  # noqa: E731
            print(f"  {species.name} seed {seed}: near {tris(near_bark)} bark + {tris(near_leaves)} foliage, "
                  f"mid {tris(mid_bark)} + {tris(mid_leaves)} triangles, half width {half:.1f} m, height {top:.1f} m")
            if args.preview and seed == species.seeds[0]:
                light = np.array([0.5, 0.8, 0.3])
                leaf = species.leaves[0][1] if species.leaves else None
                near = [(near_bark, bark_tex, False)] + ([(near_leaves, leaf_tex, True)] if near_leaves else [])
                for shown in (near, parts):
                    previews.append(render(shown, 0.6, 9.0, 20.0, 540, 600, light, leaf)[0])
            slot += 1
    Image.fromarray((sheet * 255 + 0.5).astype(np.uint8), "RGBA").save(GAME / "impostors.png")
    Image.fromarray((normals * 255 + 0.5).astype(np.uint8), "RGBA").save(GAME / "impostor_normals.png")
    (GAME / "trees.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {slot} variants -> {GAME} in {time.time() - began:.0f} s")
    if args.preview:
        bg = np.array([0.62, 0.72, 0.84], np.float32)
        tiles = [p[..., :3] * p[..., 3:] + bg * (1 - p[..., 3:]) for p in previews]
        rows = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)]
        width = max(r.shape[1] for r in rows)
        rows = [np.concatenate([r, np.broadcast_to(bg, (r.shape[0], width - r.shape[1], 3))], 1) for r in rows]
        Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(OUT / "preview.jpg", quality=90)


if __name__ == "__main__":
    main()
