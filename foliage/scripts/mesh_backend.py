"""
Grass geometry generator — MESH backend.

Turns the shared parametric blade model (see `blade_core.py`) into 3D geometry
instead of a raster silhouette. Same `BladeParams`, same compliance/width model,
so a mesh blade and a silhouette blade are the same plant.

STEP 1 (this file, for now): a single blade — a flat tapering triangle strip of
N segments, with a gentle self-weight droop baked in. STEP 2 (later): assemble N
blades into a tuft.

Design decisions, per the target look:

- **Flat strip, tunable segment count.** A blade is `segments` quads stacked up
  its length, tapering toward the tip. `segments` is the vertex-cost knob.
- **Baked droop, not stiffness.** The tip tangent is pitched `droop_deg` off
  vertical; the bend is concentrated toward the tip by `t ** droop_gamma`
  (the same compliance idea as the raster spine), so the root stays planted and
  the blade arcs over under its own weight. This is a *static per-blade*
  property — not a cluster-wide lean — so it stays consistent with the raster
  rule of keeping wind out of the baked shape.
- **Normals blended toward world-up.** A flat ribbon lit by its true geometric
  normal reads flat. Blending the normal toward the ground-up vector
  (`up_normal_blend`) makes a field of blades light as one soft surface — the
  standard stylized-grass trick. Fully geometric at blend=0, fully up at blend=1.
- **Root->tip weight baked in.** `UV.y` runs 0 at the root to 1 at the tip and
  IS the wind weight: a future vertex shader multiplies its displacement by it,
  so the root stays fixed and the tip travels most. `UV.y` doubles as the
  coordinate for the base->tip color gradient (baked into COLOR_0 for preview).

Everything is emitted in **world units** (Bevy/three.js), converted from the
model's pixel-space by `MeshParams.world_per_pixel`.
"""

import math
import random
from dataclasses import dataclass, replace

from blade_core import BladeParams, ClusterParams, blade_width, sample_blade


# Pixel -> world scale for blade geometry. At 1/600, a ~500px blade stands ~0.83
# world units tall - a sensible tuft height against the 40-unit terrain.
_DEFAULT_WORLD_PER_PIXEL = 1.0 / 600.0


@dataclass
class MeshParams:
    """3D-only knobs. Read from the optional `mesh:` block of a profile YAML;
    every field falls back to a sensible default so profiles need not set any."""
    segments: int = 5                       # quads up the blade; the vertex-cost knob
    droop_deg: float = 27.0                 # tip tangent pitch off vertical (15-40 typical)
    droop_gamma: float = 2.0                # >1 concentrates the bend toward the tip
    droop_noise_deg: float = 4.0            # small per-blade variation on the droop
    up_normal_blend: float = 0.7            # 0 = geometric normal, 1 = world-up
    world_per_pixel: float = _DEFAULT_WORLD_PER_PIXEL
    root_color: tuple[float, float, float] = (0.13, 0.26, 0.07)  # sRGB picker value: dark olive
    tip_color: tuple[float, float, float] = (0.46, 0.70, 0.22)   # sRGB picker value: yellow-green
    # --- tuft-only (STEP 2) --- None fields are resolved from the cluster at build time.
    tuft_radius_px: float | None = None     # base disk radius; default = x_spread * 0.5
    tuft_splay_deg: float | None = None      # extra droop for edge blades; default from fan_out_strength
    tuft_azimuth_jitter: float = 0.35        # radians of lean-direction randomness

    @staticmethod
    def from_profile(data: dict) -> "MeshParams":
        block = dict((data or {}).get("mesh", {}) or {})
        for key in ("root_color", "tip_color"):
            if key in block:
                block[key] = tuple(block[key])
        return MeshParams(**block)


def canonical_blade(cluster: ClusterParams) -> BladeParams:
    """A single representative blade at the midpoint of a cluster's ranges.

    Deterministic (no RNG) so the inspector always shows the same blade for a
    given profile — better for judging shape than a random cluster sample.
    """
    mid = lambda r: (r[0] + r[1]) / 2.0
    return BladeParams(
        origin_x=0.0,
        length=mid(cluster.length_range),
        initial_angle=-math.pi / 2,               # straight up; droop is added in 3D
        compliance_gamma=mid(cluster.compliance_gamma_range),
        base_thickness=mid(cluster.base_thickness_range),
        tip_thickness=mid(cluster.tip_thickness_range),
        noise_strength=cluster.noise_strength,
        spread=mid(cluster.spread_range),
        edge_noise=cluster.edge_noise,
    )


def integrate_blade_3d(
    blade: BladeParams,
    mesh: MeshParams,
    azimuth: float,
    rng: random.Random | None = None,
) -> list[tuple[float, float, float]]:
    """Walk root->tip in 3D, baking in the self-weight droop. Returns world-space
    spine positions. The blade rises in +Y and leans along the horizontal
    `azimuth` direction; the bend is concentrated toward the tip."""
    seg_len_world = (blade.length / mesh.segments) * mesh.world_per_pixel
    droop = math.radians(mesh.droop_deg)
    if rng is not None and mesh.droop_noise_deg:
        droop += math.radians(rng.gauss(0.0, mesh.droop_noise_deg))

    fx, fz = math.sin(azimuth), math.cos(azimuth)  # horizontal lean direction

    x = y = z = 0.0
    spine = [(0.0, 0.0, 0.0)]
    for i in range(mesh.segments):
        t = (i + 1) / mesh.segments
        # Tangent angle off vertical grows from ~0 at the root to `droop` at the
        # tip, weighted so the curve lives near the tip (rigid root, floppy tip).
        alpha = droop * (t ** mesh.droop_gamma)
        dy = math.cos(alpha) * seg_len_world
        dl = math.sin(alpha) * seg_len_world
        x += fx * dl
        y += dy
        z += fz * dl
        spine.append((x, y, z))
    return spine


def build_blade_mesh(
    blade: BladeParams,
    mesh: MeshParams,
    azimuth: float = 0.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rng: random.Random | None = None,
    seed2: tuple[float, float] = (0.5, 0.5),
) -> dict:
    """Build one flat, tapering, drooping blade. Returns arrays ready for
    `gltf_io.write_gltf`: positions, normals, uvs, uv1, colors, indices.

    `seed2` is a pair of per-blade randoms written to every vertex's uv1 (TEXCOORD_1);
    the wind shader uses them to give each blade its own phase/amplitude/direction so
    a tuft reads as many independent blades, not one rigid fan."""
    spine = integrate_blade_3d(blade, mesh, azimuth, rng)
    n = mesh.segments

    # Constant horizontal width axis, perpendicular to the lean direction.
    wx, wz = math.cos(azimuth), -math.sin(azimuth)
    width_axis = (wx, 0.0, wz)
    ox, oy, oz = origin

    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    uv1: list[tuple[float, float]] = []
    colors: list[tuple[float, float, float, float]] = []

    # Convert the sRGB picker colours to linear once, then lerp in linear space
    # (COLOR_0 is linear). The gradient is baked so the rendered blade matches the
    # authored root/tip swatches.
    root_lin = _srgb_to_linear3(mesh.root_color)
    tip_lin = _srgb_to_linear3(mesh.tip_color)

    for i, (px, py, pz) in enumerate(spine):
        t = i / n
        half_w = 0.5 * blade_width(t, blade.base_thickness, blade.tip_thickness, blade.spread) \
            * mesh.world_per_pixel

        # Spine tangent (averaged from neighbors), for the geometric normal.
        if 0 < i < n:
            ax, ay, az = spine[i - 1]
            bx, by, bz = spine[i + 1]
        elif i == 0:
            ax, ay, az = spine[i]
            bx, by, bz = spine[i + 1]
        else:
            ax, ay, az = spine[i - 1]
            bx, by, bz = spine[i]
        tan = _normalize((bx - ax, by - ay, bz - az))
        geom_n = _normalize(_cross(tan, width_axis))
        # Keep the geometric normal on the +Y side so the up-blend never flips it.
        if geom_n[1] < 0:
            geom_n = (-geom_n[0], -geom_n[1], -geom_n[2])
        normal = _normalize(_lerp(geom_n, (0.0, 1.0, 0.0), mesh.up_normal_blend))

        cx = px + ox
        cy = py + oy
        cz = pz + oz
        # Left / right edge vertices.
        positions.append((cx + width_axis[0] * half_w, cy, cz + width_axis[2] * half_w))
        positions.append((cx - width_axis[0] * half_w, cy, cz - width_axis[2] * half_w))
        normals.append(normal)
        normals.append(normal)
        # UV.y = root->tip weight (wind + gradient); UV.x = across-width.
        uvs.append((0.0, t))
        uvs.append((1.0, t))
        # uv1 = per-blade wind randoms (constant across the blade).
        uv1.append(seed2)
        uv1.append(seed2)
        col = _lerp(root_lin, tip_lin, t)
        colors.append((col[0], col[1], col[2], 1.0))
        colors.append((col[0], col[1], col[2], 1.0))

    # Two triangles per segment, connecting station i (verts 2i, 2i+1) to i+1.
    indices: list[int] = []
    for i in range(n):
        l0, r0 = 2 * i, 2 * i + 1
        l1, r1 = 2 * (i + 1), 2 * (i + 1) + 1
        indices += [l0, r0, l1, r1, l1, r0]

    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "uv1": uv1,
        "colors": colors,
        "indices": indices,
    }


def _merge_into(dst: dict, src: dict) -> None:
    """Append one blade mesh onto an accumulating tuft mesh, offsetting indices."""
    base = len(dst["positions"])
    dst["positions"].extend(src["positions"])
    dst["normals"].extend(src["normals"])
    dst["uvs"].extend(src["uvs"])
    dst["uv1"].extend(src["uv1"])
    dst["colors"].extend(src["colors"])
    dst["indices"].extend(i + base for i in src["indices"])


def build_tuft_mesh(cluster: ClusterParams, mesh: MeshParams, seed: int = 0) -> dict:
    """Assemble a tuft: N blades placed in a disk, radiating outward, merged into
    one mesh.

    Reuses `sample_blade` for each blade's *shape* (length, thickness, compliance,
    spread) — so a mesh tuft matches its silhouette siblings — but replaces the
    raster model's 1D x-spread with a 2D disk placement + per-blade azimuth, which
    the flat silhouette had no concept of. `fan_out_strength` becomes literal here:
    outer blades splay outward (extra droop toward their radial direction).
    """
    rng = random.Random(seed)
    n = max(1, cluster.n_blades + rng.randint(-cluster.n_blades_var, cluster.n_blades_var))

    radius_px = mesh.tuft_radius_px if mesh.tuft_radius_px is not None else cluster.x_spread * 0.5
    splay_deg = mesh.tuft_splay_deg if mesh.tuft_splay_deg is not None else cluster.fan_out_strength * 35.0
    wpp = mesh.world_per_pixel

    # Evenly-spread base positions (not pure random) so the tuft has no clumps or
    # holes. Each blade leans/arcs outward along the radial direction of its base.
    points = _even_disk_points(n, radius_px, rng)

    tuft = {"positions": [], "normals": [], "uvs": [], "uv1": [], "colors": [], "indices": []}
    for px, pz in points:
        blade = sample_blade(rng, cluster)  # shape params (origin_x/initial_angle unused in 3D)
        r = math.hypot(px, pz)
        # φ from the actual base point so lean == outward radial (sin φ, cos φ).
        phi = math.atan2(px, pz)
        origin = (px * wpp, 0.0, pz * wpp)
        azimuth = phi + rng.gauss(0.0, mesh.tuft_azimuth_jitter)
        # Edge blades splay more than center blades.
        r_norm = r / radius_px if radius_px else 0.0
        blade_mesh = replace(mesh, droop_deg=mesh.droop_deg + splay_deg * r_norm)
        # Two independent per-blade randoms → its own wind phase + amplitude/direction.
        seed2 = (rng.random(), rng.random())
        geo = build_blade_mesh(blade, blade_mesh, azimuth=azimuth, origin=origin, rng=rng, seed2=seed2)
        _merge_into(tuft, geo)
    return tuft


def _even_disk_points(n: int, radius: float, rng: random.Random, k: int = 12) -> list[tuple[float, float]]:
    """`n` points spread evenly in a disk via Mitchell's best-candidate algorithm:
    each new point is the best of `k` random candidates — the one farthest from all
    points already placed. Gives Poisson-disk-like even spacing (no clumps, no
    holes) with an exact count, and no min-distance to tune."""
    pts: list[tuple[float, float]] = []
    for _ in range(n):
        best, best_d = None, -1.0
        for _ in range(k):
            rr = radius * math.sqrt(rng.random())
            th = rng.uniform(0.0, 2.0 * math.pi)
            c = (rr * math.cos(th), rr * math.sin(th))
            if not pts:
                best = c
                break
            d = min((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 for p in pts)
            if d > best_d:
                best_d, best = d, c
        pts.append(best)
    return pts


# --- tiny vec3 helpers (avoid a numpy dependency for a handful of ops) ---

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _normalize(v):
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _lerp(a, b, t):
    return tuple(a[k] * (1.0 - t) + b[k] * t for k in range(len(a)))


def _srgb_to_linear(c: float) -> float:
    """Convert one sRGB channel (a colour-picker value) to linear light, using the
    standard sRGB transfer function. `COLOR_0` is linear by the glTF spec, so the
    authored picker colours must be converted before they're baked - otherwise a
    dark olive `0.13` is read as linear and renders ~2x too light."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb_to_linear3(c):
    return tuple(_srgb_to_linear(x) for x in c)
