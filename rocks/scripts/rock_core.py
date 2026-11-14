"""
Procedural rock mesh generation (numpy, dependency-light).

A rock is built in four steps, each a tunable knob in `RockParams`:

    1. Icosphere      - an evenly-tessellated unit sphere (no UV-sphere poles).
    2. Ellipsoid      - a random per-axis squash, so rocks aren't round.
    3. Noise displace - fBm value noise pushes each vertex along its radius,
                        giving the organic lumps.
    4. Fracture cuts  - a few random planes flatten protruding caps into flat
                        angular faces, plus one near-horizontal *base* cut so the
                        rock has a flat bottom to rest on. This is what separates
                        "reads as stone" from "blobby potato".

Output is plain vertex/face/normal arrays; `generate_rocks.py` bakes them to glTF
via the shared `common/gltf_io.py`. Texturing is NOT done here: the rock carries
only geometry, and Bevy applies a world-space triplanar material (see
`src/rocks.rs` / `assets/shaders/rock_triplanar.wgsl`), so no UVs are needed.
"""

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class RockParams:
    """Tunable rock shape knobs. Populated from a profile YAML via `from_profile`."""

    # Icosphere subdivisions. 4 = 20*4^4 = 5120 tris / 2562 verts: enough to carry
    # the noise + facets, cheap at a handful of rocks. Each +1 quadruples tris.
    subdivisions: int = 4

    # Surface displacement: radius = 1 + amp * fbm(dir * freq), over `octaves`.
    displace_amp: float = 0.35
    displace_freq: float = 1.6
    displace_octaves: int = 4

    # Per-axis ellipsoid squash range (each axis picks uniformly in [min, max]).
    ellipsoid_min: float = 0.65
    ellipsoid_max: float = 1.35
    # Optional separate range for the vertical axis: below 1 gives slabs and
    # low outcrops, above 1 gives standing stones. None = same as the others.
    ellipsoid_y_min: float | None = None
    ellipsoid_y_max: float | None = None

    # Strata: bedded rock weathers into stacked ledges. `strata` is the number
    # of beds over the rock's height (0 = off); each bed is pulled in toward the
    # vertical axis by a random share of up to `strata_depth`, with a sharp step
    # between beds. `strata_tilt` dips the bedding plane (rise per unit run).
    strata: int = 0
    strata_depth: float = 0.25
    strata_tilt: float = 0.12

    # Post-cut surface roughen: a small, higher-frequency displacement applied
    # AFTER the cuts, along each vertex's normal, so the flat cut faces gain
    # micro-relief and read as rough stone instead of flat gem facets. Set
    # `roughen_amp` to 0 to disable.
    roughen_amp: float = 0.05
    roughen_freq: float = 5.0
    roughen_octaves: int = 3

    # Random planar fracture cuts (flat faces), beyond the one flat base cut.
    side_cuts: int = 3
    # Cut offset range: how far out along each plane normal the flat face sits.
    # Smaller = deeper cut (more material removed).
    cut_offset_min: float = 0.60
    cut_offset_max: float = 0.85
    # Base cut: a mostly-downward plane giving a flat bottom. `base_tilt` jitters
    # the normal off straight-down; `base_offset_*` set how high the flat base is.
    base_tilt: float = 0.15
    base_offset_min: float = 0.55
    base_offset_max: float = 0.72

    @classmethod
    def from_profile(cls, raw: dict) -> "RockParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A unit-radius icosphere: subdivide an icosahedron `subdivisions` times and
    project every vertex back onto the sphere. Returns (verts (N,3), faces (M,3))."""
    t = (1.0 + 5.0**0.5) / 2.0
    base = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in base]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in cache:
            m = (verts[i] + verts[j]) / 2.0
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(subdivisions):
        nxt = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nxt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nxt

    return np.array(verts), np.array(faces, dtype=np.int64)


def _hash3(p: np.ndarray) -> np.ndarray:
    """Integer bit-mix lattice hash of a 3D grid point -> [0,1). `p` is (N,3) of
    integer lattice coords (as floats; callers pass `floor(p) + offset`).

    A proper murmur-style finalizer - the same one the Rust terrain hash uses
    (`src/terrain.rs`), extended to three axes. Unlike the old `fract(sin(dot))`
    trick, it doesn't lean on float32 precision loss to look random, so it's
    faster and genuinely decorrelated across the lattice.
    """
    mask = np.uint64(0xFFFFFFFF)
    # Low 32 bits of each coord (matches Rust's `coord as u32`, incl. negatives).
    ix = (p[:, 0].astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    iy = (p[:, 1].astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    iz = (p[:, 2].astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    h = (
        ((ix * np.uint64(0x16574B1D)) & mask)
        + ((iy * np.uint64(0x0B409C3F)) & mask)
        + ((iz * np.uint64(0x9E3779B1)) & mask)
    ) & mask
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x2C1B3C6D)) & mask
    h ^= h >> np.uint64(12)
    h = (h * np.uint64(0x297154C9)) & mask
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0x00FFFFFF)).astype(np.float64) / float(0x01000000)


def _value_noise(p: np.ndarray) -> np.ndarray:
    """Trilinearly-interpolated 3D value noise at points `p` (N,3) -> [0,1]."""
    pi = np.floor(p)
    pf = p - pi
    w = pf * pf * (3.0 - 2.0 * pf)  # smoothstep for continuous slope

    def h(dx: int, dy: int, dz: int) -> np.ndarray:
        return _hash3(pi + np.array([dx, dy, dz], dtype=np.float64))

    x00 = h(0, 0, 0) + (h(1, 0, 0) - h(0, 0, 0)) * w[:, 0]
    x10 = h(0, 1, 0) + (h(1, 1, 0) - h(0, 1, 0)) * w[:, 0]
    x01 = h(0, 0, 1) + (h(1, 0, 1) - h(0, 0, 1)) * w[:, 0]
    x11 = h(0, 1, 1) + (h(1, 1, 1) - h(0, 1, 1)) * w[:, 0]
    y0 = x00 + (x10 - x00) * w[:, 1]
    y1 = x01 + (x11 - x01) * w[:, 1]
    return y0 + (y1 - y0) * w[:, 2]


def _fbm(p: np.ndarray, octaves: int) -> np.ndarray:
    """Fractal sum of value noise, normalized to roughly [-1, 1]."""
    total = np.zeros(len(p))
    amp, freq, norm = 0.5, 1.0, 0.0
    for _ in range(octaves):
        total += amp * (_value_noise(p * freq) * 2.0 - 1.0)
        norm += amp
        freq *= 2.0
        amp *= 0.5
    return total / norm


def _smooth_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals (unnormalized face normals accumulated)."""
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # length ~ 2*area, so larger faces weigh more
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return normals / lens


def build_rock(params: RockParams, seed: int) -> dict:
    """Build one rock. Returns {positions, indices, normals}, with the flat base
    seated at y=0 so it drops onto the ground. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    dirs, faces = icosphere(params.subdivisions)  # unit sphere directions

    # Per-rock random form: ellipsoid squash + a noise-domain offset.
    ell = rng.uniform(params.ellipsoid_min, params.ellipsoid_max, size=3)
    noise_off = rng.uniform(-100.0, 100.0, size=3)

    # Displace radius by fBm along each vertex's own direction, then squash.
    r = 1.0 + params.displace_amp * _fbm(dirs * params.displace_freq + noise_off, params.displace_octaves)
    if params.ellipsoid_y_min is not None and params.ellipsoid_y_max is not None:
        ell[1] = rng.uniform(params.ellipsoid_y_min, params.ellipsoid_y_max)
    pts = dirs * r[:, None] * ell

    # Fracture planes: random side cuts + one near-horizontal base cut.
    cuts = []
    for _ in range(params.side_cuts):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        cuts.append((n, rng.uniform(params.cut_offset_min, params.cut_offset_max)))
    base_n = np.array([
        rng.uniform(-params.base_tilt, params.base_tilt),
        -1.0,
        rng.uniform(-params.base_tilt, params.base_tilt),
    ])
    base_n /= np.linalg.norm(base_n)
    cuts.append((base_n, rng.uniform(params.base_offset_min, params.base_offset_max)))

    # Flatten anything past each plane onto it (project the protruding cap down).
    for n, d in cuts:
        dist = pts @ n - d
        mask = dist > 0.0
        pts[mask] -= dist[mask, None] * n

    # Bedding: step the outline in and out by height, bed by bed. Done after
    # the cuts so the beds span what is left of the rock and the joint faces
    # get ledges too.
    if params.strata > 0:
        dip = rng.normal(size=2)
        dip *= params.strata_tilt / max(np.linalg.norm(dip), 1e-6)
        level = pts[:, 1] + pts[:, 0] * dip[0] + pts[:, 2] * dip[1]
        lo, hi = level.min(), level.max()
        # Uneven bed thickness: random cut heights, sorted.
        edges = np.sort(rng.uniform(0.0, 1.0, size=params.strata - 1)) if params.strata > 1 else np.array([])
        edges = np.concatenate([[0.0], edges, [1.0]])
        # Alternate hard beds (stand proud) and soft beds (weather back), so
        # neighbouring beds always differ and every boundary is a real ledge.
        u = rng.uniform(0.0, 1.0, size=params.strata)
        soft = (np.arange(params.strata) + rng.integers(0, 2)) % 2 == 1
        inset = 1.0 - params.strata_depth * np.where(soft, 0.45 + 0.55 * u, 0.2 * u)
        t = (level - lo) / max(hi - lo, 1e-6)
        bed = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, params.strata - 1)
        # Blend toward the bed above over the top 12% of each bed, so a step is
        # a steep riser the mesh can carry rather than a tear.
        top = edges[bed + 1]
        thick = np.maximum(top - edges[bed], 1e-6)
        k = np.clip((t - (top - 0.12 * thick)) / (0.12 * thick), 0.0, 1.0)
        above = np.minimum(bed + 1, params.strata - 1)
        f = inset[bed] * (1.0 - k) + inset[above] * k
        pts[:, 0] *= f
        pts[:, 2] *= f

    # Roughen: perturb every vertex along its surface normal by a little
    # high-frequency noise, so the (otherwise mathematically flat) cut faces gain
    # micro-relief and read as rough stone rather than gem facets.
    if params.roughen_amp > 0.0:
        n1 = _smooth_normals(pts, faces)
        rough = _fbm(pts * params.roughen_freq + noise_off, params.roughen_octaves)
        pts += n1 * (params.roughen_amp * rough[:, None])

    # Seat the flat base at y=0 so the caller can place the rock at ground height
    # and scale it about its base without it floating or sinking.
    pts[:, 1] -= pts[:, 1].min()

    normals = _smooth_normals(pts, faces)

    return {
        "positions": [tuple(p) for p in pts],
        "indices": faces.reshape(-1).tolist(),
        "normals": [tuple(v) for v in normals],
    }
