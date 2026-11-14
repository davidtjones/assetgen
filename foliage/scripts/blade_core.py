"""
Shared parametric grass-blade model.

Single source of truth for blade shape. Two backends consume it, so a
`dense_tuft` billboard and a `dense_tuft` mesh are the *same* plant by
construction:

- `generate_silhouette.py` (raster backend) -> 2D silhouette PNG, used as both
  ControlNet conditioning and billboard alpha. Mid-distance foliage.
- `mesh_backend.py` / `generate_geometry.py` -> 3D blade/tuft mesh (glTF).
  Close-up foliage.

The model lives in **pixel-space** (canvas 768x1024, ground at y=990) so the
existing profiles need no changes. The mesh backend converts pixels -> world
units with a single scale factor.

Physics model: each blade is a chain of segments walked base->tip. At each step
the tangent bends by a noise-driven angle change weighted by a *compliance*
profile (`t ** compliance_gamma`) — small near the rigid root, large near the
floppy tip, so blades curve at their tips rather than wandering. Blades share a
cluster's statistical distribution but sample independently: coherence without a
cluster-wide directional bias.

Note on lean: the raster path generates every blade *at rest* (no directional
lean) on purpose — a cluster-wide lean would bake wind into the texture, which
conflicts with runtime vertex-shader wind. The mesh backend adds a per-blade
*self-weight droop* (see `mesh_backend.integrate_blade_3d`); that is a static
property of an individual blade, not wind, so it does not violate this rule.
"""

import math
import random
from dataclasses import dataclass


# Canvas / coordinate space the model lives in. Pixel units.
CANVAS_W = 768
CANVAS_H = 1024
GROUND_Y = 990
# Segment count for the *raster* spine (smooth silhouette). The mesh backend
# passes its own, much lower, count to keep blades low-poly.
N_SEGMENTS = 20


@dataclass
class BladeParams:
    origin_x: float
    length: float
    initial_angle: float          # radians, -pi/2 = straight up
    compliance_gamma: float       # higher = base stays rigid longer, tip bends more sharply
    base_thickness: float
    tip_thickness: float
    noise_strength: float         # per-segment angle jitter, radians
    spread: float = 0.0           # extra bell-shaped bulge in the middle (broadleaf shapes)
    edge_noise: float = 0.0       # independent noise on left/right edges (irregular width)
    head_radius: float = 0.0
    head_elongation: float = 1.0  # 1.0 = round flower, >1.0 = elongated seed


@dataclass
class ClusterParams:
    center_x: float = CANVAS_W / 2
    x_spread: float = 200
    n_blades: int = 20
    n_blades_var: int = 4
    length_range: tuple[float, float] = (350, 600)
    initial_angle_spread: float = 0.15
    compliance_gamma_range: tuple[float, float] = (1.5, 2.5)
    base_thickness_range: tuple[float, float] = (4, 9)
    tip_thickness_range: tuple[float, float] = (1.0, 2.0)
    noise_strength: float = 0.05
    fan_out_strength: float = 0.0
    spread_range: tuple[float, float] = (0.0, 0.0)  # 0 = current tapered strokes, >0 = broader leaves
    edge_noise: float = 0.0
    head_prob: float = 0.0
    head_radius_range: tuple[float, float] = (0.0, 0.0)
    head_elongation: float = 1.0


def sample_blade(rng: random.Random, cluster: ClusterParams) -> BladeParams:
    origin_x = rng.gauss(cluster.center_x, cluster.x_spread / 3)
    # Fan-out: blades near the edge of the cluster lean outward, giving a natural spread.
    dx_from_center = origin_x - cluster.center_x
    fan_out = math.tanh(dx_from_center / max(cluster.x_spread / 2, 1)) * cluster.fan_out_strength
    initial_angle = -math.pi / 2 + rng.gauss(0, cluster.initial_angle_spread) + fan_out

    head_radius = 0.0
    if cluster.head_prob > 0 and rng.random() < cluster.head_prob:
        head_radius = rng.uniform(*cluster.head_radius_range)

    return BladeParams(
        origin_x=origin_x,
        length=rng.uniform(*cluster.length_range),
        initial_angle=initial_angle,
        compliance_gamma=rng.uniform(*cluster.compliance_gamma_range),
        base_thickness=rng.uniform(*cluster.base_thickness_range),
        tip_thickness=rng.uniform(*cluster.tip_thickness_range),
        noise_strength=cluster.noise_strength,
        spread=rng.uniform(*cluster.spread_range),
        edge_noise=cluster.edge_noise,
        head_radius=head_radius,
        head_elongation=cluster.head_elongation,
    )


def integrate_blade(
    blade: BladeParams,
    rng: random.Random,
    ground_y: float,
    n_segments: int = N_SEGMENTS,
) -> list[tuple[float, float]]:
    """Walk base->tip, accumulating noise-driven angle changes weighted by compliance.

    Returns the 2D centerline (spine) in pixel-space. Used by the raster backend.
    """
    x, y = blade.origin_x, ground_y
    angle = blade.initial_angle
    seg_len = blade.length / n_segments
    positions = [(x, y)]

    for i in range(n_segments):
        t = (i + 1) / n_segments
        compliance = t ** blade.compliance_gamma
        d_angle = rng.gauss(0, blade.noise_strength) * compliance
        angle += d_angle
        x += math.cos(angle) * seg_len
        y += math.sin(angle) * seg_len
        positions.append((x, y))
    return positions


def blade_width(
    t: float,
    base_thickness: float,
    tip_thickness: float,
    spread: float = 0.0,
) -> float:
    """Blade half-outline width at normalized length `t` in [0, 1].

    Linear base->tip taper plus an optional bell-shaped mid bulge for broadleaf
    shapes. `bell(t) = 4t(1-t)` peaks at t=0.5 and is 0 at both endpoints, so the
    two edges always meet at root and tip. Shared by both backends so raster and
    mesh agree on width exactly.
    """
    bell = 4.0 * t * (1.0 - t)
    return base_thickness * (1 - t) + tip_thickness * t + spread * bell


# ClusterParams range fields are declared as tuples; YAML doesn't distinguish
# tuples from lists, so we convert these keys after loading.
_TUPLE_FIELDS = frozenset({
    "length_range",
    "compliance_gamma_range",
    "base_thickness_range",
    "tip_thickness_range",
    "spread_range",
    "head_radius_range",
})


# Profile keys consumed only by the mesh backend (3D). Filtered out of
# ClusterParams construction here. See mesh_backend.MeshParams.
_MESH_ONLY_FIELDS = frozenset({
    "mesh",
})


def cluster_from_raw(raw: dict) -> ClusterParams:
    """Build a ClusterParams from a profile's raw YAML dict.

    Strips the mesh-only keys (the mesh backend reads those separately) and
    converts the range fields to tuples. Single source of truth for turning a
    profile dict into cluster params, shared by both backends.
    """
    data = {k: v for k, v in raw.items() if k not in _MESH_ONLY_FIELDS}
    for key in _TUPLE_FIELDS:
        if key in data:
            data[key] = tuple(data[key])
    return ClusterParams(**data)


def load_profile(path) -> tuple[ClusterParams, dict]:
    """Load one profile YAML: returns its ClusterParams and its raw dict.

    The raw dict is returned so the mesh backend can build its MeshParams from
    the `mesh:` block without re-reading the file (MeshParams lives there, not
    here, to avoid a circular import).
    """
    import yaml  # local import: keeps core importable without PyYAML for pure-geometry callers

    raw = yaml.safe_load(path.read_text()) or {}
    return cluster_from_raw(raw), raw


def load_profiles(profiles_dir) -> dict[str, ClusterParams]:
    """Load one ClusterParams per *.yaml file in profiles_dir, keyed by file stem."""
    return {
        path.stem: load_profile(path)[0]
        for path in sorted(profiles_dir.glob("*.yaml"))
    }
