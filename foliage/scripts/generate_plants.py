"""
Undergrowth plants: the small things that grow between the grass and the trees.

    pixi run plants

Bakes a few variants of each plant straight into the game
(`../game3d/assets/meshes/plants/<plant>_seed<NNN>.gltf`), where
`game3d/src/undergrowth.rs` scatters them by biome:

- fern     a rosette of arching fronds, each a row of leaflets. Forest floor.
- flower   a clump of thin stems with small bright heads and a few leaves.
           One colour per variant, so a meadow gets drifts of white, yellow
           and blue.
- heather  a low springy mound of twigs with purple tips. Moor.
- scrub    a dry, woody, grey-green tangle. Desert and waste.

Every plant is made of flat ribbons, like the grass tufts (`mesh_backend.py`),
and carries the same extra data so it shares the grass wind material:

- UV.y is the wind weight: 0 at the ground, 1 at the tips that move most.
- UV1 is a random pair per ribbon, so stems do not sway in step.
- COLOR_0 is the colour (linear). No texture.
- Normals lean toward straight up, so a plant lights as one soft clump.

Meshes are in metres at their natural size; the game scales them a little
either way.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from gltf_io import write_gltf  # noqa: E402

VARIANTS = 4
UP = np.array([0.0, 1.0, 0.0])
# How far normals lean toward straight up (0 = true surface normal).
UP_BLEND = 0.55


def _lin(c):
    """sRGB colour (0..1) to linear."""
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class Mesh:
    def __init__(self):
        self.p, self.n, self.uv, self.uv1, self.c, self.i = [], [], [], [], [], []

    def vertex(self, pos, normal, weight, pair, colour):
        n = _unit(normal * (1 - UP_BLEND) + UP * UP_BLEND)
        self.p.append(tuple(float(x) for x in pos))
        self.n.append(tuple(float(x) for x in n))
        self.uv.append((0.5, float(np.clip(weight, 0, 1))))
        self.uv1.append(pair)
        self.c.append((float(colour[0]), float(colour[1]), float(colour[2]), 1.0))
        return len(self.p) - 1

    def tri(self, a, b, c):
        self.i += [a, b, c]

    def ribbon(self, spine, side, widths, weights, colours, pair):
        """A flat strip along `spine` (k x 3), `widths[k]` wide across `side`.
        A zero final width closes it to a point."""
        k = len(spine)
        rows = []
        for j in range(k):
            tangent = _unit(spine[min(j + 1, k - 1)] - spine[max(j - 1, 0)])
            across = _unit(side - tangent * np.dot(side, tangent))
            normal = np.cross(tangent, across)
            if normal[1] < 0:
                normal = -normal
            half = across * widths[j] * 0.5
            rows.append((
                self.vertex(spine[j] - half, normal, weights[j], pair, colours[j]),
                self.vertex(spine[j] + half, normal, weights[j], pair, colours[j]),
            ))
        for (l0, r0), (l1, r1) in zip(rows, rows[1:]):
            self.tri(l0, r0, l1)
            self.tri(r1, l1, r0)

    def leaf(self, base, tip, side, width, weight, colour, pair):
        """A flat pointed leaf from `base` to `tip`: a diamond of two triangles."""
        mid = base * 0.55 + tip * 0.45
        across = _unit(side) * width * 0.5
        normal = np.cross(_unit(tip - base), _unit(side))
        if normal[1] < 0:
            normal = -normal
        a = self.vertex(base, normal, weight[0], pair, colour[0])
        b = self.vertex(mid - across, normal, weight[1], pair, colour[1])
        c = self.vertex(mid + across, normal, weight[1], pair, colour[1])
        d = self.vertex(tip, normal, weight[2], pair, colour[2])
        self.tri(a, c, b)
        self.tri(b, c, d)

    def save(self, path, name):
        write_gltf(path, positions=self.p, indices=self.i, normals=self.n, uvs=self.uv,
                   uv1=self.uv1, colors=self.c, name=name, double_sided=True)
        print(f"wrote {path.name}  [{len(self.p)} verts, {len(self.i) // 3} tris]")


def _arc(origin, heading, length, rise, droop, k, top=None):
    """A stem leaving `origin` toward compass `heading` (radians), `rise`
    (radians above flat) at the root, bending down by `droop` radians by the
    tip. Returns (points k x 3, heights 0..1 for wind weight)."""
    out = np.array([math.sin(heading), 0.0, math.cos(heading)])
    pts = [np.asarray(origin, float)]
    step = length / (k - 1)
    for j in range(1, k):
        t = (j - 0.5) / (k - 1)
        angle = rise - droop * t ** 1.6
        pts.append(pts[-1] + (out * math.cos(angle) + UP * math.sin(angle)) * step)
    pts = np.array(pts)
    top = top or max(pts[:, 1].max(), 1e-3)
    return pts, np.clip(pts[:, 1] / top, 0, 1)


def _shade(colour, rng, amount=0.08):
    return np.clip(np.asarray(colour) * (1 + rng.uniform(-amount, amount)), 0, 1)


def fern(rng):
    m = Mesh()
    dark, light = _lin([0.10, 0.26, 0.07]), _lin([0.32, 0.52, 0.15])
    fronds = rng.integers(7, 11)
    for f in range(fronds):
        heading = f / fronds * math.tau + rng.normal(0, 0.25)
        length = rng.uniform(0.55, 0.85)
        # Young fronds in the middle stand up; the outer ones arch right over.
        rise = rng.uniform(0.75, 1.3)
        k = 12
        spine, h = _arc(np.array([math.sin(heading), 0, math.cos(heading)]) * 0.03, heading,
                        length, rise, rng.uniform(1.0, 1.7), k, top=0.6)
        pair = (float(rng.random()), float(rng.random()))
        tint = _shade([1, 1, 1], rng, 0.12)
        out = np.array([math.sin(heading), 0.0, math.cos(heading)])
        side = np.cross(UP, out)
        # The stalk.
        t = np.linspace(0, 1, k)
        m.ribbon(spine, side, 0.012 * (1 - 0.7 * t), h, [dark * tint] * k, pair)
        # Leaflets: none on the bare lower stalk, longest a third of the way up,
        # tapering to the tip, each angled forward and drooping a little.
        for j in range(2, k):
            tj = j / (k - 1)
            span = length * 0.26 * math.sin(math.pi * min((tj - 0.1) / 0.9, 1.0) ** 0.7) + 0.01
            forward = _unit(spine[j] - spine[j - 1])
            colour = (dark * (1 - tj) + light * tj) * tint
            for sgn in (-1.0, 1.0):
                reach = _unit(side * sgn + forward * 0.45 - UP * 0.18) * span
                m.leaf(spine[j], spine[j] + reach, forward, span * 0.42,
                       (h[j], h[j], min(h[j] + 0.08, 1)), (colour * 0.85, colour, colour * 1.1), pair)
    return m


FLOWERS = [
    ([0.95, 0.94, 0.88], [0.93, 0.74, 0.12]),     # white, yellow eye
    ([0.96, 0.80, 0.14], [0.80, 0.52, 0.08]),     # yellow
    ([0.42, 0.44, 0.86], [0.92, 0.86, 0.40]),     # blue
    ([0.84, 0.36, 0.56], [0.95, 0.85, 0.45]),     # pink
]


def flower(rng, variant):
    m = Mesh()
    petal, eye = (_lin(c) for c in FLOWERS[variant % len(FLOWERS)])
    stem_lo, stem_hi = _lin([0.10, 0.24, 0.06]), _lin([0.26, 0.44, 0.13])
    stems = rng.integers(7, 12)
    for _ in range(stems):
        heading = rng.uniform(0, math.tau)
        r = rng.uniform(0, 0.22)
        origin = np.array([math.sin(heading) * r, 0, math.cos(heading) * r])
        length = rng.uniform(0.5, 0.85)
        k = 4
        spine, h = _arc(origin, heading, length, rng.uniform(1.3, 1.5), rng.uniform(0.1, 0.4), k, top=0.85)
        pair = (float(rng.random()), float(rng.random()))
        side = np.cross(UP, np.array([math.sin(heading), 0, math.cos(heading)]))
        t = np.linspace(0, 1, k)
        m.ribbon(spine, side, np.full(k, 0.012), h, [stem_lo * (1 - x) + stem_hi * x for x in t], pair)
        # The head: a ring of petals tilted to face up and a little outward.
        top = spine[-1]
        face = _unit(UP + np.array([math.sin(heading), 0, math.cos(heading)]) * rng.uniform(0.1, 0.6))
        a = _unit(np.cross(face, [1.0, 0.0, 0.3]))
        b = np.cross(face, a)
        # Larger than life, so a drift still reads as colour from across a field.
        size = rng.uniform(0.055, 0.08)
        petals = 6
        centre = m.vertex(top + face * 0.004, face, 1.0, pair, eye)
        ring = []
        for q in range(petals * 2):
            ang = q / (petals * 2) * math.tau
            rad = size if q % 2 == 0 else size * 0.45
            ring.append(m.vertex(top + (a * math.cos(ang) + b * math.sin(ang)) * rad, face, 1.0, pair,
                                 _shade(petal, rng, 0.05)))
        for q in range(len(ring)):
            m.tri(centre, ring[q], ring[(q + 1) % len(ring)])
    # A few leaves at the foot.
    for _ in range(rng.integers(5, 8)):
        heading = rng.uniform(0, math.tau)
        out = np.array([math.sin(heading), 0, math.cos(heading)])
        length = rng.uniform(0.15, 0.3)
        tip = out * length + UP * length * rng.uniform(0.5, 1.1)
        pair = (float(rng.random()), float(rng.random()))
        m.leaf(out * 0.01, tip, np.cross(UP, out), length * 0.35, (0, 0.15, 0.35),
               (stem_lo, stem_hi * 0.9, stem_hi), pair)
    return m


def _twigs(m, rng, count, radius, height, base, tip, tip_at, width, flat=0.0):
    """A dome of short twigs: each leaves the ground inside `radius` and leans
    outward, more so toward the rim. `tip_at` is where along a twig its colour
    turns from `base` to `tip`."""
    for _ in range(count):
        heading = rng.uniform(0, math.tau)
        r = radius * math.sqrt(rng.random())
        out = np.array([math.sin(heading), 0, math.cos(heading)])
        origin = out * r * 0.55
        lean = r / radius
        length = height * rng.uniform(0.7, 1.15) * (1 - 0.3 * lean)
        k = 4
        spine, h = _arc(origin, heading + rng.normal(0, 0.5), length,
                        1.45 - (0.75 + flat) * lean + rng.normal(0, 0.12), rng.uniform(0.1, 0.6), k, top=height)
        pair = (float(rng.random()), float(rng.random()))
        t = np.linspace(0, 1, k)
        mix = np.clip((t - tip_at) / max(1 - tip_at, 1e-3), 0, 1)
        shade = _shade([1, 1, 1], rng, 0.15)
        colours = [(base * (1 - x) + tip * x) * shade for x in mix]
        side = _unit(np.array([math.cos(heading), rng.normal(0, 0.6), -math.sin(heading)]))
        m.ribbon(spine, side, width * (1 - 0.55 * t), h, colours, pair)


def heather(rng):
    m = Mesh()
    woody, green = _lin([0.14, 0.13, 0.07]), _lin([0.20, 0.27, 0.12])
    bloom = _lin([0.56, 0.27, 0.52])
    # A tight cushion of many fine, mostly upright sprigs. Only the top third
    # of the taller ones is in bloom.
    radius = rng.uniform(0.3, 0.42)
    _twigs(m, rng, 60, radius, 0.26, woody, green, 0.1, 0.022, flat=-0.3)
    _twigs(m, rng, 50, radius, 0.34, green, bloom, 0.62, 0.02, flat=-0.3)
    return m


def scrub(rng):
    m = Mesh()
    wood, pale = _lin([0.30, 0.25, 0.19]), _lin([0.52, 0.47, 0.36])
    leaf = _lin([0.42, 0.45, 0.25])
    radius = rng.uniform(0.4, 0.6)
    _twigs(m, rng, 34, radius, 0.6, wood, pale, 0.3, 0.02, flat=0.15)
    _twigs(m, rng, 30, radius, 0.5, pale, leaf, 0.5, 0.035, flat=0.1)
    return m


PLANTS = {"fern": lambda rng, v: fern(rng), "flower": flower,
          "heather": lambda rng, v: heather(rng), "scrub": lambda rng, v: scrub(rng)}


def main():
    out = Path(__file__).resolve().parents[3] / "game3d" / "assets" / "meshes" / "plants"
    out.mkdir(parents=True, exist_ok=True)
    for index, (name, build) in enumerate(PLANTS.items()):
        for v in range(VARIANTS):
            rng = np.random.default_rng(1000 * index + v)
            build(rng, v).save(out / f"{name}_seed{v:03d}.gltf", name)


if __name__ == "__main__":
    main()
