"""Kit pieces: building pieces made by code on a grid of cubes, for builders to snap together.

A kit is one style of building pieces (walls, towers, gates, stairs...) that a builder, an
agent or a player, places on a structure's grid. It is a prop spec with `kind: kit`
(`props/specs/<name>.yaml`), run by `props/scripts/prop.py` with the stages `build`, `frame`
and `export`. This module holds what those stages do; `prop.py`'s docstring has the commands.

**The grid.** Every piece fills whole cells of a grid of cubes `cell` metres on a side (2 m).
A piece's own frame has +Y up and its cells at `[0, w) x [0, h) x [0, d)`, so its mesh lies in
`x in [0, w*cell], y in [0, h*cell], z in [0, d*cell]`, except that a piece may reach `below`
metres under its bottom (a foundation going into the ground). A piece's **outer side is -z**:
battlements stand on the -z edge, a gate's outside is -z, stairs climb toward +z.

**Turns.** A piece is placed with a turn `t` in 0..3: a rotation of `t` quarter turns about +Y,
the way Bevy's `Quat::from_rotation_y(t * pi/2)` turns (+Z goes to +X), then moved so its
cells start at the cell it was placed at. One turn takes the outer side from -z to -x.
`place()` is the reference; game3d's `world/kit.rs` does the same.

**The manifest** (`kit.json`, beside the pieces in `game3d/assets/kits/<kit>/`): `cell`
(metres), `colors` (sRGB 0..255 per surface kind: `stone`, `wood`, `roof`), `pieces` (below)
and `test`, the spec's test arrangement as `[piece, [i, j, k], turn]`. Each piece says:

- `file`, its glTF, and `parts`: the surface kind of each of its primitives, in order (each
  primitive is one surface kind, so the game can give each kind its own material).
- `cells` `[w, h, d]` and `below` (metres under the bottom).
- `faces`: for each cell on the outside of the piece and each of its outward sides (`-x`,
  `+x`, `-y`, `+y`, `-z`, `+z`), what that face is: `solid` (fully closed, measured),
  `open` (nothing there), `passage` (a way through, such as a gate), or the name of a partial
  profile (`parapet`, `stairs`...) that should meet the same profile.
- `walk`: where a player can stand, as rectangles `{x: [x0, x1], z: [z0, z1], y: [y0, y1]}`
  in the piece's frame, `y0` at `z0` and `y1` at `z1` (a flat surface has `y0 == y1`; stairs
  are a ramp).
- `passages`: the clear openings, `{axis: "x"|"z", u: [u0, u1], y: [y0, y1]}`, where `u`
  is the other horizontal axis, entered from the piece's - side along `axis`; `depth` (metres,
  default the whole piece) is how far in it stays clear, so a tower's doorway opens into its
  hollow and not through its far wall.

**The checks** (`frame`, written to `out/<kit>/frame.json`): each piece stays inside its
cells (and `below`); every side the piece says is solid is closed over every cell on it;
every passage is clear all the way through; every walk surface is where the piece says, with
2 m of headroom; and the test arrangement in the spec (`test`) has no two pieces in one cell,
no passage opening onto a solid face, and no two different partial profiles meeting.

Pieces are flat-shaded boxes and extrusions with one material per surface kind (stone, wood,
roof), coloured by the spec's `colors`. UVs are planar in metres, so a tiling material can
dress them later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SIDES = ("-x", "+x", "-y", "+y", "-z", "+z")
AXIS = {"x": 0, "y": 1, "z": 2}
HEADROOM = 2.0


# --- geometry -------------------------------------------------------------------------------

class Mesh:
    """Polygons grouped by material, flat shaded."""

    def __init__(self):
        self.parts: dict[str, dict[str, list]] = {}

    def poly(self, pts, normal, mat):
        """A convex planar polygon facing `normal` (its winding is fixed to match)."""
        p = np.asarray(pts, float)
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), n) < 0:
            p = p[::-1]
        part = self.parts.setdefault(mat, {"positions": [], "normals": [], "uvs": [], "indices": []})
        base = len(part["positions"])
        ax = int(np.argmax(np.abs(n)))
        u_ax, v_ax = {0: (2, 1), 1: (0, 2), 2: (0, 1)}[ax]
        for q in p:
            part["positions"].append(q.tolist())
            part["normals"].append(n.tolist())
            part["uvs"].append([q[u_ax] / 2.0, -q[v_ax] / 2.0])
        for i in range(1, len(p) - 1):
            part["indices"] += [base, base + i, base + i + 1]

    def rect(self, axis, at, u, v, sign, mat):
        """An axis-aligned rectangle in the plane `axis = at`, spanning `u` and `v` (the other
        two axes in x, y, z order), facing `sign` along the axis."""
        others = [a for a in range(3) if a != axis]
        if u[1] - u[0] <= 1e-9 or v[1] - v[0] <= 1e-9:
            return
        pts = []
        for a, b in ((u[0], v[0]), (u[1], v[0]), (u[1], v[1]), (u[0], v[1])):
            q = [0.0, 0.0, 0.0]
            q[axis], q[others[0]], q[others[1]] = at, a, b
            pts.append(q)
        n = [0.0, 0.0, 0.0]
        n[axis] = sign
        self.poly(pts, n, mat)

    def box(self, lo, hi, mat, skip=()):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        for axis in range(3):
            o = [a for a in range(3) if a != axis]
            u, v = (lo[o[0]], hi[o[0]]), (lo[o[1]], hi[o[1]])
            name = "xyz"[axis]
            if f"-{name}" not in skip:
                self.rect(axis, lo[axis], u, v, -1, mat)
            if f"+{name}" not in skip:
                self.rect(axis, hi[axis], u, v, +1, mat)

    def pierced(self, lo, hi, axis, hole, mat, skip=()):
        """A box with a rectangular hole straight through it along `axis` ("x" or "z").
        `hole` is `(u0, u1, y0, y1)`: `u` is the other horizontal axis. A hole that starts
        at the bottom of the box is a doorway."""
        a = AXIS[axis]
        ua = 2 if a == 0 else 0
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        u0, u1, y0, y1 = hole
        doorway = y0 <= lo[1] + 1e-9
        # The faces along the other axes, as for a plain box; a doorway has no bottom under it.
        self.box(lo, hi, mat, skip=tuple(set(skip) | {f"-{axis}", f"+{axis}"} | ({"-y"} if doorway else set())))
        if doorway and "-y" not in skip:
            for u in ((lo[ua], u0), (u1, hi[ua])):
                if a == 0:
                    self.rect(1, lo[1], (lo[0], hi[0]), u, -1, mat)
                else:
                    self.rect(1, lo[1], u, (lo[2], hi[2]), -1, mat)
        # The two pierced faces: the strips round the hole.
        for sign, at in ((-1, lo[a]), (+1, hi[a])):
            if f"{'-' if sign < 0 else '+'}{axis}" in skip:
                continue
            strips = [((lo[ua], u0), (lo[1], hi[1])), ((u1, hi[ua]), (lo[1], hi[1])),
                      ((u0, u1), (lo[1], y0)), ((u0, u1), (y1, hi[1]))]
            for u, v in strips:
                if a == 0:   # plane x = at; rect's axes are (y, z): v then u
                    self.rect(0, at, v, u, sign, mat)
                else:        # plane z = at; axes (x, y)
                    self.rect(2, at, u, v, sign, mat)
        # The reveal: the hole's four inner faces, facing into the hole.
        d = (lo[a], hi[a])
        if a == 0:
            self.rect(2, u0, d, (y0, y1), +1, mat)
            self.rect(2, u1, d, (y0, y1), -1, mat)
            self.rect(1, y1, d, (u0, u1), -1, mat)
            if y0 > lo[1] + 1e-9:
                self.rect(1, y0, d, (u0, u1), +1, mat)
        else:
            self.rect(0, u0, (y0, y1), d, +1, mat)
            self.rect(0, u1, (y0, y1), d, -1, mat)
            self.rect(1, y1, (u0, u1), d, -1, mat)
            if y0 > lo[1] + 1e-9:
                self.rect(1, y0, (u0, u1), d, +1, mat)

    def arrays(self):
        """`[(material, primitive dict)]` with numpy arrays."""
        return [(m, {k: np.asarray(v, np.float32 if k != "indices" else np.uint32) for k, v in p.items()})
                for m, p in self.parts.items()]

    def triangles(self):
        """Every triangle as an `(n, 3, 3)` array, and each one's normal."""
        tris, normals = [], []
        for p in self.parts.values():
            pos = np.asarray(p["positions"])
            idx = np.asarray(p["indices"]).reshape(-1, 3)
            tris.append(pos[idx])
            normals.append(np.asarray(p["normals"])[idx[:, 0]])
        return np.concatenate(tris), np.concatenate(normals)


# --- the pieces -----------------------------------------------------------------------------

@dataclass
class Piece:
    name: str
    cells: tuple[int, int, int]
    about: str
    below: float = 0.0
    solid: tuple[str, ...] = ()            # sides that must be closed over every cell
    walk: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    profile: str = ""                      # what a partial face is called (default: the name)
    mesh: Mesh | None = None


def merlons(m: Mesh, along: str, length: float, depth: tuple[float, float], y: tuple[float, float], cell: float):
    """Merlons 1 m wide in the middle of each cell along an edge."""
    for i in range(int(round(length / cell))):
        c = (i + 0.5) * cell
        if along == "x":
            m.box((c - 0.5, y[0], depth[0]), (c + 0.5, y[1], depth[1]), "stone", skip=("-y",))
        else:
            m.box((depth[0], y[0], c - 0.5), (depth[1], y[1], c + 0.5), "stone", skip=("-y",))


def build_pieces(cell: float) -> dict[str, Piece]:
    C = cell
    out: dict[str, Piece] = {}

    def add(p: Piece, m: Mesh):
        p.mesh = m
        p.profile = p.profile or p.name
        out[p.name] = p

    ALL = SIDES
    # foundation: a block of masonry going 4 m into the ground, to stand a wall on a slope.
    m = Mesh()
    m.box((0, -4, 0), (C, C, C), "stone")
    add(Piece("foundation", (1, 1, 1), "masonry block that reaches 4 m below its cell, for uneven ground",
              below=4.0, solid=("-x", "+x", "+y", "-z", "+z"),
              walk=[dict(x=[0, C], z=[0, C], y=[C, C])]), m)

    # wall: a cube of masonry. A curtain wall is one cell thick; its top is the wall-walk.
    m = Mesh()
    m.box((0, 0, 0), (C, C, C), "stone")
    add(Piece("wall", (1, 1, 1), "cube of masonry; stack for height, its top is a wall-walk",
              solid=ALL, walk=[dict(x=[0, C], z=[0, C], y=[C, C])]), m)

    # parapet: breastwork and a merlon on the outer edge, standing on a wall's top.
    m = Mesh()
    m.box((0, 0, 0), (C, 1.0, 0.5), "stone")
    merlons(m, "x", C, (0, 0.5), (1.0, 1.9), C)
    add(Piece("parapet", (1, 1, 1), "battlement on the outer (-z) edge; stands on a wall's top"), m)

    # parapet_corner: battlements on the -z and -x edges.
    m = Mesh()
    m.box((0, 0, 0), (C, 1.0, 0.5), "stone")
    m.box((0, 0, 0.5), (0.5, 1.0, C), "stone", skip=("-z",))
    merlons(m, "x", C, (0, 0.5), (1.0, 1.9), C)
    merlons(m, "z", C, (0, 0.5), (1.0, 1.9), C)
    add(Piece("parapet_corner", (1, 1, 1), "battlement on the -z and -x edges, for an outer corner",
              profile="parapet"), m)

    # gate: a gateway through a wall, 2 cells wide and 2 high, with a round arch.
    W, H, D = 2 * C, 2 * C, C
    x0, x1, spring = 0.6, W - 0.6, 2.2
    r, cx = (x1 - x0) / 2, W / 2
    arc = [(cx - r * math.cos(t), spring + r * math.sin(t)) for t in np.linspace(0, math.pi, 9)]
    m = Mesh()
    for sign, z in ((-1, 0.0), (+1, D)):
        m.rect(2, z, (0, x0), (0, H), sign, "stone")
        m.rect(2, z, (x1, W), (0, H), sign, "stone")
        for (ax_, ay), (bx, by) in zip(arc, arc[1:]):
            m.poly([(ax_, ay, z), (bx, by, z), (bx, H, z), (ax_, H, z)], (0, 0, sign), "stone")
    m.rect(0, 0, (0, H), (0, D), -1, "stone")
    m.rect(0, W, (0, H), (0, D), +1, "stone")
    m.rect(1, H, (0, W), (0, D), +1, "stone")
    m.rect(1, 0, (0, x0), (0, D), -1, "stone")
    m.rect(1, 0, (x1, W), (0, D), -1, "stone")
    m.rect(0, x0, (0, spring), (0, D), +1, "stone")
    m.rect(0, x1, (0, spring), (0, D), -1, "stone")
    for (ax_, ay), (bx, by) in zip(arc, arc[1:]):
        mid = np.array([(ax_ + bx) / 2, (ay + by) / 2])
        n = np.array([cx, spring]) - mid
        m.poly([(ax_, ay, 0), (bx, by, 0), (bx, by, D), (ax_, ay, D)], (n[0], n[1], 0), "stone")
    add(Piece("gate", (2, 2, 1), "gateway 2 cells wide and 2 high: a round-arched passage 2.8 m wide through the wall",
              solid=("-x", "+x", "+y"), walk=[dict(x=[0, W], z=[0, D], y=[H, H])],
              passages=[dict(axis="z", u=[x0, x1], y=[0, spring])], profile="arch"), m)

    # doorway: a wall 2 cells high with a framed door opening.
    W, H, D = C, 2 * C, C
    m = Mesh()
    m.pierced((0, 0, 0), (W, H, D), "z", (0.4, 1.6, 0, 2.4), "stone")
    for lo, hi in (((0.4, 0, 0.8), (0.5, 2.3, 1.2)), ((1.5, 0, 0.8), (1.6, 2.3, 1.2)), ((0.4, 2.3, 0.8), (1.6, 2.4, 1.2))):
        m.box(lo, hi, "wood")
    add(Piece("doorway", (1, 2, 1), "wall 2 cells high with a framed door opening 1 m wide",
              solid=("-x", "+x", "+y"), walk=[dict(x=[0, W], z=[0, D], y=[H, H])],
              passages=[dict(axis="z", u=[0.5, 1.5], y=[0, 2.3])]), m)

    # towers: a hollow square 3 cells a side, walls 0.8 m thick, in three kinds of storey.
    def tower(m: Mesh, h: float, holes: dict):
        t, S = 0.8, 3 * C
        for i in range(3):
            u = (i * C, (i + 1) * C)
            for side, (lo, hi) in (("-z", ((u[0], 0, 0), (u[1], h, t))), ("+z", ((u[0], 0, S - t), (u[1], h, S)))):
                skip = tuple(s for s, edge in (("-x", i > 0), ("+x", i < 2)) if edge)
                hole = holes.get((side, i))
                if hole:
                    m.pierced(lo, hi, "z", hole, "stone", skip=skip)
                else:
                    m.box(lo, hi, "stone", skip=skip)
            zs = (max(u[0], t), min(u[1], S - t))
            for side, (lo, hi) in (("-x", ((0, 0, zs[0]), (t, h, zs[1]))), ("+x", ((S - t, 0, zs[0]), (S, h, zs[1])))):
                skip = ("-z", "+z")
                hole = holes.get((side, i))
                if hole:
                    m.pierced(lo, hi, "x", hole, "stone", skip=skip)
                else:
                    m.box(lo, hi, "stone", skip=skip)

    S = 3 * C
    m = Mesh()
    tower(m, 2 * C, {("-z", 1): (2.3, 3.7, 0, 2.4)})
    add(Piece("tower_base", (3, 2, 3), "ground storey of a square tower: hollow, a doorway in the middle of its -z side",
              solid=("-x", "+x", "+z"), passages=[dict(axis="z", u=[2.3, 3.7], y=[0, 2.4], depth=3.0)], profile="tower"), m)

    slit = {(s, 1): (2.9, 3.1, 1.2, 2.8) for s in ("-z", "+z", "-x", "+x")}
    m = Mesh()
    tower(m, 2 * C, slit)
    add(Piece("tower_mid", (3, 2, 3), "middle storey of a square tower: hollow, an arrow slit in the middle of each side",
              profile="tower"), m)

    m = Mesh()
    m.box((0.5, 0, 0.5), (S - 0.5, 0.3, S - 0.5), "wood")
    for lo, hi in (((0, 0, 0), (S, 1.3, 0.5)), ((0, 0, S - 0.5), (S, 1.3, S)),
                   ((0, 0, 0.5), (0.5, 1.3, S - 0.5)), ((S - 0.5, 0, 0.5), (S, 1.3, S - 0.5))):
        m.box(lo, hi, "stone")
    merlons(m, "x", S, (0, 0.5), (1.3, C), C)
    merlons(m, "x", S, (S - 0.5, S), (1.3, C), C)
    merlons(m, "z", S, (0, 0.5), (1.3, C), C)
    merlons(m, "z", S, (S - 0.5, S), (1.3, C), C)
    add(Piece("tower_top", (3, 1, 3), "top of a square tower: a floor with battlements all round",
              solid=("-y",), walk=[dict(x=[0.5, S - 0.5], z=[0.5, S - 0.5], y=[0.3, 0.3])], profile="tower_top"), m)

    # stairs: ten steps rising 2 m over 2 cells, toward +z, solid underneath.
    m = Mesh()
    n, rise, run = 10, C / 10, 2 * C / 10
    for k in range(n):
        z0, z1, y = k * run, (k + 1) * run, (k + 1) * rise
        m.rect(1, y, (0, C), (z0, z1), +1, "stone")                 # tread
        m.rect(2, z0, (0, C), (k * rise, y), -1, "stone")            # riser
        m.rect(0, 0, (0, y), (z0, z1), -1, "stone")                  # sides
        m.rect(0, C, (0, y), (z0, z1), +1, "stone")
    m.rect(2, 2 * C, (0, C), (0, C), +1, "stone")
    m.rect(1, 0, (0, C), (0, 2 * C), -1, "stone")
    add(Piece("stairs", (1, 1, 2), "flight of ten steps rising one cell (2 m) over two, toward +z",
              solid=("-y", "+z"), walk=[dict(x=[0, C], z=[0, 2 * C], y=[0, C])]), m)

    # floor: a plank floor at the bottom of its cell.
    m = Mesh()
    m.box((0, 0, 0), (C, 0.25, C), "wood")
    add(Piece("floor", (1, 1, 1), "plank floor 0.25 m thick at the bottom of its cell",
              solid=("-y",), walk=[dict(x=[0, C], z=[0, C], y=[0.25, 0.25])]), m)

    # roof: a 45 degree slope rising toward +z; two back to back make a ridge.
    m = Mesh()
    m.poly([(0, 0, 0), (C, 0, 0), (C, C, C), (0, C, C)], (0, 1, -1), "roof")
    m.rect(2, C, (0, C), (0, C), +1, "stone")
    m.rect(1, 0, (0, C), (0, C), -1, "wood")
    m.poly([(0, 0, 0), (0, 0, C), (0, C, C)], (-1, 0, 0), "stone")
    m.poly([(C, 0, 0), (C, 0, C), (C, C, C)], (+1, 0, 0), "stone")
    add(Piece("roof", (1, 1, 1), "roof slope at 45 degrees rising toward +z; two back to back make a ridge",
              solid=("-y", "+z")), m)
    return out


# --- measuring ------------------------------------------------------------------------------

def _in_tris(pts2, tris2):
    """Which 2D points (n, 2) lie inside any of the 2D triangles (m, 3, 2)."""
    if len(tris2) == 0:
        return np.zeros(len(pts2), bool)
    a, b, c = tris2[:, 0], tris2[:, 1], tris2[:, 2]
    v0, v1 = (c - a)[None], (b - a)[None]
    v2 = pts2[:, None] - a[None]
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d02, d12 = (v0 * v2).sum(-1), (v1 * v2).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    return ((u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)).any(1)


def coverage(tris, normals, axis: int, at: float, sign: int, u: tuple, v: tuple, n: int = 16) -> float:
    """Share of the rectangle `u x v` in the plane `axis = at` covered by triangles lying in
    that plane and facing `sign`."""
    others = [a for a in range(3) if a != axis]
    on = (np.abs(tris[:, :, axis] - at) < 1e-4).all(1) & (normals[:, axis] * sign > 0.5)
    t2 = tris[on][:, :, others]
    g = (np.arange(n) + 0.5) / n
    uu, vv = np.meshgrid(u[0] + g * (u[1] - u[0]), v[0] + g * (v[1] - v[0]))
    return float(_in_tris(np.stack([uu.ravel(), vv.ravel()], 1), t2).mean())


def ray_hits(tris, origins, direction):
    """Distance along `direction` to the first triangle for each origin, inf where none."""
    d = np.asarray(direction, float)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(d, e2)
    det = (e1 * p).sum(-1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = origins[:, None] - v0[None]
    u = (s * p[None]).sum(-1) * inv
    q = np.cross(s, e1[None])
    v = (q * d).sum(-1) * inv
    t = (q * e2[None]).sum(-1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return np.where(hit, t, np.inf).min(1)


def faces(p: Piece, C: float, tris, normals) -> list[dict]:
    """What each outward cell face of the piece is (see the module header)."""
    w, h, d = p.cells
    size = (w, h, d)
    out = []
    for axis in range(3):
        o = [a for a in range(3) if a != axis]
        for sign in (-1, +1):
            side = ("-" if sign < 0 else "+") + "xyz"[axis]
            at = 0.0 if sign < 0 else size[axis] * C
            for i in range(size[o[0]]):
                for j in range(size[o[1]]):
                    u, v = (i * C, (i + 1) * C), (j * C, (j + 1) * C)
                    cov = coverage(tris, normals, axis, at, sign, u, v)
                    cell = [0, 0, 0]
                    cell[axis] = 0 if sign < 0 else size[axis] - 1
                    cell[o[0]], cell[o[1]] = i, j
                    if cov >= 0.99:
                        kind = "solid"
                    elif any(_passage_on(pp, side, u, v, o, C) for pp in p.passages):
                        kind = "passage"
                    elif cov <= 0.01:
                        kind = "open"
                    else:
                        kind = p.profile
                    out.append(dict(cell=cell, side=side, kind=kind, cover=round(cov, 3)))
    return out


def _passage_on(pp: dict, side: str, u, v, o, C) -> bool:
    """Whether passage `pp` opens through this face (u, v ranges on axes `o`)."""
    if side[1] != pp["axis"]:
        return False
    ua = 2 if pp["axis"] == "x" else 0
    pu = pp["u"]
    py = pp["y"]
    ru = u if o[0] == ua else v
    ry = v if o[1] == 1 else u
    return pu[0] < ru[1] and pu[1] > ru[0] and py[0] < ry[1] and py[1] > ry[0]


def check_piece(p: Piece, C: float) -> list[dict]:
    tris, normals = p.mesh.triangles()
    w, h, d = p.cells
    box_lo = np.array([0, -p.below, 0])
    box_hi = np.array([w * C, h * C, d * C])
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    checks = []

    def check(name, ok, value, limit):
        checks.append(dict(name=name, ok=bool(ok), value=value, limit=limit))

    inside = (lo >= box_lo - 1e-4).all() and (hi <= box_hi + 1e-4).all()
    check("inside_cells", inside, f"{np.round(lo, 3).tolist()}..{np.round(hi, 3).tolist()}",
          f"{box_lo.tolist()}..{box_hi.tolist()}")
    fs = faces(p, C, tris, normals)
    for side in p.solid:
        on = [f for f in fs if f["side"] == side]
        worst = min(f["cover"] for f in on)
        check(f"solid {side}", worst >= 0.99, worst, ">= 0.99 on every cell")
    for k, pp in enumerate(p.passages):
        a = AXIS[pp["axis"]]
        ua = 2 if a == 0 else 0
        g = np.linspace(0.05, 0.95, 7)
        us = pp["u"][0] + g * (pp["u"][1] - pp["u"][0])
        ys = pp["y"][0] + g * (pp["y"][1] - pp["y"][0])
        uu, yy = np.meshgrid(us, ys)
        origins = np.zeros((uu.size, 3))
        origins[:, ua], origins[:, 1], origins[:, a] = uu.ravel(), yy.ravel(), -1.0
        direction = np.zeros(3)
        direction[a] = 1.0
        # Clear from 1 m outside the -side face to `depth` into the piece (default: through it).
        depth = pp.get("depth", p.cells[a] * C)
        blocked = int((ray_hits(tris, origins, direction) < 1.0 + depth).sum())
        check(f"passage {k} clear", blocked == 0, blocked, "0 rays blocked")
    for k, wk in enumerate(p.walk):
        g = np.linspace(0.1, 0.9, 6)
        xs = wk["x"][0] + g * (wk["x"][1] - wk["x"][0])
        zs = wk["z"][0] + g * (wk["z"][1] - wk["z"][0])
        xx, zz = np.meshgrid(xs, zs)
        expect = wk["y"][0] + (zz.ravel() - wk["z"][0]) / (wk["z"][1] - wk["z"][0]) * (wk["y"][1] - wk["y"][0])
        top = h * C + 5.0
        origins = np.stack([xx.ravel(), np.full(xx.size, top), zz.ravel()], 1)
        got = top - ray_hits(tris, origins, (0, -1, 0))
        tol = 0.21 if wk["y"][0] != wk["y"][1] else 0.02
        worst = float(np.nanmax(np.abs(got - expect)))
        check(f"walk {k} at its height", worst <= tol, round(worst, 3), f"<= {tol} m")
        up = np.stack([xx.ravel(), got + 0.01, zz.ravel()], 1)
        room = float(ray_hits(tris, up, (0, 1, 0)).min())
        check(f"walk {k} headroom", room >= HEADROOM, round(room, 2) if np.isfinite(room) else "open", f">= {HEADROOM} m")
    check("tris", len(tris) <= 2000, len(tris), "<= 2000")
    return checks, fs


# --- placing pieces on a grid ---------------------------------------------------------------

def turn_matrix(t: int) -> np.ndarray:
    """Bevy's `Quat::from_rotation_y(t * pi/2)` as a 3x3 matrix (+Z goes to +X at t = 1)."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][t % 4]
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def place(cells, C: float, at, t: int):
    """Rotation and offset that place a piece with `cells` at grid cell `at` with turn `t`:
    world = R @ local + offset. The piece's cells then start at `at`."""
    R = turn_matrix(t)
    w, h, d = cells
    corners = np.array([[x, 0, z] for x in (0, w * C) for z in (0, d * C)], float)
    rot = corners @ R.T
    offset = np.asarray(at, float) * C - np.array([rot[:, 0].min(), 0, rot[:, 2].min()])
    return R, offset


def turn_side(side: str, t: int) -> str:
    """Where a piece's `side` points after turn `t`."""
    v = np.zeros(3)
    v[AXIS[side[1]]] = -1 if side[0] == "-" else 1
    r = turn_matrix(t) @ v
    a = int(np.argmax(np.abs(r)))
    return ("-" if r[a] < 0 else "+") + "xyz"[a]


def occupied(cells, at, t):
    """The grid cells a placed piece fills."""
    w, h, d = cells
    R = turn_matrix(t)
    pts = [R @ np.array([i + 0.5, j + 0.5, k + 0.5]) for i in range(w) for j in range(h) for k in range(d)]
    pts = np.array(pts)
    rot_corners = np.array([R @ np.array([x, 0, z]) for x in (0, w) for z in (0, d)])
    shift = np.array([rot_corners[:, 0].min(), 0, rot_corners[:, 2].min()])
    return [tuple(int(v) for v in np.floor(q - shift + np.asarray(at))) for q in pts]


def check_arrangement(pieces: dict[str, Piece], manifest_faces: dict, test: list, C: float) -> list[dict]:
    """No two pieces in one cell; no passage onto a solid face; no two different partial
    profiles meeting."""
    checks = []
    owner = {}
    face_at = {}
    clash = []
    for n, (name, at, t) in enumerate(test):
        p = pieces[name]
        cells = occupied(p.cells, at, t)
        local = [(i, j, k) for i in range(p.cells[0]) for j in range(p.cells[1]) for k in range(p.cells[2])]
        cell_of = dict(zip(local, cells))
        for c in cells:
            if c in owner:
                clash.append(f"{name}#{n} and {test[owner[c]][0]}#{owner[c]} at {c}")
            owner[c] = n
        for f in manifest_faces[name]:
            side = turn_side(f["side"], t)
            face_at[(cell_of[tuple(f["cell"])], side)] = (n, f["kind"])
    checks.append(dict(name="test: one piece per cell", ok=not clash, value=len(clash), limit="0", detail=clash[:5]))
    bad = []
    opposite = {"-": "+", "+": "-"}
    for (c, side), (n, kind) in face_at.items():
        a = AXIS[side[1]]
        nb = list(c)
        nb[a] += -1 if side[0] == "-" else 1
        other = face_at.get((tuple(nb), opposite[side[0]] + side[1]))
        if other is None or other[0] == n:
            continue
        k2 = other[1]
        if "open" in (kind, k2):
            continue
        if "passage" in (kind, k2) and "solid" in (kind, k2):
            bad.append(f"passage onto solid at {c} {side}")
        elif kind != k2 and "solid" not in (kind, k2) and "passage" not in (kind, k2):
            bad.append(f"{kind} meets {k2} at {c} {side}")
    checks.append(dict(name="test: faces meet", ok=not bad, value=len(bad), limit="0", detail=sorted(set(bad))[:8]))
    return checks


def merged(pieces: dict[str, Piece], test: list, C: float, gap: float = 0.0):
    """One mesh of the test arrangement (or, with `test=None`, every piece in a row)."""
    out = Mesh()
    if test is None:
        x = 0.0
        test_list = []
        for name, p in pieces.items():
            test_list.append((name, x, 0))
            x += p.cells[0] * C + gap
    for item in (test if test is not None else test_list):
        name, at, t = item
        p = pieces[name]
        if test is None:
            R, offset = np.eye(3), np.array([at, 0.0, 0.0])
        else:
            R, offset = place(p.cells, C, at, t)
        for mat, part in p.mesh.parts.items():
            pos = np.asarray(part["positions"]) @ R.T + offset
            nrm = np.asarray(part["normals"]) @ R.T
            dst = out.parts.setdefault(mat, {"positions": [], "normals": [], "uvs": [], "indices": []})
            base = len(dst["positions"])
            dst["positions"] += pos.tolist()
            dst["normals"] += nrm.tolist()
            dst["uvs"] += part["uvs"]
            dst["indices"] += [i + base for i in part["indices"]]
    return out


# --- files ----------------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def write_piece(path, mesh: Mesh, colors: dict):
    """One piece as a self-contained `.gltf`, one primitive per material."""
    from common.gltf_io import write_meshes
    prims = []
    for mat, prim in mesh.arrays():
        prim["material"] = {"name": mat, "color": srgb_to_linear(colors[mat]).round(4).tolist(), "roughness": 0.9}
        prims.append(prim)
    write_meshes(path, [(path.stem, prims)])


def read_piece(path) -> Mesh:
    """Read back a `.gltf` written by `write_piece` (or `common.gltf_io.write_meshes`)."""
    import base64
    import json
    g = json.loads(path.read_text())
    buf = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])

    def acc(i):
        a = g["accessors"][i]
        v = g["bufferViews"][a["bufferView"]]
        dt = np.float32 if a["componentType"] == 5126 else np.uint32
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.frombuffer(buf, dt, a["count"] * n, v["byteOffset"]).reshape(a["count"], n) if n > 1 else \
            np.frombuffer(buf, dt, a["count"], v["byteOffset"])

    m = Mesh()
    mats = g.get("materials", [])
    for mesh in g["meshes"]:
        for prim in mesh["primitives"]:
            name = mats[prim["material"]]["name"] if "material" in prim else "default"
            at = prim["attributes"]
            m.parts[name] = {"positions": acc(at["POSITION"]).tolist(), "normals": acc(at["NORMAL"]).tolist(),
                             "uvs": acc(at["TEXCOORD_0"]).tolist() if "TEXCOORD_0" in at else [],
                             "indices": acc(prim["indices"]).tolist()}
    return m


def manifest_entry(p: Piece, fs: list[dict], parts: list[str]) -> dict:
    return dict(file=f"{p.name}.gltf", about=p.about, cells=list(p.cells), below=p.below, parts=parts,
                faces=[{k: f[k] for k in ("cell", "side", "kind")} for f in fs],
                walk=p.walk, passages=p.passages)


def piece_meta(p: Piece) -> dict:
    """What `build` hands `frame` besides the mesh."""
    return dict(cells=list(p.cells), about=p.about, below=p.below, solid=list(p.solid), walk=p.walk,
                passages=p.passages, profile=p.profile, parts=list(p.mesh.parts))


def piece_from(name: str, meta: dict, mesh: Mesh) -> Piece:
    return Piece(name, tuple(meta["cells"]), meta["about"], meta["below"], tuple(meta["solid"]),
                 meta["walk"], meta["passages"], meta["profile"], mesh)
