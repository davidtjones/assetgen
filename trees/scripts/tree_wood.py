"""Thick wood as one surface.

Tubes laid through each other never look grown: nothing in the mesh knows that
two of them meet. Here the trunk, the limbs and the roots are instead turned
into one volume and the mesh is the skin of that volume, so a limb leaves the
trunk through a collar and a root melts into the foot.

1. Each branch becomes a run of rounded cones along a smooth curve through its
   points, a distance field: how far a place is from the wood.
2. Branches are merged with a smooth union, which fills the corner between two
   of them with a fillet as wide as the thinner one is thick.
3. The field's zero surface is meshed (surface nets), dense and even.
4. Blender's decimate brings it down to a triangle budget.
5. Bark coordinates come from the skeleton: each triangle belongs to the branch
   it is nearest, and takes "around" and "along" from it. Where two branches
   meet the bark changes direction along a seam.

Twigs are left as tubes (`tree_core._tube`): their joints are too small to see.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline

VOXEL = 0.025           # metres; the field's grid
THIN = 0.035            # nothing in the volume is thinner than this radius
FAR = 10.0
DECIMATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decimate_blender.py")


@dataclass
class Limb:
    """A branch resampled along a smooth curve, with a frame to measure round it."""
    pts: np.ndarray         # n x 3
    radii: np.ndarray       # n
    sway: np.ndarray        # n
    phase: float            # which flutter the branch moves with
    run: np.ndarray         # n, metres from its start
    side: np.ndarray        # n x 3, unit, across the branch, not twisting along it
    wraps: int              # times the bark goes round
    blend: float            # fillet where it joins the rest, metres


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


def limb(branch, bark_width: float, flare=None, blend=None) -> Limb:
    k = len(branch.pts)
    steps = np.linalg.norm(np.diff(branch.pts, axis=0), axis=1)
    n = max(k, int(steps.sum() / 0.12) + 2)
    at = np.linspace(0, k - 1, n)
    knots = np.arange(k)
    pts = CubicSpline(knots, branch.pts, axis=0)(at) if k > 2 else np.stack(
        [np.interp(at, knots, branch.pts[:, c]) for c in range(3)], 1)
    radii = np.interp(at, knots, branch.radii)
    wraps = max(1, int(round(2 * math.pi * branch.radii[0] / bark_width)))
    if flare:
        amount, trunk = flare
        radii = radii * (1 + (amount - 1) * np.exp(-np.maximum(pts[:, 1], 0) / (trunk * 2.2)))
    run = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if getattr(branch, "forks", False):
        # It ends in a split: draw the end in to the size of what leaves it,
        # or its rounded end shows between them as a knob.
        left = (run[-1] - run) / max(radii[-1] * 1.5, 1e-6)
        radii = radii * (0.6 + 0.4 * np.clip(left, 0, 1))
    radii = np.maximum(radii, THIN)
    d0 = _unit(pts[1] - pts[0])
    side = _unit(np.cross(d0, [0.0, 0.0, 1.0]) if abs(d0[2]) < 0.9 else np.cross(d0, [1.0, 0.0, 0.0]))
    sides = []
    for i in range(n):
        d = _unit(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
        side = _unit(side - d * np.dot(side, d))
        sides.append(side)
    return Limb(pts, radii, np.interp(at, knots, branch.sway), float(branch.phase), run, np.array(sides), wraps,
                min(1.5 * float(branch.radii[0]), 0.45) if blend is None else blend)


# --- the volume ------------------------------------------------------------------------

def _field(limbs: list) -> tuple[np.ndarray, np.ndarray]:
    """The distance field of all the limbs on a grid, and the grid's low corner."""
    reach = max(l.blend for l in limbs) + 3 * VOXEL
    lo = np.min([l.pts.min(0) - l.radii.max() for l in limbs], 0) - reach
    hi = np.max([l.pts.max(0) + l.radii.max() for l in limbs], 0) + reach
    lo[1] = max(lo[1], -0.45)                 # open underground: nobody sees it
    shape = np.ceil((hi - lo) / VOXEL).astype(int) + 1
    grid = np.full(shape, FAR, np.float32)
    for l in limbs:
        pad = l.blend + 3 * VOXEL
        a = np.maximum(np.floor((l.pts.min(0) - l.radii.max() - pad - lo) / VOXEL).astype(int), 0)
        b = np.minimum(np.ceil((l.pts.max(0) + l.radii.max() + pad - lo) / VOXEL).astype(int) + 1, shape)
        if (b <= a).any():
            continue
        own = np.full(b - a, FAR, np.float32)
        for i in range(len(l.pts) - 1):
            p0, p1, r0, r1 = l.pts[i], l.pts[i + 1], l.radii[i], l.radii[i + 1]
            r = max(r0, r1) + pad
            sa = np.maximum(np.floor((np.minimum(p0, p1) - r - lo) / VOXEL).astype(int), a)
            sb = np.minimum(np.ceil((np.maximum(p0, p1) + r - lo) / VOXEL).astype(int) + 1, b)
            if (sb <= sa).any():
                continue
            x, y, z = (np.arange(sa[c], sb[c], dtype=np.float32) * VOXEL + (lo[c] - p0[c]) for c in range(3))
            x, y, z = x[:, None, None], y[None, :, None], z[None, None, :]
            ba = (p1 - p0).astype(np.float32)
            h = np.clip((x * ba[0] + y * ba[1] + z * ba[2]) / max(float(ba @ ba), 1e-12), 0, 1)
            d = np.sqrt((x - ba[0] * h) ** 2 + (y - ba[1] * h) ** 2 + (z - ba[2] * h) ** 2) - (r0 + (r1 - r0) * h)
            box = tuple(slice(sa[c] - a[c], sb[c] - a[c]) for c in range(3))
            np.minimum(own[box], d, out=own[box])
        box = tuple(slice(a[c], b[c]) for c in range(3))
        rest = grid[box]
        # Smooth union: where the two are within `blend` of each other, less than either.
        h = np.clip(0.5 + 0.5 * (own - rest) / l.blend, 0, 1)
        grid[box] = own + (rest - own) * h - l.blend * h * (1 - h)
    return grid, lo


def _nets(grid: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface nets: one vertex in each cell the surface passes through, one
    quad across each grid edge it cuts."""
    inside = grid < 0
    cells = np.array(grid.shape) - 1
    ids, places, quads, outward = [], [], [], []
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(0, -1), slice(1, None)
        edge = np.argwhere(inside[tuple(low)] != inside[tuple(high)])
        if not len(edge):
            continue
        end = edge.copy()
        end[:, axis] += 1
        ga, gb = grid[tuple(edge.T)], grid[tuple(end.T)]
        place = edge.astype(np.float32)
        place[:, axis] += ga / (ga - gb)
        o1, o2 = [c for c in range(3) if c != axis]
        ring, whole = [], np.ones(len(edge), bool)
        for d1, d2 in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
            cell = edge.copy()
            cell[:, o1] += d1
            cell[:, o2] += d2
            ok = ((cell >= 0) & (cell < cells)).all(1)
            cid = np.ravel_multi_index(np.clip(cell, 0, cells - 1).T, cells)
            ids.append(cid[ok])
            places.append(place[ok])
            ring.append(cid)
            whole &= ok
        quads.append(np.stack(ring, 1)[whole])
        sign = np.where(ga[whole] < 0, 1.0, -1.0)
        out = np.zeros((int(whole.sum()), 3), np.float32)
        out[:, axis] = sign
        outward.append(out)
    ids, places = np.concatenate(ids), np.concatenate(places)
    cell_ids, inverse = np.unique(ids, return_inverse=True)
    count = np.bincount(inverse)
    verts = np.stack([np.bincount(inverse, places[:, c]) / count for c in range(3)], 1)
    quads = np.searchsorted(cell_ids, np.concatenate(quads))
    outward = np.concatenate(outward)
    # Wind each quad so it faces out of the wood.
    v = verts[quads]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    wrong = (normal * outward).sum(1) < 0
    quads[wrong] = quads[wrong][:, ::-1]
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return verts * VOXEL + lo, faces


def _decimate(verts, faces, triangles: int):
    """Blender's collapse decimation down to about `triangles`."""
    if len(faces) <= triangles:
        return verts, faces
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.obj"), os.path.join(tmp, "out.obj")
        with open(src, "w") as f:
            np.savetxt(f, verts, fmt="v %.5f %.5f %.5f")
            np.savetxt(f, faces + 1, fmt="f %d %d %d")
        run = subprocess.run(["blender", "-b", "--python", DECIMATE, "--", src, dst, str(triangles)],
                             capture_output=True, text=True)
        if not os.path.exists(dst):
            raise RuntimeError("blender decimate failed:\n" + run.stdout[-2000:] + run.stderr[-2000:])
        v, f = [], []
        for line in open(dst):
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("f "):
                f.append([part.split("/")[0] for part in line.split()[1:4]])
        return np.array(v, float), np.array(f, int) - 1


# --- bark coordinates --------------------------------------------------------------------

def _project(l: Limb, p: np.ndarray):
    """Each point as seen from one limb: distance to its surface, metres along
    it, angle round it (turns), the direction "round" points in there, sway."""
    best = np.full(len(p), np.inf)
    seg = np.zeros(len(p), int)
    frac = np.zeros(len(p))
    for i in range(len(l.pts) - 1):
        ba = l.pts[i + 1] - l.pts[i]
        pa = p - l.pts[i]
        h = np.clip(pa @ ba / max(float(ba @ ba), 1e-12), 0, 1)
        d = np.linalg.norm(pa - np.outer(h, ba), axis=1) - (l.radii[i] + (l.radii[i + 1] - l.radii[i]) * h)
        closer = d < best
        best[closer], seg[closer], frac[closer] = d[closer], i, h[closer]
    a, b = l.pts[seg], l.pts[seg + 1]
    along = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-9)
    centre = a + (b - a) * frac[:, None]
    side = l.side[seg] + (l.side[seg + 1] - l.side[seg]) * frac[:, None]
    side -= along * (side * along).sum(1, keepdims=True)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    other = np.cross(along, side)
    out = p - centre
    out -= along * (out * along).sum(1, keepdims=True)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    turn = np.arctan2((out * other).sum(1), (out * side).sum(1)) / (2 * math.pi)
    metres = l.run[seg] + (l.run[seg + 1] - l.run[seg]) * frac
    sway = l.sway[seg] + (l.sway[seg + 1] - l.sway[seg]) * frac
    return best, metres, turn, np.cross(along, out), sway


class Wood:
    """The dense surface of a skeleton's thick wood, to be cut down to size."""

    def __init__(self, limbs: list):
        self.limbs = limbs
        self.grid, self.lo = _field(limbs)
        self.verts, self.faces = _nets(self.grid, self.lo)

    def mesh(self, triangles: int, bark_height: float):
        """`(positions, normals, tangents, uvs, uv1, indices)` at about `triangles`."""
        verts, faces = _decimate(self.verts, self.faces, triangles)
        # Normals from the field, which is smoother than the triangles.
        at = (verts - self.lo) / VOXEL
        grad = np.zeros_like(verts)
        for c in range(3):
            step = np.zeros(3)
            step[c] = 1.0
            grad[:, c] = (ndimage.map_coordinates(self.grid, (at + step).T, order=1, mode="nearest")
                          - ndimage.map_coordinates(self.grid, (at - step).T, order=1, mode="nearest"))
        normals = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)

        # Sway and phase by the nearest limb, per place, so the surface cannot
        # tear in wind and moves with the tubes that grow out of it.
        near = np.full(len(verts), np.inf)
        sway = np.zeros(len(verts))
        phase = np.zeros(len(verts))
        for l in self.limbs:
            d, _, _, _, s = _project(l, verts)
            closer = d < near
            near[closer], sway[closer], phase[closer] = d[closer], s[closer], l.phase

        # Bark by the limb each triangle is nearest.
        centres = verts[faces].mean(1)
        near = np.full(len(faces), np.inf)
        owner = np.zeros(len(faces), int)
        for k, l in enumerate(self.limbs):
            d = _project(l, centres)[0]
            closer = d < near
            near[closer], owner[closer] = d[closer], k
        keys, uvs, tangents, corners = [], [], [], []
        for k, l in enumerate(self.limbs):
            mine = faces[owner == k]
            if not len(mine):
                continue
            flat = mine.reshape(-1)
            _, metres, turn, round_dir, _ = _project(l, verts[flat])
            turn = turn.reshape(-1, 3)
            shift = -np.round(turn - turn[:, :1])           # keep a triangle on one side of the seam
            u = (turn + shift) * l.wraps
            keys.append(np.stack([flat, np.full(len(flat), k), shift.reshape(-1).astype(int)], 1))
            uvs.append(np.stack([u.reshape(-1), metres / bark_height], 1))
            tangents.append(round_dir)
            corners.append(flat)
        keys, uvs, tangents, corners = (np.concatenate(x) for x in (keys, uvs, tangents, corners))
        _, first, index = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        src = corners[first]
        nrm = normals[src]
        tan = tangents[first]
        tan -= nrm * (tan * nrm).sum(1, keepdims=True)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        tan = np.concatenate([tan, -np.ones((len(tan), 1))], 1)
        uv1 = np.stack([sway[src], phase[src]], 1)
        return verts[src], nrm, tan, uvs[first], uv1, index.reshape(-1)
