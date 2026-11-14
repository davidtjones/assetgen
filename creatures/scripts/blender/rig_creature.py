"""Rig and animate a generated quadruped. Runs INSIDE Blender (needs `bpy`).

    blender -b --factory-startup --python-exit-code 1 --python rig_creature.py -- \
        mesh.glb spec.json creature.glb rig.json

1. **Frame.** Join the mesh, turn it so the body's long axis is Blender Y with the head at -Y
   (glTF +Z, the glTF "forward"), scale it to the spec's `height`, stand it on z = 0 and
   centre it.
2. **Fit a skeleton from the shape.** Horizontal slices from the ground up are split into
   connected blobs on a small grid; the highest slice that still has four separate blobs is
   the belly. Those blobs, followed down slice by slice, are the legs: their centre lines
   give hip, knee and ankle. The centroid of each slice across the body above the belly is
   the spine; the part ahead of the front legs is neck and head; the part behind the hind
   legs is the tail. Bones: `root` (on the ground, does not deform), `hips`, `spine_1..2`,
   `chest`, `neck`, `head`, `tail_1..n`, and per leg (`fl`, `fr`, `hl`, `hr`)
   `leg_xx_1` (upper), `leg_xx_2` (lower), `leg_xx_3` (foot). Every bone's local X is the
   creature's left-right axis, so pitch is rotation about X.
3. **Skin** with Blender's heat weights. If heat fails on some vertices (it does on open or
   self-intersecting meshes) the weights are solved on a voxel-remeshed copy and transferred
   back. At most four influences per vertex, normalised.
4. **Animate** two in-place loops at 30 fps, `idle` and `walk`. The walk is a lateral-sequence
   gait (hind left, fore left, hind right, fore right a quarter cycle apart): each ankle
   follows a stance-then-swing path in the body frame, and the legs are posed by an analytic
   two-bone IK each frame, so a planted foot moves backwards at exactly `stride / cycle` and
   the game can cancel it by moving the creature forwards at that speed.
5. **Export** glTF with the skin and both clips (one NLA track each), and `rig.json` with the
   fitted joints, sizes and clip lengths for the check stage and the manifest.
"""

import itertools
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

FPS = 30
DUTY = 0.65          # fraction of the walk cycle each foot is on the ground
LEGS = ["fl", "fr", "hl", "hr"]
PHASE = {"hl": 0.0, "fl": 0.25, "hr": 0.5, "fr": 0.75}   # lateral-sequence walk


# -- scene and mesh --------------------------------------------------------------------------

def load_mesh(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not objs:
        raise SystemExit(f"no mesh in {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.parent = None
    for o in list(bpy.data.objects):
        if o is not obj:
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.name = obj.data.name = "body"
    return obj


def verts(obj):
    a = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def surface_points(obj, n=300000, seed=0):
    """Points spread evenly over the surface, for shape analysis. A game mesh has too few
    vertices for a thin slice through a leg to find it; its triangles do not."""
    me = obj.data
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    v = verts(obj)
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[pick] + (b[pick] - a[pick]) * r1[:, None] + (c[pick] - a[pick]) * r2[:, None]


def frame_mesh(obj, spec):
    """Long axis -> Y, head -> -Y, height -> spec, feet on z=0, centred. Returns the transform."""
    v = verts(obj)
    xy = v[:, :2] - v[:, :2].mean(0)
    w, vecs = np.linalg.eigh(np.cov(xy.T))
    long_axis = vecs[:, np.argmax(w)]
    angle = math.atan2(long_axis[0], long_axis[1])      # rotate long axis onto +Y
    m = Matrix.Rotation(angle, 4, "Z")
    obj.data.transform(m)
    v = surface_points(obj, 100000)
    flip = head_is_positive_y(v) != bool(spec.get("flip", False))
    if flip:
        obj.data.transform(Matrix.Rotation(math.pi, 4, "Z"))
    v = verts(obj)
    s = spec["height"] / (v[:, 2].max() - v[:, 2].min())
    lo, hi = v.min(0), v.max(0)
    centre = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    obj.data.transform(Matrix.Scale(s, 4) @ Matrix.Translation(-centre))
    obj.data.update()
    return {"turn_deg": math.degrees(angle) + (180 if flip else 0), "scale": s}


def head_is_positive_y(v):
    """Guess which end is the head. The concept is drawn facing left and TRELLIS keeps the
    picture's layout, but rather than trust a convention: the head end is the one whose
    underside stands clear of the ground (a snout, against a rump over the hind legs or a tail
    lying on the ground), measured over the outer 12% of the length at each end."""
    y = v[:, 1]
    lo, hi = y.min(), y.max()
    span = hi - lo
    h = v[:, 2].max() - v[:, 2].min()
    z0 = v[:, 2].min()
    def clearance(sel):
        return np.percentile(v[sel, 2] - z0, 5) / h
    neg = clearance(y < lo + 0.12 * span)
    pos = clearance(y > hi - 0.12 * span)
    return pos > neg


# -- shape analysis --------------------------------------------------------------------------

def blobs(points, cell):
    """Connected components of 2D points on a grid of `cell`; returns a list of index arrays."""
    if len(points) == 0:
        return []
    ij = np.floor((points - points.min(0)) / cell).astype(int)
    shape = ij.max(0) + 1
    grid = -np.ones(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    occupied[ij[:, 0], ij[:, 1]] = True
    label = 0
    for start in zip(*np.nonzero(occupied)):
        if grid[start] >= 0:
            continue
        stack = [start]
        grid[start] = label
        while stack:
            a, b = stack.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    na, nb = a + da, b + db
                    if 0 <= na < shape[0] and 0 <= nb < shape[1] and occupied[na, nb] and grid[na, nb] < 0:
                        grid[na, nb] = label
                        stack.append((na, nb))
        label += 1
    lab = grid[ij[:, 0], ij[:, 1]]
    return [np.nonzero(lab == k)[0] for k in range(label)]


def find_legs(v, h):
    """Belly height and, per leg, its centre line [(z, x, y), ...] from the ground up.

    Slices go up from the ground. The first slice with four blobs names the legs (front and
    hind by Y, left and right by X). Each later slice must offer every leg a blob on its own
    side and its own half of the body, within a short step of where it was; the first slice
    that cannot is the belly. Blobs that are not legs (a tail tip, a sagging chest) cannot be
    taken for one that way."""
    cell = h / 40
    slab = h / 50
    levels = np.arange(slab, 0.8 * h, slab)
    mid_x = float(np.median(v[:, 0]))
    tracks = None
    belly = None
    for z in levels:
        sel = np.nonzero(np.abs(v[:, 2] - z) < slab / 2)[0]
        parts = [p for p in blobs(v[sel, :2], cell) if len(p) >= max(10, len(sel) // 200)]
        cents = [v[sel[p], :2].mean(0) for p in parts]
        if tracks is None:
            if len(cents) >= 4:
                big = sorted(range(len(parts)), key=lambda i: len(parts[i]), reverse=True)[:4]
                named = assign_legs([cents[i] for i in big])
                tracks = {k: [(z, *c)] for k, c in named.items()}
                mid_y = float(np.mean([c[1] for c in named.values()]))
            continue
        step = 4 * slab + 3 * cell

        def allowed(k, c):
            left, front = k[1] == "l", k[0] == "f"
            last = np.array(tracks[k][-1][1:])
            return ((c[0] > mid_x) == left and (c[1] < mid_y) == front
                    and np.linalg.norm(c - last) < step)
        options = {k: [i for i, c in enumerate(cents) if allowed(k, c)] for k in LEGS}
        picks = [p for p in itertools.product(*(options[k] for k in LEGS)) if len(set(p)) == 4]
        if not picks:
            belly = z
            break
        best = min(picks, key=lambda p: sum(np.linalg.norm(cents[p[i]] - np.array(tracks[k][-1][1:]))
                                            for i, k in enumerate(LEGS)))
        for i, k in enumerate(LEGS):
            tracks[k].append((z, *cents[best[i]]))
    if tracks is None:
        raise SystemExit("could not find four separate legs in any slice; the mesh's legs are fused")
    if belly is None:
        belly = levels[-1]
    return belly, tracks


def assign_legs(cents):
    """Name four foot blobs. Head is at -Y, so front legs have the smaller Y; the creature's
    left is +X when it faces -Y."""
    c = sorted(cents, key=lambda p: p[1])
    front, hind = sorted(c[:2], key=lambda p: p[0]), sorted(c[2:], key=lambda p: p[0])
    return {"fr": front[0], "fl": front[1], "hr": hind[0], "hl": hind[1]}


def spine_curve(v, belly, y0, y1, n):
    """Centroids of `n + 1` cross-sections across y0..y1, using only body above the belly."""
    out = []
    body = v[v[:, 2] > belly]
    step = abs(y1 - y0) / max(n, 1)
    for t in np.linspace(0, 1, n + 1):
        y = y0 + (y1 - y0) * t
        sel = body[np.abs(body[:, 1] - y) < max(step, 0.02) / 2 + 0.01]
        if len(sel) == 0:
            sel = v[np.abs(v[:, 1] - y) < 0.03]
        out.append(Vector((0.0, y, float((sel[:, 2].min() + sel[:, 2].max()) / 2))))
    return out


def end_curve(v, y_from, y_to, n, belly):
    """Centre line from y_from to y_to (an end of the body: head or tail), using every vertex
    in each cross-section, `n + 1` points. Stops short of the very tip."""
    pts = []
    for t in np.linspace(0, 0.92, n + 1):
        y = y_from + (y_to - y_from) * t
        sel = v[np.abs(v[:, 1] - y) < 0.02]
        if t == 0:
            sel = sel[sel[:, 2] > belly] if (sel[:, 2] > belly).any() else sel
        if len(sel) == 0:
            pts.append(pts[-1].copy())
            continue
        pts.append(Vector((float(np.median(sel[:, 0])), y, float((sel[:, 2].min() + sel[:, 2].max()) / 2))))
    return pts


def fit_skeleton(v, spec):
    h = spec["height"]
    belly, tracks = find_legs(v, h)
    joints = {}
    leg_info = {}
    for k in LEGS:
        t = np.array(tracks[k])                        # (z, x, y) from the bottom up
        def at(z, t=t):
            i = np.argmin(np.abs(t[:, 0] - z))
            return t[i, 1], t[i, 2]
        x_a, y_a = at(0.10 * belly)
        x_h, y_h = at(0.95 * belly)
        hip = Vector((x_h, y_h, belly + 0.25 * (h - belly)))
        ankle = Vector((x_a, y_a, 0.10 * belly))
        toe = Vector((x_a, y_a - 0.12 * belly, 0.0))
        joints[k] = [hip, None, ankle, toe, at]
        leg_info[k] = {"foot": [x_a, y_a], "track_points": len(t)}
    y = v[:, 1]
    front_y = (joints["fl"][0].y + joints["fr"][0].y) / 2
    hind_y = (joints["hl"][0].y + joints["hr"][0].y) / 2
    spine = spine_curve(v, belly, hind_y, front_y, 3)          # hips, spine_1, spine_2, chest
    head = end_curve(v, front_y, y.min(), 2, belly)            # chest -> neck -> head tip
    n_tail = int(spec.get("tail_bones", 0))
    tail = end_curve(v, hind_y, y.max(), max(n_tail, 1), belly) if n_tail else []
    for k in LEGS:
        hip, _, ankle, toe, at = joints[k]
        # hips sit on the spine's height at their end, so the leg roots are inside the body
        anchor = spine[0] if k[0] == "h" else spine[-1]
        hip.z = min(hip.z, anchor.z)
        # the knee halfway between hip and ankle: equal bones share a bend, where a long
        # thigh over a short shin folds the shin flat
        z_k = (hip.z + ankle.z) / 2
        x_k, y_k = at(min(z_k, 0.95 * belly))
        # bend it back a little (+Y) so IK has a direction. Fore legs bend back too: with one
        # joint standing for elbow and wrist, a forward bend folded the inside of the joint
        # flat on the boar (edges at 7% of rest length); backwards halved the worst stretch.
        bend = 0.06 * belly
        joints[k] = (hip, Vector((x_k, y_k + bend, z_k)), ankle, toe)
    return {"belly": belly, "legs": joints, "spine": spine, "head": head, "tail": tail, "leg_info": leg_info,
            "length": float(y.max() - y.min())}


# -- armature --------------------------------------------------------------------------------

def make_armature(fit):
    arm_data = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, deform=True):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0, 0, 0.01))
        b.parent = eb[parent] if parent else None
        b.use_connect = False
        b.use_deform = deform
        # local X = the creature's left-right axis, so pitch is rotation about X
        y_axis = (b.tail - b.head).normalized()
        x_axis = Vector((1, 0, 0)) - Vector((1, 0, 0)).dot(y_axis) * y_axis
        if x_axis.length < 1e-3:
            x_axis = Vector((0, 1, 0))
        b.align_roll(x_axis.normalized().cross(y_axis))
        return b

    s, hd = fit["spine"], fit["head"]
    bone("root", (0, 0, 0), (0, -0.1 * fit["length"], 0), deform=False)
    names = ["hips", "spine_1", "spine_2", "chest"]
    bone("hips", s[0], s[1], "root")
    bone("spine_1", s[1], s[2], "hips")
    bone("spine_2", s[2], s[3], "spine_1")
    bone("chest", s[3], s[3] + (s[3] - s[2]) * 0.5, "spine_2")
    bone("neck", s[3], hd[1], "chest")
    bone("head", hd[1], hd[2], "neck")
    names += ["neck", "head"]
    parent = "hips"
    tail_names = []
    for i in range(len(fit["tail"]) - 1):
        n = f"tail_{i + 1}"
        bone(n, fit["tail"][i], fit["tail"][i + 1], parent)
        tail_names.append(n)
        parent = n
    for k in LEGS:
        hip, knee, ankle, toe = fit["legs"][k]
        p = "hips" if k[0] == "h" else "chest"
        bone(f"leg_{k}_1", hip, knee, p)
        bone(f"leg_{k}_2", knee, ankle, f"leg_{k}_1")
        bone(f"leg_{k}_3", ankle, toe, f"leg_{k}_2")
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm, names, tail_names


# -- skinning --------------------------------------------------------------------------------

def unweighted(obj):
    return sum(1 for vx in obj.data.vertices if not any(g.weight > 1e-4 for g in vx.groups))


def skin(obj, arm, h, belly):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    missing = unweighted(obj)
    method = "heat"
    if missing > 0.001 * len(obj.data.vertices):
        method = "heat on voxel proxy"
        print(f"[rig] heat weights left {missing} vertices unweighted; solving on a voxel proxy")
        for g in list(obj.vertex_groups):
            obj.vertex_groups.remove(g)
        proxy = obj.copy()
        proxy.data = obj.data.copy()
        proxy.modifiers.clear()
        bpy.context.scene.collection.objects.link(proxy)
        proxy.parent = None
        bpy.ops.object.select_all(action="DESELECT")
        proxy.select_set(True)
        bpy.context.view_layer.objects.active = proxy
        rm = proxy.modifiers.new("remesh", "REMESH")
        rm.mode, rm.voxel_size = "VOXEL", h / 80
        bpy.ops.object.modifier_apply(modifier="remesh")
        proxy.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        for g in proxy.vertex_groups:
            obj.vertex_groups.new(name=g.name)
        dt = obj.modifiers.new("weights", "DATA_TRANSFER")
        dt.object = proxy
        dt.use_vert_data = True
        dt.data_types_verts = {"VGROUP_WEIGHTS"}
        dt.vert_mapping = "POLYINTERP_NEAREST"
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_move_to_index(modifier="weights", index=0)
        bpy.ops.object.modifier_apply(modifier="weights")
        bpy.data.objects.remove(proxy)
        missing = unweighted(obj)
    confine_legs(obj, arm, belly)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    return {"method": method, "unweighted": missing}


def confine_legs(obj, arm, belly):
    """Heat weights let a leg's bones reach far up the flank, which then stretches when the
    leg swings. Fade every leg bone's weight out between 1.5 and 3 leg radii from the leg's
    own bones, and hand what is taken to the body bones the vertex already had (or the bone
    the leg hangs from). Then smooth every weight a little over the mesh's edges."""
    v = verts(obj)
    groups = {g.name: g.index for g in obj.vertex_groups}
    W = np.zeros((len(v), len(obj.vertex_groups)))
    for vx in obj.data.vertices:
        for g in vx.groups:
            W[vx.index, g.group] = g.weight
    bones = arm.data.bones
    dist = {}
    for k in LEGS:
        d = np.full(len(v), np.inf)
        for i in (1, 2, 3):
            a, b = np.array(bones[f"leg_{k}_{i}"].head_local), np.array(bones[f"leg_{k}_{i}"].tail_local)
            ab = b - a
            t = np.clip(((v - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            d = np.minimum(d, np.linalg.norm(v - (a + t[:, None] * ab), axis=1))
        dist[k] = d
    D = np.stack([dist[k] for k in LEGS], 1)
    nearest = D.argmin(1)
    body = [groups[n] for n in groups if not n.startswith("leg_")]
    for j, k in enumerate(LEGS):
        d = dist[k]
        knee_z = bones[f"leg_{k}_2"].head_local.z
        mine = (nearest == j) & (np.abs(v[:, 2] - knee_z) < 0.1 * max(knee_z, 0.05))
        r0 = np.percentile(d[mine], 90) if mine.sum() > 10 else 0.05
        keep = np.clip(1 - (d - 1.5 * r0) / (1.5 * r0), 0, 1)
        # where another leg is nearer, this one lets go (the belly between two legs
        # follows the body, not both legs pulling opposite ways)
        other = np.delete(D, j, axis=1).min(1)
        keep *= np.clip((other - d) / max(r0, 1e-3), 0, 1)
        # above the belly the body moves with the spine: a leg bone pivots high inside the
        # body, and skin it pulls there tears against the chest and the other leg
        belly_z = belly
        hip_z = bones[f"leg_{k}_1"].head_local.z
        top = belly_z + 0.5 * max(hip_z - belly_z, 1e-3)
        keep *= np.clip((top - v[:, 2]) / max(top - belly_z, 1e-3), 0, 1)
        cols = [groups[f"leg_{k}_{i}"] for i in (1, 2, 3) if f"leg_{k}_{i}" in groups]
        # the paw belongs to the foot bone, which IK keeps flat: below the ankle the shin's
        # weight moves to the foot, blending back over the next ankle-height up, so a planted
        # paw neither tilts into the ground nor slides
        if len(cols) == 3:
            ankle_z = max(bones[f"leg_{k}_3"].head_local.z, 1e-3)
            to_foot = np.clip((2 * ankle_z - v[:, 2]) / ankle_z, 0, 1) * (nearest == j)
            moved = W[:, cols[0]] * to_foot + W[:, cols[1]] * to_foot
            W[:, cols[0]] *= 1 - to_foot
            W[:, cols[1]] *= 1 - to_foot
            W[:, cols[2]] += moved
        removed = W[:, cols].sum(1) * (1 - keep)
        W[:, cols] *= keep[:, None]
        have = W[:, body].sum(1)
        anchor = groups["hips" if k[0] == "h" else "chest"]
        share = np.where(have[:, None] > 1e-6, W[:, body] / np.maximum(have[:, None], 1e-9), 0)
        W[:, body] += share * removed[:, None]
        W[have <= 1e-6, anchor] += removed[have <= 1e-6]
    # Smooth: neighbours with very different weights tear apart when the bones move.
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    deg = np.bincount(e.ravel(), minlength=len(v)).astype(float)[:, None]
    for _ in range(6):
        acc = np.zeros_like(W)
        np.add.at(acc, e[:, 0], W[e[:, 1]])
        np.add.at(acc, e[:, 1], W[e[:, 0]])
        W = np.where(deg > 0, 0.5 * W + 0.5 * acc / np.maximum(deg, 1), W)
    for name, gi in groups.items():
        g = obj.vertex_groups[gi]
        idx = np.nonzero(W[:, gi] > 1e-4)[0]
        g.remove(list(range(len(v))))
        for i in idx:
            g.add([int(i)], float(W[i, gi]), "REPLACE")


# -- animation -------------------------------------------------------------------------------

def smoothstep(t):
    return t * t * (3 - 2 * t)


def foot_offset(phase, stride_body, lift):
    """Ankle offset (dy, dz) in the body frame at gait phase 0..1. Stance: on the ground,
    moving back (+Y) at constant speed. Swing: lifted and carried forward."""
    if phase < DUTY:
        return -stride_body / 2 + stride_body * (phase / DUTY), 0.0
    t = (phase - DUTY) / (1 - DUTY)
    return stride_body / 2 - stride_body * smoothstep(t), lift * math.sin(math.pi * t)


class Poser:
    def __init__(self, arm, fit):
        self.arm = arm
        self.pb = arm.pose.bones
        self.rest_ankle = {k: fit["legs"][k][2].copy() for k in LEGS}
        self.lengths = {}
        for k in LEGS:
            b1, b2 = arm.data.bones[f"leg_{k}_1"], arm.data.bones[f"leg_{k}_2"]
            self.lengths[k] = (b1.length, b2.length)
        self.foot_rest = {k: arm.data.bones[f"leg_{k}_3"].matrix_local.copy() for k in LEGS}
        self.prev = {}
        self.keys = {}

    def reset(self):
        for p in self.pb:
            p.rotation_mode = "QUATERNION"
            p.rotation_quaternion = Quaternion()
            p.location = Vector()
        self.prev = {}

    def rot(self, name, x=0.0, y=0.0, z=0.0):
        """Local rotation in radians: x = pitch, y = twist along the bone, z = yaw/side bend."""
        self.pb[name].rotation_quaternion = Matrix.Rotation(x, 4, "X").to_quaternion() @ \
            Matrix.Rotation(z, 4, "Z").to_quaternion() @ Matrix.Rotation(y, 4, "Y").to_quaternion()

    def solve_leg(self, k, target):
        """Two-bone IK in the leg's plane; the foot keeps its rest orientation (flat)."""
        bpy.context.view_layer.update()
        p1, p2, p3 = self.pb[f"leg_{k}_1"], self.pb[f"leg_{k}_2"], self.pb[f"leg_{k}_3"]
        hip = p1.head.copy()
        l1, l2 = self.lengths[k]
        d_vec = target - hip
        d = min(max(d_vec.length, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
        u = d_vec.normalized()
        lateral = Vector((1, 0, 0))
        w = lateral.cross(u).normalized()      # perpendicular to u in the side plane
        if w.y < 0:                            # every knee bends back (+Y), as in the rest pose
            w = -w
        a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
        b = math.sqrt(max(l1 * l1 - a * a, 0.0))
        knee = hip + u * a + w * b
        ankle = hip + u * d

        def frame(head, tail):
            yv = (tail - head).normalized()
            xv = (lateral - lateral.dot(yv) * yv).normalized()
            zv = xv.cross(yv)
            m = Matrix((xv, yv, zv)).transposed().to_4x4()
            m.translation = head
            return m
        p1.matrix = frame(hip, knee)
        bpy.context.view_layer.update()
        p2.matrix = frame(knee, ankle)
        bpy.context.view_layer.update()
        m3 = self.foot_rest[k].copy()
        m3.translation = ankle
        p3.matrix = m3

    def key(self, frame):
        """Record this frame's pose. Nothing is keyed into Blender until the clip is done, so
        no action or NLA track is live while poses are being built (a live one is evaluated
        on every view-layer update and overwrites what was just set)."""
        bpy.context.view_layer.update()
        for p in self.pb:
            q = p.rotation_quaternion.copy()
            if p.name in self.prev and self.prev[p.name].dot(q) < 0:
                q.negate()
            self.prev[p.name] = q.copy()
            path = f'pose.bones["{p.name}"]'
            for i in range(4):
                self.keys.setdefault((f"{path}.rotation_quaternion", i, p.name), []).append((frame, q[i]))
            if p.name == "root" or p.name.startswith("leg_"):
                for i in range(3):
                    self.keys.setdefault((f"{path}.location", i, p.name), []).append((frame, p.location[i]))

    def write(self, action):
        """Write the recorded keys into `action`, which must be assigned to the armature."""
        for (path, index, group), samples in self.keys.items():
            fc = action.fcurve_ensure_for_datablock(self.arm, path, index=index, group_name=group)
            fc.keyframe_points.add(len(samples))
            fc.keyframe_points.foreach_set("co", [c for f, v in samples for c in (f, v)])
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
        self.keys = {}


def animate(arm, fit, spec, tail_names):
    poser = Poser(arm, fit)
    h = spec["height"]
    arm.animation_data_create()
    clips = {}

    def clip(name, seconds, pose_at):
        arm.animation_data.action = None
        for t in arm.animation_data.nla_tracks:
            t.mute = True
        poser.reset()
        n = int(round(seconds * FPS))
        for f in range(n + 1):
            for p in poser.pb:
                p.rotation_quaternion = Quaternion()
                p.location = Vector()
            pose_at((f % n) / n)
            poser.key(f)
        act = bpy.data.actions.new(name)
        arm.animation_data.action = act
        poser.write(act)
        track = arm.animation_data.nla_tracks.new()
        track.name = name
        track.strips.new(name, 0, act)
        arm.animation_data.action = None
        clips[name] = {"frames": n, "seconds": n / FPS}

    walk = spec["walk"]
    stride_body = walk["stride"] * DUTY     # how far a planted foot travels in the body frame
    lift = walk["lift"]
    tau = 2 * math.pi

    root_rest = arm.data.bones["root"].matrix_local.to_3x3()

    def lift_root(dz):
        """Move the whole body up by dz (world). The root bone's local axes are not world axes."""
        poser.pb["root"].location = root_rest.inverted() @ Vector((0, 0, dz))

    def walk_pose(t):
        lift_root(-0.012 * h * math.cos(2 * tau * t))    # bob twice per cycle
        poser.rot("hips", z=0.05 * math.sin(tau * t))
        poser.rot("spine_1", z=-0.025 * math.sin(tau * t))
        poser.rot("chest", z=-0.04 * math.sin(tau * t))
        poser.rot("neck", x=0.04 * math.cos(2 * tau * t))
        poser.rot("head", x=-0.05 * math.cos(2 * tau * t), z=0.03 * math.sin(tau * t))
        for i, n in enumerate(tail_names):
            poser.rot(n, z=0.12 * math.sin(tau * t - 0.6 * (i + 1)), x=0.03 * math.cos(2 * tau * t))
        for k in LEGS:
            dy, dz = foot_offset((t + PHASE[k]) % 1.0, stride_body, lift)
            poser.solve_leg(k, poser.rest_ankle[k] + Vector((0, dy, dz)))

    def idle_pose(t):
        lift_root(-0.006 * h * (1 - math.cos(2 * tau * t)) / 2)   # breathing
        poser.rot("chest", x=0.01 * math.sin(2 * tau * t))
        poser.rot("neck", x=0.06 * math.sin(tau * t), z=0.12 * math.sin(tau * t + 1.0))
        poser.rot("head", x=-0.04 * math.sin(tau * t), z=0.15 * math.sin(tau * t + 1.6))
        for i, n in enumerate(tail_names):
            poser.rot(n, z=0.10 * math.sin(tau * t - 0.5 * (i + 1)))
        for k in LEGS:
            poser.solve_leg(k, poser.rest_ankle[k].copy())

    clip("idle", 4.0, idle_pose)
    clip("walk", walk["cycle"], walk_pose)
    for t in arm.animation_data.nla_tracks:
        t.mute = False
    return clips


# -- main ------------------------------------------------------------------------------------

def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    src, spec_path, out_glb, out_json = argv
    spec = json.load(open(spec_path))
    obj = load_mesh(src)
    placed = frame_mesh(obj, spec)
    fit = fit_skeleton(surface_points(obj), spec)
    arm, spine_names, tail_names = make_armature(fit)
    skinned = skin(obj, arm, spec["height"], fit["belly"])
    scene = bpy.context.scene
    scene.render.fps = FPS
    clips = animate(arm, fit, spec, tail_names)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=out_glb, export_format="GLB", export_yup=True, export_apply=False,
        export_skins=True, export_animations=True, export_animation_mode="NLA_TRACKS",
        export_force_sampling=True, export_frame_step=1, export_def_bones=False,
        export_reset_pose_bones=False, export_optimize_animation_size=False)
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    joints = {b.name: [round(c, 4) for c in b.head_local] for b in arm.data.bones}
    report = {
        "placed": placed, "belly": fit["belly"], "length": fit["length"], "tris": tris,
        "bones": [b.name for b in arm.data.bones], "joints": joints,
        "legs": fit["leg_info"], "skin": skinned, "clips": clips,
        "walk_speed": spec["walk"]["stride"] / spec["walk"]["cycle"],
    }
    json.dump(report, open(out_json, "w"), indent=2)
    print(f"[rig] {out_glb}: {tris} tris, {len(report['bones'])} bones, skin {skinned}, clips {list(clips)}")


main()
