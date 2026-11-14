"""Validate a rigged creature glTF against its spec. Runs INSIDE Blender (needs `bpy`).

    blender -b --factory-startup --python-exit-code 1 --python check_creature.py -- \
        creature.glb spec.json check.json sheet_dir

Re-imports what will ship (not the Blender scene the rig stage built) and measures:

- **mesh**: triangle count against the spec's budget, height against the spec, feet on the
  ground (lowest point near z = 0), number of loose parts.
- **skin**: vertices with no weight, vertices with more than four influences, and leg
  purity: of the vertices low down in each leg's column, the share of weight that belongs
  to that leg's bones. Heat weights bleeding from one leg into its neighbour shows up here.
- **clips**: `idle` and `walk` exist and loop (first and last frame match).
- **deformation**: for every frame of every clip, edge length against rest (ignoring the
  shortest tenth of edges, where a millimetre is a large ratio). The 99.9th percentile and
  the worst edge catch skin tearing and candy-wrapper collapse; the worst edge is only a
  warning (with its clip, frame and place in `worst_edge`), since one edge in thousands
  across every frame is too brittle to fail on.
- **feet**: during the walk, the lowest point of each foot never sinks below the ground,
  each foot is down for part of the cycle, and while down it does not slide: the vertices
  touching the ground on one frame barely move in the world by the next (body-frame speed
  minus the creature's walk speed; median over the cycle).

Writes `check.json` with each check's value, limit and pass flag, and renders the frames for
the contact sheet into `sheet_dir`: four turntable views, the skeleton seen through the
mesh, and eight walk frames from the side.
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

LEGS = ["fl", "fr", "hl", "hr"]


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # merge the vertices glTF splits along UV seams, or every UV island counts as a loose part
    bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    return arm, mesh


def rest_verts(mesh):
    a = np.empty(len(mesh.data.vertices) * 3)
    mesh.data.vertices.foreach_get("co", a)
    return (np.array(mesh.matrix_world) @ np.c_[a.reshape(-1, 3), np.ones(len(a) // 3)].T).T[:, :3]


def posed_verts(mesh):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    m = ev.to_mesh()
    a = np.empty(len(m.vertices) * 3)
    m.vertices.foreach_get("co", a)
    ev.to_mesh_clear()
    return (np.array(mesh.matrix_world) @ np.c_[a.reshape(-1, 3), np.ones(len(a) // 3)].T).T[:, :3]


def edges(mesh):
    e = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
    mesh.data.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def loose_parts(mesh):
    n = len(mesh.data.vertices)
    parent = np.arange(n)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges(mesh):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    sizes = np.bincount(roots, minlength=n)
    return int((sizes > 0).sum()), sizes[sizes > 0]


def weights(mesh):
    """(n_verts, n_groups) dense weight matrix and the group names."""
    names = [g.name for g in mesh.vertex_groups]
    W = np.zeros((len(mesh.data.vertices), len(names)))
    for v in mesh.data.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return W, names


def actions_by_name(arm):
    out = {}
    if arm.animation_data:
        for t in arm.animation_data.nla_tracks:
            for s in t.strips:
                out[t.name] = s.action
    for a in bpy.data.actions:
        out.setdefault(a.name.split("_")[0] if a.name not in ("idle", "walk") else a.name, a)
    return out


def play(arm, action):
    arm.animation_data_create()
    for t in arm.animation_data.nla_tracks:
        t.mute = True
    arm.animation_data.action = action
    if hasattr(arm.animation_data, "action_slot") and action.slots:
        arm.animation_data.action_slot = action.slots[0]
    lo, hi = action.frame_range
    return int(round(lo)), int(round(hi))


# -- rendering -------------------------------------------------------------------------------

def setup_render(size=512):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sh.show_cavity = False
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("w")
    world.color = (0.2, 0.21, 0.23)
    scene.world = world
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    scene.collection.objects.link(cam)
    scene.camera = cam
    ground = bpy.data.meshes.new("ground")
    ground.from_pydata([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], [], [(0, 1, 2, 3)])
    g = bpy.data.objects.new("ground", ground)
    scene.collection.objects.link(g)
    gm = bpy.data.materials.new("ground")
    gm.diffuse_color = (0.35, 0.37, 0.33, 1)
    g.data.materials.append(gm)
    return cam


def shoot(cam, direction, centre, extent, path):
    d = Vector(direction).normalized()
    cam.location = Vector(centre) + d * 10
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = extent * 1.25
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def joint_markers(arm, r):
    mat = bpy.data.materials.new("joint")
    mat.diffuse_color = (1.0, 0.25, 0.1, 1)
    out = []
    for b in arm.data.bones:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=arm.matrix_world @ b.head_local, segments=12, ring_count=6)
        s = bpy.context.active_object
        s.data.materials.append(mat)
        out.append(s)
        bpy.ops.mesh.primitive_cylinder_add(radius=r * 0.35, depth=1, vertices=8)
        c = bpy.context.active_object
        h, t = arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local
        c.location = (h + t) / 2
        c.scale.z = (t - h).length
        c.rotation_euler = (t - h).to_track_quat("Z", "Y").to_euler()
        c.data.materials.append(mat)
        out.append(c)
    return out


# -- main ------------------------------------------------------------------------------------

def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    src, spec_path, out_json, sheet_dir = argv
    os.makedirs(sheet_dir, exist_ok=True)
    spec = json.load(open(spec_path))
    h = spec["height"]
    arm, mesh = load(src)
    checks = []

    def check(name, value, ok, limit, warn=False):
        """`warn` checks are reported but do not fail the creature."""
        checks.append({"name": name, "value": round(float(value), 4) if isinstance(value, (int, float, np.floating, np.integer)) else value,
                       "limit": limit, "pass": bool(ok), "warn": warn})

    # rest pose: no action
    if arm.animation_data:
        arm.animation_data.action = None
        for t in arm.animation_data.nla_tracks:
            t.mute = True
    bpy.context.view_layer.update()
    V0 = rest_verts(mesh)
    tris = sum(len(p.vertices) - 2 for p in mesh.data.polygons)
    check("tris", tris, tris <= spec["tris"] * 1.1, f"<= {int(spec['tris'] * 1.1)}")
    height = V0[:, 2].max() - V0[:, 2].min()
    check("height_m", height, abs(height - h) <= 0.05 * h, f"{h} +-5%")
    check("ground_min_z_m", V0[:, 2].min(), abs(V0[:, 2].min()) <= 0.01 * h, f"|z| <= {0.01 * h:.3f}")
    n_parts, sizes = loose_parts(mesh)
    check("loose_parts", n_parts, n_parts <= 3, "<= 3")

    W, names = weights(mesh)
    unweighted = int((W.sum(1) < 1e-4).sum())
    check("unweighted_verts", unweighted, unweighted == 0, "0")
    over4 = int(((W > 1e-4).sum(1) > 4).sum())
    check("verts_over_4_influences", over4, over4 == 0, "0")

    # leg purity: vertices in the bottom third of each foot column
    joints = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
    purity = {}
    for k in LEGS:
        ankle = joints[f"leg_{k}_3"]
        knee = joints[f"leg_{k}_2"]
        radius = max(0.25 * (knee - ankle).length, 0.02)
        col = np.nonzero((np.hypot(V0[:, 0] - ankle.x, V0[:, 1] - ankle.y) < radius) & (V0[:, 2] < knee.z * 0.6))[0]
        if len(col) == 0:
            purity[k] = 0.0
            continue
        mine = [i for i, n in enumerate(names) if n.startswith(f"leg_{k}_")]
        purity[k] = float(W[col][:, mine].sum() / max(W[col].sum(), 1e-9))
    worst = min(purity, key=purity.get)
    check("leg_weight_purity_min", purity[worst], purity[worst] >= 0.9, f">= 0.9 (worst {worst})")

    E = edges(mesh)
    rest_len = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1) + 1e-9
    # the shortest edges turn millimetre moves into huge ratios; judge the rest
    long_enough = rest_len >= np.percentile(rest_len, 10)
    acts = actions_by_name(arm)
    for clip in ("idle", "walk"):
        check(f"clip_{clip}", clip in acts, clip in acts, "present")
    walk_speed = spec["walk"]["stride"] / spec["walk"]["cycle"]
    stretch_all = []
    worst_edge = {}
    foot_stats = {}
    for clip in ("idle", "walk"):
        if clip not in acts:
            continue
        lo, hi = play(arm, acts[clip])
        fps = bpy.context.scene.render.fps
        frames = list(range(lo, hi + 1))
        first = last = None
        feet = {k: [] for k in LEGS}
        foot_idx = {}
        for k in LEGS:
            mine = [i for i, n in enumerate(names) if n in (f"leg_{k}_3", f"leg_{k}_2")]
            foot_idx[k] = np.nonzero((W[:, mine].sum(1) > 0.5) & (V0[:, 2] < 0.1 * h))[0]
        for f in frames:
            bpy.context.scene.frame_set(f)
            P = posed_verts(mesh)
            if f == frames[0]:
                first = P
            last = P
            ratio = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) / rest_len
            dev_f = np.where(long_enough, np.maximum(ratio, 1 / ratio), 0)
            i = int(dev_f.argmax())
            if dev_f[i] > worst_edge.get("ratio", 0):
                worst_edge.update(ratio=round(float(dev_f[i]), 3), clip=clip, frame=f,
                                  at=[round(float(c), 3) for c in V0[E[i, 0]]],
                                  bones=sorted({names[g] for g in np.nonzero(W[E[i]].sum(0) > 0.05)[0]}))
            stretch_all.append(ratio[long_enough])
            for k in LEGS:
                if len(foot_idx[k]):
                    feet[k].append(P[foot_idx[k]])
        loop_err = float(np.abs(first - last).max())
        check(f"{clip}_loop_error_m", loop_err, loop_err < 0.002 * h + 1e-4, f"< {0.002 * h:.4f}")
        if clip == "walk":
            for k in LEGS:
                if not feet[k]:
                    continue
                F = np.stack(feet[k])                  # (frames, foot verts, 3)
                floor = F[:, :, 2].min()
                eps = 0.02 * h
                touching = F[:, :, 2] < floor + eps    # vertices on the ground, per frame
                down = touching.any(1)
                # slip: how far the vertices touching the ground on one frame move by the next,
                # in the world (body frame +Y is backwards; the creature walks at walk_speed)
                slips = []
                for i in range(len(F) - 1):
                    if touching[i].any():
                        vy = (F[i + 1, touching[i], 1] - F[i, touching[i], 1]) * fps
                        slips.append(float(np.median(np.abs(vy - walk_speed))))
                foot_stats[k] = {"min_z": float(floor), "down_fraction": float(down.mean()),
                                 "slip_mps": float(np.median(slips)) if slips else float("inf")}
            if foot_stats:
                minz = min(s["min_z"] for s in foot_stats.values())
                check("walk_foot_min_z_m", minz, minz >= -0.02 * h, f">= {-0.02 * h:.3f}")
                downs = min(s["down_fraction"] for s in foot_stats.values())
                check("walk_foot_down_fraction_min", downs, 0.3 <= downs <= 0.9, "0.3..0.9")
                slip = max(s["slip_mps"] for s in foot_stats.values())
                check("walk_foot_slip_mps", slip, slip <= 0.1 * walk_speed, f"<= {0.1 * walk_speed:.3f}")
    if stretch_all:
        S = np.concatenate(stretch_all)
        dev = np.maximum(S, 1 / S)
        check("edge_stretch_p999", np.percentile(dev, 99.9), np.percentile(dev, 99.9) <= 1.6, "<= 1.6")
        # one worst edge out of every frame: a pointer to where to look, too brittle to gate on
        check("edge_stretch_max", dev.max(), dev.max() <= 3.0, "<= 3.0 (warning)", warn=True)

    report = {"pass": all(c["pass"] or c["warn"] for c in checks), "checks": checks, "leg_purity": purity, "feet": foot_stats,
              "walk_speed": walk_speed, "worst_edge": worst_edge}
    json.dump(report, open(out_json, "w"), indent=2)

    # -- renders for the contact sheet --
    cam = setup_render()
    if arm.animation_data:
        arm.animation_data.action = None
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    lo, hi = V0.min(0), V0.max(0)
    centre = (lo + hi) / 2
    extent = float((hi - lo).max())
    views = {"front34": (1, -1, 0.5), "side": (1, 0, 0.05), "back34": (-1, 1, 0.5), "top": (0.001, 0, 1)}
    for name, d in views.items():
        shoot(cam, d, centre, extent, os.path.join(sheet_dir, f"view_{name}.png"))
    sh = bpy.context.scene.display.shading
    sh.show_xray = True
    sh.xray_alpha = 0.25
    markers = joint_markers(arm, 0.012 * extent)
    shoot(cam, (1, 0, 0.05), centre, extent, os.path.join(sheet_dir, "rig_side.png"))
    shoot(cam, (0.001, 0, 1), centre, extent, os.path.join(sheet_dir, "rig_top.png"))
    for m in markers:
        bpy.data.objects.remove(m)
    sh.show_xray = False
    if "walk" in acts:
        lo_f, hi_f = play(arm, acts["walk"])
        for i in range(8):
            f = lo_f + round(i * (hi_f - lo_f) / 8)
            bpy.context.scene.frame_set(f)
            shoot(cam, (1, 0, 0.05), centre, extent, os.path.join(sheet_dir, f"walk_{i}.png"))
    print(f"[check] {'PASS' if report['pass'] else 'FAIL'} -> {out_json}")


main()
