"""Frame a generated static prop for the game. Runs INSIDE Blender (needs `bpy`).

    blender -b --factory-startup --python-exit-code 1 --python frame_prop.py -- \
        mesh.glb spec.json prop.glb frame.json views_dir

1. Join the mesh, scale it so it stands the spec's `height` tall, stand it on z = 0 and
   centre it over the origin (by its bounding box).
2. Export glTF (+Y up, as the game reads it) with its baked textures.
3. Measure it (height, footprint radius, triangles) and check it against the spec ->
   `frame.json`, with `pass` false when a check fails.
4. Render four views (three-quarter, side, top, under) into `views_dir` for the contact sheet.
"""

import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # drop empties the importer left behind
    for o in list(bpy.context.scene.objects):
        if o is not obj:
            bpy.data.objects.remove(o)
    return obj


def verts(obj):
    a = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def frame(obj, height):
    v = verts(obj)
    lo, hi = v.min(0), v.max(0)
    s = height / float(hi[2] - lo[2])
    centre = (lo + hi) / 2
    # A Matrix, not a nested list: Mesh.transform reads a plain list as columns.
    move = Matrix.Translation((-centre[0] * s, -centre[1] * s, -lo[2] * s)) @ Matrix.Diagonal((s, s, s, 1.0))
    obj.data.transform(move)
    obj.data.update()


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def setup_render(size=512):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("w")
    world.color = (0.2, 0.21, 0.23)
    scene.world = world
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def shoot(cam, direction, centre, extent, path):
    d = Vector(direction).normalized()
    cam.location = Vector(centre) + d * 10
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = extent * 1.25
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    mesh_glb, spec_json, out_glb, out_json, views_dir = sys.argv[sys.argv.index("--") + 1:]
    spec = json.load(open(spec_json))
    obj = load(mesh_glb)
    frame(obj, float(spec["height"]))
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB", export_yup=True, use_selection=False)

    v = verts(obj)
    lo, hi = v.min(0), v.max(0)
    radius = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
    tris = tri_count(obj)
    has_texture = any(n.type == "TEX_IMAGE" and n.image for m in obj.data.materials if m and m.use_nodes
                      for n in m.node_tree.nodes)
    checks = [
        {"name": "height_m", "value": round(float(hi[2] - lo[2]), 4), "limit": f"= {spec['height']}",
         "pass": abs(float(hi[2] - lo[2]) - float(spec["height"])) < 1e-3},
        {"name": "base_on_ground", "value": round(float(lo[2]), 5), "limit": "= 0", "pass": abs(float(lo[2])) < 1e-4},
        {"name": "tris", "value": tris, "limit": f"<= {int(spec['tris'] * 1.1)}", "pass": tris <= spec["tris"] * 1.1},
        {"name": "one_piece_upright", "value": round(radius / float(spec["height"]), 2), "limit": "radius/height <= 2",
         "pass": radius <= 2 * float(spec["height"])},
        {"name": "textured", "value": has_texture, "limit": "base colour map", "pass": has_texture},
    ]
    report = {"pass": all(c["pass"] for c in checks), "checks": checks,
              "height": round(float(hi[2] - lo[2]), 4), "radius": round(radius, 4), "tris": tris}
    json.dump(report, open(out_json, "w"), indent=2)

    os.makedirs(views_dir, exist_ok=True)
    cam = setup_render()
    centre = (lo + hi) / 2
    extent = float((hi - lo).max())
    views = {"front34": (1, -1, 0.5), "side": (1, 0, 0.05), "top": (0.001, 0, 1), "under": (0.3, -0.3, -1)}
    for name, d in views.items():
        shoot(cam, d, centre, extent, os.path.join(views_dir, f"view_{name}.png"))
    print(f"[frame] {'PASS' if report['pass'] else 'FAIL'} -> {out_json}")


main()
