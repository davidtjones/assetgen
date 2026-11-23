"""Render a kit's glTF (one piece, a row of pieces, or a test arrangement) from a few views.
Runs INSIDE Blender (needs `bpy`).

    blender -b --factory-startup --python-exit-code 1 --python render_kit.py -- \
        scene.gltf out_dir name [size]

Writes `<out_dir>/<name>_<view>.png` for the views `front` (from the outer -z side, above),
`back` (from inside, +z, above) and `top`. Flat colours from the glTF's materials, studio
light with shadows and cavity shading, so seams and gaps between pieces show.
"""

import os
import sys

import bpy
from mathutils import Vector


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    path, out, name = args[:3]
    size = int(args[3]) if len(args) > 3 else 900
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo = Vector((1e9, 1e9, 1e9))
    hi = -lo
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    centre = (lo + hi) / 2

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sh.show_shadows, sh.show_cavity = True, True
    sh.cavity_type = "WORLD"
    scene.display.shadow_shift = 0.1
    scene.render.resolution_x, scene.render.resolution_y = size, int(size * 0.66)
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("w")
    world.color = (0.32, 0.34, 0.37)
    scene.world = world
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = 50
    cam.data.clip_end = 10000
    scene.collection.objects.link(cam)
    scene.camera = cam
    # Blender is +Z up; the glTF's -z (outer side) is Blender's +Y.
    views = {"front": (0.55, 1.0, 0.65), "back": (-0.6, -1.0, 0.7), "top": (0.001, 0.0, 1.0)}
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    for view, d in views.items():
        d = Vector(d).normalized()
        rot = (-d).to_track_quat("-Z", "Y")
        cam.rotation_euler = rot.to_euler()
        # Back off until every corner of the bounds is inside the frame, with a margin.
        inv = rot.inverted()
        half_w = cam.data.sensor_width / (2 * cam.data.lens) * 0.9
        half_h = half_w * scene.render.resolution_y / scene.render.resolution_x
        dist = 0.0
        for c in corners:
            q = inv @ (c - centre)  # camera space: looking down -z
            dist = max(dist, q.z + abs(q.x) / half_w, q.z + abs(q.y) / half_h)
        cam.location = centre + d * dist
        scene.render.filepath = os.path.join(out, f"{name}_{view}.png")
        bpy.ops.render.render(write_still=True)


main()
