"""Run inside Blender by `tree_wood.py`:

    blender -b --python decimate_blender.py -- in.obj out.obj <triangles>

Reads a triangle mesh, collapses it to about that many triangles, writes it back.
"""

import sys

import bpy

src, dst, triangles = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=src)
mesh = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = mesh
modifier = mesh.modifiers.new("decimate", "DECIMATE")
modifier.ratio = min(1.0, int(triangles) / max(len(mesh.data.polygons), 1))
bpy.ops.object.modifier_apply(modifier=modifier.name)
bpy.ops.wm.obj_export(filepath=dst, export_triangulated_mesh=True, export_normals=False, export_uv=False,
                      export_materials=False)
