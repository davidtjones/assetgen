"""
Minimal, dependency-free glTF 2.0 writer.

Just enough to export the baked assets (grass tufts and rocks): interleaved-free
(one bufferView per attribute), single mesh, single material, embedded base64
buffer -> one self-contained `.gltf` file. Loads natively in three.js (GLTFLoader)
and Bevy (its glTF loader), so the same asset drives both the preview and the engine.
Shared across generators; lives in `assetgen/common/`.

Supported vertex attributes (all optional except positions):
    POSITION    vec3 f32   (min/max auto-computed, required by spec)
    NORMAL      vec3 f32
    TEXCOORD_0  vec2 f32   — for grass, the root->tip wind/gradient weight in .y
    TEXCOORD_1  vec2 f32   — a second UV set; for grass, per-blade wind randoms
    COLOR_0     vec4 f32   — base->tip color gradient (multiplies base color)

`write_meshes` writes several meshes of several primitives into one file, for
assets with parts or levels of detail (trees).

Not a general exporter: no interleaving, no sparse accessors, no animation, no
textures. Add only what the baked assets need.
"""

import base64
import json
import struct
from pathlib import Path

# glTF component / accessor type constants.
_FLOAT = 5126
_UINT = 5125
_ARRAY_BUFFER = 34962          # vertex attributes
_ELEMENT_ARRAY_BUFFER = 34963  # indices


def _pack_floats(rows: list[tuple]) -> bytes:
    flat = [c for row in rows for c in row]
    return struct.pack(f"<{len(flat)}f", *flat)


def _pack_uints(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def build_gltf(
    positions: list[tuple[float, float, float]],
    indices: list[int],
    normals: list[tuple[float, float, float]] | None = None,
    uvs: list[tuple[float, float]] | None = None,
    uv1: list[tuple[float, float]] | None = None,
    colors: list[tuple[float, float, float, float]] | None = None,
    name: str = "grass",
    double_sided: bool = True,
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    unlit: bool = False,
) -> dict:
    """Build a single-mesh glTF as a dict. Vertex arrays must all be the same length.

    `uv1` becomes TEXCOORD_1 (a second UV set) — used to carry per-blade wind
    randoms into the shader (three loads it as the `uv1` attribute)."""
    n = len(positions)
    assert indices and max(indices) < n, "index out of range"
    for arr, label in ((normals, "normals"), (uvs, "uvs"), (uv1, "uv1"), (colors, "colors")):
        assert arr is None or len(arr) == n, f"{label} length must match positions"

    buffer = bytearray()
    buffer_views = []
    accessors = []
    attributes = {}

    def _add_accessor(data: bytes, *, count: int, comp_type: int, acc_type: str,
                      target: int, minmax: tuple[list, list] | None = None) -> int:
        # 4-byte align the start of every bufferView.
        while len(buffer) % 4 != 0:
            buffer.append(0)
        byte_offset = len(buffer)
        buffer.extend(data)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(data),
            "target": target,
        })
        acc = {
            "bufferView": len(buffer_views) - 1,
            "componentType": comp_type,
            "count": count,
            "type": acc_type,
        }
        if minmax is not None:
            acc["min"], acc["max"] = minmax
        accessors.append(acc)
        return len(accessors) - 1

    # POSITION (required, with min/max).
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    pos_minmax = ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])
    attributes["POSITION"] = _add_accessor(
        _pack_floats(positions), count=n, comp_type=_FLOAT, acc_type="VEC3",
        target=_ARRAY_BUFFER, minmax=pos_minmax,
    )
    if normals is not None:
        attributes["NORMAL"] = _add_accessor(
            _pack_floats(normals), count=n, comp_type=_FLOAT, acc_type="VEC3",
            target=_ARRAY_BUFFER,
        )
    if uvs is not None:
        attributes["TEXCOORD_0"] = _add_accessor(
            _pack_floats(uvs), count=n, comp_type=_FLOAT, acc_type="VEC2",
            target=_ARRAY_BUFFER,
        )
    if uv1 is not None:
        attributes["TEXCOORD_1"] = _add_accessor(
            _pack_floats(uv1), count=n, comp_type=_FLOAT, acc_type="VEC2",
            target=_ARRAY_BUFFER,
        )
    if colors is not None:
        attributes["COLOR_0"] = _add_accessor(
            _pack_floats(colors), count=n, comp_type=_FLOAT, acc_type="VEC4",
            target=_ARRAY_BUFFER,
        )

    index_accessor = _add_accessor(
        _pack_uints(indices), count=len(indices), comp_type=_UINT, acc_type="SCALAR",
        target=_ELEMENT_ARRAY_BUFFER,
    )

    material = {
        "name": f"{name}_mat",
        "doubleSided": double_sided,
        "pbrMetallicRoughness": {
            "baseColorFactor": list(base_color),
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
    }
    if unlit:
        material["extensions"] = {"KHR_materials_unlit": {}}

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(buffer)).decode("ascii")

    gltf = {
        "asset": {"version": "2.0", "generator": "assetgen/common/gltf_io.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": attributes,
                "indices": index_accessor,
                "material": 0,
                "mode": 4,  # TRIANGLES
            }],
        }],
        "materials": [material],
        "buffers": [{"byteLength": len(buffer), "uri": uri}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    if unlit:
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]

    return gltf


def dumps(gltf: dict) -> str:
    """Serialize a glTF dict to a compact JSON string."""
    return json.dumps(gltf, indent=None, separators=(",", ":"))


def write_gltf(path: str | Path, *args, **kwargs) -> None:
    """Build a single-mesh glTF and write it to `path`. See `build_gltf` for args."""
    Path(path).write_text(dumps(build_gltf(*args, **kwargs)))


def write_meshes(path: str | Path, meshes: list[tuple[str, list[dict]]]) -> None:
    """Write several meshes, each of several primitives, into one `.gltf`.

    `meshes` is `[(name, [primitive, ...]), ...]`. A primitive is a dict of
    arrays (anything `numpy.asarray` takes): `positions` (n x 3) and `indices`
    are required; `normals` (n x 3), `tangents` (n x 4, xyz and handedness),
    `uvs`, `uv1` (n x 2) and `colors` (n x 4) are optional.

    For assets whose parts take different materials in the engine (a tree's
    bark and its leaves) or that come at several levels of detail: the engine
    addresses them as mesh `i`, primitive `j`. By default no materials are
    written and the engine supplies them. A primitive may instead carry a
    `material`: a dict with `name`, `color` (linear RGB or RGBA) and optionally
    `roughness` (default 1), written as a plain PBR material (kit pieces).
    Primitives naming the same material share one.
    """
    import numpy as np

    buffer = bytearray()
    views, accessors, out = [], [], []

    def add(array, dtype, kind, target, bounds=False) -> int:
        data = np.ascontiguousarray(array, dtype=dtype)
        while len(buffer) % 4:
            buffer.append(0)
        views.append({"buffer": 0, "byteOffset": len(buffer), "byteLength": data.nbytes, "target": target})
        buffer.extend(data.tobytes())
        accessor = {"bufferView": len(views) - 1, "componentType": _FLOAT if dtype == "<f4" else _UINT,
                    "count": len(data), "type": kind}
        if bounds:
            accessor["min"], accessor["max"] = data.min(0).tolist(), data.max(0).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    materials, material_index = [], {}

    def material(spec: dict) -> int:
        if spec["name"] not in material_index:
            color = list(spec["color"]) + [1.0] * (4 - len(spec["color"]))
            materials.append({"name": spec["name"], "pbrMetallicRoughness": {
                "baseColorFactor": color, "metallicFactor": 0.0,
                "roughnessFactor": float(spec.get("roughness", 1.0))}})
            material_index[spec["name"]] = len(materials) - 1
        return material_index[spec["name"]]

    for name, primitives in meshes:
        entries = []
        for prim in primitives:
            attributes = {"POSITION": add(prim["positions"], "<f4", "VEC3", _ARRAY_BUFFER, bounds=True)}
            for key, label, kind in (("normals", "NORMAL", "VEC3"), ("tangents", "TANGENT", "VEC4"),
                                     ("uvs", "TEXCOORD_0", "VEC2"), ("uv1", "TEXCOORD_1", "VEC2"),
                                     ("colors", "COLOR_0", "VEC4")):
                if prim.get(key) is not None:
                    assert len(prim[key]) == len(prim["positions"]), f"{key} length must match positions"
                    attributes[label] = add(prim[key], "<f4", kind, _ARRAY_BUFFER)
            indices = add(np.asarray(prim["indices"]).ravel(), "<u4", "SCALAR", _ELEMENT_ARRAY_BUFFER)
            entry = {"attributes": attributes, "indices": indices, "mode": 4}
            if prim.get("material") is not None:
                entry["material"] = material(prim["material"])
            entries.append(entry)
        out.append({"name": name, "primitives": entries})

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(buffer)).decode("ascii")
    gltf = {
        "asset": {"version": "2.0", "generator": "assetgen/common/gltf_io.py"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(out)))}],
        "nodes": [{"mesh": i, "name": mesh["name"]} for i, mesh in enumerate(out)],
        "meshes": out,
        "buffers": [{"byteLength": len(buffer), "uri": uri}],
        "bufferViews": views,
        "accessors": accessors,
    }
    if materials:
        gltf["materials"] = materials
    Path(path).write_text(dumps(gltf))
