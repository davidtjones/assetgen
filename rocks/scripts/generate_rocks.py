"""
Rock geometry generator - entrypoint.

For every profile in assetgen/rocks/profiles/, bake its seed variants as
self-contained glTF meshes into assetgen/rocks/meshes/. Each mesh carries only
geometry (positions + smooth normals); Bevy applies a world-space triplanar rock
material, so no UVs or baked textures are needed here. View the bakes in the
three.js preview (assetgen/rocks/preview/index.html, `pixi run preview-rocks`).

Usage:
    pixi run rocks
"""

import json
import sys
from pathlib import Path

import yaml

# gltf_io lives in the shared assetgen/common/ module (used by every generator).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from dataclasses import replace

from rock_core import RockParams, build_rock
from gltf_io import write_gltf

# Seed variants baked per profile when the YAML doesn't set its own `seeds`.
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)

    baked: list[str] = []
    for path in sorted((root / "profiles").glob("*.yaml")):
        name = path.stem
        raw = yaml.safe_load(path.read_text()) or {}
        seeds = raw.get("seeds", DEFAULT_SEEDS)
        params = RockParams.from_profile(raw)

        # `lods: [2]` in a profile also bakes the same rock at coarser icosphere
        # subdivisions, as `<name>_seed<NNN>_lod<S>.gltf`: same shape, fewer
        # triangles, for rocks drawn at a distance. Not listed in the manifest.
        lods = raw.get("lods", [])
        for seed in seeds:
            for level in [None, *lods]:
                p = params if level is None else replace(params, subdivisions=level)
                geo = build_rock(p, seed)
                suffix = "" if level is None else f"_lod{level}"
                out = out_dir / f"{name}_seed{seed:03d}{suffix}.gltf"
                write_gltf(
                    out,
                    positions=geo["positions"],
                    indices=geo["indices"],
                    normals=geo["normals"],
                    name=f"{name}_rock",
                    double_sided=False,  # a rock is a closed solid; no back faces to show
                )
                tris = len(geo["indices"]) // 3
                if level is None:
                    baked.append(out.name)
                print(f"wrote {out.name}  [{name}: {len(geo['positions'])} verts, {tris} tris]")

    # Manifest so the three.js preview knows what to load without a dir listing.
    (out_dir / "manifest.json").write_text(json.dumps({"meshes": baked}, indent=2))
    print(f"wrote manifest.json  [{len(baked)} meshes]")


if __name__ == "__main__":
    main()
