"""
Grass geometry generator — entrypoint (STEP 1: single blade).

For every profile in assetgen/foliage/profiles/, export one canonical blade as a
self-contained glTF into assetgen/foliage/meshes/. The blade is the flat,
tapering, drooping strip built by `mesh_backend`; view it in the three.js blade
inspector (assetgen/foliage/preview/blade.html).

Usage:
    pixi run blades

STEP 2 will add tuft assembly to this same entrypoint.
"""

import sys
from pathlib import Path


from foliage.scripts.blade_core import ClusterParams, load_profile
from foliage.scripts.mesh_backend import MeshParams, build_blade_mesh, build_tuft_mesh, canonical_blade
from common.gltf_io import write_gltf

# Tuft variants baked per profile, mirroring the raster path's sample seeds.
TUFT_SEEDS = [0, 3, 7]


def _load_profile(path: Path) -> tuple[ClusterParams, MeshParams]:
    """Load both the shared cluster params and the mesh-only block from one YAML."""
    cluster, raw = load_profile(path)
    return cluster, MeshParams.from_profile(raw)


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted((root / "profiles").glob("*.yaml")):
        name = path.stem
        cluster, mesh = _load_profile(path)
        # Single canonical blade (deterministic; for the blade inspector).
        blade = canonical_blade(cluster)
        _export(out_dir / f"{name}_blade.gltf", build_blade_mesh(blade, mesh), f"{name}_blade", name)

        # Tuft variants (seeded; Bevy can split instances across them for variety).
        for seed in TUFT_SEEDS:
            geo = build_tuft_mesh(cluster, mesh, seed=seed)
            _export(out_dir / f"{name}_tuft_seed{seed:03d}.gltf", geo, f"{name}_tuft", name)


def _export(out: Path, geo: dict, mesh_name: str, label: str) -> None:
    write_gltf(
        out,
        positions=geo["positions"],
        indices=geo["indices"],
        normals=geo["normals"],
        uvs=geo["uvs"],
        uv1=geo["uv1"],
        colors=geo["colors"],
        name=mesh_name,
        double_sided=True,
    )
    tris = len(geo["indices"]) // 3
    verts = len(geo["positions"])
    print(f"wrote {out.name}  [{label}: {verts} verts, {tris} tris]")


if __name__ == "__main__":
    main()
