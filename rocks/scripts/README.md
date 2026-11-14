# Rock generator

Procedural rock meshes, baked to glTF for the Bevy `rocks` module. Geometry only -
the rock carries no UVs or textures, because Bevy applies a world-space **triplanar**
material (`assets/shaders/rock_triplanar.wgsl`). This mirrors the foliage generator:
a Python model bakes assets, a three.js page previews them, Bevy loads them.

```text
   profiles/*.yaml ─►  rock_core.py  ─►  generate_rocks.py  ─►  meshes/*.gltf
                       (mesh model)      (bake entrypoint)      + manifest.json
                                                                    │
                                              preview/index.html ◄──┘
                                              (three.js viewer)
```

- **`rock_core.py`** — the rock mesh model (numpy). `build_rock(params, seed)`
  runs five steps: icosphere → ellipsoid squash → 3D-noise displacement →
  fracture cuts (flat faces + a flat base) → post-cut roughen. Returns positions /
  indices / normals, base seated at `y = 0`. Deterministic in `seed`.
  The **roughen** pass matters: a cut projects vertices exactly onto its plane, so
  without it the faces are mathematically flat and read as cut gem facets rather
  than stone. Roughening perturbs every vertex along its normal by a little
  high-frequency noise, keeping the angular shape but killing the dead-flatness.
- **`generate_rocks.py`** — bake entrypoint. For each profile, writes its `seeds`
  variants to `meshes/<profile>_seed<NNN>.gltf` plus a `manifest.json` (the
  preview reads it to know what to load). Uses the shared `common/gltf_io.py`.

## Quick start

```text
pixi run rocks           # bake all profiles' seed variants → meshes/
pixi run preview-rocks   # three.js viewer; open http://localhost:8001/
```

After baking, copy the meshes into the app's asset tree so Bevy can load them:

```sh
cp rocks/meshes/*.gltf ../game3d/assets/meshes/rocks/
```

## Profile schema (`profiles/*.yaml`)

Every field maps to a `RockParams` knob in `rock_core.py`; all are optional and
fall back to the dataclass defaults. `seeds` is generator-level (not a shape knob).

| field                       | meaning                                                                          |
|-----------------------------|----------------------------------------------------------------------------------|
| `subdivisions`              | icosphere subdivisions (4 = 5120 tris; each +1 quadruples)                       |
| `displace_amp/freq/octaves` | surface-noise displacement: `r = 1 + amp * fbm(dir*freq)`                        |
| `ellipsoid_min/max`         | per-axis squash range (each axis picks uniformly in-range)                       |
| `side_cuts`                 | number of random planar fracture faces                                           |
| `cut_offset_min/max`        | how far out each side cut sits (smaller = deeper cut)                            |
| `roughen_amp/freq/octaves`  | post-cut micro-noise along the normal, so flat faces read as stone (0 = off)     |
| `base_tilt`                 | jitter of the base plane off straight-down                                       |
| `base_offset_min/max`       | height of the flat base cut                                                      |
| `seeds`                     | list of RNG seeds → one baked variant each                                       |

Two profiles ship: `boulder` (rounded, few soft facets) and `craggy_rock`
(stronger noise, more pronounced fracture faces). Add a profile by dropping a new
YAML in `profiles/` and re-running `pixi run rocks`.

## Tuning notes

**The mesh carries form; the normal map carries detail.** The triplanar
`rock_surface` material supplies grain, cracks and pitting. Trying to reproduce
that at the vertex level just yields crinkly, jagged rocks. Keep vertex detail low
and let the texture do its job (this is why both profiles ship `roughen_amp: 0`).

**`displace_amp` and `displace_octaves` are not the same dial.** Confusing them is
the easiest way to get a bad rock:

| symptom | cause | fix |
| --- | --- | --- |
| jagged, crinkly surface | `displace_octaves` / `displace_freq` too high | lower them (2 octaves is plenty) |
| smooth featureless egg | `displace_amp` too low, or too few `side_cuts` | raise amp; add a cut |
| flat slab, cut-gem look | `side_cuts` too many, `cut_offset_*` too deep | fewer, shallower cuts |

So: **low octaves, healthy amplitude.** Low octaves keep the surface smooth; the
amplitude and the cuts are what stop it being an egg.

**`roughen_amp` is a fallback**, not a default. It perturbs vertices after the
cuts, which were otherwise mathematically flat. It only earns its keep if a
material has no usable normal map - otherwise leave it at 0.

## Notes

- **Texturing is not here.** The mesh is bare geometry; the triplanar material and
  its tiling/tint live in the Bevy `rocks` module. Keeping it out of the bake is
  what lets an irregular mesh avoid UV seams and stretching.
- **The seam for AI 3D.** This `assetgen/rocks/ → glTF → Bevy loader` boundary is
  the same one a future text/image→3D generator (e.g. Hunyuan3D) would plug into -
  swap the mesh source, keep the loader and material. See the `props` journal.
