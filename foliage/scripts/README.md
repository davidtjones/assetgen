# Foliage generators

Parametric grass generators. One **shared blade model** feeds two backends, so a
given profile's billboard and its 3D mesh are the *same* plant by construction:

```
                    ┌──────────────────┐
   profiles/*.yaml ─►   blade_core.py   │  BladeParams, ClusterParams,
                    │  (shared model)   │  sample_blade, integrate_blade,
                    └────────┬──────────┘  blade_width
                             │
              ┌──────────────┴───────────────┐
              ▼                               ▼
   generate_silhouette.py            generate_geometry.py
   (RASTER backend)                  (MESH backend, via mesh_backend.py
   → silhouettes/*.png                + gltf_io.py)
   billboards, mid-distance          → meshes/*.gltf
                                      real geometry, close-up
```

- **`blade_core.py`** — the single source of truth for blade shape. Pure model,
  no rendering. Pixel-space (canvas 768×1024, ground at y=990).
- **`generate_silhouette.py`** — raster backend → binary silhouette PNGs, used as
  ControlNet conditioning + billboard alpha. Unchanged in behavior by the split.
- **`mesh_backend.py`** — mesh backend → flat, tapering, drooping blade strips.
- **`gltf_io.py`** (now shared, in `assetgen/common/`) — minimal dependency-free
  glTF 2.0 writer (loads in three.js and Bevy). Used by every generator (foliage,
  rocks); the foliage scripts add `common/` to `sys.path` to import it.
- **`generate_geometry.py`** — mesh **bake** entrypoint. Writes one canonical
  blade + three tuft variants (seeds 0/3/7) per profile to `meshes/`.
- **`serve.py`** — the **studio** dev server: serves the live editor and
  regenerates geometry on demand from this same model (so tuning and baking never
  diverge). Stdlib only.

## Quick start

```
pixi run silhouettes   # regenerate billboard silhouettes       → silhouettes/
pixi run blades        # bake blades + tuft variants per profile → meshes/
pixi run studio        # live blade/tuft editor; open
                       #   http://localhost:8000/preview/studio.html
```

## Studio (live editor)

`pixi run studio` starts `serve.py`, which serves `preview/studio.html` and two
endpoints:

- `GET /params?profile=NAME` → that profile's default blade/mesh/cluster values
  (used to initialize the sliders).
- `POST /generate` `{mode, profile, seed, blade{}, mesh{}, cluster{}}` → a glTF
  JSON string, generated live by `mesh_backend`.

The viewer has three modes — **single blade**, **tuft**, and **field** — plus
sliders for every tunable, root/tip color pickers, and wireframe/normals/
vertex-color toggles. Tuft mode also has **dual-handle randomness ranges**
(length, widths, compliance, leaf-spread): each blade's value is sampled anywhere
in `[lo, hi]`, mapping onto the model's `*_range` fields. **Field mode** scatters
many jittered, rotated copies of a few tuft variants over a ground plane using
three.js `InstancedMesh` — the same instancing model Bevy will use — to preview
density and variety; its `field size / tuft count / variants` sliders re-scatter
instantly (no regen), while tuft-shape sliders regenerate the variants. Drag a
slider and the mesh regenerates in ~one debounced localhost round-trip. **"Copy YAML"**
emits a `mesh:` block (plus any changed top-level cluster fields) to paste into
the profile — then `pixi run blades` bakes the committed asset. Editing and
baking share one Python model, so what you tune is exactly what you bake.

**Wind + lighting.** A live wind toggle sways the grass via a vertex shader. Bend
scales along the blade by the baked `UV.y` weight (roots planted, tips lead) and
is **length-preserving** — each vertex swings on an arc around its root, so blades
*bend* instead of stretching, and self-limit near 90°. The gust field is a
**travelling 2D value-noise (fBm)** so gusts roll patchily across the field, and
per-blade randoms baked into `UV1` give each blade its own phase / amplitude /
direction (so a tuft is a clump of individuals, not a rigid fan). Controls:
**strength** (peak tip bend; self-limits), **turbulence** (per-blade
independence — the anti-"water" knob), **base lean** (constant downwind bend),
**snap** (gust-front flick faking spring overshoot — turn down if it looks
mechanical), **speed** (overall pace; lower = longer sustained gusts), **gust
size** (noise cell size in m — large = field breathes together, small = gusts roll
across), and **direction**. Every slider has a hover tooltip. The scene has a
shadow-casting sun (elevation / azimuth sliders), ACES tone mapping (exposure
slider), and a shadows toggle — all pure client-side (no regen). The formulation
ports to Bevy WGSL, so this validates the look before the engine shader. Caveat:
shadow casting uses the undisplaced depth pass, so cast shadows don't yet sway
with the grass (fine at preview strengths).

**Presets & docs.** Every slider has a hover tooltip explaining what it does.
**Save / Save as…** write the *entire* studio state (mode, all overrides, field,
wind, lighting) to `presets/<name>.json`; the **load preset** dropdown restores
one. Presets are for the studio session; **Copy YAML** is still the path to bake a
tuning into a profile.

**Hot reload.** The server re-imports the generator modules and re-reads the
profile YAML on every request, so editing the Python (`mesh_backend.py` etc.) or
a profile file and pressing **⟳ regenerate** shows the result with no restart
(your slider values are kept; "reset to profile" re-pulls the YAML defaults). A
syntax error in an edited module surfaces as an error overlay in the viewer.

## Profile YAML schema

One YAML per grass type, auto-discovered. Two sections:

### Shared fields (both backends)

Any `ClusterParams` field. Unset fields fall back to dataclass defaults.

| Field | Default | Meaning |
| --- | --- | --- |
| `n_blades` | `20` | Mean number of blades per cluster. |
| `n_blades_var` | `4` | Random ± this many blades per cluster. |
| `length_range` | `[350, 600]` | Blade length in pixels. Canvas is 768×1024. |
| `x_spread` | `200` | Cluster width in pixels (Gaussian around center_x). |
| `initial_angle_spread` | `0.15` | Radians of jitter around straight-up. |
| `compliance_gamma_range` | `[1.5, 2.5]` | Higher = base stays rigid longer, tip bends more sharply. |
| `base_thickness_range` | `[4, 9]` | Blade width at the base, pixels. |
| `tip_thickness_range` | `[1.0, 2.0]` | Blade width at the tip, pixels. |
| `noise_strength` | `0.05` | Per-segment angle jitter (radians). Higher = curlier. |
| `fan_out_strength` | `0.0` | Outer blades lean outward (vase shape; `dense_tuft` uses 0.3). |
| `spread_range` | `[0.0, 0.0]` | Bell-shaped extra mid-width. `>0` = broader leaves. |
| `edge_noise` | `0.0` | Correlated jitter on left/right edges → irregular widths. |
| `head_prob` | `0.0` | Probability a blade has a head at the tip. |
| `head_radius_range` | `[0.0, 0.0]` | Head size range if applied. |
| `head_elongation` | `1.0` | `1.0` = round flower, `>1.0` = elongated seed head. |

### `mesh:` block (mesh backend only)

Optional. **Ignored by the raster/billboard path.** Every field has a default,
so a profile with no `mesh:` block still produces a blade.

| Field | Default | Meaning |
| --- | --- | --- |
| `segments` | `5` | Quads up the blade — the vertex-cost knob (3–7 typical). |
| `droop_deg` | `27` | Tip-tangent pitch off vertical (self-weight arc, 15–40 typical). |
| `droop_gamma` | `2.0` | `>1` concentrates the bend near the tip (rigid root). |
| `droop_noise_deg` | `4.0` | Per-blade random variation on the droop. |
| `up_normal_blend` | `0.7` | Normal blend: `0` = flat geometric, `1` = fully ground-up. |
| `world_per_pixel` | `1/600` | Pixel→world scale (a ~500px blade ≈ 0.83 units tall). |
| `root_color` | `[0.13,0.26,0.07]` | Base-of-blade vertex color (darker/olive). |
| `tip_color` | `[0.46,0.70,0.22]` | Tip vertex color (brighter/yellow-green). |
| `tuft_radius_px` | `x_spread × 0.5` | Tuft base-disk radius (blades placed in it). |
| `tuft_splay_deg` | `fan_out × 35` | Extra droop for edge blades (outward splay). |
| `tuft_azimuth_jitter` | `0.35` | Radians of randomness in each blade's lean direction. |

See `profiles/dense_tuft.yaml` for a worked `mesh:` example. Tuft rows apply only
when assembling tufts; the single blade ignores them.

## Mesh conventions (what a future wind shader / Bevy loader can rely on)

- **`UV1` (TEXCOORD_1) carries two per-blade randoms** in `[0,1]`, constant across
  a blade and different between blades: `.x` = wind phase seed, `.y` =
  amplitude/direction seed. The wind shader uses them so each blade has its own
  phase, bend amount, and lean direction — a tuft reads as many independent blades
  instead of one rigid fan. (three loads TEXCOORD_1 as the `uv1` attribute; Bevy as
  `ATTRIBUTE_UV_1`.)
- **`UV.y` is the root→tip weight**: `0` at the root, `1` at the tip. It is *both*
  the wind weight (multiply vertex displacement by it → root stays planted, tip
  travels most) and the coordinate of the base→tip color gradient. `UV.x` runs
  0→1 across the blade's width.
- **Vertex colors (`COLOR_0`)** carry the baked base→tip gradient so the mesh
  shades correctly with no texture.
- **Normals are blended toward world-up** (`up_normal_blend`) — the stylized
  "one soft surface" trick. A flat blade lit by its true geometric normal reads
  flat; blending toward up makes a field light as a continuous surface.
- **Blades are double-sided.** The material sets `doubleSided`; render with
  backface culling off.
- **Units are world units** (Bevy/three.js), Y-up, root at the origin.

## The physics model in one paragraph

Each blade is a chain of segments walked root→tip. The tangent bends by an amount
weighted by a **compliance profile** (`t ** compliance_gamma`): small at the
rigid root, large at the floppy tip, so blades curve at their tips rather than
wandering. The raster spine adds only noise (blades generated *at rest* — no
directional lean, to keep wind out of the baked texture). The **mesh** spine adds
a deterministic *self-weight droop* on top (tip pitched `droop_deg` off vertical,
concentrated toward the tip): that is a static per-blade property, not a
cluster-wide lean, so it stays consistent with the no-wind-in-the-shape rule.

## Gotchas / non-obvious things

- **Shared values are pixel-space; canvas 768×1024, ground at y=990.** The mesh
  backend converts to world units via `mesh.world_per_pixel`.
- **`spread > 0` produces leaf-shaped blades** (`width = taper + spread·bell(t)`,
  `bell(t)=4t(1−t)`), meeting at both endpoints. Applies to raster and mesh.
- **`edge_noise` uses AR(1) smoothing**, not per-vertex Gaussian (raster only).
- **Sample seeds `0, 3, 7` are hardcoded** in the raster `main()`; the mesh
  backend uses a deterministic *canonical* blade (midpoint of every range) so the
  inspector shows a stable, representative blade per profile.
- **Generated art is committed** (silhouettes and meshes), same as the rest of
  the repo — not gitignored.
