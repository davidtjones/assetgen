# Material generation workflows

Turns a text prompt into a tileable PBR material: albedo, normal, roughness,
metalness and height. Two models run one after the other in ComfyUI: FLUX.2
klein draws a photo of the surface, and CHORD estimates the maps from it. You
do not open ComfyUI for this; `pixi run material <name>` fills in the workflows
below and runs them (`../scripts/gen_material.py`, whose docstring is the CLI
reference).

    pixi run material bark_oak --seeds 0 1 2      # three candidates and a preview sheet
    pixi run material bark_oak --seeds 2 --maps-only   # keep the photo, estimate the maps again

Candidates land in `../out/mat/<name>_s<seed>/`, and `../out/mat/<name>.jpg` is
the lit preview sheet to judge them on. Prompts live in
[`../prompts/materials.yaml`](../prompts/materials.yaml); its header explains
the keys (`subject`, `avoid`, `tile`, `edit_of`).

## The workflows

Each is an API-format graph with `{{placeholders}}` the script fills in.

- **[`klein_tile.json`](klein_tile.json)** - text to a tileable photo.
  `UNETLoader` (klein 4B) -> `TorusRoll` -> `CFGGuider` -> sampler ->
  `CircularVAEDecode` -> `SaveImage`. Placeholders: models, `positive`,
  `negative`, `width`, `height`, `seed`, `steps`, `cfg`, `tiling`
  (`enable`, `x_only`, `y_only`, `disable`), `prefix`.
- **[`klein_edit.json`](klein_edit.json)** - the same, editing an existing
  photo. Adds `LoadImage({{image}})` -> `VAEEncode` -> `ReferenceLatent` on
  both prompts. Used for an entry with `edit_of`: klein repaints the same seed
  of another material (bark -> mossy bark), so the two line up and can be
  blended.
- **[`chord_maps.json`](chord_maps.json)** - photo to maps.
  `LoadImage({{image}})` -> `ChordLoadModel` -> `ChordMaps` -> four
  `SaveImage` nodes under `{{prefix}}/albedo|normal|roughness|metalness`. Any
  workflow with that shape can be added as another decomposer
  (`DECOMPOSERS` in `gen_material.py`).

## Our nodes (`../comfy_nodes/assetgen_textures`)

Symlinked into `tools/ComfyUI/custom_nodes/`. Restart ComfyUI after changing
them.

- **`TorusRoll`** (model, tiling, seed -> model) makes a transformer model
  tile. Before each denoising call the latent is rolled by a random
  wrap-around offset, one per noise level, and the prediction is rolled back,
  so the picture's edges are mended as ordinary interior. Reference latents of
  the same size are rolled with it. Pair it with `CircularVAEDecode`.
- **`ChordMaps`** (chord_model, image -> basecolor, normal, roughness,
  metalness) runs CHORD at the image's own size, which must be a multiple of
  64. CHORD's own node scales everything to 1024 and back, which leaves a
  line pattern in the maps at any other size.

## What the script does to CHORD's maps

CHORD's raw maps are not usable as tiles. `decompose` in `gen_material.py`:

1. Gives CHORD the tile with 128 px of its wrap-around neighbours attached on
   every side and keeps the middle, cross-fading each edge with the estimate
   one tile away. CHORD's maps are wrong in a strip along the image edge.
2. Removes the model's 8 px working grid from every map.
3. Flips the normal map from DirectX to OpenGL (green up).
4. Solves height from the normal map so that it tiles, keeping local relief
   only.

Do not blend several CHORD estimates of one tile: it reads the same surface at
up to twice the relief from one run to the next.

## Install (once)

- **klein:** `flux-2-klein-base-4b.safetensors`, `qwen_3_4b.safetensors`,
  `flux2-vae.safetensors`, shared with the gear pipeline
  (`../../gear/workflows/README.md`).
- **Tiling VAE decode:** the `ComfyUI-seamless-tiling` node pack.
- **CHORD:** the `comfyui-chord` node pack and `chord_v1.safetensors` in
  `tools/ComfyUI/models/checkpoints/`. The weights are gated on Hugging Face
  (log in with `pixi run python -m huggingface_hub.cli.hf auth login`) and
  under a Research-Only Copyleft licence.

## Limits

- CHORD's colours drift (grey stone comes out blue), so the bakers that take a
  material into the game set its colour themselves (`tex_core.recolour`).
- An edit changes colour far more than relief: a mossy variant's normal and
  roughness are close to the base material's.
- One candidate takes about 45 s on the 3090.

---

# The older StableMaterials workflow

Kept because some textures in use still come from it (boulders, pine and aspen
bark). `pixi run material-sm` is its headless form.

Turns a text prompt into a **PBR ground material** — `albedo` + `normal` + `roughness`, tileable,
de-lit — and writes it straight into the Bevy app assets. One model, one workflow:
[`matforger_pbr.json`](matforger_pbr.json).

The model is **StableMaterials** (`gvecchio/StableMaterials`), a joint SVBRDF diffusion model that
emits all maps together from one prompt. It runs in ComfyUI through the `ComfyUI_PBR_Maker` node
pack (for the model loader) driven by our own `asset_test_utils` nodes (prompt library, seeded
sampler, exporter, palette). Bevy renders the result via the app's `src/textures.rs`; press
**`M`** in the running app to cycle the materials listed in `GROUND_MATERIALS`.

> Background — why StableMaterials, and the dead-ends we ruled out (SDXL-albedo-then-derive,
> Ubisoft CHORD's research-only license, "AlbedoBase XL is not a de-lighting model") — is in
> the `terrain-texture` journal entries.

---

## Install (once)

1. **Model loader:** `git clone https://github.com/smthemex/ComfyUI_PBR_Maker` into
   `tools/ComfyUI/custom_nodes/` (MIT). Provides `Load_MatForger`, which we use only to load the
   model. Restart ComfyUI so it installs Python deps.
2. **Model:** `gvecchio/StableMaterials` (license: OpenRAIL — fine for game assets). Auto-downloads
   via `diffusers` on first run (`trust_remote_code=True`); fits the 3090. Optional:
   `pixi add accelerate` for faster/lighter loading.
   - ⚠️ **Do not use `gvecchio/MatForger`** — the prototype repo is gone (HF 401). StableMaterials
     is its successor: same author, same architecture, same outputs. Our workflow already points at it.
3. **Our nodes:** `asset_test_utils` (already in `custom_nodes/`) provides `MaterialPrompt`,
   `StableMaterialsSampler`, `BevyMaterialExport`, `FoliagePalette`. **After changing them, fully
   restart ComfyUI *and* hard-refresh the browser** (Ctrl/Cmd-Shift-R) — they change node
   definitions and ship a JS widget.

---

## The workflow — `matforger_pbr.json`

```
Load_MatForger ─(pipe)─┐
MaterialPrompt ─(prompt,negative)─→ StableMaterialsSampler ─→ BevyMaterialExport ─→ FoliagePalette
                       (name)                    │  (seed)          (writes maps)     (root/tip)
                                                 ├─→ PreviewImage (basecolor + normal)
                                                 └─→ PreviewImage (height/rough/metal)
```

Loads with the `ComfyUI_PBR_Maker` + `asset_test_utils` packs. The sampler's outputs are batched
(`basecolor_normal` = [basecolor, normal]; `height_rough_metal` = [height, roughness, metallic]) —
`BevyMaterialExport` slices them, so there's no manual renaming.

### Material-library nodes (`asset_test_utils`)

- **`MaterialPrompt` (Material Prompt (library))** — dropdown of named prompts from
  [`../prompts/materials.yaml`](../prompts/materials.yaml). Picking a preset fills the node's
  **editable `text` and `negative` widgets** from the library (via the bundled JS in `web/`), so
  you see and tweak the prompt before generating. Outputs `prompt`, `negative` (→ sampler) and
  `name` (→ exporter). Add a keeper by appending to the YAML — either `key: <positive>` or a `key:`
  mapping with `positive:`/`negative:`. Headless-safe: a blank widget falls back to the stored value.
- **`StableMaterialsSampler` (StableMaterials Sampler (seeded))** — the sampler we use instead of
  the stock `MatForger_Sampler`, which exposes neither a seed nor a negative prompt. Adds both: a
  seeded (portable, CPU) generator — `control` = **fixed** to reproduce/lock, **randomize** to
  explore — and a `negative` input. Outputs the `seed` (→ exporter, for the manifest). `size` is
  **512** native; upscale separately if it reads soft.
- **`BevyMaterialExport` (Bevy PBR Material Export)** — the save node. `SAVE` toggle off = previews
  only; flip on + re-queue to write `albedo/normal/roughness.png` **and** a `material.toml`
  manifest (records **prompt, seed, palette**) into `assets/textures/<name>/`. `name` is the
  folder; wire `MaterialPrompt.prompt` and `StableMaterialsSampler.seed` into it so the manifest
  captures them. A multi-line prompt writes a sibling set (`<name>_1`, `<name>_2`, …).
- **`FoliagePalette` (Foliage Palette (root/tip))** — samples the albedo into a dark (root) / light
  (tip) colour pair and prints a paste-ready `mesh:` snippet (in the ComfyUI console) for a foliage
  profile, so the 3D grass shares the ground palette. `dark_pct`/`light_pct` set how much of the
  darkest/lightest pixels to average.

### The iteration loop

Pick a preset → tweak `text`/`negative` → seed `control` = **fixed** → **Queue** → eyeball the two
previews → like it? flip **SAVE** → **Queue** (seed fixed, so the sampler is cached — you save
exactly what you previewed; the manifest records that seed) → flip SAVE off. To explore, bump the
seed (or set `control` = randomize for a run, then back to fixed to lock). Reproduce any saved
material later from the `seed` + `prompt` in its `material.toml`.

---

## The map contract (what Bevy loads)

Each material is three PNGs in the app's `assets/textures/<name>/`:

```
<name>/  albedo.png  normal.png  roughness.png   (+ material.toml)
```

Channel / colour-space rules (handled in `lib.rs`):

- **albedo** — sRGB, seamless/tileable.
- **normal** — linear (`is_srgb=false`), OpenGL convention (Y+). If bumps look *inverted* in-scene,
  set `flip_normal_map_y: true` on that material in `lib.rs`.
- **roughness** — linear grayscale; Bevy reads the **G** channel. Dark = glossy, light = matte.
- **No AO** — low-value on a flat ground plane and StableMaterials doesn't output it. Use Bevy
  runtime screen-space AO if you ever want contact shadows.

**Naming / the `M` toggle:** `BevyMaterialExport.name` is the folder under `textures/`. The
Bevy loader cycles the names in `GROUND_MATERIALS` in the app's `src/textures.rs` (default
`grassland_floor`) — add a name there to include a material in the toggle, and the scene opens on
the first.

---

## FoliagePalette → grass, in one loop

`BevyMaterialExport`'s albedo passthrough feeds `FoliagePalette`, which prints:

```
mesh:
  root_color: [0.14, 0.22, 0.07]
  tip_color:  [0.49, 0.63, 0.28]
```

Paste those into a foliage profile's `mesh:` block ([`../../foliage/profiles/dense_tuft.yaml`](../../foliage/profiles/dense_tuft.yaml)),
then `pixi run blades` to re-bake the grass geometry with the matched vertex-colour gradient. The
values are in the albedo's sRGB space — a starting point; nudge by eye against the scene.
