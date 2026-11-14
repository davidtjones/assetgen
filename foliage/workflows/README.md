# `billboard_v1.json`

ComfyUI workflow: takes a grass silhouette PNG and produces a tight-cropped RGBA grass billboard.

## Quick start

1. Start ComfyUI: `pixi run comfyui`
2. Open http://127.0.0.1:8188 in a browser.
3. Drag `billboard_v1.json` onto the canvas (or use `Workflow → Open`).
4. Ensure the silhouette you want lives in `tools/ComfyUI/input/`.
5. In the `Load Image` node, pick the silhouette PNG.
6. Queue.

Output lands at `tools/ComfyUI/output/foliage/billboard_XXXXX.png` as a tight-cropped RGBA PNG.

## Prerequisites (already installed)

- **Checkpoint:** SDXL 1.0 base at `tools/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors`
- **ControlNet:** Xinsir Union Promax SDXL at `tools/ComfyUI/models/controlnet/controlnet_union_sdxl_promax.safetensors`
- **Custom nodes:**
  - `asset_test_utils` (project-local) — provides `TilePreview` and `CropToAlpha`
  - `ComfyUI-seamless-tiling` (from spinagon) — installed but not used by this workflow
  - Native ComfyUI: `SetUnionControlNetType`, `InvertMask`, `ImageToMask`, `JoinImageWithAlpha`

If any of these is missing, ComfyUI will red-outline the corresponding node on load.

## Node graph, left → right

```
Load Checkpoint ─┬─MODEL─────────────────────────────────► KSampler ─LATENT─► VAE Decode ──►
                 ├─CLIP──► CLIPTextEncode (pos) ─┐                                          │
                 │       ► CLIPTextEncode (neg) ─┤                                          │
                 └─VAE────────────────────────►┐ │                                          │
                                               ▼ ▼                                          │
ControlNet Loader ──► SetUnionControlNetType(depth) ──► ControlNet Apply Advanced           │
                                                            ▲                               │
                                                            │                               │
Load Image (silhouette) ────► ControlNet Apply.image ───────┘                               │
                          │                                                                 │
                          └──► ImageToMask ──► InvertMask ──► JoinImageWithAlpha.alpha ◄────┘
                                                                       │
                                                                       ▼
                                                                 CropToAlpha ──► Save Image
```

## Key widget values

| Node | Widget | Value | Notes |
| --- | --- | --- | --- |
| Load Image | image | `<your silhouette>` | Files must live in `tools/ComfyUI/input/`. |
| SetUnionControlNetType | type | `depth` | ControlNet interprets the silhouette as a depth map (white = near = grass). |
| ControlNet Apply Advanced | strength | `0.8` | How strongly the silhouette constrains SDXL. Lower for more artistic freedom. |
| ControlNet Apply Advanced | start / end_percent | `0.0` / `1.0` | Fraction of denoising steps that ControlNet is active. Lower `end_percent` (~0.5) frees the model to add details not in the silhouette. |
| EmptyLatentImage | width / height | `768` / `1024` | Match silhouette canvas dimensions. |
| KSampler | steps / cfg | `25` / `6.0` | Standard SDXL settings. |
| KSampler | sampler_name / scheduler | `dpmpp_2m` / `karras` | Strong SDXL default combo. |
| CropToAlpha | padding | `12` | Transparent pixels of margin around the alpha bbox. |
| CropToAlpha | threshold | `0.01` | Alpha values above this count as "content" for the bbox. |
| Save Image | filename_prefix | `foliage/billboard` | Subfolder + prefix under `output/`. |

## Prompt guidance (from our iteration)

**Positive that works:**

```
macro photograph of a dense cluster of green grass blades, individual blades visible, close-up, sharp focus, natural detail, studio white background
```

**Negative that works:**

```
illustration, drawing, watercolor, sketch, botanical illustration, backlit, silhouette, top-down, overhead, ground, dirt, soil, grid, tile, seamless, cartoon, low quality, blurry, text, watermark
```

**Prompt words to specifically avoid:**

- `botanical specimen` — pulls toward drawn/watercolor illustration.
- `backlit` — with strong ControlNet, model renders the silhouette *as* backlit shadows instead of filling it with grass.
- `isolated on plain white` — OK but "studio white background" reads more photographic.

**For flower/seed-head tip ornaments** — lower ControlNet `end_percent` to ~`0.5` so SDXL has freedom in the final denoising steps, and describe the ornament explicitly in the positive prompt (e.g. `"...with delicate wildflowers on some blade tips..."`).

## Gotchas / non-obvious things

- **`InvertMask` between `ImageToMask` and `JoinImageWithAlpha` is required.** `JoinImageWithAlpha` internally inverts the mask (`alpha = 1 - mask`). Without the explicit invert, grass area becomes transparent and the background becomes opaque — precisely the opposite of what you want. First-attempt bug in early iterations.
- **The silhouette must be white-on-black.** The silhouette generator produces this by default. Black-on-white flips the mask logic.
- **`depth` mode is *not* universal.** Our silhouettes union all blades into one merged shape, losing internal edges. `depth` works because it treats the whole shape as "in the foreground." If you generate silhouettes with visible internal blade outlines instead of merged fills, switch `SetUnionControlNetType` to `canny/lineart/anime_lineart/mlsd`.
- **`CropToAlpha` bounds tightly (padding=12 by default).** Increase padding if the tight crop is too aggressive for your engine's texture atlas requirements.
- **First-run VRAM.** Loading SDXL + ControlNet Union takes ~15GB VRAM. On the 3090 (24GB) it fits comfortably. Watch out on lower-VRAM cards.

## Extending

- **Different silhouette / grass profile:** change the `Load Image` node's dropdown. All profile PNGs need to be copied into `tools/ComfyUI/input/` first.
- **Different aesthetic:** rewrite the positive prompt. Keep the negative prompt anchors to reject known failure modes.
- **Sibling billboards (same profile, different composition):** load each of `<profile>_seed000.png`, `_seed003.png`, `_seed007.png` in turn and queue. Three sibling billboards, same prompt, different underlying silhouettes.
- **Higher resolution:** change both the silhouette canvas size (in `generate_silhouette.py`) and `EmptyLatentImage` in the workflow, in tandem. SDXL is trained at 1024×1024; deviating too much (below ~768 or above ~1536 on either axis) degrades quality.
- **Set consistency across variants:** IP-Adapter (not yet installed) with the ground texture as reference would push the whole billboard set toward one shared palette/lighting. See *Material-set consistency* in the workspace `docs/ROADMAP.md` for context.
