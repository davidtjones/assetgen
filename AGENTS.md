# Project notes for agents - `assetgen`

**Read `../NOW.md` first** - David's current direction, which overrides this file.

Before working in this repo, read `../AGENTS.md` for shared workspace guidance
unless it is already in your context. Resolve that path relative to this repo's
root, even when working in a subdirectory.

`AGENTS.md` is canonical; `CLAUDE.md` is a relative symlink to it for Claude
compatibility. Edit `AGENTS.md` and keep the symlink intact.

This is one repo in the `game3d-ws` workspace. For the workspace layout, the
shared work journal, and the cross-repo documentation protocol, see the
workspace-level notes one directory up: `../AGENTS.md`. This file covers the
`assetgen` pipeline specifically.

## What this is

The **asset-generation pipeline**: all Python 3.12 (via **pixi**), driving
ComfyUI + SDXL for image work and dependency-free generators for geometry. It
**bakes assets that the sibling `game3d` app consumes** - the app loads them from
its own `../game3d/assets/` tree at runtime. Run pixi tasks from inside this repo.

## Layout

- `pixi.toml` - Python env (Python 3.12 + PyTorch CUDA) and tasks. See "Common commands" below. Its `[activation.env]` sets `SSL_CERT_FILE` to the env's own CA bundle: the env was created when this repo lived at `~/proj/asset-test`, the env's OpenSSL still has that prefix baked in, and without the override every HTTPS call from this Python fails with `CERTIFICATE_VERIFY_FAILED`. A `pixi reinstall` would fix the root cause but also drops the hand-installed ComfyUI dependencies.
- `foliage/` - grass silhouette/geometry generators (Python), three.js preview/studio, profiles, presets, workflows. Outputs baked glTF tufts into `foliage/meshes/`.
- `rocks/` - procedural rock geometry generator (Python/numpy), YAML profiles, three.js preview. Outputs baked glTF rocks into `rocks/meshes/`, which are then copied into the app at `../game3d/assets/meshes/rocks/`. Geometry only; the `rocks` Bevy module textures them with a triplanar material.
- `trees/` - tree generator (Python/numpy). `scripts/tree_core.py` grows a skeleton per species from a table of numbers (`Habit`) and skins it as two meshes, bark and foliage, at two levels of detail; `scripts/tree_textures.py` makes the bark (drawn, or taken from a `textures/` material named in its `GENERATED` table, with an optional mossy overlay from `MOSSY`) and the foliage cards, whose colour is relative to a leaf colour the game supplies; `scripts/generate_trees.py` bakes meshes, textures, impostor atlases and the manifest straight into `../game3d/assets/trees/`. Each file's docstring is its reference; the manifest format is in `generate_trees.py`. Look at the result with the game's asset viewer (`cargo run --release --bin viewer` in `../game3d`).
- `common/` - shared Python across generators: `gltf_io.py`, the dependency-free glTF 2.0 writer used by foliage, rocks and trees (`write_meshes` for several meshes and primitives in one file).
- `textures/` - material generation: prompt library, ComfyUI workflows, our ComfyUI nodes (`comfy_nodes/`, symlinked into `tools/ComfyUI/custom_nodes/`), and the scripts that generate candidates, preview them lit, and bake the terrain layers. The CHORD weights (`tools/ComfyUI/models/checkpoints/chord_v1.safetensors`) are under a research-only licence.
- `tools/ComfyUI/` - local ComfyUI install for image generation (SDXL 1.0 base + ControlNet Union Promax + our custom `asset_test_utils` nodes). **gitignored.**
- `tools/meshflow/` - MeshFlow (Meta's learned retopology model) with its own venv, installed but without weights: the Hugging Face repo is gated behind a noncommercial research licence that needs David's account. Nothing in this repo uses it since the gear pipeline was removed. **gitignored** (via `tools/.gitignore`).

Each experiment (foliage, rocks, textures) is split by language: its **generator**
lives here under `assetgen/<name>/`, and if it renders in-app its **Bevy plugin**
is `../game3d/src/<name>.rs`.

The learned-terrain work left this repo long ago and is not part of the asset
pipeline. It now lives under `../terrain/`: `../terrain/worldseed/` is a terrain
generator and `../terrain/terrain-diffusion/` is the upstream model clone it draws
weights from. The older `terrain-gen/` reproduction repo that used to sit beside
this one was retired on 2026-08-30 and is no longer on disk; its history is in the
journal archive. See `../AGENTS.md` for the current repo map.

## Documentation

The cross-repo documentation protocol lives in `../AGENTS.md`. **Reference docs
travel with their code** - this repo's:

- `foliage/scripts/README.md` - grass generator reference (YAML profile schema, how to add profiles).
- `foliage/workflows/README.md` - billboard ComfyUI workflow reference (node graph, prompt guidance).
- `rocks/scripts/README.md` - rock generator reference (YAML profile schema).
- `textures/workflows/README.md` - material ComfyUI workflow reference: the klein and CHORD workflows, our nodes, and what is done to CHORD's maps.

**This repo has no status doc.** The workspace is a research workspace with no
fixed deliverable; there is no `MVP.md` or `ROADMAP.md` to consult, and earlier
versions of this file pointed at both before they were removed. The per-generator
`README.md` files above describe how each thing works; the journal
(`../journal/kit/journal brief assetgen`) holds why it was done that way.

**When you change X, update Y** (in addition to the workspace-wide rules in `../AGENTS.md`):

- A generator script, its CLI, or a YAML schema -> that script's `README.md`.
- A ComfyUI workflow graph or custom node -> that workflow's `README.md`.
- Repo layout, commands, or dependencies -> this file (`AGENTS.md`).
- Finished a chunk of work, ran an experiment, or made a non-obvious decision -> add a journal entry (the `journal` skill).

## Common commands

**Python / pixi tasks** (run from inside this repo):

- `pixi run silhouettes` - regenerate all grass silhouettes from `foliage/profiles/*.yaml`
- `pixi run blades` - regenerate baked glTF grass tufts into `foliage/meshes/`
- `pixi run plants` - regenerate the undergrowth plants (fern, flower, heather, scrub) into `../game3d/assets/meshes/plants/`
- `pixi run trees [-- --preview]` - bake trees (meshes, bark and foliage textures, impostors, manifest) into `../game3d/assets/trees/`; about a minute, less when impostor views are cached
- `pixi run material <name>... [--seeds ...] [--maps-only]` - tileable PBR material candidates for a `subject` in `textures/prompts/materials.yaml`: FLUX.2 klein draws a photo, CHORD estimates the maps -> `textures/out/mat/<name>_s<seed>/`, with a lit preview sheet at `textures/out/mat/<name>.jpg`. An entry with `edit_of` is a variant of another material (mossy bark) that lines up with it. `--maps-only` keeps the photos and estimates the maps again. Starts ComfyUI itself if it is not running. `textures/scripts/gen_material.py`'s docstring is the reference.
- `pixi run ground` - bake the terrain layers into `../game3d/assets/textures/terrain/`. Each layer is a generated material graded to a set colour (`GENERATED` in `textures/scripts/bake_ground.py`), with a procedural fallback.
- `pixi run material-sm <name> [--seeds ...]` - the older StableMaterials generator -> `textures/out/sm/`; the pine and aspen bark in use came from it
- `pixi run rocks` - regenerate baked glTF rocks into `rocks/meshes/` (then copy into `../game3d/assets/meshes/rocks/`)
- `pixi run comfyui` - start ComfyUI server on `127.0.0.1:8188`
- `pixi run preview-foliage` - start three.js preview server on `:8000` (opens `foliage/preview/index.html`)
- `pixi run preview-rocks` - start three.js rock preview on `:8001` (opens `rocks/preview/index.html`)
- `pixi run studio` - foliage studio: live blade/tuft editor backed by the Python generator (`http://localhost:8000/preview/studio.html`)
