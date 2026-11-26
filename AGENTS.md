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

**How the code is wired.** Each generator is a directory (`foliage/`, `rocks/`,
`trees/`, `textures/`, `creatures/`, `props/`, `concepts/`) with its scripts in `scripts/` and a
`README.md` for someone who wants to run it. The repo root is the import root:
pixi tasks run scripts as modules (`python -m trees.scripts.generate_trees`), and
code imports shared pieces as packages (`from common.gltf_io import ...`,
`from textures.scripts import tex_core`). Do not add `sys.path` hacks; run a script
by hand as `pixi run python -m <generator>.scripts.<name>` from the repo root.
Blender scripts (`creatures/scripts/blender/`, `trees/scripts/decimate_blender.py`)
are the exception: Blender runs them by file path and they import only Blender's
own modules.

**Where the game is.** Everything written into `game3d` goes through
`common/paths.py` (`game3d_dir()`, `game3d_assets(...)`). It defaults to
`../game3d` and can be pointed elsewhere with `GAME3D_DIR=<path>`; it stops with an
error if there is no `Cargo.toml` there, rather than writing assets nobody reads.

## Layout

- `pixi.toml` - Python env (Python 3.12 + PyTorch CUDA) and tasks. See "Common commands" below. Conda packages bake the env's install path into their config files, so after moving this repo, rebuild the env (`rm -rf .pixi/envs && pixi install`); a stale path breaks HTTPS (`CERTIFICATE_VERIFY_FAILED`) and makes fontconfig recreate the old directory for its cache. A rebuild drops the hand-installed ComfyUI dependencies: snapshot `pixi run python -m pip freeze` first and reinstall the missing lines with `--no-deps`.
- `foliage/` - grass silhouette/geometry generators (Python), three.js preview/studio, profiles, presets, workflows. Outputs baked glTF tufts into `foliage/meshes/`; `scripts/generate_plants.py` bakes the undergrowth plants into `../game3d/assets/meshes/plants/`.
- `rocks/` - procedural rock geometry generator (Python/numpy), YAML profiles, three.js preview. Outputs baked glTF rocks and their `manifest.json` (read by the preview) into `rocks/meshes/` (gitignored), which are then copied into the app at `../game3d/assets/meshes/rocks/`. Geometry only; the `rocks` Bevy module textures them with a triplanar material.
- `trees/` - tree generator (Python/numpy). `scripts/tree_core.py` grows a skeleton per species from a table of numbers (`Habit`) and skins it as two meshes, bark and foliage, at two levels of detail; `scripts/tree_textures.py` makes the bark (drawn, or taken from a `textures/` material named in its `GENERATED` table, with an optional mossy overlay from `MOSSY`) and the foliage cards, whose colour is relative to a leaf colour the game supplies; `scripts/generate_trees.py` bakes meshes, textures, impostor atlases and the manifest straight into `../game3d/assets/trees/`. Each file's docstring is its reference; the manifest format is in `generate_trees.py`. Look at the result with the game's asset viewer (`cargo run --release --bin viewer` in `../game3d`).
- `common/` - shared Python across generators: `paths.py`, where the sibling `game3d` and `matdex` repos are (see above); `gltf_io.py`, the dependency-free glTF 2.0 writer used by foliage, rocks, trees and kits (`write_meshes` for several meshes and primitives in one file, optionally with flat materials); `comfy_client.py`, the standard-library ComfyUI HTTP client used by `textures/`, `creatures/` and `props/` (it validates a workflow against the server's nodes before queueing, and fails when ComfyUI accepts a prompt with node errors, which it would otherwise run in part); `image_to_mesh.py` with `workflows/`, the FLUX.2 klein concept and TRELLIS.2 mesh stages that `creatures/` and `props/` share; `game_console.py`, the standard-library client for the running game's console socket (one console line in, its output back), used by `concepts/`; `agent.py`, `ask(question, schema, record)`: one narrow question to an agent, answered as JSON in a schema, with the question, answer and cost kept in `record/` (Claude Code headless, `claude -p`, today; the backend is swappable), used by `concepts/` and the material library.
- `creatures/` - the creature pipeline: one spec in `specs/<name>.yaml` -> a rigged, animated, textured glTF in `../game3d/assets/creatures/`. FLUX.2 klein concept, TRELLIS.2 mesh with baked PBR maps (both through ComfyUI, `common/image_to_mesh.py`), then Blender scripts (`scripts/blender/`) that fit a quadruped skeleton from the mesh's shape, skin it, key in-place `idle` and gait loops (`walk`, `run`, `sprint`), and check the result against the spec (geometry, weights, deformation, foot contact and slip) with a contact sheet. `scripts/creature.py`'s docstring is the reference; outputs go to `creatures/out/<name>/` (gitignored). The game's side is `../game3d/src/creatures.rs`.
- `props/` - the prop pipeline: one spec in `specs/<name>.yaml` -> a static, textured glTF in `../game3d/assets/props/` (today the mushroom boar forage). The same concept and mesh stages as `creatures/`, then a Blender script (`scripts/blender/frame_prop.py`) that scales it to the spec's height, stands it on the ground, checks it and renders a contact sheet. `scripts/prop.py`'s docstring is the reference; outputs go to `props/out/<name>/` (gitignored). The game's side is `../game3d/src/world/food.rs`. A spec with `kind: kit` is instead a set of building pieces made by code on a grid (`scripts/kit.py`: today `fort`, a dozen stone fort pieces), checked piece by piece and in a test arrangement, rendered by `scripts/blender/render_kit.py`, and exported to `../game3d/assets/kits/<name>/` with a `kit.json` manifest that `../game3d/src/world/kit.rs` reads.
- `concepts/` - site concepts: photograph a site in the running game over its console socket (`site.shots`), then Qwen-Image-Edit-2511 draws a structure standing there, keeping the land, camera and light. What is drawn is a brief in `briefs/<name>.yaml` (today `fort`); outputs go to `concepts/out/<site>/` (gitignored). `scripts/paint.py` paints a blockout instead: the build's pictures repainted as the finished place, each part from its site plan's `finish`, from what the game says each picture shows (`site.seen`). `scripts/looks.py` dresses a painted block's building: an agent describes each part (wall, roof, frame...) in the material library's words, and the library gives back a material for each, tinted to the painting's colour, which the game dresses a structure in. `scripts/concept.py`'s, `scripts/paint.py`'s and `scripts/looks.py`'s docstrings and `concepts/README.md` are the reference.
- `textures/` - material generation: prompt library, ComfyUI workflows, our ComfyUI nodes (`comfy_nodes/`, symlinked into `tools/ComfyUI/custom_nodes/`), and the scripts that generate candidates, preview them lit, and bake the terrain layers. The material library that used to live in `library/` is now matdex (`../matdex`), which sends this repo's klein and CHORD workflows to this repo's ComfyUI. The CHORD weights (`tools/ComfyUI/models/checkpoints/chord_v1.safetensors`) are under a research-only licence.
- `tools/ComfyUI/` - local ComfyUI install (v0.45.3) for image and 3D generation: FLUX.2 klein, Qwen-Image-Edit-2511 (int8, with its Lightning 4-step LoRA), LongCat-Image-Edit Turbo (kept from the model comparison, unused), SDXL 1.0 base + ControlNet Union Promax, CHORD, native TRELLIS.2 (Comfy-Org's repackaged weights in `models/diffusion_models`, `vae`, `clip_vision`), and our custom `asset_test_utils` nodes. It needs torch 2.7 or newer, hence the cu130 torch in `pixi.toml`. Started with `--disable-dynamic-vram`: its dynamic VRAM allocator segfaulted freeing a buffer after a CHORD run. **gitignored.**
- `tools/meshflow/` - MeshFlow (Meta's learned retopology model) with its own venv, installed but without weights: the Hugging Face repo is gated behind a noncommercial research licence that needs David's account. Nothing in this repo uses it since the gear pipeline was removed. **gitignored** (via `tools/.gitignore`).

Each generator (foliage, rocks, trees, creatures) is split by language: its **generator**
lives here under `assetgen/<name>/`, and if it renders in-app its **Bevy plugin**
is `../game3d/src/<name>.rs`.

The learned-terrain work left this repo long ago and is not part of the asset
pipeline. See `../AGENTS.md` for the current repo map.

## Documentation

The cross-repo documentation protocol lives in `../AGENTS.md`. **Reference docs
travel with their code** - this repo's:

- `<generator>/README.md` - one per generator (`foliage/`, `rocks/`, `trees/`, `textures/`, `creatures/`, `props/`, `concepts/`): what it makes, how to run it, and which script does what. `foliage/README.md` and `rocks/README.md` also hold the YAML profile schemas.
- `foliage/workflows/README.md` - billboard ComfyUI workflow reference (node graph, prompt guidance).
- `textures/workflows/README.md` - material ComfyUI workflow reference: the klein and CHORD workflows, our nodes, and what is done to CHORD's maps.
- Script docstrings - the CLI reference for each entry point (`gen_material.py`, `creature.py`, `generate_trees.py`, ...).

**This repo has no status doc.** The workspace is a research workspace with no
fixed deliverable; there is no `MVP.md` or `ROADMAP.md` to consult, and earlier
versions of this file pointed at both before they were removed. The per-generator
`README.md` files above describe how each thing works; the journal
(`../journal/kit/journal brief assetgen`) holds why it was done that way.

**When you change X, update Y** (in addition to the workspace-wide rules in `../AGENTS.md`):

- A generator script, its CLI, or a YAML schema -> that generator's `README.md` (and the script's docstring).
- A ComfyUI workflow graph or custom node -> that workflow's `README.md`.
- Repo layout, commands, or dependencies -> this file (`AGENTS.md`).
- Finished a chunk of work, ran an experiment, or made a non-obvious decision -> add a journal entry (the `journal` skill).

## Common commands

**Python / pixi tasks** (run from inside this repo; pixi runs them from the repo root). Tasks that write into the game honour `GAME3D_DIR`:

- `pixi run silhouettes` - regenerate all grass silhouettes from `foliage/profiles/*.yaml`
- `pixi run blades` - regenerate baked glTF grass tufts into `foliage/meshes/`
- `pixi run plants` - regenerate the undergrowth plants (fern, flower, heather, scrub) into `../game3d/assets/meshes/plants/`
- `pixi run trees [-- --preview]` - bake trees (meshes, bark and foliage textures, impostors, manifest) into `../game3d/assets/trees/`; about a minute, less when impostor views are cached
- `pixi run material <name>... [--seeds ...] [--maps-only]` - tileable PBR material candidates for a `subject` in `textures/prompts/materials.yaml`: FLUX.2 klein draws a photo, CHORD estimates the maps -> `textures/out/mat/<name>_s<seed>/`, with a lit preview sheet at `textures/out/mat/<name>.jpg`. An entry with `edit_of` is a variant of another material (mossy bark) that lines up with it. `--maps-only` keeps the photos and estimates the maps again. Starts ComfyUI itself if it is not running. `textures/scripts/gen_material.py`'s docstring is the reference.
- `pixi run ground` - bake the terrain layers into `../game3d/assets/textures/terrain/`. Each layer is a generated material graded to a set colour (`GENERATED` in `textures/scripts/bake_ground.py`), with a procedural fallback.
- `pixi run material-sm <name> [--seeds ...]` - the older StableMaterials generator -> `textures/out/sm/`; the pine and aspen bark in use came from it
- `pixi run rocks` - regenerate baked glTF rocks into `rocks/meshes/` (then copy into `../game3d/assets/meshes/rocks/`)
- `pixi run creature <stage> <name>... [--seed N] [--force]` - the creature pipeline, stages `concept`, `mesh`, `rig`, `check`, `export` or `all`, for specs in `creatures/specs/` (today `boar`, `lizard`). About 2 minutes per creature end to end on the 3090; `check` prints each measurement and writes `creatures/out/<name>/sheet.png`; `export` copies a passing creature into `../game3d/assets/creatures/` and updates `creatures.txt` there. Starts ComfyUI itself if it is not running.
- `pixi run prop <stage> <name>... [--seed N] [--force]` - the prop pipeline, stages `concept`, `mesh`, `frame`, `export` or `all`, for specs in `props/specs/` (today `mushroom`); a kit (`fort`) has the stages `build`, `frame`, `export` and takes a few seconds. `frame` prints each check and writes `props/out/<name>/sheet.png`; `export` copies a passing prop into `../game3d/assets/props/` and updates `props.txt` there. Starts ComfyUI itself if it is not running.
- `pixi run concept <site> [--brief fort|<file.yaml>] [--views ...] [--seeds ...] [--photos <dir>] [--out <dir>]` - site photos from the running game, then concepts of the brief's structure drawn into them -> `concepts/out/<site>/` (or `--out`), with `sheet.png`. The game must be running (`WORLD_DIR=assets/terrain/crossing cargo run --release` in `../game3d`), unless `--photos` reuses photos already taken. About 20 s an image beside the game. Starts ComfyUI itself if it is not running.
- `pixi run paint <plan.toml> [--out <dir>] [--views ...] [--seeds ...] [--pictures <dir>] [--words] [--no-mask]` - paint a built blockout: the plan's structure photographed in the running game (blocks plain white), then each picture repainted as the finished place from the plan's finishes, inside the structure's mask only (the land stays as the game drew it) -> `concepts/out/<site>/paint/` (or `--out`), with `sheet.png`. `--words` prints the prompts without painting; `--no-mask` keeps the whole painting.
- `pixi run looks <paint run dir> <block> [--view <view>] [--slots wall roof footing frame ...] [--painting 0] [--structure <name>] [--region <region>] [--game]` - a block's materials from its painting: an agent describes each slot, matdex resolves each description in the site's region (the plan's `region`, or `--region`) (a match, or a new material), tinted to the painting's colour -> `<run>/looks/<block>/` (`looks.json`, `looks.con` with a `build.look` line per slot, `sheet.jpg`); `--game` exports the materials to `../game3d/assets/materials/<name>/`. Seconds a slot when the library has a match, about a minute when it draws one. Needs the `claude` CLI.
- The material library is its own repo now, `../matdex` (uv: `uv run matdex <command>`; see its `AGENTS.md`). It draws through this repo's ComfyUI, so keep `pixi run comfyui` and the klein/CHORD workflows working; `common/paths.py` `use_matdex()` is how `looks` imports its client.
- `pixi run comfyui` - start ComfyUI server on `127.0.0.1:8188`
- `pixi run preview-foliage` - start three.js preview server on `:8000` (opens `foliage/preview/index.html`)
- `pixi run preview-rocks` - start three.js rock preview on `:8001` (opens `rocks/preview/index.html`)
- `pixi run studio` - foliage studio: live blade/tuft editor backed by the Python generator (`http://localhost:8000/preview/studio.html`)
