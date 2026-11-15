# Textures

Tileable PBR materials and the terrain layers built from them.

```text
pixi run material <name>... [--seeds 0 1] [--maps-only]   # generate candidates -> out/mat/
pixi run ground [-- --size 512 --only rock]               # bake terrain layers -> ../game3d/assets/textures/terrain/
pixi run material-sm <name> [--seeds ...]                 # the older StableMaterials generator -> out/sm/
```

- `prompts/materials.yaml` - the material library: one `subject` per name.
- `scripts/gen_material.py` - FLUX.2 klein draws a tileable photo, CHORD
  estimates the maps, both through ComfyUI (started if it is not running). Writes
  `out/mat/<name>_s<seed>/` and a lit preview sheet `out/mat/<name>.jpg`. Its
  docstring is the CLI reference.
- `scripts/bake_ground.py` - grades the generated materials named in `GENERATED`
  to set colours and packs them into the arrays the terrain shader reads. A layer
  whose material is missing falls back to a procedural one.
- `scripts/gen_material_sm.py` - kept because some bark and ground in use came
  from it; new materials come from `gen_material.py`.
- `scripts/tex_core.py`, `scripts/preview.py` - shared tileable-noise helpers and
  the lit preview sheet (also used by `trees/`).
- `workflows/` - the ComfyUI graphs; `workflows/README.md` documents them.
- `comfy_nodes/` - our ComfyUI nodes (`TorusRoll` and friends), symlinked into
  `tools/ComfyUI/custom_nodes/`.

`out/` is not in git: materials are reproducible from their prompt and seed.
The CHORD weights are under a research-only licence.
