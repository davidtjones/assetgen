# Textures

Tileable PBR material candidates. Every material the game draws (buildings, terrain layers,
bark, rocks) comes from matdex (`../matdex`), the material library, which moved out of this
repo; it draws through this repo's ComfyUI and its workflows here. The game reads matdex in
place and builds its terrain layers itself when it loads, so nothing here bakes for it.

```text
pixi run material <name>... [--seeds 0 1] [--maps-only]   # generate candidates -> out/mat/
pixi run material-sm <name> [--seeds ...]                 # the older StableMaterials generator -> out/sm/
```

- `prompts/materials.yaml` - hand-written materials (bark, terrain layers): one `subject` per
  name.
- `scripts/gen_material.py` - FLUX.2 klein draws a tileable photo, CHORD
  estimates the maps, both through ComfyUI (started if it is not running). Writes
  `out/mat/<name>_s<seed>/` and a lit preview sheet `out/mat/<name>.jpg`. Its
  docstring is the CLI reference.
- `scripts/gen_material_sm.py` - the older StableMaterials generator, which some library
  materials came from; new materials come from matdex.
- `scripts/tex_core.py`, `scripts/preview.py` - shared tileable-noise helpers and
  the lit preview sheet (also used by `trees/`).
- `workflows/` - the ComfyUI graphs; `workflows/README.md` documents them.
- `comfy_nodes/` - our ComfyUI nodes (`TorusRoll` and friends), symlinked into
  `tools/ComfyUI/custom_nodes/`.

## Notes

`out/` is not in git: materials are reproducible from their prompt and seed.
The CHORD weights are under a research-only licence.
