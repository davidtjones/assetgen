# Textures

What assetgen keeps for tileable materials. The materials themselves (buildings, terrain
layers, bark, rocks) come from matdex (`../matdex`), the material library, which draws them
with its own klein and CHORD workflows on this repo's ComfyUI. The older generators that lived
here (`pixi run material`, `pixi run material-sm`) moved to matdex's `archive/` on 2026-11-26.

- `comfy_nodes/` - our ComfyUI nodes (`TorusRoll`, `CircularVAEDecode`, `ChordMaps` and
  friends), symlinked into `tools/ComfyUI/custom_nodes/`. matdex's workflows need them, so
  keep them working; their module docstring describes each.
- `scripts/tex_core.py` - tileable-noise and map helpers (normal from height, cavity, ...),
  used by the tree bake.

## Notes

`out/` (not in git) holds candidates the old generators drew. The CHORD weights are under a
research-only licence.
