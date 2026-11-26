# ComfyUI nodes

Our own ComfyUI nodes. Each package here is symlinked into `tools/ComfyUI/custom_nodes/`,
and ComfyUI loads it when it starts, so a change here needs a ComfyUI restart
(`../services/svc restart comfyui`; check matdex's `GET /jobs` is idle first).

- `assetgen_tiling/` - `TorusRoll` makes FLUX.2 klein draw an image that tiles, and
  `ChordMaps` runs CHORD at the image's own size. matdex's workflows (`klein_tile`,
  `klein_edit`, `chord_maps`) use both, so every material drawing depends on them. The
  module docstring explains each. `CircularVAEDecode`, which those workflows also use, comes
  from the `ComfyUI-seamless-tiling` custom node, not from here.

To add a package, put it here and link it:

```sh
ln -s ../../../comfy_nodes/<name> tools/ComfyUI/custom_nodes/<name>
```

The CHORD weights (`tools/ComfyUI/models/checkpoints/chord_v1.safetensors`) are under a
research-only licence.
