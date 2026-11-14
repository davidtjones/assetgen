"""ComfyUI nodes for the texture pipeline. Lives in the repo; symlinked into
`tools/ComfyUI/custom_nodes/assetgen_textures`.

`TorusRoll` makes a transformer image model (FLUX.2 klein) produce an image
that tiles. Such a model has no convolutions to pad circularly, so instead the
latent is treated as a torus: before every denoising call it is rolled by a
random offset, and the prediction is rolled back. The picture's edges land
somewhere new in the frame at each step, so the model keeps mending them as
ordinary interior. Decode with a circular VAE (`CircularVAEDecode`) so the last
step does not put a seam back. When the model is editing a reference image
(`ReferenceLatent`), the reference is rolled by the same offset, so the edit
stays lined up with it.

`ChordMaps` runs CHORD on an image at the image's own size. CHORD's own node
(`ChordMaterialEstimation`) scales every input to 1024 and the maps back,
which leaves a faint regular line pattern in them when the input is another
size, as it is when a tile is estimated with a margin round it
(`gen_material.decompose`). The model is all convolutions and attention, so it
takes any size that is a multiple of 64.
"""

import numpy as np
import torch


class TorusRoll:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tiling": (["enable", "x_only", "y_only", "disable"],),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "assetgen/textures"

    def run(self, model, tiling, seed):
        if tiling == "disable":
            return (model,)
        roll_x, roll_y = tiling in ("enable", "x_only"), tiling in ("enable", "y_only")
        rng = np.random.default_rng(seed)
        shifts = {}  # one offset per noise level, shared by every call at that level

        def wrapper(apply_model, args):
            x, t, c = args["input"], args["timestep"], args["c"]
            key = round(float(t.flatten()[0]), 6)
            if key not in shifts:
                h, w = x.shape[-2:]
                shifts[key] = (int(rng.integers(h)) if roll_y else 0, int(rng.integers(w)) if roll_x else 0)
            dy, dx = shifts[key]
            refs = c.get("ref_latents")
            if refs is not None:
                same = lambda r: r.shape[-2:] == x.shape[-2:]  # noqa: E731
                c = {**c, "ref_latents": [torch.roll(r, (dy, dx), (-2, -1)) if same(r) else r for r in refs]}
            out = apply_model(torch.roll(x, (dy, dx), (-2, -1)), t, **c)
            return torch.roll(out, (-dy, -dx), (-2, -1))

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


class ChordMaps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"chord_model": ("chord_model",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "roughness", "metalness")
    FUNCTION = "run"
    CATEGORY = "assetgen/textures"

    def run(self, chord_model, image):
        import comfy.model_management

        if image.shape[1] % 64 or image.shape[2] % 64:
            raise ValueError(f"ChordMaps: image must be a multiple of 64 each way, got {tuple(image.shape[1:3])}")
        comfy.model_management.load_models_gpu([chord_model])
        model = chord_model.model
        device = next(model.parameters()).device
        # As CHORD's node does: every convolution wraps round the image.
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.padding_mode = "circular"
        with torch.no_grad(), torch.autocast(device_type=device.type):
            out = model(image.permute(0, 3, 1, 2).to(device))
        maps = [out[key].float().cpu() for key in ("basecolor", "normal", "roughness", "metalness")]
        return tuple(m.permute(0, 2, 3, 1) if m.ndim == 4 else m for m in maps)


NODE_CLASS_MAPPINGS = {"TorusRoll": TorusRoll, "ChordMaps": ChordMaps}
NODE_DISPLAY_NAME_MAPPINGS = {"TorusRoll": "Torus Roll (tileable transformer)",
                              "ChordMaps": "CHORD maps at the image's own size"}
