"""Generate PBR ground materials with StableMaterials, headless (no ComfyUI).

The older generator, kept because the ground and bark keepers under `out/sm/`
came from it. New materials come from `gen_material.py`.

    pixi run material-sm <name> [--seeds 0 1 2 3] [--size 512] [--steps 50]

Reads the prompt for `<name>` from `textures/prompts/materials.yaml` and writes
one candidate per seed to `textures/out/sm/<name>_s<seed>/` as `albedo.png`,
`normal.png`, `roughness.png`, `height.png`, plus `textures/out/sm/<name>.jpg`,
a contact sheet of the albedos (each shown tiled 2x2 so seams and repetition
are visible). Pick a keeper by naming it as a layer source in `bake_ground.py`.

Same model and call as the `StableMaterialsSampler` ComfyUI node; this is the
batch path for when you want several seeds side by side.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import DiffusionPipeline
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out" / "sm"
QUALITY = "seamless tileable, top-down orthographic view, even flat lighting, highly detailed PBR material"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=6.0)
    args = ap.parse_args()
    library = yaml.safe_load((HERE.parent / "prompts" / "materials.yaml").read_text())
    pipe = DiffusionPipeline.from_pretrained(
        "gvecchio/StableMaterials", trust_remote_code=True, torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for name in args.names:
        entry = library[name]
        positive, negative = (entry, "") if isinstance(entry, str) else (entry["positive"], entry.get("negative", ""))
        if "tileable" not in positive:
            positive = f"{positive}, {QUALITY}"
        sheet = []
        for seed in args.seeds:
            gen = torch.Generator("cpu").manual_seed(seed)
            out = pipe(positive, negative_prompt=negative or None, guidance_scale=args.cfg, height=args.size,
                       width=args.size, tileable=True, num_inference_steps=args.steps, generator=gen).images[0]
            d = OUT / f"{name}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            out.basecolor.save(d / "albedo.png")
            out.normal.save(d / "normal.png")
            out.roughness.convert("L").save(d / "roughness.png")
            out.height.convert("L").save(d / "height.png")
            tile = np.asarray(out.basecolor.convert("RGB").resize((256, 256), Image.LANCZOS))
            sheet.append(np.tile(tile, (2, 2, 1)))
            print(f"  {name} seed {seed} -> {d}")
        Image.fromarray(np.concatenate(sheet, 1)).save(OUT / f"{name}.jpg", quality=90)


if __name__ == "__main__":
    main()
