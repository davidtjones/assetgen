"""Concepts: photograph a site in the running game, then have an image model draw a structure there.

    pixi run concept <site> [--brief fort|<file.yaml>] [--views a,b] [--seeds 0,1] [--photos <dir>]
                            [--out <dir>] [--full]

A builder (an agent, or David) runs this before it builds: the concept is what it builds
toward. `<site>` is a place of the loaded world (`place.list` in the game) or a structure
(`build.list`).

1. **Photos.** Over the game's console socket (`common/game_console.py`), `site.shots <site>`
   takes the same seven views every time (approach, north, east, south, west, above, inside;
   `../game3d/src/world/site.rs`) -> `concepts/out/<site>/photos/<n>-<view>.png`, and its reply
   -> `photos/site.txt`. The game must be running. `--photos <dir>` uses photos already taken
   instead, and needs no game.
2. **Concepts.** For each view and seed, Qwen-Image-Edit-2511 (through ComfyUI) draws the
   brief's subject into that view's photo -> `concepts/out/<site>/<brief>/<view>_s<seed>.png`.
   Each image is its own design: the model keeps a photo's camera and land well, but it cannot
   redraw one concept from another camera (given a concept as a second picture, it copies it).
3. **Sheet.** `<brief>/sheet.png`: one row per view, the photo and then each seed's concept.
   `<brief>/concept.json` records the brief, each prompt, the model settings and the times.

`--out <dir>` puts the concepts, sheet and record in `<dir>` instead (and the photos in
`<dir>/photos`, unless `--photos` is given): a builder keeps them with the rest of its build.

A brief is `concepts/briefs/<name>.yaml`, or any YAML file given by path: `subject`, `style`,
`prompt` (with `{subject}` and `{style}` filled in), `views` and `seeds`. The words live there,
not here; see `concepts/README.md`.

The model is Qwen-Image-Edit-2511 (int8) with the Lightning 4-step LoRA: about 10 s an image
on the 3090 once loaded. `--full` runs it without the LoRA, 40 steps at cfg 3: about 135 s,
a little crisper. ComfyUI is started for the run if it is not already up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from common.comfy_client import ComfyClient
from common.game_console import GameConsole, GameError
from common.image_to_mesh import comfy, fetch, run_workflow

ROOT = Path(__file__).resolve().parent.parent
BRIEFS = ROOT / "briefs"
OUT = ROOT / "out"

VIEWS = ["approach", "north", "east", "south", "west", "above", "inside"]

QWEN = {
    "unet": "qwen_image_edit_2511_int8_convrot.safetensors",
    "text_encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
    "lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "shift": 3.1,
    "megapixels": 0.92,  # 1280 x 720
}
FAST = {"steps": 4, "cfg": 1.0, "lora": True}
FULL = {"steps": 40, "cfg": 3.0, "lora": False}


def edit_workflow(photo: str, prompt: str, seed: int, prefix: str, settings: dict, mask: str | None = None) -> dict:
    """The ComfyUI graph: Qwen-Image-Edit-2511 redraws `photo` (an uploaded input image) as the
    prompt says. This follows ComfyUI's own template
    for the model (`image_qwen_image_edit_2511_int8`): the photo is scaled to about a
    megapixel, encoded both as the starting latent and as Picture 1 of the prompt, and
    sampled with AuraFlow's shift and CFG normalisation. The negative prompt is empty, and at
    cfg 1 it is not used at all. With `mask` (an uploaded grey picture, white where the photo
    may change), only that part is redrawn: at every step the rest is put back to the photo,
    so the model paints inside the mask knowing what stands round it."""
    q = QWEN
    g: dict = {
        "photo": {"class_type": "LoadImage", "inputs": {"image": photo}},
        "scaled": {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["photo", 0], "upscale_method": "lanczos", "megapixels": q["megapixels"], "resolution_steps": 32}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": q["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": q["shift"]}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "start": {"class_type": "VAEEncode", "inputs": {"pixels": ["scaled", 0], "vae": ["vae", 0]}},
    }
    for name, text in (("positive", prompt), ("negative", "")):
        g[name] = {"class_type": "TextEncodeQwenImageEditPlus",
                   "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], "image1": ["scaled", 0]}}
        g[name + "_refs"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                             "inputs": {"conditioning": [name, 0], "reference_latents_method": "index_timestep_zero"}}
    model = ["norm", 0]
    if settings["lora"]:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": q["lora"], "strength_model": 1.0}}
        model = ["lora", 0]
    latent = ["start", 0]
    if mask:
        g["mask"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
        g["masked"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": latent, "mask": ["mask", 0]}}
        latent = ["masked", 0]
    g["sample"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["positive_refs", 0], "negative": ["negative_refs", 0], "latent_image": latent,
        "seed": seed, "steps": settings["steps"], "cfg": settings["cfg"], "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return g


def brief_path(brief: str) -> Path:
    """`--brief`: a name in `concepts/briefs/`, or a path to a YAML file."""
    if brief.endswith((".yaml", ".yml")) or "/" in brief:
        return Path(brief).resolve()
    return BRIEFS / f"{brief}.yaml"


def load_brief(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"no brief {path} (briefs: {', '.join(p.stem for p in sorted(BRIEFS.glob('*.yaml')))})")
    brief = yaml.safe_load(path.read_text())
    missing = [k for k in ("subject", "style", "prompt", "views", "seeds") if k not in brief]
    if missing:
        sys.exit(f"{path}: missing {', '.join(missing)}")
    return brief


def words(text: str, brief: dict) -> str:
    """A prompt from the brief: one line, with `{subject}` and `{style}` filled in."""
    flat = {k: " ".join(str(brief[k]).split()) for k in ("subject", "style")}
    return " ".join(text.split()).format(**flat)


def take_photos(site: str, dest: Path) -> None:
    """`site.shots` in the running game -> `dest/<n>-<view>.png` and `dest/site.txt`.

    The game and ComfyUI share the GPU, and the game quits when it runs out of memory, so a
    running ComfyUI is first asked to unload its models."""
    comfy_up = ComfyClient()
    if comfy_up.is_up():
        comfy_up.free()
    try:
        with GameConsole() as game:
            reply = game.run(f'site.shots {site} "{dest}"')
    except GameError as e:
        sys.exit(f"[photos] {e}")
    (dest / "site.txt").write_text(reply + "\n")
    print(f"[photos] {reply}")


def photo_of(photos: Path, view: str) -> Path:
    path = photos / f"{VIEWS.index(view) + 1}-{view}.png"
    if not path.exists():
        sys.exit(f"no photo {path}; take the photos first (run without --photos, with the game running)")
    return path


def contact_sheet(out: Path, photos: Path, views: list[str], seeds: list[int]) -> Path:
    """One row per view: the photo, then each seed's concept, labelled."""
    from PIL import Image, ImageDraw
    w, h, label = 480, 270, 22
    sheet = Image.new("RGB", (w * (1 + len(seeds)), (h + label) * len(views)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for r, view in enumerate(views):
        cells = [(photo_of(photos, view), f"{view}: photo")] + [(out / f"{view}_s{s}.png", f"{view}: seed {s}") for s in seeds]
        for c, (path, text) in enumerate(cells):
            x, y = c * w, r * (h + label)
            draw.text((x + 6, y + 5), text, fill=(220, 220, 220))
            if path.exists():
                sheet.paste(Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS), (x, y + label))
    path = out / "sheet.png"
    sheet.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="a place of the loaded world, or a structure")
    ap.add_argument("--brief", default="fort", help="a name in concepts/briefs/ (default: fort), or a path to a brief YAML")
    ap.add_argument("--views", help="comma-separated views, overriding the brief's")
    ap.add_argument("--seeds", help="comma-separated seeds, overriding the brief's")
    ap.add_argument("--photos", type=Path, help="use the photos in this directory instead of taking new ones")
    ap.add_argument("--out", type=Path, help="write the concepts here (default: concepts/out/<site>/<brief>)")
    ap.add_argument("--full", action="store_true", help="40 steps without the Lightning LoRA (slower, a little crisper)")
    args = ap.parse_args()

    path = brief_path(args.brief)
    brief = load_brief(path)
    name = path.stem
    views = args.views.split(",") if args.views else list(brief["views"])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in brief["seeds"]]
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        sys.exit(f"unknown views {unknown}; site.shots takes {', '.join(VIEWS)}")
    settings = FULL if args.full else FAST

    out = args.out.resolve() if args.out else OUT / args.site / name
    photos = args.photos.resolve() if args.photos else (out / "photos" if args.out else OUT / args.site / "photos")
    if not args.photos:
        photos.mkdir(parents=True, exist_ok=True)
        take_photos(args.site, photos)

    out.mkdir(parents=True, exist_ok=True)
    client = comfy(out)
    uploaded = {v: client.upload_image(photo_of(photos, v), name=f"concept_{args.site}_{v}.png")["name"] for v in views}

    record = {"site": args.site, "brief": str(path), "photos": str(photos), "model": {**QWEN, **settings},
              "prompt": words(brief["prompt"], brief), "images": []}
    prompt = words(brief["prompt"], brief)
    for view in views:
        for seed in seeds:
            stem = f"{view}_s{seed}"
            t0 = time.time()
            workflow = edit_workflow(uploaded[view], prompt, seed, f"concepts/{args.site}_{name}_{stem}", settings)
            fetch(client, run_workflow(client, workflow, 1800), f"{args.site}_{name}_{stem}", out / f"{stem}.png")
            took = time.time() - t0
            record["images"].append({"view": view, "seed": seed, "file": f"{stem}.png", "seconds": round(took, 1)})
            print(f"[concept] {args.site} {view} seed {seed}: {out / (stem + '.png')} ({took:.0f} s)")

    (out / "concept.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[sheet] {contact_sheet(out, photos, views, seeds)}")


if __name__ == "__main__":
    main()
