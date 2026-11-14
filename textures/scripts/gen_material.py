"""Generate tileable PBR materials from a text prompt.

    pixi run material <name>... [--seeds 0 1] [--size 1024] [--decomposer chord] [--maps-only]

Two models, run through ComfyUI (started here if it is not already up):

1. **Image.** FLUX.2 klein draws a photo of the surface from the `subject` of
   `<name>` in `textures/prompts/materials.yaml`. `workflows/klein_tile.json`
   makes it tile (the `TorusRoll` node in `comfy_nodes/`). An entry with
   `edit_of: <other name>` is a variant instead: klein is handed the same
   seed of the other material and `subject` says what to change
   (`workflows/klein_edit.json`), so the two line up and can be blended.
2. **Maps.** A decomposer reads that photo and estimates albedo, normal and
   roughness. Each decomposer is one workflow in `DECOMPOSERS` taking an image
   and saving maps, so another model can be dropped in. Today there is one:
   `chord`, Ubisoft's CHORD (research-only licence). Height is solved here
   from the normal map so that it tiles.

Per seed it writes `textures/out/mat/<name>_s<seed>/` with `albedo.png`,
`normal.png` (OpenGL), `roughness.png`, `metalness.png`, `height.png` and
`source.png` (the photo before decomposition). `textures/out/mat/<name>.jpg`
is the lit preview sheet of every seed (`preview.py`): judge candidates there.
About half a minute per candidate on the 3090.
"""

import argparse
import atexit
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import preview
import tex_core as tc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE.parent / "out" / "mat"
WORKFLOWS = HERE.parent / "workflows"
COMFY = REPO / "tools" / "ComfyUI"

sys.path.insert(0, str(REPO / "common"))
from comfy_client import ComfyClient  # noqa: E402

# Appended to every subject: a texture is a surface seen square-on with no lighting of its own.
FRAMING = "flat frontal view filling the whole frame edge to edge, even diffuse lighting, no cast shadows"
AVOID = ("blurry, text, watermark, frame, border, vignette, perspective, depth of field, sky, "
         "object, transparent background")
KLEIN = {"unet": "flux-2-klein-base-4b.safetensors", "text_encoder": "qwen_3_4b.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 20, "cfg": 5.0}
TILING = {"both": "enable", "x": "x_only", "y": "y_only", "none": "disable"}

# Wrapped pixels added round a tile before its maps are estimated (see `decompose`).
PAD = 128

# name -> (workflow, maps it must save, whether its normal map is DirectX (green down)).
DECOMPOSERS = {
    "chord": ("chord_maps.json", {"albedo", "normal", "roughness", "metalness"}, True),
}


def _fill(template: str, values: dict) -> dict:
    for key, value in values.items():
        template = template.replace(f'"{{{{{key}}}}}"', json.dumps(value))
    return json.loads(template)


def _server() -> ComfyClient:
    """The ComfyUI client, starting a server for this run if none is up."""
    client = ComfyClient()
    if client.is_up():
        return client
    log = open(OUT / "comfyui.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--disable-dynamic-vram"], cwd=COMFY, stdout=log, stderr=log)
    atexit.register(proc.terminate)
    for _ in range(180):
        if client.is_up():
            return client
        if proc.poll() is not None:
            break
        time.sleep(1)
    sys.exit(f"ComfyUI did not start; see {OUT / 'comfyui.log'}")


def _run(client: ComfyClient, workflow: dict, dest: Path) -> list[Path]:
    """Queue `workflow`, wait, and download each saved image to `dest`, named
    by its filename prefix (`albedo_00001_.png` -> `albedo.png`)."""
    entry = client.wait(client.queue(workflow))
    return [client.download(item, dest / f"{item['filename'].split('_')[0]}.png")
            for item in client.collect_outputs(entry)]


def generate(client: ComfyClient, name: str, entry: dict, seed: int, size: int, decomposer: str) -> Path:
    d = OUT / f"{name}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    scratch = f"assetgen_material/{name}_s{seed}"
    avoid = ", ".join(filter(None, [entry.get("avoid"), AVOID]))
    values = {**KLEIN, "positive": f"{' '.join(entry['subject'].split())}, {FRAMING}", "negative": avoid,
              "width": size, "height": size, "seed": seed, "tiling": TILING[entry.get("tile", "both")],
              "prefix": f"{scratch}/source"}
    workflow = "klein_tile.json"
    if "edit_of" in entry:
        # A variant: klein repaints the same seed of another material, keeping its layout.
        base = OUT / f"{entry['edit_of']}_s{seed}" / "source.png"
        if not base.exists():
            sys.exit(f"{name}: generate {entry['edit_of']} seed {seed} first ({base} is missing)")
        values["image"] = client.upload_image(base, name=f"assetgen_material_{base.parent.name}_source.png")["name"]
        workflow = "klein_edit.json"
    source, = _run(client, _fill((WORKFLOWS / workflow).read_text(), values), d)
    decompose(client, source, d, decomposer)
    return d


def _unpad(image: np.ndarray) -> np.ndarray:
    """The middle tile of a map estimated with `PAD` wrapped pixels round it,
    made to wrap exactly: within `PAD` of each edge the tile is cross-faded
    with the estimate of the same pixels one tile away, half and half at the
    edge itself."""
    for axis in (0, 1):
        n = image.shape[axis] - 2 * PAD
        image = np.moveaxis(image, axis, 0)
        tile = image[PAD:PAD + n].copy()
        share = (0.5 * (1.0 - (np.arange(PAD) + 0.5) / PAD)).reshape((-1,) + (1,) * (image.ndim - 1))
        tile[:PAD] = tile[:PAD] * (1 - share) + image[PAD + n:PAD + n + PAD] * share
        tile[n - PAD:] = tile[n - PAD:] * (1 - share[::-1]) + image[:PAD] * share[::-1]
        image = np.moveaxis(tile, 0, axis)
    return image


def _degrid(image: np.ndarray, cell: int = 8) -> np.ndarray:
    """Remove the model's working grid from a map. A latent diffusion model
    computes on cells of `cell` pixels and leaves faint dead-straight lines at
    that spacing, which show as stripes on smooth ground under a low sun. A
    pattern that repeats exactly every `cell` pixels lives in a handful of
    frequencies and nothing in a real surface does, so those are zeroed."""
    spectrum = np.fft.fft2(image, axes=(0, 1))
    mean = spectrum[0, 0].copy()
    spectrum[::image.shape[0] // cell, ::image.shape[1] // cell] = 0
    spectrum[0, 0] = mean
    return np.fft.ifft2(spectrum, axes=(0, 1)).real.astype(np.float32)


def decompose(client: ComfyClient, source: Path, d: Path, decomposer: str) -> None:
    """Estimate the maps of the image `source` into the material dir `d`."""
    workflow, maps, directx = DECOMPOSERS[decomposer]
    d.mkdir(parents=True, exist_ok=True)

    # A decomposer sees one tile, not the tiling, and its maps go wrong in a
    # strip along the image's edge (a visible grid on smooth ground). So it is
    # given the tile with its wrap-around neighbours attached, `PAD` pixels of
    # them on every side, and only the middle is kept. Each of the middle's
    # edges then has its far side's estimate lying just outside it, and the
    # two are cross-faded so the kept tile wraps without a step.
    photo = np.pad(np.asarray(Image.open(source).convert("RGB")), ((PAD, PAD), (PAD, PAD), (0, 0)), mode="wrap")
    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(photo).save(Path(tmp) / "padded.png")
        uploaded = client.upload_image(Path(tmp) / "padded.png", name=f"assetgen_material_{d.name}.png")
        values = {"image": uploaded["name"], "prefix": f"assetgen_material/{d.name}_{decomposer}"}
        saved = _run(client, _fill((WORKFLOWS / workflow).read_text(), values), d)
    missing = maps - {p.stem for p in saved}
    if missing:
        sys.exit(f"{decomposer} did not save {sorted(missing)} for {d.name}")
    for name in maps:
        image = np.asarray(Image.open(d / f"{name}.png").convert("RGB")).astype(np.float32)
        if image.shape != photo.shape:
            sys.exit(f"{decomposer} resized {name} for {d.name}: {image.shape[:2]} from {photo.shape[:2]}")
        image = _degrid(_unpad(image))
        Image.fromarray(np.clip(image + 0.5, 0, 255).astype(np.uint8)).save(d / f"{name}.png")

    normal = np.asarray(Image.open(d / "normal.png").convert("RGB")).astype(np.float32) / 127.5 - 1.0
    if directx:
        normal[..., 1] *= -1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    Image.fromarray(tc.to8(normal * 0.5 + 0.5)).save(d / "normal.png")
    Image.fromarray(tc.to8(tc.height_from_normal(normal))).save(d / "height.png")
    for grey in ("roughness", "metalness"):
        Image.open(d / f"{grey}.png").convert("L").save(d / f"{grey}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--decomposer", choices=sorted(DECOMPOSERS), default="chord")
    ap.add_argument("--maps-only", action="store_true",
                    help="keep each seed's source.png and only estimate its maps again")
    args = ap.parse_args()
    library = yaml.safe_load((HERE.parent / "prompts" / "materials.yaml").read_text())
    for name in args.names:
        if not isinstance(library.get(name), dict) or "subject" not in library[name]:
            sys.exit(f"{name}: no `subject` in textures/prompts/materials.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    client = _server()
    for name in args.names:
        dirs = []
        for seed in args.seeds:
            start = time.time()
            if args.maps_only:
                dirs.append(OUT / f"{name}_s{seed}")
                decompose(client, dirs[-1] / "source.png", dirs[-1], args.decomposer)
            else:
                dirs.append(generate(client, name, library[name], seed, args.size, args.decomposer))
            print(f"  {name} seed {seed} -> {dirs[-1]} ({time.time() - start:.0f}s)", flush=True)
        print(f"  preview -> {preview.sheet(dirs, OUT / f'{name}.jpg')}", flush=True)


if __name__ == "__main__":
    main()
