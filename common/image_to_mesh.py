"""Image to textured game mesh: a FLUX.2 klein concept, then a TRELLIS.2 mesh, both through ComfyUI.

Shared by `creatures/` (which goes on to rig and animate the mesh) and `props/` (which only
frames it). Each generator writes its own framing into the prompt; this module only runs the
two workflows in `common/workflows/` and saves what they produce:

- [`draw_concept`] - `concept_klein.json` -> `<out>/concept.png` (and `concept_s<seed>.png`,
  `concept.json`).
- [`make_mesh`] - `mesh_trellis2.json`: cut out the concept, generate shape and PBR voxels,
  remesh watertight, decimate to `tris`, unwrap, and bake base colour, metallic/roughness,
  normal and occlusion from the dense surface onto the game mesh -> `<out>/mesh.glb` (game
  mesh, textured), `<out>/high.glb` (the dense surface the maps were baked from),
  `<out>/cutout.png` and `mesh.json`.

ComfyUI is started for the run if it is not already up (`tools/ComfyUI`, log next to the
outputs).
"""

from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common.comfy_client import ComfyClient

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = Path(__file__).resolve().parent / "workflows"
COMFY = REPO / "tools" / "ComfyUI"

KLEIN = {"unet": "flux-2-klein-base-4b.safetensors", "text_encoder": "qwen_3_4b.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 20, "cfg": 5.0}

TRELLIS = {"unet": "trellis_2_bf16.safetensors", "shape_resolution": 1536, "texture_size": 2048, "pad": 1.0}


def comfy(log_dir: Path) -> ComfyClient:
    """The ComfyUI client, starting a server for this run if none is up."""
    client = ComfyClient(timeout=120.0)
    if client.is_up():
        return client
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "comfyui.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--disable-dynamic-vram"],
                            cwd=COMFY, stdout=log, stderr=log)
    atexit.register(proc.terminate)
    for _ in range(240):
        if client.is_up():
            return client
        if proc.poll() is not None:
            break
        time.sleep(1)
    sys.exit(f"ComfyUI did not start; see {log_dir / 'comfyui.log'}")


def fill(template: str, values: dict) -> dict:
    """Substitute `"{{key}}"` placeholders (quotes included) with JSON values."""
    for key, value in values.items():
        template = template.replace(f'"{{{{{key}}}}}"', json.dumps(value))
    workflow = json.loads(template)
    return {k: v for k, v in workflow.items() if not k.startswith("_")}


def run_workflow(client: ComfyClient, workflow: dict, timeout: float) -> list[dict]:
    problems = client.validate(workflow)
    if problems:
        sys.exit("workflow rejected by the server:\n  " + "\n  ".join(problems))
    return client.collect_outputs(client.wait(client.queue(workflow), timeout=timeout))


def fetch(client: ComfyClient, outputs: list[dict], stem: str, dest: Path) -> Path:
    """Download the one output whose filename starts with `stem`."""
    match = [o for o in outputs if o["filename"].startswith(stem) and o.get("type") == "output"]
    if not match:
        sys.exit(f"ComfyUI produced no {stem!r} output (got {[o['filename'] for o in outputs]})")
    return client.download(match[0], dest)


def draw_concept(out: Path, prefix: str, positive: str, negative: str, seed: int, log_dir: Path) -> Path:
    """Draw `positive` with klein at 1024 px -> `<out>/concept.png`. `prefix` names the
    ComfyUI output folder (`<generator>/<name>`)."""
    out.mkdir(parents=True, exist_ok=True)
    client = comfy(log_dir)
    values = {**KLEIN, "positive": positive, "negative": negative, "width": 1024, "height": 1024,
              "seed": seed, "prefix": f"{prefix}/concept_s{seed}"}
    t0 = time.time()
    outputs = run_workflow(client, fill((WORKFLOWS / "concept_klein.json").read_text(), values), 600)
    fetch(client, outputs, f"concept_s{seed}", out / f"concept_s{seed}.png")
    shutil.copyfile(out / f"concept_s{seed}.png", out / "concept.png")
    (out / "concept.json").write_text(json.dumps({"seed": seed, "positive": positive, "negative": negative,
                                                  "model": KLEIN["unet"], "seconds": round(time.time() - t0, 1)},
                                                 indent=2) + "\n")
    return out / "concept.png"


def make_mesh(out: Path, prefix: str, tris: int, seed: int, log_dir: Path, texture_size: int | None = None) -> Path:
    """`<out>/concept.png` -> `<out>/mesh.glb` with `tris` triangles and baked PBR maps of
    `texture_size` px (default `TRELLIS["texture_size"]`)."""
    concept = out / "concept.png"
    if not concept.exists():
        sys.exit(f"{concept} is missing; run the `concept` stage first")
    client = comfy(log_dir)
    image = client.upload_image(concept, name=f"{prefix.replace('/', '_')}_concept.png")["name"]
    settings = {**TRELLIS, **({"texture_size": int(texture_size)} if texture_size else {})}
    mesh_prefix = f"{prefix}/mesh_s{seed}"
    values = {**settings, "image": image, "seed": seed, "tris": int(tris),
              **{f"prefix_{k}": f"{mesh_prefix}/{k}" for k in ("cutout", "game", "high")}}
    t0 = time.time()
    outputs = run_workflow(client, fill((WORKFLOWS / "mesh_trellis2.json").read_text(), values), 3600)
    fetch(client, outputs, "cutout", out / "cutout.png")
    fetch(client, outputs, "game", out / "mesh.glb")
    fetch(client, outputs, "high", out / "high.glb")
    (out / "mesh.json").write_text(json.dumps({"seed": seed, "backend": "trellis2", **settings, "tris": tris,
                                               "seconds": round(time.time() - t0, 1)}, indent=2) + "\n")
    return out / "mesh.glb"
