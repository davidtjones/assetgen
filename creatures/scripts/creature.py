"""Creature pipeline: a one-line description -> a rigged, animated, textured glTF the game loads.

    pixi run creature <stage> <name>... [--seed N] [--force]

`<name>` is a spec in `creatures/specs/<name>.yaml`. Stages run in this order, and each
reads only what the one before it wrote under `creatures/out/<name>/`, so any stage can be
rerun alone:

1. `concept` - FLUX.2 klein draws the creature side-on, standing, legs apart, on a plain
   background (`workflows/concept_klein.json`) -> `concept.png`.
2. `mesh` - TRELLIS.2, run natively in ComfyUI (`workflows/mesh_trellis2.json`): cut out the
   concept, generate shape and PBR voxels, remesh watertight, decimate to the spec's `tris`,
   unwrap, and bake base colour, metallic/roughness, normal and occlusion from the dense
   surface onto the game mesh -> `mesh.glb` (game mesh, textured) and `high.glb` (the dense
   watertight surface the maps were baked from).
3. `rig` - Blender (`blender/rig_creature.py`): orient and scale the mesh, fit a quadruped
   skeleton to its shape, skin it, key an in-place `idle` and `walk` loop -> `creature.glb`.
4. `check` - Blender (`blender/check_creature.py`): re-import `creature.glb` and measure it
   against the spec (geometry, skin, deformation, foot contact and slip) -> `check.json`,
   plus `sheet.png`, a contact sheet to look at. Exits non-zero when a check fails.
5. `export` - copy `creature.glb` into `../game3d/assets/creatures/` and update its line in
   `creatures.txt` there. Refuses a creature whose last check failed unless `--force`.

`all` runs every stage. ComfyUI is started for the image stages if it is not already up.

Spec keys: `subject` (what to draw), `body` (only `quadruped` today), `height` (metres at the
highest point when standing), `tail_bones`, `tris` (game mesh budget), `walk` (`cycle` seconds,
`stride` metres per step, `lift` metres of foot clearance), `seed`, and optionally `flip: true`
when the rig stage gets head and tail the wrong way round.

The walk is keyed in place. Its ground speed is `stride / cycle`; the manifest carries it so
the game moves the creature at exactly that speed and the feet do not slide.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
SPECS = ROOT / "specs"
OUT = ROOT / "out"
WORKFLOWS = ROOT / "workflows"
BLENDER_SCRIPTS = HERE / "blender"
COMFY = REPO / "tools" / "ComfyUI"

from common.comfy_client import ComfyClient
from common.paths import game3d_assets

KLEIN = {"unet": "flux-2-klein-base-4b.safetensors", "text_encoder": "qwen_3_4b.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 20, "cfg": 5.0}

# Appended to every subject. Side-on with the head to the left and legs apart is what the rig
# stage's skeleton fit expects; a plain background and margin all round keep the cut-out clean
# (a limb touching the frame edge becomes a flat slab in the mesh).
FRAMING = ("full body side view facing left, standing on all four legs in a neutral walking stance, "
           "legs clearly separated, tail visible, whole animal in frame with empty margin on every side, "
           "plain light grey studio background, soft even lighting, realistic game creature concept, sharp focus")
AVOID = ("cropped, cut off, partial body, multiple animals, rider, saddle, text, watermark, frame, border, "
         "shadow on the ground, scenery, grass, motion blur, blurry, lying down, sitting")

TRELLIS = {"unet": "trellis_2_bf16.safetensors", "shape_resolution": 1536, "texture_size": 2048, "pad": 1.0}


# -- spec and outputs ------------------------------------------------------------------------

def load_spec(name: str) -> dict:
    path = SPECS / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"no spec {path}")
    spec = yaml.safe_load(path.read_text())
    spec["name"] = name
    if spec.get("body") != "quadruped":
        sys.exit(f"{name}: body {spec.get('body')!r} is not supported (only quadruped)")
    return spec


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def need(path: Path, stage: str) -> Path:
    if not path.exists():
        sys.exit(f"{path} is missing; run the `{stage}` stage first")
    return path


# -- ComfyUI ---------------------------------------------------------------------------------

def comfy() -> ComfyClient:
    """The ComfyUI client, starting a server for this run if none is up."""
    client = ComfyClient(timeout=120.0)
    if client.is_up():
        return client
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "comfyui.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--disable-dynamic-vram"], cwd=COMFY, stdout=log, stderr=log)
    atexit.register(proc.terminate)
    for _ in range(240):
        if client.is_up():
            return client
        if proc.poll() is not None:
            break
        time.sleep(1)
    sys.exit(f"ComfyUI did not start; see {OUT / 'comfyui.log'}")


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


# -- stages ----------------------------------------------------------------------------------

def stage_concept(spec: dict, seed: int) -> None:
    d = out_dir(spec["name"])
    client = comfy()
    positive = f"{' '.join(spec['subject'].split())}, {FRAMING}"
    negative = ", ".join(filter(None, [spec.get("avoid"), AVOID]))
    values = {**KLEIN, "positive": positive, "negative": negative, "width": 1024, "height": 1024,
              "seed": seed, "prefix": f"creatures/{spec['name']}/concept_s{seed}"}
    t0 = time.time()
    outputs = run_workflow(client, fill((WORKFLOWS / "concept_klein.json").read_text(), values), 600)
    fetch(client, outputs, f"concept_s{seed}", d / f"concept_s{seed}.png")
    shutil.copyfile(d / f"concept_s{seed}.png", d / "concept.png")
    (d / "concept.json").write_text(json.dumps({"seed": seed, "positive": positive, "negative": negative,
                                                "model": KLEIN["unet"], "seconds": round(time.time() - t0, 1)},
                                               indent=2) + "\n")
    print(f"[concept] {spec['name']} seed {seed}: {d / 'concept.png'} ({time.time() - t0:.0f} s)")


def stage_mesh(spec: dict, seed: int) -> None:
    d = out_dir(spec["name"])
    client = comfy()
    concept = need(d / "concept.png", "concept")
    image = client.upload_image(concept, name=f"creature_{spec['name']}_concept.png")["name"]
    prefix = f"creatures/{spec['name']}/mesh_s{seed}"
    values = {**TRELLIS, "image": image, "seed": seed, "tris": int(spec["tris"]),
              **{f"prefix_{k}": f"{prefix}/{k}" for k in ("cutout", "game", "high")}}
    t0 = time.time()
    outputs = run_workflow(client, fill((WORKFLOWS / "mesh_trellis2.json").read_text(), values), 3600)
    fetch(client, outputs, "cutout", d / "cutout.png")
    fetch(client, outputs, "game", d / "mesh.glb")
    fetch(client, outputs, "high", d / "high.glb")
    (d / "mesh.json").write_text(json.dumps({"seed": seed, "backend": "trellis2", **TRELLIS, "tris": spec["tris"],
                                             "seconds": round(time.time() - t0, 1)}, indent=2) + "\n")
    print(f"[mesh] {spec['name']}: {d / 'mesh.glb'} ({time.time() - t0:.0f} s)")


def blender(script: str, *args: str) -> int:
    cmd = ["blender", "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(BLENDER_SCRIPTS / script), "--", *args]
    return subprocess.run(cmd).returncode


def stage_rig(spec: dict, seed: int) -> None:
    d = out_dir(spec["name"])
    need(d / "mesh.glb", "mesh")
    (d / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    if blender("rig_creature.py", str(d / "mesh.glb"), str(d / "spec.json"), str(d / "creature.glb"),
               str(d / "rig.json")) != 0:
        sys.exit(f"[rig] {spec['name']} failed")


def stage_check(spec: dict, seed: int) -> bool:
    d = out_dir(spec["name"])
    need(d / "creature.glb", "rig")
    (d / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (d / "check.json").unlink(missing_ok=True)
    code = blender("check_creature.py", str(d / "creature.glb"), str(d / "spec.json"), str(d / "check.json"),
                   str(d / "frames"))
    if not (d / "check.json").exists():
        sys.exit(f"[check] {spec['name']}: the check script crashed (exit {code})")
    report = json.loads((d / "check.json").read_text())
    contact_sheet(d, spec, report)
    print(f"[check] {spec['name']}: {'PASS' if report['pass'] else 'FAIL'}  sheet: {d / 'sheet.png'}")
    for c in report["checks"]:
        mark = "ok  " if c["pass"] else ("warn" if c.get("warn") else "FAIL")
        print(f"  {mark} {c['name']:<28} {c['value']!s:<14} {c['limit']}")
    return report["pass"]


def contact_sheet(d: Path, spec: dict, report: dict) -> None:
    """`sheet.png`: concept, turntable, skeleton and walk frames in one picture, with the verdict."""
    from PIL import Image, ImageDraw
    f = d / "frames"
    rows = [
        [d / "concept.png", f / "view_front34.png", f / "view_side.png", f / "view_back34.png", f / "view_top.png"],
        [f / "rig_side.png", f / "rig_top.png"] + [f / f"walk_{i}.png" for i in (0, 2, 4)],
        [f / f"walk_{i}.png" for i in (1, 3, 5, 6, 7)],
    ]
    cell, label = 320, 28
    sheet = Image.new("RGB", (cell * 5, label + cell * len(rows)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    failed = [c["name"] for c in report["checks"] if not c["pass"] and not c.get("warn")]
    draw.text((8, 7), f"{spec['name']}: {'PASS' if report['pass'] else 'FAIL ' + ', '.join(failed)}",
              fill=(120, 220, 120) if report["pass"] else (240, 110, 90))
    for r, row in enumerate(rows):
        for c, path in enumerate(row):
            if path.exists():
                sheet.paste(Image.open(path).convert("RGB").resize((cell, cell)), (c * cell, label + r * cell))
    sheet.save(d / "sheet.png")


def stage_export(spec: dict, seed: int, force: bool = False) -> None:
    d = out_dir(spec["name"])
    report = json.loads(need(d / "check.json", "check").read_text())
    if not report["pass"] and not force:
        sys.exit(f"[export] {spec['name']}: last check failed; fix it or pass --force")
    rig = json.loads(need(d / "rig.json", "rig").read_text())
    game = game3d_assets("creatures")
    game.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(d / "creature.glb", game / f"{spec['name']}.glb")
    write_manifest_entry(spec["name"], {
        "height": spec["height"],
        "length": round(rig["length"], 3),
        "walk_speed": round(spec["walk"]["stride"] / spec["walk"]["cycle"], 3),
        "walk_cycle": spec["walk"]["cycle"],
    })
    print(f"[export] {spec['name']} -> {game / (spec['name'] + '.glb')}")


def write_manifest_entry(name: str, values: dict) -> None:
    """One line per species in the game's `creatures/creatures.txt`, which `src/creatures.rs`
    reads at startup: `<name> <name>.glb key=value ...`."""
    path = game3d_assets("creatures") / "creatures.txt"
    lines = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                lines[line.split()[0]] = line
    lines[name] = " ".join([name, f"{name}.glb"] + [f"{k}={v}" for k, v in values.items()])
    header = "# name file height=<m> length=<m> walk_speed=<m/s> walk_cycle=<s>  (written by assetgen creatures)"
    path.write_text("\n".join([header] + [lines[k] for k in sorted(lines)]) + "\n")


STAGES = ["concept", "mesh", "rig", "check", "export"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=STAGES + ["all"])
    p.add_argument("names", nargs="+")
    p.add_argument("--seed", type=int, help="override the spec's seed")
    p.add_argument("--force", action="store_true", help="export even if the last check failed")
    a = p.parse_args()
    failed = []
    for name in a.names:
        spec = load_spec(name)
        seed = a.seed if a.seed is not None else int(spec.get("seed", 0))
        for stage in (STAGES if a.stage == "all" else [a.stage]):
            if stage == "concept":
                stage_concept(spec, seed)
            elif stage == "mesh":
                stage_mesh(spec, seed)
            elif stage == "rig":
                stage_rig(spec, seed)
            elif stage == "check":
                if not stage_check(spec, seed):
                    failed.append(name)
                    if a.stage == "all":
                        break
            elif stage == "export":
                stage_export(spec, seed, a.force)
    if failed:
        sys.exit(f"check failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
