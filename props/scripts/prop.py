"""Prop pipeline: a one-line description -> a static, textured glTF the game loads.

    pixi run prop <stage> <name>... [--seed N] [--force]

A prop is anything placed in the world that does not move on its own: a mushroom, later
stones, items, ruins. `<name>` is a spec in `props/specs/<name>.yaml`. Stages run in this
order, and each reads only what the one before it wrote under `props/out/<name>/`, so any
stage can be rerun alone:

1. `concept` - FLUX.2 klein draws the object alone, three-quarter view, on a plain background
   -> `concept.png` (`common/image_to_mesh.py`, shared with `creatures/`).
2. `mesh` - TRELLIS.2 makes a textured game mesh from it, decimated to the spec's `tris`, with
   baked PBR maps of `texture_size` px -> `mesh.glb`.
3. `frame` - Blender (`blender/frame_prop.py`): scale to the spec's `height`, stand on y = 0,
   centre over the origin -> `prop.glb`; measure and check it -> `frame.json`; and write
   `sheet.png`, the concept beside four renders, to look at. Exits non-zero when a check fails.
4. `export` - copy `prop.glb` into `../game3d/assets/props/` and update its line in
   `props.txt` there. Refuses a prop whose last frame check failed unless `--force`.

`all` runs every stage. ComfyUI is started for the image stages if it is not already up.

Spec keys: `subject` (what to draw), `height` (metres, ground to top), `tris` (game mesh
budget), `texture_size` (baked map size, default 1024), `seed`, and optionally `avoid` (extra
negative prompt).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common.image_to_mesh import draw_concept, make_mesh
from common.paths import game3d_assets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPECS = ROOT / "specs"
OUT = ROOT / "out"
BLENDER_SCRIPTS = HERE / "blender"

# Appended to every subject. One object, seen from a little above, whole and clear of the
# frame on a plain background, so the cut-out is clean and TRELLIS sees its top and side.
FRAMING = ("single object, three-quarter view from slightly above, whole object in frame with empty margin "
           "on every side, plain light grey studio background, soft even lighting, realistic game asset "
           "concept, sharp focus")
AVOID = ("cropped, cut off, multiple objects, group, cluster, text, watermark, frame, border, "
         "shadow on the ground, scenery, soil, grass, leaves, moss, hand, blurry")


def load_spec(name: str) -> dict:
    path = SPECS / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"no spec {path}")
    spec = yaml.safe_load(path.read_text())
    spec["name"] = name
    return spec


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def need(path: Path, stage: str) -> Path:
    if not path.exists():
        sys.exit(f"{path} is missing; run the `{stage}` stage first")
    return path


def stage_concept(spec: dict, seed: int) -> None:
    d = out_dir(spec["name"])
    positive = f"{' '.join(spec['subject'].split())}, {FRAMING}"
    negative = ", ".join(filter(None, [spec.get("avoid"), AVOID]))
    t0 = time.time()
    path = draw_concept(d, f"props/{spec['name']}", positive, negative, seed, OUT)
    print(f"[concept] {spec['name']} seed {seed}: {path} ({time.time() - t0:.0f} s)")


def stage_mesh(spec: dict, seed: int) -> None:
    d = out_dir(spec["name"])
    t0 = time.time()
    path = make_mesh(d, f"props/{spec['name']}", int(spec["tris"]), seed, OUT,
                     texture_size=int(spec.get("texture_size", 1024)))
    print(f"[mesh] {spec['name']}: {path} ({time.time() - t0:.0f} s)")


def stage_frame(spec: dict, seed: int) -> bool:
    d = out_dir(spec["name"])
    need(d / "mesh.glb", "mesh")
    (d / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (d / "frame.json").unlink(missing_ok=True)
    cmd = ["blender", "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(BLENDER_SCRIPTS / "frame_prop.py"), "--",
           str(d / "mesh.glb"), str(d / "spec.json"), str(d / "prop.glb"), str(d / "frame.json"), str(d / "views")]
    code = subprocess.run(cmd, capture_output=True, text=True)
    if not (d / "frame.json").exists():
        sys.stderr.write(code.stdout[-4000:] + code.stderr[-4000:])
        sys.exit(f"[frame] {spec['name']}: the frame script crashed (exit {code.returncode})")
    report = json.loads((d / "frame.json").read_text())
    contact_sheet(d, spec, report)
    print(f"[frame] {spec['name']}: {'PASS' if report['pass'] else 'FAIL'}  sheet: {d / 'sheet.png'}")
    for c in report["checks"]:
        print(f"  {'ok  ' if c['pass'] else 'FAIL'} {c['name']:<20} {c['value']!s:<10} {c['limit']}")
    return report["pass"]


def contact_sheet(d: Path, spec: dict, report: dict) -> None:
    """`sheet.png`: the concept and four renders in one row, with the verdict."""
    from PIL import Image, ImageDraw
    v = d / "views"
    row = [d / "concept.png"] + [v / f"view_{n}.png" for n in ("front34", "side", "top", "under")]
    cell, label = 320, 28
    sheet = Image.new("RGB", (cell * len(row), label + cell), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    failed = [c["name"] for c in report["checks"] if not c["pass"]]
    draw.text((8, 7), f"{spec['name']}: {'PASS' if report['pass'] else 'FAIL ' + ', '.join(failed)}  "
                      f"h={report['height']} m r={report['radius']} m tris={report['tris']}",
              fill=(120, 220, 120) if report["pass"] else (240, 110, 90))
    for c, path in enumerate(row):
        if path.exists():
            sheet.paste(Image.open(path).convert("RGB").resize((cell, cell)), (c * cell, label))
    sheet.save(d / "sheet.png")


def stage_export(spec: dict, seed: int, force: bool = False) -> None:
    d = out_dir(spec["name"])
    report = json.loads(need(d / "frame.json", "frame").read_text())
    if not report["pass"] and not force:
        sys.exit(f"[export] {spec['name']}: last frame check failed; fix it or pass --force")
    game = game3d_assets("props")
    game.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(d / "prop.glb", game / f"{spec['name']}.glb")
    write_manifest_entry(spec["name"], {"height": report["height"], "radius": report["radius"]})
    print(f"[export] {spec['name']} -> {game / (spec['name'] + '.glb')}")


def write_manifest_entry(name: str, values: dict) -> None:
    """One line per prop in the game's `props/props.txt`: `<name> <name>.glb key=value ...`."""
    path = game3d_assets("props") / "props.txt"
    lines = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                lines[line.split()[0]] = line
    lines[name] = " ".join([name, f"{name}.glb"] + [f"{k}={v}" for k, v in values.items()])
    header = "# name file height=<m> radius=<m>  (written by assetgen props)"
    path.write_text("\n".join([header] + [lines[k] for k in sorted(lines)]) + "\n")


STAGES = ["concept", "mesh", "frame", "export"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=STAGES + ["all"])
    p.add_argument("names", nargs="+")
    p.add_argument("--seed", type=int, help="override the spec's seed")
    p.add_argument("--force", action="store_true", help="export even if the last frame check failed")
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
            elif stage == "frame":
                if not stage_frame(spec, seed):
                    failed.append(name)
                    if a.stage == "all":
                        break
            elif stage == "export":
                stage_export(spec, seed, a.force)
    if failed:
        sys.exit(f"frame check failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
