"""Paint a blockout: pictures of a build in the game, repainted as the finished place.

    pixi run paint <plan.toml> [--out <dir>] [--views a,b] [--seeds 0,1] [--pictures <dir>]
                               [--brief paint|<file.yaml>] [--words] [--full]

A builder runs this on its blockout (a site plan built with `plan.build`, its buildings still
plain blocks) to see the place finished, from the same cameras, on the real land. Unlike a
concept drawn on bare land, every painting keeps the blockout's layout, so the views show one
place from every side. The painting is what a layout is judged and tuned by.

`<plan.toml>` is the site plan: a path from here, or from the game's root (as `plan.build`
takes it). Its `site` is the structure to paint.

1. **Pictures.** Over the game's console socket, with the blocks' labels off:
   `site.shots <site>` (approach, north, east, south, west, above, inside) and `site.eye` for
   each of the plan's `[[view]]`s (a player's eye on the ground) -> `<out>/blockout/<n>-<view>.png`,
   each with a `.json` from `site.seen` saying what it shows: how many of the structure's pieces,
   and for each block how much, where on the picture, and whether its door is in sight. The game
   must be running with the blockout built. The blocks are drawn plain white and unnamed for the
   pictures (`blockout.colours off`, `blockout.labels off`): the model keeps a bright coloured
   box as it is. `--pictures <dir>` reuses pictures already taken.
2. **Words.** One prompt per picture (`words_for`): what the site is for (its programme's role),
   then one sentence per part in sight, biggest first, from the plan's finishes: the walls and
   towers (the plan's `finish`, a wall's or tower's own, else the brief's `default_finish`), and
   each block, a white box standing for its building, by where it stands, with its door: in
   sight, on a side out of sight, or none. Doors and gates are only where the blockout has a dark opening or an arch.
3. **Paintings.** Qwen-Image-Edit-2511 repaints each picture per seed ->
   `<out>/painted/<view>_s<seed>.png`.
4. **Sheet.** `<out>/sheet.png`: one row per view, the blockout and then each painting.
   `<out>/paint.json` records the plan, each view's prompt, the model settings and the times.

`--out` defaults to `concepts/out/<site>/paint`. The brief (`concepts/briefs/paint.yaml`) holds
the instruction's frame and the default finish.

The model and its settings are the concept tool's (`concept.py`): about 20 s an image beside
the running game. ComfyUI is asked to unload before the pictures and is started for the
paintings if it is not up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import yaml

from common.comfy_client import ComfyClient
from common.game_console import GameConsole, GameError
from common.image_to_mesh import comfy, fetch, run_workflow
from common.paths import game3d_dir
from concepts.scripts.concept import BRIEFS, FAST, FULL, OUT, QWEN, edit_workflow

SHOTS = ["approach", "north", "east", "south", "west", "above", "inside"]
# Smaller than this share of the picture, a part is not worth a sentence.
TINY = 0.0005


def find_plan(path: str) -> Path:
    """A plan path from here, else from the game's root."""
    for p in (Path(path), game3d_dir() / path):
        if p.exists():
            return p.resolve()
    sys.exit(f"no plan {path} (here or in {game3d_dir()})")


def load_brief(name: str) -> tuple[Path, dict]:
    path = Path(name).resolve() if name.endswith((".yaml", ".yml")) or "/" in name else BRIEFS / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"no brief {path}")
    brief = yaml.safe_load(path.read_text())
    missing = [k for k in ("default_finish", "style", "prompt", "doors", "views", "seeds") if k not in brief]
    if missing:
        sys.exit(f"{path}: missing {', '.join(missing)}")
    return path, brief


def flat(text: str) -> str:
    return " ".join(str(text).split())


def subject_of(plan: dict) -> str:
    """What the site is for: the first sentence of its programme's role, else the programme's
    own words, else "a fort"."""
    programme = plan.get("programme")
    if not programme:
        return "a fort"
    path = find_plan(programme)
    data = tomllib.loads(path.read_text())
    role = data.get("role", {}).get(plan.get("role", ""), {})
    about = flat(role.get("about", "")) or flat(data.get("about", "a fort"))
    first = about.split(". ")[0].rstrip(".")
    # "A pass fort: holds the only way through ..." -> "a pass fort that holds the only way through ..."
    if ": " in first:
        what, does = first.split(": ", 1)
        first = f"{what[0].lower()}{what[1:]}, which {does}"
    return first


def take_pictures(site: str, views: list[dict], dest: Path) -> None:
    """`site.shots` and a `site.eye` per plan view, with the labels off -> `dest`."""
    up = ComfyClient()
    if up.is_up():
        up.free()  # the game and ComfyUI share the GPU
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with GameConsole() as game:
            # Plain white blocks, no names: a bright coloured box is kept as it is.
            switches = {s: game.run(f"blockout.{s}").strip().endswith("on") for s in ("labels", "colours")}
            for s in switches:
                game.run(f"blockout.{s} off")
            try:
                print(f"[pictures] {game.run(f'site.shots {site} \"{dest}\"').splitlines()[0]}")
                for n, v in enumerate(views, start=len(SHOTS) + 1):
                    (x, z), (tx, tz) = v["at"], v["toward"]
                    reply = game.run(f'site.eye {site} {x} {z} {tx} {tz} "{dest}/{n}-{v["name"]}.png"')
                    print(f"[pictures] {v['name']}: {reply.splitlines()[0]}")
            finally:
                for s, on in switches.items():
                    if on:
                        game.run(f"blockout.{s} on")
    except GameError as e:
        sys.exit(f"[pictures] {e}")


def pictures_in(folder: Path) -> dict[str, Path]:
    """The pictures with what they show beside them, by view name: `<n>-<view>.png`."""
    out = {}
    for png in sorted(folder.glob("*.png"), key=lambda p: int(p.stem.split("-", 1)[0]) if p.stem.split("-", 1)[0].isdigit() else 99):
        if "-" in png.stem and png.with_suffix(".json").exists():
            out[png.stem.split("-", 1)[1]] = png
    return out


def where(box: list[float], seen: float | None = None) -> str:
    """Where on the picture a box is, in words, set off by commas when it runs to more than a
    place ("in the middle", "on the left, small and far off,")."""
    left, top, right, bottom = box
    x = (left + right) / 2
    side = "on the left" if x < 0.36 else "on the right" if x > 0.64 else "in the middle"
    area = (right - left) * (bottom - top)
    size = ", small and far off" if area < 0.004 else ""
    hidden = ", partly hidden" if seen is not None and seen < 0.35 else ""
    return f"{side}{size}{hidden}," if size or hidden else side


def area(box: list[float] | None) -> float:
    return 0.0 if not box else (box[2] - box[0]) * (box[3] - box[1])


def words_for(plan: dict, brief: dict, seen: dict) -> str:
    """The prompt for one picture, from the plan, the brief and what the picture shows."""
    parts: list[tuple[float, str]] = []
    pieces = seen["pieces"]
    if pieces["seen"] and area(pieces["box"]) >= TINY:
        finish = flat(plan.get("finish") or brief["default_finish"])
        text = f"The grey walls and towers {where(pieces['box'])} are {finish}."
        for kind in ("wall", "tower"):
            for shape in plan.get(kind, []):
                if shape.get("finish"):
                    text += f" The {shape['name']} {kind} is {flat(shape['finish'])}."
        parts.append((area(pieces["box"]), text))
    blocks = {b["name"]: b for b in plan.get("block", [])}
    for b in seen["blocks"]:
        if not b["box"] or area(b["box"]) < TINY or b["seen"] < 0.05:
            continue
        block = blocks.get(b["name"], {})
        kind = b["kind"] if b["name"] == b["kind"] else f"{b['kind']} (the {b['name']})"
        finish = flat(block.get("finish") or brief["default_finish"])
        text = f"The plain white box {where(b['box'], b['seen'])} stands for a {kind}: paint it as the {kind}, {finish}."
        door = b.get("door")
        if door is None:
            text += " It has no door."
        elif door["seen"]:
            text += " Its door is the dark opening on it."
        else:
            text += " Its door is on a side out of sight, so the walls in view have no door."
        parts.append((area(b["box"]), text))
    parts.sort(key=lambda p: -p[0])
    return flat(brief["prompt"]).format(
        subject=subject_of(plan),
        parts=" ".join(t for _, t in parts),
        doors=flat(brief["doors"]),
        style=flat(brief["style"]),
    )


def contact_sheet(out: Path, pictures: dict[str, Path], views: list[str], seeds: list[int]) -> Path:
    """One row per view: the blockout, then each seed's painting, labelled."""
    from PIL import Image, ImageDraw
    w, h, label = 480, 270, 22
    sheet = Image.new("RGB", (w * (1 + len(seeds)), (h + label) * len(views)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for r, view in enumerate(views):
        cells = [(pictures[view], f"{view}: blockout")] + [(out / "painted" / f"{view}_s{s}.png", f"{view}: seed {s}") for s in seeds]
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
    ap.add_argument("plan", help="the site plan (TOML), from here or from the game's root")
    ap.add_argument("--out", type=Path, help="where to write (default: concepts/out/<site>/paint)")
    ap.add_argument("--views", help="comma-separated views, overriding the brief's (plan views are added unless this is given)")
    ap.add_argument("--seeds", help="comma-separated seeds, overriding the brief's")
    ap.add_argument("--pictures", type=Path, help="use the pictures (and their .json) in this directory instead of taking new ones")
    ap.add_argument("--brief", default="paint", help="a name in concepts/briefs/ (default: paint), or a path to a YAML file")
    ap.add_argument("--words", action="store_true", help="print each view's prompt and stop: no painting")
    ap.add_argument("--full", action="store_true", help="40 steps without the Lightning LoRA (slower, a little crisper)")
    args = ap.parse_args()

    plan_path = find_plan(args.plan)
    plan = tomllib.loads(plan_path.read_text())
    site = plan["site"]
    brief_path, brief = load_brief(args.brief)
    eye_views = plan.get("view", [])
    out = (args.out or OUT / site / "paint").resolve()
    folder = args.pictures.resolve() if args.pictures else out / "blockout"
    if not args.pictures:
        take_pictures(site, eye_views, folder)
    pictures = pictures_in(folder)

    views = args.views.split(",") if args.views else list(brief["views"]) + [v["name"] for v in eye_views]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in brief["seeds"]]
    missing = [v for v in views if v not in pictures]
    if missing:
        sys.exit(f"no pictures for {missing} in {folder} (have: {', '.join(pictures)})")
    prompts = {v: words_for(plan, brief, json.loads(pictures[v].with_suffix(".json").read_text())) for v in views}
    if args.words:
        for v in views:
            print(f"[{v}] {prompts[v]}\n")
        return

    settings = FULL if args.full else FAST
    (out / "painted").mkdir(parents=True, exist_ok=True)
    client = comfy(out)
    record = {"plan": str(plan_path), "site": site, "brief": str(brief_path), "pictures": str(folder),
              "model": {**QWEN, **settings}, "views": {v: {"prompt": prompts[v], "images": []} for v in views}}
    for view in views:
        uploaded = client.upload_image(pictures[view], name=f"paint_{site}_{view}.png")["name"]
        for seed in seeds:
            stem = f"{view}_s{seed}"
            t0 = time.time()
            workflow = edit_workflow(uploaded, prompts[view], seed, f"paint/{site}_{stem}", settings)
            fetch(client, run_workflow(client, workflow, 1800), f"{site}_{stem}", out / "painted" / f"{stem}.png")
            took = time.time() - t0
            record["views"][view]["images"].append({"seed": seed, "file": f"painted/{stem}.png", "seconds": round(took, 1)})
            print(f"[paint] {site} {view} seed {seed}: {out / 'painted' / (stem + '.png')} ({took:.0f} s)")
    (out / "paint.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[sheet] {contact_sheet(out, pictures, views, seeds)}")


if __name__ == "__main__":
    main()
