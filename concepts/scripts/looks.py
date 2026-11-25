"""A building's materials from its painting: what a painted block shows its walls, roof, footing
and frames made of, as materials from the material library, each with the colour and tile size
the game dresses that building's slots in.

    pixi run looks <paint run dir> <block> [--view <view>] [--slots wall roof ...] [--painting 0]
                   [--structure <name>] [--game]

The paint step (`pixi run paint`) decides how each block of a site plan looks. This step turns
that look into materials, so the building put up in the block wears what its painting showed,
and no one designs a material by hand.

1. **The pictures.** From the paint run's record (`paint.json`), the painted view in which the
   block is biggest (or `--view`). It is cropped to the block's box as the game saw it
   (`site.seen`, written beside each blockout picture), with room above for its roof. The
   colour of the game's light is taken out of it first: the blockout picture from the same
   camera shows the block plain white under that light, and a painter keeps its colour (the
   blue of a wall in shade), which the game would add again (`light_on`, `balanced`). Each slot
   is cropped from its own band of the box (`BANDS`): the roof from above it, the walls from its
   lower part, the frames from round the block's door when it is in sight.
2. **The words.** An agent is shown the block and each slot's crop and asked one question: what
   each slot's surface is made of, in the library's words (`textures/library/words.py`: a kind
   and pattern from its vocabulary, the finish, the colour, a detail), and which slots are the
   same material as another (a footing of the walls' stone). Its question and answer are kept
   in `describe/`.
3. **The colour.** Each slot's colour is the painting's own: the median of the pixels in the
   slot's band the painter changed from the game's picture (`painted_colour`), which leaves out
   the sky, the ground and anything in front. Round a door the painter changed the wall too, so
   the frame's colour is the median of the door's upper half (things in front of a door stand
   on the ground).
4. **The material.** The library (`textures/library/`, through its API) gives back a material
   for each slot's words: one it has whose words mean about the same, or a new one it draws
   (about a minute). A slot the agent called the same as another gets that slot's material.
   Every library material is colour-neutral; the building uses it tinted to the slot's colour,
   at the tile size the material's courses give it. Who uses it is recorded in the library as
   `<site>/<block>/<slot>`.
5. **Into the game.** With `--game`, each material is exported to
   `../game3d/assets/materials/<name>/`.

Writes `<run>/looks/<block>/`: `crop.png` and `<slot>.crop.png`, `describe/` (the agent's
question and answer), `looks.json` (each slot's material, version, tint, tile size, and whether
it was matched or made), `looks.con` (a `build.look <structure> <slot> <material> <tile> <r> <g>
<b>` line per slot; `--structure` defaults to `<site>-<block>`), and `sheet.jpg`: each slot's
crop beside its material, tinted.

The slots are the kit's (`assetgen/props/scripts/kit.py`): `footing`, `wall`, `upper` (an upper
storey's walls and the gables), `frame`, `floor` and `roof`. By default it describes `wall`,
`roof`, `footing` and `frame`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common.agent import ask
from concepts.scripts.paint import ROOF, pictures_in
from textures.library import words
from textures.library.client import LibraryClient
from textures.library.match import tint
from textures.library.measure import LUMA, to_linear, to_srgb

# What each slot is, for the agent: which part of the building its crop shows.
SLOTS = {
    "wall": "the walls of the building's lower storey",
    "upper": "the walls of its upper storey and gable ends",
    "roof": "the roof covering",
    "footing": "the foot of the walls, where they meet the ground",
    "frame": "the door, and the timber round it: describe the door's own boards",
    "floor": "the floor",
}
DEFAULT_SLOTS = ["wall", "roof", "footing", "frame"]
# The colour the game draws a plain block (`blockout.colours off`, game3d's blockout.rs), sRGB.
BLOCK_WHITE = (0.86, 0.85, 0.82)
# Which part of the block's box each slot is cropped from, as shares of its height down from its
# top: the roof from above the box (where the paint step lets it be painted) into its upper
# part, the walls from its lower part, the footing from its foot.
BANDS = {"roof": (-ROOF, 0.4), "upper": (-0.2, 0.6), "footing": (0.75, 1.0)}
BAND = (0.25, 1.0)
# A pixel the painter changed: further than this from the game's picture in some channel (sRGB).
CHANGED = 0.08
# Fewer changed pixels than this in a band, and the slot's colour is not known (left untinted).
CHANGED_MIN = 64
# The frames are small in a picture of a whole building: they are cropped round the block's door
# when it is in sight, a square this share of the box's height on a side. The door's colour is
# taken from the door's upper half: above its point on the picture (`site.seen`: the door's
# middle, a metre up), where grass, a cart or another roof in front of it rarely reach, in a
# strip this share of the crop's side across.
DOOR_CROP = 0.45
DOOR_MIDDLE = 0.15
# How much wider than the block's box the crop is, each side, as a share of the box.
MARGIN = 0.08
# The crop's shorter side is scaled up to at least this many pixels before the agent sees it.
CROP_MIN = 512


def crop_box(box: list[float], size: tuple[int, int], band: tuple[float, float] = (-ROOF, 1.0)) -> tuple[int, int, int, int]:
    """Part of the block's box on the picture in pixels: from `band[0]` to `band[1]` of its height
    down from its top (by default with room above for its roof), and a margin each side."""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    left, right = left - MARGIN * w, right + MARGIN * w
    top, bottom = top + band[0] * h, top + band[1] * h
    W, H = size
    return (max(0, int(left * W)), max(0, int(top * H)), min(W, int(right * W) + 1), min(H, int(bottom * H) + 1))


def scaled(im: Image.Image) -> Image.Image:
    s = max(1.0, CROP_MIN / min(im.size))
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def light_on(blockout: Path, box: list[float]) -> np.ndarray:
    """The light on the block where the painting was made, in linear RGB: the block's own pixels
    in the blockout picture (drawn plain white, the brighter half of what `site.mask` marks inside
    its box) over the white it is drawn in. A painting's colours are its materials times this."""
    pic = to_linear(np.asarray(Image.open(blockout).convert("RGB"), np.float32) / 255.0)
    H, W = pic.shape[:2]
    mask = np.asarray(Image.open(blockout.with_suffix(".mask.png")).convert("L").resize((W, H), Image.NEAREST)) > 127
    x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H))
    inside = pic[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    if len(inside) < 16:
        return to_linear(np.array(BLOCK_WHITE))
    luma = inside @ LUMA
    return inside[luma >= np.median(luma)].mean(0)


def balanced(picture: Image.Image, light: np.ndarray) -> Image.Image:
    """The painting with the colour of the game's light taken out and its brightness kept. The
    painter keeps the light's colour (the blue of a wall in shade), which the game would add a
    second time; the brightness is the painter's own, and painted stone and slate come out about
    as bright as the real thing, which is what the game's light expects."""
    colour = np.maximum(light / to_linear(np.array(BLOCK_WHITE)), 1e-4)
    colour = colour / (colour @ LUMA)
    lin = to_linear(np.asarray(picture.convert("RGB"), np.float32) / 255.0) / colour
    return Image.fromarray((to_srgb(lin) * 255 + 0.5).astype(np.uint8))


def painted_colour(painting: Image.Image, game: Image.Image, part: tuple[int, int, int, int]) -> np.ndarray | None:
    """A slot's colour on the painting, linear RGB: the median of the pixels in `part` the painter
    changed from the game's picture. What the paint step left alone (sky, ground, a rock in
    front) is the game's own and is skipped, as are windows by the median. None when the painter
    changed too little there."""
    p = np.asarray(painting.crop(part), np.float32) / 255.0
    g = np.asarray(game.resize(painting.size).crop(part).convert("RGB"), np.float32) / 255.0
    changed = np.abs(p - g).max(-1) > CHANGED
    if changed.sum() < CHANGED_MIN:
        return None
    return np.median(to_linear(p[changed]), 0)


def door_box(box: list[float], door: list[float] | None, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A square round the block's door on the picture, in pixels; None when it is out of sight."""
    if not door:
        return None
    W, H = size
    half = DOOR_CROP * (box[3] - box[1]) * H / 2
    x, y = door[0] * W, door[1] * H
    return (max(0, int(x - half)), max(0, int(y - 1.4 * half)), min(W, int(x + half)), min(H, int(y + 0.6 * half)))


def door_colour(painting: Image.Image, door: list[float], part: tuple[int, int, int, int]) -> np.ndarray:
    """The door's colour, linear RGB: the median of a strip of its upper half, above its point."""
    r = DOOR_MIDDLE * (part[2] - part[0]) / 2
    x, y = door[0] * painting.width, door[1] * painting.height
    p = np.asarray(painting.crop((int(x - r), int(y - 3 * r), int(x + r) + 1, int(y - r) + 1)), np.float32) / 255.0
    return np.median(to_linear(p.reshape(-1, 3)), 0)


def best_view(run: dict, pictures: Path, block: str, view: str | None) -> tuple[str, list[float], list[float] | None]:
    """The view the block is biggest in (or `view`), the block's box on its picture, and where
    its door is on it (None when out of sight)."""
    found = []
    for name, png in pictures_in(pictures).items():
        if name not in run["views"] or (view and name != view):
            continue
        seen = json.loads(png.with_suffix(".json").read_text())
        for b in seen["blocks"]:
            if b["name"] == block and b.get("box"):
                x0, y0, x1, y1 = b["box"]
                door = (b.get("door") or {}).get("at")
                found.append(((x1 - x0) * (y1 - y0), name, b["box"], door))
    if not found:
        sys.exit(f"block `{block}` is in no painted view" + (f" named `{view}`" if view else ""))
    _, name, box, door = max(found, key=lambda f: f[0])
    return name, box, door


def block_of(plan: dict, block: str) -> dict:
    for b in plan.get("block", []):
        if b["name"] == block:
            return b
    sys.exit(f"no block `{block}` in the plan")


def describe(out: Path, slots: list[str], block: dict) -> tuple[dict[str, words.Description], dict[str, str]]:
    """One question to an agent: each slot's material, in the library's words, and which slots
    are the same material as another (`same_as`)."""
    parts = "\n".join(f"- `{s}.crop.png`: {SLOTS[s]}" for s in slots)
    question = (
        f"These pictures are crops of one building in a painting of a fort. `crop.png` shows the whole "
        f"building; it was painted from these words: \"{' '.join((block.get('finish') or '').split())}\". "
        f"Trust the pictures over the words where they differ.\n\nEach other picture shows one part of it:\n"
        f"{parts}\n\nRead every picture. For each part, say what its bare surface is made of, as a material "
        f"for a game: the kind and pattern from the list that fits it best, its finish (its surface and "
        f"wear in a few words, no colour), its colour in a few words, and anything else about it in one "
        f"short sentence (or nothing). Describe only the surface, not windows, doors, ironwork, edges or "
        f"anything standing in front of it. If a part is plainly the same material as another part "
        f"(the foot of the walls built of the same stone as the walls), name that part in `same_as`.")

    def slot_schema(slot: str) -> dict:
        one = words.schema()
        one["properties"]["same_as"] = {"type": "string", "enum": [""] + [o for o in slots if o != slot],
                                        "description": "the part this one is the same material as, or empty"}
        one["required"].append("same_as")
        return one

    schema = {"type": "object", "properties": {s: slot_schema(s) for s in slots},
              "required": slots, "additionalProperties": False}
    files = [out / "crop.png"] + [out / f"{s}.crop.png" for s in slots]
    answer = ask(question, schema, out / "describe", files=files)
    described = {s: words.from_choice(answer[s]["material"], finish=answer[s]["finish"], colour=answer[s]["colour"],
                                      detail=answer[s]["detail"]) for s in slots}
    # A slot the same as another is dressed in that slot's material, unless that one is itself
    # the same as something (no chains).
    same = {s: answer[s]["same_as"] for s in slots
            if answer[s]["same_as"] in slots and not answer[answer[s]["same_as"]]["same_as"]}
    return described, same


def tinted(albedo: Image.Image, tint: list[float], size: int) -> Image.Image:
    lin = to_linear(np.asarray(albedo.convert("RGB").resize((size, size), Image.LANCZOS), np.float32) / 255.0)
    return Image.fromarray((to_srgb(lin * np.asarray(tint)) * 255 + 0.5).astype(np.uint8))


def sheet(lib: LibraryClient, out: Path, looks: dict) -> Path:
    """Each slot's crop beside its material's albedo, tinted, and its lit preview."""
    cell, pad = 300, 12
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for slot, look in looks.items():
            albedo = Image.open(lib.file(look["material"], look["version"], "albedo.png", Path(tmp) / "a.png"))
            lit = Image.open(lib.file(look["material"], look["version"], "sheet.jpg", Path(tmp) / "s.jpg")).convert("RGB")
            lit = lit.crop((0, 70 + 36, lit.width, lit.height))
            lit = lit.resize((int(lit.width * cell / lit.height), cell))
            crop = Image.open(out / f"{slot}.crop.png").convert("RGB")
            crop = crop.resize((int(crop.width * cell / crop.height), cell))
            row = Image.new("RGB", (crop.width + cell + lit.width + 4 * pad, cell + 40), (28, 28, 30))
            how = ("made" if look["made"] else f"the same as the {look['same_as']}" if look.get("same_as")
                   else f"matched, {look['similarity']}")
            ImageDraw.Draw(row).text((pad, 8), f"{slot}: {look['material']} v{look['version']} ({how}): "
                                     f"{look['description']['finish']}; tint {look['tint']}, "
                                     f"{look['tile_metres']} m tile", fill=(240, 240, 240))
            row.paste(crop, (pad, 32))
            row.paste(tinted(albedo, look["tint"], cell), (crop.width + 2 * pad, 32))
            row.paste(lit, (crop.width + cell + 3 * pad, 32))
            rows.append(row)
    im = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (28, 28, 30))
    y = 0
    for r in rows:
        im.paste(r, (0, y))
        y += r.height
    im.save(out / "sheet.jpg", quality=90)
    return out / "sheet.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="a paint run's folder (holds paint.json)")
    ap.add_argument("block", help="the block's name in the plan")
    ap.add_argument("--view", help="the painted view to crop (default: where the block is biggest)")
    ap.add_argument("--slots", nargs="+", choices=sorted(SLOTS), default=DEFAULT_SLOTS)
    ap.add_argument("--painting", type=int, default=0, help="which seed's painting to crop (default 0)")
    ap.add_argument("--structure", help="the structure `looks.con` dresses (default <site>-<block>)")
    ap.add_argument("--game", action="store_true", help="export each material into ../game3d/assets/materials/")
    a = ap.parse_args()

    run_dir = Path(a.run).resolve()
    run = json.loads((run_dir / "paint.json").read_text())
    plan = tomllib.loads(Path(run["plan"]).read_text())
    block = block_of(plan, a.block)
    view, box, door = best_view(run, Path(run["pictures"]), a.block, a.view)
    painted = [im for im in run["views"][view]["images"] if im["seed"] == a.painting]
    if not painted:
        sys.exit(f"view {view} has no painting of seed {a.painting}")
    out = run_dir / "looks" / a.block
    out.mkdir(parents=True, exist_ok=True)
    picture = Image.open(run_dir / painted[0]["file"]).convert("RGB")
    blockout = pictures_in(Path(run["pictures"]))[view]
    light = light_on(blockout, box)
    game = balanced(Image.open(blockout).convert("RGB"), light)
    picture = balanced(picture, light)
    scaled(picture.crop(crop_box(box, picture.size))).save(out / "crop.png")
    print(f"[looks] {a.block}: the {view} painting, the colour of light {np.round(light, 3).tolist()} taken out -> {out / 'crop.png'}", flush=True)

    colours = {}
    for slot in a.slots:
        around_door = door_box(box, door, picture.size) if slot == "frame" else None
        part = around_door or crop_box(box, picture.size, BANDS.get(slot, BAND))
        scaled(picture.crop(part)).save(out / f"{slot}.crop.png")
        colours[slot] = door_colour(picture, door, part) if around_door else painted_colour(picture, game, part)

    described, same = describe(out, a.slots, block)
    lib = LibraryClient().ensure()
    looks = {}
    for slot in sorted(a.slots, key=lambda s: s in same):
        desc = described[slot]
        colour = None if colours[slot] is None else [round(float(c), 5) for c in colours[slot]]
        user = f"{run['site']}/{a.block}/{slot}"
        print(f"  {slot}: {desc.kind}, {desc.pattern}: {desc.finish} ({desc.colour})", flush=True)
        if slot in same:
            # The same material as another slot, at this slot's own colour.
            of = looks[same[slot]]
            mean = lib.material(of["material"])["current"]["albedo_mean"]
            hit = {"name": of["material"], "version": of["version"], "tint": tint(mean, colour),
                   "tile_metres": of["tile_metres"], "made": False, "similarity": None}
            lib.use(user, hit["name"], hit["tint"], hit["tile_metres"])
            how = f"the same as the {same[slot]}"
        else:
            source = {"by": "concepts/scripts/looks.py", "run": str(run_dir), "block": a.block, "view": view,
                      "painting": painted[0]["file"], "slot": slot}
            hit = lib.resolve(desc.as_dict(), colour, user=user, source=source)
            how = "drew it" if hit["made"] else f"matched at {hit['similarity']}"
        looks[slot] = {"material": hit["name"], "version": hit["version"], "tint": hit["tint"],
                       "tile_metres": hit["tile_metres"], "made": hit["made"], "similarity": hit["similarity"],
                       "same_as": same.get(slot), "colour": colour, "description": desc.as_dict()}
        print(f"    -> {hit['name']} v{hit['version']} ({how}), tint {hit['tint']}, a {hit['tile_metres']} m tile", flush=True)
        if a.game and slot not in same:
            print(f"    -> {lib.export(hit['name'])}", flush=True)
    looks = {s: looks[s] for s in a.slots}
    (out / "looks.json").write_text(json.dumps(looks, indent=2) + "\n")
    structure = a.structure or f"{run['site']}-{a.block}"
    lines = [f"build.look {structure} {slot} {lk['material']} {lk['tile_metres']} " + " ".join(f"{t:g}" for t in lk["tint"])
             for slot, lk in looks.items()]
    (out / "looks.con").write_text("\n".join(lines) + "\n")
    print(f"[looks] {out / 'looks.json'}, {out / 'looks.con'}, sheet {sheet(lib, out, looks)}")


if __name__ == "__main__":
    main()
