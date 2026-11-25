"""Draw one version of a material from its description, and measure it.

klein photographs the description's words (`Description.words`) as a tileable surface, and
CHORD estimates its maps, through `textures/scripts/gen_material.py`. Each drawing is then
checked, and drawn again with another seed if it fails, `TRIES` times at most:

1. **Even** (patterns laid in rows: stone courses, rows of slates): a tile uneven at a large
   scale (`UNEVEN`) has had a facade or a seam drawn across it.
2. **Bare** (every pattern): an agent is shown klein's photo and asked one question, whether it
   is the one surface it should be with nothing else in it (`bare`). klein draws in what the
   words mention: a door's "stone surround" became a pale strip down the side of its boards.
   The question and answer are kept in the version's `check/`.

The first drawing that passes is kept, else the best (bare first, then evenest), with what was
wrong in its `source.check`. Then its courses are counted on the height map and its tile sized
from them.

Last, the albedo is made colour-neutral (`neutral`): each channel divided by its mean and
multiplied by the mean's brightness, so it is grey on average with its light and dark, and its
patches of another colour (moss, lichen), kept as they were drawn. Every use tints it to its
own colour, so what colour klein happened to draw (asked for no colour, slate came out pale
yellow-green) never matters. The albedo as drawn is kept as `albedo_drawn.png`.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common.agent import AgentError, ask
from textures.library import measure
from textures.library.words import NOT_A_SURFACE, Description
from textures.scripts import gen_material as gm

SIZE = 1024
UNEVEN = 0.25
TRIES = 3


def neutral(d: Path) -> list[float]:
    """Make `d/albedo.png` colour-neutral (keeping the drawn one as `albedo_drawn.png`); its new
    mean, linear RGB."""
    drawn = d / "albedo_drawn.png"
    (d / "albedo.png").rename(drawn)
    lin = measure.to_linear(np.asarray(Image.open(drawn).convert("RGB"), np.float32) / 255.0)
    mean = lin.reshape(-1, 3).mean(0)
    grey = float(mean @ measure.LUMA)
    lin = lin * (grey / np.maximum(mean, 1e-4))
    Image.fromarray((measure.to_srgb(lin) * 255 + 0.5).astype(np.uint8)).save(d / "albedo.png")
    return [round(float(c), 5) for c in measure.mean_colour(d / "albedo.png")]


def measured(d: Path, desc: Description) -> dict:
    info = desc.info
    n = measure.courses(d / "height.png", info["runs"] == "rows") if info["runs"] != "none" else None
    return {"courses": n,
            "tile_metres": round(n * info["course"], 3) if n else float(info["tile"]),
            "albedo_mean": [round(float(c), 5) for c in measure.mean_colour(d / "albedo.png")],
            "unevenness": round(measure.unevenness(d / "albedo.png"), 3)}


def bare(d: Path, desc: Description) -> dict:
    """Ask an agent whether klein's photo in `d` is one bare surface of `desc`: `{bare, other}`."""
    question = (f"Read source.png. It should be a close-up photograph of one bare surface: {desc.kind}, "
                f"{desc.pattern} ({desc.finish}), filling the whole picture edge to edge and repeating "
                f"evenly. Is there anything else in it: a frame, an edge or corner, a window or door, a band "
                f"or strip of something different across or along it, another material (stone among timber, "
                f"timber among stone), sky or grass? Say bare only if it is that one surface and nothing else.")
    schema = {"type": "object", "properties": {
        "bare": {"type": "boolean", "description": "true if it is only that one surface"},
        "other": {"type": "string", "description": "what else is in it, in a few words, or empty"}},
        "required": ["bare", "other"], "additionalProperties": False}
    try:
        return ask(question, schema, d / "check", files=[d / "source.png"])
    except AgentError as e:
        return {"bare": True, "other": f"not checked: {e}"}


def make(desc: Description, dest: Path, seed: int = 0, log=print) -> tuple[int, dict]:
    """Draw `desc` into the empty folder `dest`. Returns the seed kept and its measurements,
    with the check of the drawing kept (`check`)."""
    client = gm._server()
    entry = {"subject": desc.words(), "avoid": NOT_A_SURFACE}
    rows = desc.info["runs"] == "rows"
    tries = []
    for n in range(TRIES):
        s = seed + 1000 * n
        d = dest.parent / f"{dest.name}.s{s}"
        start = time.time()
        gm.generate(client, dest.name, entry, s, SIZE, "chord", dest=d)
        m = measured(d, desc)
        even = not rows or m["unevenness"] <= UNEVEN
        check = bare(d, desc) if even else {"bare": None, "other": "not asked: uneven"}
        m["check"] = {**check, "even": even}
        tries.append(((check["bare"] is not True, m["unevenness"]), s, d, m))
        log(f"  {desc.kind} {desc.pattern} seed {s}: {m['courses']} courses, a {m['tile_metres']} m tile, "
            f"unevenness {m['unevenness']}{'' if even else ' (uneven)'}, "
            f"{'bare' if check['bare'] else 'not bare: ' + check['other']} ({time.time() - start:.0f}s)")
        if even and check["bare"]:
            break
    _, s, d, m = min(tries, key=lambda t: t[0])
    if not (m["check"]["even"] and m["check"]["bare"]):
        log(f"  none of {len(tries)} drawings passed; keeping seed {s} ({m['check']['other']})")
    for _, _, other, _ in tries:
        if other != d:
            shutil.rmtree(other)
    dest.rmdir()
    d.rename(dest)
    m["albedo_mean"] = neutral(dest)
    return s, m
