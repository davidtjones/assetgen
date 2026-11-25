"""The material library from the command line: run its service, and ask it things.

    pixi run library serve [--port 8190]
    pixi run library list [--kind K] [--pattern P] [--review new|approved|rejected]
    pixi run library show <name>
    pixi run library search <kind> <pattern> <finish> [--colour-words W] [--detail D] [--rgb R G B]
    pixi run library resolve <kind> <pattern> <finish> [--colour-words W] [--detail D] [--rgb R G B] [--user U]
    pixi run library make <kind> <pattern> <finish> [--colour-words W] [--detail D] [--seed N]
    pixi run library revise <name> [--finish F] [--colour-words W] [--detail D] [--seed N]
    pixi run library review <name> <version> new|approved|rejected [--note N]
    pixi run library export <name>...
    pixi run library delete <name> [--version N] [--force]
    pixi run library uses [<name>]
    pixi run library sheet <out.jpg> [--kind K] [--pattern P]

Every command but `serve` goes through the service's API (`textures/library/service.py`),
starting it for the run if it is not up. `--rgb` is a colour in linear RGB to tint a match to.
`sheet` draws every current material as a tile with its name, description and review state,
for looking the library over. `delete` moves a material, or one version, to the desktop trash;
a material that is used needs `--force`, which drops its uses and trashes its copy in the game.
The service also serves a review page at http://127.0.0.1:8190/ for doing all this by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from textures.library.client import LibraryClient, LibraryError


def _desc(a) -> dict:
    return {"kind": a.kind, "pattern": a.pattern, "finish": a.finish, "colour": a.colour_words or "",
            "detail": a.detail or ""}


def _describe(a: argparse.ArgumentParser) -> None:
    a.add_argument("kind")
    a.add_argument("pattern")
    a.add_argument("finish")
    a.add_argument("--colour-words", help="its colour in words (steers what is drawn)")
    a.add_argument("--detail")


def sheet(lib: LibraryClient, out: Path, kind: str | None, pattern: str | None) -> Path:
    found = lib.materials(kind, pattern)
    if not found:
        sys.exit("the library has no materials" + (" of that kind" if kind or pattern else ""))
    cell, cols = 256, 5
    rows = (len(found) + cols - 1) // cols
    im = Image.new("RGB", (cols * cell, rows * (cell + 64)), (28, 28, 30))
    draw = ImageDraw.Draw(im)
    uses = {}
    for u in lib.uses():
        uses[u["name"]] = uses.get(u["name"], 0) + 1
    with tempfile.TemporaryDirectory() as tmp:
        for i, m in enumerate(found):
            x, y = (i % cols) * cell, (i // cols) * (cell + 64)
            tile = Image.open(lib.file(m["name"], m["version"], "albedo.png", Path(tmp) / "a.png")).convert("RGB")
            im.paste(tile.resize((cell - 8, cell - 8)), (x + 4, y + 4))
            draw.text((x + 6, y + cell), f"{m['name']} v{m['version']} ({m['review']}, {uses.get(m['name'], 0)} uses)", fill=(240, 240, 240))
            draw.text((x + 6, y + cell + 16), m["finish"][:42], fill=(190, 190, 190))
            draw.text((x + 6, y + cell + 32), f"{m['tile_metres']} m tile", fill=(150, 150, 150))
    im.save(out, quality=90)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8190)
    p = sub.add_parser("list")
    p.add_argument("--kind")
    p.add_argument("--pattern")
    p.add_argument("--review")
    sub.add_parser("show").add_argument("name")
    for name in ("search", "resolve", "make"):
        p = sub.add_parser(name)
        _describe(p)
        if name != "make":
            p.add_argument("--rgb", type=float, nargs=3)
        if name == "resolve":
            p.add_argument("--user")
        if name == "make":
            p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("revise")
    p.add_argument("name")
    p.add_argument("--finish")
    p.add_argument("--colour-words")
    p.add_argument("--detail")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("review")
    p.add_argument("name")
    p.add_argument("version", type=int)
    p.add_argument("state", choices=["new", "approved", "rejected"])
    p.add_argument("--note", default="")
    sub.add_parser("export").add_argument("names", nargs="+")
    p = sub.add_parser("delete")
    p.add_argument("name")
    p.add_argument("--version", type=int, help="delete only this version")
    p.add_argument("--force", action="store_true", help="delete a used material, dropping its uses")
    sub.add_parser("uses").add_argument("name", nargs="?")
    p = sub.add_parser("sheet")
    p.add_argument("out")
    p.add_argument("--kind")
    p.add_argument("--pattern")
    a = ap.parse_args()

    if a.command == "serve":
        from textures.library.service import serve
        serve(a.port)
        return
    lib = LibraryClient().ensure()
    show = lambda x: print(json.dumps(x, indent=2))
    try:
        if a.command == "list":
            for m in lib.materials(a.kind, a.pattern, a.review):
                print(f"{m['name']:<32} v{m['version']} {m['review']:<9} {m['tile_metres']:>6} m  {m['finish']}")
        elif a.command == "show":
            show(lib.material(a.name))
        elif a.command == "search":
            show(lib.search(_desc(a), a.rgb))
        elif a.command == "resolve":
            show(lib.resolve(_desc(a), a.rgb, user=a.user, source={"by": "pixi run library resolve"}))
        elif a.command == "make":
            show(lib.create(_desc(a), seed=a.seed, source={"by": "pixi run library make"}))
        elif a.command == "revise":
            changes = {k: v for k, v in (("finish", a.finish), ("colour", a.colour_words), ("detail", a.detail)) if v}
            show(lib.revise(a.name, seed=a.seed, **changes))
        elif a.command == "review":
            lib.review(a.name, a.version, a.state, a.note)
            print(f"{a.name} v{a.version}: {a.state}")
        elif a.command == "export":
            for name in a.names:
                print(f"{name} -> {lib.export(name)}")
        elif a.command == "delete":
            done = lib.delete(a.name, a.version, a.force)
            print(f"deleted {a.name}" + (f" v{a.version}" if a.version and not done["material_deleted"] else "")
                  + "".join(f"\n  trashed {t}" for t in done["trashed"])
                  + "".join(f"\n  dropped use {u}" for u in done["uses_dropped"]))
        elif a.command == "uses":
            for u in lib.uses(a.name):
                print(f"{u['user']:<32} {u['name']:<32} tint {u['tint']} tile {u['tile_metres']} m")
        elif a.command == "sheet":
            print(sheet(lib, Path(a.out), a.kind, a.pattern))
    except LibraryError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
