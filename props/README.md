# Props

One spec in `specs/<name>.yaml` becomes a static, textured glTF that the game
places in the world (`../game3d/src/world/food.rs` for food plants). A prop is
anything that does not move on its own: today a mushroom. A spec can also be a
**kit**: a set of building pieces made by code that snap together on a grid
(see "Kits" below).

```text
pixi run prop all mushroom             # every stage
pixi run prop <stage> <name>...        # concept, mesh, frame or export, alone
```

1. `concept` and `mesh` run through ComfyUI (`common/image_to_mesh.py`, shared
   with `creatures/`): FLUX.2 klein draws the object alone on a plain
   background, TRELLIS.2 makes a textured game mesh from it.
2. `frame` runs in Blender (`scripts/blender/frame_prop.py`): scale to the
   spec's height, stand it on the ground, centre it, check it, and write
   `out/<name>/sheet.png` (the concept beside four renders) to look at.
3. `export` copies a passing prop into `../game3d/assets/props/` and updates
   `props.txt` there.

Each stage reads only what the one before wrote under `out/<name>/` (not in
git), so any stage can be rerun alone. `scripts/prop.py`'s docstring is the
reference for stages and spec keys. Needs `blender` on `PATH`.

## Spec

```yaml
subject: a single common wild mushroom, ...   # what to draw
height: 0.12          # metres, ground to top
tris: 600             # game mesh budget; detail goes in the baked maps
texture_size: 512     # baked map size in pixels (default 1024)
seed: 1
```

`avoid` adds to the negative prompt. If a concept comes out wrong, try another
seed with `--seed N` before changing the subject.

## Kits

A kit is one style of building pieces - walls, towers, a gate, stairs, floors,
roofs - that a builder (an agent or a player) snaps together on a structure's
grid. Pieces are made by code, not drawn and meshed, so they fit the grid
exactly. Today there is one: `fort`, twelve stone pieces.

```text
pixi run prop all fort                 # build, frame, export: a few seconds
```

1. `build` (`scripts/kit.py`) makes every piece and the spec's test arrangement
   as glTF under `out/<name>/`.
2. `frame` checks each piece and the arrangement, renders them with Blender
   (`scripts/blender/render_kit.py`) and writes `out/<name>/sheet.png`: the
   pieces in a row above the arrangement from outside, inside and above.
3. `export` writes the pieces and `kit.json` into `../game3d/assets/kits/<name>/`.
   The game's asset viewer shows each piece and builds the test arrangement with
   the game's own placement (`../game3d/src/world/kit.rs`).

**The grid.** Every piece fills whole cells of a grid of `cell`-metre cubes
(2 m). It is placed at a cell with a turn of 0-3 quarter turns about up. Its
outer side, where battlements stand and a gate opens to, is -z before turning.

**What is checked.** Each piece stays inside its cells; each side it says is
solid is closed over every cell; each passage (a gateway, a door) is clear; each
walk surface (a wall-walk, a tread) is at the height it says with 2 m of
headroom. In the test arrangement no two pieces share a cell, no passage opens
onto a solid face, and no two different partial profiles meet. `kit.json`
records each piece's cells and what each of its outward faces is, so a builder
can tell what fits next to what without looking.

**Slots.** Each part of a piece is one slot, the part of a building it is:
`footing`, `wall`, `upper` (an upper storey and the gables), `frame` (doors,
windows, posts), `floor` or `roof`. The kit gives each slot a flat colour; in the
game a structure names a material for each slot (`build.look`), made from its
painting by `pixi run looks` (`concepts/README.md`).

**Small buildings.** `kit.house(at, size, storeys, doors, windows, open_sides,
stairs)` lays out a gable-roofed building from the house pieces (walls, windows,
doors, corners, posts, upper-storey walls, gables, and 45 degree roof courses),
as `[piece, cell, turn]` entries. A kit spec's `houses` list adds them to the test
arrangement.

```yaml
kind: kit
cell: 2.0                       # metres per grid cell
colors:                         # sRGB per slot
  wall: [128, 124, 116]
  frame: [112, 80, 52]
  roof: [74, 80, 92]
inherit:                        # a slot with no material takes this slot's
  upper: wall
test:                           # [piece, [i, j, k], turn]
  - [tower_base, [0, 0, 0], 2]
  - [wall, [3, 0, 0], 0]
houses:                         # kit.house arguments, added to the test arrangement
  - {name: stable, at: [24, 0, 0], size: [6, 4], open_sides: ["-z"], doors: [["+x", 1, "wide"]]}
```

`scripts/kit.py`'s docstring is the reference for the grid, turns, the manifest
and the checks; the pieces themselves are in `build_pieces`.

**Limits.** Pieces are boxes and extrusions with flat colours per slot until a
structure dresses them. Roofs are 45 degrees only, a storey is 4 m, and there is
no lean-to layout yet. Joins are checked by face kind, not by drawing the seam,
so two pieces whose faces both say `solid` are trusted to meet. A kit is one
fixed set of pieces in code; there is no generated (concept and mesh) path for
pieces yet.
