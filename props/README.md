# Props

One spec in `specs/<name>.yaml` becomes a static, textured glTF that the game
places in the world (`../game3d/src/world/food.rs` for food plants). A prop is
anything that does not move on its own: today a mushroom.

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
