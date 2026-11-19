# Creatures

One spec in `specs/<name>.yaml` becomes a rigged, animated, textured glTF that
the game loads (`../game3d/src/creatures.rs`).

```text
pixi run creature all boar             # every stage
pixi run creature <stage> <name>...    # concept, mesh, rig, check or export, alone
```

1. `concept` and `mesh` run through ComfyUI (`common/image_to_mesh.py`, shared
   with `props/`): FLUX.2 klein draws the creature, TRELLIS.2 makes a textured
   game mesh from it.
2. `rig` and `check` run in Blender (`scripts/blender/`): fit a quadruped
   skeleton, skin it, key `idle` and gait loops (`walk`, and `run` and
   `sprint` if the spec has them), then measure the result
   against the spec and write a contact sheet.
3. `export` copies a passing creature into `../game3d/assets/creatures/` and
   updates `creatures.txt` there.

Each stage reads only what the one before wrote under `out/<name>/` (not in
git), so any stage can be rerun alone. `scripts/creature.py`'s docstring is the
reference for stages and spec keys. Needs `blender` on `PATH`; about two
minutes per creature on the 3090.
