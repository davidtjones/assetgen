# Tree generator

Procedural trees for the game's forest (`../game3d/src/trees.rs`). Each species
is a table of numbers; the generator grows a skeleton from it, skins it, and
bakes everything the game needs straight into `../game3d/assets/trees/`.

```text
pixi run trees                # bake every species (about a minute; impostor views are cached in out/)
pixi run trees -- --preview   # also write a preview sheet to out/preview.jpg
```

Look at the result in the game's asset viewer: `cargo run --release --bin viewer`
in `../game3d`. It reloads as files are rewritten.

The scripts, in `scripts/`:

- `tree_core.py` - grows a skeleton per species (`Habit`) and skins it as two
  meshes, bark and foliage, at two levels of detail.
- `tree_wood.py` - turns trunk, limbs and roots into one surface (a distance
  field meshed as a whole), decimated with Blender through `decimate_blender.py`.
- `tree_textures.py` - bark (the matdex material for the use `trees/<kind>/bark` at its
  tint, with a mossy overlay from `trees/<kind>/bark-mossy`; drawn when matdex has none) and
  foliage cards. Revise a bark in matdex, then bake again.
- `tex_core.py` - tileable noise and map helpers (normal from height, cavity, ...) that
  `tree_textures.py` draws with.
- `generate_trees.py` - the bake: meshes, textures, impostor atlases and the
  `trees.txt` manifest. Its docstring documents the manifest format.

Needs `blender` on `PATH` for the decimation step. Bark is read from what matdex's service last published
(`../matdex/data/current.json`, or `MATDEX_DATA`; `common/matdex.py`); a bark kind with no use
there, or no matdex at all, falls back to drawn bark.
