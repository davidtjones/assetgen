# Textures

Tileable PBR materials, the material library the game's buildings are dressed from, and the
terrain layers.

```text
pixi run library serve                                    # the material library's API on :8190
pixi run library list | show <name> | sheet <out.jpg>     # look the library over
pixi run material <name>... [--seeds 0 1] [--maps-only]   # generate candidates -> out/mat/
pixi run ground [-- --size 512 --only rock]               # bake terrain layers -> ../game3d/assets/textures/terrain/
pixi run material-sm <name> [--seeds ...]                 # the older StableMaterials generator -> out/sm/
```

- `prompts/materials.yaml` - hand-written materials (bark, terrain layers): one `subject` per
  name.
- `library/` - the material library (below).
- `scripts/gen_material.py` - FLUX.2 klein draws a tileable photo, CHORD
  estimates the maps, both through ComfyUI (started if it is not running). Writes
  `out/mat/<name>_s<seed>/` and a lit preview sheet `out/mat/<name>.jpg`. Its
  docstring is the CLI reference.
- `scripts/bake_ground.py` - grades the generated materials named in `GENERATED`
  to set colours and packs them into the arrays the terrain shader reads. A layer
  whose material is missing falls back to a procedural one.
- `scripts/gen_material_sm.py` - kept because some bark and ground in use came
  from it; new materials come from `gen_material.py`.
- `scripts/tex_core.py`, `scripts/preview.py` - shared tileable-noise helpers and
  the lit preview sheet (also used by `trees/`).
- `workflows/` - the ComfyUI graphs; `workflows/README.md` documents them.
- `comfy_nodes/` - our ComfyUI nodes (`TorusRoll` and friends), symlinked into
  `tools/ComfyUI/custom_nodes/`.

## The material library

One shared set of materials for the whole game, described in words, found by meaning and
reused, so a world built from many paintings is dressed in a few dozen materials instead of
one per building. Tools ask it for "a material like this, for this"; it gives back one it has,
or draws a new one.

```sh
pixi run library serve            # started for you by any client that finds it down
pixi run library resolve stone "coursed blocks" "rough-hewn, weathered" --rgb 0.1 0.1 0.09
pixi run library sheet library.jpg
```

1. **A material is a description.** A kind and a pattern from `library/vocabulary.yaml`
   (`stone: coursed blocks`, `slate: rows`, `timber: vertical boards`...), a finish in a few
   words ("rough-hewn, weathered, lichen in the joints"), a colour in words and a detail
   (`library/words.py`). Each pattern says what klein is asked to draw, which way its courses
   run, and how big a course is.
2. **Matching.** Two descriptions match when their kind and pattern are the same and their
   finishes mean about the same: the cosine similarity of their embeddings (bge-small, on the
   CPU) is at least `MATCH` in `library/match.py` (0.78: a paraphrase scores about 0.83, an
   unrelated finish about 0.6). `MATCH` is how cohesive the world looks: looser, and one stone
   serves every wall; tighter, and the library fills with near twins.
3. **Colour never stops a match.** Every material's albedo is stored colour-neutral (grey on
   average, its light, dark and patches of moss kept), and each use is tinted to its own
   colour. Klein cannot be trusted with colour words: asked for dark blue-grey slate, it drew
   bright teal.
4. **Drawing one.** klein and CHORD (`scripts/gen_material.py`), then two checks, up to three
   seeds (`library/make.py`): patterns laid in rows must be even (no facade or seam across the
   tile), and an agent is shown the photo and asked whether it is that one bare surface and
   nothing else. The tile is sized from the courses counted on the height map. About a minute.
5. **Versions and review.** Revising a material's words draws a new version; every version
   has a review state (`new`, `approved`, `rejected`), and a material's current version is its
   newest not rejected. Who uses what (`colfort/hall/wall`, with its tint and tile size) is
   recorded, so a revision can be carried to everything that uses it.

The library lives in `library/data/` (not in git; `$MATERIAL_LIBRARY` moves it): one SQLite
file and a folder of maps per version (`library/store.py`). Only the service writes there.
Everything else goes through its HTTP API (`library/service.py`, whose docstring lists the
routes) or its Python client (`library/client.py`, `LibraryClient().ensure().resolve(...)`).
Export copies a material into `../game3d/assets/materials/<name>/`, where the game's
`build.look` finds it by name. `scripts/library.py`'s docstring is the CLI reference.

`concepts/scripts/looks.py` is its first user: it describes each slot of a painted building
and resolves the descriptions.

## Notes

`out/` is not in git: materials are reproducible from their prompt and seed.
The library's service and embeddings use aiohttp and transformers, which come with ComfyUI's
hand-installed requirements, not `pixi.toml`: a rebuilt env needs them reinstalled (`AGENTS.md`).
The CHORD weights are under a research-only licence.
