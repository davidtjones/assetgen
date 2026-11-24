# Concepts

Photograph a site in the running game, then have an image model draw a structure standing
there: a fort on a mountain pass, with the land, rocks, camera and light kept as they are.
A builder (an agent, or David) makes concepts before it builds; they are what it builds
toward. Changing the brief or the model changes every concept, so they are where the look
of what gets built is tuned.

```sh
# in ../game3d: a world with places, and the game running
WORLD_DIR=assets/terrain/crossing cargo run --release
# here
pixi run concept col                      # photos, then the fort brief's concepts
pixi run concept col --views approach,west --seeds 3,4
pixi run concept col --photos concepts/out/col/photos   # reuse photos; no game needed
pixi run concept col --brief /abs/path/brief.yaml --photos <dir> --out <dir>   # a builder's own
```

`<site>` is a place of the loaded world (`place.list` in the game's console) or a structure
(`build.list`).

## What it does

1. **Photos.** The tool sends `site.shots <site>` over the game's console socket
   (`common/game_console.py`). The game takes the same seven views every time, lit at 10:00:
   approach, north, east, south, west, above, inside (`../game3d/src/world/site.rs`). They
   go to `concepts/out/<site>/photos/<n>-<view>.png`, and the game's reply to `site.txt`.
2. **Concepts.** For each view and seed in the brief, Qwen-Image-Edit-2511 draws the
   subject into that view's photo -> `concepts/out/<site>/<brief>/<view>_s<seed>.png`.
3. **Sheet.** `<brief>/sheet.png` puts each view's photo beside its concepts.
   `<brief>/concept.json` records the prompt, the model settings and each image's time. Both
   cover the latest run only; concept images from earlier runs stay beside them.

`--out <dir>` writes the concepts, sheet and record to `<dir>` instead of
`concepts/out/<site>/<brief>/` (and new photos to `<dir>/photos`). A builder uses it to keep
its concepts in its own build directory, with `--photos` pointing at the site pictures it has
already taken.

The game and ComfyUI share the GPU, and the game quits when it runs out of memory. Before the
photos, the tool asks a running ComfyUI to unload its models. While the concepts are drawn
ComfyUI loads the model beside the running game, holding about 10 GB, and the game keeps
running.

## Briefs

A brief is `briefs/<name>.yaml`, the words the model is given, passed as `--brief <name>`; or
any YAML file with the same keys, passed by its path. Edit it to change what is drawn; no code
needs to change.

| Key | What it is |
| --- | --- |
| `subject` | What to draw: what it is for, its size in metres, and how it came to be. Without a size the model draws a castle. |
| `style` | How it looks: material, mood, how it meets the ground. The house style. |
| `prompt` | The full instruction. Picture 1 is the site photo; `{subject}` and `{style}` are filled in. |
| `views` | Which of `site.shots`' views to draw on. |
| `seeds` | One concept per seed per view. |

Today there is one brief, `fort`.

**Write the subject as a history, not a list of parts.** The model draws a list as listed and
adds nothing: "a curtain wall with battlements, one gate, a square tower at each corner" gave
plain walled yards that matched the fort kit and nothing more. Purpose alone ("built on the
rock, hard to take") gave a bare ring of wall. A history ("an old stone tower first, then
walls, then a gatehouse where the road comes in, timber hoardings, sheds and repairs") gave a
round keep, a gatehouse, timber, sheds and moss. Those were samples of six images each on the col,
2026-11-24. A concept can show what the kit cannot build yet; that gap is worth knowing.

## The model

Qwen-Image-Edit-2511 (int8, Apache 2.0) with the Lightning 4-step LoRA, run in ComfyUI. The
graph follows ComfyUI's own template for the model and is built in `edit_workflow` in
`scripts/concept.py`. An image takes about 10 s on the 3090 with ComfyUI alone, and 16-25 s
with the game running beside it. `--full` runs 40 steps without the LoRA: about 135 s, a
little crisper.

It was picked on 2026-11-23 over FLUX.2 klein 4B and LongCat-Image-Edit Turbo, all three run
on the same col photos. Klein changed the land under the fort. LongCat drew toy-like keeps.
Qwen kept the photo's light and drew the walls, towers and gate asked for. LongCat's weights
are still in `tools/ComfyUI/models`.

## Limits

- **Each view is its own design.** The model keeps a photo's camera and land, but it cannot
  redraw one concept from another camera. Given a concept as a second picture, it copies
  that picture instead.
- **From above it fails.** The photo is mostly grass, so the model invents a new camera with
  a horizon. The low and side views work.
- **Size is only roughly followed.** The metres in `subject` help, but the drawn fort is
  not measured against anything.
