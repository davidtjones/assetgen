"""The material library: every material the game is dressed in, described in words, drawn
once, and reused wherever something like it is asked for.

- `words.py` - a material in words (`Description`) and the vocabulary of kinds and patterns.
- `match.py` - whether the library already has a material for a description; the settings
  that decide how cohesive the world looks.
- `make.py` - drawing a version (FLUX.2 klein, then CHORD) and measuring it.
- `store.py` - SQLite and the map files; versions, review state, who uses what.
- `service.py` - the HTTP API, the one way in (`pixi run library serve`).
- `client.py` - the client other tools use.

`textures/README.md` is the reference for using it.
"""
