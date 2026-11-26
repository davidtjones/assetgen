"""Where assetgen finds its sibling repo `game3d`, the app it bakes assets into.

Every generator that writes into the game goes through `game3d_dir()`, so the
location is decided in one place. Set `GAME3D_DIR` to point somewhere else (a
scratch copy, say); by default it is the `game3d` directory beside this repo.
matdex is reached over HTTP and through its published `current.json` (`matdex.py`).
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def game3d_dir() -> Path:
    """The game3d repo root. Fails loudly if it does not look like one, so a moved
    directory stops an export instead of writing assets somewhere nobody reads."""
    env = os.environ.get("GAME3D_DIR")
    root = Path(env).expanduser().resolve() if env else (REPO.parent / "game3d").resolve()
    if not (root / "Cargo.toml").is_file():
        hint = "check GAME3D_DIR" if env else "it should sit beside assetgen, or set GAME3D_DIR"
        raise SystemExit(f"game3d not found at {root}: no Cargo.toml there ({hint})")
    return root


def game3d_assets(*parts: str) -> Path:
    """A path under game3d's `assets/` tree."""
    return game3d_dir().joinpath("assets", *parts)

