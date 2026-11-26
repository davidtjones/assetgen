"""Where assetgen finds its sibling repos: the `game3d` app it bakes assets into, and
`matdex`, the material index whose client and words `concepts/scripts/looks.py` uses.

Every generator that writes into the game goes through `game3d_dir()`, so the
location is decided in one place. Set `GAME3D_DIR` to point somewhere else (a
scratch copy, say); by default it is the `game3d` directory beside this repo.
`MATDEX_DIR` does the same for matdex.
"""

import os
import sys
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


def use_matdex() -> Path:
    """Make the `matdex` package importable, from the matdex repo beside this one (or
    `MATDEX_DIR`). Only its plain modules are used here (`client`, `words`, `match`,
    `measure`: the standard library, numpy and Pillow); the service runs in matdex's own uv
    environment, which its client starts. matdex is not a pixi dependency because syncing the
    pixi environment drops ComfyUI's hand-installed packages."""
    env = os.environ.get("MATDEX_DIR")
    root = Path(env).expanduser().resolve() if env else (REPO.parent / "matdex").resolve()
    if not (root / "matdex" / "client.py").is_file():
        hint = "check MATDEX_DIR" if env else "it should sit beside assetgen, or set MATDEX_DIR"
        raise SystemExit(f"matdex not found at {root} ({hint})")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
