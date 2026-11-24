"""Drive a running game3d over its console socket. Standard library only.

The game listens on a Unix socket (`../game3d/src/dev/remote.rs`): at `CONSOLE_SOCKET`, else
`$XDG_RUNTIME_DIR/game3d-console.sock`, else `/tmp/game3d-console.sock`. A client sends one
console line; the game runs it as if it were typed, sends back each line it prints, then one
line starting with `\\x04`: `\\x04ok` or `\\x04failed`. A line that waits (`wait`, `shot`,
`settle`, or a command that queues more lines, such as `site.shots`) answers when it is done.

    with GameConsole() as game:
        print(game.run("place.info col"))
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

END = "\x04"


class GameError(RuntimeError):
    pass


def socket_path() -> Path:
    if path := os.environ.get("CONSOLE_SOCKET"):
        if path == "off":
            raise GameError("CONSOLE_SOCKET=off: the game's console socket is turned off")
        return Path(path)
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "game3d-console.sock"


class GameConsole:
    def __init__(self, path: Path | None = None, timeout: float = 600.0):
        self.path = path or socket_path()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(str(self.path))
        except OSError as e:
            raise GameError(
                f"no game listening at {self.path} ({e.strerror}); start it in ../game3d with "
                "`WORLD_DIR=assets/terrain/<world> cargo run --release`"
            ) from None
        self.reader = self.sock.makefile("r", encoding="utf-8", newline="\n")

    def __enter__(self) -> GameConsole:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.reader.close()
        self.sock.close()

    def send(self, line: str) -> tuple[bool, str]:
        """Run one console line. Returns whether it succeeded, and what it printed."""
        if "\n" in line:
            raise ValueError("one console line at a time")
        self.sock.sendall((line + "\n").encode())
        out: list[str] = []
        for raw in self.reader:
            raw = raw.rstrip("\n")
            if raw.startswith(END):
                return raw[1:] == "ok", "\n".join(out)
            out.append(raw)
        raise GameError(f"the game closed the console socket during `{line}`")

    def run(self, line: str) -> str:
        """Run one console line and return what it printed; raise if it failed."""
        ok, text = self.send(line)
        if not ok:
            raise GameError(f"`{line}` failed:\n{text}")
        return text
