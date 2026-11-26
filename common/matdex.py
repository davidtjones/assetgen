"""matdex, the material index (`../matdex`), as assetgen reaches it. assetgen imports none of
matdex's code: it speaks HTTP to matdex's service, and reads the `current.json` the service
publishes, the same file the game reads. Standard library only, like `comfy_client.py`.

- `Matdex` is the service, at `MATDEX_URL` (default http://127.0.0.1:8190). It is not started
  from here: `Matdex().check()` stops a tool with a hint when it is down. The routes and their
  shapes are in matdex's `service.py`, and at `/openapi.json` on the running service.
- `Published` is the library as the service last published it (`current.json`): every current
  material's folder and every use, read with no service running. The folder is `MATDEX_DATA`,
  or `data/` in the matdex repo beside this one (`MATDEX_DIR` moves that).

    from common.matdex import Matdex
    lib = Matdex().check()
    hit = lib.resolve({"kind": "slate", "pattern": "rows", "finish": "thin, worn"},
                      colour=[0.05, 0.06, 0.07], user="colfort/hall/roof", region="moor")
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from common.paths import REPO

URL = os.environ.get("MATDEX_URL", "http://127.0.0.1:8190")


class MatdexError(RuntimeError):
    pass


def _q(part: str) -> str:
    return urllib.parse.quote(part, safe="")


class Matdex:
    def __init__(self, url: str = URL):
        self.url = url.rstrip("/")

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> tuple[int, dict | list]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", str(e))
            except ValueError:
                message = str(e)
            raise MatdexError(f"matdex {method} {path}: {message}") from None
        except urllib.error.URLError as e:
            raise MatdexError(f"matdex is not running at {self.url} ({e.reason}); it runs under the "
                              f"workspace's services/ (`./svc status` there)") from None

    def check(self) -> Matdex:
        """This client, once the service has answered. Raises MatdexError if it is down."""
        self._call("GET", "/health", timeout=5)
        return self

    def description_schema(self) -> dict:
        """The JSON schema an agent answers a description in: `material` ("kind: pattern"),
        `finish`, `colour`, `detail`."""
        return self._call("GET", "/vocabulary/schema")[1]

    def file(self, name: str, version: int, file: str, dest: Path) -> Path:
        """Download one of a version's files (`albedo.png`, `sheet.jpg`, ...) to `dest`."""
        try:
            with urllib.request.urlopen(f"{self.url}/materials/{_q(name)}/v{version}/{_q(file)}", timeout=60) as r:
                Path(dest).write_bytes(r.read())
        except urllib.error.URLError as e:
            raise MatdexError(f"matdex {name} v{version} {file}: {e}") from None
        return Path(dest)

    def use(self, user: str, name: str, colour: list[float] | None = None, tile_metres: float | None = None,
            region: str | None = None) -> dict:
        """Record that `user` uses `name`, tinted so it comes out `colour` on average (white
        when None). Returns the tint and tile size recorded."""
        body = {"user": user, "name": name, "colour": colour, "tile_metres": tile_metres, "region": region}
        return self._call("PUT", "/uses", {k: v for k, v in body.items() if v is not None})[1]

    def wait(self, job_id: int, poll: float = 3.0, log=print) -> dict:
        """Wait for a drawing job; its result. Raises MatdexError if it failed."""
        said = None
        while True:
            j = self._call("GET", f"/jobs/{job_id}")[1]
            if j["state"] == "done":
                return j["result"]
            if j["state"] == "failed":
                raise MatdexError(f"matdex job {job_id} failed: {j['error']}")
            if j["state"] != said:
                log(f"[matdex] job {job_id} {j['state']}")
                said = j["state"]
            time.sleep(poll)

    def resolve(self, description: dict, colour: list[float] | None = None, user: str | None = None,
                source: dict | None = None, region: str | None = None, log=print) -> dict:
        """A material for `description` in `region`: the library's best match, or a new one it
        draws (about a minute). The result names it, with the tint and tile size to use it at,
        and whether it was made."""
        body = {"description": description, "colour": colour, "region": region, "user": user, "source": source}
        status, out = self._call("POST", "/resolve", body)
        if status == 200:
            return out["match"]
        log(f"[matdex] nothing matches {description['kind']} {description['pattern']} ({description['finish']}); "
            f"drawing one, job {out['job']}")
        return self.wait(out["job"], log=log)


def data_dir() -> Path:
    """matdex's data folder: `MATDEX_DATA`, or `data/` in the matdex repo (`MATDEX_DIR`, or
    beside this repo)."""
    if os.environ.get("MATDEX_DATA"):
        return Path(os.environ["MATDEX_DATA"]).expanduser()
    env = os.environ.get("MATDEX_DIR")
    return (Path(env).expanduser() if env else REPO.parent / "matdex").resolve() / "data"


class Published:
    """The library as matdex's service last published it (`current.json`): every material's
    current version and every use. Read-only, and needs no service."""

    def __init__(self, data: Path | str | None = None):
        root = Path(data) if data else data_dir()
        file = root / "current.json"
        if not file.is_file():
            raise MatdexError(f"{file} not found: start matdex's service once (`uv run matdex serve`) to publish it")
        current = json.loads(file.read_text())
        self.root = root
        self.materials: dict[str, dict] = current["materials"]
        self.uses: dict[str, dict] = current["uses"]

    def use(self, user: str) -> dict | None:
        """What `user` wears: its material, tint (linear RGB), tile size and region, or None."""
        return self.uses.get(user)

    def map(self, name: str, map: str) -> Path:
        """A map (`albedo`, `normal`, `roughness`, `height`) of a material's current version."""
        if name not in self.materials:
            raise MatdexError(f"no current material `{name}` in {self.root / 'current.json'}")
        return self.root / self.materials[name]["dir"] / f"{map}.png"
