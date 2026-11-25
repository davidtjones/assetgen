"""A client for the material library's API (`service.py`), standard library only.

    from textures.library.client import LibraryClient
    lib = LibraryClient().ensure()          # starts the service if it is not running
    hit = lib.resolve({"kind": "slate", "pattern": "rows", "finish": "thin, worn, mossy"},
                      colour=[0.05, 0.06, 0.07], user="colfort/hall/roof")
    lib.export(hit["name"])                 # into ../game3d/assets/materials/<name>/

`MATERIAL_LIBRARY_URL` points it at a service elsewhere (default http://127.0.0.1:8190).
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = os.environ.get("MATERIAL_LIBRARY_URL", "http://127.0.0.1:8190")
REPO = Path(__file__).resolve().parents[2]


class LibraryError(RuntimeError):
    pass


class LibraryClient:
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
            except Exception:
                message = str(e)
            raise LibraryError(f"{method} {path}: {message}") from None

    def is_up(self) -> bool:
        try:
            return self._call("GET", "/health", timeout=2)[0] == 200
        except (OSError, LibraryError):
            return False

    def ensure(self, log=print) -> LibraryClient:
        """Start the service for the rest of this run if none is up (local only)."""
        if self.is_up():
            return self
        logfile = REPO / "textures" / "library" / "service.log"
        log(f"[library] starting the service ({logfile})")
        proc = subprocess.Popen([sys.executable, "-m", "textures.scripts.library", "serve"], cwd=REPO,
                                stdout=open(logfile, "a"), stderr=subprocess.STDOUT)
        atexit.register(proc.terminate)
        for _ in range(60):
            if self.is_up():
                return self
            if proc.poll() is not None:
                break
            time.sleep(1)
        raise LibraryError(f"the library service did not start; see {logfile}")

    # -- reading ---------------------------------------------------------------------------------

    def vocabulary(self) -> dict:
        return self._call("GET", "/vocabulary")[1]

    def materials(self, kind: str | None = None, pattern: str | None = None, review: str | None = None) -> list[dict]:
        query = "&".join(f"{k}={urllib.request.quote(v)}" for k, v in
                         (("kind", kind), ("pattern", pattern), ("review", review)) if v)
        return self._call("GET", "/materials" + (f"?{query}" if query else ""))[1]

    def material(self, name: str) -> dict:
        return self._call("GET", f"/materials/{name}")[1]

    def file(self, name: str, version: int, file: str, dest: Path) -> Path:
        with urllib.request.urlopen(f"{self.url}/materials/{name}/v{version}/{file}", timeout=60) as r:
            Path(dest).write_bytes(r.read())
        return Path(dest)

    def search(self, description: dict, colour: list[float] | None = None) -> list[dict]:
        return self._call("POST", "/search", {"description": description, "colour": colour})[1]["candidates"]

    def uses(self, name: str | None = None) -> list[dict]:
        return self._call("GET", "/uses" + (f"?name={name}" if name else ""))[1]

    def job(self, job_id: int) -> dict:
        return self._call("GET", f"/jobs/{job_id}")[1]

    # -- asking for materials -------------------------------------------------------------------

    def wait(self, job_id: int, poll: float = 3.0, log=print) -> dict:
        """Wait for a job; its result. Raises LibraryError if it failed."""
        said = None
        while True:
            j = self.job(job_id)
            if j["state"] == "done":
                return j["result"]
            if j["state"] == "failed":
                raise LibraryError(f"job {job_id} failed: {j['error']}")
            if j["state"] != said:
                log(f"[library] job {job_id} {j['state']}")
                said = j["state"]
            time.sleep(poll)

    def resolve(self, description: dict, colour: list[float] | None = None, user: str | None = None,
                source: dict | None = None, wait: bool = True, log=print) -> dict:
        """A material for `description`: the library's best match, or a new one drawn for it.
        The result names it, with the tint and tile size to use it at, and whether it was made.
        With `wait=False`, a new material's result is `{"job": id}`."""
        body = {"description": description, "colour": colour, "user": user, "source": source}
        status, out = self._call("POST", "/resolve", body)
        if status == 200:
            return out["match"]
        log(f"[library] nothing matches {description['kind']} {description['pattern']} ({description['finish']}); "
            f"drawing one, job {out['job']}")
        return self.wait(out["job"], log=log) if wait else out

    def create(self, description: dict, seed: int = 0, source: dict | None = None, wait: bool = True, log=print) -> dict:
        out = self._call("POST", "/materials", {"description": description, "seed": seed, "source": source})[1]
        return self.wait(out["job"], log=log) if wait else out

    def revise(self, name: str, wait: bool = True, log=print, **changes) -> dict:
        out = self._call("POST", f"/materials/{name}/revise", changes)[1]
        return self.wait(out["job"], log=log) if wait else out

    def review(self, name: str, version: int, review: str, note: str = "") -> None:
        self._call("POST", f"/materials/{name}/v{version}/review", {"review": review, "note": note})

    def use(self, user: str, name: str, tint: list[float] | None = None, tile_metres: float | None = None) -> None:
        body = {"user": user, "name": name, **({"tint": tint} if tint else {}),
                **({"tile_metres": tile_metres} if tile_metres else {})}
        self._call("PUT", "/uses", body)

    def export(self, name: str) -> Path:
        """Copy a material's current version into the game's `assets/materials/<name>/`."""
        return Path(self._call("POST", f"/materials/{name}/export", {})[1]["path"])

    def delete(self, name: str, version: int | None = None, force: bool = False) -> dict:
        """Delete a material, or one version of it, into the trash (`Store.delete`). A used
        material needs `force`, which drops its uses."""
        path = f"/materials/{name}" + (f"/v{version}" if version is not None else "") + ("?force=1" if force else "")
        return self._call("DELETE", path)[1]

    def jobs(self, limit: int = 50) -> list[dict]:
        return self._call("GET", f"/jobs?limit={limit}")[1]
