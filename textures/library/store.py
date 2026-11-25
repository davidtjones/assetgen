"""The library's store: one SQLite file and a folder of maps per material version.

    <root>/library.sqlite
    <root>/materials/<name>/v<version>/   albedo.png normal.png roughness.png height.png
                                          albedo_drawn.png metalness.png source.png sheet.jpg

`<root>` is `textures/library/data/` (not in git), or `$MATERIAL_LIBRARY`.

Tables:

- `material`: one row per material, its name, kind and pattern. A name is the kind and pattern
  and a number (`slate-rows-2`), and never changes.
- `version`: each time a material was drawn: its description, the words klein was given and the
  seed, what was measured on the maps (courses, tile size, the neutral albedo's mean colour,
  which a use's tint is worked out from, and unevenness), an embedding
  of its free words, where the request came from, and a review state (`new`, `approved`,
  `rejected`). A material's current version is its newest one not rejected.
- `use`: who uses which material (`colfort/hall/wall`), with the tint and tile size it is used at.
- `job`: requests to draw a material, which take a minute each and run one at a time.
- `deleted`: what was deleted (a whole material, or one version) and when, with its rows. A
  deleted material's name and a deleted version's number are never given out again, so a
  name a building still holds cannot come to mean a different material.

Deleting is a real removal from the library, but the files go to the desktop trash (`gio
trash`), each folder with a `deleted.json` saying what it was, so a mistake can be undone by
hand. The library never deletes a file outright.

Only the service (`service.py`) writes here; everything else goes through its API.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

from textures.library.words import Description
from textures.scripts import preview

ROOT = Path(os.environ.get("MATERIAL_LIBRARY") or Path(__file__).parent / "data")
MAPS = ("albedo", "normal", "roughness", "height")

SCHEMA = """
CREATE TABLE IF NOT EXISTS material (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    pattern TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS version (
    name TEXT NOT NULL REFERENCES material(name),
    version INTEGER NOT NULL,
    finish TEXT NOT NULL,
    colour TEXT NOT NULL,
    detail TEXT NOT NULL,
    words TEXT NOT NULL,
    seed INTEGER NOT NULL,
    courses INTEGER,
    tile_metres REAL NOT NULL,
    albedo_mean TEXT NOT NULL,
    unevenness REAL NOT NULL,
    embedding BLOB NOT NULL,
    source TEXT NOT NULL,
    review TEXT NOT NULL DEFAULT 'new' CHECK (review IN ('new', 'approved', 'rejected')),
    note TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS use (
    user TEXT PRIMARY KEY,
    name TEXT NOT NULL REFERENCES material(name),
    tint TEXT NOT NULL,
    tile_metres REAL NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    request TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'done', 'failed')),
    result TEXT,
    error TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deleted (
    name TEXT NOT NULL,
    version INTEGER,
    record TEXT NOT NULL,
    deleted TEXT NOT NULL
);
"""

REVIEWS = ("new", "approved", "rejected")


class InUse(ValueError):
    """A material that is used was asked to be deleted without `force`."""

    def __init__(self, name: str, uses: list[dict]):
        super().__init__(f"`{name}` is used by {', '.join(u['user'] for u in uses)}; delete it with force "
                         f"to drop those uses")
        self.uses = uses


def trash(path: Path) -> None:
    """Move `path` to the desktop trash. Raises RuntimeError if it cannot: the library does not
    fall back to deleting outright."""
    if shutil.which("gio") is None:
        raise RuntimeError("cannot delete: `gio` is not installed, and the library only deletes into the trash")
    done = subprocess.run(["gio", "trash", str(path)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"could not move {path} to the trash: {done.stderr.strip()}")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class Store:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        (self.root / "materials").mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(exist_ok=True)
        self.path = self.root / "library.sqlite"
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    # -- materials and versions --------------------------------------------------------------

    def dir(self, name: str, version: int) -> Path:
        return self.root / "materials" / name / f"v{version}"

    def staging(self, label: str) -> Path:
        """A fresh folder to draw a version in before it is added."""
        d = self.root / "staging" / f"{slug(label)}-{time.time_ns()}"
        d.mkdir(parents=True)
        return d

    @staticmethod
    def _version(row: sqlite3.Row, embedding: bool = False) -> dict:
        out = {k: row[k] for k in row.keys() if k != "embedding"}
        out["albedo_mean"] = json.loads(out["albedo_mean"])
        out["source"] = json.loads(out["source"])
        if embedding:
            out["embedding"] = np.frombuffer(row["embedding"], np.float32)
        return out

    def add(self, desc: Description, staged: Path, measured: dict, embedding: np.ndarray, seed: int,
            source: dict, name: str | None = None) -> tuple[str, int]:
        """Add a version drawn in `staged` (moved into place): to material `name`, or to a new
        material named from its kind and pattern. Returns the name and version."""
        with self._db() as db:
            if name is None:
                stem = slug(f"{desc.kind}-{desc.pattern}")
                taken = {r["name"] for r in db.execute(
                    "SELECT name FROM material WHERE kind = ? AND pattern = ? "
                    "UNION SELECT name FROM deleted WHERE version IS NULL", (desc.kind, desc.pattern))}
                n = 1
                while f"{stem}-{n}" in taken:
                    n += 1
                name = f"{stem}-{n}"
                db.execute("INSERT INTO material VALUES (?, ?, ?, ?)", (name, desc.kind, desc.pattern, now()))
            else:
                row = db.execute("SELECT kind, pattern FROM material WHERE name = ?", (name,)).fetchone()
                if row is None or (row["kind"], row["pattern"]) != (desc.kind, desc.pattern):
                    raise ValueError(f"`{name}` is not a {desc.kind} of {desc.pattern}")
            version = 1 + (db.execute("SELECT MAX(v) FROM (SELECT version AS v FROM version WHERE name = ? "
                                      "UNION ALL SELECT version FROM deleted WHERE name = ?)",
                                      (name, name)).fetchone()[0] or 0)
            dest = self.dir(name, version)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(staged, dest)
            preview.sheet([dest], dest / "sheet.jpg", [f"{name} v{version}: {desc.finish}"])
            db.execute(
                "INSERT INTO version (name, version, finish, colour, detail, words, seed, courses, tile_metres, "
                "albedo_mean, unevenness, embedding, source, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, version, desc.finish, desc.colour, desc.detail, desc.words(), seed, measured["courses"],
                 measured["tile_metres"], json.dumps(measured["albedo_mean"]), measured["unevenness"],
                 np.asarray(embedding, np.float32).tobytes(), json.dumps(source), now()))
        return name, version

    _CURRENT = """
        SELECT m.kind, m.pattern, v.* FROM material m JOIN version v ON v.name = m.name
        WHERE v.version = (SELECT MAX(version) FROM version w WHERE w.name = m.name AND w.review != 'rejected')
    """

    # A material whose every version is rejected has no current version; this is its newest.
    _ALL_REJECTED = """
        SELECT m.kind, m.pattern, v.* FROM material m JOIN version v ON v.name = m.name
        WHERE v.version = (SELECT MAX(version) FROM version w WHERE w.name = m.name)
          AND NOT EXISTS (SELECT 1 FROM version w WHERE w.name = m.name AND w.review != 'rejected')
    """

    def materials(self, kind: str | None = None, pattern: str | None = None, review: str | None = None,
                  embedding: bool = False, rejected: bool = False) -> list[dict]:
        """Every material's current version, filtered. With `rejected`, also every material
        whose versions are all rejected, by its newest version."""
        rows = self._CURRENT + (f" UNION ALL {self._ALL_REJECTED}" if rejected else "")
        sql, args = f"SELECT * FROM ({rows}) WHERE 1", []
        for column, value in (("kind", kind), ("pattern", pattern), ("review", review)):
            if value:
                sql += f" AND {column} = ?"
                args.append(value)
        with self._db() as db:
            return [self._version(r, embedding) for r in db.execute(sql + " ORDER BY name", args)]

    def material(self, name: str) -> dict | None:
        """A material: its current version, every version, and who uses it."""
        with self._db() as db:
            head = db.execute("SELECT * FROM material WHERE name = ?", (name,)).fetchone()
            if head is None:
                return None
            versions = [self._version(r) for r in db.execute(
                "SELECT m.kind, m.pattern, v.* FROM version v JOIN material m ON m.name = v.name "
                "WHERE v.name = ? ORDER BY v.version", (name,))]
            uses = [self._use(r) for r in db.execute("SELECT * FROM use WHERE name = ? ORDER BY user", (name,))]
        current = next((v for v in reversed(versions) if v["review"] != "rejected"), None)
        return {**dict(head), "current": current, "versions": versions, "uses": uses}

    def review(self, name: str, version: int, state: str, note: str = "") -> None:
        if state not in REVIEWS:
            raise ValueError(f"a review is one of {', '.join(REVIEWS)}")
        with self._db() as db:
            if db.execute("UPDATE version SET review = ?, note = ? WHERE name = ? AND version = ?",
                          (state, note, name, version)).rowcount == 0:
                raise KeyError(f"no version {version} of `{name}`")

    def delete(self, name: str, version: int | None = None, force: bool = False,
               materials_dir: Path | None = None) -> dict:
        """Delete a material, or one version of it: its rows leave the library and its folder goes
        to the trash, with a `deleted.json` beside the maps. Deleting a material's only version
        deletes the material. A material that is used is deleted only with `force`, which drops
        its uses (raises InUse otherwise); its copy in the game, `materials_dir/<name>/`, goes to
        the trash too. Returns what was deleted and what went to the trash."""
        entry = self.material(name)
        if entry is None:
            raise KeyError(f"no material `{name}`")
        if version is not None and all(v["version"] != version for v in entry["versions"]):
            raise KeyError(f"no version {version} of `{name}`")
        whole = version is None or len(entry["versions"]) == 1
        if whole and entry["uses"] and not force:
            raise InUse(name, entry["uses"])

        if whole:
            folder = self.root / "materials" / name
            record = {k: entry[k] for k in ("name", "kind", "pattern", "created", "versions", "uses")}
            game = Path(materials_dir) / name if materials_dir else None
        else:
            folder = self.dir(name, version)
            record = {"name": name, "kind": entry["kind"], "pattern": entry["pattern"],
                      "version": next(v for v in entry["versions"] if v["version"] == version)}
            game = None
        record["deleted"] = now()
        trashed = []
        with self._db() as db:  # rolled back if a folder cannot go to the trash
            if whole:
                db.execute("DELETE FROM use WHERE name = ?", (name,))
                db.execute("DELETE FROM version WHERE name = ?", (name,))
                db.execute("DELETE FROM material WHERE name = ?", (name,))
            else:
                db.execute("DELETE FROM version WHERE name = ? AND version = ?", (name, version))
            db.execute("INSERT INTO deleted VALUES (?, ?, ?, ?)",
                       (name, None if whole else version, json.dumps(record), record["deleted"]))
            if folder.is_dir():
                (folder / "deleted.json").write_text(json.dumps(record, indent=2) + "\n")
                try:
                    trash(folder)
                except RuntimeError:
                    (folder / "deleted.json").unlink()
                    raise
                trashed.append(str(folder))
        if game is not None and game.is_dir():
            trash(game)
            trashed.append(str(game))
        return {"name": name, "version": None if whole else version, "material_deleted": whole,
                "uses_dropped": [u["user"] for u in entry["uses"]] if whole else [], "trashed": trashed}

    # -- uses ------------------------------------------------------------------------------------

    @staticmethod
    def _use(row: sqlite3.Row) -> dict:
        return {**dict(row), "tint": json.loads(row["tint"])}

    def use(self, user: str, name: str, tint: list[float], tile_metres: float) -> None:
        with self._db() as db:
            db.execute("INSERT INTO use VALUES (?, ?, ?, ?, ?) ON CONFLICT(user) DO UPDATE SET "
                       "name = excluded.name, tint = excluded.tint, tile_metres = excluded.tile_metres, "
                       "updated = excluded.updated", (user, name, json.dumps(tint), tile_metres, now()))

    def uses(self, name: str | None = None) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM use" + (" WHERE name = ?" if name else "") + " ORDER BY user",
                              (name,) if name else ())
            return [self._use(r) for r in rows]

    # -- jobs ------------------------------------------------------------------------------------

    def add_job(self, kind: str, request: dict) -> int:
        with self._db() as db:
            return db.execute("INSERT INTO job (kind, request, state, created, updated) VALUES (?, ?, 'queued', ?, ?)",
                              (kind, json.dumps(request), now(), now())).lastrowid

    def job(self, job_id: int) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM job WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else self._job(row)

    def jobs(self, limit: int = 50) -> list[dict]:
        """The newest jobs, newest first."""
        with self._db() as db:
            return [self._job(r) for r in db.execute("SELECT * FROM job ORDER BY id DESC LIMIT ?", (limit,))]

    @staticmethod
    def _job(row: sqlite3.Row) -> dict:
        out = dict(row)
        out["request"] = json.loads(out["request"])
        out["result"] = json.loads(out["result"]) if out["result"] else None
        return out

    def set_job(self, job_id: int, state: str, result: dict | None = None, error: str | None = None) -> None:
        with self._db() as db:
            db.execute("UPDATE job SET state = ?, result = ?, error = ?, updated = ? WHERE id = ?",
                       (state, json.dumps(result) if result is not None else None, error, now(), job_id))

    def unfinished_jobs(self) -> list[int]:
        """Jobs queued or cut off mid-run, oldest first: the service runs them again on start."""
        with self._db() as db:
            return [r["id"] for r in db.execute("SELECT id FROM job WHERE state IN ('queued', 'running') ORDER BY id")]

    # -- the game --------------------------------------------------------------------------------

    def export(self, name: str, materials_dir: Path) -> Path:
        """Copy a material's current version to `materials_dir/<name>/`, where the game loads it
        by name: its maps and a `material.toml` with its description and `tile_metres`."""
        entry = self.material(name)
        if entry is None or entry["current"] is None:
            raise KeyError(f"no material `{name}` with a version that is not rejected")
        v = entry["current"]
        src, dest = self.dir(name, v["version"]), Path(materials_dir) / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for m in MAPS:
            shutil.copyfile(src / f"{m}.png", dest / f"{m}.png")
        fields = {"name": name, "version": v["version"], "kind": v["kind"], "pattern": v["pattern"],
                  "finish": v["finish"], "colour": v["colour"], "detail": v["detail"], "review": v["review"],
                  "courses": v["courses"], "tile_metres": v["tile_metres"], "albedo_mean": v["albedo_mean"],
                  "source": "assetgen textures/library: FLUX.2 klein from words, CHORD maps",
                  "maps": [f"{m}.png" for m in MAPS]}
        lines = [f"{k} = {json.dumps(val)}" for k, val in fields.items() if val is not None]
        (dest / "material.toml").write_text("\n".join(lines) + "\n")
        return dest
