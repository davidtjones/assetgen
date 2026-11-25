"""The material library's HTTP API: the one way into the library, for tools, the game and the
review page.

    pixi run library serve [--port 8190]      then open http://127.0.0.1:8190/ for the page

JSON in and out. A description is `{"kind", "pattern", "finish", "colour", "detail"}` (see
`words.py`); a colour is linear RGB, three numbers. Drawing a material takes a minute, so a
request that draws one returns `202 {"job": id}`, and `GET /jobs/<id>` says when it is done.
Jobs run one at a time, in order, and a job left unfinished when the service stopped is run
again when it starts.

    GET  /                                    the review page (`web/index.html`)
    GET  /health
    GET  /vocabulary                          the kinds and patterns a description can name
    GET  /materials?kind=&pattern=&review=&all=1
                                              every material's current version; with `all`,
                                              also those whose every version is rejected
    GET  /materials/<name>                    a material: current version, all versions, uses
    GET  /materials/<name>/v<n>/<file>        a version's maps and sheet.jpg; thumb.jpg is
                                              its albedo at 256 pixels, made on first ask
    POST /search    {description, colour?}    the candidates of its kind and pattern, ranked
    POST /resolve   {description, colour?, user?, source?}
                    the best match (200, `use` recorded for `user`), or a job drawing a new
                    material (202). The main call: "a material like this, for this".
    POST /materials {description, seed?, source?}     draw a new material, matched or not (202)
    POST /materials/<name>/revise {finish?, colour?, detail?, seed?}
                    draw a new version from changed words (202). Everything that uses the
                    material gets the new version when it is next exported.
    POST /materials/<name>/v<n>/review {review: new|approved|rejected, note?}
    POST /materials/<name>/export             copy its current version into the game
    DELETE /materials/<name>?force=1          delete a material: its folder and its copy in the
                                              game go to the trash. A used one needs `force`
                                              (409 with its uses otherwise), which drops them.
    DELETE /materials/<name>/v<n>             delete one version (its only one: the material)
    GET  /uses?name=                          who uses which material
    PUT  /uses      {user, name, tint?, tile_metres?}
    GET  /jobs?limit=                         the newest jobs, newest first
    GET  /jobs/<id>
"""

from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path

from aiohttp import web
from PIL import Image

from common.paths import game3d_assets
from textures.library.embed import embed
from textures.library.make import make
from textures.library.match import rank, tint
from textures.library.store import InUse, Store
from textures.library.words import VOCABULARY, Description

PORT = 8190
PAGE = Path(__file__).parent / "web" / "index.html"
THUMB = 256
STORE = web.AppKey("store", Store)
QUEUE = web.AppKey("queue", asyncio.Queue)


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def best(store: Store, desc: Description, colour: list[float] | None) -> tuple[dict | None, list[dict]]:
    """The best match for `desc` in the library (None if nothing matches), and every candidate."""
    ranked = rank(desc, embed([desc.text()])[0], colour,
                  store.materials(desc.kind, desc.pattern, embedding=True))
    return (ranked[0] if ranked and ranked[0]["matches"] else None), ranked


def run_job(store: Store, job_id: int) -> None:
    """Draw what job `job_id` asks for: a `resolve` (unless something now matches), a `create`
    or a `revise`. Runs in a worker thread."""
    job = store.job(job_id)
    store.set_job(job_id, "running")
    req = job["request"]
    try:
        colour, user = req.get("colour"), req.get("user")
        name = None
        if job["kind"] == "revise":
            name = req["name"]
            current = store.material(name)["current"]
            desc = Description.of({**current, **{k: req[k] for k in ("finish", "colour", "detail") if k in req}})
        else:
            desc = Description.of(req["description"])
        if job["kind"] == "resolve":
            # A job queued earlier may have drawn a match since this one was asked for.
            found, _ = best(store, desc, colour)
            if found:
                result = {**found, "made": False}
                if user:
                    store.use(user, found["name"], found["tint"], found["tile_metres"])
                store.set_job(job_id, "done", result)
                return
        print(f"[library] job {job_id}: drawing {desc.kind} {desc.pattern}: {desc.finish}", flush=True)
        staged = store.staging(f"{desc.kind}-{desc.pattern}")
        seed, measured = make(desc, staged, int(req.get("seed", 0)), log=lambda m: print(m, flush=True))
        source = {**(req.get("source") or {}), "job": job_id, "check": measured.pop("check"),
                  **({"user": user} if user else {})}
        name, version = store.add(desc, staged, measured, embed([desc.text()])[0], seed, source, name)
        result = {"name": name, "version": version, "made": True, "similarity": None,
                  "tint": tint(measured["albedo_mean"], colour), "tile_metres": measured["tile_metres"]}
        if user:
            store.use(user, name, result["tint"], result["tile_metres"])
        store.set_job(job_id, "done", result)
        print(f"[library] job {job_id}: {name} v{version}", flush=True)
    except (Exception, SystemExit) as e:
        store.set_job(job_id, "failed", error=f"{e}\n{traceback.format_exc()}")
        print(f"[library] job {job_id} failed: {e}", flush=True)


async def worker(app: web.Application):
    loop = asyncio.get_running_loop()
    for job_id in app[STORE].unfinished_jobs():
        app[QUEUE].put_nowait(job_id)

    async def run():
        while True:
            job_id = await app[QUEUE].get()
            await loop.run_in_executor(None, run_job, app[STORE], job_id)

    task = asyncio.create_task(run())
    yield
    task.cancel()


def _bad(message: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(text=json.dumps({"error": message}), content_type="application/json")


async def _body(request: web.Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise _bad("send a JSON object")
    if not isinstance(body, dict):
        raise _bad("send a JSON object")
    return body


def _desc(body: dict) -> Description:
    try:
        return Description.of(body.get("description") or {})
    except ValueError as e:
        raise _bad(str(e))


def _enqueue(request: web.Request, kind: str, body: dict) -> int:
    job_id = request.app[STORE].add_job(kind, body)
    request.app[QUEUE].put_nowait(job_id)
    return job_id


routes = web.RouteTableDef()


@routes.get("/")
async def page(request):
    return web.FileResponse(PAGE, headers={"Cache-Control": "no-cache"})


def _thumb(version_dir: Path) -> Path:
    """The version's albedo at THUMB pixels, made once and kept beside it."""
    out = version_dir / "thumb.jpg"
    if not out.is_file():
        im = Image.open(version_dir / "albedo.png").convert("RGB")
        im.thumbnail((THUMB, THUMB))
        part = out.with_suffix(".part.jpg")
        im.save(part, quality=88)
        part.replace(out)
    return out


@routes.get("/health")
async def health(request):
    return web.json_response({"ok": True, "root": str(request.app[STORE].root), "queued": request.app[QUEUE].qsize()})


@routes.get("/vocabulary")
async def vocabulary(request):
    return web.json_response(VOCABULARY)


@routes.get("/materials")
async def materials(request):
    q = request.query
    return web.json_response(request.app[STORE].materials(q.get("kind"), q.get("pattern"), q.get("review"),
                                                          rejected=q.get("all") in ("1", "true")))


@routes.get("/materials/{name}")
async def material(request):
    found = request.app[STORE].material(request.match_info["name"])
    return web.json_response(found) if found else _error(404, f"no material `{request.match_info['name']}`")


@routes.get("/materials/{name}/v{version:\\d+}/{file}")
async def material_file(request):
    store = request.app[STORE]
    path = store.dir(request.match_info["name"], int(request.match_info["version"])) / request.match_info["file"]
    if path.name == "thumb.jpg" and (path.parent / "albedo.png").is_file():
        path = await asyncio.get_running_loop().run_in_executor(None, _thumb, path.parent)
    if path.suffix not in (".png", ".jpg") or not path.is_file():
        return _error(404, "no such file")
    return web.FileResponse(path)


@routes.post("/search")
async def search(request):
    body = await _body(request)
    desc = _desc(body)
    _, ranked = await asyncio.get_running_loop().run_in_executor(None, best, request.app[STORE], desc, body.get("colour"))
    return web.json_response({"candidates": ranked})


@routes.post("/resolve")
async def resolve(request):
    body = await _body(request)
    desc = _desc(body)
    store = request.app[STORE]
    found, ranked = await asyncio.get_running_loop().run_in_executor(None, best, store, desc, body.get("colour"))
    if found is None:
        return web.json_response({"job": _enqueue(request, "resolve", body), "candidates": ranked[:5]}, status=202)
    if body.get("user"):
        store.use(body["user"], found["name"], found["tint"], found["tile_metres"])
    return web.json_response({"match": {**found, "made": False}, "candidates": ranked[:5]})


@routes.post("/materials")
async def create(request):
    body = await _body(request)
    _desc(body)
    return web.json_response({"job": _enqueue(request, "create", body)}, status=202)


@routes.post("/materials/{name}/revise")
async def revise(request):
    body = await _body(request)
    name = request.match_info["name"]
    if request.app[STORE].material(name) is None:
        return _error(404, f"no material `{name}`")
    return web.json_response({"job": _enqueue(request, "revise", {**body, "name": name})}, status=202)


@routes.post("/materials/{name}/v{version:\\d+}/review")
async def review(request):
    body = await _body(request)
    try:
        request.app[STORE].review(request.match_info["name"], int(request.match_info["version"]),
                                  body.get("review", ""), body.get("note", ""))
    except ValueError as e:
        return _error(400, str(e))
    except KeyError as e:
        return _error(404, str(e))
    return web.json_response({"ok": True})


@routes.post("/materials/{name}/export")
async def export(request):
    try:
        path = request.app[STORE].export(request.match_info["name"], game3d_assets("materials"))
    except KeyError as e:
        return _error(404, str(e))
    return web.json_response({"path": str(path)})


async def _delete(request: web.Request, version: int | None) -> web.Response:
    force = request.query.get("force") in ("1", "true")
    try:
        done = await asyncio.get_running_loop().run_in_executor(
            None, lambda: request.app[STORE].delete(request.match_info["name"], version, force=force,
                                                    materials_dir=game3d_assets("materials")))
    except InUse as e:
        return web.json_response({"error": str(e), "uses": e.uses}, status=409)
    except KeyError as e:
        return _error(404, str(e))
    except RuntimeError as e:
        return _error(500, str(e))
    print(f"[library] deleted {done['name']}" + (f" v{done['version']}" if done["version"] else "")
          + f"; trashed {', '.join(done['trashed']) or 'nothing'}", flush=True)
    return web.json_response(done)


@routes.delete("/materials/{name}")
async def delete_material(request):
    return await _delete(request, None)


@routes.delete("/materials/{name}/v{version:\\d+}")
async def delete_version(request):
    return await _delete(request, int(request.match_info["version"]))


@routes.get("/uses")
async def uses(request):
    return web.json_response(request.app[STORE].uses(request.query.get("name")))


@routes.put("/uses")
async def put_use(request):
    body = await _body(request)
    store = request.app[STORE]
    found = store.material(body.get("name", ""))
    if not body.get("user") or found is None or found["current"] is None:
        return _error(400, "send a user and the name of a material in the library")
    store.use(body["user"], body["name"], body.get("tint", [1.0, 1.0, 1.0]),
              float(body.get("tile_metres", found["current"]["tile_metres"])))
    return web.json_response({"ok": True})


@routes.get("/jobs")
async def jobs(request):
    try:
        limit = max(1, min(500, int(request.query.get("limit", 50))))
    except ValueError:
        return _error(400, "limit is a number")
    return web.json_response(request.app[STORE].jobs(limit))


@routes.get("/jobs/{id:\\d+}")
async def job(request):
    found = request.app[STORE].job(int(request.match_info["id"]))
    return web.json_response(found) if found else _error(404, "no such job")


def app(store: Store | None = None) -> web.Application:
    a = web.Application()
    a[STORE] = store or Store()
    a[QUEUE] = asyncio.Queue()
    a.add_routes(routes)
    a.cleanup_ctx.append(worker)
    return a


def serve(port: int = PORT) -> None:
    web.run_app(app(), host="127.0.0.1", port=port, print=lambda m: print(f"[library] {m}", flush=True))
