"""Minimal ComfyUI HTTP client. Standard library only.

Talks to a running ComfyUI server (`pixi run comfyui`) over its REST API:
check the node registry, upload an input image, queue an API-format workflow,
poll until it finishes, fetch what it produced. No websocket; polling
`/history` is plenty for batch work.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex
        self._object_info: dict | None = None

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, params=None, data=None, headers=None) -> bytes:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise ComfyError(f"{method} {path} -> HTTP {e.code}: {body[:2000]}") from None
        except urllib.error.URLError as e:
            raise ComfyError(
                f"{method} {path} -> {e.reason} (is ComfyUI running? start it with `pixi run comfyui`)"
            ) from None

    def _get_json(self, path: str, params=None):
        return json.loads(self._request("GET", path, params=params))

    def _post_json(self, path: str, payload):
        data = json.dumps(payload).encode()
        return json.loads(self._request("POST", path, data=data, headers={"Content-Type": "application/json"}))

    def is_up(self) -> bool:
        try:
            self._get_json("/system_stats")
            return True
        except ComfyError:
            return False

    # -- node registry -------------------------------------------------------

    def object_info(self, refresh: bool = False) -> dict:
        if self._object_info is None or refresh:
            self._object_info = self._get_json("/object_info")
        return self._object_info

    def validate(self, workflow: dict, refresh: bool = False) -> list[str]:
        """Check an API-format workflow against the server's node registry.

        Catches the cheap mistakes before queueing: unknown node classes,
        unknown input names, links to missing nodes, and combo values the
        server does not offer, which is how a missing model file shows up.
        """
        info = self.object_info(refresh=refresh)
        problems: list[str] = []
        for nid, node in workflow.items():
            ct = node.get("class_type")
            spec = info.get(ct)
            if spec is None:
                problems.append(f"node {nid}: unknown class_type {ct!r}")
                continue
            declared: dict = {}
            for group in ("required", "optional"):
                declared.update(spec.get("input", {}).get(group, {}) or {})
            # A dynamic combo's options carry their own inputs, sent as "<combo>.<input>".
            for name, decl in list(declared.items()):
                if isinstance(decl, (list, tuple)) and decl and decl[0] == "COMFY_DYNAMICCOMBO_V3":
                    for option in (decl[1] if len(decl) > 1 else {}).get("options", []):
                        for group in ("required", "optional"):
                            for sub, sub_decl in (option.get("inputs", {}).get(group, {}) or {}).items():
                                declared[f"{name}.{sub}"] = sub_decl
            for name, value in (node.get("inputs") or {}).items():
                if name not in declared:
                    problems.append(f"node {nid} ({ct}): unknown input {name!r}")
                    continue
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    if value[0] not in workflow:
                        problems.append(f"node {nid} ({ct}): input {name!r} links to missing node {value[0]!r}")
                    continue
                options = self._combo_options(declared[name])
                if options is not None and value not in options:
                    shown = ", ".join(map(str, options[:6])) + (" ..." if len(options) > 6 else "")
                    problems.append(
                        f"node {nid} ({ct}): {name}={value!r} is not available on the server "
                        f"(it offers: {shown or 'nothing'})"
                    )
        return problems

    @staticmethod
    def _combo_options(decl) -> list | None:
        if not isinstance(decl, (list, tuple)) or not decl:
            return None
        kind = decl[0]
        extra = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
        if isinstance(kind, list):
            return kind
        if kind == "COMBO" and isinstance(extra.get("options"), list):
            return extra["options"]
        return None

    # -- inputs --------------------------------------------------------------

    def upload_image(self, path: Path, name: str | None = None, subfolder: str = "", overwrite: bool = True) -> dict:
        """POST an image into ComfyUI's input folder. Returns {name, subfolder, type}."""
        path = Path(path)
        name = name or path.name
        boundary = "----assetgen" + uuid.uuid4().hex
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        parts: list[bytes] = []

        def field(key: str, value: str) -> None:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())

        field("overwrite", "true" if overwrite else "false")
        if subfolder:
            field("subfolder", subfolder)
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
            + path.read_bytes()
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        raw = self._request(
            "POST", "/upload/image", data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return json.loads(raw)

    # -- execution -----------------------------------------------------------

    def queue(self, workflow: dict) -> str:
        resp = self._post_json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if "prompt_id" not in resp:
            raise ComfyError(f"queue rejected: {json.dumps(resp)[:2000]}")
        if resp.get("node_errors"):
            # ComfyUI still runs the outputs that validated and silently drops the rest.
            raise ComfyError(f"queue accepted with node errors (some outputs would not run): "
                             f"{json.dumps(resp['node_errors'])[:2000]}")
        return resp["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 1800.0, poll: float = 1.0, heartbeat: float = 30.0) -> dict:
        """Block until the prompt shows up in /history, then return its entry."""
        t0 = time.time()
        last_beat = t0
        while True:
            entry = self._get_json(f"/history/{prompt_id}").get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyError(self._describe_error(entry))
                if status.get("completed", True):
                    return entry
            now = time.time()
            if now - t0 > timeout:
                raise ComfyError(f"timed out after {timeout:.0f}s waiting for prompt {prompt_id}")
            if heartbeat and now - last_beat >= heartbeat:
                print(f"    ... still running ({now - t0:.0f}s)")
                last_beat = now
            time.sleep(poll)

    @staticmethod
    def _describe_error(entry: dict) -> str:
        lines = ["ComfyUI reported an execution error"]
        for msg in (entry.get("status") or {}).get("messages", []):
            if not (isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict)):
                continue
            kind, info = msg
            if kind == "execution_error":
                lines.append(
                    f"  node {info.get('node_id')} ({info.get('node_type')}): "
                    f"{info.get('exception_type')}: {info.get('exception_message')}"
                )
        return "\n".join(lines)

    # -- outputs -------------------------------------------------------------

    def collect_outputs(self, entry: dict) -> list[dict]:
        """Flatten a history entry into [{node, kind, filename, subfolder, type}, ...]."""
        found: list[dict] = []
        for node_id, out in (entry.get("outputs") or {}).items():
            for kind in ("images", "3d", "gifs", "audio", "files"):
                for item in out.get(kind) or []:
                    if isinstance(item, dict) and "filename" in item:
                        found.append({"node": node_id, "kind": kind, **item})
        return found

    def download(self, item: dict, dest: Path) -> Path:
        params = {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        data = self._request("GET", "/view", params=params)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
