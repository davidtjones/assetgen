"""
Foliage studio dev server.

Serves the studio viewer (preview/studio.html) AND regenerates geometry live from
the SAME Python model the bake uses — so tuning in the browser and the final
`pixi run blades` bake never diverge. Stdlib only (no Flask); mesh generation is
microseconds and the glTF is a couple KB, so the localhost round-trip feels
instant under a debounced slider.

Endpoints:
    GET  /                      -> static files under assetgen/foliage/
    GET  /params?profile=NAME   -> that profile's default blade/mesh/cluster values
    POST /generate              -> {mode, profile, seed, blade{}, mesh{}, cluster{}}
                                   returns a glTF JSON string

Run:
    pixi run studio             # then open http://localhost:8000/preview/studio.html
"""

import importlib
import json
import shutil
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Hot-reload relies on always reading LIVE source. The .pyc bytecode cache records
# source mtime truncated to whole seconds, so two edits in the same second can make
# importlib.reload reuse stale bytecode. Disable bytecode writing and clear any
# existing cache before the first import so reloads can never go stale.
sys.dont_write_bytecode = True
shutil.rmtree(Path(__file__).resolve().parent / "__pycache__", ignore_errors=True)

# Import the generator as MODULES (not names) so we can hot-reload them: editing
# blade_core / mesh_backend / gltf_io / generate_geometry — or any profile YAML —
# then hitting "regenerate" in the studio picks up the change with no restart.
# gltf_io lives in the shared assetgen/common/ module (used by every generator).
from common import gltf_io
from foliage.scripts import blade_core, generate_geometry, mesh_backend

# Dependency order: blade_core ← gltf_io, mesh_backend ← generate_geometry.
_RELOADABLE = (blade_core, gltf_io, mesh_backend, generate_geometry)

MODULE_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = MODULE_ROOT / "profiles"
# Studio presets: full session snapshots (mode, all overrides, field/wind/light)
# saved as JSON so a tuning can be reloaded later. Distinct from the profile YAML
# + "Copy YAML" bake path — presets are for the studio, not the asset bake.
PRESETS_DIR = MODULE_ROOT / "presets"


def _reload():
    """Re-import the generator modules so source edits take effect live. A syntax
    error here propagates to the request handler and surfaces in the studio."""
    for m in _RELOADABLE:
        importlib.reload(m)

# Which fields the studio may override, per group. Keeps POST handling explicit
# and stops a bad key from crashing setattr.
_CLUSTER_RANGES = {
    "length_range", "base_thickness_range", "tip_thickness_range",
    "compliance_gamma_range", "spread_range",
}
_CLUSTER_KEYS = {
    "n_blades", "n_blades_var", "x_spread", "fan_out_strength", "noise_strength",
} | _CLUSTER_RANGES
# tuft_radius_px / tuft_splay_deg are intentionally NOT overridable from the studio:
# they resolve from x_spread / fan_out_strength, and echoing a resolved value back
# would pin them and make those sliders do nothing. Set them in the profile YAML
# only if you want to override the derivation.
_MESH_KEYS = {
    "segments", "droop_deg", "droop_gamma", "droop_noise_deg", "up_normal_blend",
    "root_color", "tip_color", "tuft_azimuth_jitter",
}
_BLADE_KEYS = {"length", "base_thickness", "tip_thickness", "compliance_gamma", "spread"}


def _profile_names() -> list[str]:
    return [p.stem for p in sorted(PROFILES_DIR.glob("*.yaml"))]


def _safe_name(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9 _-]", "", name or "").strip().replace(" ", "_") or "preset"


def _preset_names() -> list[str]:
    return [p.stem for p in sorted(PRESETS_DIR.glob("*.json"))] if PRESETS_DIR.exists() else []


def _read_preset(name: str) -> dict:
    return json.loads((PRESETS_DIR / f"{_safe_name(name)}.json").read_text())


def _write_preset(name: str, data) -> str:
    PRESETS_DIR.mkdir(exist_ok=True)
    safe = _safe_name(name)
    (PRESETS_DIR / f"{safe}.json").write_text(json.dumps(data, indent=2))
    return safe


def _apply(obj, overrides: dict, allowed: set):
    for k, v in (overrides or {}).items():
        if k not in allowed:
            continue
        if isinstance(v, list) and (k in ("root_color", "tip_color") or k in _CLUSTER_RANGES):
            v = tuple(v)
        if k in ("segments", "n_blades", "n_blades_var"):
            v = int(round(v))
        setattr(obj, k, v)
    return obj


def _resolved_defaults(name: str) -> dict:
    """The values the studio's sliders should initialize to for a profile."""
    _reload()
    cluster, mesh = generate_geometry._load_profile(PROFILES_DIR / f"{name}.yaml")
    blade = mesh_backend.canonical_blade(cluster)
    return {
        "profiles": _profile_names(),
        "blade": {
            "length": blade.length,
            "base_thickness": blade.base_thickness,
            "tip_thickness": blade.tip_thickness,
            "compliance_gamma": blade.compliance_gamma,
            "spread": blade.spread,
        },
        "mesh": {
            "segments": mesh.segments,
            "droop_deg": mesh.droop_deg,
            "droop_gamma": mesh.droop_gamma,
            "droop_noise_deg": mesh.droop_noise_deg,
            "up_normal_blend": mesh.up_normal_blend,
            "root_color": list(mesh.root_color),
            "tip_color": list(mesh.tip_color),
            "tuft_azimuth_jitter": mesh.tuft_azimuth_jitter,
        },
        "cluster": {
            "n_blades": cluster.n_blades,
            "n_blades_var": cluster.n_blades_var,
            "x_spread": cluster.x_spread,
            "fan_out_strength": cluster.fan_out_strength,
            "noise_strength": cluster.noise_strength,
            "length_range": list(cluster.length_range),
            "base_thickness_range": list(cluster.base_thickness_range),
            "tip_thickness_range": list(cluster.tip_thickness_range),
            "compliance_gamma_range": list(cluster.compliance_gamma_range),
            "spread_range": list(cluster.spread_range),
        },
    }


def _generate(req: dict) -> str:
    _reload()
    name = req.get("profile") or _profile_names()[0]
    cluster, mesh = generate_geometry._load_profile(PROFILES_DIR / f"{name}.yaml")
    _apply(cluster, req.get("cluster"), _CLUSTER_KEYS)
    _apply(mesh, req.get("mesh"), _MESH_KEYS)

    if req.get("mode") == "tuft":
        geo = mesh_backend.build_tuft_mesh(cluster, mesh, seed=int(req.get("seed", 0)))
        mesh_name = f"{name}_tuft"
    else:
        blade = mesh_backend.canonical_blade(cluster)
        _apply(blade, req.get("blade"), _BLADE_KEYS)
        geo = mesh_backend.build_blade_mesh(blade, mesh)
        mesh_name = f"{name}_blade"

    gltf = gltf_io.build_gltf(
        positions=geo["positions"], indices=geo["indices"], normals=geo["normals"],
        uvs=geo["uvs"], uv1=geo["uv1"], colors=geo["colors"], name=mesh_name, double_sided=True,
    )
    return gltf_io.dumps(gltf)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(MODULE_ROOT), **kw)

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode() if not isinstance(obj, str) else obj.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/params":
            name = parse_qs(parsed.query).get("profile", [None])[0] or _profile_names()[0]
            try:
                self._send_json(_resolved_defaults(name))
            except Exception as e:  # bad profile name, etc.
                self._send_json({"error": str(e)}, code=400)
            return
        if parsed.path == "/presets":
            self._send_json({"presets": _preset_names()})
            return
        if parsed.path == "/preset":
            name = parse_qs(parsed.query).get("name", [None])[0]
            try:
                self._send_json(_read_preset(name))
            except Exception as e:
                self._send_json({"error": str(e)}, code=400)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
                saved = _write_preset(req.get("name", ""), req.get("preset", {}))
                self._send_json({"ok": True, "name": saved, "presets": _preset_names()})
            except Exception as e:
                self._send_json({"error": repr(e)}, code=400)
            return
        if parsed.path != "/generate":
            self._send_json({"error": "unknown endpoint"}, code=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(_generate(req))  # already a JSON string
        except Exception as e:
            self._send_json({"error": repr(e)}, code=400)

    def log_message(self, fmt, *args):
        pass  # quiet; the studio polls a lot


def main(port: int = 8000):
    print(f"foliage studio: http://localhost:{port}/preview/studio.html")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
