#!/usr/bin/env python3
"""Local web editor for reviewing generated LIBERO spawn positions."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MUJOCO_GL", "egl")

VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
for path in (TOOLS_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_profile_tools import (  # noqa: E402
    FINAL_SPEED_REVIEW_THRESHOLD,
    normalize_pose,
    render_pose_png,
    setup_logging,
    simulate_pose,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_POSSIBLE_DIR = ROOT / "possible_spawn_positions"
DEFAULT_HOLDED_DIR = ROOT / "holded_spawn_positions"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790
TRANSLATION_STEP = 0.1
ROTATION_STEP_DEG = 90.0
ROTATION_STEP_RAD = 1.5707963267948966
POSE_KEYS = ("x", "y", "z", "r", "p", "h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and edit LIBERO spawn positions in a local browser.")
    parser.add_argument("--possible-dir", type=Path, default=DEFAULT_POSSIBLE_DIR)
    parser.add_argument("--holded-dir", type=Path, default=DEFAULT_HOLDED_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sim-seconds", type=float, default=5.0)
    parser.add_argument("--sim-fps", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pose_from_entry(entry: dict[str, Any]) -> dict[str, float]:
    for key in ("relative_scene_table_fixed_pose", "pose", "anchor_pose"):
        if isinstance(entry.get(key), dict):
            return normalize_pose(entry[key])
    simulation = entry.get("simulation")
    if isinstance(simulation, dict) and isinstance(simulation.get("final_anchor_pose"), dict):
        return normalize_pose(simulation["final_anchor_pose"])
    return {key: 0.0 for key in POSE_KEYS}


def corrected_location(entry: dict[str, Any], fallback: str) -> str:
    location = str(entry.get("location") or fallback)
    label = str(entry.get("label") or "")
    if label == "front" and location.endswith(".front_side"):
        return location[: -len(".front_side")] + ".back_side"
    if label == "back" and location.endswith(".back_side"):
        return location[: -len(".back_side")] + ".front_side"
    return location


class SpawnReviewStore:
    def __init__(
        self,
        possible_dir: Path,
        holded_dir: Path,
        camera_name: str,
        image_size: int,
        sim_seconds: float,
        sim_fps: int,
    ) -> None:
        self.possible_dir = possible_dir
        self.holded_dir = holded_dir
        self.camera_name = camera_name
        self.image_size = image_size
        self.sim_seconds = sim_seconds
        self.sim_fps = sim_fps
        self.modified_dir = possible_dir / f"modified_{time.strftime('%Y%m%d_%H%M%S')}"

    def source_root(self, source: str) -> Path:
        if source == "holded":
            return self.holded_dir
        if source == "possible":
            return self.possible_dir
        raise ValueError(f"Unknown source: {source}")

    def list_entries(self, source: str = "holded") -> list[dict[str, Any]]:
        root = self.source_root(source)
        entries = []
        for path in sorted(root.glob("*/*/*.json")):
            if path.name.startswith("_"):
                continue
            data = read_json(path)
            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) < 3:
                continue
            scene, fixed_type, label_file = parts[0], parts[1], parts[2]
            label = Path(label_file).stem
            for object_type, entry in sorted(data.items()):
                if object_type.startswith("_") or not isinstance(entry, dict):
                    continue
                entries.append(
                    {
                        "source": source,
                        "path": str(rel),
                        "scene": entry.get("scene", scene),
                        "fixed_type": entry.get("fixed_type", fixed_type),
                        "label": entry.get("label", label),
                        "location": corrected_location(entry, f"{fixed_type}.{label}"),
                        "object_type": entry.get("object_type", object_type),
                        "status": entry.get("status"),
                        "reason": entry.get("reason") or entry.get("simulation", {}).get("reason"),
                        "pose": pose_from_entry(entry),
                    }
                )
        return entries

    def load_entry(self, source: str, rel_path: str, object_type: str) -> tuple[Path, dict[str, Any]]:
        path = self.source_root(source) / rel_path
        data = read_json(path)
        entry = data.get(object_type)
        if not isinstance(entry, dict):
            raise KeyError(f"No entry for {object_type} in {path}")
        return path, entry

    def render(self, source: str, rel_path: str, object_type: str, pose: dict[str, float] | None = None) -> dict[str, Any]:
        _path, entry = self.load_entry(source, rel_path, object_type)
        pose = normalize_pose(pose if pose is not None else pose_from_entry(entry))
        location = corrected_location(entry, str(entry.get("location", "")))
        png = render_pose_png(
            scene=str(entry["scene"]),
            location=location,
            object_type=str(entry["object_type"]),
            pose=pose,
            camera_name=self.camera_name,
            image_size=self.image_size,
            settle_steps=0,
            force_safe_object_init=True,
        )
        return {
            "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "pose": pose,
            "entry": {**entry, "location": location},
        }

    def settle(self, source: str, rel_path: str, object_type: str, pose: dict[str, float]) -> dict[str, Any]:
        _source_path, entry = self.load_entry(source, rel_path, object_type)
        pose = normalize_pose(pose)
        location = corrected_location(entry, str(entry.get("location", "")))
        with tempfile.TemporaryDirectory(prefix="vlapb_settled_preview_") as tmp:
            final_image_path = Path(tmp) / "final.png"
            result = simulate_pose(
                scene=str(entry["scene"]),
                location=location,
                object_type=str(entry["object_type"]),
                pose=pose,
                camera_name=self.camera_name,
                image_size=self.image_size,
                sim_seconds=self.sim_seconds,
                sim_fps=self.sim_fps,
                final_image_path=final_image_path,
                force_safe_object_init=True,
            )
            final_png = final_image_path.read_bytes()
        stable = bool(result.get("velocity_stable")) and float(result.get("final_speed", 999.0)) <= FINAL_SPEED_REVIEW_THRESHOLD
        payload = {
            **entry,
            "location": location,
            "status": "possible" if stable else "holded",
            "modified_from": {
                "source": source,
                "path": rel_path,
                "object_type": object_type,
            },
            "pose": pose,
            "relative_scene_table_fixed_pose": result.get("final_anchor_pose"),
            "actual_pose": result.get("final_pose"),
            "start_pose": result.get("start_pose"),
            "simulation": result,
            "timestamp": time.time(),
        }
        return {
            "stable": stable,
            "result": payload,
            "image": "data:image/png;base64," + base64.b64encode(final_png).decode("ascii"),
        }

    def confirm_save(self, source: str, rel_path: str, object_type: str, result: dict[str, Any]) -> dict[str, Any]:
        source_path, entry = self.load_entry(source, rel_path, object_type)
        payload = dict(result)
        payload.setdefault("modified_from", {"source": source, "path": rel_path, "object_type": object_type})
        stable = payload.get("status") == "possible"
        if stable:
            target_path = self.modified_dir / safe_name(str(entry["scene"])) / safe_name(str(entry["fixed_type"])) / f"{safe_name(str(entry['label']))}.json"
            target_data = read_json(target_path)
            target_data[str(entry["object_type"])] = payload
            write_json(target_path, target_data)
            if source == "holded":
                source_data = read_json(source_path)
                source_data.pop(object_type, None)
                write_json(source_path, source_data)
        else:
            target_path = self.holded_dir / rel_path
            target_data = read_json(target_path)
            target_data[str(entry["object_type"])] = payload
            write_json(target_path, target_data)
        return {"stable": stable, "path": str(target_path), "result": payload}


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LIBERO Spawn Position Review</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    main { display: grid; grid-template-columns: 360px 1fr; height: 100vh; }
    aside { border-right: 1px solid #333; overflow: auto; padding: 12px; }
    section { padding: 12px; overflow: auto; }
    button, select { background: #222; color: #eee; border: 1px solid #555; padding: 7px 9px; margin: 3px; }
    button:hover { background: #333; }
    .row { border-bottom: 1px solid #2a2a2a; padding: 8px; cursor: pointer; }
    .row:hover, .row.active { background: #242424; }
    img { max-width: min(900px, 100%); image-rendering: auto; background: #000; }
    code { background: #222; padding: 2px 4px; }
    .pose { display: grid; grid-template-columns: repeat(6, 1fr); max-width: 720px; gap: 6px; }
    .pose div { background: #1d1d1d; padding: 8px; }
    .status { white-space: pre-wrap; color: #b7e3ff; }
  </style>
</head>
<body>
<main>
  <aside>
    <div>
      <button onclick="loadList('holded')">Holded</button>
      <button onclick="loadList('possible')">Possible</button>
    </div>
    <div id="list"></div>
  </aside>
  <section>
    <h2 id="title">Select an entry</h2>
    <p>Keyboard: <code>W/A/S/D</code> move XY by 0.1m, <code>Z/X</code> move Z, <code>R/F</code> roll, <code>P/O</code> pitch, <code>H/J</code> heading by 90 degrees. Press <code>Enter</code> to settle 5s, then confirm before saving. Mouse: click list item to auto-spawn + 5s check, click image to re-run 5s check.</p>
    <div class="pose" id="pose"></div>
    <div>
      <button onclick="renderCurrent()">Render</button>
      <button onclick="settleCurrent()">Settle 5s</button>
      <button onclick="confirmSave()">Confirm Save</button>
    </div>
    <p class="status" id="status"></p>
    <img id="image" onclick="settleCurrent()">
  </section>
</main>
<script>
let entries = [];
let current = null;
let pose = {x:0,y:0,z:0,r:0,p:0,h:0};
let pendingResult = null;
const tStep = 0.1;
const rStep = Math.PI / 2;

function setStatus(msg) { document.getElementById('status').textContent = msg; }
function poseHtml() {
  document.getElementById('pose').innerHTML = ['x','y','z','r','p','h']
    .map(k => `<div><b>${k}</b><br>${Number(pose[k] || 0).toFixed(4)}</div>`).join('');
}
async function api(path, payload=null) {
  const opts = payload ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)} : {};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
async function loadList(source) {
  setStatus('Loading ' + source + '...');
  entries = await api('/api/list?source=' + source);
  document.getElementById('list').innerHTML = entries.map((e,i) =>
    `<div class="row" id="row-${i}" onclick="selectEntry(${i})"><b>${e.scene}</b><br>${e.fixed_type}/${e.label}/${e.object_type}<br>${e.status || ''} ${e.reason || ''}</div>`
  ).join('');
  setStatus(`Loaded ${entries.length} entries from ${source}`);
}
async function selectEntry(i) {
  current = entries[i];
  pose = {...current.pose};
  pendingResult = null;
  document.querySelectorAll('.row').forEach(x => x.classList.remove('active'));
  document.getElementById('row-' + i).classList.add('active');
  document.getElementById('title').textContent = `${current.scene} / ${current.fixed_type} / ${current.label} / ${current.object_type}`;
  poseHtml();
  // On mouse selection, immediately spawn and run 5-second stability check.
  // For holded entries we do not auto-arm saving; user must explicitly settle again
  // (or click Settle 5s / image) before Confirm Save.
  if (current.source === 'possible') await settleCurrent(false);
  else await settleCurrent(false);
}
async function renderCurrent() {
  if (!current) return;
  setStatus('Rendering...');
  poseHtml();
  const data = await api('/api/render', {...current, pose});
  document.getElementById('image').src = data.image;
  pose = data.pose;
  poseHtml();
  setStatus('Rendered.');
}
async function settleCurrent(allowConfirm=true) {
  if (!current) return;
  setStatus('Simulating 5 seconds...');
  const data = await api('/api/settle', {...current, pose});
  pendingResult = allowConfirm ? data.result : null;
  if (data.image) document.getElementById('image').src = data.image;
  if (pendingResult.relative_scene_table_fixed_pose) {
    pose = {...pendingResult.relative_scene_table_fixed_pose};
    poseHtml();
  }
  if (!pendingResult && data.result && data.result.relative_scene_table_fixed_pose) {
    pose = {...data.result.relative_scene_table_fixed_pose};
    poseHtml();
  }
  const confirmText = allowConfirm ? ' Click Confirm Save only if this pose is acceptable.' : ' This possible entry was previewed after 5 seconds; nothing was saved.';
  setStatus(`Stable: ${data.stable}\nReview the rendered final result.${confirmText}`);
}
async function confirmSave() {
  if (!current || !pendingResult) {
    setStatus('No settled result to save. Run Settle 5s first.');
    return;
  }
  setStatus('Saving confirmed result...');
  const data = await api('/api/confirm-save', {...current, result: pendingResult});
  setStatus(`Stable: ${data.stable}\nSaved: ${data.path}`);
  pendingResult = null;
  if (data.stable && current.source === 'holded') await loadList('holded');
}
document.addEventListener('keydown', async (ev) => {
  if (!current) return;
  const k = ev.key.toLowerCase();
  if (k === 'w') pose.x += tStep;
  else if (k === 's') pose.x -= tStep;
  else if (k === 'd') pose.y += tStep;
  else if (k === 'a') pose.y -= tStep;
  else if (k === 'z') pose.z += tStep;
  else if (k === 'x') pose.z -= tStep;
  else if (k === 'r') pose.r += rStep;
  else if (k === 'f') pose.r -= rStep;
  else if (k === 'p') pose.p += rStep;
  else if (k === 'o') pose.p -= rStep;
  else if (k === 'h') pose.h += rStep;
  else if (k === 'j') pose.h -= rStep;
  else if (ev.key === 'Enter') { await settleCurrent(); return; }
  else return;
  ev.preventDefault();
  pendingResult = null;
  poseHtml();
  await renderCurrent();
});
loadList('holded');
</script>
</body>
</html>
"""


def json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(store: SpawnReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/list":
                    source = parse_qs(parsed.query).get("source", ["holded"])[0]
                    json_response(self, HTTPStatus.OK, store.list_entries(source))
                    return
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except Exception as exc:
                json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/api/render":
                    json_response(self, HTTPStatus.OK, store.render(payload["source"], payload["path"], payload["object_type"], payload.get("pose")))
                    return
                if self.path == "/api/settle":
                    json_response(self, HTTPStatus.OK, store.settle(payload["source"], payload["path"], payload["object_type"], payload["pose"]))
                    return
                if self.path == "/api/confirm-save":
                    json_response(self, HTTPStatus.OK, store.confirm_save(payload["source"], payload["path"], payload["object_type"], payload["result"]))
                    return
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except Exception as exc:
                json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    args.possible_dir.mkdir(parents=True, exist_ok=True)
    args.holded_dir.mkdir(parents=True, exist_ok=True)
    store = SpawnReviewStore(
        args.possible_dir,
        args.holded_dir,
        args.camera_name,
        args.image_size,
        args.sim_seconds,
        args.sim_fps,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Open http://{args.host}:{args.port}")
    print(f"Possible dir: {args.possible_dir}")
    print(f"Holded dir: {args.holded_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
