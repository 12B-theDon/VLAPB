#!/usr/bin/env python3
"""Browser-based validator/editor for possible spawn positions.

The server edits JSON files under docs/possible_spawn_positions. Entries with
"N/A" are queued first so the user can assign an initial relative pose before
judging whether the placement is possible.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
import random
import time
from collections import Counter, defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("MUJOCO_GL", "egl")

VLAPB_LIBERO_ROOT = Path("/home/artemis/Documents/VLAPB/libero")
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
DEFAULT_SPAWN_DIR = VLAPB_LIBERO_ROOT / "docs" / "possible_spawn_positions"
LIBERO_DATA_ROOT = Path("/home/artemis/libero_data")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
FLOOR_SCENE = "FLOOR_SCENE"
POSE_KEYS = ("x", "y", "z", "r", "p", "h")
CANONICAL_LOCATION_ALIASES = {
    "cabinet.drawer.front_side": "cabinet.bottom_drawer.inside",
}
TOP_DRAWER_LOCATION = "cabinet.top_drawer.inside"
BOTTOM_DRAWER_LOCATION = "cabinet.bottom_drawer.inside"
ZERO_POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "p": 0.0, "h": 0.0}
DRAWER_OPEN_QPOS = -0.15
SIM_SECONDS = 5.0
SIM_FPS = 10
CONTROL_FREQ = 20
DISPLACEMENT_REVIEW_THRESHOLD = 0.08
HEIGHT_DROP_REVIEW_THRESHOLD = 0.05
FINAL_SPEED_REVIEW_THRESHOLD = 0.03
CLOSE_CHECK_STEPS = 40
CLOSE_SETTLE_STEPS = 20
ARTICULATION_CLOSED_TOLERANCE = 0.015
CLOSE_DISPLACEMENT_REVIEW_THRESHOLD = 0.04
CLOSE_HEIGHT_DROP_REVIEW_THRESHOLD = 0.03
SHELF_LAYER_MAX_OBJECT_HEIGHT = 0.18
SHELF_LAYER_MAX_CENTER_Z_ABOVE_FIXED = 0.35
REDROP_Z_OFFSET = 0.12
REDROP_MIN_Z = 0.12
FIXED_CENTER_LOCK_PREFIXES = ("basket.", "wooden_tray.", "plate.")
FIXED_CENTER_LOCK_LOCATIONS = {"flat_stove.surface"}
FIXED_LEFT_LOCK_PREFIXES = ("cabinet.",)
CABINET_LEFT_XY = (0.0, -0.30)
CABINET_LEFT_YAW = math.pi
OBJECT_CENTER_LOCK_LOCATIONS = {
    "basket.inside",
    "wooden_tray.inside",
    "plate.center",
}
SIDE_DEFAULT_OFFSET = 0.14
SAFE_OBJECT_INIT_XY = (0.24, -0.24)
SAFE_OBJECT_INIT_HALF_SIZE = 0.025
REFERENCE_FIXED_CLASSES = {
    "basket",
    "desk_caddy",
    "flat_stove",
    "floor",
    "kitchen_table",
    "living_room_table",
    "main_table",
    "microwave",
    "plate",
    "table",
    "white_cabinet",
    "wine_rack",
    "wooden_cabinet",
    "wooden_tray",
    "wooden_two_layer_shelf",
}
SUITE_BDDL_DIRS = (
    LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_10",
    LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_90",
    LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_object",
    LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_spatial",
)
LIBERO_DATA_DIRS = (
    LIBERO_DATA_ROOT / "libero_10",
    LIBERO_DATA_ROOT / "libero_90",
    LIBERO_DATA_ROOT / "libero_object",
    LIBERO_DATA_ROOT / "libero_spatial",
)
LOGGER = logging.getLogger("validate_profiles")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

try:
    import cv2
    import h5py
    import imageio
    import numpy as np
    from robosuite.utils.errors import RandomizationError
    from robosuite.utils.transform_utils import convert_quat, euler2mat, mat2quat
    from libero.envs import TASK_MAPPING
    import libero.utils.utils as libero_utils
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm is optional.
        tqdm = None
    from test_env_reconfiguration import (
        append_entries_to_section,
        build_env_kwargs,
        detect_workspace_name,
        extract_section_span,
        get_action_dim,
        parse_typed_section,
    )
except Exception as exc:  # pragma: no cover - surfaced in /api/render.
    RENDER_IMPORT_ERROR = exc
else:
    RENDER_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and edit possible spawn positions.")
    parser.add_argument("--spawn-dir", type=Path, default=DEFAULT_SPAWN_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--sim-seconds", type=float, default=SIM_SECONDS)
    parser.add_argument("--sim-fps", type=int, default=SIM_FPS)
    parser.add_argument("--verbose", action="store_true", help="Print detailed server/debug logs.")
    return parser.parse_args()

# ===

def stable_seed(scene: str, location: str, object_type: str) -> int:
    key = f"{scene}|{location}|{object_type}"
    return abs(hash(key)) % (2**31)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def debug_log(action: str, payload: dict[str, object], exc: Exception | None = None) -> None:
    parts = [
        f"[validate_profiles:{action}]",
        f"scene={payload.get('scene')}",
        f"location={payload.get('location')}",
        f"object={payload.get('object')}",
        f"path={payload.get('path')}",
        f"pose={payload.get('pose') or payload.get('anchor')}",
    ]
    if exc is not None:
        parts.append(f"error={type(exc).__name__}: {exc}")
    LOGGER.warning(" ".join(parts))


def log_object_state(env, label: str, object_instance: str, fixed_instance: str, location: str, pose: dict[str, float]) -> None:
    print(f"\n===== DEBUG {label} =====")
    print("object_instance:", object_instance)
    print("fixed_instance:", fixed_instance)
    print("location:", location)
    print("pose:", pose)

    print("object joints:")
    for j in env.sim.model.joint_names:
        if object_instance in j:
            try:
                print(" ", j, "qpos =", env.sim.data.get_joint_qpos(j))
            except Exception as exc:
                print(" ", j, "qpos read failed:", exc)

    print("object bodies:")
    for b in env.sim.model.body_names:
        if object_instance in b:
            bid = env.sim.model.body_name2id(b)
            print(" ", b, "xpos =", env.sim.data.body_xpos[bid])

    ref_body = resolve_reference_body(env, fixed_instance, location)
    ref_id = env.sim.model.body_name2id(ref_body)
    print("reference body:", ref_body)
    print("reference xpos:", env.sim.data.body_xpos[ref_id])

    target_world = env.sim.data.body_xpos[ref_id] + np.array([pose["x"], pose["y"], pose["z"]])
    print("target world pos:", target_world)

    print("camera:", env.sim.model.camera_id2name(env.sim.model.camera_name2id("agentview")) if "agentview" in env.sim.model.camera_names else "no agentview")
    print("========================\n")



# ==

def spawn_json_text(items: dict[str, object]) -> str:
    lines = ["{"]
    names = list(items)
    for idx, name in enumerate(names):
        value = items[name]
        encoded_value = json.dumps(value, sort_keys=True, separators=(", ", ": "))
        comma = "," if idx + 1 < len(names) else ""
        lines.append(f"  {json.dumps(name)}: {encoded_value}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(spawn_json_text(data))


def normalize_pose(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return {key: round(float(value.get(key, 0.0)), 4) for key in POSE_KEYS}
    return dict(ZERO_POSE)


def is_zero_pose(pose: dict[str, float]) -> bool:
    return all(abs(float(pose[key])) < 1e-9 for key in POSE_KEYS)


def fixed_center_locked(location: str) -> bool:
    return location in FIXED_CENTER_LOCK_LOCATIONS or location.startswith(FIXED_CENTER_LOCK_PREFIXES)


def fixed_relocation_xy(location: str) -> tuple[float, float] | None:
    if location.startswith(FIXED_LEFT_LOCK_PREFIXES):
        return CABINET_LEFT_XY
    if fixed_center_locked(location):
        return (0.0, 0.0)
    return None


def fixed_relocation_yaw(location: str) -> float:
    if location.startswith(FIXED_LEFT_LOCK_PREFIXES):
        return CABINET_LEFT_YAW
    return 0.0


def object_center_locked(location: str) -> bool:
    return location in OBJECT_CENTER_LOCK_LOCATIONS


def default_pose_for_location(location: str) -> dict[str, float]:
    location = canonical_location(location)
    pose = dict(ZERO_POSE)
    if location.endswith(".right_side"):
        pose["y"] = SIDE_DEFAULT_OFFSET
    elif location.endswith(".left_side"):
        pose["y"] = -SIDE_DEFAULT_OFFSET
    elif location.endswith(".front_side"):
        pose["x"] = -SIDE_DEFAULT_OFFSET
    elif location.endswith(".back_side"):
        pose["x"] = SIDE_DEFAULT_OFFSET
    return pose


def canonical_location(location: str) -> str:
    return CANONICAL_LOCATION_ALIASES.get(location, location)


def drawer_xy_source_location(location: str) -> str | None:
    if canonical_location(location) == TOP_DRAWER_LOCATION:
        return BOTTOM_DRAWER_LOCATION
    return None


def copy_xy_from_pose(pose: dict[str, float], source_pose: dict[str, float] | None) -> dict[str, float]:
    if source_pose is None:
        return pose
    copied = dict(pose)
    copied["x"] = source_pose["x"]
    copied["y"] = source_pose["y"]
    return copied


def effective_pose_for_location(location: str, value: object) -> dict[str, float]:
    location = canonical_location(location)
    pose = normalize_pose(value)
    if is_zero_pose(pose):
        return default_pose_for_location(location)
    return pose


def is_pose(value: object) -> bool:
    return isinstance(value, dict) and all(key in value for key in POSE_KEYS)


def anchor_pose(data: dict[str, object]) -> dict[str, float] | None:
    value = data.get("_anchor")
    return normalize_pose(value) if is_pose(value) else None


def object_items(data: dict[str, object]) -> list[str]:
    return sorted(key for key in data if not key.startswith("_"))


def ensure_anchor(data: dict[str, object], pose: dict[str, float]) -> dict[str, float]:
    normalized = normalize_pose(pose)
    data["_anchor"] = normalized
    return normalized


def scene_bddl_candidates(scene: str) -> list[Path]:
    candidates: list[Path] = []
    if scene == FLOOR_SCENE:
        for bddl_dir in SUITE_BDDL_DIRS:
            candidates.extend(path for path in sorted(bddl_dir.glob("*.bddl")) if "Floor_Manipulation" in path.read_text(encoding="utf-8", errors="ignore"))
        return candidates
    for bddl_dir in SUITE_BDDL_DIRS:
        candidates.extend(sorted(bddl_dir.glob(f"{scene}_*.bddl")))
    return candidates


def typed_entries_from_bddl(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    text = path.read_text(encoding="utf-8")
    fixtures = parse_typed_section(extract_section_span(text, "fixtures")[2])
    objects = parse_typed_section(extract_section_span(text, "objects")[2])
    return fixtures, objects, text


def fixed_type_candidates(location: str) -> set[str]:
    fixed = location.split(".", 1)[0]
    if fixed == "cabinet":
        return {"white_cabinet", "wooden_cabinet", "cabinet"}
    if fixed == "wooden_two_layer_shelf":
        return {"wooden_two_layer_shelf"}
    return {fixed}


def find_fixed_instance(entries: list[tuple[str, str]], location: str) -> str | None:
    candidates = fixed_type_candidates(location)
    for name, entity_type in entries:
        if entity_type in candidates:
            return name
    return None


def resolve_scene_bddl(scene: str, location: str) -> tuple[Path, str, str]:
    for path in scene_bddl_candidates(scene):
        fixtures, objects, _text = typed_entries_from_bddl(path)
        fixed_instance = find_fixed_instance(fixtures + objects, location)
        if fixed_instance:
            workspace = detect_workspace_name(fixtures)
            return path, fixed_instance, workspace
    raise FileNotFoundError(f"No BDDL found for {scene} with fixed item for {location}")


def safe_instance_name(object_type: str) -> str:
    return f"vlapb_{re.sub(r'[^A-Za-z0-9_]+', '_', object_type)}_1"


def region_block(
    name: str,
    target: str,
    xy: tuple[float, float] = (0.0, 0.0),
    half_size: float = 0.01,
    yaw: tuple[float, float] | None = None,
) -> str:
    x, y = xy
    yaw_block = ""
    if yaw is not None:
        yaw_block = (
            "          (:yaw_rotation (\n"
            f"              ({yaw[0]:.6f} {yaw[1]:.6f})\n"
            "            )\n"
            "          )\n"
        )
    return (
        "\n"
        f"      ({name}\n"
        f"          (:target {target})\n"
        "          (:ranges (\n"
        f"              ({x - half_size:.6f} {y - half_size:.6f} {x + half_size:.6f} {y + half_size:.6f})\n"
        "            )\n"
        "          )\n"
        f"{yaw_block}"
        "      )"
    )


def is_drawer_location(location: str) -> bool:
    return "drawer" in location


def is_microwave_inside(location: str) -> bool:
    return location == "microwave.inside"


def is_shelf_layer_location(location: str) -> bool:
    return location in {
        "wooden_two_layer_shelf.top_shelf",
        "wooden_two_layer_shelf.bottom_shelf",
    }


def requires_close_check(location: str) -> bool:
    return is_drawer_location(location) or is_microwave_inside(location)


def drawer_open_regions(location: str, fixed_instance: str) -> list[str]:
    if not is_drawer_location(location):
        return []
    if "bottom_drawer" in location:
        return [f"{fixed_instance}_bottom_region"]
    if "middle_drawer" in location:
        return [f"{fixed_instance}_middle_region"]
    if "top_drawer" in location:
        return [f"{fixed_instance}_top_region"]
    return []


def is_drawer_state_line(raw_line: str, fixed_instance: str) -> bool:
    pattern = rf"\((Open|Close)\s+{re.escape(fixed_instance)}_(top|bottom|middle)_region\)"
    return re.search(pattern, raw_line) is not None


def is_articulated_state_line(raw_line: str, fixed_instance: str) -> bool:
    if is_drawer_state_line(raw_line, fixed_instance):
        return True
    pattern = rf"\((Open|Close)\s+{re.escape(fixed_instance)}\)"
    return re.search(pattern, raw_line) is not None


def is_fixed_placement_line(raw_line: str, fixed_instance: str) -> bool:
    pattern = rf"\((On|In)\s+{re.escape(fixed_instance)}\s+"
    return re.search(pattern, raw_line) is not None


def filter_init_for_fixed(
    init_section: str,
    fixed_instance: str | None,
    location: str,
    recenter_fixed: bool = False,
) -> str:
    lines = [init_section.splitlines()[0]]
    if fixed_instance:
        for raw_line in init_section.splitlines()[1:-1]:
            tokens = set(re.findall(r"[A-Za-z0-9_]+", raw_line))
            if fixed_instance in tokens:
                if is_articulated_state_line(raw_line, fixed_instance):
                    continue
                if recenter_fixed and is_fixed_placement_line(raw_line, fixed_instance):
                    continue
                lines.append(raw_line)
    lines.append("  )")
    return "\n".join(lines)


def region_center(regions_section: str, region_name: str) -> tuple[float, float] | None:
    pattern = re.compile(
        rf"\({re.escape(region_name)}\s+.*?:ranges\s+\(\s*"
        rf"\(\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*\)",
        re.DOTALL,
    )
    match = pattern.search(regions_section)
    if not match:
        return None
    x1, y1, x2, y2 = (float(value) for value in match.groups())
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def fixed_anchor_from_bddl(
    regions_section: str,
    init_section: str,
    fixed_instance: str,
    fallback_target: str,
) -> tuple[str, tuple[float, float]]:
    init_match = re.search(rf"\(On\s+{re.escape(fixed_instance)}\s+([A-Za-z0-9_]+)\)", init_section)
    if not init_match:
        return fallback_target, (0.0, 0.0)

    target_region = init_match.group(1)
    prefix = f"{fallback_target}_"
    if target_region.startswith(prefix):
        target = fallback_target
        region_name = target_region[len(prefix):]
    else:
        target, _, region_name = target_region.partition("_")
    center = region_center(regions_section, region_name)
    if center is None:
        return fallback_target, (0.0, 0.0)
    return target, center


def effective_fixed_anchor_from_bddl(
    regions_section: str,
    init_section: str,
    fixed_instance: str,
    workspace: str,
    location: str,
) -> tuple[str, tuple[float, float], bool]:
    """Return the anchor actually used by our generated validation BDDL.

    Some fixed items are intentionally relocated for validation so they are
    reachable and visible. Any pose saved by the web UI must be relative to
    this effective anchor, not the original LIBERO BDDL anchor.
    """
    anchor_target, anchor_xy = fixed_anchor_from_bddl(regions_section, init_section, fixed_instance, workspace)
    fixed_xy = fixed_relocation_xy(location)
    recenter_fixed = fixed_xy is not None
    if recenter_fixed:
        anchor_xy = fixed_xy
    return anchor_target, anchor_xy, recenter_fixed


def placement_site_for_location(location: str) -> tuple[str, str] | None:
    if location == "basket.inside":
        return "In", "contain_region"
    if location == "wooden_tray.inside":
        return "In", "contain_region"
    if location == "plate.center":
        return "On", ""
    if location == "flat_stove.surface":
        return "On", "cook_region"
    if location == "microwave.inside":
        return "In", "heating_region"
    if location == "microwave.top_surface":
        return "On", "top_side"
    if location == "cabinet.top_surface":
        return "On", "top_side"
    if location.startswith("cabinet.top_drawer."):
        return "In", "top_region"
    if location.startswith("cabinet.middle_drawer."):
        return "In", "middle_region"
    if location.startswith("cabinet.bottom_drawer."):
        return "In", "bottom_region"
    if location == "wooden_two_layer_shelf.top_shelf":
        return "In", "top_region"
    if location == "wooden_two_layer_shelf.top_surface":
        return "On", "top_side"
    if location == "wooden_two_layer_shelf.bottom_shelf":
        return "In", "bottom_region"
    return None


def needs_manual_site_spawn(location: str) -> bool:
    return (
        "drawer" in location
        or location in {
            "microwave.inside",
            "wooden_two_layer_shelf.top_shelf",
            "wooden_two_layer_shelf.bottom_shelf",
        }
    )


def location_demo_keywords(location: str) -> tuple[str, ...]:
    if location.startswith("cabinet.top_drawer."):
        return ("in_the_top_drawer",)
    if location.startswith("cabinet.middle_drawer."):
        return ("middle_drawer", "middle_layer_of_the_drawer")
    if location.startswith("cabinet.bottom_drawer."):
        return ("in_the_bottom_drawer",)
    mapping = {
        "basket.inside": ("in_the_basket", "put_it_in_the_basket", "place_it_in_the_basket"),
        "wooden_tray.inside": ("in_the_tray", "put_it_in_the_tray", "place_it_in_the_tray"),
        "plate.center": ("on_the_plate",),
        "flat_stove.surface": ("on_the_stove", "on_it"),
        "cabinet.top_surface": ("on_top_of_the_cabinet", "on_top_of_it"),
        "microwave.inside": ("in_the_microwave",),
        "microwave.top_surface": ("on_the_microwave", "on_top_of_the_microwave"),
        "wooden_two_layer_shelf.top_shelf": ("on_the_cabinet_shelf",),
        "wooden_two_layer_shelf.top_surface": ("on_top_of_the_shelf", "on_top_of_the_cabinet"),
        "wooden_two_layer_shelf.bottom_shelf": ("under_the_cabinet_shelf",),
        "floor.center": ("place_it_in_the_basket", "put_it_in_the_basket"),
    }
    return mapping.get(location, ())


@lru_cache(maxsize=None)
def reference_demo_candidates(scene: str, location: str) -> tuple[Path, ...]:
    keywords = location_demo_keywords(location)
    if not keywords:
        return ()
    exact_candidates = []
    fallback_candidates = []
    for data_dir in LIBERO_DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("*_demo.hdf5")):
            name = path.name.lower()
            if re.search(r"^[a-z_]+_scene\d+_(open|close)_", name):
                continue
            if any(keyword in name for keyword in keywords):
                if name.startswith(f"{scene.lower()}_"):
                    exact_candidates.append(path)
                else:
                    fallback_candidates.append(path)
    candidates = exact_candidates or fallback_candidates
    return tuple(sorted(
        candidates,
        key=lambda path: min(
            (idx for idx, keyword in enumerate(keywords) if keyword in path.name.lower()),
            default=len(keywords),
        ),
    ))


def scene_from_demo_path(path: Path) -> str | None:
    match = re.match(r"([A-Z]+(?:_[A-Z]+)*_SCENE\d+)_", path.name)
    return match.group(1) if match else None


def bddl_path_from_demo(hdf5_file) -> Path:
    raw = hdf5_file["data"].attrs.get("bddl_file_name", "")
    value = raw.decode() if isinstance(raw, bytes) else str(raw)
    path = Path(value)
    if path.is_absolute():
        return path
    return LIBERO_ROOT / path


def first_object_of_interest(bddl_text: str) -> str:
    _obj_start, _obj_end, objects_section = extract_section_span(bddl_text, "objects")
    object_types = dict(parse_typed_section(objects_section))
    _start, _end, section = extract_section_span(bddl_text, "obj_of_interest")
    for line in section.splitlines()[1:-1]:
        token = line.strip()
        if token and object_types.get(token) not in REFERENCE_FIXED_CLASSES:
            return token
    raise RuntimeError("No obj_of_interest found in reference BDDL")


def object_type_from_bddl(bddl_text: str, object_instance: str) -> str:
    _obj_start, _obj_end, objects_section = extract_section_span(bddl_text, "objects")
    object_types = dict(parse_typed_section(objects_section))
    object_type = object_types.get(object_instance)
    if not object_type:
        raise RuntimeError(f"No object type found for {object_instance}")
    return object_type


def localize_demo_model_xml(model_xml: str) -> str:
    libero_asset_root = LIBERO_ROOT / "libero" / "libero"
    replacements = (
        ("/home/yifengz/workspace/libero-dev/chiliocosm", str(libero_asset_root)),
        ("/home/yifengz/workspace/LIBERO/libero/libero", str(libero_asset_root)),
        ("/home/yifengz/workspace/LIBERO/libero", str(LIBERO_ROOT / "libero")),
    )
    for old, new in replacements:
        model_xml = model_xml.replace(old, new)
    return libero_utils.postprocess_model_xml(model_xml, {})


def yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def demo_env_kwargs(hdf5_file, bddl_path: Path, camera_name: str, image_size: int) -> tuple[str, dict[str, object]]:
    env_args = json.loads(hdf5_file["data"].attrs["env_args"])
    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs.update(
        {
            "bddl_file_name": str(bddl_path),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "camera_names": [camera_name],
            "camera_heights": image_size,
            "camera_widths": image_size,
        }
    )
    return env_args["problem_name"], env_kwargs


def extract_reference_pose_from_demo(
    demo_path: Path,
    scene: str,
    location: str,
    camera_name: str,
    image_size: int,
) -> dict[str, object]:
    with h5py.File(demo_path, "r") as f:
        bddl_path = bddl_path_from_demo(f)
        bddl_text = bddl_path.read_text(encoding="utf-8")
        object_instance = first_object_of_interest(bddl_text)
        object_type = object_type_from_bddl(bddl_text, object_instance)
        problem_name, env_kwargs = demo_env_kwargs(f, bddl_path, camera_name, image_size)
        demo_name = sorted(f["data"].keys())[0]
        model_xml = localize_demo_model_xml(f["data"][demo_name].attrs["model_file"])
        states = f["data"][demo_name]["states"][()]

        env = TASK_MAPPING[problem_name](**env_kwargs)
        try:
            env.reset()
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[-1])
            env.sim.forward()
            body_name = body_name_for_instance(env, object_instance)
            body_id = env.sim.model.body_name2id(body_name)
            final_pos = np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)
            final_quat = np.array(env.sim.data.body_xquat[body_id], dtype=np.float64)
        finally:
            env.close()

    reference_scene = scene_from_demo_path(demo_path) or scene
    original_bddl, fixed_instance, workspace = resolve_scene_bddl(reference_scene, location)
    text = original_bddl.read_text(encoding="utf-8")
    regions_section = extract_section_span(text, "regions")[2]
    init_section = extract_section_span(text, "init")[2]
    _anchor_target, anchor_xy = fixed_anchor_from_bddl(regions_section, init_section, fixed_instance, workspace)
    pose = {
        "x": round(float(final_pos[0] - anchor_xy[0]), 4),
        "y": round(float(final_pos[1] - anchor_xy[1]), 4),
        "z": 0.0,
        "r": 0.0,
        "p": 0.0,
        "h": round(float(yaw_from_wxyz(final_quat)), 4),
    }
    return {
        "pose": normalize_pose(pose),
        "demo_path": str(demo_path),
        "demo_object_instance": object_instance,
        "demo_object_type": object_type,
        "demo_bddl": str(bddl_path),
        "reference_scene": reference_scene,
        "target_scene": scene,
        "cross_scene_reference": reference_scene != scene,
        "final_world_position": {
            "x": round(float(final_pos[0]), 4),
            "y": round(float(final_pos[1]), 4),
            "z": round(float(final_pos[2]), 4),
        },
    }


def reference_anchor_for(scene: str, location: str, camera_name: str, image_size: int) -> dict[str, object] | None:
    errors = []
    for demo_path in reference_demo_candidates(scene, location):
        try:
            return extract_reference_pose_from_demo(demo_path, scene, location, camera_name, image_size)
        except Exception as exc:
            errors.append(f"{demo_path.name}: {exc}")
    if errors:
        return {"error": "; ".join(errors[:3]), "pose": None}
    return None


def reference_body_z_for(scene: str, location: str, camera_name: str, image_size: int) -> float | None:
    reference = reference_anchor_for(scene, location, camera_name, image_size)
    if not reference:
        return None
    final_world = reference.get("final_world_position")
    if not isinstance(final_world, dict) or "z" not in final_world:
        return None
    return float(final_world["z"])


def build_render_bddl(
    original_bddl: Path,
    object_type: str,
    workspace: str,
    fixed_instance: str,
    location: str,
    pose: dict[str, float],
    safe_object_init: bool = False,
) -> tuple[str, str, tuple[float, float], bool]:
    text = original_bddl.read_text(encoding="utf-8")
    instance = safe_instance_name(object_type)
    region = f"{instance}_init_region"

    _fixtures_start, _fixtures_end, fixtures_section = extract_section_span(text, "fixtures")
    objects_start, objects_end, objects_section = extract_section_span(text, "objects")
    regions_start, regions_end, regions_section = extract_section_span(text, "regions")
    init_start, init_end, init_section = extract_section_span(text, "init")
    interest_start, interest_end, _interest_section = extract_section_span(text, "obj_of_interest")
    goal_start, goal_end, _goal_section = extract_section_span(text, "goal")
    original_fixtures = parse_typed_section(fixtures_section)
    original_objects = parse_typed_section(objects_section)
    original_fixed_names = {name for name, _entity_type in original_fixtures + original_objects}
    original_object_names = {name for name, _entity_type in original_objects}
    fixed_xy = fixed_relocation_xy(location)

    kept_object_lines = []
    for name, entity_type in original_objects:
        if name == fixed_instance:
            kept_object_lines.append(f"    {name} - {entity_type}")
    kept_object_lines.append(f"    {instance} - {object_type}")
    new_objects_section = "(:objects\n" + "\n".join(kept_object_lines) + "\n  )"
    anchor_target, anchor_xy, should_recenter_fixed = effective_fixed_anchor_from_bddl(
        regions_section,
        init_section,
        fixed_instance,
        workspace,
        location,
    )
    recenter_fixed = should_recenter_fixed and fixed_instance in original_fixed_names
    placement_site = placement_site_for_location(location)
    site_based = placement_site is not None
    use_site_init = site_based and not safe_object_init
    region_entry = ""
    if not use_site_init:
        region_xy = (anchor_xy[0] + pose["x"], anchor_xy[1] + pose["y"])
        region_half_size = 0.005
        if safe_object_init:
            region_xy = SAFE_OBJECT_INIT_XY
            region_half_size = SAFE_OBJECT_INIT_HALF_SIZE
        region_entry = region_block(
            region,
            anchor_target,
            xy=region_xy,
            half_size=region_half_size,
            yaw=(pose["h"], pose["h"]),
        )
    fixed_center_region = f"{fixed_instance}_center_region"
    if recenter_fixed:
        fixed_region_entry = region_block(
            fixed_center_region,
            anchor_target,
            xy=anchor_xy,
            half_size=0.005,
            yaw=(fixed_relocation_yaw(location), fixed_relocation_yaw(location)),
        )
        region_entry = fixed_region_entry + region_entry

    filtered_init = filter_init_for_fixed(init_section, fixed_instance, location, recenter_fixed)
    if not use_site_init:
        init_lines = [f"\n    (On {instance} {anchor_target}_{region})"]
        goal_predicate = f"(On {instance} {anchor_target}_{region})"
    else:
        relation, site_region = placement_site
        placement_target = fixed_instance if not site_region else f"{fixed_instance}_{site_region}"
        init_lines = [f"\n    ({relation} {instance} {placement_target})"]
        goal_predicate = f"({relation} {instance} {placement_target})"
    if recenter_fixed:
        init_lines.insert(0, f"\n    (On {fixed_instance} {anchor_target}_{fixed_center_region})")

    for open_region in drawer_open_regions(location, fixed_instance):
        open_line = f"\n    (Open {open_region})"
        if open_line.strip() not in filtered_init:
            init_lines.append(open_line)
    if is_microwave_inside(location):
        open_line = f"\n    (Open {fixed_instance})"
        if open_line.strip() not in filtered_init:
            init_lines.append(open_line)
    new_interest_section = f"(:obj_of_interest\n    {instance}\n  )"
    new_goal_section = f"(:goal\n    (And {goal_predicate})\n  )"

    new_regions_section = regions_section
    if region_entry:
        new_regions_section = append_entries_to_section(regions_section, region_entry, "    ")

    replacements = [
        (objects_start, objects_end, new_objects_section),
        (regions_start, regions_end, new_regions_section),
        (init_start, init_end, append_entries_to_section(filtered_init, "".join(init_lines), "  ")),
        (interest_start, interest_end, new_interest_section),
        (goal_start, goal_end, new_goal_section),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, instance, anchor_xy, site_based or safe_object_init


def reset_with_retries(env, max_attempts: int = 5):
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return env.reset()
        except RandomizationError as exc:
            last_error = exc
    raise RuntimeError("Could not reset MuJoCo env") from last_error

def resolve_reference_body(env, fixed_instance: str, location: str) -> str:
    body_names = list(env.sim.model.body_names)

    if f"{fixed_instance}_main" in body_names:
        return f"{fixed_instance}_main"
    if fixed_instance in body_names:
        return fixed_instance

    raise RuntimeError(f"Cannot find reference body for {fixed_instance}")


def set_relative_pose(env, object_instance: str, fixed_instance: str, location: str, pose: dict[str, float]) -> None:
    fixed_body_name = resolve_reference_body(env, fixed_instance, location)

    fixed_body_id = env.sim.model.body_name2id(fixed_body_name)
    fixed_pos = np.array(env.sim.data.body_xpos[fixed_body_id], dtype=np.float64)

    world_pos = fixed_pos + np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    quat = convert_quat(mat2quat(euler2mat([pose["r"], pose["p"], pose["h"]])), to="wxyz")

    candidates = [f"{object_instance}_joint", f"{object_instance}_joint0"]
    joint_name = next((j for j in candidates if j in env.sim.model.joint_names), None)
    if joint_name is None:
        raise RuntimeError(f"Could not find free joint for {object_instance}")

    print("\n===== DEBUG set_relative_pose =====")
    print("fixed_body_name:", fixed_body_name)
    print("fixed_pos:", fixed_pos)
    print("pose offset:", pose)
    print("computed world_pos:", world_pos)
    print("joint_name:", joint_name)
    print("qpos before:", env.sim.data.get_joint_qpos(joint_name))

    env.sim.data.set_joint_qpos(joint_name, np.concatenate([world_pos, quat]))
    env.sim.forward()

    print("qpos after:", env.sim.data.get_joint_qpos(joint_name))
    obj_body = f"{object_instance}_main"
    if obj_body in env.sim.model.body_names:
        bid = env.sim.model.body_name2id(obj_body)
        print("object body xpos after:", env.sim.data.body_xpos[bid])
    print("==================================\n")


def object_joint_name(env, object_instance: str) -> str:
    candidates = [f"{object_instance}_joint", f"{object_instance}_joint0"]
    joint_name = next((j for j in candidates if j in env.sim.model.joint_names), None)
    if joint_name is None:
        raise RuntimeError(f"Could not find free joint for {object_instance}")
    return joint_name


def apply_sampled_pose_adjustments(
    env,
    object_instance: str,
    fixed_instance: str,
    location: str,
    pose: dict[str, float],
    anchor_xy: tuple[float, float] | None = None,
    apply_xy: bool = False,
    target_body_z: float | None = None,
) -> None:
    center_locked = object_center_locked(location)
    has_xy_offset = center_locked or (
        apply_xy and anchor_xy is not None and (abs(pose["x"]) > 1e-9 or abs(pose["y"]) > 1e-9)
    )
    has_z_offset = target_body_z is not None or abs(pose["z"]) > 1e-9
    has_roll_pitch = abs(pose["r"]) > 1e-9 or abs(pose["p"]) > 1e-9
    if not has_xy_offset and not has_z_offset and not has_roll_pitch:
        return

    joint_name = object_joint_name(env, object_instance)
    qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64)
    if center_locked:
        fixed_body = resolve_reference_body(env, fixed_instance, location)
        fixed_body_id = env.sim.model.body_name2id(fixed_body)
        fixed_pos = np.array(env.sim.data.body_xpos[fixed_body_id], dtype=np.float64)
        qpos[0] = fixed_pos[0]
        qpos[1] = fixed_pos[1]
    elif has_xy_offset:
        qpos[0] = anchor_xy[0] + pose["x"]
        qpos[1] = anchor_xy[1] + pose["y"]
    if target_body_z is not None:
        body_z = object_position(env, object_instance)[2]
        qpos[2] += float(target_body_z) - float(body_z)
    elif has_z_offset:
        qpos[2] += pose["z"]
    if has_roll_pitch:
        qpos[3:7] = convert_quat(mat2quat(euler2mat([pose["r"], pose["p"], pose["h"]])), to="wxyz")
    env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.forward()


def drawer_joint_candidates(fixed_object, location: str) -> list[str]:
    joints = list(getattr(fixed_object, "joints", []))

    if "top_drawer" in location:
        keys = ("top", "upper")
    elif "middle_drawer" in location:
        keys = ("middle", "center")
    elif "bottom_drawer" in location:
        keys = ("bottom", "lower")
    else:
        return []

    matched = [
        joint for joint in joints
        if any(key in joint.lower() for key in keys)
    ]

    # joint 이름에 top/bottom이 안 들어간 경우를 위한 fallback
    if matched:
        return matched
    if "top_drawer" in location and len(joints) >= 1:
        return [joints[0]]
    if "middle_drawer" in location and len(joints) >= 2:
        return [joints[len(joints) // 2]]
    if "bottom_drawer" in location and len(joints) >= 2:
        return [joints[-1]]

    return joints


def open_drawer_if_needed(env, fixed_instance: str, location: str) -> None:
    if not is_drawer_location(location):
        return
    try:
        fixed_object = env.get_object(fixed_instance)
    except Exception:
        return

    open_ranges = fixed_object.object_properties.get("articulation", {}).get("default_open_ranges", [])
    if open_ranges:
        first_range = open_ranges[0]
        if isinstance(first_range, (int, float)):
            qpos = float(sum(open_ranges) / len(open_ranges))
        else:
            qpos = float(sum(first_range) / len(first_range))
    else:
        qpos = DRAWER_OPEN_QPOS

    for joint_name in drawer_joint_candidates(fixed_object, location):
        if joint_name in env.sim.model.joint_names:
            env.sim.data.set_joint_qpos(joint_name, qpos)

    env.sim.forward()


def open_microwave_if_needed(env, fixed_instance: str, location: str) -> None:
    if not is_microwave_inside(location):
        return
    try:
        fixed_object = env.get_object(fixed_instance)
    except Exception:
        return
    open_ranges = fixed_object.object_properties.get("articulation", {}).get("default_open_ranges", [])
    if open_ranges:
        qpos = float(sum(open_ranges) / len(open_ranges))
    else:
        qpos = -1.7
    for joint_name in getattr(fixed_object, "joints", []):
        if joint_name in env.sim.model.joint_names:
            env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.forward()


def open_articulated_if_needed(env, fixed_instance: str, location: str) -> None:
    open_drawer_if_needed(env, fixed_instance, location)
    open_microwave_if_needed(env, fixed_instance, location)


def selected_articulation_joints(env, fixed_instance: str, location: str) -> list[str]:
    try:
        fixed_object = env.get_object(fixed_instance)
    except Exception:
        return []
    if is_drawer_location(location):
        return [joint for joint in drawer_joint_candidates(fixed_object, location) if joint in env.sim.model.joint_names]
    if is_microwave_inside(location):
        return [joint for joint in getattr(fixed_object, "joints", []) if joint in env.sim.model.joint_names]
    return []


def joint_closed_qpos(env, joint_name: str) -> float:
    joint_id = env.sim.model.joint_name2id(joint_name)
    limited = bool(env.sim.model.jnt_limited[joint_id])
    if limited:
        low, high = (float(value) for value in env.sim.model.jnt_range[joint_id])
        return min(max(0.0, low), high)
    return 0.0


def joint_qpos(env, joint_name: str) -> float:
    value = env.sim.data.get_joint_qpos(joint_name)
    try:
        return float(value)
    except TypeError:
        return float(np.asarray(value).reshape(-1)[0])


def set_joint_scalar_qpos(env, joint_name: str, value: float) -> None:
    current = env.sim.data.get_joint_qpos(joint_name)
    if np.isscalar(current):
        env.sim.data.set_joint_qpos(joint_name, float(value))
        return
    qpos = np.asarray(current, dtype=np.float64).copy()
    qpos.reshape(-1)[0] = float(value)
    env.sim.data.set_joint_qpos(joint_name, qpos)


def object_body_ids(env, object_instance: str) -> set[int]:
    body_ids = set()
    for body_id, body_name in enumerate(env.sim.model.body_names):
        if body_name and str(body_name).startswith(object_instance):
            body_ids.add(body_id)
    return body_ids


def geom_radius(env, geom_id: int) -> np.ndarray:
    # Conservative axis-aligned radius. MuJoCo geom_size has type-dependent
    # semantics, so use the largest declared extent for each axis.
    radius = float(np.max(env.sim.model.geom_size[geom_id]))
    return np.array([radius, radius, radius], dtype=np.float64)


def geom_bounding_box(env, object_instance: str) -> dict[str, object] | None:
    mins = []
    maxs = []
    body_ids = object_body_ids(env, object_instance)
    for geom_id, geom_name in enumerate(env.sim.model.geom_names):
        geom_body_id = int(env.sim.model.geom_bodyid[geom_id])
        name_matches = bool(geom_name and str(geom_name).startswith(object_instance))
        if body_ids:
            if geom_body_id not in body_ids:
                continue
        elif not name_matches:
            continue
        center = np.array(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
        radius = geom_radius(env, geom_id)
        mins.append(center - radius)
        maxs.append(center + radius)
    if not mins:
        return None
    min_xyz = np.min(np.stack(mins), axis=0)
    max_xyz = np.max(np.stack(maxs), axis=0)
    size = max_xyz - min_xyz
    return {
        "min": [round(float(value), 5) for value in min_xyz],
        "max": [round(float(value), 5) for value in max_xyz],
        "size": [round(float(value), 5) for value in size],
        "height": round(float(size[2]), 5),
    }


def close_articulation_check(env, fixed_instance: str, object_instance: str, location: str, action: np.ndarray) -> dict[str, object] | None:
    if not requires_close_check(location):
        return None
    joints = selected_articulation_joints(env, fixed_instance, location)
    if not joints:
        return {"status": "blocked", "reason": "missing_articulation_joint", "joints": []}

    start_pos = object_position(env, object_instance)
    start_z = float(start_pos[2])
    start_qpos = {joint: joint_qpos(env, joint) for joint in joints}
    target_qpos = {joint: joint_closed_qpos(env, joint) for joint in joints}

    for step in range(1, CLOSE_CHECK_STEPS + 1):
        alpha = step / CLOSE_CHECK_STEPS
        for joint in joints:
            qpos = start_qpos[joint] + (target_qpos[joint] - start_qpos[joint]) * alpha
            set_joint_scalar_qpos(env, joint, qpos)
        env.sim.forward()
        env.step(action)

    for _ in range(CLOSE_SETTLE_STEPS):
        env.step(action)

    final_pos = object_position(env, object_instance)
    final_speed = object_speed(env, object_instance)
    displacement = float(np.linalg.norm(final_pos[:2] - start_pos[:2]))
    height_drop = float(max(0.0, start_z - final_pos[2]))
    final_qpos = {joint: joint_qpos(env, joint) for joint in joints}
    joint_errors = {joint: abs(final_qpos[joint] - target_qpos[joint]) for joint in joints}
    closed = all(error <= ARTICULATION_CLOSED_TOLERANCE for error in joint_errors.values())
    stable = final_speed <= FINAL_SPEED_REVIEW_THRESHOLD
    stayed_put = displacement <= CLOSE_DISPLACEMENT_REVIEW_THRESHOLD
    no_drop = height_drop <= CLOSE_HEIGHT_DROP_REVIEW_THRESHOLD
    ok = closed and stable and stayed_put and no_drop

    reasons = []
    if not closed:
        reasons.append("not_closed")
    if not stable:
        reasons.append("object_unsettled_after_close")
    if not stayed_put:
        reasons.append("object_moved_during_close")
    if not no_drop:
        reasons.append("object_dropped_during_close")

    return {
        "status": "closed" if ok else "blocked",
        "reason": "closed_stable" if ok else "+".join(reasons),
        "joints": joints,
        "start_qpos": {joint: round(float(value), 5) for joint, value in start_qpos.items()},
        "target_qpos": {joint: round(float(value), 5) for joint, value in target_qpos.items()},
        "final_qpos": {joint: round(float(value), 5) for joint, value in final_qpos.items()},
        "joint_errors": {joint: round(float(value), 5) for joint, value in joint_errors.items()},
        "closed_tolerance": ARTICULATION_CLOSED_TOLERANCE,
        "displacement": round(displacement, 5),
        "height_drop": round(height_drop, 5),
        "final_speed": round(final_speed, 5),
        "thresholds": {
            "displacement": CLOSE_DISPLACEMENT_REVIEW_THRESHOLD,
            "height_drop": CLOSE_HEIGHT_DROP_REVIEW_THRESHOLD,
            "final_speed": FINAL_SPEED_REVIEW_THRESHOLD,
        },
    }


def shelf_layer_check(env, fixed_instance: str, object_instance: str, location: str, layer_start_pos: np.ndarray, layer_final_pos: np.ndarray) -> dict[str, object] | None:
    if not is_shelf_layer_location(location):
        return None
    object_box = geom_bounding_box(env, object_instance)
    fixed_body = resolve_reference_body(env, fixed_instance, location)
    fixed_body_id = env.sim.model.body_name2id(fixed_body)
    fixed_z = float(env.sim.data.body_xpos[fixed_body_id][2])
    center_z_above_fixed = float(layer_final_pos[2] - fixed_z)
    displacement = float(np.linalg.norm(layer_final_pos[:2] - layer_start_pos[:2]))
    height_drop = float(max(0.0, layer_start_pos[2] - layer_final_pos[2]))
    object_height = float(object_box.get("height", 0.0)) if object_box else 0.0
    ok = (
        displacement <= DISPLACEMENT_REVIEW_THRESHOLD
        and height_drop <= HEIGHT_DROP_REVIEW_THRESHOLD
        and object_height <= SHELF_LAYER_MAX_OBJECT_HEIGHT
        and center_z_above_fixed <= SHELF_LAYER_MAX_CENTER_Z_ABOVE_FIXED
    )
    reasons = []
    if displacement > DISPLACEMENT_REVIEW_THRESHOLD:
        reasons.append("large_displacement")
    if height_drop > HEIGHT_DROP_REVIEW_THRESHOLD:
        reasons.append("height_drop")
    if object_height > SHELF_LAYER_MAX_OBJECT_HEIGHT:
        reasons.append("object_too_tall_for_layer")
    if center_z_above_fixed > SHELF_LAYER_MAX_CENTER_Z_ABOVE_FIXED:
        reasons.append("object_above_layer_clearance")
    return {
        "status": "clear" if ok else "blocked",
        "reason": "layer_clear" if ok else "+".join(reasons),
        "object_bounds": object_box,
        "center_z_above_fixed": round(center_z_above_fixed, 5),
        "displacement": round(displacement, 5),
        "height_drop": round(height_drop, 5),
        "thresholds": {
            "object_height": SHELF_LAYER_MAX_OBJECT_HEIGHT,
            "center_z_above_fixed": SHELF_LAYER_MAX_CENTER_Z_ABOVE_FIXED,
            "displacement": DISPLACEMENT_REVIEW_THRESHOLD,
            "height_drop": HEIGHT_DROP_REVIEW_THRESHOLD,
        },
    }


def apply_post_settle_checks(
    env,
    fixed_instance: str,
    object_instance: str,
    location: str,
    action: np.ndarray,
    settle_start_pos: np.ndarray,
    settle_final_pos: np.ndarray,
) -> tuple[dict[str, object], list[str]]:
    checks: dict[str, object] = {}
    blocking_reasons: list[str] = []
    shelf_check = shelf_layer_check(env, fixed_instance, object_instance, location, settle_start_pos, settle_final_pos)
    if shelf_check is not None:
        checks["shelf_layer_check"] = shelf_check
        if shelf_check.get("status") != "clear":
            blocking_reasons.append(f"shelf_layer_{shelf_check.get('reason', 'blocked')}")
    close_check = close_articulation_check(env, fixed_instance, object_instance, location, action)
    if close_check is not None:
        key = "microwave_close_check" if is_microwave_inside(location) else "drawer_close_check"
        checks[key] = close_check
        if close_check.get("status") != "closed":
            blocking_reasons.append(f"{key}_{close_check.get('reason', 'blocked')}")
    return checks, blocking_reasons


def post_settle_checks_blocked(result: dict[str, object]) -> bool:
    checks = result.get("post_settle_checks")
    if not isinstance(checks, dict):
        return False
    shelf_check = checks.get("shelf_layer_check")
    if isinstance(shelf_check, dict) and shelf_check.get("status") != "clear":
        return True
    for key in ("drawer_close_check", "microwave_close_check"):
        close_check = checks.get(key)
        if isinstance(close_check, dict) and close_check.get("status") != "closed":
            return True
    return False


def render_pose_png(
    scene: str,
    location: str,
    object_type: str,
    pose: dict[str, float],
    camera_name: str,
    image_size: int,
    settle_steps: int,
    force_safe_object_init: bool = False,
) -> bytes:
    if RENDER_IMPORT_ERROR is not None:
        raise RuntimeError(f"MuJoCo render imports failed: {RENDER_IMPORT_ERROR}")

    original_bddl, fixed_instance, workspace = resolve_scene_bddl(scene, location)
    manual_site_spawn = needs_manual_site_spawn(location) or force_safe_object_init
    bddl_text, object_instance, anchor_xy, site_based = build_render_bddl(
        original_bddl,
        object_type,
        workspace,
        fixed_instance,
        location,
        pose,
        safe_object_init=manual_site_spawn,
    )
    reference_body_z = reference_body_z_for(scene, location, camera_name, image_size) if manual_site_spawn else None
    target_body_z = None
    if reference_body_z is not None:
        target_body_z = reference_body_z + float(pose.get("z", 0.0)) + 0.02
    with tempfile.TemporaryDirectory(prefix="vlapb_render_") as tmp:
        bddl_path = Path(tmp) / f"{scene}_{location}_{object_type}.bddl"
        bddl_path.write_text(bddl_text, encoding="utf-8")
        env_kwargs, problem_name = build_env_kwargs(bddl_path, camera_name, image_size)
        
        seed = stable_seed(scene, location, object_type)
        seed_everything(seed)
        
        env = TASK_MAPPING[problem_name](**env_kwargs)
        try:
            seed_everything(seed)
            if hasattr(env, "seed"):
                env.seed(seed)

            reset_with_retries(env)

            #_object_state(env, "after reset / before move", object_instance, fixed_instance, location, pose)
            open_articulated_if_needed(env, fixed_instance, location)
            apply_sampled_pose_adjustments(
                env,
                object_instance,
                fixed_instance,
                location,
                pose,
                anchor_xy,
                site_based,
                target_body_z=target_body_z,
            )
            #log_object_state(env, "after reset / before move", object_instance, fixed_instance, location, pose)

            env.sim.forward()

            image = env.sim.render(
                width=image_size,
                height=image_size,
                camera_name=camera_name
            )[::-1, :, :].copy()

            return encode_png(image)
        finally:
            env.close()


def render_env_image(env, camera_name: str, image_size: int) -> np.ndarray:
    return env.sim.render(width=image_size, height=image_size, camera_name=camera_name)[::-1, :, :].copy()


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image[:, :, ::-1])
    if not ok:
        raise RuntimeError("Could not encode render image")
    return encoded.tobytes()


def render_plain_bddl_png(bddl_path: Path, camera_name: str, image_size: int) -> bytes:
    env_kwargs, problem_name = build_env_kwargs(bddl_path, camera_name, image_size)
    env = TASK_MAPPING[problem_name](**env_kwargs)
    try:
        reset_with_retries(env)
        return encode_png(render_env_image(env, camera_name, image_size))
    finally:
        env.close()


def body_name_for_instance(env, instance: str) -> str:
    if f"{instance}_main" in env.sim.model.body_names:
        return f"{instance}_main"
    if instance in env.sim.model.body_names:
        return instance
    raise RuntimeError(f"Cannot find body for {instance}")


def object_position(env, object_instance: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name_for_instance(env, object_instance))
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def object_speed(env, object_instance: str) -> float:
    joint_name = object_joint_name(env, object_instance)
    joint_id = env.sim.model.joint_name2id(joint_name)
    qvel_addr = env.sim.model.jnt_dofadr[joint_id]
    qvel = np.array(env.sim.data.qvel[qvel_addr : qvel_addr + 6], dtype=np.float64)
    return float(np.linalg.norm(qvel[:3]))


def status_from_motion(displacement: float, height_drop: float, final_speed: float) -> tuple[str, str]:
    reasons = []
    if displacement > DISPLACEMENT_REVIEW_THRESHOLD:
        reasons.append("large_displacement")
    if height_drop > HEIGHT_DROP_REVIEW_THRESHOLD:
        reasons.append("height_drop")
    if final_speed > FINAL_SPEED_REVIEW_THRESHOLD:
        reasons.append("unsettled")
    if not reasons:
        return "auto_possible", "settled"
    if "unsettled" in reasons and len(reasons) == 1:
        return "unsettled", "unsettled"
    return "review", "+".join(reasons)


def gif_relative_path(spawn_dir: Path, scene: str, location: str, object_type: str) -> Path:
    safe_location = location.replace("/", "_")
    safe_object = object_type.replace("/", "_")
    return Path("_gifs") / scene / safe_location / f"{safe_object}.gif"


def final_image_relative_path(spawn_dir: Path, scene: str, location: str, object_type: str) -> Path:
    safe_location = location.replace("/", "_")
    safe_object = object_type.replace("/", "_")
    return Path("_finals") / scene / safe_location / f"{safe_object}.png"


def save_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [frame.astype(np.uint8) for frame in frames]
    imageio.mimsave(path, rgb_frames, duration=1.0 / max(1, fps), loop=0)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image[:, :, ::-1])
    if not ok:
        raise RuntimeError("Could not encode final image")
    path.write_bytes(encoded.tobytes())


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def simulate_pose(
    *,
    scene: str,
    location: str,
    object_type: str,
    pose: dict[str, float],
    camera_name: str,
    image_size: int,
    sim_seconds: float,
    sim_fps: int,
    gif_path: Path | None = None,
    final_image_path: Path | None = None,
    force_safe_object_init: bool = False,
) -> dict[str, object]:
    if RENDER_IMPORT_ERROR is not None:
        raise RuntimeError(f"MuJoCo render imports failed: {RENDER_IMPORT_ERROR}")

    original_bddl, fixed_instance, workspace = resolve_scene_bddl(scene, location)
    manual_site_spawn = needs_manual_site_spawn(location) or force_safe_object_init
    bddl_text, object_instance, anchor_xy, site_based = build_render_bddl(
        original_bddl,
        object_type,
        workspace,
        fixed_instance,
        location,
        pose,
        safe_object_init=manual_site_spawn,
    )
    reference_body_z = reference_body_z_for(scene, location, camera_name, image_size) if manual_site_spawn else None
    target_body_z = None
    if reference_body_z is not None:
        target_body_z = reference_body_z + float(pose.get("z", 0.0)) + 0.02
    total_steps = max(1, int(sim_seconds * CONTROL_FREQ))
    frame_interval = max(1, int(CONTROL_FREQ / max(1, sim_fps)))
    frames: list[np.ndarray] = []

    with tempfile.TemporaryDirectory(prefix="vlapb_sim_") as tmp:
        bddl_path = Path(tmp) / f"{scene}_{location}_{object_type}.bddl"
        bddl_path.write_text(bddl_text, encoding="utf-8")
        env_kwargs, problem_name = build_env_kwargs(bddl_path, camera_name, image_size)
        seed = stable_seed(scene, location, object_type)
        seed_everything(seed)
        env = TASK_MAPPING[problem_name](**env_kwargs)
        try:
            seed_everything(seed)
            if hasattr(env, "seed"):
                env.seed(seed)
            reset_with_retries(env)
            open_articulated_if_needed(env, fixed_instance, location)
            apply_sampled_pose_adjustments(
                env,
                object_instance,
                fixed_instance,
                location,
                pose,
                anchor_xy,
                site_based,
                target_body_z=target_body_z,
            )
            env.sim.forward()
            start_pos = object_position(env, object_instance)
            action = np.zeros(get_action_dim(env), dtype=np.float32)

            for step in range(total_steps + 1):
                if step % frame_interval == 0:
                    frames.append(render_env_image(env, camera_name, image_size))
                if step < total_steps:
                    env.step(action)

            final_pos = object_position(env, object_instance)
            final_speed = object_speed(env, object_instance)
            displacement = float(np.linalg.norm(final_pos[:2] - start_pos[:2]))
            height_drop = float(max(0.0, start_pos[2] - final_pos[2]))
            velocity_stable = final_speed <= FINAL_SPEED_REVIEW_THRESHOLD
            velocity_warning = "" if velocity_stable else (
                f"Object velocity is not stable after {sim_seconds:.1f}s: "
                f"{final_speed:.5f} > {FINAL_SPEED_REVIEW_THRESHOLD:.5f}"
            )
            status, reason = status_from_motion(displacement, height_drop, final_speed)
            post_checks, post_check_reasons = apply_post_settle_checks(
                env,
                fixed_instance,
                object_instance,
                location,
                action,
                start_pos,
                final_pos,
            )
            if post_check_reasons:
                status = "review"
                reason = "+".join([reason, *post_check_reasons]) if reason != "settled" else "+".join(post_check_reasons)
                final_pos = object_position(env, object_instance)
                final_speed = object_speed(env, object_instance)
                displacement = float(np.linalg.norm(final_pos[:2] - start_pos[:2]))
                height_drop = float(max(0.0, start_pos[2] - final_pos[2]))
                velocity_stable = final_speed <= FINAL_SPEED_REVIEW_THRESHOLD
                velocity_warning = "" if velocity_stable else (
                    f"Object velocity is not stable after post-settle checks: "
                    f"{final_speed:.5f} > {FINAL_SPEED_REVIEW_THRESHOLD:.5f}"
                )
            final_anchor_pose = dict(normalize_pose(pose))
            final_anchor_pose["x"] = round(float(final_pos[0] - anchor_xy[0]), 4)
            final_anchor_pose["y"] = round(float(final_pos[1] - anchor_xy[1]), 4)

            if gif_path is not None:
                save_gif(gif_path, frames, sim_fps)
            if final_image_path is not None and frames:
                save_png(final_image_path, frames[-1])

            return {
                "scene": scene,
                "location": location,
                "object": object_type,
                "anchor_pose": normalize_pose(pose),
                "start_pose": {
                    "x": round(float(start_pos[0]), 4),
                    "y": round(float(start_pos[1]), 4),
                    "z": round(float(start_pos[2]), 4),
                },
                "final_pose": {
                    "x": round(float(final_pos[0]), 4),
                    "y": round(float(final_pos[1]), 4),
                    "z": round(float(final_pos[2]), 4),
                },
                "final_anchor_pose": final_anchor_pose,
                "displacement": round(displacement, 5),
                "height_drop": round(height_drop, 5),
                "final_speed": round(final_speed, 5),
                "velocity_stable": velocity_stable,
                "velocity_warning": velocity_warning,
                "velocity_threshold": FINAL_SPEED_REVIEW_THRESHOLD,
                "post_settle_checks": post_checks,
                "status": status,
                "reason": reason,
                "gif_path": str(gif_path) if gif_path is not None else None,
                "final_image_path": str(final_image_path) if final_image_path is not None else None,
                "timestamp": time.time(),
            }
        finally:
            env.close()


def camera_ray(env, camera_name: str, pixel_x: float, pixel_y: float, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    camera_id = env.sim.model.camera_name2id(camera_name)
    camera_pos = np.array(env.sim.data.cam_xpos[camera_id], dtype=np.float64)
    camera_mat = np.array(env.sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    fovy = np.deg2rad(float(env.sim.model.cam_fovy[camera_id]))
    x_ndc = (2.0 * pixel_x / image_size) - 1.0
    y_ndc = 1.0 - (2.0 * pixel_y / image_size)
    ray_camera = np.array([x_ndc * np.tan(fovy / 2.0), y_ndc * np.tan(fovy / 2.0), -1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)
    ray_world = camera_mat @ ray_camera
    ray_world /= np.linalg.norm(ray_world)
    return camera_pos, ray_world


def click_anchor_pose(
    *,
    scene: str,
    location: str,
    object_type: str,
    current_pose: dict[str, float],
    pixel_x: float,
    pixel_y: float,
    camera_name: str,
    image_size: int,
    force_safe_object_init: bool = False,
) -> dict[str, float]:
    if RENDER_IMPORT_ERROR is not None:
        raise RuntimeError(f"MuJoCo render imports failed: {RENDER_IMPORT_ERROR}")

    original_bddl, fixed_instance, workspace = resolve_scene_bddl(scene, location)
    manual_site_spawn = needs_manual_site_spawn(location) or force_safe_object_init
    bddl_text, object_instance, anchor_xy, site_based = build_render_bddl(
        original_bddl,
        object_type,
        workspace,
        fixed_instance,
        location,
        current_pose,
        safe_object_init=manual_site_spawn,
    )
    reference_body_z = reference_body_z_for(scene, location, camera_name, image_size) if manual_site_spawn else None
    target_body_z = None
    if reference_body_z is not None:
        target_body_z = reference_body_z + float(current_pose.get("z", 0.0)) + 0.02

    with tempfile.TemporaryDirectory(prefix="vlapb_pick_") as tmp:
        bddl_path = Path(tmp) / f"{scene}_{location}_{object_type}.bddl"
        bddl_path.write_text(bddl_text, encoding="utf-8")
        env_kwargs, problem_name = build_env_kwargs(bddl_path, camera_name, image_size)
        env = TASK_MAPPING[problem_name](**env_kwargs)
        try:
            reset_with_retries(env)
            open_articulated_if_needed(env, fixed_instance, location)
            apply_sampled_pose_adjustments(
                env,
                object_instance,
                fixed_instance,
                location,
                current_pose,
                anchor_xy,
                site_based,
                target_body_z=target_body_z,
            )
            env.sim.forward()
            ray_origin, ray_direction = camera_ray(env, camera_name, pixel_x, pixel_y, image_size)
            plane_z = float(object_position(env, object_instance)[2])
            if abs(ray_direction[2]) < 1e-6:
                raise RuntimeError("Camera ray is parallel to placement plane")
            t = (plane_z - ray_origin[2]) / ray_direction[2]
            world_point = ray_origin + t * ray_direction
            new_pose = dict(current_pose)
            new_pose["x"] = round(float(world_point[0] - anchor_xy[0]), 4)
            new_pose["y"] = round(float(world_point[1] - anchor_xy[1]), 4)
            LOGGER.debug(
                "click-anchor scene=%s location=%s object=%s pixel=(%.1f, %.1f) "
                "anchor_xy=(%.4f, %.4f) world=(%.4f, %.4f, %.4f) pose=(%.4f, %.4f)",
                scene,
                location,
                object_type,
                pixel_x,
                pixel_y,
                anchor_xy[0],
                anchor_xy[1],
                world_point[0],
                world_point[1],
                world_point[2],
                new_pose["x"],
                new_pose["y"],
            )
            return normalize_pose(new_pose)
        finally:
            env.close()


def click_anchor_pose_from_reference_bddl(
    *,
    reference_bddl: Path,
    target_scene: str,
    location: str,
    current_pose: dict[str, float],
    pixel_x: float,
    pixel_y: float,
    camera_name: str,
    image_size: int,
) -> dict[str, float]:
    if RENDER_IMPORT_ERROR is not None:
        raise RuntimeError(f"MuJoCo render imports failed: {RENDER_IMPORT_ERROR}")

    target_bddl, fixed_instance, workspace = resolve_scene_bddl(target_scene, location)
    target_text = target_bddl.read_text(encoding="utf-8")
    target_regions = extract_section_span(target_text, "regions")[2]
    target_init = extract_section_span(target_text, "init")[2]
    _target_anchor, target_anchor_xy, _target_recentered = effective_fixed_anchor_from_bddl(
        target_regions,
        target_init,
        fixed_instance,
        workspace,
        location,
    )

    reference_text = reference_bddl.read_text(encoding="utf-8")
    reference_fixtures = parse_typed_section(extract_section_span(reference_text, "fixtures")[2])
    reference_objects = parse_typed_section(extract_section_span(reference_text, "objects")[2])
    reference_fixed_instance = find_fixed_instance(reference_fixtures + reference_objects, location)
    reference_workspace = detect_workspace_name(reference_fixtures)
    if reference_fixed_instance is None:
        reference_fixed_instance = fixed_instance
    reference_regions = extract_section_span(reference_text, "regions")[2]
    reference_init = extract_section_span(reference_text, "init")[2]
    _reference_anchor, reference_anchor_xy = fixed_anchor_from_bddl(
        reference_regions,
        reference_init,
        reference_fixed_instance,
        reference_workspace,
    )
    reference_object = first_object_of_interest(reference_text)
    env_kwargs, problem_name = build_env_kwargs(reference_bddl, camera_name, image_size)
    env = TASK_MAPPING[problem_name](**env_kwargs)
    try:
        reset_with_retries(env)
        ray_origin, ray_direction = camera_ray(env, camera_name, pixel_x, pixel_y, image_size)
        plane_z = float(object_position(env, reference_object)[2])
        if abs(ray_direction[2]) < 1e-6:
            raise RuntimeError("Camera ray is parallel to placement plane")
        t = (plane_z - ray_origin[2]) / ray_direction[2]
        world_point = ray_origin + t * ray_direction
        new_pose = dict(current_pose)
        new_pose["x"] = round(float(world_point[0] - reference_anchor_xy[0]), 4)
        new_pose["y"] = round(float(world_point[1] - reference_anchor_xy[1]), 4)
        LOGGER.debug(
            "click-anchor-reference target_scene=%s location=%s pixel=(%.1f, %.1f) "
            "reference_anchor_xy=(%.4f, %.4f) target_anchor_xy=(%.4f, %.4f) "
            "world=(%.4f, %.4f, %.4f) pose=(%.4f, %.4f)",
            target_scene,
            location,
            pixel_x,
            pixel_y,
            reference_anchor_xy[0],
            reference_anchor_xy[1],
            target_anchor_xy[0],
            target_anchor_xy[1],
            world_point[0],
            world_point[1],
            world_point[2],
            new_pose["x"],
            new_pose["y"],
        )
        return normalize_pose(new_pose)
    finally:
        env.close()


class SpawnStore:
    def __init__(
        self,
        spawn_dir: Path,
        camera_name: str,
        image_size: int,
        settle_steps: int,
        sim_seconds: float,
        sim_fps: int,
    ):
        self.spawn_dir = spawn_dir
        self.feedback_path = spawn_dir / "_feedback.jsonl"
        self.results_path = spawn_dir / "_validation_results.jsonl"
        self.stats_path = spawn_dir / "_validation_stats.json"
        self.reference_path = spawn_dir / "_libero_reference_anchors.json"
        self.review_queue_path = spawn_dir / "_human_review_queue.json"
        self.gif_dir = spawn_dir / "_gifs"
        self.camera_name = camera_name
        self.image_size = image_size
        self.settle_steps = settle_steps
        self.sim_seconds = sim_seconds
        self.sim_fps = sim_fps
        self.migrate_deprecated_location_files()
        self.migrate_top_drawer_xy_from_bottom()

    def migrate_deprecated_location_files(self) -> None:
        for deprecated, canonical in CANONICAL_LOCATION_ALIASES.items():
            for old_path in sorted(self.spawn_dir.glob(f"*/{deprecated}.json")):
                new_path = old_path.with_name(f"{canonical}.json")
                old_data = read_json(old_path)
                new_data = read_json(new_path)
                changed = False
                if "_anchor" not in new_data and is_pose(old_data.get("_anchor")):
                    new_data["_anchor"] = normalize_pose(old_data["_anchor"])
                    changed = True
                for obj, value in old_data.items():
                    if obj.startswith("_"):
                        continue
                    if obj not in new_data or new_data[obj] == "N/A":
                        new_data[obj] = normalize_pose(value) if is_pose(value) else value
                        changed = True
                if changed:
                    write_json(new_path, new_data)
                    LOGGER.info("migrated deprecated %s to %s", old_path, new_path)

    def migrate_top_drawer_xy_from_bottom(self) -> None:
        for top_path in sorted(self.spawn_dir.glob(f"*/{TOP_DRAWER_LOCATION}.json")):
            bottom_path = top_path.with_name(f"{BOTTOM_DRAWER_LOCATION}.json")
            if not bottom_path.exists():
                continue
            bottom_anchor = anchor_pose(read_json(bottom_path))
            if bottom_anchor is None:
                continue
            top_data = read_json(top_path)
            changed = False
            for key, value in list(top_data.items()):
                if key == "_anchor" and is_pose(value):
                    updated = copy_xy_from_pose(normalize_pose(value), bottom_anchor)
                    if updated != value:
                        top_data[key] = updated
                        changed = True
                elif not key.startswith("_") and is_pose(value):
                    updated = copy_xy_from_pose(normalize_pose(value), bottom_anchor)
                    if updated != value:
                        top_data[key] = updated
                        changed = True
            if changed:
                write_json(top_path, top_data)
                LOGGER.info("aligned top drawer xy from bottom drawer: %s", top_path)

    def files(self) -> list[Path]:
        deprecated_names = {f"{location}.json" for location in CANONICAL_LOCATION_ALIASES}
        files = sorted(
            path for path in self.spawn_dir.glob("*/*.json")
            if not path.name.startswith("_") and path.name not in deprecated_names
        )
        LOGGER.debug(
            "spawn_dir=%s exists=%s json_files=%d",
            self.spawn_dir,
            self.spawn_dir.exists(),
            len(files),
        )
        if files:
            LOGGER.debug("first spawn files: %s", [str(path) for path in files[:5]])
        return files

    def debug_info(self) -> dict[str, object]:
        files = self.files()
        scenes = sorted({path.parent.name for path in files})
        return {
            "spawn_dir": str(self.spawn_dir),
            "spawn_dir_exists": self.spawn_dir.exists(),
            "spawn_dir_is_dir": self.spawn_dir.is_dir(),
            "json_file_count": len(files),
            "scene_count": len(scenes),
            "first_scenes": scenes[:10],
            "first_files": [str(path) for path in files[:10]],
        }

    def entries(self) -> list[dict[str, object]]:
        queue = []
        files = self.files()
        latest_by_combination = self.latest_results_by_combination()
        pending_review_keys = {
            (str(item.get("scene", "")), str(item.get("location", "")), str(item.get("object", "")))
            for item in self.pending_review_items()
        }
        LOGGER.info("building queue from %d spawn position files", len(files))
        for path in files:
            scene = path.parent.name
            location = path.stem
            data = read_json(path)
            objects = object_items(data)
            anchor = anchor_pose(data)
            object_validation: dict[str, str] = {}
            for obj in objects:
                key = (scene, location, obj)
                object_validation[obj] = self.validation_state_for_result(
                    latest_by_combination.get(key),
                    key in pending_review_keys,
                )
            location_needs_review = any(value == "need_verification" for value in object_validation.values())
            location_stable = bool(object_validation) and all(value == "stable" for value in object_validation.values())
            reference_candidates = reference_demo_candidates(scene, location)
            reference_match_type = ""
            if reference_candidates:
                reference_match_type = (
                    "exact_scene"
                    if any(scene_from_demo_path(candidate) == scene for candidate in reference_candidates)
                    else "similar_scene"
                )
            queue.append(
                {
                    "scene": scene,
                    "location": location,
                    "object": objects[0] if objects else "",
                    "objects": objects,
                    "path": str(path),
                    "pose": anchor,
                    "anchor": anchor,
                    "needs_initial_pose": anchor is None,
                    "validation_state": (
                        "need_verification"
                        if location_needs_review
                        else "stable" if location_stable else "not_run"
                    ),
                    "object_validation": object_validation,
                    "libero_reference_available": bool(reference_candidates),
                    "libero_reference_match_type": reference_match_type,
                }
            )
        queue.sort(
            key=lambda item: (
                not item["needs_initial_pose"],
                item["validation_state"] != "need_verification",
                item["scene"],
                item["location"],
            )
        )
        LOGGER.info("queue entries=%d missing_anchors=%d", len(queue), sum(1 for item in queue if item["needs_initial_pose"]))
        return queue

    def update(self, payload: dict[str, object]) -> dict[str, object]:
        path = Path(str(payload["path"]))
        obj = str(payload["object"])
        status = str(payload.get("status", "possible"))
        pose = normalize_pose(payload.get("pose"))
        data = read_json(path)
        if status == "impossible":
            data[obj] = "N/A"
        else:
            data[obj] = pose
            ensure_anchor(data, pose)
        write_json(path, data)
        feedback = {
            "scene": payload.get("scene"),
            "location": payload.get("location"),
            "object": obj,
            "path": str(path),
            "status": status,
            "pose": data[obj],
        }
        with self.feedback_path.open("a") as f:
            f.write(json.dumps(feedback, sort_keys=True) + "\n")
        return feedback

    def save_anchor(self, payload: dict[str, object], pose: dict[str, float]) -> dict[str, object]:
        path = Path(str(payload["path"]))
        data = read_json(path)
        pose = self.align_drawer_xy(str(payload["scene"]), str(payload["location"]), normalize_pose(pose))
        anchor = ensure_anchor(data, pose)
        write_json(path, data)
        return {"path": str(path), "anchor": anchor}

    def save_simulation_final_pose(self, payload: dict[str, object], result: dict[str, object]) -> None:
        final_pose = result.get("final_anchor_pose")
        if not is_pose(final_pose):
            return
        path = Path(str(payload["path"]))
        obj = str(payload["object"])
        data = read_json(path)
        normalized = normalize_pose(final_pose)
        data[obj] = normalized
        if payload.get("save_anchor_from_sim", False):
            ensure_anchor(data, normalized)
        write_json(path, data)

    def object_pose_for_scene_location(self, scene: str, location: str, object_type: str) -> dict[str, float] | None:
        path = self.spawn_dir / scene / f"{canonical_location(location)}.json"
        if not path.exists():
            return None
        value = read_json(path).get(object_type)
        return normalize_pose(value) if is_pose(value) else None

    def save_reference_cache(self, key: str, payload: dict[str, object]) -> None:
        cache = read_json(self.reference_path)
        cache[key] = payload
        write_json(self.reference_path, cache)

    def anchor_for_scene_location(self, scene: str, location: str) -> dict[str, float] | None:
        path = self.spawn_dir / scene / f"{canonical_location(location)}.json"
        if not path.exists():
            return None
        return anchor_pose(read_json(path))

    def reference_pose_for_scene_location(self, scene: str, location: str) -> dict[str, float] | None:
        reference = reference_anchor_for(scene, canonical_location(location), self.camera_name, self.image_size)
        if reference and reference.get("pose"):
            return normalize_pose(reference["pose"])
        return None

    def align_drawer_xy(self, scene: str, location: str, pose: dict[str, float]) -> dict[str, float]:
        source_location = drawer_xy_source_location(location)
        if source_location is None:
            return pose
        source_pose = (
            self.anchor_for_scene_location(scene, source_location)
            or self.reference_pose_for_scene_location(scene, source_location)
        )
        aligned = copy_xy_from_pose(pose, source_pose)
        if source_pose is not None and (aligned["x"] != pose["x"] or aligned["y"] != pose["y"]):
            LOGGER.debug(
                "aligned drawer xy scene=%s location=%s source=%s xy=(%.4f, %.4f)",
                scene,
                location,
                source_location,
                aligned["x"],
                aligned["y"],
            )
        return aligned

    def pose_from_payload(self, payload: dict[str, object]) -> dict[str, float]:
        scene = str(payload["scene"])
        location = str(payload["location"])
        pose = effective_pose_for_location(location, payload.get("pose") or payload.get("anchor"))
        return self.align_drawer_xy(scene, location, pose)

    def use_reference_anchor(self, payload: dict[str, object]) -> dict[str, object]:
        scene = str(payload["scene"])
        location = str(payload["location"])
        reference = reference_anchor_for(scene, location, self.camera_name, self.image_size)
        if not reference or not reference.get("pose"):
            raise ValueError(f"No usable LIBERO reference anchor for {scene}/{location}: {reference}")
        pose = self.align_drawer_xy(scene, location, normalize_pose(reference["pose"]))
        result = self.save_anchor(payload, pose)
        result.update({"scene": scene, "location": location, "reference": reference})
        self.save_reference_cache(f"{scene}/{location}", result)
        return result

    def render(self, payload: dict[str, object]) -> bytes:
        scene = str(payload["scene"])
        location = str(payload["location"])
        pose = self.pose_from_payload(payload)
        try:
            return render_pose_png(
                scene=scene,
                location=location,
                object_type=str(payload["object"]),
                pose=pose,
                camera_name=self.camera_name,
                image_size=self.image_size,
                settle_steps=self.settle_steps,
            )
        except RandomizationError as exc:
            debug_log("render retry safe object init", payload, exc)
            try:
                return render_pose_png(
                    scene=scene,
                    location=location,
                    object_type=str(payload["object"]),
                    pose=pose,
                    camera_name=self.camera_name,
                    image_size=self.image_size,
                    settle_steps=self.settle_steps,
                    force_safe_object_init=True,
                )
            except RandomizationError as safe_exc:
                exc = safe_exc
            reference = reference_anchor_for(scene, location, self.camera_name, self.image_size)
            reference_object = reference.get("demo_object_type") if reference else None
            reference_pose = reference.get("pose") if reference else None
            if not reference_object or not reference_pose:
                raise
            fallback_payload = dict(payload)
            fallback_payload["object"] = reference_object
            fallback_payload["pose"] = reference_pose
            debug_log("render fallback reference object", fallback_payload, exc)
            try:
                return render_pose_png(
                    scene=scene,
                    location=location,
                    object_type=str(reference_object),
                    pose=normalize_pose(reference_pose),
                    camera_name=self.camera_name,
                    image_size=self.image_size,
                    settle_steps=self.settle_steps,
                )
            except RandomizationError as fallback_exc:
                demo_bddl = reference.get("demo_bddl")
                if not demo_bddl:
                    raise
                debug_log("render fallback original reference bddl", fallback_payload, fallback_exc)
                return render_plain_bddl_png(Path(str(demo_bddl)), self.camera_name, self.image_size)

    def click_anchor(self, payload: dict[str, object]) -> dict[str, object]:
        scene = str(payload["scene"])
        location = str(payload["location"])
        current_pose = self.pose_from_payload(payload)
        try:
            pose = click_anchor_pose(
                scene=scene,
                location=location,
                object_type=str(payload["object"]),
                current_pose=current_pose,
                pixel_x=float(payload["pixel_x"]),
                pixel_y=float(payload["pixel_y"]),
                camera_name=self.camera_name,
                image_size=self.image_size,
            )
        except RandomizationError as exc:
            debug_log("click-anchor retry safe object init", payload, exc)
            try:
                pose = click_anchor_pose(
                    scene=scene,
                    location=location,
                    object_type=str(payload["object"]),
                    current_pose=current_pose,
                    pixel_x=float(payload["pixel_x"]),
                    pixel_y=float(payload["pixel_y"]),
                    camera_name=self.camera_name,
                    image_size=self.image_size,
                    force_safe_object_init=True,
                )
            except RandomizationError as safe_exc:
                exc = safe_exc
            else:
                result = self.save_anchor(payload, self.align_drawer_xy(scene, location, pose))
                result["scene"] = payload.get("scene")
                result["location"] = payload.get("location")
                return result
            reference = reference_anchor_for(scene, location, self.camera_name, self.image_size)
            demo_bddl = reference.get("demo_bddl") if reference else None
            if not demo_bddl:
                raise
            debug_log("click-anchor fallback original reference bddl", payload, exc)
            pose = click_anchor_pose_from_reference_bddl(
                reference_bddl=Path(str(demo_bddl)),
                target_scene=scene,
                location=location,
                current_pose=current_pose,
                pixel_x=float(payload["pixel_x"]),
                pixel_y=float(payload["pixel_y"]),
                camera_name=self.camera_name,
                image_size=self.image_size,
            )
        result = self.save_anchor(payload, self.align_drawer_xy(scene, location, pose))
        result["scene"] = payload.get("scene")
        result["location"] = payload.get("location")
        return result

    def simulate(self, payload: dict[str, object]) -> dict[str, object]:
        scene = str(payload["scene"])
        location = str(payload["location"])
        object_type = str(payload["object"])
        pose = self.pose_from_payload(payload)
        rel_gif = gif_relative_path(self.spawn_dir, scene, location, object_type)
        rel_final = final_image_relative_path(self.spawn_dir, scene, location, object_type)
        gif_path = self.spawn_dir / rel_gif
        final_image_path = self.spawn_dir / rel_final
        try:
            result = simulate_pose(
                scene=scene,
                location=location,
                object_type=object_type,
                pose=pose,
                camera_name=self.camera_name,
                image_size=self.image_size,
                sim_seconds=self.sim_seconds,
                sim_fps=self.sim_fps,
                gif_path=gif_path,
                final_image_path=final_image_path,
                force_safe_object_init=bool(payload.get("force_safe_object_init", False)),
            )
            result["gif_url"] = "/" + rel_gif.as_posix()
            result["final_image_url"] = "/" + rel_final.as_posix()
            self.save_simulation_final_pose(payload, result)
        except Exception as exc:
            debug_log("simulate result failed_render", payload, exc)
            result = {
                "scene": scene,
                "location": location,
                "object": object_type,
                "anchor_pose": pose,
                "status": "failed_render",
                "reason": repr(exc),
                "gif_path": None,
                "gif_url": None,
                "final_image_path": None,
                "final_image_url": None,
                "timestamp": time.time(),
            }
        append_jsonl(self.results_path, result)
        self.write_stats()
        return result

    def lifted_review_pose(self, item: dict[str, object]) -> dict[str, float]:
        scene = str(item["scene"])
        location = str(item["location"])
        object_type = str(item["object"])
        source_pose = (
            self.object_pose_for_scene_location(scene, location, object_type)
            or normalize_pose(item.get("pose"))
        )
        pose = self.align_drawer_xy(scene, location, source_pose)
        pose["z"] = round(max(float(pose.get("z", 0.0)) + REDROP_Z_OFFSET, REDROP_MIN_Z), 4)
        return pose

    def redrop_result_is_stable(self, result: dict[str, object]) -> bool:
        return (
            result.get("status") != "failed_render"
            and not post_settle_checks_blocked(result)
            and float(result.get("displacement", 999.0)) <= DISPLACEMENT_REVIEW_THRESHOLD
            and result.get("velocity_stable") is not False
        )

    def mark_redrop_result(self, result: dict[str, object], stable: bool) -> dict[str, object]:
        result = dict(result)
        result["redrop_review"] = True
        result["redrop_ignored_height_drop"] = True
        if stable:
            result["status"] = "auto_possible"
            result["reason"] = "redrop_settled"
        else:
            reasons = []
            if float(result.get("displacement", 999.0)) > DISPLACEMENT_REVIEW_THRESHOLD:
                reasons.append("large_displacement")
            if result.get("velocity_stable") is False:
                reasons.append("unsettled")
            if result.get("status") == "failed_render":
                reasons.append("failed_render")
            if post_settle_checks_blocked(result):
                reasons.append("post_settle_check_blocked")
            result["status"] = "review"
            result["reason"] = "redrop_" + ("+".join(reasons) if reasons else "needs_review")
        return result

    def redrop_review_queue(self) -> dict[str, object]:
        items = self.pending_review_items()
        settled = []
        still_review = []
        errors = []

        iterator = items
        if tqdm is not None:
            iterator = tqdm(items, desc="redrop need verification", unit="object")
        else:
            print(f"[validate_profiles:redrop] 0/{len(items)} need-verification objects", file=sys.stderr, flush=True)

        for idx, item in enumerate(iterator, start=1):
            scene = str(item["scene"])
            location = str(item["location"])
            object_type = str(item["object"])
            try:
                pose = self.lifted_review_pose(item)
                payload = {
                    "scene": scene,
                    "location": location,
                    "object": object_type,
                    "path": str(item.get("path") or self.spawn_dir / scene / f"{canonical_location(location)}.json"),
                    "pose": pose,
                    "anchor": pose,
                    "save_anchor_from_sim": False,
                    "force_safe_object_init": True,
                }
                result = self.simulate(payload)
                stable = self.redrop_result_is_stable(result)
                result = self.mark_redrop_result(result, stable)
                append_jsonl(self.results_path, result)
                self.save_simulation_final_pose(payload, result)
                if stable:
                    self.remove_review_item(result)
                    settled.append(result)
                else:
                    updated_item = dict(item)
                    updated_item.update(
                        {
                            "pose": result.get("final_anchor_pose") or pose,
                            "status": result.get("status"),
                            "reason": result.get("reason"),
                            "review_reason": "redrop_still_needs_verification",
                            "gif_url": result.get("gif_url"),
                            "final_image_url": result.get("final_image_url"),
                            "velocity_stable": result.get("velocity_stable"),
                            "velocity_warning": result.get("velocity_warning"),
                            "final_speed": result.get("final_speed"),
                        }
                    )
                    still_review.append(updated_item)
                print(
                    f"[validate_profiles:redrop] {idx}/{len(items)} {scene}/{location}/{object_type}: "
                    f"{'stable' if stable else 'need verification'}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                debug_log("redrop-review failed", item, exc)
                failed_item = dict(item)
                failed_item.update(
                    {
                        "status": "failed_render",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "review_reason": "redrop_failed",
                    }
                )
                still_review.append(failed_item)
                errors.append(failed_item)

        self.review_queue_path.write_text(json.dumps(still_review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "attempted": len(items),
            "settled_count": len(settled),
            "still_review_count": len(still_review),
            "error_count": len(errors),
            "settled": settled[:50],
            "review_items": still_review[:50],
            "review_items_note": "First 50 only in browser response; full queue is saved in _human_review_queue.json.",
            "z_offset": REDROP_Z_OFFSET,
            "min_z": REDROP_MIN_Z,
            "stats": self.write_stats(),
        }

    def validate_auto(self, payload: dict[str, object], include_results: bool = True) -> dict[str, object]:
        path = Path(str(payload["path"]))
        data = read_json(path)
        pose = anchor_pose(data)
        if pose is None:
            raise ValueError(f"Anchor is not set for {path}")
        pose = self.align_drawer_xy(str(payload["scene"]), str(payload["location"]), pose)
        objects = object_items(data)
        results = []
        iterator = objects
        if tqdm is not None:
            iterator = tqdm(objects, desc=f"{path.parent.name}/{path.stem}", unit="object")
        else:
            print(f"[validate_profiles:auto] {path.parent.name}/{path.stem} 0/{len(objects)}", file=sys.stderr, flush=True)
        for idx, obj in enumerate(iterator, start=1):
            row = dict(payload)
            row["object"] = obj
            row["pose"] = pose
            row["anchor"] = pose
            row["save_anchor_from_sim"] = False
            results.append(self.simulate(row))
            if tqdm is None:
                print(
                    f"[validate_profiles:auto] {path.parent.name}/{path.stem} {idx}/{len(objects)} {obj}",
                    file=sys.stderr,
                    flush=True,
                )
        forced_review = [
            result for result in results
            if result.get("status") in {"review", "unsettled", "failed_render"}
            or result.get("velocity_stable") is False
        ]
        stable_results = [
            result for result in results
            if result.get("status") == "auto_possible" and result.get("velocity_stable") is not False
        ]
        sample_count = int(len(stable_results) * 0.1)
        if stable_results and sample_count == 0:
            sample_count = 1
        sample_count = min(sample_count, len(stable_results))
        sampled_review = random.sample(stable_results, sample_count) if sample_count else []
        review_items = []
        seen = set()
        for reason, group in (("not_stable_or_failed", forced_review), ("random_10_percent_stable_check", sampled_review)):
            for result in group:
                key = (result.get("scene"), result.get("location"), result.get("object"))
                if key in seen:
                    continue
                seen.add(key)
                review_item = {
                    "scene": result.get("scene"),
                    "location": result.get("location"),
                    "object": result.get("object"),
                    "path": str(path),
                    "pose": result.get("final_anchor_pose") or result.get("anchor_pose") or pose,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "review_reason": reason,
                    "gif_url": result.get("gif_url"),
                    "final_image_url": result.get("final_image_url"),
                    "velocity_stable": result.get("velocity_stable"),
                    "velocity_warning": result.get("velocity_warning"),
                    "final_speed": result.get("final_speed"),
                }
                review_items.append(review_item)
        self.review_queue_path.write_text(json.dumps(review_items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        response = {
            "count": len(results),
            "review_count": len(review_items),
            "review_items": review_items,
            "stable_count": len(stable_results),
            "need_verification_count": len(forced_review),
            "random_sample_count": len(sampled_review),
            "not_stable_or_failed_count": len(forced_review),
            "stats": self.stats(),
        }
        if include_results:
            response["results"] = results
        return response

    def validate_all_auto(self) -> dict[str, object]:
        files = self.files()
        eligible = []
        skipped_missing_anchor = []
        for path in files:
            data = read_json(path)
            if anchor_pose(data) is None:
                skipped_missing_anchor.append(str(path))
                continue
            objects = object_items(data)
            if not objects:
                continue
            eligible.append((path, objects))

        review_items = []
        errors = []
        total_locations = len(eligible)
        total_objects = sum(len(objects) for _path, objects in eligible)
        completed_locations = 0
        completed_objects = 0

        iterator = eligible
        if tqdm is not None:
            iterator = tqdm(eligible, desc="all scene/location", unit="location")
        else:
            print(
                f"[validate_profiles:auto-all] 0/{total_locations} locations, 0/{total_objects} objects",
                file=sys.stderr,
                flush=True,
            )

        for path, objects in iterator:
            scene = path.parent.name
            location = path.stem
            payload = {
                "scene": scene,
                "location": location,
                "object": objects[0],
                "objects": objects,
                "path": str(path),
                "pose": anchor_pose(read_json(path)),
                "anchor": anchor_pose(read_json(path)),
            }
            try:
                result = self.validate_auto(payload, include_results=False)
                completed_locations += 1
                completed_objects += int(result.get("count", 0))
                review_items.extend(result.get("review_items", []))
                location_need = int(result.get("not_stable_or_failed_count", 0))
                location_sampled = int(result.get("random_sample_count", 0))
                location_stable = max(0, int(result.get("count", 0)) - location_need)
                status_text = (
                    f"need verification={location_need} sampled={location_sampled} stable={location_stable}"
                    if location_need or location_sampled
                    else f"stable={location_stable}"
                )
                if tqdm is not None and hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str(f"{scene}/{location}: {status_text}")
                print(
                    f"[validate_profiles:auto-all] done {scene}/{location}: {status_text}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                debug_log("validate-all-auto location failed", payload, exc)
                errors.append(
                    {
                        "scene": scene,
                        "location": location,
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if tqdm is None:
                print(
                    f"[validate_profiles:auto-all] {completed_locations}/{total_locations} locations, "
                    f"{completed_objects}/{total_objects} objects: {scene}/{location}",
                    file=sys.stderr,
                    flush=True,
                )

        self.review_queue_path.write_text(json.dumps(review_items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "locations_total": len(files),
            "locations_eligible": total_locations,
            "locations_completed": completed_locations,
            "locations_skipped_missing_anchor": len(skipped_missing_anchor),
            "skipped_missing_anchor": skipped_missing_anchor,
            "objects_total": total_objects,
            "objects_completed": completed_objects,
            "review_count": len(review_items),
            "review_items": review_items[:50],
            "review_items_note": "First 50 only in browser response; full queue is saved in _human_review_queue.json.",
            "error_count": len(errors),
            "errors": errors,
            "stats": self.write_stats(),
        }

    def review(self, payload: dict[str, object]) -> dict[str, object]:
        path = Path(str(payload["path"]))
        status = str(payload.get("status", "human_possible"))
        row = {
            "scene": payload.get("scene"),
            "location": payload.get("location"),
            "object": payload.get("object"),
            "path": str(path),
            "status": status,
            "reason": "human_review",
            "anchor_pose": normalize_pose(payload.get("pose") or payload.get("anchor")),
            "timestamp": time.time(),
        }
        append_jsonl(self.results_path, row)
        self.remove_review_item(row)
        self.write_stats()
        return row

    def result_rows(self) -> list[dict[str, object]]:
        if not self.results_path.exists():
            return []
        rows = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def validation_state_for_result(row: dict[str, object] | None, queued_for_review: bool = False) -> str:
        if queued_for_review:
            return "need_verification"
        if not row:
            return "not_run"
        status = str(row.get("status", "unknown"))
        if status in {"auto_possible", "human_possible"} and row.get("velocity_stable") is not False:
            return "stable"
        if status == "impossible":
            return "impossible"
        if status in {"review", "unsettled", "failed_render"} or row.get("velocity_stable") is False:
            return "need_verification"
        return "not_run"

    def latest_results_by_combination(self) -> dict[tuple[str, str, str], dict[str, object]]:
        latest_by_combination = {}
        for row in self.result_rows():
            key = (str(row.get("scene", "")), str(row.get("location", "")), str(row.get("object", "")))
            if all(key):
                latest_by_combination[key] = row
        return latest_by_combination

    def pending_review_items(self) -> list[dict[str, object]]:
        if not self.review_queue_path.exists():
            return []
        try:
            items = json.loads(self.review_queue_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(items, list):
            return []
        latest = self.latest_results_by_combination()
        pending = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("scene", "")), str(item.get("location", "")), str(item.get("object", "")))
            latest_status = str((latest.get(key) or {}).get("status", ""))
            if latest_status in {"human_possible", "impossible"}:
                continue
            pending.append(item)
        if len(pending) != len(items):
            self.review_queue_path.write_text(json.dumps(pending, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return pending

    def remove_review_item(self, row: dict[str, object]) -> None:
        if not self.review_queue_path.exists():
            return
        key = (str(row.get("scene", "")), str(row.get("location", "")), str(row.get("object", "")))
        if not all(key):
            return
        pending = [
            item for item in self.pending_review_items()
            if (str(item.get("scene", "")), str(item.get("location", "")), str(item.get("object", ""))) != key
        ]
        self.review_queue_path.write_text(json.dumps(pending, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def stats(self) -> dict[str, object]:
        files = self.files()
        anchor_complete = 0
        anchor_missing = 0
        object_combinations_total = 0
        object_pose_done = 0
        for path in files:
            data = read_json(path)
            objects = object_items(data)
            object_combinations_total += len(objects)
            object_pose_done += sum(1 for obj in objects if is_pose(data.get(obj)))
            if anchor_pose(data) is None:
                anchor_missing += 1
            else:
                anchor_complete += 1

        rows = self.result_rows()
        latest_by_combination = self.latest_results_by_combination()
        unique_validated_combinations = {
            (str(row.get("scene", "")), str(row.get("location", "")), str(row.get("object", "")))
            for row in rows
            if row.get("scene") and row.get("location") and row.get("object")
        }
        status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
        latest_status_counts = Counter(str(row.get("status", "unknown")) for row in latest_by_combination.values())
        reason_counts = Counter(str(row.get("reason", "unknown")) for row in rows)
        scene_counts: dict[str, Counter] = defaultdict(Counter)
        location_counts: dict[str, Counter] = defaultdict(Counter)
        object_counts: dict[str, Counter] = defaultdict(Counter)
        gifs_present = 0
        gifs_missing = 0
        for row in rows:
            status = str(row.get("status", "unknown"))
            scene_counts[str(row.get("scene", ""))][status] += 1
            location_counts[str(row.get("location", ""))][status] += 1
            object_counts[str(row.get("object", ""))][status] += 1
            gif_path = row.get("gif_path")
            if gif_path:
                if Path(str(gif_path)).exists():
                    gifs_present += 1
                else:
                    gifs_missing += 1
        review_queue_count = len(self.pending_review_items())

        latest_stable = latest_status_counts.get("auto_possible", 0) + latest_status_counts.get("human_possible", 0)
        latest_need_verification = sum(
            latest_status_counts.get(status, 0)
            for status in ("review", "unsettled", "failed_render")
        )
        latest_impossible = latest_status_counts.get("impossible", 0)
        validation_not_run = max(0, object_combinations_total - len(unique_validated_combinations))
        object_pose_na = max(0, object_combinations_total - object_pose_done)

        stats = {
            "anchors_total": len(files),
            "anchors_complete": anchor_complete,
            "anchors_missing": anchor_missing,
            "representative_anchor_total": len(files),
            "representative_anchor_done": anchor_complete,
            "object_combinations_total": object_combinations_total,
            "object_pose_done": object_pose_done,
            "object_pose_na": object_pose_na,
            "object_validation_unique_done": len(unique_validated_combinations),
            "object_validation_not_run": validation_not_run,
            "object_validation_note": (
                "Representative anchors are scene/location-level. Object counts are separate; "
                "a representative object click does not mean every object for that location was verified."
            ),
            "validation_runs": len(rows),
            "status_counts": dict(status_counts),
            "latest_status_counts": dict(latest_status_counts),
            "latest_stable_combinations": latest_stable,
            "latest_need_verification_combinations": latest_need_verification,
            "latest_impossible_combinations": latest_impossible,
            "review_queue_count": review_queue_count,
            "reason_counts": dict(reason_counts),
            "scene_counts": {key: dict(value) for key, value in scene_counts.items() if key},
            "location_counts": {key: dict(value) for key, value in location_counts.items() if key},
            "object_counts": {key: dict(value) for key, value in object_counts.items() if key},
            "gifs_present": gifs_present,
            "gifs_missing": gifs_missing,
        }
        return stats

    def write_stats(self) -> dict[str, object]:
        stats = self.stats()
        self.stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return stats


HTML_PATH = TOOLS_DIR / "validate_profiles_page.html"


def load_html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def make_handler(store: SpawnStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def send_json(self, data: object, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_bytes(body, "application/json", status)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = load_html().encode()
                self.send_bytes(body, "text/html; charset=utf-8")
            elif path == "/api/queue":
                self.send_json(store.entries())
            elif path == "/api/stats":
                self.send_json(store.write_stats())
            elif path == "/api/debug":
                self.send_json(store.debug_info())
            elif path == "/api/review-queue":
                self.send_json(store.pending_review_items())
            elif path.startswith("/_gifs/"):
                gif_path = store.spawn_dir / path.lstrip("/")
                if not gif_path.exists() or not gif_path.is_file():
                    self.send_error(404)
                    return
                body = gif_path.read_bytes()
                self.send_bytes(body, "image/gif")
            elif path.startswith("/_finals/"):
                image_path = store.spawn_dir / path.lstrip("/")
                if not image_path.exists() or not image_path.is_file():
                    self.send_error(404)
                    return
                body = image_path.read_bytes()
                self.send_bytes(body, "image/png")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if path == "/api/save":
                try:
                    self.send_json(store.update(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/save-anchor":
                try:
                    self.send_json(store.save_anchor(payload, normalize_pose(payload.get("pose") or payload.get("anchor"))))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/click-anchor":
                try:
                    self.send_json(store.click_anchor(payload))
                except Exception as exc:
                    debug_log("click-anchor failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/use-reference-anchor":
                try:
                    self.send_json(store.use_reference_anchor(payload))
                except Exception as exc:
                    debug_log("use-reference-anchor failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/render":
                try:
                    body = store.render(payload)
                    self.send_bytes(body, "image/png")
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as exc:
                    debug_log("render failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/simulate":
                try:
                    self.send_json(store.simulate(payload))
                except Exception as exc:
                    debug_log("simulate failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/validate-auto":
                try:
                    self.send_json(store.validate_auto(payload))
                except Exception as exc:
                    debug_log("validate-auto failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/validate-auto-all":
                try:
                    self.send_json(store.validate_all_auto())
                except Exception as exc:
                    debug_log("validate-auto-all failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/redrop-review-queue":
                try:
                    self.send_json(store.redrop_review_queue())
                except Exception as exc:
                    debug_log("redrop-review-queue failed", payload, exc)
                    self.send_json({"error": str(exc)}, status=400)
            elif path == "/api/save-review":
                try:
                    self.send_json(store.review(payload))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=400)
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    args.spawn_dir.mkdir(parents=True, exist_ok=True)
    store = SpawnStore(args.spawn_dir, args.camera_name, args.image_size, args.settle_steps, args.sim_seconds, args.sim_fps)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Open http://{args.host}:{args.port}")
    print(f"Editing {args.spawn_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
