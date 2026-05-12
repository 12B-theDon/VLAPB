#!/usr/bin/env python3
"""Render placement-suite fixture previews for left/right layouts.

This script renders fixture-only preview images before generating placement
episodes. It uses the validated left/right poses in
``scripts/data_generation/possible_spawn_positions`` and writes images under
``scripts/suites/placement_fixture_previews``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = SCRIPT_DIR.parents[1]
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from test_env_reconfiguration import render_task_image  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
        return iterable


DEFAULT_POSSIBLE_DIR = VLAPB_LIBERO_ROOT / "scripts" / "data_generation" / "possible_spawn_positions"
DEFAULT_HOLDED_DIR = VLAPB_LIBERO_ROOT / "scripts" / "data_generation" / "holded_spawn_positions"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "placement_fixture_previews"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_LIBERO_OBJECTS = VLAPB_LIBERO_ROOT / "docs" / "libero_objects.json"
DEFAULT_PROFILES = VLAPB_LIBERO_ROOT / "profiles" / "profiles.json"

PLACEMENT_FIXTURES = (
    "basket",
    "cabinet",
    "flat_stove",
    "microwave",
    "white_cabinet",
    "wooden_cabinet",
    "wooden_tray",
    "wooden_two_layer_shelf",
)
FLOOR_FIXTURE = "floor"
DEFAULT_PREVIEW_FIXTURES = (*PLACEMENT_FIXTURES, FLOOR_FIXTURE)
SIDES = ("left", "right")
FLOOR_LABELS = ("on", "front", "back", "left", "right")
FLOOR_SINGLE_Y = {"left": -0.26, "right": 0.26, "center": 0.0}
FLOOR_SINGLE_X = 0.0
FLOOR_MULTIPLE_X = 0.0
FLOOR_MULTIPLE_Y = {"left": -0.30, "right": 0.30}
AREA_TABLES = {
    "kitchen": "kitchen_table",
    "living_room": "living_room_table",
    "study": "study_table",
    "floor": "floor",
}
AREA_PROBLEMS = {
    "kitchen": "LIBERO_Kitchen_Tabletop_Manipulation",
    "living_room": "LIBERO_Living_Room_Tabletop_Manipulation",
    "study": "LIBERO_Study_Tabletop_Manipulation",
    "floor": "LIBERO_Floor_Manipulation",
}
RENDER_TYPE_ALIASES = {
    "cabinet": "wooden_cabinet",
}
OBJECT_SECTION_TYPES = {"basket", "plate", "wooden_tray"}
SEMANTIC_FAMILIES = {
    "cabinet": ("wooden_cabinet", "white_cabinet"),
    "white_cabinet": ("cabinet", "wooden_cabinet"),
    "wooden_cabinet": ("cabinet", "white_cabinet"),
    "basket": ("wooden_tray",),
    "wooden_tray": ("basket", "plate"),
    "plate": ("wooden_tray",),
    "flat_stove": ("plate",),
    "microwave": ("wooden_cabinet", "white_cabinet", "cabinet"),
    "wooden_two_layer_shelf": ("wooden_cabinet", "white_cabinet", "cabinet"),
}

# Approximate size proxy used only after same-fixture and semantic-family fallback.
APPROX_FIXTURE_SIZE = {
    "basket": (0.26, 0.22, 0.14),
    "cabinet": (0.32, 0.24, 0.32),
    "flat_stove": (0.28, 0.20, 0.04),
    "microwave": (0.32, 0.24, 0.22),
    "plate": (0.20, 0.20, 0.03),
    "white_cabinet": (0.32, 0.24, 0.32),
    "wooden_cabinet": (0.32, 0.24, 0.32),
    "wooden_tray": (0.28, 0.20, 0.05),
    "wooden_two_layer_shelf": (0.34, 0.22, 0.34),
}

LEFT_MIRRORED_FIXTURES = {
    "cabinet",
    "microwave",
    "white_cabinet",
    "wooden_cabinet",
    "wooden_two_layer_shelf",
}
DRAWER_FIXTURES = {
    "cabinet",
    "white_cabinet",
    "wooden_cabinet",
}
DRAWER_CONFLICT_FIXTURES = {
    "microwave",
    "wooden_two_layer_shelf",
}
MUTUALLY_EXCLUDED_MULTIPLE_PAIRS = {
    frozenset(("microwave", "wooden_two_layer_shelf")),
}
PREVIEW_SIDE_Y_LIMIT = {
    "kitchen": 0.34,
    "living_room": 0.34,
    "study": 0.34,
}
SINGLE_BASKET_Y_LIMIT = 0.24
MULTIPLE_LANE_CANDIDATES = (
    (-0.34, 0.34),
    (-0.32, 0.32),
    (-0.30, 0.30),
    (-0.28, 0.28),
    (-0.26, 0.26),
    (-0.24, 0.24),
    (-0.22, 0.22),
    (-0.20, 0.20),
)
MULTIPLE_BASKET_Y_LIMIT = 0.22
MULTIPLE_FLAT_STOVE_Y_LIMIT = 0.22
CENTER_X_BY_AREA = {
    "kitchen": 0.0,
    "living_room": 0.0,
    "study": 0.0,
}
CANONICAL_FLAT_STOVE_X_BY_AREA = {
    "kitchen": -0.12,
    "living_room": -0.12,
    "study": -0.12,
}
CANONICAL_FLAT_STOVE_Y = {
    "left": -0.22,
    "right": 0.22,
}
CANONICAL_FLAT_STOVE_YAW = 0.0
FIXTURE_Y_HALF_SIZE = {
    "basket": 0.16,
    "cabinet": 0.18,
    "flat_stove": 0.13,
    "microwave": 0.19,
    "white_cabinet": 0.18,
    "wooden_cabinet": 0.18,
    "wooden_tray": 0.13,
    "wooden_two_layer_shelf": 0.20,
}
ACCESS_MARGIN_Y = {
    "basket": 0.10,
    "cabinet": 0.07,
    "microwave": 0.08,
    "white_cabinet": 0.07,
    "wooden_cabinet": 0.07,
    "wooden_two_layer_shelf": 0.08,
}
SIDE_LABELS = {"left", "right", "front", "back"}
SIDE_LABEL_OFFSETS = {
    "left": {"x": 0.0, "y": -0.21},
    "right": {"x": 0.0, "y": 0.21},
    "front": {"x": 0.12, "y": 0.0},
    "back": {"x": -0.12, "y": 0.0},
}
USER_OBJECT_MARKERS = ()
GRASPABLE_OBJECT_POSE = {"x": 0.0, "y": 0.0, "z": 0.1, "h": 0.0}
FIXED_OBJECT_POSE = {"x": 0.0, "y": 0.0, "z": 0.1, "h": 0.0}


@dataclass(frozen=True)
class PoseSource:
    requested_fixture: str
    source_fixture: str
    side: str
    scene: str
    area: str
    pose: dict[str, float]
    path: str
    fallback: str


@dataclass(frozen=True)
class RenderCase:
    kind: str
    area: str
    scene: str
    fixtures: tuple[tuple[str, str], ...]
    output_path: str
    pose_sources: tuple[PoseSource, ...]
    object_markers: tuple[dict[str, Any], ...] = USER_OBJECT_MARKERS


def replace_pose(source: PoseSource, pose: dict[str, float]) -> PoseSource:
    return PoseSource(
        source.requested_fixture,
        source.source_fixture,
        source.side,
        source.scene,
        source.area,
        pose,
        source.path,
        source.fallback,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--possible-dir", type=Path, default=DEFAULT_POSSIBLE_DIR)
    parser.add_argument("--holded-dir", type=Path, default=DEFAULT_HOLDED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--libero-objects", type=Path, default=DEFAULT_LIBERO_OBJECTS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fixtures", nargs="+", default=list(DEFAULT_PREVIEW_FIXTURES))
    parser.add_argument("--graspable-objects", nargs="+", default=None, help="Optional graspable object type filter.")
    parser.add_argument("--scenes", nargs="+", default=None, help="Optional numbered scene filter, e.g. KITCHEN_SCENE1.")
    parser.add_argument("--user-id", default=None, help="Optional profile user_id filter for --graspable-only.")
    parser.add_argument("--manual-object", default=None, help="Render one explicit graspable object with --manual-placements.")
    parser.add_argument(
        "--manual-placements",
        nargs="+",
        default=None,
        help="Explicit fixture:label placements, e.g. wooden_cabinet:top basket:front microwave:front.",
    )
    parser.add_argument(
        "--placements-per-item",
        type=int,
        default=3,
        help="Number of placement preview images per graspable item; includes one preferred placement.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--only-left-mirrored",
        action="store_true",
        help="Render only cases affected by left-side mirroring corrections.",
    )
    parser.add_argument(
        "--graspable-only",
        action="store_true",
        help="Render only graspable object markers, without placement fixtures.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")


def area_for_scene(scene: str) -> str:
    if scene == "FLOOR_SCENE":
        return "floor"
    if scene.startswith("KITCHEN_SCENE"):
        return "kitchen"
    if scene.startswith("LIVING_ROOM_SCENE"):
        return "living_room"
    if scene.startswith("STUDY_SCENE"):
        return "study"
    return scene.lower()


def scene_sort_key(scene: str) -> tuple[str, int, str]:
    match = re.match(r"(.+?)(\d+)$", scene)
    if not match:
        return (scene, 0, scene)
    return (match.group(1), int(match.group(2)), scene)


def render_type(fixture: str) -> str:
    return RENDER_TYPE_ALIASES.get(fixture, fixture)


def instance_name(fixture: str, side: str) -> str:
    """Stable BDDL instance name for preview fixtures.

    Filenames and manifest entries keep the requested fixture name. BDDL uses
    the render type so aliases such as cabinet -> wooden_cabinet match LIBERO's
    object registry while still avoiding numeric suffixes.
    """
    return f"{render_type(fixture)}_{side}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_graspable_object_types(path: Path, object_filter: list[str] | None = None) -> tuple[str, ...]:
    data = read_json(path)
    requested = set(object_filter or [])
    types = []
    seen = set()
    for item in data.get("graspable_objects", []):
        object_type = str(item.get("type", ""))
        if not object_type or object_type in seen:
            continue
        if requested and object_type not in requested:
            continue
        seen.add(object_type)
        types.append(object_type)
    if requested:
        missing = sorted(requested - seen)
        if missing:
            raise ValueError(f"Unknown graspable object type(s): {', '.join(missing)}")
    if not types:
        raise ValueError(f"No graspable objects found in {path}")
    return tuple(types)


def graspable_object_marker(
    object_type: str,
    pose: dict[str, Any] | None = None,
    *,
    variant: str = "preview",
    label: str | None = None,
    user_id: str | None = None,
    fixed_type: str | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    safe_object_type = safe_name(object_type)
    marker = {
        "object_type": object_type,
        "instance": f"{safe_object_type}_preview_1",
        "region": f"{safe_object_type}_preview_region",
        "pose": normalize_pose(pose or GRASPABLE_OBJECT_POSE),
        "variant": variant,
    }
    if label is not None:
        marker["label"] = label
    if user_id is not None:
        marker["user_id"] = user_id
    if fixed_type is not None:
        marker["fixed_type"] = fixed_type
    if source_file is not None:
        marker["source_file"] = source_file
    return marker


def normalize_pose(value: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round(float(value.get("x", 0.0)), 4),
        "y": round(float(value.get("y", 0.0)), 4),
        "z": round(float(value.get("z", 0.0)), 4),
        "r": round(float(value.get("r", 0.0)), 4),
        "p": round(float(value.get("p", 0.0)), 4),
        "h": round(float(value.get("h", 0.0)), 4),
    }


def normalize_yaw(yaw: float) -> float:
    """Keep yaw in [-pi, pi] for readable BDDL output."""
    return round((yaw + math.pi) % (2 * math.pi) - math.pi, 4)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def rotate_offset(offset_x: float, offset_y: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        offset_x * cos_yaw - offset_y * sin_yaw,
        offset_x * sin_yaw + offset_y * cos_yaw,
    )


def side_goal_center(label: str, fixed_pose: dict[str, float]) -> tuple[float, float]:
    offset = SIDE_LABEL_OFFSETS[label]
    yaw = float(fixed_pose.get("h", 0.0))
    dx, dy = rotate_offset(offset["x"], offset["y"], yaw)
    return float(fixed_pose.get("x", 0.0)) + dx, float(fixed_pose.get("y", 0.0)) + dy


def side_label_reachable_for_pose(label: str, fixed_pose: dict[str, float], area: str) -> bool:
    if label not in SIDE_LABELS:
        return True
    _, goal_y = side_goal_center(label, fixed_pose)
    y_limit = PREVIEW_SIDE_Y_LIMIT.get(area, 0.34)
    if abs(goal_y) > y_limit:
        return False
    fixture_y = float(fixed_pose.get("y", 0.0))
    if abs(fixture_y) < 1e-6:
        return True
    same_side = goal_y * fixture_y > 0.0
    farther_outward = abs(goal_y) > abs(fixture_y) + 1e-3
    if same_side and farther_outward:
        return False
    return True


def canonical_flat_stove_pose(side: str, pose: dict[str, float], area: str) -> dict[str, float]:
    adjusted = dict(pose)
    adjusted["x"] = CANONICAL_FLAT_STOVE_X_BY_AREA.get(area, -0.12)
    adjusted["y"] = CANONICAL_FLAT_STOVE_Y[side]
    adjusted["h"] = CANONICAL_FLAT_STOVE_YAW
    return adjusted


def enforce_side_y_policy(requested_fixture: str, side: str, pose: dict[str, float], area: str) -> dict[str, float]:
    adjusted = dict(pose)
    if requested_fixture == "flat_stove":
        adjusted = canonical_flat_stove_pose(side, adjusted, area)

    y_limit = PREVIEW_SIDE_Y_LIMIT.get(area)
    if y_limit is not None:
        if side == "right":
            adjusted["y"] = clamp(float(adjusted.get("y", 0.0)), 0.0, y_limit)
        elif side == "left":
            adjusted["y"] = clamp(float(adjusted.get("y", 0.0)), -y_limit, 0.0)

    if requested_fixture == "basket":
        if side == "right":
            adjusted["y"] = min(float(adjusted.get("y", 0.0)), SINGLE_BASKET_Y_LIMIT)
        elif side == "left":
            adjusted["y"] = max(float(adjusted.get("y", 0.0)), -SINGLE_BASKET_Y_LIMIT)

    if requested_fixture != "flat_stove":
        adjusted["x"] = CENTER_X_BY_AREA.get(area, adjusted.get("x", 0.0))
    return adjusted


def adjust_preview_pose(requested_fixture: str, side: str, pose: dict[str, float], area: str) -> dict[str, float]:
    adjusted = dict(pose)
    if side == "left" and requested_fixture in LEFT_MIRRORED_FIXTURES:
        adjusted["h"] = normalize_yaw(float(adjusted.get("h", 0.0)) + math.pi)
    return enforce_side_y_policy(requested_fixture, side, adjusted, area)


def y_clearance_half_size(fixture: str) -> float:
    return FIXTURE_Y_HALF_SIZE.get(fixture, 0.16) + ACCESS_MARGIN_Y.get(fixture, 0.05)


def multiple_candidate_score(
    left_fixture: str,
    right_fixture: str,
    left_y: float,
    right_y: float,
) -> tuple[bool, float]:
    if left_fixture == "basket" and abs(left_y) > MULTIPLE_BASKET_Y_LIMIT:
        return False, float("inf")
    if right_fixture == "basket" and abs(right_y) > MULTIPLE_BASKET_Y_LIMIT:
        return False, float("inf")
    if left_fixture == "flat_stove" and abs(left_y) > MULTIPLE_FLAT_STOVE_Y_LIMIT:
        return False, float("inf")
    if right_fixture == "flat_stove" and abs(right_y) > MULTIPLE_FLAT_STOVE_Y_LIMIT:
        return False, float("inf")

    left_clearance = y_clearance_half_size(left_fixture)
    right_clearance = y_clearance_half_size(right_fixture)
    clearance_gap = abs(right_y - left_y) - left_clearance - right_clearance
    if clearance_gap < 0.04:
        return False, float("inf")

    lane_balance = abs(abs(left_y) - abs(right_y))
    center_bias = abs(left_y + right_y)
    excessive_spread = max(0.0, abs(left_y) - 0.30) + max(0.0, abs(right_y) - 0.30)
    return True, lane_balance + center_bias + excessive_spread


def choose_multiple_y_positions(left_fixture: str, right_fixture: str) -> tuple[float, float]:
    scored = []
    for left_y, right_y in MULTIPLE_LANE_CANDIDATES:
        ok, score = multiple_candidate_score(left_fixture, right_fixture, left_y, right_y)
        if ok:
            scored.append((score, left_y, right_y))
    if scored:
        _, left_y, right_y = min(scored, key=lambda item: item[0])
        return left_y, right_y

    left_y, right_y = MULTIPLE_LANE_CANDIDATES[0]
    if left_fixture == "basket":
        left_y = -MULTIPLE_BASKET_Y_LIMIT
    if right_fixture == "basket":
        right_y = MULTIPLE_BASKET_Y_LIMIT
    if left_fixture == "flat_stove":
        left_y = -MULTIPLE_FLAT_STOVE_Y_LIMIT
    if right_fixture == "flat_stove":
        right_y = MULTIPLE_FLAT_STOVE_Y_LIMIT
    return left_y, right_y


def should_skip_multiple_pair(left_fixture: str, right_fixture: str) -> bool:
    """Skip fixture pairs whose interaction clearance is too tight to be useful."""
    pair = {left_fixture, right_fixture}
    if frozenset(pair) in MUTUALLY_EXCLUDED_MULTIPLE_PAIRS:
        return True
    if left_fixture in DRAWER_FIXTURES and right_fixture in DRAWER_FIXTURES:
        return True
    return bool(pair & DRAWER_FIXTURES and pair & DRAWER_CONFLICT_FIXTURES)


def adjust_multiple_pose_sources(left: PoseSource, right: PoseSource) -> tuple[PoseSource, PoseSource]:
    """Keep two-fixture previews visually comparable across fixtures.

    The raw possible_spawn_positions are validated for graspable objects around
    fixtures, not for placing two large fixtures side by side. For multiple
    previews, preserve scene-specific x/z/yaw while enforcing stable left/right
    lanes so different fixture combinations remain fair to inspect.
    """
    left_pose = dict(left.pose)
    right_pose = dict(right.pose)
    left_y, right_y = choose_multiple_y_positions(left.requested_fixture, right.requested_fixture)
    left_pose["y"] = left_y
    right_pose["y"] = right_y
    return replace_pose(left, left_pose), replace_pose(right, right_pose)


def pose_from_file(path: Path) -> dict[str, float] | None:
    data = read_json(path)
    poses = []
    for object_type, entry in data.items():
        if str(object_type).startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("skip_reason"):
            continue
        pose = entry.get("relative_scene_table_fixed_pose") or entry.get("pose") or entry.get("anchor_pose")
        if isinstance(pose, dict) and "x" in pose and "y" in pose:
            poses.append(normalize_pose(pose))
    if not poses:
        return None
    return {
        key: round(sum(pose[key] for pose in poses) / len(poses), 4)
        for key in ("x", "y", "z", "r", "p", "h")
    }


def available_scenes(possible_dir: Path, scene_filter: list[str] | None = None) -> list[str]:
    scenes = [path.name for path in possible_dir.iterdir() if path.is_dir() and not path.name.startswith("_") and not path.name.startswith("modified_")]
    if scene_filter:
        allowed = set(scene_filter)
        scenes = [scene for scene in scenes if scene in allowed]
    return sorted(scenes, key=scene_sort_key)


def modified_roots(possible_dir: Path) -> list[Path]:
    return sorted(path for path in possible_dir.iterdir() if path.is_dir() and path.name.startswith("modified_"))


def direct_pose_path(possible_dir: Path, scene: str, fixture: str, side: str) -> Path | None:
    candidates = [possible_dir / scene / fixture / f"{side}.json"]
    candidates.extend(root / scene / fixture / f"{side}.json" for root in modified_roots(possible_dir))
    for path in candidates:
        if path.exists():
            return path
    return None


def size_distance(a: str, b: str) -> float:
    sa = APPROX_FIXTURE_SIZE.get(a, (1.0, 1.0, 1.0))
    sb = APPROX_FIXTURE_SIZE.get(b, (1.0, 1.0, 1.0))
    return math.dist(sa, sb)


def fallback_fixtures(requested_fixture: str, all_fixtures: list[str]) -> list[str]:
    ordered = []
    for fixture in SEMANTIC_FAMILIES.get(requested_fixture, ()):
        if fixture in all_fixtures and fixture not in ordered:
            ordered.append(fixture)
    for fixture in sorted(all_fixtures, key=lambda item: (size_distance(requested_fixture, item), item)):
        if fixture != requested_fixture and fixture not in ordered:
            ordered.append(fixture)
    return ordered


def available_pose_fixtures(possible_dir: Path) -> list[str]:
    fixtures = set(PLACEMENT_FIXTURES)
    roots = [possible_dir, *modified_roots(possible_dir)]
    for root in roots:
        for scene_dir in root.iterdir() if root.exists() else []:
            if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
                continue
            for fixture_dir in scene_dir.iterdir():
                if fixture_dir.is_dir() and not fixture_dir.name.startswith("_"):
                    fixtures.add(fixture_dir.name)
    return sorted(fixtures)


def resolve_pose_source(possible_dir: Path, scene: str, requested_fixture: str, side: str, all_fixtures: list[str]) -> PoseSource:
    area = area_for_scene(scene)
    direct = direct_pose_path(possible_dir, scene, requested_fixture, side)
    if direct:
        pose = pose_from_file(direct)
        if pose:
            return PoseSource(requested_fixture, requested_fixture, side, scene, area, adjust_preview_pose(requested_fixture, side, pose, area), str(direct), "direct")

    for source_fixture in fallback_fixtures(requested_fixture, all_fixtures):
        same_scene = direct_pose_path(possible_dir, scene, source_fixture, side)
        if same_scene:
            pose = pose_from_file(same_scene)
            if pose:
                return PoseSource(requested_fixture, source_fixture, side, scene, area, adjust_preview_pose(requested_fixture, side, pose, area), str(same_scene), "same_scene_fallback")

    area_scenes = [candidate.name for candidate in possible_dir.iterdir() if candidate.is_dir() and area_for_scene(candidate.name) == area]
    for source_fixture in [requested_fixture, *fallback_fixtures(requested_fixture, all_fixtures)]:
        for candidate_scene in sorted(area_scenes, key=scene_sort_key):
            candidate = direct_pose_path(possible_dir, candidate_scene, source_fixture, side)
            if candidate:
                pose = pose_from_file(candidate)
                if pose:
                    return PoseSource(requested_fixture, source_fixture, side, candidate_scene, area, adjust_preview_pose(requested_fixture, side, pose, area), str(candidate), "same_area_fallback")

    for source_fixture in [requested_fixture, *fallback_fixtures(requested_fixture, all_fixtures)]:
        for candidate_scene in available_scenes(possible_dir):
            candidate = direct_pose_path(possible_dir, candidate_scene, source_fixture, side)
            if candidate:
                pose = pose_from_file(candidate)
                if pose:
                    return PoseSource(requested_fixture, source_fixture, side, candidate_scene, area_for_scene(candidate_scene), adjust_preview_pose(requested_fixture, side, pose, area), str(candidate), "global_size_fallback")

    raise FileNotFoundError(f"No {side} pose found for {requested_fixture} or fallback fixtures in {possible_dir}")


def region_block(name: str, table: str, pose: dict[str, float], half_size: float = 0.01) -> str:
    x = float(pose["x"])
    y = float(pose["y"])
    h = float(pose.get("h", 0.0))
    return f"""      ({name}
          (:target {table})
          (:ranges (
              ({x - half_size:.4f} {y - half_size:.4f} {x + half_size:.4f} {y + half_size:.4f})
            )
          )
          (:yaw_rotation (
              ({h:.4f} {h:.4f})
            )
          )
      )"""


def object_region_block(name: str, target: str) -> str:
    return f"""      ({name}
          (:target {target})
      )"""


def semantic_region_for_marker(marker: dict[str, Any]) -> tuple[str, str, str] | None:
    label = marker.get("label")
    fixed_type = marker.get("fixed_type")
    if not label or not fixed_type:
        return None
    fixed_instance = instance_name(str(fixed_type), "center")
    if label == "in":
        region = "heating_region" if fixed_type == "microwave" else "contain_region"
        return "In", fixed_instance, region
    if label == "top":
        return "In", fixed_instance, "top_region"
    if label == "bottom":
        return "In", fixed_instance, "bottom_region"
    if label == "on":
        if fixed_type == "flat_stove":
            return "On", fixed_instance, "cook_region"
        return "On", fixed_instance, ""
    return None


def marker_init_predicate_and_target(marker: dict[str, Any], table: str) -> tuple[str, str]:
    semantic = semantic_region_for_marker(marker)
    if semantic:
        predicate, fixed_instance, region = semantic
        target = fixed_instance if not region else f"{fixed_instance}_{region}"
        return predicate, target
    return "On", f"{table}_{marker['region']}"


def render_bddl(scene: str, pose_sources: tuple[PoseSource, ...], object_markers: tuple[dict[str, Any], ...]) -> str:
    area = area_for_scene(scene)
    table = AREA_TABLES.get(area, "main_table")
    problem = AREA_PROBLEMS.get(area, "LIBERO_Tabletop_Manipulation")
    regions = []
    init_lines = []
    fixture_instances: dict[str, list[str]] = defaultdict(list)
    object_instances: dict[str, list[str]] = defaultdict(list)
    fixture_instances[table if area in AREA_TABLES else "table"].append(table)
    interests = []

    for source in pose_sources:
        requested = source.requested_fixture
        fixture_type = render_type(requested)
        instance = instance_name(requested, source.side)
        region = f"{instance}_{source.side}_region"
        regions.append(region_block(region, table, source.pose))
        init_lines.append(f"    (On {instance} {table}_{region})")
        interests.append(instance)
        if fixture_type in OBJECT_SECTION_TYPES:
            object_instances[fixture_type].append(instance)
        else:
            fixture_instances[fixture_type].append(instance)

    added_object_regions = set()
    for marker in object_markers:
        semantic = semantic_region_for_marker(marker)
        if semantic:
            _, fixed_instance, region = semantic
            if region and (fixed_instance, region) not in added_object_regions:
                regions.append(object_region_block(region, fixed_instance))
                added_object_regions.add((fixed_instance, region))
        else:
            regions.append(region_block(marker["region"], table, marker["pose"], half_size=0.025))
        predicate, target = marker_init_predicate_and_target(marker, table)
        init_lines.append(f"    ({predicate} {marker['instance']} {target})")
        object_instances[marker["object_type"]].append(marker["instance"])
        interests.append(marker["instance"])

    fixture_lines = [
        f"    {' '.join(instances)} - {fixture_type}"
        for fixture_type, instances in fixture_instances.items()
    ]
    object_lines = [
        f"    {' '.join(instances)} - {object_type}"
        for object_type, instances in object_instances.items()
    ]

    goal_terms = " ".join(
        [
            *(
                f"(On {instance_name(src.requested_fixture, src.side)} {table}_{instance_name(src.requested_fixture, src.side)}_{src.side}_region)"
                for src in pose_sources
            ),
            *(
                f"({marker_init_predicate_and_target(marker, table)[0]} {marker['instance']} {marker_init_predicate_and_target(marker, table)[1]})"
                for marker in object_markers
            ),
        ]
    )
    return f"""(define (problem {problem})
  (:domain robosuite)
  (:language Render placement fixture preview for {scene})
    (:regions
{chr(10).join(regions)}
    )

  (:fixtures
{chr(10).join(fixture_lines)}
  )

  (:objects
{chr(10).join(object_lines)}
  )

  (:obj_of_interest
{chr(10).join(f'    {item}' for item in interests)}
  )

  (:init
{chr(10).join(init_lines)}
  )

  (:goal
    (And {goal_terms})
  )

)
"""


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def pose_from_libero_region(region: dict[str, Any]) -> dict[str, float] | None:
    ranges = region.get("ranges") or []
    if not ranges or not ranges[0] or len(ranges[0][0]) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in ranges[0][0][:4]]
    yaw_rotation = region.get("yaw_rotation") or []
    yaw = 0.0
    if yaw_rotation and yaw_rotation[0] and yaw_rotation[0][0]:
        yaw_values = yaw_rotation[0][0]
        yaw = sum(float(value) for value in yaw_values) / len(yaw_values)
    return normalize_pose({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "z": 0.1, "h": yaw})


def original_fixed_object_pose(libero_objects_path: Path, scene: str, fixed_type: str) -> tuple[dict[str, float], str] | None:
    data = read_json(libero_objects_path)
    for item in data.get("fixed_objects", []):
        if str(item.get("scene")) != scene or str(item.get("type")) != fixed_type:
            continue
        region = item.get("location", {}).get("region")
        if not isinstance(region, dict):
            continue
        pose = pose_from_libero_region(region)
        if pose:
            return pose, f"{libero_objects_path}:{item.get('id', fixed_type)}"
    return None


def fixed_object_source(scene: str, fixed_type: str, libero_objects_path: Path) -> PoseSource:
    area = area_for_scene(scene)
    original = original_fixed_object_pose(libero_objects_path, scene, fixed_type)
    pose, path, fallback = (
        (original[0], original[1], "libero_original")
        if original
        else (normalize_pose(FIXED_OBJECT_POSE), "profile_fixed_object", "profile_fixed_object")
    )
    return PoseSource(
        requested_fixture=fixed_type,
        source_fixture=fixed_type,
        side="center",
        scene=scene,
        area=area,
        pose=pose,
        path=path,
        fallback=fallback,
    )


def floor_fixture_pose_source(fixture: str, side: str, *, multiple: bool = False) -> PoseSource:
    if multiple:
        pose = {"x": FLOOR_MULTIPLE_X, "y": FLOOR_MULTIPLE_Y.get(side, 0.0), "z": 0.1, "r": 0.0, "p": 0.0, "h": 0.0}
    else:
        pose = {"x": FLOOR_SINGLE_X, "y": FLOOR_SINGLE_Y.get(side, 0.0), "z": 0.1, "r": 0.0, "p": 0.0, "h": 0.0}
    area = area_for_scene("FLOOR_SCENE")
    return PoseSource(
        requested_fixture=fixture,
        source_fixture=fixture,
        side=side,
        scene="FLOOR_SCENE",
        area=area,
        pose=adjust_preview_pose(fixture, side, normalize_pose(pose), area),
        path="floor_canonical_pose",
        fallback="floor_canonical_pose",
    )


def floor_possible_path(possible_dir: Path, holded_dir: Path, fixed_type: str, label: str) -> Path | None:
    for root in (possible_dir, holded_dir):
        path = root / "FLOOR_SCENE" / fixed_type / f"{label}.json"
        if path.exists():
            return path
    return None


def has_drawer_preference(profile: dict[str, Any]) -> bool:
    return any(
        str(pref.get("fixed_type")) in DRAWER_FIXTURES or str(pref.get("label")) in {"top", "bottom"}
        for pref in profile.get("placements", [])
    )


def choose_profile(payload: dict[str, Any], user_id: str | None, rng: random.Random, object_filter: list[str] | None) -> dict[str, Any]:
    profiles = list(payload.get("profiles", []))
    if user_id:
        profiles = [profile for profile in profiles if str(profile.get("user_id")) == user_id]
        if not profiles:
            raise ValueError(f"user_id not found in profiles: {user_id}")
    if object_filter:
        allowed = set(object_filter)
        profiles = [
            profile
            for profile in profiles
            if any(str(pref.get("object_type")) in allowed for pref in profile.get("placements", []))
        ]
    if not profiles:
        raise ValueError("No profile has matching placement preferences")
    if not user_id:
        drawer_profiles = [profile for profile in profiles if has_drawer_preference(profile)]
        if drawer_profiles:
            profiles = drawer_profiles
    return rng.choice(profiles)


def choose_profile_scene(profile: dict[str, Any], scene_filter: list[str] | None, rng: random.Random, object_filter: list[str] | None) -> str:
    allowed_objects = set(object_filter or [])
    scenes = []
    for pref in profile.get("placements", []):
        if allowed_objects and str(pref.get("object_type")) not in allowed_objects:
            continue
        scene = str(pref.get("scene", ""))
        if not scene:
            continue
        if scene_filter and scene not in scene_filter:
            continue
        if scene not in scenes:
            scenes.append(scene)
    if not scenes:
        raise ValueError(f"No matching placement scene for user {profile.get('user_id')}")
    return rng.choice(scenes)


def pose_for_object_from_possible(path: Path, object_type: str) -> dict[str, float] | None:
    data = read_json(path)
    entry = data.get(object_type)
    if not isinstance(entry, dict) or entry.get("skip_reason"):
        return None
    pose = entry.get("relative_scene_table_fixed_pose") or entry.get("pose") or entry.get("anchor_pose")
    if not isinstance(pose, dict):
        return None
    return normalize_pose(pose)


def floor_pose_file(possible_dir: Path, holded_dir: Path, label: str) -> Path | None:
    for root in (possible_dir, holded_dir):
        path = root / "FLOOR_SCENE" / FLOOR_FIXTURE / f"{label}.json"
        if path.exists():
            return path
    return None


def floor_markers_from_file(path: Path, label: str, object_filter: list[str] | None = None) -> list[dict[str, Any]]:
    data = read_json(path)
    allowed = set(object_filter or [])
    markers = []
    for object_type in sorted(key for key in data if not str(key).startswith("_")):
        if allowed and object_type not in allowed:
            continue
        pose = pose_for_object_from_possible(path, object_type)
        if pose is None:
            continue
        markers.append(
            graspable_object_marker(
                object_type,
                pose,
                variant=f"floor_{safe_name(label)}",
                label=label,
                source_file=str(path),
            )
        )
    return markers


def build_floor_cases(
    args: argparse.Namespace,
    *,
    kind: str = "floor",
    output_group: str = "floor",
) -> list[RenderCase]:
    if args.scenes and "FLOOR_SCENE" not in set(args.scenes):
        return []
    cases = []
    for label in FLOOR_LABELS:
        path = floor_pose_file(args.possible_dir, args.holded_dir, label)
        if path is None:
            continue
        for marker in floor_markers_from_file(path, label, args.graspable_objects):
            output = args.output_dir / "floor" / "FLOOR_SCENE" / output_group / safe_name(label) / f"{safe_name(marker['object_type'])}.png"
            cases.append(
                RenderCase(
                    kind,
                    "floor",
                    "FLOOR_SCENE",
                    ((FLOOR_FIXTURE, label),),
                    str(output),
                    (),
                    (marker,),
                )
            )
    return cases


def random_possible_markers(
    possible_dir: Path,
    pref: dict[str, Any],
    rng: random.Random,
    fixed_pose: dict[str, float],
    count: int = 2,
) -> tuple[dict[str, Any], ...]:
    scene = str(pref["scene"])
    fixed_type = str(pref["fixed_type"])
    object_type = str(pref["object_type"])
    preferred_label = str(pref["label"])
    root = possible_dir / scene / fixed_type
    candidates = []
    for path in sorted(root.glob("*.json")):
        label = path.stem
        if label == preferred_label:
            continue
        if label in SIDE_LABELS and not side_label_reachable_for_pose(label, fixed_pose, area_for_scene(scene)):
            continue
        pose = pose_for_object_from_possible(path, object_type)
        if pose is None:
            continue
        candidates.append((label, path, pose))
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} non-preferred possible placement(s) found for "
            f"{object_type} in {scene}/{fixed_type}; need {count}"
        )
    if len(candidates) > count:
        candidates = rng.sample(candidates, count)
    markers = []
    for index, (label, path, pose) in enumerate(candidates[:count], start=1):
        markers.append(
            graspable_object_marker(
                object_type,
                pose,
                variant=f"random_{index}",
                label=label,
                user_id=str(pref.get("user_id", "")) or None,
                fixed_type=fixed_type,
                source_file=str(path),
            )
        )
    return tuple(markers)


def parse_manual_placement(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Manual placement must be fixture:label, got {value!r}")
    fixture, label = value.split(":", 1)
    fixture = fixture.strip()
    label = label.strip()
    if not fixture or not label:
        raise ValueError(f"Manual placement must be fixture:label, got {value!r}")
    return fixture, label


def build_manual_placement_cases(args: argparse.Namespace) -> list[RenderCase]:
    if not args.manual_object:
        raise ValueError("--manual-object is required with --manual-placements")
    scene = (args.scenes or ["KITCHEN_SCENE1"])[0]
    area = area_for_scene(scene)
    object_type = str(args.manual_object)
    cases = []
    for raw_placement in args.manual_placements or []:
        fixed_type, label = parse_manual_placement(raw_placement)
        if scene == "FLOOR_SCENE":
            path = floor_possible_path(args.possible_dir, args.holded_dir, fixed_type, label)
            if path is None:
                raise FileNotFoundError(f"No floor scene placement file for {fixed_type}:{label}")
            pose = pose_for_object_from_possible(path, object_type)
            if pose is None:
                raise ValueError(f"{object_type} has no {fixed_type}:{label} placement in {path}")
            marker = graspable_object_marker(
                object_type,
                pose,
                variant=f"{safe_name(fixed_type)}_{safe_name(label)}",
                label=label,
                fixed_type=fixed_type,
                source_file=str(path),
            )
            fixture_sources = () if fixed_type == FLOOR_FIXTURE else (floor_fixture_pose_source(fixed_type, "center"),)
            output = args.output_dir / "floor" / "FLOOR_SCENE" / "manual_placements" / safe_name(object_type) / f"{safe_name(fixed_type)}__{safe_name(label)}.png"
            cases.append(
                RenderCase(
                    "manual_placement",
                    "floor",
                    "FLOOR_SCENE",
                    ((fixed_type, label),),
                    str(output),
                    fixture_sources,
                    (marker,),
                )
            )
            continue

        path = args.possible_dir / scene / fixed_type / f"{label}.json"
        if not path.exists():
            raise FileNotFoundError(f"No possible placement file: {path}")
        pose = pose_for_object_from_possible(path, object_type)
        if pose is None:
            raise ValueError(f"{object_type} has no possible {fixed_type}:{label} placement in {path}")
        marker = graspable_object_marker(
            object_type,
            pose,
            variant=f"{safe_name(fixed_type)}_{safe_name(label)}",
            label=label,
            fixed_type=fixed_type,
            source_file=str(path),
        )
        fixture_source = fixed_object_source(scene, fixed_type, args.libero_objects)
        if label in SIDE_LABELS and not side_label_reachable_for_pose(label, fixture_source.pose, area):
            raise ValueError(
                f"Manual placement {fixed_type}:{label} is unreachable for pose "
                f"(x={fixture_source.pose.get('x')}, y={fixture_source.pose.get('y')}, h={fixture_source.pose.get('h')})"
            )
        output = args.output_dir / area / scene / "manual_placements" / safe_name(object_type) / f"{safe_name(fixed_type)}__{safe_name(label)}.png"
        cases.append(
            RenderCase(
                "manual_placement",
                area,
                scene,
                ((fixed_type, "center"),),
                str(output),
                (fixture_source,),
                (marker,),
            )
        )
    return cases


def build_profile_graspable_cases(args: argparse.Namespace) -> list[RenderCase]:
    if args.placements_per_item < 3:
        raise ValueError("--placements-per-item must be at least 3")
    rng = random.Random(args.seed)
    payload = read_json(args.profiles)
    profile = choose_profile(payload, args.user_id, rng, args.graspable_objects)
    scene = choose_profile_scene(profile, args.scenes, rng, args.graspable_objects)
    user_id = str(profile["user_id"])
    allowed_objects = set(args.graspable_objects or [])
    cases = []
    for pref in profile.get("placements", []):
        if str(pref.get("scene")) != scene:
            continue
        object_type = str(pref["object_type"])
        if allowed_objects and object_type not in allowed_objects:
            continue
        pref = {**pref, "user_id": user_id}
        area = area_for_scene(scene)
        fixture_source = fixed_object_source(scene, str(pref["fixed_type"]), args.libero_objects)
        preferred_marker = graspable_object_marker(
            object_type,
            pref.get("pose"),
            variant="preferred",
            label=str(pref["label"]),
            user_id=user_id,
            fixed_type=str(pref["fixed_type"]),
            source_file=str(pref.get("source_file", "")) or None,
        )
        base = args.output_dir / area / scene / "profile_graspable" / user_id / safe_name(object_type)
        cases.append(
            RenderCase(
                "profile_preferred",
                area,
                scene,
                ((str(pref["fixed_type"]), "center"),),
                str(base / "preferred.png"),
                (fixture_source,),
                (preferred_marker,),
            )
        )
        for marker in random_possible_markers(
            args.possible_dir,
            pref,
            rng,
            fixture_source.pose,
            count=args.placements_per_item - 1,
        ):
            cases.append(
                RenderCase(
                    "profile_random",
                    area,
                    scene,
                    ((str(pref["fixed_type"]), "center"),),
                    str(base / f"{marker['variant']}.png"),
                    (fixture_source,),
                    (marker,),
                )
            )
    return cases


def build_floor_fixture_cases(args: argparse.Namespace, fixtures: list[str]) -> list[RenderCase]:
    if args.scenes and "FLOOR_SCENE" not in set(args.scenes):
        return []
    cases = []
    for fixture in fixtures:
        for side in SIDES:
            source = floor_fixture_pose_source(fixture, side)
            output = args.output_dir / "floor" / "FLOOR_SCENE" / "single" / f"{fixture}__{side}.png"
            cases.append(RenderCase("single", "floor", "FLOOR_SCENE", ((fixture, side),), str(output), (source,)))
    for left_fixture, right_fixture in itertools.permutations(fixtures, 2):
        if should_skip_multiple_pair(left_fixture, right_fixture):
            continue
        left = floor_fixture_pose_source(left_fixture, "left", multiple=True)
        right = floor_fixture_pose_source(right_fixture, "right", multiple=True)
        output = args.output_dir / "floor" / "FLOOR_SCENE" / "multiple" / f"{left_fixture}__left__{right_fixture}__right.png"
        cases.append(RenderCase("multiple", "floor", "FLOOR_SCENE", ((left_fixture, "left"), (right_fixture, "right")), str(output), (left, right)))
    return cases


def build_cases(args: argparse.Namespace) -> list[RenderCase]:
    requested_fixtures = list(dict.fromkeys(args.fixtures))
    floor_requested = FLOOR_FIXTURE in requested_fixtures or (args.scenes and "FLOOR_SCENE" in set(args.scenes))
    if args.manual_placements:
        return build_manual_placement_cases(args)
    if args.graspable_only:
        if floor_requested:
            return build_floor_cases(args, kind="graspable_only", output_group="graspable_only")
        return build_profile_graspable_cases(args)

    include_floor = floor_requested
    fixtures = [fixture for fixture in requested_fixtures if fixture != FLOOR_FIXTURE]
    pose_fixtures = available_pose_fixtures(args.possible_dir)
    cases = []
    for scene in available_scenes(args.possible_dir, args.scenes):
        area = area_for_scene(scene)
        for fixture in fixtures:
            for side in SIDES:
                source = resolve_pose_source(args.possible_dir, scene, fixture, side, pose_fixtures)
                output = args.output_dir / area / scene / "single" / f"{fixture}__{side}.png"
                cases.append(RenderCase("single", area, scene, ((fixture, side),), str(output), (source,)))
        for left_fixture, right_fixture in itertools.permutations(fixtures, 2):
            if should_skip_multiple_pair(left_fixture, right_fixture):
                continue
            left = resolve_pose_source(args.possible_dir, scene, left_fixture, "left", pose_fixtures)
            right = resolve_pose_source(args.possible_dir, scene, right_fixture, "right", pose_fixtures)
            left, right = adjust_multiple_pose_sources(left, right)
            output = args.output_dir / area / scene / "multiple" / f"{left_fixture}__left__{right_fixture}__right.png"
            cases.append(RenderCase("multiple", area, scene, ((left_fixture, "left"), (right_fixture, "right")), str(output), (left, right)))
    if include_floor and not args.only_left_mirrored:
        cases.extend(build_floor_fixture_cases(args, fixtures))
        cases.extend(build_floor_cases(args))
    if args.only_left_mirrored:
        cases = [
            case
            for case in cases
            if any(source.side == "left" and source.requested_fixture in LEFT_MIRRORED_FIXTURES for source in case.pose_sources)
        ]
    return cases


def write_manifest(path: Path, cases: list[RenderCase], dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": dry_run,
        "case_count": len(cases),
        "single_count": sum(1 for case in cases if case.kind == "single"),
        "multiple_count": sum(1 for case in cases if case.kind == "multiple"),
        "floor_count": sum(1 for case in cases if case.area == "floor" or any(fixture == FLOOR_FIXTURE for fixture, _label in case.fixtures)),
        "cases": [
            {
                **{k: v for k, v in asdict(case).items() if k != "pose_sources"},
                "pose_sources": [asdict(source) for source in case.pose_sources],
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_case(case: RenderCase, camera_name: str, image_size: int) -> None:
    output = Path(case.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    error_path = output.with_suffix(output.suffix + ".error.txt")
    with tempfile.TemporaryDirectory(prefix="vlapb_placement_preview_") as tmp:
        bddl_path = Path(tmp) / f"{safe_name(case.scene)}_{safe_name(output.stem)}.bddl"
        bddl_path.write_text(render_bddl(case.scene, case.pose_sources, case.object_markers), encoding="utf-8")
        render_task_image(
            bddl_file=bddl_path,
            output_path=output,
            camera_name=camera_name,
            image_size=image_size,
            settle_steps=0,
            max_reset_attempts=5,
        )
    if error_path.exists():
        error_path.unlink()


def write_render_error(case: RenderCase, exc: Exception) -> None:
    output = Path(case.output_path)
    error_path = output.with_suffix(output.suffix + ".error.txt")
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        "\n".join(
            [
                f"case={case.kind}",
                f"scene={case.scene}",
                f"fixtures={case.fixtures}",
                f"output={case.output_path}",
                f"error={type(exc).__name__}: {exc}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    cases = build_cases(args)
    filtered_run = bool(args.scenes or args.only_left_mirrored or args.graspable_only or args.fixtures != list(DEFAULT_PREVIEW_FIXTURES))
    manifest_name = "manifest.filtered.json" if filtered_run else "manifest.json"
    write_manifest(args.output_dir / manifest_name, cases, args.dry_run)

    logging.info("scenes: %d", len(set(case.scene for case in cases)))
    logging.info("render cases: %d", len(cases))
    logging.info("single: %d | multiple: %d | floor: %d", sum(1 for case in cases if case.kind == "single"), sum(1 for case in cases if case.kind == "multiple"), sum(1 for case in cases if case.area == "floor" or any(fixture == FLOOR_FIXTURE for fixture, _label in case.fixtures)))
    logging.info("output: %s", args.output_dir)
    logging.info("manifest: %s", args.output_dir / manifest_name)
    if args.dry_run:
        logging.info("dry run enabled; manifest written, no images rendered")
        return

    failures = 0
    for case in tqdm(cases, desc="Rendering placement fixture previews", unit="image"):
        if args.skip_existing and Path(case.output_path).exists():
            continue
        try:
            render_case(case, args.camera_name, args.image_size)
        except Exception as exc:  # noqa: BLE001 - keep batch preview rendering going.
            failures += 1
            logging.warning("render failed: %s -> %s: %s", case.scene, case.output_path, exc)
            write_render_error(case, exc)
    if failures:
        logging.warning("render completed with %d failed cases; see *.error.txt files", failures)


if __name__ == "__main__":
    main()
