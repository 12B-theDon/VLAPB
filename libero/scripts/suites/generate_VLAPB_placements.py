#!/usr/bin/env python3
"""Generate VLAPB placement-suite BDDL and episode metadata.

Input/output:
  Read user profiles from profiles.json and LIBERO object-pair statistics from
  docs/libero_objects.json, then write episode-level .bddl and metadata .json
  files under VLAPB_suites/placements.

Generation:
  Build feasible placement episodes from user placement preferences. The
  generator filters impossible fixture pairs, side-blocked basket/tray regions,
  and internal/surface placements not observed in the original LIBERO dataset.

Hyperparameters:
  Episode budget, seed, output root, fixture list, and seen/unseen split ratio
  are configurable through CLI arguments and VLAPB_config.yaml.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable: Iterable[Any] | None = None, **_: Any) -> None:
            self.iterable = iterable

        def __iter__(self) -> Iterator[Any]:
            return iter(self.iterable or ())

        def update(self, _count: int = 1) -> None:
            return None

        def close(self) -> None:
            return None


SCRIPT_DIR = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = SCRIPT_DIR.parents[1]
VLAPB_PROJECT_ROOT = VLAPB_LIBERO_ROOT.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "VLAPB_config.yaml"
DEFAULT_PROFILES_PATH = VLAPB_LIBERO_ROOT / "profiles" / "profiles.json"
DEFAULT_LIBERO_OBJECTS_PATH = VLAPB_LIBERO_ROOT / "docs" / "libero_objects.json"
DEFAULT_OUTPUT_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"

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
FIXTURE_ALIASES = {
    "cabinet": "wooden_cabinet",
}
OBJECT_SECTION_TYPES = {"basket", "wooden_tray"}
DRAWER_FIXTURES = {"cabinet", "white_cabinet", "wooden_cabinet"}
DRAWER_CONFLICT_FIXTURES = {"microwave", "wooden_two_layer_shelf"}
LEFT_MIRRORED_FIXTURES = {
    "cabinet",
    "microwave",
    "white_cabinet",
    "wooden_cabinet",
    "wooden_two_layer_shelf",
}
MUTUALLY_EXCLUDED_MULTIPLE_PAIRS = {
    frozenset(("microwave", "wooden_two_layer_shelf")),
}
STRUCTURAL_LABELS = {"in", "on", "top", "middle", "bottom"}
SIDE_LABELS = {"left", "right", "front", "back"}
DRAWER_LAYER_LABELS = {"top", "middle", "bottom"}
DRAWER_SIDE_LABEL_RE = re.compile(r"^(top|middle|bottom)_(front|back|left|right)$")
SPLIT_ORDER = ("type1", "type2", "type3", "type4", "adaptability", "multiuser", "consistency")
PROBLEM_BY_AREA = {
    "kitchen": "LIBERO_Kitchen_Tabletop_Manipulation",
    "living_room": "LIBERO_Living_Room_Tabletop_Manipulation",
    "study": "LIBERO_Study_Tabletop_Manipulation",
}
TABLE_BY_AREA = {
    "kitchen": "kitchen_table",
    "living_room": "living_room_table",
    "study": "study_table",
}
SCENE_BY_AREA = {
    "kitchen": "KITCHEN_SCENE1",
    "living_room": "LIVING_ROOM_SCENE1",
    "study": "STUDY_SCENE1",
}
FIXTURE_POSES = {
    "left": {"x": 0.0, "y": -0.30, "h": 0.0},
    "right": {"x": 0.0, "y": 0.30, "h": 0.0},
}
SPECIAL_FIXTURE_POSES = {
    ("flat_stove", "left"): {"x": -0.12, "y": -0.22, "h": 0.0},
    ("flat_stove", "right"): {"x": -0.12, "y": 0.22, "h": 0.0},
    ("basket", "left"): {"x": 0.0, "y": -0.22, "h": 0.0},
    ("basket", "right"): {"x": 0.0, "y": 0.22, "h": 0.0},
    ("wooden_tray", "left"): {"x": 0.0, "y": -0.22, "h": 0.0},
    ("wooden_tray", "right"): {"x": 0.0, "y": 0.22, "h": 0.0},
}
TARGET_INIT_POSE = {"x": -0.155, "y": 0.005, "h": 0.0}
SECONDARY_TARGET_INIT_POSE = {"x": 0.155, "y": 0.005, "h": 0.0}
DISTRACTOR_POSES = (
    {"x": -0.18, "y": 0.0, "h": 0.0},
    {"x": 0.0, "y": 0.0, "h": 0.0},
    {"x": 0.12, "y": -0.12, "h": 0.0},
)
CAMERA_VISIBLE_SIDE_OFFSETS = {
    "left": {"x": 0.0, "y": -0.21},
    "right": {"x": 0.0, "y": 0.21},
    "front": {"x": 0.12, "y": 0.0},
    "back": {"x": -0.12, "y": 0.0},
}
SIDE_GOAL_HALF_SIZE_BY_LABEL = {
    "left": {"x": 0.14, "y": 0.18},
    "right": {"x": 0.14, "y": 0.18},
    "front": {"x": 0.14, "y": 0.20},
    "back": {"x": 0.14, "y": 0.20},
}
SECONDARY_SIDE_GOAL_EXTRA_HALF_SIZE = {"x": 0.02, "y": 0.02}
SIDE_Y_LIMIT_BY_AREA = {
    "kitchen": 0.34,
    "living_room": 0.34,
    "study": 0.34,
}


@dataclass(frozen=True)
class PlacementPreference:
    user_id: str
    object_type: str
    fixed_type: str
    label: str
    location: str
    area: str
    scene: str
    table: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    belongings: tuple[str, ...]
    placements: tuple[PlacementPreference, ...]
    seen_belongings: tuple[str, ...]
    unseen_belongings: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class FixtureSpec:
    fixture: str
    side: str


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    suite: str
    split: str
    task_type: str
    scene: str
    area: str
    table: str
    target_user: str
    target_object: str
    target_object_seen: bool
    preference: dict[str, Any]
    fixtures: tuple[FixtureSpec, ...]
    secondary_user: str | None = None
    secondary_object: str | None = None
    secondary_preference: dict[str, Any] | None = None
    distractor_object: str | None = None
    distractor_pose_index: int | None = None
    participating_users: tuple[str, ...] = ()
    ownership: dict[str, list[str]] | None = None
    adaptation: dict[str, Any] | None = None
    consistency: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--libero-objects", type=Path, default=DEFAULT_LIBERO_OBJECTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixtures", nargs="+", default=list(PLACEMENT_FIXTURES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seen-ratio", type=float, default=0.75)
    parser.add_argument("--object-box-size", type=float, default=0.025)
    parser.add_argument("--fixture-box-size", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", help="Reserved for future rendering/simulation checks.")
    parser.add_argument("--max-episodes-per-split", type=int, default=None)
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format="[%(levelname)s] %(message)s")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logging.warning("config file not found: %s; using default budgets", path)
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read VLAPB_config.yaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_nested_config(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def render_type(fixture: str) -> str:
    return FIXTURE_ALIASES.get(fixture, fixture)


def area_for_scene(scene: str, fallback: str = "kitchen") -> str:
    if scene.startswith("KITCHEN_SCENE"):
        return "kitchen"
    if scene.startswith("LIVING_ROOM_SCENE"):
        return "living_room"
    if scene.startswith("STUDY_SCENE"):
        return "study"
    return fallback


def split_seen_unseen(items: tuple[str, ...], ratio: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(dict.fromkeys(items)))
    if len(ordered) <= 1:
        return ordered, ()
    seen_count = max(1, min(len(ordered) - 1, round(len(ordered) * ratio)))
    return ordered[:seen_count], ordered[seen_count:]


def normalize_preference(raw: dict[str, Any], user_id: str) -> PlacementPreference:
    scene = str(raw.get("scene") or SCENE_BY_AREA.get(str(raw.get("area", "kitchen")), "KITCHEN_SCENE1"))
    area = str(raw.get("area") or area_for_scene(scene))
    table = str(raw.get("table") or TABLE_BY_AREA.get(area, "kitchen_table"))
    return PlacementPreference(
        user_id=user_id,
        object_type=str(raw["object_type"]),
        fixed_type=str(raw["fixed_type"]),
        label=str(raw["label"]),
        location=str(raw.get("location", "")),
        area=area,
        scene=scene,
        table=table,
        raw=raw,
    )


def load_profiles(path: Path, seen_ratio: float) -> list[UserProfile]:
    payload = read_json(path)
    profiles = []
    for raw_profile in payload.get("profiles", []):
        user_id = str(raw_profile["user_id"])
        belongings = tuple(str(item) for item in raw_profile.get("belongings", []))
        if not belongings:
            continue
        seen, unseen = split_seen_unseen(belongings, seen_ratio)
        placements = tuple(normalize_preference(raw, user_id) for raw in raw_profile.get("placements", []))
        profiles.append(
            UserProfile(
                user_id=user_id,
                belongings=belongings,
                placements=placements,
                seen_belongings=seen,
                unseen_belongings=unseen,
                raw=raw_profile,
            )
        )
    if not profiles:
        raise ValueError(f"No profiles with belongings found in {path}")
    return profiles


def ownership_index(profiles: list[UserProfile]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for profile in profiles:
        for item in profile.belongings:
            owners[item].append(profile.user_id)
    return {item: sorted(user_ids) for item, user_ids in owners.items()}


def all_profile_objects(profiles: list[UserProfile]) -> tuple[str, ...]:
    seen = []
    for profile in profiles:
        for item in profile.belongings:
            if item not in seen:
                seen.append(item)
    return tuple(seen)


def read_libero_pair_stats(path: Path) -> dict[tuple[str, str], int]:
    data = read_json(path)
    stats: Counter[tuple[str, str]] = Counter()
    for coupling in data.get("couplings", []):
        fixed = coupling.get("fixed_object", {}).get("type")
        graspable = coupling.get("graspable_object", {}).get("type")
        if fixed and graspable:
            stats[(str(fixed), str(graspable))] += 1
    return dict(stats)


def should_skip_multiple_pair(left_fixture: str, right_fixture: str) -> bool:
    pair = {left_fixture, right_fixture}
    if frozenset(pair) in MUTUALLY_EXCLUDED_MULTIPLE_PAIRS:
        return True
    if left_fixture in DRAWER_FIXTURES and right_fixture in DRAWER_FIXTURES:
        return True
    return bool(pair & DRAWER_FIXTURES and pair & DRAWER_CONFLICT_FIXTURES)


def allowed_other_fixtures(target_fixture: str, fixtures: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(fixture for fixture in fixtures if fixture != target_fixture and not should_skip_multiple_pair(target_fixture, fixture))


def side_allowed_for_preference(pref: PlacementPreference, fixture_side: str) -> bool:
    if pref.label not in SIDE_LABELS:
        return True
    if fixture_side == "left" and pref.label == "left":
        return False
    if fixture_side == "right" and pref.label == "right":
        return False
    return True


def drawer_layer_for_label(label: str) -> str | None:
    if label in DRAWER_LAYER_LABELS:
        return label
    match = DRAWER_SIDE_LABEL_RE.match(label)
    if match:
        return match.group(1)
    return None


def is_drawer_label(label: str) -> bool:
    return drawer_layer_for_label(label) is not None


def is_structural_preference(pref: PlacementPreference) -> bool:
    return pref.label in STRUCTURAL_LABELS or is_drawer_label(pref.label)


def is_preference_feasible(
    pref: PlacementPreference,
    fixture_side: str,
    allowed_fixtures: set[str],
    libero_pair_stats: dict[tuple[str, str], int],
) -> bool:
    if pref.fixed_type not in allowed_fixtures:
        return False
    if is_structural_preference(pref):
        # Structural candidates are already validated by possible_spawn_positions.
        # Do not hard-prune them again by LIBERO pair frequency evidence.
        return True
    if pref.label not in SIDE_LABELS:
        return False
    if not side_allowed_for_preference(pref, fixture_side):
        return False
    return side_label_reachable(pref, fixture_side)


def preference_pair_count(pref: PlacementPreference, libero_pair_stats: dict[tuple[str, str], int]) -> int:
    return libero_pair_stats.get((render_type(pref.fixed_type), pref.object_type), 0)


def relation_for(pref: PlacementPreference) -> str:
    if pref.label in SIDE_LABELS:
        return "AtXY"
    if pref.fixed_type in DRAWER_FIXTURES and (pref.label == "in" or is_drawer_label(pref.label)):
        return "In"
    if pref.label in {"in", "top", "bottom"}:
        return "In"
    return "On"


def goal_region_for(pref: PlacementPreference, fixture_id: str) -> str:
    drawer_layer = drawer_layer_for_label(pref.label)
    if pref.fixed_type in DRAWER_FIXTURES:
        if pref.label == "in":
            return f"{fixture_id}_bottom_region"
        if drawer_layer:
            return f"{fixture_id}_{drawer_layer}_region"
    if pref.label == "in":
        if pref.fixed_type == "microwave":
            return f"{fixture_id}_heating_region"
        return f"{fixture_id}_contain_region"
    if pref.label == "top":
        return f"{fixture_id}_top_region"
    if pref.label == "bottom":
        return f"{fixture_id}_bottom_region"
    if pref.label == "middle":
        return f"{fixture_id}_middle_region"
    if pref.label == "on":
        if pref.fixed_type == "flat_stove":
            return f"{fixture_id}_cook_region"
        if pref.fixed_type in DRAWER_FIXTURES or pref.fixed_type == "wooden_two_layer_shelf":
            return f"{fixture_id}_top_side"
        return fixture_id
    if pref.label in SIDE_LABELS:
        return "placement_goal_region"
    return fixture_id


def fixture_instance_name(fixture: str, side: str) -> str:
    return f"{render_type(fixture)}_{side}"


def fixture_pose(fixture: str, side: str) -> dict[str, float]:
    pose = dict(SPECIAL_FIXTURE_POSES.get((fixture, side), FIXTURE_POSES[side]))
    if side == "left" and fixture in LEFT_MIRRORED_FIXTURES:
        pose["h"] = round((float(pose.get("h", 0.0)) + math.pi + math.pi) % (2 * math.pi) - math.pi, 4)
    return pose


def square_range(center_x: float, center_y: float, half_size: float) -> tuple[float, float, float, float]:
    return (
        round(center_x - half_size, 4),
        round(center_y - half_size, 4),
        round(center_x + half_size, 4),
        round(center_y + half_size, 4),
    )


def rectangular_range(center_x: float, center_y: float, half_x: float, half_y: float) -> tuple[float, float, float, float]:
    return (
        round(center_x - half_x, 4),
        round(center_y - half_y, 4),
        round(center_x + half_x, 4),
        round(center_y + half_y, 4),
    )


def side_goal_center(label: str, fixture_pose_payload: dict[str, float]) -> tuple[float, float]:
    # Viewer-centric label policy:
    # Front/back/left/right are anchored to camera/world axes and translated by
    # the fixture's current position. They do not rotate with fixture heading.
    offset = CAMERA_VISIBLE_SIDE_OFFSETS[label]
    return fixture_pose_payload["x"] + offset["x"], fixture_pose_payload["y"] + offset["y"]


def side_label_reachable(pref: PlacementPreference, fixture_side: str) -> bool:
    if pref.label not in SIDE_LABELS:
        return True
    pose = fixture_pose(pref.fixed_type, fixture_side)
    _, goal_y = side_goal_center(pref.label, pose)
    y_limit = SIDE_Y_LIMIT_BY_AREA.get(pref.area, 0.34)
    if abs(goal_y) > y_limit:
        return False

    fixture_y = float(pose.get("y", 0.0))
    if abs(fixture_y) < 1e-6:
        return True

    same_side = goal_y * fixture_y > 0.0
    farther_outward = abs(goal_y) > abs(fixture_y) + 1e-3
    if same_side and farther_outward:
        return False
    return True


def camera_visible_side_goal_range(
    label: str,
    fixture_pose_payload: dict[str, float],
    *,
    secondary: bool = False,
) -> tuple[float, float, float, float]:
    center_x, center_y = side_goal_center(label, fixture_pose_payload)
    half_size = SIDE_GOAL_HALF_SIZE_BY_LABEL.get(label, {"x": 0.07, "y": 0.08})
    if secondary:
        half_size = {
            "x": half_size["x"] + SECONDARY_SIDE_GOAL_EXTRA_HALF_SIZE["x"],
            "y": half_size["y"] + SECONDARY_SIDE_GOAL_EXTRA_HALF_SIZE["y"],
        }
    return rectangular_range(
        center_x,
        center_y,
        half_size["x"],
        half_size["y"],
    )


def render_region(
    name: str,
    target: str,
    ranges: tuple[float, float, float, float] | None = None,
    yaw: float | None = None,
) -> str:
    if ranges is None:
        return f"""      ({name}
          (:target {target})
      )"""
    yaw_block = ""
    if yaw is not None:
        yaw_block = f"""
          (:yaw_rotation (
              ({yaw:.4f} {yaw:.4f})
            )
          )"""
    x1, y1, x2, y2 = ranges
    return f"""      ({name}
          (:target {target})
          (:ranges (
              ({x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f})
            )
          ){yaw_block}
      )"""


def grouped_declarations(id_type_pairs: list[tuple[str, str]]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for object_id, object_type in id_type_pairs:
        grouped[object_type].append(object_id)
    return [f"    {' '.join(ids)} - {object_type}" for object_type, ids in sorted(grouped.items())]


def split_budget(max_counts: dict[str, int], total_budget: int | None) -> dict[str, int]:
    if total_budget is None:
        return dict(max_counts)
    if total_budget < 0:
        raise ValueError(f"placements.total_episodes must be non-negative, got {total_budget}")
    positive = {split: count for split, count in max_counts.items() if count > 0}
    if not positive:
        return dict(max_counts)
    total_available = sum(positive.values())
    planned = {split: 0 for split in max_counts}
    remaining_budget = min(total_budget, total_available)
    raw = {
        split: (count / total_available) * remaining_budget
        for split, count in positive.items()
    }
    for split, value in raw.items():
        planned[split] = min(max_counts[split], int(value))
    remainder = remaining_budget - sum(planned.values())
    fractions = sorted(
        positive,
        key=lambda split: (raw[split] - int(raw[split]), max_counts[split]),
        reverse=True,
    )
    while remainder > 0 and fractions:
        progressed = False
        for split in fractions:
            if planned[split] < max_counts[split]:
                planned[split] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            break
    return planned


def sample_specs(specs: list[EpisodeSpec], count: int, rng: random.Random) -> list[EpisodeSpec]:
    if count >= len(specs):
        return specs
    by_user: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        by_user[spec.target_user].append(spec)
    for bucket in by_user.values():
        rng.shuffle(bucket)
    ordered_users = sorted(by_user)
    sampled = []
    while len(sampled) < count and any(by_user.values()):
        for user_id in ordered_users:
            if by_user[user_id]:
                sampled.append(by_user[user_id].pop())
                if len(sampled) == count:
                    break
    return sampled


def preference_payload(pref: PlacementPreference) -> dict[str, Any]:
    payload = {
        "object_type": pref.object_type,
        "fixed_type": pref.fixed_type,
        "label": pref.label,
        "location": pref.location,
        "area": pref.area,
        "scene": pref.scene,
        "table": pref.table,
    }
    if pref.label in SIDE_LABELS:
        payload["goal_region_policy"] = "camera-visible side label encoded as explicit table placement_goal_region"
        payload["goal_predicate"] = "AtXY"
    return payload


def make_episode(
    index: int,
    split: str,
    task_type: str,
    pref: PlacementPreference,
    target_seen: bool,
    fixtures: tuple[FixtureSpec, ...],
    ownership: dict[str, list[str]],
    secondary_user: str | None = None,
    secondary_object: str | None = None,
    secondary_preference: PlacementPreference | None = None,
    distractor_object: str | None = None,
    distractor_pose_index: int | None = None,
    participating_users: tuple[str, ...] = (),
    adaptation: dict[str, Any] | None = None,
    consistency: dict[str, Any] | None = None,
    debug: dict[str, Any] | None = None,
) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=f"placements_{split}_{index:06d}",
        suite="placements",
        split=split,
        task_type=task_type,
        scene=pref.scene,
        area=pref.area,
        table=pref.table,
        target_user=pref.user_id,
        target_object=pref.object_type,
        target_object_seen=target_seen,
        preference=preference_payload(pref),
        fixtures=fixtures,
        secondary_user=secondary_user,
        secondary_object=secondary_object,
        secondary_preference=preference_payload(secondary_preference) if secondary_preference else None,
        distractor_object=distractor_object,
        distractor_pose_index=distractor_pose_index,
        participating_users=participating_users or (pref.user_id,),
        ownership=ownership,
        adaptation=adaptation,
        consistency=consistency,
        debug=debug,
    )


def build_type1_specs(
    profiles: list[UserProfile],
    allowed_fixtures: set[str],
    pair_stats: dict[tuple[str, str], int],
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        seen = set(profile.seen_belongings)
        for pref in profile.placements:
            target_seen = pref.object_type in seen
            if not target_seen:
                continue
            for side in ("left", "right"):
                if not is_preference_feasible(pref, side, allowed_fixtures, pair_stats):
                    continue
                specs.append(
                    make_episode(
                        index,
                        "type1",
                        "type1",
                        pref,
                        target_seen,
                        (FixtureSpec(pref.fixed_type, side),),
                        {profile.user_id: [pref.object_type]},
                        debug={"libero_pair_count": preference_pair_count(pref, pair_stats)},
                    )
                )
                index += 1
    return specs


def build_type2_specs(
    profiles: list[UserProfile],
    allowed_fixtures: tuple[str, ...],
    pair_stats: dict[tuple[str, str], int],
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    allowed_set = set(allowed_fixtures)
    for profile in profiles:
        seen = set(profile.seen_belongings)
        for pref in profile.placements:
            target_seen = pref.object_type in seen
            if not target_seen:
                continue
            for target_side in ("left", "right"):
                if not is_preference_feasible(pref, target_side, allowed_set, pair_stats):
                    continue
                other_side = "right" if target_side == "left" else "left"
                for other_fixture in allowed_other_fixtures(pref.fixed_type, allowed_fixtures):
                    specs.append(
                        make_episode(
                            index,
                            "type2",
                            "type2",
                            pref,
                            target_seen,
                            (FixtureSpec(pref.fixed_type, target_side), FixtureSpec(other_fixture, other_side)),
                            {profile.user_id: [pref.object_type]},
                            debug={"libero_pair_count": preference_pair_count(pref, pair_stats)},
                        )
                    )
                    index += 1
    return specs


def build_type3_specs(
    profiles: list[UserProfile],
    allowed_fixtures: tuple[str, ...],
    object_pool: tuple[str, ...],
    pair_stats: dict[tuple[str, str], int],
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    base_specs = build_type2_specs(profiles, allowed_fixtures, pair_stats)
    for base in base_specs:
        profile = next(profile for profile in profiles if profile.user_id == base.target_user)
        excluded = set(profile.belongings) | {base.target_object}
        distractors = [item for item in object_pool if item not in excluded]
        for pose_index, distractor in enumerate(distractors[: len(DISTRACTOR_POSES)]):
            specs.append(
                make_episode(
                    index,
                    "type3",
                    "type3",
                    PlacementPreference(
                        user_id=base.target_user,
                        object_type=base.target_object,
                        fixed_type=base.preference["fixed_type"],
                        label=base.preference["label"],
                        location=base.preference["location"],
                        area=base.area,
                        scene=base.scene,
                        table=base.table,
                        raw=base.preference,
                    ),
                    base.target_object_seen,
                    base.fixtures,
                    base.ownership or {base.target_user: [base.target_object]},
                    distractor_object=distractor,
                    distractor_pose_index=pose_index,
                    debug=base.debug,
                )
            )
            index += 1
    return specs


def build_type4_specs(type2_specs: list[EpisodeSpec]) -> list[EpisodeSpec]:
    specs = []
    for index, base in enumerate(type2_specs, start=1):
        pref = PlacementPreference(
            user_id=base.target_user,
            object_type=base.target_object,
            fixed_type=base.preference["fixed_type"],
            label=base.preference["label"],
            location=base.preference["location"],
            area=base.area,
            scene=base.scene,
            table=base.table,
            raw=base.preference,
        )
        fixture_state = fixture_state_for(pref)
        specs.append(
            make_episode(
                index,
                "type4",
                "type4",
                pref,
                base.target_object_seen,
                base.fixtures,
                base.ownership or {base.target_user: [base.target_object]},
                debug={**(base.debug or {}), "fixture_state": fixture_state},
            )
        )
    return specs


def build_adaptability_specs(
    profiles: list[UserProfile],
    allowed_fixtures: set[str],
    pair_stats: dict[tuple[str, str], int],
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        feasible = [
            pref
            for pref in profile.placements
            if pref.object_type in set(profile.seen_belongings)
            and any(is_preference_feasible(pref, side, allowed_fixtures, pair_stats) for side in ("left", "right"))
        ]
        for old_pref, new_pref in itertools.permutations(feasible, 2):
            if old_pref.fixed_type == new_pref.fixed_type and old_pref.label == new_pref.label:
                continue
            for side in ("left", "right"):
                if not is_preference_feasible(new_pref, side, allowed_fixtures, pair_stats):
                    continue
                specs.append(
                    make_episode(
                        index,
                        "adaptability",
                        "adaptability",
                        new_pref,
                        True,
                        (FixtureSpec(new_pref.fixed_type, side),),
                        {profile.user_id: [new_pref.object_type]},
                        adaptation={
                            "user_id": profile.user_id,
                            "previous_placement": preference_payload(old_pref),
                            "updated_placement": preference_payload(new_pref),
                        },
                        debug={"libero_pair_count": preference_pair_count(new_pref, pair_stats)},
                    )
                )
                index += 1
    return specs


def build_multiuser_specs(
    profiles: list[UserProfile],
    allowed_fixtures: tuple[str, ...],
    pair_stats: dict[tuple[str, str], int],
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    allowed_set = set(allowed_fixtures)
    for user_a, user_b in itertools.permutations(profiles, 2):
        prefs_a = [pref for pref in user_a.placements if pref.object_type in set(user_a.seen_belongings)]
        prefs_b = [pref for pref in user_b.placements if pref.object_type in set(user_b.seen_belongings)]
        for pref_a in prefs_a:
            for pref_b in prefs_b:
                if pref_a.fixed_type == pref_b.fixed_type:
                    continue
                if should_skip_multiple_pair(pref_a.fixed_type, pref_b.fixed_type):
                    continue
                if not is_preference_feasible(pref_a, "left", allowed_set, pair_stats):
                    continue
                if not is_preference_feasible(pref_b, "right", allowed_set, pair_stats):
                    continue
                shared_pref_b = PlacementPreference(
                    user_id=pref_b.user_id,
                    object_type=pref_b.object_type,
                    fixed_type=pref_b.fixed_type,
                    label=pref_b.label,
                    location=pref_b.location,
                    area=pref_a.area,
                    scene=pref_a.scene,
                    table=pref_a.table,
                    raw={**pref_b.raw, "area": pref_a.area, "scene": pref_a.scene, "table": pref_a.table},
                )
                specs.append(
                    make_episode(
                        index,
                        "multiuser",
                        "multiuser",
                        pref_a,
                        True,
                        (FixtureSpec(pref_a.fixed_type, "left"), FixtureSpec(pref_b.fixed_type, "right")),
                        {user_a.user_id: [pref_a.object_type], user_b.user_id: [pref_b.object_type]},
                        secondary_user=user_b.user_id,
                        secondary_object=pref_b.object_type,
                        secondary_preference=shared_pref_b,
                        participating_users=(user_a.user_id, user_b.user_id),
                        debug={
                            "libero_pair_count": preference_pair_count(pref_a, pair_stats),
                        },
                    )
                )
                index += 1
    return specs


def build_consistency_specs(
    profiles: list[UserProfile],
    allowed_fixtures: set[str],
    pair_stats: dict[tuple[str, str], int],
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        unseen = set(profile.unseen_belongings)
        for pref in profile.placements:
            if pref.object_type not in unseen:
                continue
            for side in ("left", "right"):
                if not is_preference_feasible(pref, side, allowed_fixtures, pair_stats):
                    continue
                specs.append(
                    make_episode(
                        index,
                        "consistency",
                        "consistency",
                        pref,
                        False,
                        (FixtureSpec(pref.fixed_type, side),),
                        {profile.user_id: [pref.object_type]},
                        consistency={
                            "user_id": profile.user_id,
                            "unseen_object": pref.object_type,
                            "seen_objects": list(profile.seen_belongings),
                            "rule": "consistency uses only user-local unseen belongings",
                        },
                        debug={"libero_pair_count": preference_pair_count(pref, pair_stats)},
                    )
                )
                index += 1
    return specs


def fixture_state_for(pref: PlacementPreference) -> dict[str, str]:
    if pref.fixed_type == "microwave":
        return {"microwave_door": "open"}
    if pref.fixed_type in DRAWER_FIXTURES:
        drawer_layer = drawer_layer_for_label(pref.label)
        if pref.label == "in":
            return {"bottom_drawer": "open"}
        if drawer_layer:
            return {f"{drawer_layer}_drawer": "open"}
        if pref.label == "on":
            return {"drawer_or_door": "closed"}
        if pref.label == "front":
            return {"drawer_or_door": "closed"}
        return {"drawer_or_door": "open"}
    return {}


def language_for(spec: EpisodeSpec) -> str:
    pref = spec.preference
    place = pref["label"]
    fixture = pref["fixed_type"]
    if spec.secondary_user and spec.secondary_object and spec.secondary_preference:
        secondary_place = spec.secondary_preference["label"]
        secondary_fixture = spec.secondary_preference["fixed_type"]
        return (
            f"For {spec.target_user}, pick the {spec.target_object} and place it at the {place} of the {fixture}, "
            f"for {spec.secondary_user}, pick the {spec.secondary_object} and place it at the {secondary_place} "
            f"of the {secondary_fixture}."
        )
    return (
        f"Pick the {spec.target_object} belonging to {spec.target_user} "
        f"and place it at the {place} of the {fixture}"
    )


def object_ids(types: list[str]) -> list[tuple[str, str]]:
    counts: Counter[str] = Counter()
    ids = []
    for object_type in types:
        counts[object_type] += 1
        ids.append((f"{object_type}_{counts[object_type]}", object_type))
    return ids


def render_bddl(spec: EpisodeSpec, object_box_size: float, fixture_box_size: float) -> str:
    problem = PROBLEM_BY_AREA.get(spec.area, "LIBERO_Tabletop_Manipulation")
    object_types = [spec.target_object]
    if spec.secondary_object:
        object_types.append(spec.secondary_object)
    if spec.distractor_object:
        object_types.append(spec.distractor_object)
    object_id_pairs = object_ids(object_types)
    target_id = object_id_pairs[0][0]
    secondary_id = object_id_pairs[1][0] if spec.secondary_object else None

    fixture_instances = [(fixture, fixture_instance_name(fixture.fixture, fixture.side)) for fixture in spec.fixtures]
    target_fixture = spec.fixtures[0]
    target_fixture_id = fixture_instance_name(target_fixture.fixture, target_fixture.side)
    secondary_fixture = spec.fixtures[1] if spec.secondary_object and len(spec.fixtures) > 1 else None
    secondary_fixture_id = fixture_instance_name(secondary_fixture.fixture, secondary_fixture.side) if secondary_fixture else None

    regions = []
    fixture_lines = [f"    {spec.table} - {spec.table}"]
    object_lines = grouped_declarations(object_id_pairs)
    init_lines = []

    for fixture, instance in fixture_instances:
        pose = fixture_pose(fixture.fixture, fixture.side)
        region = f"{instance}_{fixture.side}_region"
        regions.append(
            render_region(
                region,
                spec.table,
                square_range(pose["x"], pose["y"], fixture_box_size),
                yaw=pose.get("h", 0.0),
            )
        )
        init_lines.append(f"    (On {instance} {spec.table}_{region})")
        fixture_type = render_type(fixture.fixture)
        if fixture_type in OBJECT_SECTION_TYPES:
            object_lines.append(f"    {instance} - {fixture_type}")
        else:
            fixture_lines.append(f"    {instance} - {fixture_type}")

    regions.append(
        render_region(
            "target_object_init_region",
            spec.table,
            square_range(TARGET_INIT_POSE["x"], TARGET_INIT_POSE["y"], object_box_size),
        )
    )
    init_lines.append(f"    (On {target_id} {spec.table}_target_object_init_region)")

    if spec.secondary_object:
        regions.append(
            render_region(
                "secondary_object_init_region",
                spec.table,
                square_range(SECONDARY_TARGET_INIT_POSE["x"], SECONDARY_TARGET_INIT_POSE["y"], object_box_size),
            )
        )
        init_lines.append(f"    (On {secondary_id} {spec.table}_secondary_object_init_region)")

    if spec.distractor_object:
        pose = DISTRACTOR_POSES[spec.distractor_pose_index or 0]
        distractor_id = object_id_pairs[-1][0]
        regions.append(
            render_region(
                "distractor_object_init_region",
                spec.table,
                square_range(pose["x"], pose["y"], object_box_size),
            )
        )
        init_lines.append(f"    (On {distractor_id} {spec.table}_distractor_object_init_region)")

    pref = PlacementPreference(
        user_id=spec.target_user,
        object_type=spec.target_object,
        fixed_type=spec.preference["fixed_type"],
        label=spec.preference["label"],
        location=spec.preference["location"],
        area=spec.area,
        scene=spec.scene,
        table=spec.table,
        raw=spec.preference,
    )
    goal_target = goal_region_for(pref, target_fixture_id)
    if pref.label in SIDE_LABELS:
        goal_region_name = goal_target
        target_pose = fixture_pose(target_fixture.fixture, target_fixture.side)
        regions.append(
            render_region(
                goal_region_name,
                spec.table,
                camera_visible_side_goal_range(pref.label, target_pose),
            )
        )
        goal_target = f"{spec.table}_{goal_region_name}"
    elif "_" in goal_target and goal_target != target_fixture_id:
        regions.append(render_region(goal_target.split(f"{target_fixture_id}_", 1)[-1], target_fixture_id))
    relation = relation_for(pref)
    goal_items = [f"({relation} {target_id} {goal_target})"]

    if spec.secondary_object and spec.secondary_preference and secondary_fixture and secondary_fixture_id:
        secondary_pref = PlacementPreference(
            user_id=spec.secondary_user or "",
            object_type=spec.secondary_object,
            fixed_type=spec.secondary_preference["fixed_type"],
            label=spec.secondary_preference["label"],
            location=spec.secondary_preference["location"],
            area=spec.area,
            scene=spec.scene,
            table=spec.table,
            raw=spec.secondary_preference,
        )
        secondary_goal_target = goal_region_for(secondary_pref, secondary_fixture_id)
        if secondary_pref.label in SIDE_LABELS:
            secondary_goal_region_name = "secondary_placement_goal_region"
            target_pose = fixture_pose(secondary_fixture.fixture, secondary_fixture.side)
            regions.append(
                render_region(
                    secondary_goal_region_name,
                    spec.table,
                    camera_visible_side_goal_range(secondary_pref.label, target_pose, secondary=True),
                )
            )
            secondary_goal_target = f"{spec.table}_{secondary_goal_region_name}"
        elif "_" in secondary_goal_target and secondary_goal_target != secondary_fixture_id:
            regions.append(render_region(secondary_goal_target.split(f"{secondary_fixture_id}_", 1)[-1], secondary_fixture_id))
        goal_items.append(f"({relation_for(secondary_pref)} {secondary_id} {secondary_goal_target})")

    goal_line = f"    (And {' '.join(goal_items)})"

    return f"""(define (problem {problem})
  (:domain robosuite)
  (:language {language_for(spec)})
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
    {target_id}{f' {secondary_id}' if secondary_id else ''}
    {target_fixture_id}{f' {secondary_fixture_id}' if secondary_fixture_id else ''}
  )

  (:init
{chr(10).join(init_lines)}
  )

  (:goal
{goal_line}
  )

)
"""


def metadata_for(spec: EpisodeSpec, profiles_path: Path, libero_objects_path: Path, bddl_path: Path) -> dict[str, Any]:
    return {
        **asdict(spec),
        "fixtures": [asdict(fixture) for fixture in spec.fixtures],
        "language": language_for(spec),
        "profile_source": str(profiles_path),
        "libero_objects_source": str(libero_objects_path),
        "bddl_file": str(bddl_path),
        "constraints": {
            "allowed_fixtures": list(PLACEMENT_FIXTURES),
            "fixture_pair_policy": "drawer-drawer, drawer-shelf, drawer-microwave, and shelf-microwave pairs are skipped",
            "structural_pair_policy": "in/on/top/middle/bottom and drawer-side placements require validated spawn-position evidence",
            "side_policy": "left-side fixtures block left placement; right-side fixtures block right placement",
            "side_goal_policy": "left/right/front/back labels use camera-visible explicit table placement_goal_region with AtXY predicate",
            "seen_unseen_policy": "type/adaptability/multiuser use seen belongings; consistency uses unseen belongings only",
        },
    }


def output_paths(output_root: Path, spec: EpisodeSpec) -> tuple[Path, Path]:
    base = output_root / "placements" / spec.split
    return (
        base / "bddl_files" / f"{spec.episode_id}.bddl",
        base / "metadata" / f"{spec.episode_id}.json",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_fixtures(fixtures: list[str]) -> tuple[str, ...]:
    unknown = sorted(set(fixtures) - set(PLACEMENT_FIXTURES))
    if unknown:
        raise ValueError(f"Unsupported placement fixtures: {unknown}. Valid fixtures: {sorted(PLACEMENT_FIXTURES)}")
    return tuple(fixtures)


def build_all_specs(
    profiles: list[UserProfile],
    fixtures: tuple[str, ...],
    object_pool: tuple[str, ...],
    pair_stats: dict[tuple[str, str], int],
) -> dict[str, list[EpisodeSpec]]:
    allowed_set = set(fixtures)
    type1 = build_type1_specs(profiles, allowed_set, pair_stats)
    type2 = build_type2_specs(profiles, fixtures, pair_stats)
    return {
        "type1": type1,
        "type2": type2,
        "type3": build_type3_specs(profiles, fixtures, object_pool, pair_stats),
        "type4": build_type4_specs(type2),
        "adaptability": build_adaptability_specs(profiles, allowed_set, pair_stats),
        "multiuser": build_multiuser_specs(profiles, fixtures, pair_stats),
        "consistency": build_consistency_specs(profiles, allowed_set, pair_stats),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    config = read_config(args.config)
    fixtures = validate_fixtures(args.fixtures)
    profiles = load_profiles(args.profiles, args.seen_ratio)
    pair_stats = read_libero_pair_stats(args.libero_objects)
    object_pool = all_profile_objects(profiles)
    object_owners = ownership_index(profiles)
    specs_by_split = build_all_specs(profiles, fixtures, object_pool, pair_stats)
    counts = {split: len(specs_by_split[split]) for split in SPLIT_ORDER}

    configured_total = get_nested_config(config, ("placements", "total_episodes"))
    if configured_total is not None:
        configured_total = int(configured_total)
    planned_counts = split_budget(counts, configured_total)
    if args.max_episodes_per_split is not None:
        planned_counts = {split: min(count, args.max_episodes_per_split) for split, count in planned_counts.items()}
    planned_total = sum(planned_counts.values())

    rng = random.Random(args.seed)
    planned_specs: dict[str, list[EpisodeSpec]] = {}
    for offset, split in enumerate(SPLIT_ORDER):
        planned_specs[split] = sample_specs(specs_by_split[split], planned_counts[split], random.Random(args.seed + offset + 1))

    logging.info("profiles: %d users", len(profiles))
    logging.info("object pool: %d unique objects from profiles.json", len(object_pool))
    logging.info("fixtures: %s", ", ".join(fixtures))
    logging.info("output root: %s", args.output_root)
    logging.info("seed: %d | seen_ratio: %.2f | gpu: %s", args.seed, args.seen_ratio, args.use_gpu)
    logging.info("libero objects: %s", args.libero_objects)
    logging.info("user seen/unseen counts: %s", {p.user_id: [len(p.seen_belongings), len(p.unseen_belongings)] for p in profiles})
    logging.info("episode counts by split: %s", counts)
    logging.info("total episodes possible: %d", sum(counts.values()))
    logging.info("config: %s", args.config)
    if configured_total is not None:
        logging.info("configured placements.total_episodes: %d", configured_total)
    logging.info("planned episodes: %s", planned_counts)
    logging.info("planned total episodes: %d", planned_total)
    if args.max_episodes_per_split is not None:
        logging.info("debug cap enabled: max %d episodes per split", args.max_episodes_per_split)

    if args.dry_run:
        logging.info("dry run enabled; no files written")
        return

    summary = {
        "suite": "placements",
        "profiles": str(args.profiles),
        "config": str(args.config),
        "libero_objects": str(args.libero_objects),
        "output_root": str(args.output_root),
        "counts": counts,
        "total": sum(counts.values()),
        "planned_counts": planned_counts,
        "planned_total": planned_total,
        "configured_total_episodes": configured_total,
        "user_seen_unseen_counts": {p.user_id: {"seen": len(p.seen_belongings), "unseen": len(p.unseen_belongings)} for p in profiles},
        "hyperparameters": {
            "fixtures": fixtures,
            "seed": args.seed,
            "seen_ratio": args.seen_ratio,
            "object_box_size": args.object_box_size,
            "fixture_box_size": args.fixture_box_size,
            "use_gpu": args.use_gpu,
            "max_episodes_per_split": args.max_episodes_per_split,
            "fixture_pair_policy": {
                "drawer_fixtures": sorted(DRAWER_FIXTURES),
                "drawer_conflict_fixtures": sorted(DRAWER_CONFLICT_FIXTURES),
                "mutually_excluded_pairs": [sorted(pair) for pair in MUTUALLY_EXCLUDED_MULTIPLE_PAIRS],
            },
            "object_owners": object_owners,
        },
    }

    progress = tqdm(total=planned_total, desc="Generating placement episodes", unit="episode")
    written = 0
    for split in SPLIT_ORDER:
        for spec in planned_specs[split]:
            bddl_path, metadata_path = output_paths(args.output_root, spec)
            write_text(bddl_path, render_bddl(spec, args.object_box_size, args.fixture_box_size))
            write_json(metadata_path, metadata_for(spec, args.profiles, args.libero_objects, bddl_path))
            written += 1
            progress.update(1)
            if args.debug and written <= 5:
                logging.debug("sample episode %s: %s", spec.episode_id, asdict(spec))
    progress.close()
    summary["written"] = written
    write_json(args.output_root / "placements" / "generation_summary.json", summary)
    logging.info("written episodes: %d", written)
    logging.info("summary: %s", args.output_root / "placements" / "generation_summary.json")

    # Keep rng referenced so deterministic behavior stays explicit if future
    # randomized builders are added.
    _ = rng


if __name__ == "__main__":
    main()
