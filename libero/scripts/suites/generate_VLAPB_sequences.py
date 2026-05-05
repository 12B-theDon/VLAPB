#!/usr/bin/env python3
"""Generate VLAPB sequence-suite BDDL and episode metadata.

Input/output:
  Read user profiles from profiles.json and graspable object types from
  docs/libero_objects.json, then write episode-level .bddl and metadata .json
  files under VLAPB_suites/sequences.

Generation:
  Spawn exactly three graspable objects around a wooden_tray. The BDDL goal
  checks only the final tray state, while metadata records the required
  user-specific target sequence for temporal evaluation.

Hyperparameters:
  Episode budget, seed, output root, spawn radius, and seen/unseen split are
  configurable through CLI arguments and VLAPB_config.yaml.
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

OBJECTS_PER_EPISODE = 3
FIXTURE = "wooden_tray"
RELATION = "In"
SPLIT_ORDER = ("type1", "type2", "adaptability", "consistency")
DEFAULT_SCENE_VARIANTS = (
    ("KITCHEN_SCENE1", "kitchen_table"),
    ("KITCHEN_SCENE5", "kitchen_table"),
    ("LIVING_ROOM_SCENE1", "living_room_table"),
    ("LIVING_ROOM_SCENE6", "living_room_table"),
    ("STUDY_SCENE1", "study_table"),
    ("STUDY_SCENE3", "study_table"),
)
ADAPTATION_STRATEGY_PAIRS = {
    "alphabetical": "reverse_alphabetical",
    "reverse_alphabetical": "alphabetical",
    "fixed_near_to_far": "fixed_far_to_near",
    "fixed_far_to_near": "fixed_near_to_far",
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
    "odd_positions_first": "even_positions_first",
    "even_positions_first": "odd_positions_first",
}
LARGE_FOOTPRINT_OBJECTS = {"chefmate_8_frypan"}
LARGE_FOOTPRINT_RADIUS_SCALE = 1.55


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    belongings: tuple[str, ...]
    sequence_strategy: str
    seen_belongings: tuple[str, ...]
    unseen_belongings: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class SceneVariant:
    scene: str
    table: str


@dataclass(frozen=True)
class SpawnPoint:
    object_type: str
    x: float
    y: float
    distance_to_tray: float
    ordinal_position: int


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    suite: str
    split: str
    task_type: str
    scene: str
    table: str
    fixture: str
    relation: str
    target_user: str
    sequence_strategy: str
    target_sequence: tuple[str, ...]
    graspable_objects: tuple[str, ...]
    spawn_points: tuple[SpawnPoint, ...]
    ownership: dict[str, list[str]]
    participating_users: tuple[str, ...]
    object_seen_flags: dict[str, bool] | None = None
    adaptation: dict[str, Any] | None = None
    consistency: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--libero-objects", type=Path, default=DEFAULT_LIBERO_OBJECTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seen-ratio", type=float, default=0.75)
    parser.add_argument("--spawn-radius", type=float, default=0.24)
    parser.add_argument("--object-box-size", type=float, default=0.025)
    parser.add_argument("--fixture-box-size", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", help="Reserved for future rendering/simulation checks.")
    parser.add_argument("--max-episodes-per-split", type=int, default=None)
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format="[%(levelname)s] %(message)s")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


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


def split_seen_unseen(items: tuple[str, ...], ratio: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(dict.fromkeys(items)))
    if len(ordered) <= 1:
        return ordered, ()
    seen_count = max(1, min(len(ordered) - 1, round(len(ordered) * ratio)))
    return ordered[:seen_count], ordered[seen_count:]


def load_profiles(path: Path, seen_ratio: float) -> list[UserProfile]:
    data = read_json(path)
    profiles = []
    for raw_profile in data.get("profiles", []):
        user_id = str(raw_profile["user_id"])
        belongings = tuple(str(item) for item in raw_profile.get("belongings", []))
        strategy = str(raw_profile.get("placement_order_strategy", "alphabetical"))
        if len(set(belongings)) < OBJECTS_PER_EPISODE:
            continue
        seen, unseen = split_seen_unseen(belongings, seen_ratio)
        profiles.append(
            UserProfile(
                user_id=user_id,
                belongings=belongings,
                sequence_strategy=strategy,
                seen_belongings=seen,
                unseen_belongings=unseen,
                raw=raw_profile,
            )
        )
    if not profiles:
        raise ValueError(f"No profiles with at least {OBJECTS_PER_EPISODE} belongings found in {path}")
    return profiles


def load_libero_graspable_types(path: Path) -> tuple[str, ...]:
    data = read_json(path)
    types = []
    seen = set()
    for item in data.get("graspable_objects", []):
        object_type = str(item.get("type", ""))
        if object_type and object_type not in seen:
            seen.add(object_type)
            types.append(object_type)
    if not types:
        raise ValueError(f"No graspable objects found in {path}")
    return tuple(types)


def all_profile_objects(profiles: list[UserProfile]) -> tuple[str, ...]:
    seen = []
    for profile in profiles:
        for item in profile.belongings:
            if item not in seen:
                seen.append(item)
    return tuple(seen)


def ownership_index(profiles: list[UserProfile]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for profile in profiles:
        for item in profile.belongings:
            owners[item].append(profile.user_id)
    return {item: sorted(user_ids) for item, user_ids in owners.items()}


def choose_scene_variant(index: int) -> SceneVariant:
    scene, table = DEFAULT_SCENE_VARIANTS[(index - 1) % len(DEFAULT_SCENE_VARIANTS)]
    return SceneVariant(scene=scene, table=table)


def circular_spawn_points(
    objects: tuple[str, ...],
    radius: float,
    rng: random.Random,
) -> tuple[SpawnPoint, ...]:
    base_angles = [math.radians(angle) for angle in (70, 180, 300)]
    rotation = rng.uniform(-0.20, 0.20)
    points = []
    for ordinal, (object_type, angle) in enumerate(zip(objects, base_angles), start=1):
        jittered_radius = radius * rng.uniform(0.78, 1.10)
        if object_type in LARGE_FOOTPRINT_OBJECTS:
            jittered_radius *= LARGE_FOOTPRINT_RADIUS_SCALE
        x = round(jittered_radius * math.cos(angle + rotation), 4)
        y = round(jittered_radius * math.sin(angle + rotation), 4)
        points.append(
            SpawnPoint(
                object_type=object_type,
                x=x,
                y=y,
                distance_to_tray=round(math.hypot(x, y), 4),
                ordinal_position=ordinal,
            )
        )
    return tuple(points)


def order_objects(
    objects: tuple[str, ...],
    strategy: str,
    spawn_points: tuple[SpawnPoint, ...],
) -> tuple[str, ...]:
    position_by_object = {point.object_type: point for point in spawn_points}
    if strategy == "alphabetical":
        return tuple(sorted(objects))
    if strategy == "reverse_alphabetical":
        return tuple(sorted(objects, reverse=True))
    if strategy == "fixed_near_to_far":
        return tuple(sorted(objects, key=lambda item: (position_by_object[item].distance_to_tray, item)))
    if strategy == "fixed_far_to_near":
        return tuple(sorted(objects, key=lambda item: (-position_by_object[item].distance_to_tray, item)))
    if strategy == "left_to_right":
        return tuple(sorted(objects, key=lambda item: (position_by_object[item].y, item)))
    if strategy == "right_to_left":
        return tuple(sorted(objects, key=lambda item: (-position_by_object[item].y, item)))
    if strategy == "odd_positions_first":
        return tuple(sorted(objects, key=lambda item: (position_by_object[item].ordinal_position % 2 == 0, position_by_object[item].ordinal_position)))
    if strategy == "even_positions_first":
        return tuple(sorted(objects, key=lambda item: (position_by_object[item].ordinal_position % 2 == 1, position_by_object[item].ordinal_position)))
    logging.warning("unknown sequence strategy %s; falling back to profile/object order", strategy)
    return objects


def make_ownership(objects_by_user: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    return {
        user_id: sorted(set(objects))
        for user_id, objects in sorted(objects_by_user.items())
        if set(objects)
    }


def make_episode(
    index: int,
    split: str,
    task_type: str,
    scene: SceneVariant,
    profile: UserProfile,
    objects: tuple[str, ...],
    strategy: str,
    rng: random.Random,
    ownership: dict[str, list[str]],
    object_seen_flags: dict[str, bool] | None = None,
    adaptation: dict[str, Any] | None = None,
    consistency: dict[str, Any] | None = None,
    debug: dict[str, Any] | None = None,
) -> EpisodeSpec:
    spawn_points = circular_spawn_points(objects, radius=1.0, rng=rng)
    target_sequence = order_objects(objects, strategy, spawn_points)
    if adaptation and "previous_strategy" in adaptation:
        previous_sequence = order_objects(objects, str(adaptation["previous_strategy"]), spawn_points)
        adaptation = {
            **adaptation,
            "previous_sequence": previous_sequence,
            "updated_sequence": target_sequence,
            "same_object_set": objects,
            "same_scene": scene.scene,
            "same_table": scene.table,
        }
    episode_id = f"sequences_{split}_{index:06d}"
    return EpisodeSpec(
        episode_id=episode_id,
        suite="sequences",
        split=split,
        task_type=task_type,
        scene=scene.scene,
        table=scene.table,
        fixture=FIXTURE,
        relation=RELATION,
        target_user=profile.user_id,
        sequence_strategy=strategy,
        target_sequence=target_sequence,
        graspable_objects=objects,
        spawn_points=spawn_points,
        ownership=ownership,
        participating_users=(profile.user_id,),
        object_seen_flags=object_seen_flags,
        adaptation=adaptation,
        consistency=consistency,
        debug=debug,
    )


def rescale_spawn_points(spec: EpisodeSpec, spawn_radius: float) -> tuple[SpawnPoint, ...]:
    scaled = []
    for point in spec.spawn_points:
        x = round(point.x * spawn_radius, 4)
        y = round(point.y * spawn_radius, 4)
        scaled.append(
            SpawnPoint(
                object_type=point.object_type,
                x=x,
                y=y,
                distance_to_tray=round(math.hypot(x, y), 4),
                ordinal_position=point.ordinal_position,
            )
        )
    return tuple(scaled)


def unique_combinations(values: Iterable[str], size: int) -> Iterator[tuple[str, ...]]:
    yield from itertools.combinations(sorted(set(values)), size)


def build_type1_specs(
    profiles: list[UserProfile],
    neutral_object_pool: tuple[str, ...],
    seed: int,
) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        for objects in unique_combinations(neutral_object_pool, OBJECTS_PER_EPISODE):
            scene = choose_scene_variant(index)
            specs.append(
                make_episode(
                    index=index,
                    split="type1",
                    task_type="type1",
                    scene=scene,
                    profile=profile,
                    objects=objects,
                    strategy=profile.sequence_strategy,
                    rng=random.Random(f"{seed}:type1:{profile.user_id}:{objects}"),
                    ownership={},
                    object_seen_flags={item: False for item in objects},
                    debug={"object_policy": "three unique graspable objects not owned by any profile user"},
                )
            )
            index += 1
    return specs


def build_type2_specs(profiles: list[UserProfile], seed: int) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        for objects in unique_combinations(profile.belongings, OBJECTS_PER_EPISODE):
            seen = set(profile.seen_belongings)
            scene = choose_scene_variant(index)
            specs.append(
                make_episode(
                    index=index,
                    split="type2",
                    task_type="type2",
                    scene=scene,
                    profile=profile,
                    objects=objects,
                    strategy=profile.sequence_strategy,
                    rng=random.Random(f"{seed}:type2:{profile.user_id}:{objects}"),
                    ownership=make_ownership({profile.user_id: objects}),
                    object_seen_flags={item: item in seen for item in objects},
                    debug={"object_policy": "three objects sampled only from target user's belongings"},
                )
            )
            index += 1
    return specs


def build_adaptability_specs(profiles: list[UserProfile], seed: int) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        updated_strategy = ADAPTATION_STRATEGY_PAIRS.get(profile.sequence_strategy)
        if updated_strategy is None:
            continue
        for objects in unique_combinations(profile.seen_belongings, OBJECTS_PER_EPISODE):
            scene = choose_scene_variant(index)
            specs.append(
                make_episode(
                    index=index,
                    split="adaptability",
                    task_type="adaptability",
                    scene=scene,
                    profile=profile,
                    objects=objects,
                    strategy=updated_strategy,
                    rng=random.Random(f"{seed}:adaptability:{profile.user_id}:{objects}"),
                    ownership=make_ownership({profile.user_id: objects}),
                    object_seen_flags={item: True for item in objects},
                    adaptation={
                        "user_id": profile.user_id,
                        "previous_strategy": profile.sequence_strategy,
                        "updated_strategy": updated_strategy,
                        "rule": "adaptability changes only the user's sequence strategy and uses no unseen objects",
                    },
                )
            )
            index += 1
    return specs


def build_consistency_specs(profiles: list[UserProfile], seed: int) -> list[EpisodeSpec]:
    specs = []
    index = 1
    for profile in profiles:
        if len(profile.seen_belongings) < 2 or len(profile.unseen_belongings) < 1:
            continue
        for seen_objects in unique_combinations(profile.seen_belongings, 2):
            for unseen_object in profile.unseen_belongings:
                objects = tuple(sorted((*seen_objects, unseen_object)))
                scene = choose_scene_variant(index)
                specs.append(
                    make_episode(
                        index=index,
                        split="consistency",
                        task_type="consistency",
                        scene=scene,
                        profile=profile,
                        objects=objects,
                        strategy=profile.sequence_strategy,
                        rng=random.Random(f"{seed}:consistency:{profile.user_id}:{objects}"),
                        ownership=make_ownership({profile.user_id: objects}),
                        object_seen_flags={item: item != unseen_object for item in objects},
                        consistency={
                            "user_id": profile.user_id,
                            "seen_objects": list(seen_objects),
                            "unseen_object": unseen_object,
                            "rule": "consistency uses two seen belongings and one user-local unseen belonging",
                        },
                    )
                )
                index += 1
    return specs


def build_all_specs(
    profiles: list[UserProfile],
    neutral_object_pool: tuple[str, ...],
    seed: int,
) -> dict[str, list[EpisodeSpec]]:
    return {
        "type1": build_type1_specs(profiles, neutral_object_pool, seed + 11),
        "type2": build_type2_specs(profiles, seed + 22),
        "adaptability": build_adaptability_specs(profiles, seed + 33),
        "consistency": build_consistency_specs(profiles, seed + 44),
    }


def split_budget(max_counts: dict[str, int], total_budget: int | None) -> dict[str, int]:
    if total_budget is None:
        return dict(max_counts)
    if total_budget < 0:
        raise ValueError(f"sequences.total_episodes must be non-negative, got {total_budget}")
    positive = {split: count for split, count in max_counts.items() if count > 0}
    if not positive:
        return dict(max_counts)
    total_available = sum(positive.values())
    remaining_budget = min(total_budget, total_available)
    planned = {split: 0 for split in max_counts}
    raw = {split: (count / total_available) * remaining_budget for split, count in positive.items()}
    for split, value in raw.items():
        planned[split] = min(max_counts[split], int(value))
    remainder = remaining_budget - sum(planned.values())
    fractions = sorted(positive, key=lambda split: (raw[split] - int(raw[split]), max_counts[split]), reverse=True)
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
    users = sorted(by_user)
    sampled = []
    while len(sampled) < count and any(by_user.values()):
        for user_id in users:
            if by_user[user_id]:
                sampled.append(by_user[user_id].pop())
                if len(sampled) == count:
                    break
    return sampled


def object_ids(object_types: tuple[str, ...]) -> list[tuple[str, str]]:
    counts: Counter[str] = Counter()
    ids = []
    for object_type in object_types:
        counts[object_type] += 1
        ids.append((f"{object_type}_{counts[object_type]}", object_type))
    return ids


def grouped_declarations(id_type_pairs: list[tuple[str, str]]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for object_id, object_type in id_type_pairs:
        grouped[object_type].append(object_id)
    return [f"    {' '.join(ids)} - {object_type}" for object_type, ids in sorted(grouped.items())]


def square_range(center_x: float, center_y: float, half_size: float) -> tuple[float, float, float, float]:
    return (
        round(center_x - half_size, 4),
        round(center_y - half_size, 4),
        round(center_x + half_size, 4),
        round(center_y + half_size, 4),
    )


def render_region(name: str, target: str, ranges: tuple[float, float, float, float] | None = None) -> str:
    if ranges is None:
        return f"""      ({name}
          (:target {target})
      )"""
    x1, y1, x2, y2 = ranges
    return f"""      ({name}
          (:target {target})
          (:ranges (
              ({x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f})
            )
          )
      )"""


def language_for(spec: EpisodeSpec) -> str:
    sequence = ", ".join(spec.target_sequence)
    if spec.split == "adaptability" and spec.adaptation:
        previous = ", ".join(spec.adaptation.get("previous_sequence", ()))
        return (
            f"After {spec.target_user}'s sequence preference changes from "
            f"{previous} to {sequence}, put the same objects into the wooden_tray "
            f"in the new order: {sequence}"
        )
    if spec.split == "type1":
        return (
            f"Use {spec.target_user}'s sequence preference and put the objects "
            f"into the wooden_tray in this order: {sequence}"
        )
    return (
        f"Put the objects belonging to {spec.target_user} into the wooden_tray "
        f"in this order: {sequence}"
    )


def render_bddl(spec: EpisodeSpec, spawn_radius: float, object_box_size: float, fixture_box_size: float) -> str:
    scaled_points = rescale_spawn_points(spec, spawn_radius)
    object_id_pairs = object_ids(spec.graspable_objects)
    object_id_by_type = {object_type: object_id for object_id, object_type in object_id_pairs}
    fixture_id = f"{FIXTURE}_1"

    regions = [render_region("wooden_tray_region", spec.table, square_range(0.0, 0.0, fixture_box_size))]
    for idx, point in enumerate(scaled_points):
        regions.append(render_region(f"object_init_region_{idx}", spec.table, square_range(point.x, point.y, object_box_size)))
    regions.append(render_region("contain_region", fixture_id))

    init_lines = [f"    (On {fixture_id} {spec.table}_wooden_tray_region)"]
    for idx, (object_id, _object_type) in enumerate(object_id_pairs):
        init_lines.append(f"    (On {object_id} {spec.table}_object_init_region_{idx})")

    goal_items = [f"({RELATION} {object_id_by_type[item]} {fixture_id}_contain_region)" for item in spec.target_sequence]
    goal_line = f"    (And {' '.join(goal_items)})"

    return f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language_for(spec)})
    (:regions
{chr(10).join(regions)}
    )

  (:fixtures
    {spec.table} - table
  )

  (:objects
{chr(10).join(grouped_declarations(object_id_pairs))}
    {fixture_id} - {FIXTURE}
  )

  (:obj_of_interest
    {' '.join(object_id_by_type[item] for item in spec.target_sequence)}
    {fixture_id}
  )

  (:init
{chr(10).join(init_lines)}
  )

  (:goal
{goal_line}
  )

)
"""


def metadata_for(
    spec: EpisodeSpec,
    profiles_path: Path,
    libero_objects_path: Path,
    bddl_path: Path,
    spawn_radius: float,
) -> dict[str, Any]:
    scaled_points = rescale_spawn_points(spec, spawn_radius)
    payload = {
        **asdict(spec),
        "spawn_points": [asdict(point) for point in scaled_points],
        "language": language_for(spec),
        "profile_source": str(profiles_path),
        "libero_objects_source": str(libero_objects_path),
        "bddl_file": str(bddl_path),
        "constraints": {
            "fixture": FIXTURE,
            "graspable_object_count": OBJECTS_PER_EPISODE,
            "bddl_goal_policy": "BDDL checks final In relation for all three objects; temporal order is evaluated from target_sequence metadata",
            "type1_policy": "three unique non-profile graspable objects from libero_objects.json",
            "type2_policy": "three target-user belongings",
            "consistency_policy": "two seen target-user belongings and one unseen target-user belonging",
            "adaptability_policy": "target-user belongings only; no unseen objects; sequence strategy changes",
            "spawn_radius": spawn_radius,
        },
    }
    return payload


def output_paths(output_root: Path, spec: EpisodeSpec) -> tuple[Path, Path]:
    base = output_root / "sequences" / spec.split
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


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    config = read_config(args.config)
    profiles = load_profiles(args.profiles, args.seen_ratio)
    profile_object_pool = all_profile_objects(profiles)
    profile_object_set = set(profile_object_pool)
    libero_object_pool = load_libero_graspable_types(args.libero_objects)
    neutral_object_pool = tuple(item for item in libero_object_pool if item not in profile_object_set)
    object_owners = ownership_index(profiles)

    if len(neutral_object_pool) < OBJECTS_PER_EPISODE:
        raise ValueError(
            f"Need at least {OBJECTS_PER_EPISODE} non-profile graspable objects for type1; "
            f"found {len(neutral_object_pool)}"
        )

    specs_by_split = build_all_specs(profiles, neutral_object_pool, args.seed)
    counts = {split: len(specs_by_split[split]) for split in SPLIT_ORDER}
    configured_total = get_nested_config(config, ("sequences", "total_episodes"))
    if configured_total is not None:
        configured_total = int(configured_total)
    planned_counts = split_budget(counts, configured_total)
    if args.max_episodes_per_split is not None:
        planned_counts = {split: min(count, args.max_episodes_per_split) for split, count in planned_counts.items()}
    planned_total = sum(planned_counts.values())

    planned_specs = {}
    for offset, split in enumerate(SPLIT_ORDER):
        planned_specs[split] = sample_specs(specs_by_split[split], planned_counts[split], random.Random(args.seed + offset + 1))

    logging.info("profiles: %d users", len(profiles))
    logging.info("profile object pool: %d unique objects from profiles.json", len(profile_object_pool))
    logging.info("neutral object pool: %d unique non-profile graspable objects from libero_objects.json", len(neutral_object_pool))
    logging.info("fixture: %s", FIXTURE)
    logging.info("output root: %s", args.output_root)
    logging.info("seed: %d | seen_ratio: %.2f | spawn_radius: %.3f | gpu: %s", args.seed, args.seen_ratio, args.spawn_radius, args.use_gpu)
    logging.info("libero objects: %s", args.libero_objects)
    logging.info("strategy counts: %s", dict(Counter(profile.sequence_strategy for profile in profiles)))
    logging.info("user seen/unseen counts: %s", {p.user_id: [len(p.seen_belongings), len(p.unseen_belongings)] for p in profiles})
    logging.info("episode counts by split: %s", counts)
    logging.info("total episodes possible: %d", sum(counts.values()))
    logging.info("config: %s", args.config)
    if configured_total is not None:
        logging.info("configured sequences.total_episodes: %d", configured_total)
    logging.info("planned episodes: %s", planned_counts)
    logging.info("planned total episodes: %d", planned_total)
    if args.max_episodes_per_split is not None:
        logging.info("debug cap enabled: max %d episodes per split", args.max_episodes_per_split)

    if args.dry_run:
        logging.info("dry run enabled; no files written")
        return

    summary = {
        "suite": "sequences",
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
        "strategy_counts": dict(Counter(profile.sequence_strategy for profile in profiles)),
        "hyperparameters": {
            "fixture": FIXTURE,
            "graspable_object_count": OBJECTS_PER_EPISODE,
            "scene_variants": [list(item) for item in DEFAULT_SCENE_VARIANTS],
            "seed": args.seed,
            "seen_ratio": args.seen_ratio,
            "spawn_radius": args.spawn_radius,
            "object_box_size": args.object_box_size,
            "fixture_box_size": args.fixture_box_size,
            "use_gpu": args.use_gpu,
            "max_episodes_per_split": args.max_episodes_per_split,
            "object_owners": object_owners,
            "adaptation_strategy_pairs": ADAPTATION_STRATEGY_PAIRS,
        },
    }

    progress = tqdm(total=planned_total, desc="Generating sequence episodes", unit="episode")
    written = 0
    for split in SPLIT_ORDER:
        for spec in planned_specs[split]:
            bddl_path, metadata_path = output_paths(args.output_root, spec)
            write_text(bddl_path, render_bddl(spec, args.spawn_radius, args.object_box_size, args.fixture_box_size))
            write_json(metadata_path, metadata_for(spec, args.profiles, args.libero_objects, bddl_path, args.spawn_radius))
            written += 1
            progress.update(1)
            if args.debug and written <= 5:
                logging.debug("sample episode %s: %s", spec.episode_id, asdict(spec))
    progress.close()

    summary["written"] = written
    write_json(args.output_root / "sequences" / "generation_summary.json", summary)
    logging.info("written episodes: %d", written)
    logging.info("summary: %s", args.output_root / "sequences" / "generation_summary.json")


if __name__ == "__main__":
    main()
