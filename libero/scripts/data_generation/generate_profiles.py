#!/usr/bin/env python3
"""Generate personalized LIBERO user profiles from stable spawn positions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - YAML is optional.
    yaml = None


VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = VLAPB_LIBERO_ROOT / "docs"
PROFILES_DIR = VLAPB_LIBERO_ROOT / "profiles"
DATA_GEN_DIR = Path(__file__).resolve().parent

DEFAULT_SUMMARY = DOCS_DIR / "libero_objects_summary.md"
DEFAULT_OBJECTS = DOCS_DIR / "libero_objects.json"
DEFAULT_POSSIBLE_DIR = DATA_GEN_DIR / "possible_spawn_positions"
DEFAULT_HOLDED_DIR = DATA_GEN_DIR / "holded_spawn_positions"
DEFAULT_CONFIG = VLAPB_LIBERO_ROOT / "teleop_moveit" / "episode_selection.yaml"
DEFAULT_OUTPUT = PROFILES_DIR / "profiles.json"
DEFAULT_MD = PROFILES_DIR / "profiles_generation_summary.md"
DEFAULT_OBJECTS_PER_USER = 4

ORDER_STRATEGIES = (
    "alphabetical",
    "reverse_alphabetical",
    "fixed_near_to_far",
    "fixed_far_to_near",
    "left_to_right",
    "right_to_left",
    "odd_positions_first",
    "even_positions_first",
)
STRUCTURAL_LABELS = {"in", "on", "top", "middle", "bottom"}
DRAWER_SIDE_LABEL_RE = re.compile(r"^(top|middle|bottom)_(front|back|left|right)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate personalized belongings and placement profiles.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--possible-dir", type=Path, default=DEFAULT_POSSIBLE_DIR)
    parser.add_argument("--holded-dir", type=Path, default=DEFAULT_HOLDED_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--objects-per-user", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def scene_area(scene: str | None) -> str:
    scene = scene or ""
    if scene.startswith("KITCHEN_"):
        return "kitchen"
    if scene.startswith("LIVING_ROOM_"):
        return "living_room"
    if scene.startswith("STUDY_"):
        return "study"
    return scene.split("_SCENE", 1)[0].lower() if "_SCENE" in scene else "unknown"


def safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_graspable_items(objects_path: Path, config: dict[str, Any], placeable_items: set[str] | None = None) -> list[str]:
    objects = safe_read_json(objects_path)
    items = {
        item.get("type")
        for task in objects.get("tasks", [])
        for item in task.get("objects", [])
        if isinstance(item, dict) and item.get("type")
    }
    fixed_types = {
        item.get("type")
        for task in objects.get("tasks", [])
        for item in task.get("fixtures", [])
        if isinstance(item, dict) and item.get("type")
    }
    yaml_items = set(config.get("available", {}).get("items", []) or [])
    if yaml_items:
        items |= yaml_items
    unwanted = set(config.get("selection", {}).get("unwanted_items", []) or [])
    fixed_items = set(config.get("available", {}).get("fixed_items", []) or [])
    output = [item for item in items if item and item not in fixed_types and item not in fixed_items and item not in unwanted]
    if placeable_items is not None:
        output = [item for item in output if item in placeable_items]
    return sorted(output)


def load_spawn_positions(root: Path) -> list[dict[str, Any]]:
    placements = []
    for path in sorted(root.glob("**/*.json")):
        if path.name.startswith("_"):
            continue
        data = safe_read_json(path)
        for object_type, entry in sorted(data.items()):
            if object_type.startswith("_") or not isinstance(entry, dict):
                continue
            status = entry.get("status")
            # Use only validated-possible candidates as profile inputs.
            if status not in {None, "possible"}:
                continue
            scene = entry.get("scene") or path.parts[-4] if len(path.parts) >= 4 else None
            area = entry.get("area") or scene_area(scene)
            placements.append(
                {
                    "object_type": entry.get("object_type") or object_type,
                    "scene": scene,
                    "area": area,
                    "table": entry.get("table"),
                    "fixed_type": entry.get("fixed_type"),
                    "label": entry.get("label"),
                    "location": entry.get("location"),
                    "pose": entry.get("relative_scene_table_fixed_pose") or entry.get("pose"),
                    "actual_pose": entry.get("actual_pose"),
                    "source_file": str(path),
                    "modified": any(part.startswith("modified_") for part in path.parts),
                }
            )
    return placements


def combo_has_two_area_options(items: tuple[str, ...], grouped: dict[str, dict[str, list[dict[str, Any]]]]) -> bool:
    area_options = sorted({area for item in items for area in grouped.get(item, {})})
    for area_a, area_b in itertools.combinations(area_options, 2):
        choices = [[area for area in (area_a, area_b) if grouped.get(item, {}).get(area)] for item in items]
        if any(not choice for choice in choices):
            continue
        if any(len(set(area_assignment)) == 2 for area_assignment in itertools.product(*choices)):
            return True
    return False


def max_profiles(items: list[str], objects_per_user: int, grouped: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[int, list[tuple[str, ...]]]:
    selected: list[tuple[str, ...]] = []
    combos = itertools.combinations(items, objects_per_user)
    for combo in combos:
        if not combo_has_two_area_options(tuple(combo), grouped):
            continue
        combo_set = set(combo)
        if all(len(combo_set & set(other)) <= 1 for other in selected):
            selected.append(tuple(combo))
    return len(selected), selected


def prompt_num_users(max_users: int, default_users: int | None) -> int:
    default = min(default_users or max_users, max_users)
    print(f"Maximum users possible with pairwise belongings overlap <= 1: {max_users}")
    raw = input(f"How many users should be generated? [default {default}]: ").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 1 or value > max_users:
        raise ValueError(f"num-users must be between 1 and {max_users}")
    return value


def placements_by_object_and_area(placements: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for placement in placements:
        grouped[placement["object_type"]][placement["area"]].append(placement)
    return grouped


def placement_sort_key(placement: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(placement.get("area") or ""),
        str(placement.get("scene") or ""),
        str(placement.get("fixed_type") or ""),
        str(placement.get("label") or ""),
    )


def choose_two_area_placements(items: tuple[str, ...], grouped: dict[str, dict[str, list[dict[str, Any]]]], user_index: int) -> list[dict[str, Any]]:
    area_options = sorted({area for item in items for area in grouped.get(item, {})})
    pairs = list(itertools.combinations(area_options, 2))
    if not pairs:
        raise RuntimeError(f"No two-area placement options for {items}")
    pairs = pairs[user_index % len(pairs) :] + pairs[: user_index % len(pairs)]
    for area_a, area_b in pairs:
        choices = [[area for area in (area_a, area_b) if grouped.get(item, {}).get(area)] for item in items]
        if any(not choice for choice in choices):
            continue
        best_area_assignment = None
        best_balance = None
        for area_assignment in itertools.product(*choices):
            counts = Counter(area_assignment)
            if counts[area_a] == 0 or counts[area_b] == 0:
                continue
            balance = abs(counts[area_a] - counts[area_b])
            if best_balance is None or balance < best_balance:
                best_area_assignment = area_assignment
                best_balance = balance
        if best_area_assignment is None:
            continue
        assignments = []
        counts = Counter()
        for item, area in zip(items, best_area_assignment):
            options = sorted(grouped[item][area], key=placement_sort_key)
            placement = dict(options[(user_index + counts[area]) % len(options)])
            placement["object_type"] = item
            assignments.append(placement)
            counts[area] += 1
        if len(assignments) == len(items):
            return assignments
    raise RuntimeError(f"Could not assign all items to exactly two areas: {items}")


def has_structural_label(placement: dict[str, Any]) -> bool:
    label = str(placement.get("label") or "").lower()
    return label in STRUCTURAL_LABELS or DRAWER_SIDE_LABEL_RE.match(label) is not None


def ensure_structural_minimum(
    placements: list[dict[str, Any]],
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    user_index: int,
    minimum_structural: int,
) -> list[dict[str, Any]]:
    if minimum_structural <= 0:
        return placements
    current = sum(1 for placement in placements if has_structural_label(placement))
    if current >= minimum_structural:
        return placements

    updated = list(placements)
    for idx, placement in enumerate(updated):
        object_type = str(placement.get("object_type"))
        area = str(placement.get("area"))
        options = sorted(grouped.get(object_type, {}).get(area, []), key=placement_sort_key)
        structural_options = [option for option in options if has_structural_label(option)]
        if not structural_options:
            continue
        replacement = dict(structural_options[user_index % len(structural_options)])
        replacement["object_type"] = object_type
        updated[idx] = replacement
        current += 1
        if current >= minimum_structural:
            return updated
    return updated


def order_key(strategy: str, placement: dict[str, Any], index: int) -> Any:
    pose = placement.get("pose") or {}
    if strategy == "alphabetical":
        return placement["object_type"]
    if strategy == "reverse_alphabetical":
        return tuple(reversed(placement["object_type"]))
    if strategy == "fixed_near_to_far":
        return math.hypot(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
    if strategy == "fixed_far_to_near":
        return -math.hypot(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
    if strategy == "left_to_right":
        return float(pose.get("y", 0.0))
    if strategy == "right_to_left":
        return -float(pose.get("y", 0.0))
    if strategy == "odd_positions_first":
        return (index % 2 == 0, index)
    if strategy == "even_positions_first":
        return (index % 2 == 1, index)
    return index


def assign_order(strategy: str, placements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(placements, start=1))
    ordered = sorted(indexed, key=lambda pair: order_key(strategy, pair[1], pair[0]))
    return [{**placement, "order": idx} for idx, (_old_idx, placement) in enumerate(ordered, start=1)]


def build_profiles(
    num_users: int,
    combos: list[tuple[str, ...]],
    placements: list[dict[str, Any]],
    seed: int,
    minimum_structural: int,
) -> list[dict[str, Any]]:
    grouped = placements_by_object_and_area(placements)
    profiles = []
    for user_idx in range(num_users):
        belongings = combos[user_idx]
        user_placements = choose_two_area_placements(belongings, grouped, user_idx + seed)
        user_placements = ensure_structural_minimum(user_placements, grouped, user_idx + seed, minimum_structural)
        strategy = ORDER_STRATEGIES[user_idx % len(ORDER_STRATEGIES)]
        ordered = assign_order(strategy, user_placements)
        profiles.append(
            {
                "user_id": f"user_{user_idx + 1:03d}",
                "belongings": list(belongings),
                "areas": sorted({placement["area"] for placement in ordered}),
                "placement_order_strategy": strategy,
                "placements": ordered,
            }
        )
    return profiles


def bar(count: int, total: int, width: int = 24) -> str:
    filled = 0 if total == 0 else round(width * count / total)
    return "#" * filled + "." * (width - filled)


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    profiles = data["profiles"]
    area_counts = Counter(area for profile in profiles for area in profile["areas"])
    object_counts = Counter(item for profile in profiles for item in profile["belongings"])
    strategy_counts = Counter(profile["placement_order_strategy"] for profile in profiles)
    lines = [
        "# Profile Generation Summary",
        "",
        f"- Users generated: `{len(profiles)}`",
        f"- Objects per user: `{data['metadata']['objects_per_user']}`",
        f"- Maximum users possible: `{data['metadata']['max_users_possible']}`",
        f"- Graspable item count: `{len(data['metadata']['graspable_items'])}`",
        f"- Stable placement entries loaded: `{data['metadata']['placement_count']}`",
        "",
        "## Area Coverage",
        "",
        "| Area | Profiles | Bar |",
        "| --- | ---: | --- |",
    ]
    for area, count in sorted(area_counts.items()):
        lines.append(f"| `{area}` | {count} | `{bar(count, len(profiles))}` |")
    lines.extend(["", "## Order Strategy Distribution", "", "| Strategy | Users | Bar |", "| --- | ---: | --- |"])
    for strategy, count in sorted(strategy_counts.items()):
        lines.append(f"| `{strategy}` | {count} | `{bar(count, len(profiles))}` |")
    lines.extend(["", "## Belongings Distribution", "", "| Object | Users | Bar |", "| --- | ---: | --- |"])
    for item, count in object_counts.most_common():
        lines.append(f"| `{item}` | {count} | `{bar(count, len(profiles))}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    objects_per_user = args.objects_per_user or int(config.get("generation", {}).get("objects_per_user", DEFAULT_OBJECTS_PER_USER))
    minimum_structural = int(config.get("generation", {}).get("min_structural_per_user", 1))
    default_users = config.get("generation", {}).get("num_users")
    summary_text = args.summary.read_text(encoding="utf-8")
    objects_data = safe_read_json(args.objects)
    placements = load_spawn_positions(args.possible_dir)
    placeable_items = {placement["object_type"] for placement in placements}
    graspable_items = load_graspable_items(args.objects, config, placeable_items)
    grouped_for_capacity = placements_by_object_and_area(placements)
    max_users, combos = max_profiles(graspable_items, objects_per_user, grouped_for_capacity)
    if max_users == 0:
        raise RuntimeError("No valid user profiles can be generated with the current item pool")
    num_users = args.num_users if args.num_users is not None else prompt_num_users(max_users, int(default_users) if default_users else None)
    if num_users > max_users:
        raise ValueError(f"Requested {num_users} users but only {max_users} are possible")
    profiles = build_profiles(num_users, combos, placements, args.seed, minimum_structural)
    data = {
        "metadata": {
            "source_files": {
                "summary": str(args.summary),
                "objects": str(args.objects),
                "possible_dir": str(args.possible_dir),
                "holded_dir": str(args.holded_dir),
                "config": str(args.config),
            },
            "summary_chars": len(summary_text),
            "objects_per_user": objects_per_user,
            "pairwise_overlap_limit": 1,
            "min_structural_per_user": minimum_structural,
            "max_users_possible": max_users,
            "graspable_items": graspable_items,
            "placement_count": len(placements),
            "libero_task_count": objects_data.get("metadata", {}).get("task_count"),
        },
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown, data)
    print(f"Maximum users possible: {max_users}")
    print(f"Wrote {len(profiles)} profiles to {args.output}")
    print(f"Wrote summary to {args.markdown}")


if __name__ == "__main__":
    main()
