#!/usr/bin/env python3
"""Generate personnel profiles for VLAPB LIBERO personalization.

The generator is intentionally evidence-driven: objects and placements are
extracted from the four LIBERO instruction summaries under /home/artemis/Documents/VLAPB/libero/docs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

VLAPB_LIBERO_ROOT = Path("/home/artemis/Documents/VLAPB/libero")
DOCS_ROOT = VLAPB_LIBERO_ROOT / "docs"
DEFAULT_OUTPUT_DIR = VLAPB_LIBERO_ROOT / "profiles"
FIXED_ITEMS_PATH = DOCS_ROOT / "fixed_items.json"
DEFAULT_SPAWN_POSITIONS_DIR = DOCS_ROOT / "possible_spawn_positions"
POSE_KEYS = ("x", "y", "z", "r", "p", "h")
CANONICAL_LOCATION_ALIASES = {
    "cabinet.drawer.front_side": "cabinet.bottom_drawer.inside",
    "wooden_two_layer_shelf.shelf": "wooden_two_layer_shelf.top_shelf",
    "wooden_two_layer_shelf.under_shelf": "wooden_two_layer_shelf.bottom_shelf",
}

INSTRUCTION_FILES = {
    "libero_10": DOCS_ROOT / "libero_10_instructions.txt",
    "libero_90": DOCS_ROOT / "libero_90_instructions.txt",
    "libero_object": DOCS_ROOT / "libero_object_instructions.txt",
    "libero_spatial": DOCS_ROOT / "libero_spatial_instructions.txt",
}

FIXTURE_CLASSES = {
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

HARD_EXCLUDED_OBJECTS = {"wine_bottle"}
DISALLOWED_TARGET_CLASSES = {"desk_caddy", "wine_rack"}
TARGET_HINTS = (
    "basket",
    "cabinet",
    "drawer",
    "microwave",
    "plate",
    "shelf",
    "stove",
    "white_cabinet",
    "wooden_cabinet",
    "wooden_tray",
    "wooden_two_layer_shelf",
)
STABLE_PLACEMENT_HINTS = (
    "in the basket",
    "to the right of the basket",
    "to the left of the basket",
    "to the front of the basket",
    "to the back of the basket",
    "in the bottom drawer",
    "in the middle drawer",
    "in the top drawer",
    "to the front of the drawer",
    "in the microwave",
    "on top of the microwave",
    "to the front of the microwave",
    "on the stove",
    "on the top drawer of the cabinet shelf",
    "on the bottom drawer of the cabinet shelf",
    "to the front of the cabinet shelf",
    "on the plate",
    "to the right of the plate",
    "to the left of the plate",
    "to the front of the plate",
    "to the back of the plate",
)
STABLE_PLACEMENT_OPTIONS = (
    {"hint": "in the basket", "location": "basket.inside", "fixed_item": "basket"},
    {"hint": "to the right of the basket", "location": "basket.right_side", "fixed_item": "basket"},
    {"hint": "to the left of the basket", "location": "basket.left_side", "fixed_item": "basket"},
    {"hint": "to the front of the basket", "location": "basket.front_side", "fixed_item": "basket"},
    {"hint": "to the back of the basket", "location": "basket.back_side", "fixed_item": "basket"},
    {"hint": "in the bottom drawer", "location": "cabinet.bottom_drawer.inside", "fixed_item": "cabinet"},
    {"hint": "in the middle drawer", "location": "cabinet.middle_drawer.inside", "fixed_item": "cabinet"},
    {"hint": "in the top drawer", "location": "cabinet.top_drawer.inside", "fixed_item": "cabinet"},
    {"hint": "to the front of the drawer", "location": "cabinet.bottom_drawer.front_side", "fixed_item": "cabinet"},
    {"hint": "in the microwave", "location": "microwave.inside", "fixed_item": "microwave"},
    {"hint": "on top of the microwave", "location": "microwave.top_surface", "fixed_item": "microwave"},
    {"hint": "to the front of the microwave", "location": "microwave.front_side", "fixed_item": "microwave"},
    {"hint": "on the stove", "location": "flat_stove.surface", "fixed_item": "flat_stove"},
    {
        "hint": "on the top drawer of the cabinet shelf",
        "location": "wooden_two_layer_shelf.top_shelf",
        "fixed_item": "wooden_two_layer_shelf",
    },
    {
        "hint": "on the bottom drawer of the cabinet shelf",
        "location": "wooden_two_layer_shelf.bottom_shelf",
        "fixed_item": "wooden_two_layer_shelf",
    },
    {
        "hint": "to the front of the cabinet shelf",
        "location": "wooden_two_layer_shelf.front_side",
        "fixed_item": "wooden_two_layer_shelf",
    },
    {"hint": "on the plate", "location": "plate.center", "fixed_item": "plate"},
    {"hint": "to the right of the plate", "location": "plate.right_side", "fixed_item": "plate"},
    {"hint": "to the left of the plate", "location": "plate.left_side", "fixed_item": "plate"},
    {"hint": "to the front of the plate", "location": "plate.front_side", "fixed_item": "plate"},
    {"hint": "to the back of the plate", "location": "plate.back_side", "fixed_item": "plate"},

    {"hint": "to the left of the tray", "location": "wooden_tray.left_side", "fixed_item": "wooden_tray"},
    {"hint": "to the right of the tray", "location": "wooden_tray.right_side", "fixed_item": "wooden_tray"},
    {"hint": "to the front of the tray", "location": "wooden_tray.front_side", "fixed_item": "wooden_tray"},
    {"hint": "to the back of the tray", "location": "wooden_tray.back_side", "fixed_item": "wooden_tray"},
    {"hint": "in the tray", "location": "wooden_tray.inside", "fixed_item": "wooden_tray"},

)
CENTER_FIXED_LOCATIONS = {
    "basket.inside",
    "wooden_tray.inside",
    "plate.center",
    "flat_stove.surface",
}
SEQUENCE_RULES = (
    "left_to_right",
    "right_to_left",
    "center_outward",
    "outer_to_center",
    "alphabetical",
    "reverse_alphabetical",
    "front_to_back",
    "back_to_front",
    "close_to_far",
    "far_to_close",
)


@dataclass(frozen=True)
class Instruction:
    source: str
    task_id: int
    instruction: str
    bddl_file: str
    fixtures: dict[str, str]
    objects: dict[str, str]
    objects_of_interest: tuple[str, ...]
    fixed_items: tuple[dict[str, str], ...] = ()


@dataclass
class PlacementEvidence:
    location: str
    source: str
    task_id: int
    instruction: str
    bddl_file: str
    count: int = 1
    hint: str = ""
    fixed_item: str = ""
    coordinate_pending: bool = False


@dataclass
class ObjectRecord:
    object_class: str
    sources: set[str] = field(default_factory=set)
    action_count: int = 0
    basket_count: int = 0
    spatial_count: int = 0
    placements: dict[str, PlacementEvidence] = field(default_factory=dict)


def parse_typed_entities(value: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+_\d+) \(([^)]+)\)", value))


def read_instructions(file_path: Path, source: str) -> list[Instruction]:
    text = file_path.read_text()
    blocks = re.split(r"\n(?=\d+\.\s)", text.strip())
    instructions: list[Instruction] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        header = re.match(r"(\d+)\.\s*(.*)", lines[0])
        if not header:
            continue

        task_id = int(header.group(1))
        instruction = header.group(2).strip()
        bddl_file = ""
        fixtures: dict[str, str] = {}
        objects: dict[str, str] = {}
        objects_of_interest: tuple[str, ...] = ()

        for line in lines[1:]:
            key, _, raw_value = line.strip().partition(":")
            value = raw_value.strip()
            if key == "file":
                bddl_file = value
            elif key == "fixtures":
                fixtures = parse_typed_entities(value)
            elif key == "objects":
                objects = parse_typed_entities(value)
            elif key == "objects_of_interest":
                objects_of_interest = tuple(item.strip() for item in value.split(",") if item.strip())

        instructions.append(
            Instruction(
                source=source,
                task_id=task_id,
                instruction=instruction,
                bddl_file=bddl_file,
                fixtures=fixtures,
                objects=objects,
                objects_of_interest=objects_of_interest,
            )
        )

    return instructions


def load_fixed_items(path: Path = FIXED_ITEMS_PATH) -> dict[tuple[str, int], tuple[dict[str, str], ...]]:
    if not path.exists():
        return {}

    source_by_filename = {file_path.name: source for source, file_path in INSTRUCTION_FILES.items()}
    raw_data = json.loads(path.read_text())
    fixed_items: dict[tuple[str, int], tuple[dict[str, str], ...]] = {}

    for filename, entries in raw_data.items():
        source = source_by_filename.get(filename)
        if not source:
            continue
        for entry in entries:
            task_id = entry.get("index")
            if task_id is None:
                continue
            fixed_items[(source, int(task_id))] = tuple(entry.get("fixed_items", ()))

    return fixed_items


def attach_fixed_items(
    instructions: list[Instruction],
    fixed_items: dict[tuple[str, int], tuple[dict[str, str], ...]],
) -> list[Instruction]:
    return [
        Instruction(
            source=item.source,
            task_id=item.task_id,
            instruction=item.instruction,
            bddl_file=item.bddl_file,
            fixtures=item.fixtures,
            objects=item.objects,
            objects_of_interest=item.objects_of_interest,
            fixed_items=fixed_items.get((item.source, item.task_id), ()),
        )
        for item in instructions
    ]


def is_movable_object(object_class: str) -> bool:
    return object_class not in FIXTURE_CLASSES and object_class not in HARD_EXCLUDED_OBJECTS


def normalize_location(instruction: str, target_classes: Iterable[str]) -> str | None:
    text = instruction.lower()
    targets = set(target_classes)

    if "in the basket" in text or "place it in the basket" in text:
        return "basket.inside"
    if "on top of it" in text and "cabinet" in text:
        cabinet = "white_cabinet" if "white_cabinet" in targets else "wooden_cabinet"
        return f"{cabinet}.top_surface"
    if "bottom drawer" in text or "middle drawer" in text or "middle layer" in text or "top drawer" in text or "top layer" in text:
        side = "white_cabinet" if "white_cabinet" in targets else "wooden_cabinet"
    if "bottom drawer" in text:
        return f"{side}.bottom_drawer.inside"
    if "middle drawer" in text or "middle layer" in text:
        return f"{side}.middle_drawer.inside"
    if "top drawer" in text or "top layer" in text:
        if " at the front " in text:
            return f"{side}.top_drawer.front_side"
        if " at the back " in text:
            return f"{side}.top_drawer.back_side"
        return f"{side}.top_drawer.inside"
    if "in the microwave" in text:
        return "microwave.inside"
    if "on the stove" in text or " on it" in text and "stove" in text:
        return "flat_stove.surface"
    if "on the cabinet shelf" in text:
        return "wooden_two_layer_shelf.top_shelf"
    if "under the cabinet shelf" in text:
        return "wooden_two_layer_shelf.bottom_shelf"
    if "on top of the cabinet" in text or "on the wooden cabinet" in text:
        cabinet = "white_cabinet" if "white_cabinet" in targets else "wooden_cabinet"
        return f"{cabinet}.top_surface"
    if "to the right of the plate" in text:
        return "plate.right_side"
    if "on the left plate" in text:
        return "plate.left.center"
    if "on the right plate" in text:
        return "plate.right.center"
    if "on the plate" in text or "place it on the plate" in text:
        return "plate.center"
    if "in the tray" in text or "put it in the tray" in text or "place it in the tray" in text:
        return "wooden_tray.inside"

    for target in sorted(targets):
        if target in DISALLOWED_TARGET_CLASSES:
            continue
        if any(hint in target for hint in TARGET_HINTS):
            return f"{target}.surface"
    return None


def normalize_fixed_item_location(fixed_items: Iterable[dict[str, str]]) -> str | None:
    placement_relations = {
        "in",
        "on",
        "under",
        "to the right of",
        "to the left of",
        "to the front of",
        "to the back of",
    }

    for fixed_item in fixed_items:
        relation = fixed_item.get("relation", "")
        if relation not in placement_relations:
            continue

        target_class = fixed_item.get("type") or fixed_item.get("item", "")
        target_class = target_class.replace(" ", "_")
        detail = fixed_item.get("detail", "").lower()

        location = location_from_fixed_item(target_class, relation, detail)
        if location:
            return location

    return None


def location_from_fixed_item(target_class: str, relation: str, detail: str) -> str | None:
    if target_class == "basket":
        return "basket.inside"
    if target_class == "microwave":
        return "microwave.inside"
    if target_class == "flat_stove" or target_class == "stove":
        return "flat_stove.surface"
    if target_class in {"white_cabinet", "wooden_cabinet", "cabinet"}:
        cabinet = "white_cabinet" if target_class == "white_cabinet" else "wooden_cabinet"
        if relation == "on" and ("top drawer" in detail or "top layer" in detail):
            return f"{cabinet}.top_surface"
        if "bottom drawer" in detail:
            return f"{cabinet}.bottom_drawer.inside"
        if "middle drawer" in detail or "middle layer" in detail:
            return f"{cabinet}.middle_drawer.inside"
        if "top drawer" in detail or "top layer" in detail:
            if "front" in detail:
                return f"{cabinet}.top_drawer.front_side"
            if "back" in detail:
                return f"{cabinet}.top_drawer.back_side"
            return f"{cabinet}.top_drawer.inside"
        if relation == "under":
            return f"{cabinet}.under"
        return f"{cabinet}.top_surface"
    if target_class in {"wooden_two_layer_shelf", "shelf"}:
        if relation == "under":
            return "wooden_two_layer_shelf.bottom_shelf"
        if relation == "on" and "top" in detail:
            return "wooden_two_layer_shelf.top_surface"
        return "wooden_two_layer_shelf.top_shelf"
    if target_class == "plate":
        if relation == "to the right of":
            return "plate.right_side"
        if relation == "to the left of":
            return "plate.left_side"
        if relation == "to the front of":
            return "plate.front_side"
        if relation == "to the back of":
            return "plate.back_side"
        if "left plate" in detail:
            return "plate.left.center"
        if "right plate" in detail:
            return "plate.right.center"
        if "front" in detail:
            return "plate.front_side"
        if "back" in detail:
            return "plate.back_side"
        if "left" in detail:
            return "plate.left_side"
        if "right" in detail:
            return "plate.right_side"
        return "plate.center"
    if target_class == "wooden_tray":
        return "wooden_tray.inside"
    
    if target_class in DISALLOWED_TARGET_CLASSES:
        return None
    if any(hint in target_class for hint in TARGET_HINTS):
        return f"{target_class}.surface"
    return None


def canonical_location(location: str | None) -> str | None:
    if location is None:
        return None
    return CANONICAL_LOCATION_ALIASES.get(location, location)


def unique_stable_placement_options() -> list[dict[str, str]]:
    options = []
    seen = set()
    for option in STABLE_PLACEMENT_OPTIONS:
        location = canonical_location(option["location"])
        if location in seen:
            continue
        seen.add(location)
        normalized = dict(option)
        normalized["location"] = location
        options.append(normalized)
    return options


def stable_option_by_location() -> dict[str, dict[str, str]]:
    return {option["location"]: option for option in unique_stable_placement_options()}


def collect_records(instructions: list[Instruction]) -> dict[str, ObjectRecord]:
    records: dict[str, ObjectRecord] = {}

    for item in instructions:
        entity_classes = {**item.fixtures, **item.objects}
        interest_classes = [entity_classes[name] for name in item.objects_of_interest if name in entity_classes]
        movable = [cls for cls in interest_classes if is_movable_object(cls)]
        target_classes = [
            cls for cls in interest_classes
            if cls in FIXTURE_CLASSES and cls not in DISALLOWED_TARGET_CLASSES
        ]
        location = normalize_fixed_item_location(item.fixed_items)
        if location is None:
            location = normalize_location(item.instruction, target_classes)
        location = canonical_location(location)

        for object_class in item.objects.values():
            if is_movable_object(object_class):
                records.setdefault(object_class, ObjectRecord(object_class)).sources.add(item.source)

        for object_class in movable:
            record = records.setdefault(object_class, ObjectRecord(object_class))
            record.sources.add(item.source)
            record.action_count += 1
            if item.source == "libero_spatial":
                record.spatial_count += 1
            if location == "basket.inside":
                record.basket_count += 1

            if location:
                existing = record.placements.get(location)
                if existing:
                    existing.count += 1
                else:
                    option = stable_option_by_location().get(location, {})
                    record.placements[location] = PlacementEvidence(
                        location=location,
                        source=item.source,
                        task_id=item.task_id,
                        instruction=item.instruction,
                        bddl_file=item.bddl_file,
                        hint=option.get("hint", ""),
                        fixed_item=option.get("fixed_item", ""),
                    )

    return {key: value for key, value in records.items() if value.placements}


def add_stable_hint_placements(records: dict[str, ObjectRecord]) -> None:
    for record in records.values():
        for option in unique_stable_placement_options():
            location = option["location"]
            if location in record.placements:
                continue
            record.placements[location] = PlacementEvidence(
                location=location,
                source="synthetic_stable_hint",
                task_id=-1,
                instruction=option["hint"],
                bddl_file="",
                hint=option["hint"],
                fixed_item=option["fixed_item"],
                coordinate_pending=True,
            )


def rotated_stable_locations(user_id: int) -> list[str]:
    locations = [option["location"] for option in unique_stable_placement_options()]
    rotation = user_id % len(locations)
    return locations[rotation:] + locations[:rotation]


def best_placement(
    record: ObjectRecord,
    used_locations: set[str],
    preferred_locations: Iterable[str] = (),
) -> PlacementEvidence:
    preferred_rank = {location: idx for idx, location in enumerate(preferred_locations)}
    fallback_rank = len(preferred_rank)
    placements = sorted(
        record.placements.values(),
        key=lambda p: (
            p.location in used_locations,
            preferred_rank.get(p.location, fallback_rank),
            p.coordinate_pending,
            -p.count,
            p.location,
        ),
    )
    return placements[0]


def object_priority(record: ObjectRecord) -> tuple[int, int, int, str]:
    has_basket = 1 if record.basket_count else 0
    return (has_basket, record.spatial_count, record.action_count, record.object_class)


def find_belonging_sets(records: dict[str, ObjectRecord], num_profiles: int, objects_per_profile: int) -> list[tuple[str, ...]]:
    candidates = sorted(records, key=lambda obj: object_priority(records[obj]), reverse=True)
    all_combos = list(combinations(candidates, objects_per_profile))

    def combo_priority(combo: tuple[str, ...]) -> tuple[int, int, int, str]:
        basket = sum(records[obj].basket_count > 0 for obj in combo)
        spatial = sum(records[obj].spatial_count for obj in combo)
        action = sum(records[obj].action_count for obj in combo)
        return (basket, spatial, action, "|".join(combo))

    all_combos.sort(key=combo_priority, reverse=True)
    selected: list[tuple[str, ...]] = []

    for combo in all_combos:
        if all(len(set(combo) & set(existing)) <= 1 for existing in selected):
            selected.append(combo)
            if len(selected) == num_profiles:
                return selected

    raise ValueError(
        f"Could only build {len(selected)} profiles with pairwise overlap <= 1; "
        f"requested {num_profiles}."
    )


def make_linear_layout(objects: tuple[str, ...], user_id: int) -> list[dict[str, object]]:
    return [
        {"slot": idx, "position": f"linear_slot_{idx}", "object": obj}
        for idx, obj in enumerate(objects)
    ]


def center_outward_order(items: list[str]) -> list[str]:
    center_left = (len(items) - 1) // 2
    center_right = len(items) // 2
    indices: list[int] = []
    for offset in range(len(items)):
        left = center_left - offset
        right = center_right + offset
        if left >= 0:
            indices.append(left)
        if right != left and right < len(items):
            indices.append(right)
    return [items[idx] for idx in indices]


def outer_to_center_order(items: list[str]) -> list[str]:
    indices: list[int] = []
    left = 0
    right = len(items) - 1
    while left <= right:
        indices.append(left)
        if right != left:
            indices.append(right)
        left += 1
        right -= 1
    return [items[idx] for idx in indices]


def order_objects(objects: tuple[str, ...], records: dict[str, ObjectRecord], user_id: int) -> tuple[str, list[str]]:
    rule = SEQUENCE_RULES[user_id % len(SEQUENCE_RULES)]
    ordered = list(objects)

    if rule in {"left_to_right", "front_to_back", "close_to_far"}:
        pass
    elif rule in {"right_to_left", "back_to_front", "far_to_close"}:
        ordered.reverse()
    elif rule == "center_outward":
        ordered = center_outward_order(ordered)
    elif rule == "outer_to_center":
        ordered = outer_to_center_order(ordered)
    elif rule == "alphabetical":
        ordered = sorted(ordered)
    elif rule == "reverse_alphabetical":
        ordered = sorted(ordered, reverse=True)
    elif rule == "evidence_high_to_low":
        ordered = sorted(ordered, key=lambda obj: records[obj].action_count, reverse=True)
    elif rule == "evidence_low_to_high":
        ordered = sorted(ordered, key=lambda obj: records[obj].action_count)

    rotation = user_id // len(SEQUENCE_RULES)
    if ordered:
        rotation %= len(ordered)
        ordered = ordered[rotation:] + ordered[:rotation]
    return rule, ordered


def instruction_signatures(
    instructions: list[Instruction],
) -> tuple[set[tuple[tuple[str, str], ...]], set[tuple[str, str]]]:
    scene_signatures: set[tuple[tuple[str, str], ...]] = set()
    object_placement_signatures: set[tuple[str, str]] = set()

    for item in instructions:
        entity_classes = {**item.fixtures, **item.objects}
        interest_classes = [entity_classes[name] for name in item.objects_of_interest if name in entity_classes]
        movable = [cls for cls in interest_classes if is_movable_object(cls)]
        if not movable:
            continue

        target_classes = [
            cls for cls in interest_classes
            if cls in FIXTURE_CLASSES and cls not in DISALLOWED_TARGET_CLASSES
        ]
        location = normalize_fixed_item_location(item.fixed_items)
        if location is None:
            location = normalize_location(item.instruction, target_classes)
        if location is None:
            continue

        scene_signature = tuple(sorted((object_class, location) for object_class in movable))
        scene_signatures.add(scene_signature)
        object_placement_signatures.update(scene_signature)

    return scene_signatures, object_placement_signatures


def scene_name_from_bddl(bddl_file: str) -> str | None:
    match = re.match(r"([A-Z]+(?:_[A-Z]+)*_SCENE\d+)", bddl_file)
    return match.group(1) if match else None


def scene_fixed_classes(instruction: Instruction) -> set[str]:
    classes = set(instruction.fixtures.values()) | set(instruction.objects.values())
    if "white_cabinet" in classes or "wooden_cabinet" in classes:
        classes.add("cabinet")
    if "wooden_two_layer_shelf" in classes:
        classes.add("shelf")
    return classes


def scene_supports_fixed_item(fixed_item: str, fixed_classes: set[str]) -> bool:
    if fixed_item == "cabinet":
        return "cabinet" in fixed_classes
    if fixed_item == "wooden_two_layer_shelf":
        return "wooden_two_layer_shelf" in fixed_classes or "shelf" in fixed_classes
    return fixed_item in fixed_classes


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


def normalized_spawn_entries(
    graspable_objects: list[str],
    existing: dict[str, object],
    location: str,
) -> dict[str, object]:
    entries = {}
    existing_anchor = existing.get("_anchor")
    if isinstance(existing_anchor, dict) and all(key in existing_anchor for key in POSE_KEYS):
        entries["_anchor"] = {key: round(float(existing_anchor[key]), 4) for key in POSE_KEYS}
    elif location in CENTER_FIXED_LOCATIONS:
        entries["_anchor"] = {key: 0.0 for key in POSE_KEYS}

    for obj in graspable_objects:
        value = existing.get(obj, "N/A")
        if isinstance(value, dict) and all(key in value for key in POSE_KEYS):
            entries[obj] = {key: round(float(value[key]), 4) for key in POSE_KEYS}
        else:
            entries[obj] = "N/A"
    return entries


def write_possible_spawn_positions(
    instructions: list[Instruction],
    records: dict[str, ObjectRecord],
    spawn_dir: Path,
) -> int:
    graspable_objects = sorted(records)
    scene_classes: dict[str, set[str]] = {}
    for instruction in instructions:
        scene = scene_name_from_bddl(instruction.bddl_file)
        if not scene:
            continue
        scene_classes.setdefault(scene, set()).update(scene_fixed_classes(instruction))

    written = 0
    for scene in sorted(scene_classes):
        scene_dir = spawn_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        for option in unique_stable_placement_options():
            if not scene_supports_fixed_item(option["fixed_item"], scene_classes[scene]):
                continue
            path = scene_dir / f"{option['location']}.json"
            if path.exists():
                continue
            existing = {}
            entries = normalized_spawn_entries(graspable_objects, existing, option["location"])
            path.write_text(spawn_json_text(entries))
            written += 1
    return written


def generate_profiles(
    num_profiles: int = 23,
    objects_per_profile: int = 4,
    fixed_items_path: Path = FIXED_ITEMS_PATH,
    spawn_positions_dir: Path = DEFAULT_SPAWN_POSITIONS_DIR,
) -> dict[str, object]:
    instructions: list[Instruction] = []
    for source, path in INSTRUCTION_FILES.items():
        instructions.extend(read_instructions(path, source))
    fixed_items = load_fixed_items(fixed_items_path)
    instructions = attach_fixed_items(instructions, fixed_items)

    records = collect_records(instructions)
    add_stable_hint_placements(records)
    if len(records) < objects_per_profile:
        raise ValueError(f"Not enough evidence-backed objects: found {len(records)}")

    real_scene_signatures, real_object_placement_signatures = instruction_signatures(instructions)
    spawn_position_files = write_possible_spawn_positions(instructions, records, spawn_positions_dir)
    belonging_sets = find_belonging_sets(records, num_profiles, objects_per_profile)
    profiles = []
    used_arrangement_signatures: set[tuple[tuple[str, str], ...]] = set()
    used_sequences: set[tuple[str, ...]] = set()

    for user_id, belongings in enumerate(belonging_sets):
        used_locations: set[str] = set()
        preferred_locations = rotated_stable_locations(user_id)
        placements = {}
        placement_exists_in_libero = {}
        placement_coordinate_pending = {}
        evidence = {}
        for obj in belongings:
            placement = best_placement(records[obj], used_locations, preferred_locations)
            used_locations.add(placement.location)
            placements[obj] = placement.location
            placement_exists_in_libero[obj] = (obj, placement.location) in real_object_placement_signatures
            placement_coordinate_pending[obj] = placement.coordinate_pending
            evidence[obj] = {
                "source": placement.source,
                "task_id": placement.task_id,
                "instruction": placement.instruction,
                "file": placement.bddl_file,
                "fixed_item": placement.fixed_item,
                "placement_hint": placement.hint,
                "placement_exists_in_libero": placement_exists_in_libero[obj],
                "coordinate_pending": placement.coordinate_pending,
            }

        arrangement_signature = tuple(sorted(placements.items()))
        scene_exists_in_libero = arrangement_signature in real_scene_signatures
        if arrangement_signature in used_arrangement_signatures:
            raise ValueError(f"Duplicate arrangement generated for user {user_id}")
        used_arrangement_signatures.add(arrangement_signature)

        sequence_rule, sequence_order = order_objects(belongings, records, user_id)
        sequence_signature = tuple(sequence_order)
        if sequence_signature in used_sequences:
            sequence_order = sequence_order[1:] + sequence_order[:1]
            sequence_signature = tuple(sequence_order)
        if sequence_signature in used_sequences:
            raise ValueError(f"Duplicate sequence generated for user {user_id}")
        used_sequences.add(sequence_signature)

        profiles.append(
            {
                "user_id": user_id,
                "belongings": list(belongings),
                "arrangement": {
                    "type": "final_placement",
                    "placements": placements,
                    "evidence": evidence,
                },
                "placement": placements,
                "placement_coordinate_pending": placement_coordinate_pending,
                "placement_exists_in_libero": placement_exists_in_libero,
                "scene_exists_in_libero": scene_exists_in_libero,
                "sequence": {
                    "type": "basket_linear_order",
                    "input_layout": make_linear_layout(belongings, user_id),
                    "rule": sequence_rule,
                    "order": sequence_order,
                    "target": "basket.inside",
                },
                "source": ["libero_spatial", "libero_10", "libero_90", "libero_object"],
            }
        )

    return {
        "metadata": {
            "num_profiles": num_profiles,
            "objects_per_profile": objects_per_profile,
            "hard_excluded_objects": sorted(HARD_EXCLUDED_OBJECTS),
            "disallowed_target_classes": sorted(DISALLOWED_TARGET_CLASSES),
            "belongings_overlap_limit": 1,
            "instruction_files": {key: str(value) for key, value in INSTRUCTION_FILES.items()},
            "fixed_items_file": str(fixed_items_path),
            "fixed_items_entries": len(fixed_items),
            "real_scene_signature_count": len(real_scene_signatures),
            "real_object_placement_signature_count": len(real_object_placement_signatures),
            "spawn_positions_dir": str(spawn_positions_dir),
            "spawn_position_files": spawn_position_files,
            "stable_placement_options": unique_stable_placement_options(),
        },
        "profiles": profiles,
    }


def validate_profiles(data: dict[str, object]) -> None:
    profiles = data["profiles"]
    seen_belongings = set()
    seen_arrangements = set()
    seen_sequences = set()

    for profile in profiles:
        belongings = tuple(profile["belongings"])
        if len(belongings) != len(set(belongings)):
            raise ValueError(f"Duplicate belongings inside user {profile['user_id']}")
        if any(obj in HARD_EXCLUDED_OBJECTS for obj in belongings):
            raise ValueError(f"Hard-excluded object used by user {profile['user_id']}")
        blocked_targets = tuple(DISALLOWED_TARGET_CLASSES)
        if any(loc.startswith(blocked_targets) for loc in profile["placement"].values()):
            raise ValueError(f"Disallowed target used by user {profile['user_id']}")
        if belongings in seen_belongings:
            raise ValueError(f"Duplicate belongings for user {profile['user_id']}")
        seen_belongings.add(belongings)

        arrangement = tuple(sorted(profile["placement"].items()))
        if arrangement in seen_arrangements:
            raise ValueError(f"Duplicate arrangement for user {profile['user_id']}")
        seen_arrangements.add(arrangement)

        sequence = tuple(profile["sequence"]["order"])
        if sequence in seen_sequences:
            raise ValueError(f"Duplicate sequence for user {profile['user_id']}")
        seen_sequences.add(sequence)

    for left, right in combinations(profiles, 2):
        overlap = set(left["belongings"]) & set(right["belongings"])
        if len(overlap) > 1:
            raise ValueError(
                f"Users {left['user_id']} and {right['user_id']} share too many belongings: {sorted(overlap)}"
            )


def write_outputs(data: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_json = output_dir / "profiles.json"
    profiles_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    summary_csv = output_dir / "profiles_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user_id",
                "belongings",
                "scene_exists_in_libero",
                "placement_exists_in_libero",
                "placement_coordinate_pending",
                "sequence_rule",
                "sequence_order",
                "placements",
                "placement_hints",
            ],
        )
        writer.writeheader()
        for profile in data["profiles"]:
            writer.writerow(
                {
                    "user_id": profile["user_id"],
                    "belongings": ";".join(profile["belongings"]),
                    "scene_exists_in_libero": profile["scene_exists_in_libero"],
                    "placement_exists_in_libero": ";".join(
                        f"{obj}->{exists}"
                        for obj, exists in profile["placement_exists_in_libero"].items()
                    ),
                    "placement_coordinate_pending": ";".join(
                        f"{obj}->{pending}"
                        for obj, pending in profile["placement_coordinate_pending"].items()
                    ),
                    "sequence_rule": profile["sequence"]["rule"],
                    "sequence_order": ";".join(profile["sequence"]["order"]),
                    "placements": ";".join(
                        f"{obj}->{loc}" for obj, loc in profile["placement"].items()
                    ),
                    "placement_hints": ";".join(
                        f"{obj}->{profile['arrangement']['evidence'][obj].get('placement_hint', '')}"
                        for obj in profile["placement"]
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LIBERO personnel profiles.")
    parser.add_argument("--num-profiles", type=int, default=23)
    parser.add_argument("--objects-per-profile", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-items", type=Path, default=FIXED_ITEMS_PATH)
    parser.add_argument("--spawn-positions-dir", type=Path, default=DEFAULT_SPAWN_POSITIONS_DIR)
    args = parser.parse_args()

    data = generate_profiles(
        num_profiles=args.num_profiles,
        objects_per_profile=args.objects_per_profile,
        fixed_items_path=args.fixed_items,
        spawn_positions_dir=args.spawn_positions_dir,
    )
    validate_profiles(data)
    write_outputs(data, args.output_dir)
    print(f"Wrote {len(data['profiles'])} profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
