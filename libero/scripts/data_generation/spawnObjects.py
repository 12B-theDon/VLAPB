#!/usr/bin/env python3
"""Batch-check physically stable LIBERO object spawn positions.

The script reads the extracted LIBERO metadata, expands object/fixed-location
combinations, drops each movable object from 10cm above the candidate pose, and
stores stable placements under ``possible_spawn_positions``. Unstable or failed
placements are written under ``holded_spawn_positions`` with diagnostic data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
for path in (TOOLS_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_profile_tools import (  # noqa: E402
    FINAL_SPEED_REVIEW_THRESHOLD,
    default_pose_for_location,
    simulate_pose,
    setup_logging,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    def tqdm(iterable, **_kwargs):
        return iterable


DOCS_DIR = VLAPB_LIBERO_ROOT / "docs"
DEFAULT_CONSISTENCY = DOCS_DIR / "libero_fixed_location_consistency.txt"
DEFAULT_SUMMARY = DOCS_DIR / "libero_objects_summary.md"
DEFAULT_OBJECTS = DOCS_DIR / "libero_objects.json"
DEFAULT_POSSIBLE_DIR = Path(__file__).resolve().parent / "possible_spawn_positions"
DEFAULT_HOLDED_DIR = Path(__file__).resolve().parent / "holded_spawn_positions"

FIXED_TYPE_ALIASES = {
    "white_cabinet": "cabinet",
    "wooden_cabinet": "cabinet",
    "flat_stove": "flat_stove",
    "wooden_tray": "wooden_tray",
    "desk_caddy": "desk_caddy",
    "wooden_two_layer_shelf": "wooden_two_layer_shelf",
}

LOCATION_BY_FIXED_AND_LABEL = {
    ("basket", "in"): "basket.inside",
    ("basket", "contain"): "basket.inside",
    ("desk_caddy", "front"): "desk_caddy.front_side",
    ("desk_caddy", "back"): "desk_caddy.back_side",
    ("desk_caddy", "left"): "desk_caddy.left_side",
    ("desk_caddy", "right"): "desk_caddy.right_side",
    ("flat_stove", "on"): "flat_stove.surface",
    ("flat_stove", "cook"): "flat_stove.surface",
    ("microwave", "in"): "microwave.inside",
    ("microwave", "contain"): "microwave.inside",
    ("microwave", "on"): "microwave.top_surface",
    ("plate", "on"): "plate.center",
    ("wooden_tray", "in"): "wooden_tray.inside",
    ("wooden_tray", "contain"): "wooden_tray.inside",
    ("wooden_two_layer_shelf", "top"): "wooden_two_layer_shelf.top_shelf",
    ("wooden_two_layer_shelf", "on"): "wooden_two_layer_shelf.top_shelf",
    ("wooden_two_layer_shelf", "bottom"): "wooden_two_layer_shelf.bottom_shelf",
    ("wooden_two_layer_shelf", "in"): "wooden_two_layer_shelf.bottom_shelf",
    ("wooden_cabinet", "top"): "cabinet.top_drawer.inside",
    ("white_cabinet", "top"): "cabinet.top_drawer.inside",
    ("wooden_cabinet", "bottom"): "cabinet.bottom_drawer.inside",
    ("white_cabinet", "bottom"): "cabinet.bottom_drawer.inside",
    ("wooden_cabinet", "in"): "cabinet.bottom_drawer.inside",
    ("white_cabinet", "in"): "cabinet.bottom_drawer.inside",
}

SIDE_LABELS = ("front", "back", "left", "right")
DRAWER_ONLY_LABELS = {"top", "bottom"}
TOP_BOTTOM_EXCLUDED = {"basket", "desk_caddy", "flat_stove", "kitchen_table", "living_room_table", "study_table", "plate", "wooden_tray"}
NON_FIXED_TARGET_TYPES = {"akita_black_bowl"}
FIXED_OBJECT_TYPES = {
    "basket",
    "desk_caddy",
    "flat_stove",
    "kitchen_table",
    "living_room_table",
    "microwave",
    "plate",
    "study_table",
    "white_cabinet",
    "wine_rack",
    "wooden_cabinet",
    "wooden_tray",
    "wooden_two_layer_shelf",
}

# Physically impossible / undesired combinations.
# Keyed by (fixed_type, fixed_side, label, object_type).
FORBIDDEN_CASES = {
    ("wooden_cabinet", "left", "left", "chefmate_8_frypan"),
    ("wooden_cabinet", "left", "left", "porcelain_mug"),
    ("white_cabinet", "left", "left", "chefmate_8_frypan"),
}
SAME_SIDE_BLOCKED_FIXTURES = {
    "wooden_cabinet",
    "white_cabinet",
    "microwave",
    "cabinet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate physical spawn positions for LIBERO objects.")
    parser.add_argument("--consistency", type=Path, default=DEFAULT_CONSISTENCY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--possible-dir", type=Path, default=DEFAULT_POSSIBLE_DIR)
    parser.add_argument("--holded-dir", type=Path, default=DEFAULT_HOLDED_DIR)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sim-seconds", type=float, default=5.0)
    parser.add_argument("--sim-fps", type=int, default=5)
    parser.add_argument("--drop-z", type=float, default=0.10)
    parser.add_argument("--max-cases", type=int, default=None, help="Smoke-test only the first N generated cases.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not rerun cases with an existing output JSON.")
    parser.add_argument("--include-existing-libero-pairs", action="store_true", help="Also simulate pairs already observed in LIBERO.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the candidate manifest; do not run MuJoCo.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_required_inputs(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    consistency = args.consistency.read_text(encoding="utf-8")
    summary = args.summary.read_text(encoding="utf-8")
    objects = json.loads(args.objects.read_text(encoding="utf-8"))
    return consistency, summary, objects


def scene_area(scene: str | None, dataset: str) -> str:
    if scene:
        if scene.startswith("KITCHEN_"):
            return "kitchen"
        if scene.startswith("LIVING_ROOM_"):
            return "living_room"
        if scene.startswith("STUDY_"):
            return "study"
    return dataset


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def unique_sorted(values: set[str]) -> list[str]:
    return sorted(value for value in values if value)


def load_movable_object_types(objects: dict[str, Any]) -> list[str]:
    movable = set()
    for task in objects.get("tasks", []):
        for item in task.get("objects", []):
            object_type = item.get("type")
            if object_type and object_type not in FIXED_OBJECT_TYPES:
                movable.add(object_type)
    return unique_sorted(movable)


def observed_existing_pairs(objects: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    existing = set()
    for coupling in objects.get("couplings", []):
        fixed_type = coupling.get("fixed_object", {}).get("type")
        object_type = coupling.get("graspable_object", {}).get("type")
        scene = coupling.get("scene")
        for label in normalized_labels(coupling):
            if fixed_type and object_type and scene:
                existing.add((scene, fixed_type, label, object_type))
    return existing


def normalized_labels(coupling: dict[str, Any]) -> set[str]:
    labels = set()
    relation = str(coupling.get("relation", "")).lower()
    if relation == "in":
        labels.add("in")
    elif relation == "on":
        labels.add("on")
    goal = coupling.get("goal_location") or {}
    region = goal.get("region") or {}
    haystack = " ".join(str(part) for part in (goal.get("reference", ""), region.get("name", ""), region.get("target", ""))).lower()
    for label in ("front", "back", "left", "right", "top", "bottom"):
        if label in haystack:
            labels.add(label)
    if "contain" in haystack:
        labels.add("in")
    if "cook" in haystack:
        labels.add("on")
    return labels


def fixed_contexts(objects: dict[str, Any]) -> list[dict[str, str]]:
    contexts = {}
    for coupling in objects.get("couplings", []):
        fixed = coupling.get("fixed_object", {})
        fixed_type = fixed.get("type")
        if not fixed_type or fixed_type in NON_FIXED_TARGET_TYPES:
            continue
        if fixed_type == "wine_rack":
            # User policy: do not use wine rack for generated spawn candidates.
            continue
        scene = coupling.get("scene")
        table = coupling.get("table")
        dataset = coupling.get("dataset", "")
        fixed_side = str(coupling.get("fixed_object", {}).get("side") or "")
        if not scene or not table:
            continue
        key = (scene, table, fixed_type, fixed_side)
        contexts[key] = {
            "scene": scene,
            "table": table,
            "area": scene_area(scene, dataset),
            "fixed_type": fixed_type,
            "fixed_side": fixed_side,
        }
    return sorted(contexts.values(), key=lambda item: (item["scene"], item["fixed_type"], item["fixed_side"], item["table"]))


def is_forbidden_case(scene: str, fixed_type: str, fixed_side: str, label: str, object_type: str) -> bool:
    _ = scene  # Scene is currently not needed because the rule is structural (left-left).
    if (fixed_type, fixed_side, label, object_type) in FORBIDDEN_CASES:
        return True
    # General physical usability rule:
    # for bulky fixed items (cabinet/microwave), avoid same-side placement
    # (left-left and right-right).
    if fixed_type in SAME_SIDE_BLOCKED_FIXTURES and fixed_side in {"left", "right"} and label in {"left", "right"}:
        if fixed_side == label:
            return True
    return False


def candidate_labels_for_fixed(fixed_type: str) -> list[str]:
    labels = {"on", "in", *SIDE_LABELS}
    if fixed_type not in TOP_BOTTOM_EXCLUDED:
        labels.update(DRAWER_ONLY_LABELS)
    if fixed_type == "flat_stove":
        labels.add("cook")
    if fixed_type in {"basket", "wooden_tray", "desk_caddy"}:
        labels.add("contain")
    return sorted(labels)


def location_for(fixed_type: str, label: str) -> str | None:
    label = "in" if label == "contain" else "on" if label == "cook" else label
    direct = LOCATION_BY_FIXED_AND_LABEL.get((fixed_type, label))
    if direct:
        return direct
    canonical = FIXED_TYPE_ALIASES.get(fixed_type, fixed_type)
    if label in SIDE_LABELS:
        if label == "front":
            return f"{canonical}.back_side"
        if label == "back":
            return f"{canonical}.front_side"
        return f"{canonical}.{label}_side"
    if label == "on":
        if fixed_type.endswith("_table"):
            return f"{fixed_type}.center"
        if fixed_type == "wine_rack":
            return "wine_rack.top_side"
    return None


def initial_pose(location: str, drop_z: float) -> dict[str, float]:
    pose = default_pose_for_location(location)
    pose["z"] = round(float(pose.get("z", 0.0)) + drop_z, 4)
    return pose


def generate_cases(objects: dict[str, Any], include_existing: bool, drop_z: float) -> list[dict[str, Any]]:
    movable_types = load_movable_object_types(objects)
    contexts = fixed_contexts(objects)
    existing = observed_existing_pairs(objects)
    cases = []
    seen = set()
    for context in contexts:
        fixed_type = context["fixed_type"]
        fixed_side = context.get("fixed_side", "")
        for label in candidate_labels_for_fixed(fixed_type):
            location = location_for(fixed_type, label)
            if not location:
                continue
            normalized_label = "in" if label == "contain" else "on" if label == "cook" else label
            for object_type in movable_types:
                if is_forbidden_case(context["scene"], fixed_type, fixed_side, normalized_label, object_type):
                    continue
                key = (context["scene"], fixed_type, normalized_label, object_type)
                case_id = f"{context['scene']}__{fixed_type}__{normalized_label}__{object_type}"
                if case_id in seen:
                    continue
                seen.add(case_id)
                cases.append(
                    {
                        "id": case_id,
                        "scene": context["scene"],
                        "table": context["table"],
                        "area": context["area"],
                        "fixed_type": fixed_type,
                        "label": normalized_label,
                        "location": location,
                        "object_type": object_type,
                        "pose": initial_pose(location, drop_z),
                        "already_in_libero": key in existing,
                        "skip_reason": "existing_libero_pair" if key in existing and not include_existing else None,
                    }
                )
    return cases


def output_path(root: Path, case: dict[str, Any]) -> Path:
    return (
        root
        / safe_name(case["scene"])
        / safe_name(case["fixed_type"])
        / f"{safe_name(case['label'])}.json"
    )


def append_case_result(path: Path, case: dict[str, Any], result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    key = case["object_type"]
    data[key] = result
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def skipped_result(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": case["skip_reason"],
        "scene": case["scene"],
        "table": case["table"],
        "fixed_type": case["fixed_type"],
        "location": case["location"],
        "label": case["label"],
        "object_type": case["object_type"],
        "pose": case["pose"],
        "timestamp": time.time(),
    }


def run_case(case: dict[str, Any], args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    result = simulate_pose(
        scene=case["scene"],
        location=case["location"],
        object_type=case["object_type"],
        pose=case["pose"],
        camera_name=args.camera_name,
        image_size=args.image_size,
        sim_seconds=args.sim_seconds,
        sim_fps=args.sim_fps,
        force_safe_object_init=True,
    )
    stable = bool(result.get("velocity_stable")) and float(result.get("final_speed", 999.0)) <= FINAL_SPEED_REVIEW_THRESHOLD
    root = args.possible_dir if stable else args.holded_dir
    payload = {
        **case,
        "status": "possible" if stable else "holded",
        "velocity_threshold": FINAL_SPEED_REVIEW_THRESHOLD,
        "drop_z": args.drop_z,
        "relative_scene_table_fixed_pose": result.get("final_anchor_pose"),
        "actual_pose": result.get("final_pose"),
        "start_pose": result.get("start_pose"),
        "simulation": result,
        "timestamp": time.time(),
    }
    return output_path(root, case), payload


def write_manifest(root: Path, name: str, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress_counts_text(counts: dict[str, int]) -> str:
    return (
        f"ok={counts.get('possible', 0)} "
        f"hold={counts.get('holded', 0)} "
        f"fail={counts.get('failed', 0)} "
        f"skip={counts.get('skipped_existing_libero_pair', 0) + counts.get('already_done', 0)}"
    )


def update_progress(progress: Any, counts: dict[str, int]) -> None:
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(progress_counts_text(counts), refresh=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    consistency_text, summary_text, objects = read_required_inputs(args)
    cases = generate_cases(objects, args.include_existing_libero_pairs, args.drop_z)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    manifest = {
        "source_files": {
            "consistency": str(args.consistency),
            "summary": str(args.summary),
            "objects": str(args.objects),
        },
        "read_lengths": {
            "consistency_chars": len(consistency_text),
            "summary_chars": len(summary_text),
        },
        "case_count": len(cases),
        "sim_seconds": args.sim_seconds,
        "drop_z": args.drop_z,
        "cases": cases,
    }
    write_manifest(args.possible_dir, "_candidate_manifest.json", manifest)

    counts = defaultdict(int)
    progress = tqdm(
        cases,
        desc="Validating spawn positions",
        unit="case",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for index, case in enumerate(progress, start=1):
        possible_path = output_path(args.possible_dir, case)
        holded_path = output_path(args.holded_dir, case)
        if args.skip_existing and (possible_path.exists() or holded_path.exists()):
            counts["already_done"] += 1
            update_progress(progress, counts)
            continue
        if args.dry_run:
            counts["dry_run"] += 1
            update_progress(progress, counts)
            continue
        if case["skip_reason"]:
            append_case_result(possible_path, case, skipped_result(case))
            counts["skipped_existing_libero_pair"] += 1
            update_progress(progress, counts)
            continue
        try:
            path, result = run_case(case, args)
        except Exception as exc:
            path = output_path(args.holded_dir, case)
            result = {
                **case,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "drop_z": args.drop_z,
                "timestamp": time.time(),
            }
        append_case_result(path, case, result)
        counts[str(result.get("status", "unknown"))] += 1
        if hasattr(progress, "set_postfix_str"):
            update_progress(progress, counts)
        else:
            print(f"[{index}/{len(cases)}] {case['id']} -> {result.get('status')} ({path})", flush=True)

    stats = {
        "case_count": len(cases),
        "counts": dict(sorted(counts.items())),
        "possible_dir": str(args.possible_dir),
        "holded_dir": str(args.holded_dir),
        "timestamp": time.time(),
    }
    write_manifest(args.possible_dir, "_run_stats.json", stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
