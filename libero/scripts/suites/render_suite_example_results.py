#!/usr/bin/env python3
"""Render representative before/after images for VLAPB suite tasks.

The script selects one generated episode for each VLAPB task split and writes
two images: the original initial state and a goal-state approximation where the
goal predicates are moved into the BDDL init section.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = SCRIPT_DIR.parents[1]
VLAPB_PROJECT_ROOT = VLAPB_LIBERO_ROOT.parent
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from test_env_reconfiguration import (  # noqa: E402
    TASK_MAPPING,
    build_env_kwargs,
    get_action_dim,
    render_task_image,
    reset_with_retries,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
        return iterable


DEFAULT_SUITES_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"
DEFAULT_OUTPUT_DIR = DEFAULT_SUITES_ROOT / "example_results"
TYPE4_TOP_DRAWER_BDDL = (
    Path("/home/artemis/Documents/LIBERO/libero/libero/bddl_files/libero_90")
    / "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.bddl"
)
SEQUENCE_TYPE1_VISUAL_BDDL = (
    DEFAULT_SUITES_ROOT / "sequences" / "type2" / "bddl_files" / "sequences_type2_000004.bddl"
)
TASK_SPLITS = (
    ("belongings", "type1"),
    ("belongings", "type2"),
    ("belongings", "type3"),
    ("belongings", "adaptability"),
    ("belongings", "multiuser"),
    ("placements", "type1"),
    ("placements", "type2"),
    ("placements", "type3"),
    ("placements", "type4"),
    ("placements", "adaptability"),
    ("placements", "multiuser"),
    ("placements", "consistency"),
    ("sequences", "type1"),
    ("sequences", "type2"),
    ("sequences", "adaptability"),
    ("sequences", "consistency"),
)


@dataclass(frozen=True)
class RenderRecord:
    task_index: int
    suite: str
    split: str
    episode_id: str
    bddl_file: str
    metadata_file: str
    before_image: str
    after_image: str
    after_bddl_file: str
    sequence_step_images: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites-root", type=Path, default=DEFAULT_SUITES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format="[%(levelname)s] %(message)s")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_metadata_files(suites_root: Path, suite: str, split: str) -> list[Path]:
    metadata_dir = suites_root / suite / split / "metadata"
    candidates = sorted(metadata_dir.glob("*.json"))
    if suite == "placements" and split in {"type1", "type3"}:
        def priority(path: Path) -> tuple[int, str]:
            metadata = read_json(path)
            pref = metadata.get("preference", {})
            if pref.get("fixed_type") == "basket" and pref.get("object_type") == "ketchup":
                return (0, path.name)
            if pref.get("fixed_type") == "basket":
                return (1, path.name)
            return (2, path.name)

        candidates.sort(key=priority)
    if suite == "placements" and split == "type4":
        def priority(path: Path) -> tuple[int, str]:
            metadata = read_json(path)
            pref = metadata.get("preference", {})
            if pref.get("fixed_type") in {"wooden_cabinet", "white_cabinet", "cabinet"} and pref.get("label") == "top":
                return (0, path.name)
            if pref.get("fixed_type") in {"wooden_cabinet", "white_cabinet", "cabinet"} and pref.get("label") == "front":
                return (1, path.name)
            return (2, path.name)

        candidates.sort(key=priority)
    if suite == "sequences":
        large_penalty = {
            "chefmate_8_frypan": 10,
            "moka_pot": 5,
            "black_book": 2,
            "yellow_book": 2,
            "akita_black_bowl": 3,
            "porcelain_mug": 3,
            "red_coffee_mug": 3,
            "white_bowl": 3,
            "white_yellow_mug": 3,
            "wine_bottle": 5,
        }

        def priority(path: Path) -> tuple[int, str]:
            metadata = read_json(path)
            objects = metadata.get("graspable_objects", [])
            penalty = sum(large_penalty.get(obj, 0) for obj in objects)
            return (penalty, path.name)

        candidates.sort(key=priority)
    return candidates


def bddl_for_metadata(metadata_file: Path, suites_root: Path, suite: str, split: str) -> Path:
    if suite == "placements" and split == "type4" and TYPE4_TOP_DRAWER_BDDL.exists():
        return TYPE4_TOP_DRAWER_BDDL
    if suite == "sequences" and split == "type1" and SEQUENCE_TYPE1_VISUAL_BDDL.exists():
        return SEQUENCE_TYPE1_VISUAL_BDDL
    metadata = read_json(metadata_file)
    bddl_path = Path(str(metadata.get("bddl_file", "")))
    if bddl_path.exists():
        return bddl_path
    return suites_root / suite / split / "bddl_files" / f"{metadata_file.stem}.bddl"


def extract_init_span(text: str) -> tuple[int, int, str]:
    marker = "(:init"
    start = text.index(marker)
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1, text[start : index + 1]
    raise ValueError("Could not find complete :init section")


def extract_section_span(text: str, section_name: str) -> tuple[int, int, str]:
    marker = f"(:{section_name}"
    start = text.index(marker)
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1, text[start : index + 1]
    raise ValueError(f"Could not find complete :{section_name} section")


def extract_goal_terms(text: str) -> list[tuple[str, str, str]]:
    goal_start = text.index("(:goal")
    goal_text = text[goal_start:]
    terms = re.findall(r"\((In|On)\s+([^\s()]+)\s+([^\s()]+)\)", goal_text)
    if not terms:
        raise ValueError("No In/On goal terms found")
    return terms


def replace_goal_with_render_safe_goal(text: str) -> str:
    init_terms = re.findall(r"\((In|On)\s+([^\s()]+)\s+([^\s()]+)\)", extract_init_span(text)[2])
    if not init_terms:
        return text
    _pred, obj, _target = init_terms[-1]
    target = init_terms[0][1]
    start, end, _section = extract_section_span(text, "goal")
    replacement = f"(:goal\n    (And (On {obj} {target}))\n  )"
    return text[:start] + replacement + text[end:]


def parse_region_centers(text: str) -> dict[str, tuple[str, float, float]]:
    centers: dict[str, tuple[str, float, float]] = {}
    _start, _end, regions_section = extract_section_span(text, "regions")
    region_re = re.compile(
        r"\(([A-Za-z0-9_]+)\s+"
        r"\(:target\s+([A-Za-z0-9_]+)\).*?"
        r"\(:ranges\s+\(\s+\(([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\)",
        re.DOTALL,
    )
    for name, target, x1, y1, x2, y2 in region_re.findall(regions_section):
        centers[name] = (
            target,
            (float(x1) + float(x2)) / 2.0,
            (float(y1) + float(y2)) / 2.0,
        )
    return centers


def parse_entity_centers(text: str) -> tuple[str, dict[str, tuple[float, float]]]:
    centers = parse_region_centers(text)
    _start, _end, init_section = extract_init_span(text)
    entity_centers: dict[str, tuple[float, float]] = {}
    workspace = "main_table"
    for entity, reference in re.findall(r"\(On\s+([^\s()]+)\s+([^\s()]+)\)", init_section):
        for region_name, (target, x, y) in centers.items():
            if reference == f"{target}_{region_name}":
                entity_centers[entity] = (x, y)
                workspace = target
                break
    return workspace, entity_centers


def fixture_from_goal_target(goal_target: str, entity_centers: dict[str, tuple[float, float]]) -> tuple[float, float, str]:
    if goal_target in entity_centers:
        x, y = entity_centers[goal_target]
        return x, y, "center"
    suffix_offsets = {
        "_left_side": (0.0, -0.11, "left"),
        "_right_side": (0.0, 0.11, "right"),
        "_front_side": (0.11, 0.0, "front"),
        "_back_side": (-0.11, 0.0, "back"),
        "_contain_region": (0.0, 0.0, "contain"),
        "_cook_region": (0.0, 0.0, "cook"),
        "_top_region": (0.0, 0.0, "top"),
        "_bottom_region": (0.0, 0.0, "bottom"),
        "_heating_region": (0.0, 0.0, "heating"),
    }
    for suffix, (dx, dy, label) in suffix_offsets.items():
        if goal_target.endswith(suffix):
            fixture = goal_target[: -len(suffix)]
            x, y = entity_centers.get(fixture, (0.0, 0.0))
            return x + dx, y + dy, label
    return 0.0, 0.0, "fallback"


def append_regions(text: str, region_blocks: list[str]) -> str:
    start, end, section = extract_section_span(text, "regions")
    replacement = section[:-1] + "\n" + "\n".join(region_blocks) + "\n    )"
    return text[:start] + replacement + text[end:]


def object_ids_by_type(text: str) -> dict[str, list[str]]:
    objects_section = extract_section_span(text, "objects")[2]
    mapping: dict[str, list[str]] = {}
    for line in objects_section.splitlines():
        if " - " not in line:
            continue
        ids_part, object_type = line.split(" - ", 1)
        object_type = object_type.strip()
        for object_id in ids_part.strip().split():
            mapping.setdefault(object_type, []).append(object_id)
    return mapping


def sequence_goal_object_ids(text: str) -> list[str]:
    return [obj for pred, obj, target in extract_goal_terms(text) if pred == "In" and "wooden_tray" in target]


def sanitize_type4_top_drawer_bddl(text: str) -> str:
    text = text.replace("    butter_1 butter_2 - butter", "    butter_1 - butter")
    text = re.sub(r"\n\s+\(On butter_2 kitchen_table_butter_back_init_region\)", "", text)
    return text


def spread_sequence_objects(text: str, mode: str) -> str:
    goal_terms = extract_goal_terms(text)
    if len(goal_terms) <= 1 or not all(pred == "In" for pred, _obj, _target in goal_terms):
        return text

    init_start, init_end, init_section = extract_init_span(text)
    workspace, entity_centers = parse_entity_centers(text)
    tray_id = goal_terms[0][2].split("_contain_region", 1)[0]
    tray_x, tray_y = entity_centers.get(tray_id, (0.0, 0.0))
    objects = [obj for _pred, obj, _target in goal_terms]
    if mode == "before":
        offsets = ((0.22, 0.15), (-0.23, -0.08), (0.19, -0.20))
        label = "sequence_before"
        region_target = workspace
        base_x = tray_x
        base_y = tray_y
    else:
        offsets = ((0.00, 0.045), (-0.055, -0.030), (0.055, -0.030))
        label = "sequence_after_on_tray"
        region_target = tray_id
        base_x = 0.0
        base_y = 0.0

    kept_lines = []
    for line in init_section.splitlines()[1:-1]:
        match = re.search(r"\((In|On)\s+([^\s()]+)\s+([^\s()]+)\)", line)
        if match and match.group(2) in set(objects):
            continue
        kept_lines.append(line)

    region_blocks = []
    init_lines = []
    for index, obj in enumerate(objects):
        dx, dy = offsets[index % len(offsets)]
        x = base_x + dx
        y = base_y + dy
        region_name = f"{label}_{obj}_region"
        region_blocks.append(
            "      ("
            f"{region_name}\n"
            f"          (:target {region_target})\n"
            "          (:ranges (\n"
            f"              ({x - 0.025:.4f} {y - 0.025:.4f} {x + 0.025:.4f} {y + 0.025:.4f})\n"
            "            )\n"
            "          )\n"
            "      )"
        )
        init_lines.append(f"    (On {obj} {region_target}_{region_name})")

    new_init = "(:init\n" + "\n".join(kept_lines + init_lines) + "\n  )"
    updated = text[:init_start] + new_init + text[init_end:]
    return append_regions(updated, region_blocks)


def render_sequence_after_image(
    bddl_file: Path,
    output_path: Path,
    camera_name: str,
    image_size: int,
    settle_steps: int,
    max_reset_attempts: int,
    object_ids_to_place: list[str],
) -> None:
    text = bddl_file.read_text(encoding="utf-8")
    tray_id = "wooden_tray_1"
    for _pred, _obj, target in extract_goal_terms(text):
        if "wooden_tray" in target:
            tray_id = target.split("_contain_region", 1)[0]
            break

    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    env = TASK_MAPPING[problem_name](**env_kwargs)
    try:
        reset_with_retries(env, max_attempts=max_reset_attempts)
        tray_qpos = env.sim.data.get_joint_qpos(f"{tray_id}_joint0").copy()
        tray_x, tray_y, tray_z = tray_qpos[:3]
        if len(object_ids_to_place) == 1:
            offsets = ((0.000, 0.000, 0.080),)
        elif len(object_ids_to_place) == 2:
            offsets = ((-0.055, -0.030, 0.074), (0.055, 0.030, 0.086))
        else:
            offsets = ((-0.085, -0.045, 0.072), (0.000, 0.000, 0.084), (0.085, 0.045, 0.096))
        for index, obj in enumerate(object_ids_to_place):
            joint_name = f"{obj}_joint0"
            try:
                qpos = env.sim.data.get_joint_qpos(joint_name).copy()
            except Exception:
                continue
            dx, dy, dz = offsets[index % len(offsets)]
            qpos[:3] = [tray_x + dx, tray_y + dy, tray_z + dz]
            env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.forward()
        _ = settle_steps  # Keep overridden sequence poses fixed for the representative after image.
        image = env.sim.render(
            camera_name=camera_name,
            width=image_size,
            height=image_size,
            depth=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image[::-1, :, ::-1])
    finally:
        env.close()


def render_objects_inside_fixture_image(
    bddl_file: Path,
    output_path: Path,
    camera_name: str,
    image_size: int,
    max_reset_attempts: int,
    object_ids: list[str],
    fixture_id: str,
    fixture_kind: str,
) -> None:
    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    env = TASK_MAPPING[problem_name](**env_kwargs)
    try:
        reset_with_retries(env, max_attempts=max_reset_attempts)
        fixture_qpos = env.sim.data.get_joint_qpos(f"{fixture_id}_joint0").copy()
        fixture_x, fixture_y, fixture_z = fixture_qpos[:3]
        if fixture_kind == "basket":
            offsets = ((0.000, -0.025, 0.105), (0.000, 0.035, 0.120), (0.035, 0.000, 0.115))
        else:
            offsets = ((0.000, 0.000, 0.070), (-0.030, 0.020, 0.080), (0.030, -0.020, 0.090))
        for index, object_id in enumerate(object_ids):
            joint_name = f"{object_id}_joint0"
            try:
                qpos = env.sim.data.get_joint_qpos(joint_name).copy()
            except Exception:
                continue
            dx, dy, dz = offsets[index % len(offsets)]
            qpos[:3] = [fixture_x + dx, fixture_y + dy, fixture_z + dz]
            qpos[3:] = [0.0, 0.0, 0.0, 1.0]
            env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.forward()
        image = env.sim.render(
            camera_name=camera_name,
            width=image_size,
            height=image_size,
            depth=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image[::-1, :, ::-1])
    finally:
        env.close()


def build_after_bddl(text: str) -> str:
    init_start, init_end, init_section = extract_init_span(text)
    goal_terms = extract_goal_terms(text)
    goal_objects = {obj for _pred, obj, _target in goal_terms}
    workspace, entity_centers = parse_entity_centers(text)

    init_lines = init_section.splitlines()
    kept_lines = []
    for line in init_lines[1:-1]:
        match = re.search(r"\((In|On)\s+([^\s()]+)\s+([^\s()]+)\)", line)
        if match and match.group(2) in goal_objects:
            continue
        kept_lines.append(line)

    synthetic_regions = []
    after_lines = []
    occupied: list[tuple[float, float]] = []
    multi_in_goal = len(goal_terms) > 1 and all(pred == "In" for pred, _obj, _target in goal_terms)
    sequence_offsets = ((0.00, 0.11), (-0.095, -0.04), (0.095, -0.04), (0.00, -0.12))
    for index, (_pred, obj, target) in enumerate(goal_terms):
        if (_pred == "In" and not multi_in_goal) or not (
            target.endswith("_left_side")
            or target.endswith("_right_side")
            or target.endswith("_front_side")
            or target.endswith("_back_side")
            or multi_in_goal
        ):
            after_lines.append(f"    ({_pred} {obj} {target})")
            continue
        x, y, label = fixture_from_goal_target(target, entity_centers)
        if multi_in_goal:
            dx, dy = sequence_offsets[index % len(sequence_offsets)]
            x += dx
            y += dy
            label = "sequence_done"
        else:
            x += (index % 2) * 0.045
            y += (index // 2) * 0.045
        while any(abs(x - ox) < 0.035 and abs(y - oy) < 0.035 for ox, oy in occupied):
            x += 0.04
        occupied.append((x, y))
        region_name = f"after_{obj}_{label}_region"
        synthetic_regions.append(
            "      ("
            f"{region_name}\n"
            f"          (:target {workspace})\n"
            "          (:ranges (\n"
            f"              ({x - 0.025:.4f} {y - 0.025:.4f} {x + 0.025:.4f} {y + 0.025:.4f})\n"
            "            )\n"
            "          )\n"
            "      )"
        )
        after_lines.append(f"    (On {obj} {workspace}_{region_name})")

    new_init = "(:init\n" + "\n".join(kept_lines + after_lines) + "\n  )"
    updated = text[:init_start] + new_init + text[init_end:]
    if synthetic_regions:
        return append_regions(updated, synthetic_regions)
    return updated


def render_pair(
    bddl_file: Path,
    before_bddl_path: Path,
    before_path: Path,
    after_path: Path,
    after_bddl_path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    suite: str,
    split: str,
) -> None:
    source_text = bddl_file.read_text(encoding="utf-8")
    if suite == "placements" and split == "type4" and bddl_file == TYPE4_TOP_DRAWER_BDDL:
        source_text = sanitize_type4_top_drawer_bddl(source_text)
    is_sequence = "/sequences/" in str(bddl_file)
    before_text = spread_sequence_objects(source_text, "before") if is_sequence else source_text
    before_bddl_path.write_text(replace_goal_with_render_safe_goal(before_text), encoding="utf-8")
    render_task_image(
        bddl_file=before_bddl_path,
        output_path=before_path,
        camera_name=args.camera_name,
        image_size=args.image_size,
        settle_steps=args.settle_steps,
        max_reset_attempts=args.max_reset_attempts,
    )
    if is_sequence:
        after_bddl_path.write_text(replace_goal_with_render_safe_goal(before_text), encoding="utf-8")
        step1_objects = sequence_goal_object_ids(source_text)[:1]
        render_sequence_after_image(
            bddl_file=after_bddl_path,
            output_path=after_path,
            camera_name=args.camera_name,
            image_size=args.image_size,
            settle_steps=args.settle_steps,
            max_reset_attempts=args.max_reset_attempts,
            object_ids_to_place=step1_objects,
        )
    elif suite == "belongings" and split == "multiuser" and metadata.get("fixture") == "basket":
        after_bddl_path.write_text(replace_goal_with_render_safe_goal(before_text), encoding="utf-8")
        type_to_ids = object_ids_by_type(before_text)
        owned_types = []
        for objects in metadata.get("ownership", {}).values():
            owned_types.extend(objects)
        object_ids = [
            type_to_ids[object_type][0]
            for object_type in owned_types
            if type_to_ids.get(object_type)
        ]
        render_objects_inside_fixture_image(
            bddl_file=after_bddl_path,
            output_path=after_path,
            camera_name=args.camera_name,
            image_size=args.image_size,
            max_reset_attempts=args.max_reset_attempts,
            object_ids=object_ids[:2],
            fixture_id="basket_1",
            fixture_kind="basket",
        )
    elif suite == "placements" and metadata.get("preference", {}).get("fixed_type") == "basket":
        after_bddl_path.write_text(replace_goal_with_render_safe_goal(before_text), encoding="utf-8")
        type_to_ids = object_ids_by_type(before_text)
        target_type = metadata.get("target_object") or metadata.get("preference", {}).get("object_type")
        target_ids = type_to_ids.get(str(target_type), [])
        fixtures = metadata.get("fixtures", [])
        basket_side = "left"
        for fixture in fixtures:
            if fixture.get("fixture") == "basket":
                basket_side = fixture.get("side", basket_side)
                break
        render_objects_inside_fixture_image(
            bddl_file=after_bddl_path,
            output_path=after_path,
            camera_name=args.camera_name,
            image_size=args.image_size,
            max_reset_attempts=args.max_reset_attempts,
            object_ids=target_ids[:1],
            fixture_id=f"basket_{basket_side}",
            fixture_kind="basket",
        )
    else:
        after_text = build_after_bddl(source_text)
        after_text = replace_goal_with_render_safe_goal(after_text)
        after_bddl_path.write_text(after_text, encoding="utf-8")
        render_task_image(
            bddl_file=after_bddl_path,
            output_path=after_path,
            camera_name=args.camera_name,
            image_size=args.image_size,
            settle_steps=args.settle_steps,
            max_reset_attempts=args.max_reset_attempts,
        )


def render_sequence_step_images(
    source_text: str,
    before_bddl_path: Path,
    step_bddl_path: Path,
    output_prefix: Path,
    args: argparse.Namespace,
) -> list[str]:
    step_paths = [output_prefix.with_name(f"{output_prefix.name}_step{step}.png") for step in range(4)]
    render_task_image(
        bddl_file=before_bddl_path,
        output_path=step_paths[0],
        camera_name=args.camera_name,
        image_size=args.image_size,
        settle_steps=args.settle_steps,
        max_reset_attempts=args.max_reset_attempts,
    )
    step_bddl_path.write_text(before_bddl_path.read_text(encoding="utf-8"), encoding="utf-8")
    goal_objects = sequence_goal_object_ids(source_text)
    for step in range(1, 4):
        render_sequence_after_image(
            bddl_file=step_bddl_path,
            output_path=step_paths[step],
            camera_name=args.camera_name,
            image_size=args.image_size,
            settle_steps=args.settle_steps,
            max_reset_attempts=args.max_reset_attempts,
            object_ids_to_place=goal_objects[:step],
        )
    return [str(path) for path in step_paths]


def make_distribution_png(suites_root: Path, output_path: Path) -> None:
    summaries = {}
    for suite in ("belongings", "placements", "sequences"):
        summary_path = suites_root / suite / "generation_summary.json"
        if summary_path.exists():
            summaries[suite] = read_json(summary_path).get("planned_counts", {})

    try:
        import matplotlib.pyplot as plt

        labels = []
        values = []
        colors = []
        palette = {"belongings": "#4C78A8", "placements": "#F58518", "sequences": "#54A24B"}
        for suite, counts in summaries.items():
            for split, count in counts.items():
                labels.append(f"{suite}\n{split}")
                values.append(int(count))
                colors.append(palette.get(suite, "#777777"))

        width = max(10, len(labels) * 0.75)
        fig, ax = plt.subplots(figsize=(width, 5.2))
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_title("VLAPB Suite Distribution")
        ax.set_ylabel("Written episodes")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - fallback for lean envs
        logging.warning("matplotlib chart failed (%s); writing JSON fallback next to png path", exc)
        output_path.with_suffix(".distribution.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in output_dir.glob("*.png"):
        if stale_png.name != "suite_distribution.png":
            stale_png.unlink()
    after_bddl_dir = output_dir / "after_bddl"
    after_bddl_dir.mkdir(parents=True, exist_ok=True)
    for stale_bddl in after_bddl_dir.glob("*.bddl"):
        stale_bddl.unlink()

    records: list[RenderRecord] = []
    failed: dict[str, list[str]] = {}
    progress = tqdm(TASK_SPLITS, desc="Rendering suite examples", unit="task")
    for task_index, (suite, split) in enumerate(progress, start=1):
        key = f"{suite}_{split}"
        errors = []
        candidates = candidate_metadata_files(args.suites_root, suite, split)[: args.max_candidates]
        if not candidates:
            raise FileNotFoundError(f"No metadata candidates for {suite}/{split}")
        for metadata_file in candidates:
            bddl_file = bddl_for_metadata(metadata_file, args.suites_root, suite, split)
            metadata = read_json(metadata_file)
            if suite == "placements" and split == "type4" and bddl_file == TYPE4_TOP_DRAWER_BDDL:
                episode_id = "placements_type4_top_drawer_butter"
            elif suite == "sequences" and split == "type1" and bddl_file == SEQUENCE_TYPE1_VISUAL_BDDL:
                episode_id = "sequences_type1_visual_small_objects"
            else:
                episode_id = str(metadata.get("episode_id", metadata_file.stem))
            prefix = f"{task_index:02d}_{suite}_{split}_{episode_id}"
            before_path = output_dir / f"{prefix}_before.png"
            after_path = output_dir / f"{prefix}_after.png"
            before_bddl_path = after_bddl_dir / f"{prefix}_before.bddl"
            after_bddl_path = after_bddl_dir / f"{prefix}_after.bddl"
            sequence_step_images = None
            try:
                render_pair(
                    bddl_file,
                    before_bddl_path,
                    before_path,
                    after_path,
                    after_bddl_path,
                    args,
                    metadata,
                    suite,
                    split,
                )
                if suite == "sequences":
                    sequence_step_images = render_sequence_step_images(
                        source_text=bddl_file.read_text(encoding="utf-8"),
                        before_bddl_path=before_bddl_path,
                        step_bddl_path=after_bddl_dir / f"{prefix}_steps.bddl",
                        output_prefix=output_dir / prefix,
                        args=args,
                    )
            except Exception as exc:
                errors.append(f"{metadata_file.name}: {exc}")
                logging.warning("render failed for %s/%s candidate %s: %s", suite, split, metadata_file.name, exc)
                cleanup_paths = [before_path, after_path, before_bddl_path, after_bddl_path]
                if sequence_step_images:
                    cleanup_paths.extend(Path(path) for path in sequence_step_images)
                for path in cleanup_paths:
                    if path.exists():
                        path.unlink()
                continue
            records.append(
                RenderRecord(
                    task_index=task_index,
                    suite=suite,
                    split=split,
                    episode_id=episode_id,
                    bddl_file=str(bddl_file),
                    metadata_file=str(metadata_file),
                    before_image=str(before_path),
                    after_image=str(after_path),
                    after_bddl_file=str(after_bddl_path),
                    sequence_step_images=sequence_step_images,
                )
            )
            break
        else:
            failed[key] = errors
            raise RuntimeError(f"Could not render any candidate for {suite}/{split}: {errors[:3]}")

    chart_path = output_dir / "suite_distribution.png"
    make_distribution_png(args.suites_root, chart_path)
    rendered_image_count = sum(2 + len(record.sequence_step_images or []) for record in records)
    manifest = {
        "output_dir": str(output_dir),
        "image_count": rendered_image_count,
        "chart": str(chart_path),
        "records": [asdict(record) for record in records],
        "failed_candidates": failed,
        "render_args": {
            "camera_name": args.camera_name,
            "image_size": args.image_size,
            "settle_steps": args.settle_steps,
            "max_reset_attempts": args.max_reset_attempts,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("rendered task pairs: %d", len(records))
    logging.info("rendered png images: %d", rendered_image_count)
    logging.info("manifest: %s", output_dir / "manifest.json")
    logging.info("distribution chart: %s", chart_path)


if __name__ == "__main__":
    main()
