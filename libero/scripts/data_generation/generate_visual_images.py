#!/usr/bin/env python3
"""Materialize visual-injection images for compare_vis_tex bundles.

Reads evaluations/compare_injection_methods/manifest.json and creates:
- observation images from generated table-scene BDDL files,
- per-object reference render views,
- text-free reference panels,
- concatenated observation + reference-panel images.

The script intentionally keeps image generation separate from compare_vis_tex.py
so the benchmark manifest can be inspected or edited before the expensive render
step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

# Set before robosuite / mujoco imports. Users can override in the shell.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the script usable without tqdm.
    def tqdm(iterable, **_kwargs):
        return iterable

VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = VLAPB_LIBERO_ROOT / "evaluations" / "compare_injection_methods"
DEFAULT_MANIFEST = DEFAULT_EVAL_DIR / "manifest.json"
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from test_env_reconfiguration import render_task_image  # noqa: E402

REGION_HALF_SIZE = 0.025
BASKET_REGION_HALF_SIZE = 0.035
# Match the feasible table slots used by compare_vis_tex scene generation,
# but render only the referenced object on the table.
REFERENCE_POSITIONS = ((-0.09, -0.27), (-0.09, -0.09), (0.11, -0.27))
DEFAULT_CAMERA_NAME = "agentview"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_REFERENCE_TILE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate observation/reference-panel images for injection comparison."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--reference-tile-size", type=int, default=DEFAULT_REFERENCE_TILE_SIZE)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument(
        "--limit-trials",
        type=int,
        default=None,
        help="Render only the first N trials for smoke testing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerender files that already exist.",
    )
    parser.add_argument(
        "--placeholder-on-failure",
        action="store_true",
        help="Write labeled placeholder images if MuJoCo rendering fails.",
    )
    parser.add_argument(
        "--update-json",
        action="store_true",
        help="Mark generated visual condition JSON files as generated.",
    )
    return parser.parse_args()


def safe_instance_name(object_type: str, index: int) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_]+', '_', object_type)}_{index}"


def object_label(object_type: str) -> str:
    return object_type.replace("_", " ")


def region_block(name: str, target: str, xy: tuple[float, float], half_size: float, yaw: float | None = None) -> str:
    x, y = xy
    yaw_block = ""
    if yaw is not None:
        yaw_block = (
            "          (:yaw_rotation (\n"
            f"              ({yaw:.10f} {yaw:.10f})\n"
            "            )\n"
            "          )\n"
        )
    return (
        f"      ({name}\n"
        f"          (:target {target})\n"
        "          (:ranges (\n"
        f"              ({x - half_size:.6f} {y - half_size:.6f} {x + half_size:.6f} {y + half_size:.6f})\n"
        "            )\n"
        "          )\n"
        f"{yaw_block}"
        "      )"
    )


def write_table_scene_bddl(scene_config: dict, output_path: Path) -> dict[str, str]:
    layout = scene_config["scene_layout"]
    target_object = scene_config["target_object"]
    objects = list(layout["objects"])
    basket = layout["basket"]

    instance_by_role_object: dict[str, str] = {}
    object_lines = []
    region_lines = []
    init_lines = []
    interest_lines = []

    for index, item in enumerate(objects, start=1):
        object_type = item["object"]
        instance = safe_instance_name(object_type, index)
        region = f"{instance}_init_region"
        instance_by_role_object[f"{item['role']}:{object_type}"] = instance
        object_lines.append(f"    {instance} - {object_type}")
        region_lines.append(
            region_block(
                region,
                target="main_table",
                xy=(float(item["xy"][0]), float(item["xy"][1])),
                half_size=REGION_HALF_SIZE,
            )
        )
        init_lines.append(f"    (On {instance} main_table_{region})")
        if item["role"] == "target":
            interest_lines.append(f"    {instance}")

    basket_instance = "basket_1"
    basket_region = "basket_init_region"
    object_lines.append(f"    {basket_instance} - basket")
    region_lines.append(
        region_block(
            basket_region,
            target="main_table",
            xy=(float(basket["xy"][0]), float(basket["xy"][1])),
            half_size=BASKET_REGION_HALF_SIZE,
        )
    )
    region_lines.append("      (contain_region\n          (:target basket_1)\n      )")
    init_lines.append(f"    (On {basket_instance} main_table_{basket_region})")
    interest_lines.append(f"    {basket_instance}")

    target_instance = instance_by_role_object[f"target:{target_object}"]
    language = f"Pick sks's object and put it in the basket"
    bddl = (
        "(define (problem LIBERO_Tabletop_Manipulation)\n"
        "  (:domain robosuite)\n"
        f"  (:language {language})\n"
        "    (:regions\n"
        + "\n".join(region_lines)
        + "\n    )\n\n"
        "  (:fixtures\n"
        "    main_table - table\n"
        "  )\n\n"
        "  (:objects\n"
        + "\n".join(object_lines)
        + "\n  )\n\n"
        "  (:obj_of_interest\n"
        + "\n".join(interest_lines)
        + "\n  )\n\n"
        "  (:init\n"
        + "\n".join(init_lines)
        + "\n  )\n\n"
        "  (:goal\n"
        f"    (And (In {target_instance} basket_1_contain_region))\n"
        "  )\n\n"
        ")\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bddl, encoding="utf-8")
    return {"target_instance": target_instance, "basket_instance": basket_instance}


def write_reference_bddl(object_type: str, output_path: Path, xy: tuple[float, float]) -> None:
    instance = safe_instance_name(object_type, 1)
    region = f"{instance}_init_region"
    bddl = (
        "(define (problem LIBERO_Tabletop_Manipulation)\n"
        "  (:domain robosuite)\n"
        f"  (:language show {object_label(object_type)})\n"
        "    (:regions\n"
        + region_block(region, target="main_table", xy=xy, half_size=REGION_HALF_SIZE)
        + "\n    )\n\n"
        "  (:fixtures\n"
        "    main_table - table\n"
        "  )\n\n"
        "  (:objects\n"
        f"    {instance} - {object_type}\n"
        "  )\n\n"
        "  (:obj_of_interest\n"
        f"    {instance}\n"
        "  )\n\n"
        "  (:init\n"
        f"    (On {instance} main_table_{region})\n"
        "  )\n\n"
        "  (:goal\n"
        f"    (And (On {instance} main_table_{region}))\n"
        "  )\n\n"
        ")\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bddl, encoding="utf-8")


def placeholder_image(path: Path, text: str, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (4, 4), (size - 5, size - 5), (80, 80, 80), 1)
    words = text.split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 18:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    y = size // 2 - 12 * len(lines)
    for line in lines[:5]:
        cv2.putText(image, line, (12, max(24, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24
    cv2.imwrite(str(path), image)


def render_bddl_to_image(
    bddl_path: Path,
    image_path: Path,
    camera_name: str,
    image_size: int,
    settle_steps: int,
    max_reset_attempts: int,
    skip_existing: bool,
    placeholder_on_failure: bool,
    placeholder_text: str,
) -> bool:
    if skip_existing and image_path.exists():
        return True
    try:
        render_task_image(
            bddl_file=bddl_path,
            output_path=image_path,
            camera_name=camera_name,
            image_size=image_size,
            settle_steps=settle_steps,
            max_reset_attempts=max_reset_attempts,
        )
        return True
    except Exception:
        if not placeholder_on_failure:
            raise
        placeholder_image(image_path, placeholder_text, image_size)
        return False


def load_image(path: Path, fallback_text: str, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        placeholder_image(path, fallback_text, size)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not load or create image: {path}")
    return image


def make_reference_panel(
    reference_items: list[dict],
    output_path: Path,
    tile_size: int = DEFAULT_REFERENCE_TILE_SIZE,
) -> None:
    rows = []
    for item in reference_items:
        row_tiles = []
        for view_path in item["views"]:
            image = load_image(Path(view_path), item["object"], tile_size)
            # Individual reference view files preserve the raw render; resize
            # only for consistent panel tiling.
            image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
            row_tiles.append(image)
        if row_tiles:
            rows.append(cv2.hconcat(row_tiles))

    if not rows:
        raise ValueError("Cannot build an empty reference panel")

    # Object-major layout: each belongings object gets one row, and its three
    # position views are placed left-to-right. This produces a 4x3 panel for the
    # current 4-belongings profiles.
    panel = cv2.vconcat(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def make_concat(observation_path: Path, panel_path: Path, output_path: Path, image_size: int) -> None:
    observation = load_image(observation_path, "observation", image_size)
    panel = load_image(panel_path, "reference panel", image_size)
    panel_h = panel.shape[0]
    scale = panel_h / observation.shape[0]
    observation = cv2.resize(
        observation,
        (max(1, int(observation.shape[1] * scale)), panel_h),
        interpolation=cv2.INTER_CUBIC,
    )
    concat = cv2.hconcat([observation, panel])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), concat)


def update_json_status(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_status"] = "generated"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_dir = Path(manifest["metadata"]["output_dir"]).resolve()
    bddl_dir = eval_dir / "bddl"

    trials = manifest["trials"]
    if args.limit_trials is not None:
        trials = trials[: args.limit_trials]

    rendered_observations = 0
    rendered_references = 0
    built_panels = 0
    built_concats = 0

    for trial in tqdm(trials, desc="Generating visual images", unit="trial"):
        trial_id = trial["trial_id"]
        scene_config_path = Path(trial["scene_config"])
        scene_config = json.loads(scene_config_path.read_text(encoding="utf-8"))
        observation_path = eval_dir / "images" / "observations" / f"{trial_id}.png"
        scene_bddl_path = bddl_dir / "scenes" / f"{trial_id}.bddl"
        write_table_scene_bddl(scene_config, scene_bddl_path)
        render_bddl_to_image(
            scene_bddl_path,
            observation_path,
            camera_name=args.camera_name,
            image_size=args.image_size,
            settle_steps=args.settle_steps,
            max_reset_attempts=args.max_reset_attempts,
            skip_existing=args.skip_existing,
            placeholder_on_failure=args.placeholder_on_failure,
            placeholder_text=trial_id,
        )
        rendered_observations += 1

        # All visual condition JSON files share the same reference items for a trial.
        visual_json_paths = [Path(path) for path in trial["visual_conditions"]]
        visual_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in visual_json_paths]
        reference_payload = next(
            (payload for payload in visual_payloads if payload.get("reference_items")),
            None,
        )
        if reference_payload is None:
            continue

        for item in reference_payload["reference_items"]:
            object_type = item["object"]
            for view_index, view_path in enumerate(item["views"]):
                view_path = Path(view_path)
                reference_bddl = bddl_dir / "reference_objects" / f"{object_type}_view_{view_index:02d}.bddl"
                xy = REFERENCE_POSITIONS[view_index % len(REFERENCE_POSITIONS)]
                write_reference_bddl(object_type, reference_bddl, xy=xy)
                rendered = render_bddl_to_image(
                    reference_bddl,
                    view_path,
                    camera_name=args.camera_name,
                    image_size=args.image_size,
                    settle_steps=args.settle_steps,
                    max_reset_attempts=args.max_reset_attempts,
                    skip_existing=args.skip_existing,
                    placeholder_on_failure=args.placeholder_on_failure,
                    placeholder_text=object_label(object_type),
                )
                rendered_references += 1

        for payload, path in zip(visual_payloads, visual_json_paths):
            condition = payload["visual_condition"]
            if condition == "obs_only":
                if args.update_json:
                    update_json_status(path)
                continue

            panel_path = Path(payload["input_images"]["reference_panel"])
            make_reference_panel(
                payload["reference_items"],
                panel_path,
                tile_size=args.reference_tile_size,
            )
            built_panels += 1

            concat_image = payload.get("concat_image")
            if concat_image:
                make_concat(observation_path, panel_path, Path(concat_image), args.image_size)
                built_concats += 1
            if args.update_json:
                update_json_status(path)

        scene_config["image_status"] = "generated"
        scene_config["bddl_file"] = str(scene_bddl_path)
        scene_config["observation_image"] = str(observation_path)
        scene_config_path.write_text(json.dumps(scene_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Rendered observations: {rendered_observations}")
    print(f"Rendered reference views: {rendered_references}")
    print(f"Built reference panels: {built_panels}")
    print(f"Built concat images: {built_concats}")
    print(f"Output root: {eval_dir}")


if __name__ == "__main__":
    main()
