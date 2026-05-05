#!/usr/bin/env python3
"""Generate belongings visual-injection object images and episode inputs.

The object image cache contains one rendered table scene per belongings object:
``VLAPB_suites/belongings/visual_inputs/objects/{object_type}.png``.

Episode-level JSON files are written under each belongings split:
``VLAPB_suites/belongings/{split}/visual_inputs/{episode_id}.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = SCRIPT_DIR.parents[1]
VLAPB_PROJECT_ROOT = VLAPB_LIBERO_ROOT.parent
DEFAULT_PROFILES = VLAPB_LIBERO_ROOT / "profiles" / "profiles.json"
DEFAULT_SUITE_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"
DEFAULT_BELONGINGS_ROOT = DEFAULT_SUITE_ROOT / "belongings"
DEFAULT_OBJECT_IMAGE_DIR = DEFAULT_BELONGINGS_ROOT / "visual_inputs" / "objects"
DEFAULT_OBJECT_BDDL_DIR = DEFAULT_BELONGINGS_ROOT / "visual_inputs" / "bddl_objects"
DEFAULT_MANIFEST = DEFAULT_BELONGINGS_ROOT / "visual_inputs" / "objects_manifest.json"

EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

try:
    from test_env_reconfiguration import render_task_image
except Exception:  # pragma: no cover - dry-run and JSON-only workflows can still work.
    render_task_image = None


SPLIT_ORDER = ("type1", "type2", "type3", "adaptability", "multiuser")
REGION_HALF_SIZE = 0.025
DEFAULT_XY = (0.0, -0.12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--belongings-root", type=Path, default=DEFAULT_BELONGINGS_ROOT)
    parser.add_argument("--object-image-dir", type=Path, default=DEFAULT_OBJECT_IMAGE_DIR)
    parser.add_argument("--object-bddl-dir", type=Path, default=DEFAULT_OBJECT_BDDL_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument(
        "--episode-image-source",
        choices=("profile", "metadata-ownership", "target-object"),
        default="metadata-ownership",
        help="Which object list to attach to each episode visual input.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-only", action="store_true", help="Write BDDL/JSON but skip PNG rendering.")
    parser.add_argument("--limit-objects", type=int, default=None, help="Debug cap for object image generation.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Debug cap for episode JSON generation.")
    parser.add_argument(
        "--all-existing-metadata",
        dest="respect_generation_summary",
        action="store_false",
        help="Process every belongings metadata JSON on disk instead of generation_summary planned counts.",
    )
    parser.set_defaults(respect_generation_summary=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def object_label(object_type: str) -> str:
    return object_type.replace("_", " ")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    profiles = data.get("profiles", data)
    if isinstance(profiles, dict):
        return {str(user_id): dict(profile) for user_id, profile in profiles.items()}
    return {str(profile["user_id"]): dict(profile) for profile in profiles}


def profile_belongings(profiles: Mapping[str, dict[str, Any]], user_id: str) -> list[str]:
    profile = profiles.get(user_id, {})
    return [str(item) for item in profile.get("belongings", [])]


def unique_belongings(profiles: Mapping[str, dict[str, Any]]) -> list[str]:
    objects: set[str] = set()
    for profile in profiles.values():
        objects.update(str(item) for item in profile.get("belongings", []))
    return sorted(objects)


def planned_counts(belongings_root: Path, respect_generation_summary: bool) -> dict[str, int]:
    if not respect_generation_summary:
        return {}
    summary_path = belongings_root / "generation_summary.json"
    if not summary_path.exists():
        return {}
    summary = read_json(summary_path)
    return {str(split): int(count) for split, count in (summary.get("planned_counts") or {}).items()}


def metadata_files(belongings_root: Path, respect_generation_summary: bool) -> list[Path]:
    limits = planned_counts(belongings_root, respect_generation_summary)
    files: list[Path] = []
    seen: set[str] = set()
    for split in SPLIT_ORDER:
        metadata_dir = belongings_root / split / "metadata"
        if not metadata_dir.exists():
            continue
        seen.add(split)
        split_files = sorted(metadata_dir.glob("*.json"))
        limit = limits.get(split)
        files.extend(split_files[:limit] if limit is not None else split_files)
    for metadata_dir in sorted(belongings_root.glob("*/metadata")):
        split = metadata_dir.parent.name
        if split in seen:
            continue
        split_files = sorted(metadata_dir.glob("*.json"))
        limit = limits.get(split)
        files.extend(split_files[:limit] if limit is not None else split_files)
    return sorted(set(files))


def relation_text(relation: str | None, fixture: str | None) -> str:
    fixture_text = object_label(fixture or "target fixture")
    if str(relation or "").lower() == "on":
        return f"on the {fixture_text}"
    return f"in the {fixture_text}"


def general_input(metadata: Mapping[str, Any]) -> str:
    return (
        f"Pick <sks>'s item and place it "
        f"{relation_text(metadata.get('relation'), metadata.get('fixture'))}."
    )


def visual_prefix(image_count: int) -> str:
    if image_count == 1:
        return "This is a picture of <sks> belongings."
    return "These are pictures of <sks> belongings."


def visual_textual_input(image_count: int, metadata: Mapping[str, Any]) -> str:
    return f"{visual_prefix(image_count)} {general_input(metadata)}"


def region_block(name: str, target: str, xy: tuple[float, float], half_size: float) -> str:
    x, y = xy
    return (
        f"      ({name}\n"
        f"          (:target {target})\n"
        "          (:ranges (\n"
        f"              ({x - half_size:.6f} {y - half_size:.6f} {x + half_size:.6f} {y + half_size:.6f})\n"
        "            )\n"
        "          )\n"
        "      )"
    )


def write_object_bddl(object_type: str, path: Path, xy: tuple[float, float] = DEFAULT_XY) -> None:
    instance = f"{safe_name(object_type)}_1"
    region = f"{safe_name(object_type)}_init_region"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bddl, encoding="utf-8")


def render_object_image(
    bddl_path: Path,
    image_path: Path,
    *,
    camera_name: str,
    image_size: int,
    settle_steps: int,
    max_reset_attempts: int,
) -> None:
    if render_task_image is None:
        raise RuntimeError("Could not import render_task_image from libero/examples/test_env_reconfiguration.py")
    render_task_image(
        bddl_file=bddl_path,
        output_path=image_path,
        camera_name=camera_name,
        image_size=image_size,
        settle_steps=settle_steps,
        max_reset_attempts=max_reset_attempts,
    )


def episode_objects(
    metadata: Mapping[str, Any],
    profiles: Mapping[str, dict[str, Any]],
    source: str,
) -> list[str]:
    target_user = str(metadata.get("target_user"))
    if source == "target-object":
        target = metadata.get("target_object")
        return [str(target)] if target else []
    if source == "metadata-ownership":
        ownership = metadata.get("ownership") or {}
        return [str(item) for item in ownership.get(target_user, [])]
    return profile_belongings(profiles, target_user)


def episode_output_path(metadata_path: Path) -> Path:
    return metadata_path.parent.parent / "visual_inputs" / metadata_path.name


def build_episode_payload(
    metadata: Mapping[str, Any],
    metadata_path: Path,
    profiles: Mapping[str, dict[str, Any]],
    object_image_dir: Path,
    source: str,
) -> dict[str, Any]:
    objects = episode_objects(metadata, profiles, source)
    image_paths = [str(object_image_dir / f"{safe_name(obj)}.png") for obj in objects]
    return {
        "task_id": metadata.get("episode_id"),
        "suite": "belongings",
        "split": metadata.get("split"),
        "task_type": metadata.get("task_type"),
        "user_id": metadata.get("target_user"),
        "target_object": metadata.get("target_object"),
        "image_source": source,
        "object_types": objects,
        "image_paths": image_paths,
        "visual_textual_input": visual_textual_input(len(image_paths), metadata),
        "general_input": general_input(metadata),
        "metadata_file": str(metadata_path),
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    profiles = load_profiles(args.profiles)
    objects = unique_belongings(profiles)
    if args.limit_objects is not None:
        objects = objects[: args.limit_objects]

    rendered = 0
    skipped = 0
    object_records: dict[str, dict[str, str]] = {}
    iterator = tqdm(objects, desc="Generating object images", unit="object") if tqdm else objects
    for object_type in iterator:
        bddl_path = args.object_bddl_dir / f"{safe_name(object_type)}.bddl"
        image_path = args.object_image_dir / f"{safe_name(object_type)}.png"
        object_records[object_type] = {
            "object_type": object_type,
            "bddl_file": str(bddl_path),
            "image_path": str(image_path),
        }
        if args.dry_run:
            continue
        write_object_bddl(object_type, bddl_path)
        if args.json_only:
            continue
        if args.skip_existing and image_path.exists():
            skipped += 1
            continue
        render_object_image(
            bddl_path,
            image_path,
            camera_name=args.camera_name,
            image_size=args.image_size,
            settle_steps=args.settle_steps,
            max_reset_attempts=args.max_reset_attempts,
        )
        rendered += 1

    metadata_paths = metadata_files(args.belongings_root, args.respect_generation_summary)
    if args.limit_episodes is not None:
        metadata_paths = metadata_paths[: args.limit_episodes]

    episode_counts: Counter[str] = Counter()
    written_episode_json = 0
    episode_iterator = tqdm(metadata_paths, desc="Generating visual episode inputs", unit="episode") if tqdm else metadata_paths
    for metadata_path in episode_iterator:
        metadata = read_json(metadata_path)
        episode_counts[str(metadata.get("split", "unknown"))] += 1
        if args.dry_run:
            continue
        payload = build_episode_payload(
            metadata,
            metadata_path,
            profiles,
            args.object_image_dir,
            args.episode_image_source,
        )
        write_json(episode_output_path(metadata_path), payload)
        written_episode_json += 1

    summary = {
        "profiles": str(args.profiles),
        "belongings_root": str(args.belongings_root),
        "object_image_dir": str(args.object_image_dir),
        "object_bddl_dir": str(args.object_bddl_dir),
        "manifest": str(args.manifest),
        "object_count": len(objects),
        "rendered_object_images": rendered,
        "skipped_existing_images": skipped,
        "episode_json_count": len(metadata_paths),
        "written_episode_json": 0 if args.dry_run else written_episode_json,
        "episode_image_source": args.episode_image_source,
        "respect_generation_summary": args.respect_generation_summary,
        "dry_run": args.dry_run,
        "json_only": args.json_only,
        "distribution": dict(sorted(episode_counts.items())),
        "objects": object_records,
    }
    if not args.dry_run:
        write_json(args.manifest, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = generate(args)
    print(f"[INFO] object image dir: {summary['object_image_dir']}")
    print(f"[INFO] object count: {summary['object_count']}")
    print(f"[INFO] rendered object images: {summary['rendered_object_images']}")
    print(f"[INFO] skipped existing images: {summary['skipped_existing_images']}")
    print(f"[INFO] episode json count: {summary['episode_json_count']}")
    print(f"[INFO] written episode json: {summary['written_episode_json']}")
    print(f"[INFO] distribution: {summary['distribution']}")
    if summary["dry_run"]:
        print("[INFO] dry-run: no files written")
    else:
        print(f"[INFO] manifest: {summary['manifest']}")


if __name__ == "__main__":
    main()
