#!/usr/bin/env python3
"""
Render an original LIBERO-90 task scene and a reconfigured variant.

The script resolves a 1-based task index using ``libero_90_instructions.txt``
when available so that the numbering matches the user's extracted instruction
file. It then:

1. Renders the original task environment and saves a PNG.
2. Creates a temporary BDDL variant that adds two random extra objects not
   already present in the scene.
3. Resets the environment with that modified BDDL and saves a second PNG.

Example:
    python3 test_env_reconfiguration.py --task-id 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = ROOT.parent
DOCS_ROOT = VLAPB_LIBERO_ROOT / "docs"
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
sys.path.insert(0, str(LIBERO_ROOT / "libero"))

import cv2
import numpy as np
import robosuite as suite
from robosuite.utils.errors import RandomizationError

from libero.envs import TASK_MAPPING
from libero.envs.bddl_utils import get_problem_info, robosuite_parse_problem
from libero.envs.objects import get_object_dict


DEFAULT_DATA_ROOT = Path("/home/artemis/libero_data")
SUITE_BDDL_DIRS = {
    "libero_10": LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_10",
    "libero_90": LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_90",
    "libero_goal": LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_goal",
    "libero_object": LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_object",
    "libero_spatial": LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_spatial",
}
DEFAULT_OUTPUT_DIR = VLAPB_LIBERO_ROOT / "examples"

WORKSPACE_TYPES = {
    "main_table",
    "table",
    "kitchen_table",
    "living_room_table",
    "study_table",
    "coffee_table",
    "floor",
}

# Prefer small / medium movable objects that already appear across LIBERO tasks.
EXTRA_OBJECT_POOL = [
    "alphabet_soup",
    "akita_black_bowl",
    "basket",
    "black_book",
    "butter",
    "chocolate_pudding",
    "cream_cheese",
    "ketchup",
    "milk",
    "orange_juice",
    "porcelain_mug",
    "red_coffee_mug",
    "tomato_sauce",
    "white_bowl",
    "white_yellow_mug",
    "wine_bottle",
    "wooden_tray",
]

# Candidate XY placements on the workspace surface. We choose a subset that is
# sufficiently far from existing placements to reduce reset collisions.
CANDIDATE_POSITIONS = [
    (-0.25, -0.22),
    (-0.22, -0.18),
    (-0.22, -0.05),
    (-0.22, 0.10),
    (-0.25, 0.22),
    (-0.14, -0.22),
    (-0.14, 0.12),
    (-0.05, -0.22),
    (-0.05, 0.18),
    (0.05, -0.22),
    (0.05, 0.18),
    (0.14, -0.22),
    (0.14, 0.12),
    (0.25, -0.22),
    (0.22, -0.18),
    (0.22, -0.05),
    (0.22, 0.10),
    (0.25, 0.22),
]
CANDIDATE_GRID_X = [-0.30, -0.22, -0.14, -0.06, 0.02, 0.10, 0.18, 0.26, 0.30]
CANDIDATE_GRID_Y = [-0.24, -0.16, -0.08, 0.00, 0.08, 0.16, 0.24]

FIXTURE_CLEARANCE = {
    "basket": 0.11,
    "desk_caddy": 0.12,
    "flat_stove": 0.14,
    "microwave": 0.17,
    "white_cabinet": 0.24,
    "wooden_cabinet": 0.18,
    "wooden_two_layer_shelf": 0.15,
    "wooden_tray": 0.17,
    "wine_rack": 0.10,
}

# Approximate XY radii used only for conservative initial placement. These are
# deliberately larger than the 5cm BDDL sampling regions because MuJoCo object
# meshes can extend well beyond the sampled centroid.
OBJECT_CLEARANCE = {
    "basket": 0.13,
    "wooden_tray": 0.18,
    "akita_black_bowl": 0.09,
    "white_bowl": 0.09,
    "porcelain_mug": 0.08,
    "red_coffee_mug": 0.08,
    "white_yellow_mug": 0.08,
    "wine_bottle": 0.08,
    "bottled_spirit_1": 0.08,
    "cloud_computing": 0.10,
    "black_book": 0.12,
    "alphabet_soup": 0.08,
    "butter": 0.07,
    "chocolate_pudding": 0.08,
    "cream_cheese": 0.08,
    "ketchup": 0.08,
    "milk": 0.08,
    "orange_juice": 0.08,
    "tomato_sauce": 0.08,
    "laptop": 0.14,
    "yellow_book": 0.12,
    "desk_caddy": 0.15,
    "wooden_shelf": 0.16,
    "white_storage_box": 0.14,
    "cutting_board_0": 0.13,
    "chefmate_8_frypan": 0.14,
    "moka_pot": 0.09,
    "knife_0": 0.11,
    "scissors_0": 0.10,
    "hammer_1": 0.12,
    "vase_0": 0.09,
    "giftbox_0": 0.09,
    "mirror": 0.10,
    "alien_0": 0.08,
    "number_cube": 0.07,
}
DEFAULT_OBJECT_CLEARANCE = 0.10
EXISTING_OBJECT_CLEARANCE = 0.08
REGION_HALF_SIZE = 0.05

# Conservative XY half-extents for placement-only collision checks. These
# deliberately include handles, rims, and drawer travel that are easy to miss
# with centroid-distance checks.
OBJECT_FOOTPRINTS = {
    "basket": (0.12, 0.12),
    "wooden_tray": (0.18, 0.12),
    "akita_black_bowl": (0.08, 0.08),
    "white_bowl": (0.08, 0.08),
    "plate": (0.09, 0.09),
    "porcelain_mug": (0.07, 0.07),
    "red_coffee_mug": (0.07, 0.07),
    "white_yellow_mug": (0.07, 0.07),
    "wine_bottle": (0.06, 0.06),
    "black_book": (0.12, 0.09),
    "alphabet_soup": (0.06, 0.06),
    "butter": (0.06, 0.05),
    "chocolate_pudding": (0.06, 0.06),
    "cream_cheese": (0.07, 0.05),
    "ketchup": (0.06, 0.06),
    "milk": (0.06, 0.06),
    "orange_juice": (0.06, 0.06),
    "tomato_sauce": (0.06, 0.06),
    "laptop": (0.14, 0.10),
    "yellow_book": (0.12, 0.09),
    "cloud_computing": (0.12, 0.09),
    "desk_caddy": (0.13, 0.15),
    "wooden_shelf": (0.16, 0.12),
    "white_storage_box": (0.12, 0.12),
    "cutting_board_0": (0.13, 0.09),
    "chefmate_8_frypan": (0.13, 0.13),
    "moka_pot": (0.08, 0.08),
    "knife_0": (0.12, 0.04),
    "scissors_0": (0.10, 0.06),
    "hammer_1": (0.12, 0.07),
    "vase_0": (0.08, 0.08),
    "giftbox_0": (0.08, 0.08),
    "mirror": (0.10, 0.06),
    "alien_0": (0.07, 0.07),
    "number_cube": (0.06, 0.06),
}
DEFAULT_OBJECT_FOOTPRINT = (0.08, 0.08)

FIXTURE_FOOTPRINTS = {
    "basket": (0.12, 0.12),
    "flat_stove": (0.18, 0.16),
    "microwave": (0.22, 0.18),
    "white_cabinet": (0.22, 0.22),
    "wooden_cabinet": (0.22, 0.22),
    "wooden_two_layer_shelf": (0.18, 0.16),
    "wooden_tray": (0.18, 0.12),
    "wine_rack": (0.16, 0.12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an original and reconfigured LIBERO scene."
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_BDDL_DIRS),
        default="libero_90",
        help="LIBERO suite used to resolve --task-id.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Optional suite directory containing *_demo.hdf5 files or .bddl files, "
            "e.g. /home/artemis/libero_data/libero_object."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing downloaded LIBERO demo folders.",
    )
    parser.add_argument(
        "--bddl-dir",
        type=Path,
        default=None,
        help="Optional directory containing .bddl files for the selected suite.",
    )
    parser.add_argument(
        "--instructions-file",
        type=Path,
        default=None,
        help="Optional *_instructions.txt file whose file: lines define task order.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="1-based task index matching the selected suite order.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help=(
            "Scene / task selector. Accepts a .bddl path, BDDL stem, or BDDL "
            "(:scene ...) name. If several tasks share the scene, --object-name "
            "is used to pick a matching task."
        ),
    )
    parser.add_argument(
        "--object-name",
        "--object-names",
        nargs="+",
        default=None,
        help=(
            "Object type(s) to generate explicitly, e.g. --object-name basket. "
            "This is an alias for --force-objects and disables random object choice."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PNGs and temporary BDDL will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling extra objects and placements.",
    )
    parser.add_argument(
        "--num-extra-objects",
        type=int,
        default=2,
        help="Number of random objects to add to the scene.",
    )
    parser.add_argument(
        "--force-objects",
        nargs="+",
        default=None,
        help=(
            "Optional list of object types to add explicitly, e.g. "
            "--force-objects basket wooden_tray"
        ),
    )
    parser.add_argument(
        "--replace-scene-objects",
        action="store_true",
        help=(
            "Replace the base task's movable objects with --force-objects instead "
            "of appending them as *_extra_N instances. Fixtures remain intact."
        ),
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="agentview",
        help="Camera to render from, e.g. agentview or frontview.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square output image size in pixels.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=3,
        help="Number of zero-action steps after reset before saving an image.",
    )
    parser.add_argument(
        "--max-reset-attempts",
        type=int,
        default=25,
        help="Maximum reset retries if random placement fails.",
    )
    parser.add_argument(
        "--placement-metadata-output",
        type=Path,
        default=None,
        help="Optional JSON path where added object placements will be written.",
    )
    args = parser.parse_args()
    if args.task_id is None and args.scene_name is None:
        parser.error("Provide either --task-id or --scene-name")
    if args.object_name is not None and args.force_objects is not None:
        parser.error("Use either --object-name or --force-objects, not both")
    if args.object_name is not None:
        args.force_objects = args.object_name
        args.num_extra_objects = len(args.object_name)
    return args


def resolve_bddl_reference(reference: str, base_dir: Path | None = None) -> Path:
    candidates = []
    ref_path = Path(reference)
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        if base_dir is not None:
            candidates.append(base_dir / ref_path)
        candidates.append(LIBERO_ROOT / ref_path)
        candidates.append(ROOT / ref_path)
        candidates.append(Path.cwd() / ref_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def bddl_path_from_demo(demo_path: Path) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading .hdf5 demo directories requires h5py. "
            "Install h5py or pass --bddl-dir."
        ) from exc

    with h5py.File(demo_path, "r") as h5_file:
        attrs = h5_file["data"].attrs if "data" in h5_file else h5_file.attrs
        bddl_file_name = attrs.get("bddl_file_name")
        if bddl_file_name is None:
            raise KeyError(f"{demo_path} has no bddl_file_name attribute")
        if isinstance(bddl_file_name, bytes):
            bddl_file_name = bddl_file_name.decode("utf-8")
    return resolve_bddl_reference(str(bddl_file_name), base_dir=demo_path.parent)


def collect_task_files_from_data_dir(data_dir: Path) -> list[Path]:
    bddl_files = sorted(data_dir.glob("*.bddl"))
    if bddl_files:
        return [path.resolve() for path in bddl_files]

    demo_files = sorted(data_dir.glob("*_demo.hdf5"))
    if not demo_files:
        demo_files = sorted(data_dir.glob("*.hdf5"))
    return [bddl_path_from_demo(demo_path) for demo_path in demo_files]


def official_suite_files(suite: str, bddl_dir: Path) -> list[Path]:
    try:
        from libero.benchmark.libero_suite_task_map import libero_task_map
    except ImportError:
        return []

    task_names = libero_task_map.get(suite, [])
    return [bddl_dir / f"{task_name}.bddl" for task_name in task_names]


def sort_by_official_order(paths: list[Path], suite: str) -> list[Path]:
    official_files = official_suite_files(suite, Path("."))
    if not official_files:
        return sorted(paths)
    order = {path.stem: index for index, path in enumerate(official_files)}
    return sorted(paths, key=lambda path: (order.get(path.stem, len(order)), path.name))


def load_instruction_order(
    suite: str,
    bddl_dir: Path,
    instructions_file: Path | None,
    data_dir: Path | None,
) -> list[Path]:
    instruction_path = instructions_file or (DOCS_ROOT / f"{suite}_instructions.txt")
    if instruction_path.exists():
        ordered_files: list[Path] = []
        for line in instruction_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("file: "):
                continue
            file_name = stripped.split("file: ", maxsplit=1)[1].strip()
            ordered_files.append(resolve_bddl_reference(file_name, base_dir=bddl_dir))
        if ordered_files:
            return ordered_files

    official_files = [path.resolve() for path in official_suite_files(suite, bddl_dir)]
    existing_official_files = [path for path in official_files if path.exists()]
    if existing_official_files:
        return existing_official_files

    if data_dir is not None and data_dir.exists():
        data_order = collect_task_files_from_data_dir(data_dir)
        if data_order:
            return sort_by_official_order(data_order, suite)

    return sort_by_official_order(list(bddl_dir.glob("*.bddl")), suite)


def resolve_task_file(
    task_id: int,
    suite: str,
    bddl_dir: Path,
    instructions_file: Path | None,
    data_dir: Path | None,
) -> Path:
    ordered_files = load_instruction_order(
        suite=suite,
        bddl_dir=bddl_dir,
        instructions_file=instructions_file,
        data_dir=data_dir,
    )
    if not ordered_files:
        raise FileNotFoundError(
            f"No .bddl or .hdf5 tasks found for suite {suite} "
            f"(bddl_dir={bddl_dir}, data_dir={data_dir})"
        )
    if task_id < 1 or task_id > len(ordered_files):
        raise IndexError(
            f"task-id must be between 1 and {len(ordered_files)}, got {task_id}"
        )
    task_file = ordered_files[task_id - 1]
    if not task_file.exists():
        raise FileNotFoundError(f"Resolved task file does not exist: {task_file}")
    return task_file


def bddl_scene_name(bddl_path: Path) -> str | None:
    match = re.search(r"\(:scene\s+([^\s\)]+)", bddl_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def bddl_object_types(bddl_path: Path) -> set[str]:
    try:
        objects_section = extract_section_span(
            bddl_path.read_text(encoding="utf-8"), "objects"
        )[2]
    except ValueError:
        return set()
    return {object_type for _name, object_type in parse_typed_section(objects_section)}


def resolve_scene_file(
    scene_name: str,
    suite: str,
    bddl_dir: Path,
    instructions_file: Path | None,
    data_dir: Path | None,
    object_types: list[str] | None = None,
) -> Path:
    scene_ref = Path(scene_name)
    if scene_ref.exists():
        return scene_ref.resolve()

    ordered_files = load_instruction_order(
        suite=suite,
        bddl_dir=bddl_dir,
        instructions_file=instructions_file,
        data_dir=data_dir,
    )
    all_files = ordered_files + [path for path in sorted(bddl_dir.glob("*.bddl")) if path not in ordered_files]
    if not all_files:
        raise FileNotFoundError(
            f"No .bddl tasks found for suite {suite} "
            f"(bddl_dir={bddl_dir}, data_dir={data_dir})"
        )

    query = scene_ref.stem.lower() if scene_ref.suffix == ".bddl" else scene_name.lower()
    exact_matches: list[Path] = []
    prefix_matches: list[Path] = []
    loose_matches: list[Path] = []
    query_prefix = f"{query}_"
    for path in all_files:
        stem = path.stem.lower()
        internal_scene = (bddl_scene_name(path) or "").lower()
        if query in {stem, path.name.lower(), internal_scene}:
            exact_matches.append(path)
        elif stem.startswith(query_prefix) or (internal_scene and internal_scene.startswith(query_prefix)):
            prefix_matches.append(path)
        elif query in stem or (internal_scene and query in internal_scene):
            loose_matches.append(path)

    matches = exact_matches or prefix_matches or loose_matches
    if object_types:
        requested = set(object_types)
        object_matches = [
            path for path in matches
            if requested.issubset(bddl_object_types(path))
        ]
        if object_matches:
            matches = object_matches

    unique_matches = []
    seen = set()
    for path in matches:
        resolved = path.resolve()
        if resolved not in seen:
            unique_matches.append(resolved)
            seen.add(resolved)

    if not unique_matches:
        raise FileNotFoundError(
            f"Could not resolve scene-name '{scene_name}' in suite {suite}"
        )
    if len(unique_matches) > 1:
        preview = ", ".join(path.name for path in unique_matches[:10])
        raise ValueError(
            f"scene-name '{scene_name}' matched {len(unique_matches)} tasks. "
            f"Pass a more specific BDDL stem/path or an --object-name that narrows it. "
            f"Matches: {preview}"
        )
    return unique_matches[0]


def extract_section_span(text: str, section_name: str) -> tuple[int, int, str]:
    token = f"(:{section_name}"
    start = text.find(token)
    if start == -1:
        raise ValueError(f"Could not find section {token}")

    depth = 0
    end = -1
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end == -1:
        raise ValueError(f"Could not find the end of section {token}")
    return start, end, text[start:end]


def append_entries_to_section(section_text: str, entries: str, closing_indent: str) -> str:
    stripped = section_text.rstrip()
    if not stripped.endswith(")"):
        raise ValueError("Section does not end with ')'")
    return stripped[:-1] + entries + f"\n{closing_indent})"


def parse_typed_section(section_text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in section_text.splitlines()[1:-1]:
        line = raw_line.strip()
        if not line or line.startswith("(") or " - " not in line:
            continue
        names_part, object_type = [part.strip() for part in line.split(" - ", maxsplit=1)]
        for name in names_part.split():
            entries.append((name, object_type))
    return entries


def make_instance_name(object_type: str, existing_names: Iterable[str], extra_index: int) -> str:
    existing_name_set = set(existing_names)
    base = f"{object_type}_extra_{extra_index}"
    if base not in existing_name_set:
        return base

    suffix = 1
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in existing_name_set:
            return candidate
        suffix += 1


def detect_workspace_name(fixtures: list[tuple[str, str]]) -> str:
    for fixture_name, fixture_type in fixtures:
        if fixture_type in WORKSPACE_TYPES:
            return fixture_name
    if not fixtures:
        raise ValueError("No fixtures found in task file")
    return fixtures[0][0]


def collect_existing_centroids(problem_dict: dict, workspace_name: str) -> list[tuple[float, float]]:
    centroids: list[tuple[float, float]] = []
    for region_info in problem_dict["regions"].values():
        if region_info["target"] != workspace_name:
            continue
        for rect_range in region_info["ranges"]:
            xmin, ymin, xmax, ymax = rect_range
            centroids.append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))
    return centroids


def object_footprint(object_type: str) -> tuple[float, float]:
    return OBJECT_FOOTPRINTS.get(object_type, DEFAULT_OBJECT_FOOTPRINT)


def extract_on_region_mapping(init_section: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern = re.compile(r"\(On\s+([^\s]+)\s+([^\s\)]+)\)")
    for instance_name, region_name in pattern.findall(init_section):
        mapping[instance_name] = region_name
    return mapping


def region_centroid(region_info: dict) -> tuple[float, float]:
    if not region_info["ranges"]:
        raise ValueError("Region has no ranges")
    xmin, ymin, xmax, ymax = region_info["ranges"][0]
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def collect_fixture_exclusion_zones(
    parsed_problem: dict,
    fixtures: list[tuple[str, str]],
    init_section: str,
) -> list[tuple[float, float, float]]:
    on_region_mapping = extract_on_region_mapping(init_section)
    zones: list[tuple[float, float, float]] = []

    for fixture_name, fixture_type in fixtures:
        region_name = on_region_mapping.get(fixture_name)
        if not region_name:
            continue
        region_info = parsed_problem["regions"].get(region_name)
        if not region_info:
            continue
        clearance = FIXTURE_CLEARANCE.get(fixture_type)
        if clearance is None:
            continue
        cx, cy = region_centroid(region_info)
        zones.append((cx, cy, clearance))

    return zones


def collect_fixture_exclusion_rects(
    parsed_problem: dict,
    fixtures: list[tuple[str, str]],
    init_section: str,
) -> list[tuple[float, float, float, float]]:
    on_region_mapping = extract_on_region_mapping(init_section)
    rects: list[tuple[float, float, float, float]] = []

    for fixture_name, fixture_type in fixtures:
        region_name = on_region_mapping.get(fixture_name)
        if not region_name:
            continue
        region_info = parsed_problem["regions"].get(region_name)
        if not region_info:
            continue
        half_extents = FIXTURE_FOOTPRINTS.get(fixture_type)
        if half_extents is None:
            continue
        cx, cy = region_centroid(region_info)
        hx, hy = half_extents
        rects.append((cx, cy, hx, hy))

    return rects


def collect_existing_object_rects(
    parsed_problem: dict,
    workspace_name: str,
    existing_objects: list[tuple[str, str]],
    init_section: str,
) -> list[tuple[float, float, float, float]]:
    on_region_mapping = extract_on_region_mapping(init_section)
    rects: list[tuple[float, float, float, float]] = []

    for object_name, object_type in existing_objects:
        region_name = on_region_mapping.get(object_name)
        if not region_name:
            continue
        region_info = parsed_problem["regions"].get(region_name)
        if not region_info or region_info["target"] != workspace_name:
            continue
        cx, cy = region_centroid(region_info)
        hx, hy = object_footprint(object_type)
        rects.append((cx, cy, hx, hy))

    return rects


def rects_too_close(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
    margin: float,
) -> bool:
    ax, ay, ahx, ahy = rect_a
    bx, by, bhx, bhy = rect_b
    return (
        abs(ax - bx) < ahx + bhx + margin
        and abs(ay - by) < ahy + bhy + margin
    )


def shuffled_candidate_positions(rng: random.Random) -> list[tuple[float, float]]:
    candidates = CANDIDATE_POSITIONS[:]
    seen = {candidate for candidate in candidates}
    for x in CANDIDATE_GRID_X:
        for y in CANDIDATE_GRID_Y:
            candidate = (x, y)
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    rng.shuffle(candidates)
    return candidates


def sample_extra_positions(
    rng: random.Random,
    existing_centroids: list[tuple[float, float]],
    fixture_zones: list[tuple[float, float, float]],
    object_types: list[str],
    existing_object_rects: list[tuple[float, float, float, float]] | None = None,
    fixture_rects: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float]]:
    candidates = shuffled_candidate_positions(rng)
    existing_object_rects = existing_object_rects or []
    fixture_rects = fixture_rects or []

    object_radii = [
        OBJECT_CLEARANCE.get(object_type, DEFAULT_OBJECT_CLEARANCE)
        for object_type in object_types
    ]
    object_footprints = [object_footprint(object_type) for object_type in object_types]
    order = sorted(
        range(len(object_types)),
        key=lambda index: (
            object_footprints[index][0] * object_footprints[index][1],
            object_radii[index],
        ),
        reverse=True,
    )
    placements: list[tuple[float, float] | None] = [None] * len(object_types)
    chosen: list[tuple[float, float, float, float, float]] = []

    def backtrack(order_index: int, slack: float, use_existing_centroids: bool) -> bool:
        if order_index == len(order):
            return True

        object_index = order[order_index]
        object_radius = object_radii[object_index]
        object_hx, object_hy = object_footprints[object_index]
        for x, y in candidates:
            candidate_rect = (x, y, object_hx, object_hy)
            if use_existing_centroids and any(
                np.hypot(x - cx, y - cy)
                < max(object_radius, EXISTING_OBJECT_CLEARANCE) + slack
                for cx, cy in existing_centroids
            ):
                continue
            if any(
                np.hypot(x - cx, y - cy) < object_radius + radius + slack
                for cx, cy, radius in fixture_zones
            ):
                continue
            if any(
                rects_too_close(candidate_rect, rect, margin=slack)
                for rect in existing_object_rects
            ):
                continue
            if any(
                rects_too_close(candidate_rect, rect, margin=slack)
                for rect in fixture_rects
            ):
                continue
            if any(
                rects_too_close(candidate_rect, (cx, cy, hx, hy), margin=slack)
                for cx, cy, _chosen_radius, hx, hy in chosen
            ):
                continue

            placements[object_index] = (x, y)
            chosen.append((x, y, object_radius, object_hx, object_hy))
            if backtrack(order_index + 1, slack, use_existing_centroids):
                return True
            chosen.pop()
            placements[object_index] = None
        return False

    for slack, use_existing_centroids in [
        (0.05, True),
        (0.035, True),
        (0.02, True),
        (0.01, False),
        (0.0, False),
    ]:
        placements = [None] * len(object_types)
        chosen.clear()
        if backtrack(0, slack, use_existing_centroids):
            return [placement for placement in placements if placement is not None]

    center_chosen: list[tuple[float, float, float]] = []

    def center_backtrack(order_index: int) -> bool:
        if order_index == len(order):
            return True

        object_index = order[order_index]
        object_radius = min(object_radii[object_index], 0.10)
        object_hx, object_hy = object_footprints[object_index]
        for x, y in candidates:
            candidate_rect = (x, y, object_hx, object_hy)
            if any(
                rects_too_close(candidate_rect, rect, margin=0.0)
                for rect in fixture_rects
            ):
                continue
            if any(
                rects_too_close(candidate_rect, rect, margin=0.0)
                for rect in existing_object_rects
            ):
                continue
            if any(
                np.hypot(x - cx, y - cy) < max(object_radius, radius, 0.075)
                for cx, cy, radius in center_chosen
            ):
                continue

            placements[object_index] = (x, y)
            center_chosen.append((x, y, object_radius))
            if center_backtrack(order_index + 1):
                return True
            center_chosen.pop()
            placements[object_index] = None
        return False

    placements = [None] * len(object_types)
    if center_backtrack(0):
        return [placement for placement in placements if placement is not None]

    raise ValueError(
        "Could not find collision-aware placements for: "
        + ", ".join(object_types)
    )


def choose_extra_object_types(
    rng: random.Random, existing_object_types: Iterable[str], num_extra_objects: int
) -> list[str]:
    existing_types = set(existing_object_types)
    available = [obj for obj in EXTRA_OBJECT_POOL if obj not in existing_types]
    if len(available) < num_extra_objects:
        raise ValueError(
            f"Only {len(available)} candidate objects remain, need {num_extra_objects}"
        )
    return rng.sample(available, num_extra_objects)


def list_available_extra_object_types(
    existing_object_types: Iterable[str],
) -> list[str]:
    existing_types = set(existing_object_types)
    return [obj for obj in EXTRA_OBJECT_POOL if obj not in existing_types]


def validate_forced_object_types(
    forced_object_types: list[str],
    existing_object_types: Iterable[str],
    allow_existing_types: bool = False,
) -> list[str]:
    normalized = [obj.strip() for obj in forced_object_types if obj.strip()]
    if not normalized:
        raise ValueError("--force-objects was provided but no valid object types were found")

    registered_object_types = set(get_object_dict().keys())
    unknown = [obj for obj in normalized if obj not in registered_object_types]
    if unknown:
        raise ValueError(
            "Unsupported forced object types: " + ", ".join(sorted(unknown))
        )

    existing_types = set(existing_object_types)
    duplicates_in_scene = [obj for obj in normalized if obj in existing_types]
    if duplicates_in_scene and not allow_existing_types:
        raise ValueError(
            "Forced object types already exist in this scene: "
            + ", ".join(sorted(duplicates_in_scene))
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError("--force-objects contains duplicates")

    return normalized


def replace_section(original_text: str, section_name: str, replacement: str) -> str:
    start, end, _section = extract_section_span(original_text, section_name)
    return original_text[:start] + replacement + original_text[end:]


def remove_object_init_predicates(init_section: str, object_names: Iterable[str]) -> str:
    object_name_set = set(object_names)
    filtered_lines: list[str] = []
    for raw_line in init_section.splitlines():
        tokens = set(re.findall(r"[A-Za-z0-9_]+", raw_line))
        if tokens.intersection(object_name_set):
            continue
        filtered_lines.append(raw_line)
    return "\n".join(filtered_lines)


def build_modified_bddl(
    original_bddl_path: Path,
    rng: random.Random,
    num_extra_objects: int,
    forced_object_types: list[str] | None = None,
    replace_scene_objects: bool = False,
) -> tuple[str, list[str]]:
    original_text = original_bddl_path.read_text(encoding="utf-8")
    parsed = robosuite_parse_problem(str(original_bddl_path))

    objects_start, objects_end, objects_section = extract_section_span(original_text, "objects")
    regions_start, regions_end, regions_section = extract_section_span(original_text, "regions")
    init_start, init_end, init_section = extract_section_span(original_text, "init")

    existing_objects = parse_typed_section(objects_section)
    fixtures = parse_typed_section(extract_section_span(original_text, "fixtures")[2])
    workspace_name = detect_workspace_name(fixtures)

    existing_names = [name for name, _ in existing_objects]
    existing_types = [obj_type for _, obj_type in existing_objects]
    if forced_object_types is not None:
        extra_object_types = validate_forced_object_types(
            forced_object_types,
            existing_types,
            allow_existing_types=replace_scene_objects,
        )
        num_extra_objects = len(extra_object_types)
    else:
        extra_object_types = choose_extra_object_types(
            rng, existing_types, num_extra_objects
        )

    existing_centroids = (
        []
        if replace_scene_objects
        else collect_existing_centroids(parsed, workspace_name)
    )
    fixture_zones = collect_fixture_exclusion_zones(parsed, fixtures, init_section)
    existing_object_rects = (
        []
        if replace_scene_objects
        else collect_existing_object_rects(
            parsed, workspace_name, existing_objects, init_section
        )
    )
    fixture_rects = collect_fixture_exclusion_rects(parsed, fixtures, init_section)
    if forced_object_types is not None:
        extra_positions = sample_extra_positions(
            rng,
            existing_centroids,
            fixture_zones,
            object_types=extra_object_types,
            existing_object_rects=existing_object_rects,
            fixture_rects=fixture_rects,
        )
    else:
        available_object_types = list_available_extra_object_types(existing_types)
        last_placement_error: Exception | None = None
        extra_positions = []
        for _ in range(50):
            extra_object_types = rng.sample(available_object_types, num_extra_objects)
            try:
                extra_positions = sample_extra_positions(
                    rng,
                    existing_centroids,
                    fixture_zones,
                    object_types=extra_object_types,
                    existing_object_rects=existing_object_rects,
                    fixture_rects=fixture_rects,
                )
                break
            except ValueError as exc:
                last_placement_error = exc
        else:
            raise ValueError(
                "Could not find collision-aware placements after resampling "
                f"{len(available_object_types)} candidate object types"
            ) from last_placement_error

    object_lines: list[str] = []
    region_lines: list[str] = []
    init_lines: list[str] = []
    added_instance_names: list[str] = []
    placement_metadata: list[dict] = []

    for extra_index, (object_type, (x, y)) in enumerate(
        zip(extra_object_types, extra_positions), start=1
    ):
        instance_name = (
            object_type
            if replace_scene_objects
            else make_instance_name(object_type, existing_names, extra_index)
        )
        region_name = f"{instance_name}_init_region"
        added_instance_names.append(instance_name)
        existing_names.append(instance_name)
        region_range = [
            round(x - REGION_HALF_SIZE, 6),
            round(y - REGION_HALF_SIZE, 6),
            round(x + REGION_HALF_SIZE, 6),
            round(y + REGION_HALF_SIZE, 6),
        ]
        placement_metadata.append(
            {
                "instance_name": instance_name,
                "object_type": object_type,
                "workspace": workspace_name,
                "region_name": region_name,
                "xy": [round(x, 6), round(y, 6)],
                "region_range": region_range,
                "clearance_radius": OBJECT_CLEARANCE.get(
                    object_type, DEFAULT_OBJECT_CLEARANCE
                ),
            }
        )

        object_lines.append(f"\n    {instance_name} - {object_type}")
        region_lines.append(
            "\n"
            f"      ({region_name}\n"
            f"          (:target {workspace_name})\n"
            "          (:ranges (\n"
            f"              ({region_range[0]:.6f} {region_range[1]:.6f} {region_range[2]:.6f} {region_range[3]:.6f})\n"
            "            )\n"
            "          )\n"
            "          (:yaw_rotation (\n"
            "              (0.0 0.0)\n"
            "            )\n"
            "          )\n"
            "      )"
        )
        init_lines.append(f"\n    (On {instance_name} {workspace_name}_{region_name})")

    if replace_scene_objects:
        new_objects_section = "(:objects" + "".join(object_lines) + "\n  )"
    else:
        new_objects_section = append_entries_to_section(
            objects_section, "".join(object_lines), closing_indent="  "
        )
    new_regions_section = append_entries_to_section(
        regions_section, "".join(region_lines), closing_indent="    "
    )
    if replace_scene_objects:
        fixture_init_section = remove_object_init_predicates(
            init_section, [name for name, _object_type in existing_objects]
        )
        new_init_section = append_entries_to_section(
            fixture_init_section, "".join(init_lines), closing_indent="  "
        )
    else:
        new_init_section = append_entries_to_section(
            init_section, "".join(init_lines), closing_indent="  "
        )

    if replace_scene_objects and added_instance_names:
        first_instance = added_instance_names[0]
        first_region = f"{workspace_name}_{first_instance}_init_region"
        new_goal_section = (
            "(:goal\n"
            f"    (And (On {first_instance} {first_region}))\n"
            "  )"
        )
        new_obj_interest_section = (
            "(:obj_of_interest\n"
            + "".join(f"    {name}\n" for name in added_instance_names)
            + "  )"
        )

    updated_text = original_text
    replacements = [
        (objects_start, objects_end, new_objects_section),
        (regions_start, regions_end, new_regions_section),
        (init_start, init_end, new_init_section),
    ]
    if replace_scene_objects and added_instance_names:
        try:
            goal_start, goal_end, _goal_section = extract_section_span(original_text, "goal")
            replacements.append((goal_start, goal_end, new_goal_section))
        except ValueError:
            pass
        try:
            interest_start, interest_end, _interest_section = extract_section_span(
                original_text, "obj_of_interest"
            )
            replacements.append((interest_start, interest_end, new_obj_interest_section))
        except ValueError:
            pass

    for start, end, replacement in sorted(replacements, reverse=True):
        updated_text = updated_text[:start] + replacement + updated_text[end:]

    return updated_text, extra_object_types, placement_metadata


def build_env_kwargs(
    bddl_file: Path, camera_name: str, image_size: int
) -> tuple[dict, str]:
    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    problem_info = get_problem_info(str(bddl_file))
    problem_name = problem_info["problem_name"]
    env_kwargs = {
        "bddl_file_name": str(bddl_file),
        "robots": ["Panda"],
        "controller_configs": controller_configs,
        "gripper_types": "default",
        "initialization_noise": None,
        "use_camera_obs": True,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "render_camera": camera_name,
        "render_collision_mesh": False,
        "render_visual_mesh": True,
        "render_gpu_device_id": -1,
        "control_freq": 20,
        "horizon": 1000,
        "ignore_done": False,
        "hard_reset": True,
        "camera_names": [camera_name],
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_depths": False,
        "camera_segmentations": None,
        "renderer": "mujoco",
        "renderer_config": None,
    }
    return env_kwargs, problem_name


def get_action_dim(env) -> int:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "action_spec"):
        low, _high = env.action_spec
        return int(np.asarray(low).shape[0])
    raise AttributeError("Could not determine action dimension for environment")


def reset_with_retries(env, max_attempts: int):
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return env.reset()
        except RandomizationError as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to reset the environment after {max_attempts} attempts"
    ) from last_error


def render_task_image(
    bddl_file: Path,
    output_path: Path,
    camera_name: str,
    image_size: int,
    settle_steps: int,
    max_reset_attempts: int,
) -> None:
    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    env = TASK_MAPPING[problem_name](**env_kwargs)
    try:
        obs = reset_with_retries(env, max_attempts=max_reset_attempts)
        action = np.zeros(get_action_dim(env), dtype=np.float32)
        for _ in range(max(0, settle_steps)):
            obs, _, _, _ = env.step(action)

        image_key = f"{camera_name}_image"
        if image_key not in obs:
            raise KeyError(f"{image_key} not found in observation keys: {list(obs.keys())}")
        image = obs[image_key]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image[::-1, :, ::-1])
    finally:
        env.close()


def slugify_task_name(task_file: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", task_file.stem)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    bddl_dir = (args.bddl_dir or SUITE_BDDL_DIRS[args.suite]).resolve()
    data_dir = args.data_dir
    if data_dir is None:
        default_data_dir = args.data_root / args.suite
        data_dir = default_data_dir if default_data_dir.exists() else None
    if data_dir is not None:
        data_dir = data_dir.resolve()
    instructions_file = (
        args.instructions_file.resolve()
        if args.instructions_file is not None
        else None
    )

    if args.scene_name is not None:
        task_file = resolve_scene_file(
            args.scene_name,
            suite=args.suite,
            bddl_dir=bddl_dir,
            instructions_file=instructions_file,
            data_dir=data_dir,
            object_types=args.force_objects,
        )
        task_id_for_output = 0
    else:
        task_file = resolve_task_file(
            args.task_id,
            suite=args.suite,
            bddl_dir=bddl_dir,
            instructions_file=instructions_file,
            data_dir=data_dir,
        )
        task_id_for_output = args.task_id
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_task_name(task_file)
    if args.scene_name is not None:
        prefix = f"{args.suite}_scene_{slug}"
    else:
        prefix = f"{args.suite}_{task_id_for_output:02d}_{slug}"
    original_png = output_dir / f"{prefix}_original.png"
    modified_bddl = output_dir / f"{prefix}_reconfigured.bddl"
    modified_png = output_dir / f"{prefix}_reconfigured.png"

    render_task_image(
        bddl_file=task_file,
        output_path=original_png,
        camera_name=args.camera_name,
        image_size=args.image_size,
        settle_steps=args.settle_steps,
        max_reset_attempts=args.max_reset_attempts,
    )

    modified_text, added_object_types, placement_metadata = build_modified_bddl(
        task_file,
        rng=rng,
        num_extra_objects=args.num_extra_objects,
        forced_object_types=args.force_objects,
        replace_scene_objects=args.replace_scene_objects,
    )
    modified_bddl.write_text(modified_text, encoding="utf-8")
    if args.placement_metadata_output is not None:
        args.placement_metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.placement_metadata_output.write_text(
            json.dumps(
                {
                    "suite": args.suite,
                    "task_id": task_id_for_output,
                    "scene_name": args.scene_name,
                    "task_file": str(task_file),
                    "workspace": placement_metadata[0]["workspace"]
                    if placement_metadata
                    else None,
                    "added_objects": placement_metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    render_task_image(
        bddl_file=modified_bddl,
        output_path=modified_png,
        camera_name=args.camera_name,
        image_size=args.image_size,
        settle_steps=args.settle_steps,
        max_reset_attempts=args.max_reset_attempts,
    )

    print(f"Suite: {args.suite}")
    if args.scene_name is not None:
        print(f"Scene: {args.scene_name} -> {task_file.name}")
    else:
        print(f"Task #{args.task_id}: {task_file.name}")
    print(f"Added extra object types: {', '.join(added_object_types)}")
    print(f"Original render: {original_png}")
    print(f"Modified BDDL: {modified_bddl}")
    print(f"Reconfigured render: {modified_png}")


if __name__ == "__main__":
    main()
