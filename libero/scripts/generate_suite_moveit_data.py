#!/usr/bin/env python3
"""Generate MoveIt pick/place trajectories from VLAPB suite BDDL files.

The script is intentionally usable in two modes:

* --dry-run converts suite episodes into pick/place jobs and writes JSON
  plans without importing ROS.
* normal execution imports moveit_commander at runtime and asks MoveIt to plan
  each primitive segment.

Pose input is kept explicit because the suite metadata knows which object should
be moved, but not where that object currently is in the robot/world frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


VLAPB_LIBERO_ROOT = Path("/home/artemis/Documents/VLAPB/libero")
VLAPB_PROJECT_ROOT = Path("/home/artemis/Documents/VLAPB")
DEFAULT_SUITES_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"
DEFAULT_POSES = VLAPB_LIBERO_ROOT / "teleop_moveit" / "moveit_poses.json"
DEFAULT_OUTPUT_DIR = VLAPB_LIBERO_ROOT / "teleop_moveit" / "moveit"

DEFAULT_GROUP = "panda_arm"
DEFAULT_GRIPPER_GROUP = "hand"
DEFAULT_EEF_LINK = "panda_link8"
DEFAULT_REFERENCE_FRAME = "world"


@dataclass(frozen=True)
class PoseSpec:
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class RegionSpec:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class SceneObject:
    instance_name: str
    object_class: str
    pose: PoseSpec
    is_fixture: bool = False


@dataclass(frozen=True)
class PickPlaceStep:
    object_name: str
    receptacle_name: str
    pick_pose: PoseSpec
    place_pose: PoseSpec
    place_region: RegionSpec | None
    success_mode: str
    sequence_index: int


@dataclass(frozen=True)
class EpisodeJob:
    episode_id: str
    episode_type: str
    user_id: str
    instruction: str
    sequence_matters: bool
    success_mode: str
    steps: tuple[PickPlaceStep, ...]
    scene_objects: tuple[SceneObject, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobFailure:
    episode_id: str
    error: str


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def suite_bddl_files(suites_root: Path = DEFAULT_SUITES_ROOT) -> list[Path]:
    return sorted(suites_root.glob("*/**/bddl_files/*.bddl"))


def metadata_path_for_bddl(bddl_file: Path) -> Path:
    split_dir = bddl_file.parent.parent
    return split_dir / "metadata" / f"{bddl_file.stem}.json"


def read_metadata_for_bddl(bddl_file: Path) -> dict[str, Any]:
    metadata_path = metadata_path_for_bddl(bddl_file)
    if metadata_path.exists():
        return read_json(metadata_path)
    split_dir = bddl_file.parent.parent
    return {
        "episode_id": bddl_file.stem,
        "suite": split_dir.parent.name,
        "split": split_dir.name,
        "task_type": split_dir.name,
        "language": "",
        "bddl_file": str(bddl_file),
    }


def row_from_bddl(bddl_file: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    metadata = read_metadata_for_bddl(bddl_file)
    episode_id = str(metadata.get("episode_id") or bddl_file.stem)
    suite_name = str(metadata.get("suite") or bddl_file.parent.parent.parent.name)
    split = str(metadata.get("split") or bddl_file.parent.parent.name)
    task_type = str(metadata.get("task_type") or split)
    target_user = str(metadata.get("target_user") or (metadata.get("participating_users") or [""])[0])
    target_object = metadata.get("target_object")
    target_sequence = metadata.get("target_sequence")
    if target_object is None and isinstance(target_sequence, list) and target_sequence:
        target_object = target_sequence[0]
    target_receptacle = metadata.get("fixture")
    if target_receptacle is None and metadata.get("fixtures"):
        target_receptacle = metadata["fixtures"][0].get("fixture")

    output_base = output_dir / "hdf5" / suite_name / split
    raw_base = output_dir / "raw" / suite_name / split
    return {
        "episode_id": episode_id,
        "suite": suite_name,
        "split": split,
        "episode_type": f"{suite_name}_{task_type}",
        "task_type": task_type,
        "user_id": target_user,
        "instruction": str(metadata.get("language") or ""),
        "target_object": target_object,
        "target_sequence": target_sequence,
        "target_receptacle": target_receptacle,
        "bddl_file": str(bddl_file),
        "output_hdf5": str(output_base / f"{episode_id}_demo.hdf5"),
        "raw_hdf5": str(raw_base / f"{episode_id}_raw.hdf5"),
        "metadata": metadata,
    }


def selected_suite_rows(
    *,
    suites_root: Path,
    output_dir: Path,
    episode_id: str | None,
    collect_all: bool,
    suite_name: str | None,
    split: str | None,
    limit: int | None,
    offset: int,
) -> list[dict[str, Any]]:
    rows = [row_from_bddl(path, output_dir) for path in suite_bddl_files(suites_root)]
    if suite_name:
        rows = [row for row in rows if row["suite"] == suite_name]
    if split:
        rows = [row for row in rows if row["split"] == split]
    if episode_id:
        requested = Path(episode_id)
        if requested.exists():
            requested_resolved = requested.resolve()
            suites_root_resolved = suites_root.resolve()
            try:
                requested_resolved.relative_to(suites_root_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Suite episode path must be under {suites_root_resolved}: {requested_resolved}"
                ) from exc
            rows = [row_from_bddl(requested, output_dir)]
        else:
            rows = [
                row
                for row in rows
                if row["episode_id"] == episode_id or Path(row["bddl_file"]).stem == episode_id
            ]
        if not rows:
            raise ValueError(f"Suite episode not found: {episode_id}")
    elif not collect_all:
        if not rows:
            raise ValueError(f"No suite BDDL files found under {suites_root}")
        rows = [rows[0]]
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No suite episodes matched the selection")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def path_from_row(row: dict[str, Any]) -> Path | None:
    value = row.get("bddl_file")
    return Path(value) if value else None


def canonical_object(name: str, aliases: dict[str, str]) -> str:
    normalized = normalize(name)
    return aliases.get(normalized, normalized)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def tuple_of_float(values: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must be a list of {length} numbers, got {values!r}")
    return tuple(float(value) for value in values)


def pose_from_payload(payload: dict[str, Any], label: str) -> PoseSpec:
    xyz = tuple_of_float(payload.get("xyz"), 3, f"{label}.xyz")
    quat = tuple_of_float(payload.get("quat_xyzw", (0.0, 1.0, 0.0, 0.0)), 4, f"{label}.quat_xyzw")
    return PoseSpec(xyz=xyz, quat_xyzw=quat)  # type: ignore[arg-type]


def region_from_payload(payload: dict[str, Any], label: str) -> RegionSpec:
    center = tuple_of_float(payload.get("center"), 3, f"{label}.center")
    size = tuple_of_float(payload.get("size"), 3, f"{label}.size")
    quat = tuple_of_float(payload.get("quat_xyzw", (0.0, 1.0, 0.0, 0.0)), 4, f"{label}.quat_xyzw")
    return RegionSpec(center=center, size=size, quat_xyzw=quat)  # type: ignore[arg-type]


def load_pose_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Pose DB does not exist: {path}. "
            "Create a template with --write-pose-template, then replace placeholder coordinates "
            "with robot/world-frame poses."
        )
    payload = read_json(path)
    if "objects" not in payload:
        raise ValueError(f"{path} must contain an 'objects' mapping")
    if "receptacles" not in payload:
        raise ValueError(f"{path} must contain a 'receptacles' mapping")
    return payload


def object_pick_pose(pose_db: dict[str, Any], object_name: str) -> PoseSpec:
    objects = pose_db["objects"]
    if object_name not in objects:
        raise KeyError(f"Missing pick pose for object '{object_name}' in pose DB")
    entry = objects[object_name]
    if "pick" in entry:
        entry = entry["pick"]
    return pose_from_payload(entry, f"objects.{object_name}.pick")


def receptacle_region(pose_db: dict[str, Any], receptacle_name: str) -> RegionSpec:
    receptacles = pose_db["receptacles"]
    if receptacle_name not in receptacles:
        raise KeyError(f"Missing receptacle '{receptacle_name}' in pose DB")
    entry = receptacles[receptacle_name]
    if "place_region" in entry:
        entry = entry["place_region"]
    return region_from_payload(entry, f"receptacles.{receptacle_name}.place_region")


def parse_bddl_typed_section(text: str, section_name: str) -> dict[str, str]:
    objects: dict[str, str] = {}
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(f"(:{section_name}"):
            in_section = True
            continue
        if in_section and line.startswith(")"):
            break
        if in_section and " - " in line:
            instance, object_class = line.rsplit(" - ", 1)
            objects[instance.strip()] = normalize(object_class)
    return objects


def parse_bddl_objects(text: str) -> dict[str, str]:
    return parse_bddl_typed_section(text, "objects")


def parse_bddl_fixtures(text: str) -> dict[str, str]:
    return parse_bddl_typed_section(text, "fixtures")


def parse_bddl_region_centers(text: str) -> dict[str, tuple[float, float]]:
    regions: dict[str, tuple[float, float]] = {}
    pattern = re.compile(
        r"\((?P<name>[A-Za-z0-9_]+)\s+"
        r"\(:target\s+(?P<target>[A-Za-z0-9_]+)\)\s+"
        r"\(:ranges\s+\(\s+\("
        r"(?P<x0>-?\d+(?:\.\d+)?)\s+"
        r"(?P<y0>-?\d+(?:\.\d+)?)\s+"
        r"(?P<x1>-?\d+(?:\.\d+)?)\s+"
        r"(?P<y1>-?\d+(?:\.\d+)?)"
        r"\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        x0 = float(match.group("x0"))
        y0 = float(match.group("y0"))
        x1 = float(match.group("x1"))
        y1 = float(match.group("y1"))
        regions[match.group("name")] = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    return regions


def parse_bddl_init_regions(text: str, region_names: Iterable[str]) -> dict[str, str]:
    init_regions: dict[str, str] = {}
    sorted_regions = sorted(region_names, key=len, reverse=True)
    for instance, region_ref in re.findall(r"\(On\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", text):
        for region_name in sorted_regions:
            if region_ref.endswith(region_name):
                init_regions[instance] = region_name
                break
    return init_regions


def bddl_path_for_episode(row: dict[str, Any], bddl_dir: Path) -> Path:
    row_path = path_from_row(row)
    if row_path and row_path.exists():
        return row_path
    raise FileNotFoundError(f"Missing BDDL for {row['episode_id']}: {row_path}")


def libero_xy_to_moveit_xyz(
    xy: tuple[float, float],
    args: argparse.Namespace,
    z: float,
) -> tuple[float, float, float]:
    x, y = xy
    source_x, source_y = args.libero_origin_xy
    target_x, target_y, target_z = args.moveit_origin_xyz
    yaw = math.radians(args.libero_to_moveit_yaw_deg)
    sx, sy = args.libero_to_moveit_scale_xy
    dx = (x - source_x) * sx
    dy = (y - source_y) * sy
    rotated_x = math.cos(yaw) * dx - math.sin(yaw) * dy
    rotated_y = math.sin(yaw) * dx + math.cos(yaw) * dy
    return (
        target_x + rotated_x,
        target_y + rotated_y,
        target_z + z,
    )


def bddl_pose_db_for_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    aliases: dict[str, str],
) -> dict[str, Any]:
    text = bddl_path_for_episode(row, args.suites_root).read_text(encoding="utf-8")
    objects_by_instance = parse_bddl_objects(text)
    fixtures_by_instance = parse_bddl_fixtures(text)
    regions = parse_bddl_region_centers(text)
    init_regions = parse_bddl_init_regions(text, regions)

    object_payload: dict[str, Any] = {}
    receptacle_payload: dict[str, Any] = {}
    scene_objects: list[dict[str, Any]] = []

    target_receptacle = normalize(row.get("target_receptacle") or "basket")
    target_receptacle = aliases.get(target_receptacle, target_receptacle)
    target_objects = set(episode_objects(row, aliases))

    scene_entities = {
        **objects_by_instance,
        **{
            instance: object_class
            for instance, object_class in fixtures_by_instance.items()
            if object_class == target_receptacle
        },
    }
    for instance, object_class in scene_entities.items():
        region_name = init_regions.get(instance)
        if not region_name or region_name not in regions:
            continue
        xyz = libero_xy_to_moveit_xyz(regions[region_name], args, args.bddl_object_z)
        entry = {
            "pick": {
                "xyz": [round(value, 6) for value in xyz],
                "quat_xyzw": list(args.default_quat_xyzw),
            }
        }
        scene_object = {
            "instance_name": instance,
            "object_class": object_class,
            "pose": entry["pick"],
        }
        if instance in fixtures_by_instance:
            scene_object["is_fixture"] = True
        scene_objects.append(scene_object)
        region_entry = {
            "place_region": {
                "center": [round(value, 6) for value in libero_xy_to_moveit_xyz(regions[region_name], args, args.bddl_receptacle_z)],
                "size": list(args.bddl_place_region_size),
                "quat_xyzw": list(args.default_quat_xyzw),
            }
        }
        if object_class == target_receptacle:
            receptacle_payload[object_class] = region_entry
        elif object_class in target_objects:
            object_payload[object_class] = entry

    if target_receptacle not in receptacle_payload:
        raise KeyError(f"BDDL did not provide init region for receptacle '{target_receptacle}' in {row['episode_id']}")

    missing = sorted(target_objects - set(object_payload))
    if missing:
        raise KeyError(f"BDDL did not provide init regions for target object(s) {missing} in {row['episode_id']}")

    return {
        "objects": object_payload,
        "receptacles": receptacle_payload,
        "scene_objects": scene_objects,
    }


def deterministic_region_pose(region: RegionSpec, step_index: int, total_steps: int) -> PoseSpec:
    """Pick a stable place pose inside a container or tray region."""

    if total_steps <= 1:
        offset_x = 0.0
        offset_y = 0.0
    else:
        cols = math.ceil(math.sqrt(total_steps))
        row = step_index // cols
        col = step_index % cols
        rows = math.ceil(total_steps / cols)
        usable_x = region.size[0] * 0.5
        usable_y = region.size[1] * 0.5
        offset_x = ((col + 0.5) / cols - 0.5) * usable_x
        offset_y = ((row + 0.5) / rows - 0.5) * usable_y

    return PoseSpec(
        xyz=(region.center[0] + offset_x, region.center[1] + offset_y, region.center[2]),
        quat_xyzw=region.quat_xyzw,
    )


def episode_objects(row: dict[str, Any], aliases: dict[str, str]) -> list[str]:
    metadata = row.get("metadata", {})
    if row.get("target_sequence"):
        raw_objects = row["target_sequence"]
    elif metadata.get("target_sequence"):
        raw_objects = metadata["target_sequence"]
    elif metadata.get("graspable_objects") and metadata.get("suite") == "sequences":
        raw_objects = metadata["graspable_objects"]
    else:
        raw_objects = [row["target_object"]]

    return unique_preserve_order(canonical_object(str(item), aliases) for item in raw_objects)


def sequence_matters(row: dict[str, Any]) -> bool:
    if row.get("suite") == "sequences":
        return True
    metadata = row.get("metadata", {})
    return bool(metadata.get("target_sequence") or metadata.get("order") or metadata.get("rule") in {"left_to_right", "right_to_left"})


def success_mode_for_receptacle(receptacle_name: str) -> str:
    if receptacle_name in {"basket", "wooden_tray"}:
        return "inside_container"
    return "on_receptacle"


def make_episode_job(row: dict[str, Any], pose_db: dict[str, Any], aliases: dict[str, str]) -> EpisodeJob:
    target_receptacle = canonical_object(row.get("target_receptacle") or "basket", aliases)
    _ = target_receptacle
    scene_objects = tuple(
        SceneObject(
            instance_name=str(item["instance_name"]),
            object_class=canonical_object(str(item["object_class"]), aliases),
            pose=pose_from_payload(item["pose"], f"scene_objects.{item.get('instance_name', 'unknown')}"),
            is_fixture=bool(item.get("is_fixture", False)),
        )
        for item in pose_db.get("scene_objects", [])
    )
    objects = episode_objects(row, aliases)
    region = receptacle_region(pose_db, target_receptacle)
    total_steps = len(objects)
    success_mode = success_mode_for_receptacle(target_receptacle)

    steps: list[PickPlaceStep] = []
    for index, object_name in enumerate(objects):
        steps.append(
            PickPlaceStep(
                object_name=object_name,
                receptacle_name=target_receptacle,
                pick_pose=object_pick_pose(pose_db, object_name),
                place_pose=deterministic_region_pose(region, index, total_steps),
                place_region=region,
                success_mode=success_mode,
                sequence_index=index,
            )
        )

    return EpisodeJob(
        episode_id=row["episode_id"],
        episode_type=row["episode_type"],
        user_id=row["user_id"],
        instruction=row["instruction"],
        sequence_matters=sequence_matters(row),
        success_mode=success_mode,
        steps=tuple(steps),
        scene_objects=scene_objects,
        metadata=dict(row.get("metadata", {})),
    )


def job_to_plan(job: EpisodeJob) -> dict[str, Any]:
    payload = asdict(job)
    for step in payload["steps"]:
        step["gripper_actions"] = [
            {
                "after_segment": "grasp",
                "command": "close",
                "reason": "end effector is at the object grasp pose",
            },
            {
                "after_segment": "place",
                "command": "open",
                "reason": "end effector is at the target receptacle/place pose",
            },
        ]
    payload["notes"] = [
        f"sequence_matters is {str(job.sequence_matters).lower()}; replay steps in sequence_index order.",
        "place_pose is sampled deterministically inside place_region.",
        "The gripper closes after reaching the object grasp pose and opens after reaching the target place pose.",
    ]
    return payload


def write_dry_run_plan(job: EpisodeJob, output_dir: Path) -> Path:
    path = output_dir / job.episode_id / "plan.json"
    write_json(path, job_to_plan(job))
    return path


def pose_to_payload(pose: PoseSpec) -> dict[str, Any]:
    return {
        "position": {
            "x": pose.xyz[0],
            "y": pose.xyz[1],
            "z": pose.xyz[2],
        },
        "orientation": {
            "x": pose.quat_xyzw[0],
            "y": pose.quat_xyzw[1],
            "z": pose.quat_xyzw[2],
            "w": pose.quat_xyzw[3],
        },
    }


def region_to_payload(region: RegionSpec) -> dict[str, Any]:
    return {
        "center": {
            "x": region.center[0],
            "y": region.center[1],
            "z": region.center[2],
        },
        "size": {
            "x": region.size[0],
            "y": region.size[1],
            "z": region.size[2],
        },
        "orientation": {
            "x": region.quat_xyzw[0],
            "y": region.quat_xyzw[1],
            "z": region.quat_xyzw[2],
            "w": region.quat_xyzw[3],
        },
    }


def job_to_ros2_mtc_spec(job: EpisodeJob, args: argparse.Namespace) -> dict[str, Any]:
    """Build a declarative MoveIt 2 MTC task spec for a separate ROS2 C++ runner."""

    return {
        "schema": "vlapb.moveit2_mtc_task.v1",
        "episode": {
            "episode_id": job.episode_id,
            "episode_type": job.episode_type,
            "user_id": job.user_id,
            "instruction": job.instruction,
            "sequence_matters": job.sequence_matters,
            "success_mode": job.success_mode,
            "metadata": job.metadata,
        },
        "moveit2": {
            "node_expected_api": "MoveIt Task Constructor C++",
            "reference_frame": args.reference_frame,
            "arm_group_name": args.group,
            "hand_group_name": args.gripper_group,
            "hand_frame": args.hand_frame,
            "ik_frame": args.hand_frame,
            "open_gripper_goal": args.open_gripper_goal,
            "closed_gripper_goal": args.closed_gripper_goal,
            "planning_attempts": args.planning_attempts,
            "planning_time": args.planning_time,
        },
        "scene_objects": [
            {
                "instance_name": item.instance_name,
                "object_class": item.object_class,
                "pose": pose_to_payload(item.pose),
                "is_fixture": item.is_fixture,
            }
            for item in job.scene_objects
        ],
        "recommended_stage_order": [
            "CurrentState",
            "MoveTo(open hand)",
            "Connect(move to pick)",
            "SerialContainer(pick object)",
            "GenerateGraspPose",
            "ComputeIK(grasp pose IK)",
            "ModifyPlanningScene(allow hand/object collision)",
            "MoveTo(close hand)",
            "ModifyPlanningScene(attach object)",
            "MoveRelative(lift object)",
            "Connect(move to place)",
            "SerialContainer(place object)",
            "GeneratePlacePose",
            "ComputeIK(place pose IK)",
            "MoveTo(open hand)",
            "ModifyPlanningScene(detach object)",
            "MoveRelative(retreat)",
        ],
        "steps": [
            {
                "sequence_index": step.sequence_index,
                "object_id": step.object_name,
                "object_name": step.object_name,
                "receptacle_name": step.receptacle_name,
                "pick_pose": pose_to_payload(step.pick_pose),
                "place_pose": pose_to_payload(step.place_pose),
                "place_region": region_to_payload(step.place_region) if step.place_region else None,
                "approach": {
                    "frame_id": args.reference_frame,
                    "direction": {"x": 0.0, "y": 0.0, "z": -1.0},
                    "min_distance": args.approach_min_distance,
                    "max_distance": args.approach_height,
                },
                "lift": {
                    "frame_id": args.reference_frame,
                    "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "min_distance": args.lift_min_distance,
                    "max_distance": args.lift_height,
                },
                "retreat": {
                    "frame_id": args.reference_frame,
                    "direction": {"x": -0.5, "y": 0.0, "z": 0.0},
                    "min_distance": args.retreat_min_distance,
                    "max_distance": args.retreat_max_distance,
                },
                "success_mode": step.success_mode,
            }
            for step in job.steps
        ],
        "notes": [
            "This file is not executed by generate_suite_moveit_data.py.",
            "Use it as input to a ROS2 C++ MoveIt Task Constructor node inside the MoveIt 2 Docker container.",
            "The stage names intentionally match the official MoveIt 2 MTC pick/place tutorial.",
        ],
    }


def write_ros2_mtc_spec(job: EpisodeJob, output_dir: Path, args: argparse.Namespace) -> Path:
    path = output_dir / job.episode_id / "mtc_task.json"
    write_json(path, job_to_ros2_mtc_spec(job, args))
    return path


def make_pose_msg(pose: PoseSpec) -> Any:
    import geometry_msgs.msg

    msg = geometry_msgs.msg.Pose()
    msg.position.x, msg.position.y, msg.position.z = pose.xyz
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = pose.quat_xyzw
    return msg


def trajectory_to_dict(plan: Any) -> dict[str, Any]:
    trajectory = getattr(plan, "joint_trajectory", plan)
    return {
        "joint_names": list(getattr(trajectory, "joint_names", [])),
        "points": [
            {
                "positions": list(point.positions),
                "velocities": list(point.velocities),
                "accelerations": list(point.accelerations),
                "time_from_start": float(point.time_from_start.to_sec()),
            }
            for point in getattr(trajectory, "points", [])
        ],
    }


def moveit_error_code_to_dict(error_code: Any) -> dict[str, Any]:
    if error_code is None:
        return {}
    return {
        "value": getattr(error_code, "val", None),
        "repr": repr(error_code),
    }


class MoveItPlanner:
    def __init__(self, args: argparse.Namespace) -> None:
        import moveit_commander
        import rospy

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("vlapb_suite_moveit_generator", anonymous=True)

        self.moveit_commander = moveit_commander
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(args.group)
        self.group.set_pose_reference_frame(args.reference_frame)
        self.group.set_end_effector_link(args.eef_link)
        self.group.set_planning_time(args.planning_time)
        self.group.set_num_planning_attempts(args.planning_attempts)
        self.gripper_group = None
        if args.gripper_group:
            self.gripper_group = moveit_commander.MoveGroupCommander(args.gripper_group)
        self.approach_height = args.approach_height
        self.lift_height = args.lift_height
        self.execute = args.execute
        self.open_gripper_values = tuple(args.open_gripper_values)
        self.closed_gripper_values = tuple(args.closed_gripper_values)
        self.stop_on_plan_failure = args.stop_on_plan_failure
        self._last_joint_state: tuple[list[str], list[float]] | None = None

    def pose_with_z_offset(self, pose: PoseSpec, offset: float) -> PoseSpec:
        return PoseSpec(
            xyz=(pose.xyz[0], pose.xyz[1], pose.xyz[2] + offset),
            quat_xyzw=pose.quat_xyzw,
        )

    def reset_chain_start(self) -> None:
        self._last_joint_state = None
        self.group.set_start_state_to_current_state()

    def _set_chained_start_state(self) -> None:
        if self.execute or self._last_joint_state is None:
            self.group.set_start_state_to_current_state()
            return

        import moveit_msgs.msg
        import sensor_msgs.msg

        names, positions = self._last_joint_state
        state = moveit_msgs.msg.RobotState()
        state.joint_state = sensor_msgs.msg.JointState()
        state.joint_state.name = names
        state.joint_state.position = positions
        self.group.set_start_state(state)

    def _remember_plan_endpoint(self, payload: dict[str, Any]) -> None:
        if not payload["success"] or not payload["points"]:
            return
        payload["chained_from_previous_segment"] = self._last_joint_state is not None
        self._last_joint_state = (
            list(payload["joint_names"]),
            list(payload["points"][-1]["positions"]),
        )

    def plan_to_pose(self, pose: PoseSpec) -> dict[str, Any]:
        self._set_chained_start_state()
        self.group.set_pose_target(make_pose_msg(pose))
        result = self.group.plan()
        success = True
        planning_time = None
        error_code = None
        if isinstance(result, tuple):
            if len(result) >= 2:
                success = bool(result[0])
                plan = result[1]
            else:
                plan = result[0]
            if len(result) >= 3:
                planning_time = float(result[2])
            if len(result) >= 4:
                error_code = result[3]
        else:
            plan = result
        self.group.clear_pose_targets()
        payload = trajectory_to_dict(plan)
        payload["success"] = bool(success and payload["points"])
        payload["planning_time"] = planning_time
        payload["moveit_error_code"] = moveit_error_code_to_dict(error_code)
        payload["executed"] = False
        if self.stop_on_plan_failure and not payload["success"]:
            raise RuntimeError(f"MoveIt failed to plan to pose xyz={pose.xyz} quat_xyzw={pose.quat_xyzw}")
        self._remember_plan_endpoint(payload)
        if self.execute:
            payload["executed"] = bool(self.group.execute(plan, wait=True))
            self.group.stop()
        return payload

    def command_gripper(self, command: str) -> dict[str, Any]:
        values = self.closed_gripper_values if command == "close" else self.open_gripper_values
        payload = {
            "command": command,
            "joint_values": list(values),
            "executed": False,
        }
        if self.gripper_group is None:
            payload["skipped_reason"] = "no gripper group configured"
            return payload
        if self.execute:
            self.gripper_group.go(list(values), wait=True)
            self.gripper_group.stop()
            payload["executed"] = True
        return payload

    def plan_step(self, step: PickPlaceStep) -> dict[str, Any]:
        pre_grasp = self.plan_to_pose(self.pose_with_z_offset(step.pick_pose, self.approach_height))
        grasp = self.plan_to_pose(step.pick_pose)
        close_gripper = self.command_gripper("close")
        lift = self.plan_to_pose(self.pose_with_z_offset(step.pick_pose, self.lift_height))
        pre_place = self.plan_to_pose(self.pose_with_z_offset(step.place_pose, self.approach_height))
        place = self.plan_to_pose(step.place_pose)
        open_gripper = self.command_gripper("open")
        retreat = self.plan_to_pose(self.pose_with_z_offset(step.place_pose, self.lift_height))

        return {
            "object_name": step.object_name,
            "receptacle_name": step.receptacle_name,
            "segments": {
                "pre_grasp": pre_grasp,
                "grasp": grasp,
                "lift": lift,
                "pre_place": pre_place,
                "place": place,
                "retreat": retreat,
            },
            "gripper_actions": [
                {
                    "after_segment": "grasp",
                    "reason": "end effector is at the object grasp pose",
                    **close_gripper,
                },
                {
                    "after_segment": "place",
                    "reason": "end effector is at the target receptacle/place pose",
                    **open_gripper,
                },
            ],
            "ordered_primitives": [
                "pre_grasp",
                "grasp",
                "close_gripper",
                "lift",
                "pre_place",
                "place",
                "open_gripper",
                "retreat",
            ],
            "success_mode": step.success_mode,
            "place_region": asdict(step.place_region) if step.place_region else None,
        }

    def plan_job(self, job: EpisodeJob, output_dir: Path) -> Path:
        self.reset_chain_start()
        payload = job_to_plan(job)
        payload["moveit_trajectories"] = [self.plan_step(step) for step in job.steps]
        path = output_dir / job.episode_id / "trajectory_plan.json"
        write_json(path, payload)
        return path

    def shutdown(self) -> None:
        self.moveit_commander.roscpp_shutdown()


def build_pose_template(rows: list[dict[str, Any]], aliases: dict[str, str]) -> dict[str, Any]:
    objects: list[str] = []
    receptacles: list[str] = []

    for row in rows:
        for object_name in episode_objects(row, aliases):
            if object_name not in objects:
                objects.append(object_name)
        receptacle = canonical_object(row.get("target_receptacle") or "basket", aliases)
        if receptacle not in receptacles:
            receptacles.append(receptacle)

    object_payload: dict[str, Any] = {}
    for index, object_name in enumerate(sorted(objects)):
        col = index % 5
        row = index // 5
        object_payload[object_name] = {
            "pick": {
                "xyz": [round(0.35 + col * 0.06, 4), round(-0.30 + row * 0.06, 4), 0.05],
                "quat_xyzw": [0.0, 1.0, 0.0, 0.0],
            }
        }

    receptacle_payload: dict[str, Any] = {}
    for index, receptacle_name in enumerate(sorted(receptacles)):
        receptacle_payload[receptacle_name] = {
            "place_region": {
                "center": [round(0.50 + index * 0.08, 4), 0.22, 0.10],
                "size": [0.20, 0.16, 0.08],
                "quat_xyzw": [0.0, 1.0, 0.0, 0.0],
            }
        }

    return {
        "description": (
            "Template pose DB for generate_suite_moveit_data.py. "
            "Coordinates are placeholders; replace them with world-frame values."
        ),
        "objects": object_payload,
        "receptacles": receptacle_payload,
    }


def write_pose_template(path: Path, rows: list[dict[str, Any]], aliases: dict[str, str], suites_root: Path) -> None:
    payload = {
        **build_pose_template(rows, aliases),
        "source_suites_root": str(suites_root),
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites-root", type=Path, default=DEFAULT_SUITES_ROOT)
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--pose-source", choices=("json", "bddl"), default="json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode", dest="episode_id", type=str, default=None)
    parser.add_argument("--suite", choices=("belongings", "placements", "sequences"), default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--all", dest="collect_all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("ros1-moveit-commander", "ros2-mtc-spec"),
        default="ros1-moveit-commander",
        help=(
            "ros1-moveit-commander plans with moveit_commander/rospy. "
            "ros2-mtc-spec writes JSON specs for a MoveIt 2 Task Constructor C++ runner."
        ),
    )
    parser.add_argument("--write-pose-template", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP)
    parser.add_argument("--gripper-group", type=str, default=DEFAULT_GRIPPER_GROUP)
    parser.add_argument("--hand-frame", type=str, default="panda_hand")
    parser.add_argument("--open-gripper-goal", type=str, default="open")
    parser.add_argument("--closed-gripper-goal", type=str, default="close")
    parser.add_argument("--open-gripper-values", nargs="+", type=float, default=[0.04, 0.04])
    parser.add_argument("--closed-gripper-values", nargs="+", type=float, default=[0.0, 0.0])
    parser.add_argument("--default-quat-xyzw", nargs=4, type=float, default=[0.0, 1.0, 0.0, 0.0])
    parser.add_argument("--eef-link", type=str, default=DEFAULT_EEF_LINK)
    parser.add_argument("--reference-frame", type=str, default=DEFAULT_REFERENCE_FRAME)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--planning-attempts", type=int, default=10)
    parser.add_argument("--approach-height", type=float, default=0.10)
    parser.add_argument("--approach-min-distance", type=float, default=0.03)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--lift-min-distance", type=float, default=0.05)
    parser.add_argument("--retreat-min-distance", type=float, default=0.05)
    parser.add_argument("--retreat-max-distance", type=float, default=0.20)
    parser.add_argument("--libero-origin-xy", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--moveit-origin-xyz", nargs=3, type=float, default=[0.50, 0.0, 0.0])
    parser.add_argument("--libero-to-moveit-scale-xy", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument("--libero-to-moveit-yaw-deg", type=float, default=0.0)
    parser.add_argument("--bddl-object-z", type=float, default=0.05)
    parser.add_argument("--bddl-receptacle-z", type=float, default=0.05)
    parser.add_argument("--bddl-place-region-size", nargs=3, type=float, default=[0.20, 0.16, 0.08])
    parser.add_argument("--stop-on-plan-failure", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def output_path_for_job(job: EpisodeJob, output_dir: Path, dry_run: bool) -> Path:
    filename = "plan.json" if dry_run else "trajectory_plan.json"
    return output_dir / job.episode_id / filename


def backend_output_path_for_job(job: EpisodeJob, output_dir: Path, args: argparse.Namespace) -> Path:
    if args.dry_run:
        return output_path_for_job(job, output_dir, dry_run=True)
    if args.backend == "ros2-mtc-spec":
        return output_dir / job.episode_id / "mtc_task.json"
    return output_path_for_job(job, output_dir, dry_run=False)


def build_jobs(
    rows: list[dict[str, Any]],
    pose_db: dict[str, Any] | None,
    aliases: dict[str, str],
    continue_on_error: bool,
    args: argparse.Namespace,
) -> tuple[list[EpisodeJob], list[JobFailure]]:
    jobs: list[EpisodeJob] = []
    failures: list[JobFailure] = []
    for row in rows:
        try:
            row_pose_db = pose_db
            if args.pose_source == "bddl":
                row_pose_db = bddl_pose_db_for_row(row, args, aliases)
            if row_pose_db is None:
                raise ValueError("pose_db is required when pose_source=json")
            jobs.append(make_episode_job(row, row_pose_db, aliases))
        except Exception as exc:
            failure = JobFailure(episode_id=str(row.get("episode_id", "<unknown>")), error=str(exc))
            failures.append(failure)
            if not continue_on_error:
                raise
    return jobs, failures


def write_failure_report(output_dir: Path, failures: list[JobFailure]) -> Path | None:
    if not failures:
        return None
    path = output_dir / "failures.json"
    write_json(path, {"failures": [asdict(failure) for failure in failures]})
    return path


def main() -> None:
    args = parse_args()
    aliases: dict[str, str] = {}
    rows = selected_suite_rows(
        suites_root=args.suites_root,
        output_dir=args.output_dir,
        episode_id=args.episode_id,
        collect_all=args.collect_all,
        suite_name=args.suite,
        split=args.split,
        limit=args.limit,
        offset=args.offset,
    )

    if args.write_pose_template:
        write_pose_template(args.poses, rows, aliases, args.suites_root)
        print(f"wrote pose template: {args.poses}")
        return

    pose_db = None if args.pose_source == "bddl" else load_pose_db(args.poses)
    jobs, failures = build_jobs(rows, pose_db, aliases, args.continue_on_error, args)

    if args.dry_run:
        for job in jobs:
            if args.skip_existing and backend_output_path_for_job(job, args.output_dir, args).exists():
                print(f"[dry-run] {job.episode_id}: skipped existing plan")
                continue
            path = write_dry_run_plan(job, args.output_dir)
            print(f"[dry-run] {job.episode_id}: {len(job.steps)} pick/place step(s) -> {path}")
        failure_path = write_failure_report(args.output_dir, failures)
        if failure_path:
            print(f"[failures] wrote {len(failures)} failure(s): {failure_path}")
        return

    if args.backend == "ros2-mtc-spec":
        for job in jobs:
            if args.skip_existing and backend_output_path_for_job(job, args.output_dir, args).exists():
                print(f"[ros2-mtc-spec] {job.episode_id}: skipped existing task spec")
                continue
            path = write_ros2_mtc_spec(job, args.output_dir, args)
            print(f"[ros2-mtc-spec] {job.episode_id}: {len(job.steps)} MTC step spec(s) -> {path}")
        failure_path = write_failure_report(args.output_dir, failures)
        if failure_path:
            print(f"[failures] wrote {len(failures)} failure(s): {failure_path}")
        return

    if os.environ.get("ROS_VERSION") == "2":
        raise RuntimeError(
            "ROS_VERSION=2 detected, but backend=ros1-moveit-commander uses ROS1 moveit_commander/rospy. "
            "Use --backend ros2-mtc-spec in the MoveIt 2 Docker container, then run a ROS2 C++ "
            "MoveIt Task Constructor node with the generated mtc_task.json files."
        )

    planner = MoveItPlanner(args)
    try:
        for job in jobs:
            if args.skip_existing and output_path_for_job(job, args.output_dir, dry_run=False).exists():
                print(f"[moveit] {job.episode_id}: skipped existing trajectory plan")
                continue
            try:
                path = planner.plan_job(job, args.output_dir)
                print(f"[moveit] {job.episode_id}: {len(job.steps)} pick/place step(s) -> {path}")
            except Exception as exc:
                failures.append(JobFailure(episode_id=job.episode_id, error=str(exc)))
                if not args.continue_on_error:
                    raise
                print(f"[moveit] {job.episode_id}: failed: {exc}")
    finally:
        planner.shutdown()

    failure_path = write_failure_report(args.output_dir, failures)
    if failure_path:
        print(f"[failures] wrote {len(failures)} failure(s): {failure_path}")


if __name__ == "__main__":
    main()
