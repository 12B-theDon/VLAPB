#!/usr/bin/env python3
"""Render MoveIt-generated VLAPB trajectories inside the LIBERO MuJoCo scene.

This validator is intentionally pragmatic:

* If a MoveIt trajectory JSON exists, it replays joint positions in MuJoCo and
  saves an MP4 preview. By default it also visualizes the MTC attach/detach
  phase by moving the picked object's free joint with the gripper site.
* If no trajectory JSON exists yet, it still loads the episode BDDL scene and
  saves a static MP4 preview so the episode/camera/BDDL wiring can be checked.

The script does not claim task success from joint replay alone. It is a visual
sanity check for the path/scene alignment before converting trajectories into
LIBERO action datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


VLAPB_LIBERO_ROOT = Path("/home/artemis/Documents/VLAPB/libero")
LIBERO_REPO_ROOT = Path("/home/artemis/Documents/LIBERO")
LIBERO_PACKAGE_ROOT = LIBERO_REPO_ROOT / "libero"
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"

for path in (TOOLS_DIR, LIBERO_PACKAGE_ROOT, LIBERO_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from libero.envs import TASK_MAPPING  # noqa: E402
    import libero.envs.bddl_utils as BDDLUtils  # noqa: E402
except ModuleNotFoundError:
    from libero.libero.envs import TASK_MAPPING  # type: ignore[no-redef] # noqa: E402
    import libero.libero.envs.bddl_utils as BDDLUtils  # type: ignore[no-redef] # noqa: E402

from test_env_reconfiguration import build_env_kwargs, get_action_dim, reset_with_retries  # noqa: E402


DEFAULT_MOVEIT_DIR = VLAPB_LIBERO_ROOT / "teleop_moveit" / "moveit"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def default_episode_dir(episode_id: str, moveit_dir: Path) -> Path:
    return moveit_dir / episode_id


def default_trajectory_path(episode_id: str, moveit_dir: Path, step_index: int) -> Path:
    return default_episode_dir(episode_id, moveit_dir) / f"step_{step_index}_trajectory.json"


def default_task_path(episode_id: str, moveit_dir: Path) -> Path:
    return default_episode_dir(episode_id, moveit_dir) / "mtc_task.json"


def default_video_path(episode_id: str, moveit_dir: Path, step_index: int) -> Path:
    return default_episode_dir(episode_id, moveit_dir) / f"step_{step_index}_libero_preview.mp4"


def bddl_file_for_episode(episode_id: str, moveit_dir: Path, task_file: Path | None) -> Path:
    task_path = task_file or default_task_path(episode_id, moveit_dir)
    if not task_path.exists():
        raise FileNotFoundError(f"Task spec not found: {task_path}")
    task = read_json(task_path)
    episode = task.get("episode", {})
    metadata = episode.get("metadata", {})
    bddl_file = metadata.get("bddl_file") or episode.get("bddl_file")
    if not bddl_file:
        raise KeyError(f"{task_path} does not contain episode.metadata.bddl_file")
    return Path(bddl_file)


def load_trajectory(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if "points" in payload and "joint_names" in payload:
        return payload
    if "trajectory" in payload:
        return payload["trajectory"]
    if "joint_trajectory" in payload:
        return payload["joint_trajectory"]
    raise ValueError(f"Could not find trajectory points in {path}")


def image_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    key = f"{camera_name}_image"
    if key not in obs:
        raise KeyError(f"{key} not found in observation keys: {list(obs.keys())}")
    image = np.asarray(obs[key])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def render_env_image(env: Any, camera_name: str, image_size: int, flip_vertical: bool) -> np.ndarray:
    image = env.sim.render(width=image_size, height=image_size, camera_name=camera_name)
    if flip_vertical:
        image = image[::-1, :, :]
    return np.asarray(image).copy()


def stretch_frames(frames: list[np.ndarray], fps: int, min_video_seconds: float) -> list[np.ndarray]:
    if not frames or min_video_seconds <= 0:
        return frames
    target_count = max(1, int(round(fps * min_video_seconds)))
    if len(frames) >= target_count:
        return frames
    indices = np.linspace(0, len(frames) - 1, target_count)
    return [frames[int(round(index))] for index in indices]


def env_joint_indexes(env: Any) -> list[int]:
    robot = env.robots[0]
    for name in ("_ref_joint_pos_indexes", "joint_indexes"):
        value = getattr(robot, name, None)
        if value is not None:
            return [int(index) for index in value]
    raise AttributeError("Could not find robot joint qpos indexes on env.robots[0]")


def trajectory_joint_columns(joint_names: list[str], positions_len: int) -> list[int]:
    if positions_len == 7:
        return list(range(7))
    panda_cols = [
        index
        for index, name in enumerate(joint_names)
        if name.startswith("panda_joint") and name.removeprefix("panda_joint").isdigit()
    ]
    if len(panda_cols) >= 7:
        return panda_cols[:7]
    return list(range(min(7, positions_len)))


def set_robot_joints(env: Any, joint_positions: np.ndarray) -> None:
    indexes = env_joint_indexes(env)
    if len(joint_positions) < len(indexes):
        raise ValueError(f"Trajectory has {len(joint_positions)} joints, env expects at least {len(indexes)}")
    env.sim.data.qpos[indexes] = joint_positions[: len(indexes)]
    env.sim.forward()


def site_position(env: Any, site_name: str) -> np.ndarray:
    site_id = env.sim.model.site_name2id(site_name)
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def find_free_joint_qpos_address(env: Any, object_id: str) -> tuple[int, str]:
    normalized = normalize(object_id)
    exact_pattern = re.compile(rf"^{re.escape(normalized)}_\d+_joint\d+$")
    exact_candidates: list[tuple[int, str]] = []
    fallback_candidates: list[tuple[int, str]] = []
    for joint_id in range(env.sim.model.njnt):
        if int(env.sim.model.jnt_type[joint_id]) != 0:
            continue
        name = env.sim.model.joint_id2name(joint_id) or ""
        normalized_name = normalize(name)
        candidate = (int(env.sim.model.jnt_qposadr[joint_id]), name)
        if exact_pattern.fullmatch(normalized_name):
            exact_candidates.append(candidate)
        elif normalized_name.startswith(f"{normalized}_"):
            fallback_candidates.append(candidate)
    candidates = exact_candidates or fallback_candidates
    if not candidates:
        raise KeyError(f"Could not find MuJoCo free joint for object_id={object_id!r}")
    return candidates[0]


def moveit_position_to_mujoco(position: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float64) + np.asarray(offset, dtype=np.float64)


def set_free_joint_pose(env: Any, qpos_address: int, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
    env.sim.data.qpos[qpos_address : qpos_address + 3] = position[:3]
    env.sim.data.qpos[qpos_address + 3 : qpos_address + 7] = quat_wxyz[:4]
    env.sim.forward()


def set_gripper_visual(env: Any, closed: bool) -> None:
    indexes = getattr(env.robots[0], "_ref_gripper_joint_pos_indexes", None)
    if indexes is None:
        return
    indexes = [int(index) for index in indexes]
    if not indexes:
        return
    if closed:
        env.sim.data.qpos[indexes] = 0.0
    else:
        current = np.asarray(env.sim.data.qpos[indexes], dtype=np.float64)
        if np.all(np.abs(current) < 1e-6):
            env.sim.data.qpos[indexes] = np.array([0.04, -0.04])[: len(indexes)]
    env.sim.forward()


def load_step_spec(task_file: Path, step_index: int) -> dict[str, Any] | None:
    if not task_file.exists():
        return None
    task = read_json(task_file)
    steps = list(task.get("steps", []))
    if step_index < 0 or step_index >= len(steps):
        return None
    return dict(steps[step_index])


def pose_position(pose: dict[str, Any]) -> np.ndarray:
    position = pose.get("position", {})
    return np.array([position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)], dtype=np.float64)


def pose_quat_wxyz(pose: dict[str, Any]) -> np.ndarray:
    orientation = pose.get("orientation", {})
    return np.array(
        [
            orientation.get("w", 1.0),
            orientation.get("x", 0.0),
            orientation.get("y", 0.0),
            orientation.get("z", 0.0),
        ],
        dtype=np.float64,
    )


def make_env(bddl_file: Path, camera_name: str, image_size: int, max_reset_attempts: int) -> tuple[Any, dict[str, Any]]:
    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    env_kwargs["ignore_done"] = True
    env = TASK_MAPPING[problem_name](**env_kwargs)
    obs = reset_with_retries(env, max_attempts=max_reset_attempts)
    return env, obs


def render_static_preview(
    *,
    env: Any,
    obs: dict[str, Any],
    camera_name: str,
    image_size: int,
    frames: int,
    settle_steps: int,
    flip_vertical: bool,
) -> list[np.ndarray]:
    action = np.zeros(get_action_dim(env), dtype=np.float32)
    rendered: list[np.ndarray] = []
    for frame_index in range(max(1, frames)):
        if frame_index < settle_steps:
            obs, _reward, _done, _info = env.step(action)
        rendered.append(render_env_image(env, camera_name, image_size, flip_vertical))
    return rendered


def render_trajectory_preview(
    *,
    env: Any,
    obs: dict[str, Any],
    trajectory: dict[str, Any],
    camera_name: str,
    image_size: int,
    stride: int,
    settle_steps: int,
    frames_per_point: int,
    flip_vertical: bool,
    step_spec: dict[str, Any] | None,
    visualize_attachment: bool,
    attach_segment: int,
    release_segment: int,
    gripper_site: str,
    grasp_z_offset: float,
    moveit_to_mujoco_offset: np.ndarray,
    hold_after_release_seconds: float,
    fps: int,
    release_mode: str,
    attach_distance_threshold: float,
) -> list[np.ndarray]:
    action = np.zeros(get_action_dim(env), dtype=np.float32)
    for _ in range(max(0, settle_steps)):
        obs, _reward, _done, _info = env.step(action)

    joint_names = list(trajectory.get("joint_names", []))
    points = list(trajectory.get("points", []))
    if not points:
        raise ValueError("Trajectory contains no points")

    first_positions = np.asarray(points[0].get("positions", []), dtype=np.float64)
    columns = trajectory_joint_columns(joint_names, len(first_positions))
    object_qpos_address: int | None = None
    object_position: np.ndarray | None = None
    object_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    release_position: np.ndarray | None = None
    release_quat = object_quat
    object_released = False
    attached_offset: np.ndarray | None = None
    last_attached_position: np.ndarray | None = None
    warned_attach_distance = False
    reported_stage_distances: set[str] = set()
    if visualize_attachment and step_spec:
        object_id = str(step_spec.get("object_id", ""))
        object_qpos_address, object_joint_name = find_free_joint_qpos_address(env, object_id)
        object_position = np.asarray(env.sim.data.qpos[object_qpos_address : object_qpos_address + 3], dtype=np.float64).copy()
        object_quat = np.asarray(env.sim.data.qpos[object_qpos_address + 3 : object_qpos_address + 7], dtype=np.float64).copy()
        release_position = moveit_position_to_mujoco(
            pose_position(dict(step_spec.get("place_pose", {}))),
            moveit_to_mujoco_offset,
        )
        release_quat = object_quat
        print(f"[attach] target object_id={object_id} mujoco_joint={object_joint_name}")
        set_gripper_visual(env, closed=False)

    rendered: list[np.ndarray] = []
    for point in points[:: max(1, stride)]:
        positions = np.asarray(point.get("positions", []), dtype=np.float64)
        set_robot_joints(env, positions[columns])
        segment_index = int(point.get("segment_index", -1))
        stage_name = str(point.get("stage_name", ""))
        if object_qpos_address is not None:
            if object_position is not None and stage_name in {"pre_grasp", "pick_hover", "close_gripper", "secure_grasp"} and stage_name not in reported_stage_distances:
                gripper_position = site_position(env, gripper_site)
                delta = object_position - gripper_position
                print(
                    f"[distance] {stage_name}: gripper={gripper_position.round(4).tolist()} "
                    f"target={object_position.round(4).tolist()} target_minus_gripper={delta.round(4).tolist()} "
                    f"dist={float(np.linalg.norm(delta)):.4f}"
                )
                reported_stage_distances.add(stage_name)
            if stage_name:
                is_gripper_closing_phase = stage_name in {"close_gripper", "secure_grasp"}
                is_attached_phase = stage_name in {
                    "secure_grasp",
                    "lift object",
                    "move to place",
                    "lift",
                    "pre_place",
                    "place_hover",
                }
                is_release_phase = stage_name in {"release_object", "retreat"}
            else:
                is_gripper_closing_phase = False
                is_attached_phase = attach_segment <= segment_index < release_segment
                is_release_phase = segment_index >= release_segment
            if is_attached_phase:
                set_gripper_visual(env, closed=True)
                gripper_position = site_position(env, gripper_site)
                if attached_offset is None and object_position is not None:
                    distance = float(np.linalg.norm(object_position - gripper_position))
                    if distance <= attach_distance_threshold:
                        attached_offset = object_position - gripper_position
                        print(f"[attach] locked offset distance={distance:.3f} offset={attached_offset.round(4).tolist()}")
                    elif not warned_attach_distance:
                        print(
                            f"[attach-warning] gripper is {distance:.3f}m from target object; "
                            f"not attaching because threshold={attach_distance_threshold:.3f}m"
                        )
                        warned_attach_distance = True
                if attached_offset is not None:
                    attached_position = gripper_position + attached_offset
                    last_attached_position = attached_position.copy()
                    set_free_joint_pose(env, object_qpos_address, attached_position, object_quat)
            elif is_gripper_closing_phase:
                set_gripper_visual(env, closed=True)
            elif is_release_phase:
                if not object_released:
                    if release_mode == "target" and release_position is not None:
                        set_free_joint_pose(env, object_qpos_address, release_position, release_quat)
                    elif last_attached_position is not None:
                        set_free_joint_pose(env, object_qpos_address, last_attached_position, object_quat)
                    object_released = True
                set_gripper_visual(env, closed=False)
        frame = render_env_image(env, camera_name, image_size, flip_vertical)
        rendered.extend([frame] * max(1, frames_per_point))

    if rendered and hold_after_release_seconds > 0:
        rendered.extend([rendered[-1]] * int(round(max(0.0, hold_after_release_seconds) * fps)))
    return rendered


def write_video(path: Path, frames: list[np.ndarray], fps: int, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        print(f"[skip] video exists: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to write MP4 previews: pip install imageio imageio-ffmpeg") from exc

    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path


def validate_one(args: argparse.Namespace, episode_id: str, step_index: int | None = None) -> Path:
    step_index = args.step_index if step_index is None else step_index
    trajectory_path = args.trajectory or default_trajectory_path(episode_id, args.moveit_dir, step_index)
    task_path = args.task_file or default_task_path(episode_id, args.moveit_dir)
    bddl_file = bddl_file_for_episode(episode_id, args.moveit_dir, task_path)
    video_path = args.output or default_video_path(episode_id, args.moveit_dir, step_index)

    trajectory = load_trajectory(trajectory_path)
    step_spec = load_step_spec(task_path, step_index)
    env, obs = make_env(bddl_file, args.camera, args.image_size, args.max_reset_attempts)
    try:
        if trajectory is None:
            print(f"[preview] no trajectory found at {trajectory_path}; rendering static scene preview")
            frames = render_static_preview(
                env=env,
                obs=obs,
                camera_name=args.camera,
                image_size=args.image_size,
                frames=args.static_frames,
                settle_steps=args.settle_steps,
                flip_vertical=args.flip_vertical,
            )
        else:
            print(f"[preview] replaying trajectory: {trajectory_path}")
            frames = render_trajectory_preview(
                env=env,
                obs=obs,
                trajectory=trajectory,
                camera_name=args.camera,
                image_size=args.image_size,
                stride=args.stride,
                settle_steps=args.settle_steps,
                frames_per_point=args.frames_per_point,
                flip_vertical=args.flip_vertical,
                step_spec=step_spec,
                visualize_attachment=args.visualize_attachment,
                attach_segment=args.attach_segment,
                release_segment=args.release_segment,
                gripper_site=args.gripper_site,
                grasp_z_offset=args.grasp_z_offset,
                moveit_to_mujoco_offset=np.asarray(args.moveit_to_mujoco_offset, dtype=np.float64),
                hold_after_release_seconds=args.hold_after_release_seconds,
                fps=args.fps,
                release_mode=args.release_mode,
                attach_distance_threshold=args.attach_distance_threshold,
            )
    finally:
        env.close()

    frames = stretch_frames(frames, args.fps, args.min_video_seconds)
    output = write_video(video_path, frames, args.fps, args.overwrite)
    print(f"[video] {episode_id} step {step_index} -> {output}")
    return output


def selected_episodes(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(path.name for path in args.moveit_dir.iterdir() if (path / "mtc_task.json").exists())
    if args.episode:
        return [args.episode]
    if args.trajectory:
        parent = args.trajectory.parent
        if parent.name.startswith("ep_"):
            return [parent.name]
    raise ValueError("Provide --episode EPISODE_ID, --trajectory PATH, or --all")


def step_indices_for_episode(args: argparse.Namespace, episode_id: str) -> list[int]:
    if not args.all_steps:
        return [args.step_index]

    task_path = args.task_file or default_task_path(episode_id, args.moveit_dir)
    if task_path.exists():
        steps = read_json(task_path).get("steps", [])
        if steps:
            return list(range(len(steps)))

    episode_dir = default_episode_dir(episode_id, args.moveit_dir)
    discovered = sorted(
        int(path.stem.removeprefix("step_").removesuffix("_trajectory"))
        for path in episode_dir.glob("step_*_trajectory.json")
        if path.stem.removeprefix("step_").removesuffix("_trajectory").isdigit()
    )
    return discovered or [args.step_index]


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.trajectory and args.all_steps:
        raise ValueError("--trajectory points to one file; do not combine it with --all-steps")
    if args.output and (args.all or args.all_steps):
        raise ValueError("--output points to one file; do not combine it with --all or --all-steps")
    jobs: list[tuple[str, int]] = []
    for episode_id in selected_episodes(args):
        for step_index in step_indices_for_episode(args, episode_id):
            jobs.append((episode_id, step_index))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--moveit-dir", type=Path, default=DEFAULT_MOVEIT_DIR)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--task-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames-per-point", type=int, default=4)
    parser.add_argument("--min-video-seconds", type=float, default=10.0)
    parser.add_argument("--hold-after-release-seconds", type=float, default=2.0)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--static-frames", type=int, default=160)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument("--flip-vertical", dest="flip_vertical", action="store_true", default=True)
    parser.add_argument("--no-flip-vertical", dest="flip_vertical", action="store_false")
    parser.add_argument("--visualize-attachment", dest="visualize_attachment", action="store_true", default=True)
    parser.add_argument("--no-visualize-attachment", dest="visualize_attachment", action="store_false")
    parser.add_argument("--attach-segment", type=int, default=4)
    parser.add_argument("--release-segment", type=int, default=7)
    parser.add_argument("--gripper-site", type=str, default="gripper0_grip_site")
    parser.add_argument("--grasp-z-offset", type=float, default=-0.11, help="Deprecated visual offset; preserved for CLI compatibility.")
    parser.add_argument("--attach-distance-threshold", type=float, default=0.065)
    parser.add_argument(
        "--release-mode",
        choices=("gripper", "target"),
        default="gripper",
        help="gripper leaves the object where the hand actually releases it; target snaps it to place_pose.",
    )
    parser.add_argument(
        "--moveit-to-mujoco-offset",
        type=float,
        nargs=3,
        default=[-0.5, 0.0, 0.92],
        metavar=("X", "Y", "Z"),
        help="XYZ offset added to MoveIt world poses when visualizing object release in MuJoCo.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render-video", action="store_true", help="Accepted for CLI symmetry; MP4 is always written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[tuple[str, str]] = []
    jobs = selected_jobs(args)
    iterator = tqdm(jobs, total=len(jobs), unit="step", dynamic_ncols=True) if tqdm is not None else jobs
    for episode_id, step_index in iterator:
        label = f"{episode_id} step {step_index}"
        if tqdm is not None:
            iterator.set_description(label)
        else:
            print(f"[render] {label}")
        try:
            validate_one(args, episode_id, step_index)
        except Exception as exc:  # noqa: BLE001 - keep batch rendering going.
            label = f"{episode_id}/step_{step_index}"
            failures.append((label, repr(exc)))
            print(f"[failed] {label}: {exc!r}")
            if not args.all and not args.all_steps:
                raise

    if failures:
        print("failures:")
        for episode_id, error in failures:
            print(f"- {episode_id}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
