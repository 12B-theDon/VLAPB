"""Shared helpers for collecting VLAPB suite teleop demonstrations."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite as suite
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - only needed for HDF5 teleop export paths
    h5py = None

VLAPB_PROJECT_ROOT = Path("/home/artemis/Documents/VLAPB")
VLAPB_LIBERO_ROOT = VLAPB_PROJECT_ROOT / "libero"
LIBERO_REPO_ROOT = Path("/home/artemis/Documents/LIBERO")
LIBERO_PACKAGE_ROOT = LIBERO_REPO_ROOT / "libero"
DEFAULT_SUITES_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"
DEFAULT_OUTPUT_DIR = VLAPB_LIBERO_ROOT / "teleop_moveit"

TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
for path in (TOOLS_DIR, EXAMPLES_DIR, LIBERO_PACKAGE_ROOT, LIBERO_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import libero.utils.utils as libero_utils  # noqa: E402
    from libero.envs import TASK_MAPPING  # noqa: E402
    import libero.envs.bddl_utils as BDDLUtils  # noqa: E402
except ModuleNotFoundError:
    import libero.libero.utils.utils as libero_utils  # type: ignore[no-redef] # noqa: E402
    from libero.libero.envs import TASK_MAPPING  # type: ignore[no-redef] # noqa: E402
    import libero.libero.envs.bddl_utils as BDDLUtils  # type: ignore[no-redef] # noqa: E402

from test_env_reconfiguration import render_task_image  # noqa: E402


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def suite_bddl_files(suites_root: Path = DEFAULT_SUITES_ROOT) -> list[Path]:
    return sorted(suites_root.glob("*/**/bddl_files/*.bddl"))


def metadata_path_for_bddl(bddl_file: Path) -> Path:
    split_dir = bddl_file.parent.parent
    return split_dir / "metadata" / f"{bddl_file.stem}.json"


def read_metadata_for_bddl(bddl_file: Path) -> dict[str, Any]:
    metadata_path = metadata_path_for_bddl(bddl_file)
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    problem_info = BDDLUtils.get_problem_info(str(bddl_file))
    split_dir = bddl_file.parent.parent
    return {
        "episode_id": bddl_file.stem,
        "suite": split_dir.parent.name,
        "split": split_dir.name,
        "task_type": split_dir.name,
        "language": problem_info.get("language_instruction", ""),
        "bddl_file": str(bddl_file),
    }


def row_from_bddl(bddl_file: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    metadata = read_metadata_for_bddl(bddl_file)
    episode_id = str(metadata.get("episode_id") or bddl_file.stem)
    suite_name = str(metadata.get("suite") or bddl_file.parent.parent.parent.name)
    split = str(metadata.get("split") or bddl_file.parent.parent.name)
    task_type = str(metadata.get("task_type") or split)
    language = str(metadata.get("language") or "")
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
        "instruction": language,
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


def load_raw_demo(raw_demo_path: Path) -> h5py.File:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to load raw teleop HDF5 demos")
    if not raw_demo_path.exists():
        raise FileNotFoundError(raw_demo_path)
    return h5py.File(raw_demo_path, "r")


def create_processed_hdf5(
    *,
    raw_demo_path: Path,
    output_path: Path,
    use_camera_obs: bool = True,
    use_depth: bool = False,
    no_proprio: bool = False,
    cap_index: int = 5,
) -> Path:
    """Convert a raw LIBERO teleop HDF5 into processed LIBERO dataset format."""

    if h5py is None:
        raise ModuleNotFoundError("h5py is required to create processed teleop HDF5 demos")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_file = load_raw_demo(raw_demo_path)

    env_name = raw_file["data"].attrs["env"]
    env_kwargs = json.loads(raw_file["data"].attrs["env_info"])
    problem_info = json.loads(raw_file["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    bddl_file_name = raw_file["data"].attrs["bddl_file_name"]
    if isinstance(bddl_file_name, bytes):
        bddl_file_name = bddl_file_name.decode("utf-8")

    demos = sorted(raw_file["data"].keys())

    processed = h5py.File(output_path, "w")
    grp = processed.create_group("data")
    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = raw_file["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = Path(bddl_file_name).read_text(encoding="utf-8")

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not use_camera_obs,
        has_offscreen_renderer=use_camera_obs,
        ignore_done=True,
        use_camera_obs=use_camera_obs,
        camera_depths=use_depth,
        camera_names=["robot0_eye_in_hand", "agentview"],
        reward_shaping=True,
        control_freq=20,
        camera_heights=256,
        camera_widths=256,
        camera_segmentations=None,
    )
    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": bddl_file_name,
        "env_kwargs": env_kwargs,
    }
    grp.attrs["env_args"] = json.dumps(env_args)

    env = TASK_MAPPING[problem_name](**env_kwargs)
    total_len = 0

    try:
        for demo_index, ep in enumerate(demos):
            model_xml = raw_file[f"data/{ep}"].attrs["model_file"]
            states = raw_file[f"data/{ep}/states"][()]
            actions = np.asarray(raw_file[f"data/{ep}/actions"][()])

            env.reset()
            env.reset_from_xml_string(libero_utils.postprocess_model_xml(model_xml, {}))
            env.sim.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()

            valid_indices: list[int] = []
            robot_states = []
            gripper_states = []
            joint_states = []
            ee_states = []
            agentview_images = []
            eye_in_hand_images = []
            agentview_depths = []
            eye_in_hand_depths = []

            for step_index, action in enumerate(actions):
                obs, _reward, _done, _info = env.step(action)
                if step_index < cap_index:
                    continue
                valid_indices.append(step_index)

                if not no_proprio:
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])
                    joint_states.append(obs["robot0_joint_pos"])
                    ee_states.append(
                        np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
                    )

                robot_states.append(env.get_robot_state_vector(obs))
                if use_camera_obs:
                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                    if use_depth:
                        agentview_depths.append(obs["agentview_depth"])
                        eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])
                else:
                    env.render()

            states_out = states[valid_indices]
            actions_out = actions[valid_indices]
            if len(actions_out) == 0:
                raise ValueError(f"{raw_demo_path} demo {ep} produced no samples after cap_index={cap_index}")

            dones = np.zeros(len(actions_out), dtype=np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions_out), dtype=np.uint8)
            rewards[-1] = 1

            ep_grp = grp.create_group(f"demo_{demo_index}")
            obs_grp = ep_grp.create_group("obs")

            if not no_proprio:
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                ee_array = np.stack(ee_states, axis=0)
                obs_grp.create_dataset("ee_states", data=ee_array)
                obs_grp.create_dataset("ee_pos", data=ee_array[:, :3])
                obs_grp.create_dataset("ee_ori", data=ee_array[:, 3:])

            if use_camera_obs:
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                if use_depth:
                    obs_grp.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
                    obs_grp.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0))

            ep_grp.create_dataset("actions", data=actions_out)
            ep_grp.create_dataset("states", data=states_out)
            ep_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_grp.create_dataset("rewards", data=rewards)
            ep_grp.create_dataset("dones", data=dones)
            ep_grp.attrs["num_samples"] = len(actions_out)
            ep_grp.attrs["model_file"] = model_xml
            ep_grp.attrs["init_state"] = states[0]
            total_len += len(actions_out)

        grp.attrs["num_demos"] = len(demos)
        grp.attrs["total"] = total_len
    finally:
        env.close()
        raw_file.close()
        processed.close()

    return output_path


def xbox_input2action(device, robot, active_arm: str = "right", env_configuration: str | None = None):
    """Convert LinuxXboxController state to a robosuite action without importing keyboard devices."""

    state = device.get_controller_state()
    dpos = state["dpos"]
    raw_drotation = state["raw_drotation"]
    grasp = state["grasp"]
    reset = state["reset"]
    if reset:
        return None, None

    is_bimanual = robot.__class__.__name__ == "Bimanual"
    controller = robot.controller if not is_bimanual else robot.controller[active_arm]
    gripper = robot.gripper if not is_bimanual else robot.gripper[active_arm]
    gripper_dof = gripper.dof

    drotation = raw_drotation[[1, 0, 2]]
    if controller.name == "IK_POSE":
        if robot.robot_model.__class__.__name__ == "Panda":
            drotation = drotation[[1, 0, 2]]
        else:
            drotation[0] = -drotation[0]
        drotation *= 10
        dpos *= 5
        drotation = T.mat2quat(T.euler2mat(drotation))

        if env_configuration == "single-arm-opposed":
            dpos = dpos[[1, 0, 2]]
            drotation[0] = -drotation[0]
            drotation[1] = -drotation[1]
            if active_arm == "left":
                dpos[0] = -dpos[0]
            else:
                dpos[1] = -dpos[1]

        drotation = T.quat2axisangle(drotation)
    elif controller.name == "OSC_POSE":
        drotation[2] = -drotation[2]
        drotation *= 50
        dpos *= 125
    elif controller.name == "OSC_POSITION":
        dpos *= 125
    else:
        raise ValueError("Unsupported controller specified; expected IK_POSE, OSC_POSE, or OSC_POSITION")

    grasp = 1 if grasp else -1
    if controller.name == "OSC_POSITION":
        action = np.concatenate([dpos, [grasp] * gripper_dof])
    else:
        action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])
    return action, grasp


def _body_position(env: Any, name: str) -> np.ndarray | None:
    if name not in getattr(env, "obj_body_id", {}):
        return None
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[name]])


def _site_position(env: Any, name: str) -> np.ndarray | None:
    try:
        return np.asarray(env.sim.data.get_site_xpos(name))
    except Exception:
        return None


def _format_pos(pos: np.ndarray | None) -> str:
    if pos is None:
        return "missing"
    return "(" + ", ".join(f"{value:.4f}" for value in pos[:3]) + ")"


def _debug_name_candidates(env: Any, row: dict[str, Any]) -> dict[str, Any]:
    goal_states = list(getattr(env, "parsed_problem", {}).get("goal_state", []))
    goal_pairs: list[tuple[str, str]] = []
    for goal_state in goal_states:
        if len(goal_state) >= 3:
            goal_pairs.append((str(goal_state[1]), str(goal_state[2])))

    target_name = goal_pairs[0][0] if goal_pairs else None
    goal_name = goal_pairs[0][1] if goal_pairs else None

    fixed_class = row.get("target_receptacle") or row.get("metadata", {}).get("fixture")
    fixed_names = []
    if fixed_class:
        fixed_class = str(fixed_class)
        fixed_names = [
            name
            for name in getattr(env, "object_states_dict", {})
            if name != target_name and (name == fixed_class or name.startswith(f"{fixed_class}_") or fixed_class in name)
        ]

    return {
        "goal_states": goal_states,
        "goal_pairs": goal_pairs,
        "target_name": target_name,
        "goal_name": goal_name,
        "fixed_names": fixed_names,
    }


def _print_teleop_debug(env: Any, row: dict[str, Any], step_count: int, obs: dict[str, Any] | None = None) -> None:
    names = _debug_name_candidates(env, row)
    goal_pairs = names["goal_pairs"]

    eef_pos = None
    if obs is not None and "robot0_eef_pos" in obs:
        eef_pos = np.asarray(obs["robot0_eef_pos"])
    if eef_pos is None:
        eef_pos = _site_position(env, "gripper0_grip_site")

    try:
        success = bool(env._check_success())
    except Exception as exc:  # noqa: BLE001 - debug path should report predicate errors.
        success = f"error:{exc!r}"

    print(f"[debug step={step_count}] success={success} eef={_format_pos(eef_pos)}")
    for idx, (target_name, goal_name) in enumerate(goal_pairs, start=1):
        target_pos = _body_position(env, target_name)
        goal_pos = _site_position(env, goal_name)
        if goal_pos is None:
            goal_pos = _body_position(env, goal_name)
        print(
            f"  goal[{idx}] target:{target_name}={_format_pos(target_pos)} "
            f"goal:{goal_name}={_format_pos(goal_pos)}"
        )
    for fixed_name in names["fixed_names"]:
        fixed_pos = _body_position(env, fixed_name)
        if fixed_pos is None:
            fixed_pos = _site_position(env, fixed_name)
        print(f"  fixed:{fixed_name}={_format_pos(fixed_pos)}")
    if step_count == 1:
        print(f"  goal_states={names['goal_states']}")


def collect_human_trajectory(
    env,
    device,
    arm: str,
    env_configuration: str,
    remove_directories: list[str],
    row: dict[str, Any],
    debug_teleop: bool = False,
    debug_every: int = 20,
) -> bool:
    """Collect one teleop rollout with robosuite's DataCollectionWrapper."""

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    env.render()
    task_completion_hold_count = -1
    device.start_control()
    saving = True
    step_count = 0
    last_obs = None
    min_success_step = 20
    min_target_displacement = 0.03

    # Prevent immediate auto-termination when objects start inside enlarged goal regions.
    goal_states = row.get("goal_states") or []
    target_names: list[str] = []
    for state in goal_states:
        if isinstance(state, (list, tuple)) and len(state) >= 3 and str(state[0]).lower() == "atxy":
            target_names.append(str(state[1]))
    target_names = sorted(set(target_names))

    initial_target_positions: dict[str, np.ndarray] = {}
    for name in target_names:
        pos = _body_position(env, name)
        if pos is not None:
            initial_target_positions[name] = np.asarray(pos, dtype=float)

    while True:
        step_count += 1
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
        action, _grasp = xbox_input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )
        if action is None:
            print("trajectory cancelled")
            saving = False
            break

        last_obs, _reward, _done, _info = env.step(action)
        env.render()
        if debug_teleop and (step_count == 1 or step_count % max(1, debug_every) == 0):
            _print_teleop_debug(env, row, step_count, last_obs)

        if task_completion_hold_count == 0:
            break

        targets_moved = True
        if initial_target_positions:
            targets_moved = False
            for name, init_pos in initial_target_positions.items():
                cur_pos = _body_position(env, name)
                if cur_pos is None:
                    continue
                cur_pos = np.asarray(cur_pos, dtype=float)
                if float(np.linalg.norm(cur_pos - init_pos)) >= min_target_displacement:
                    targets_moved = True
                    break

        success_ready = step_count >= min_success_step and targets_moved
        if env._check_success() and success_ready:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

    print(f"teleop steps: {step_count}")
    if not saving and getattr(env, "ep_directory", None):
        remove_directories.append(env.ep_directory.split("/")[-1])
    env.close()
    if hasattr(device, "close"):
        device.close()
    return saving


def gather_raw_demonstrations_as_hdf5(
    *,
    directory: Path,
    raw_hdf5_path: Path,
    env_info: str,
    problem_info: dict[str, Any],
    bddl_file: Path,
    remove_directories: list[str],
) -> Path:
    """Pack robosuite DataCollectionWrapper npz files into raw LIBERO HDF5."""

    if h5py is None:
        raise ModuleNotFoundError("h5py is required to gather raw teleop HDF5 demos")
    raw_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_hdf5_path, "w") as h5_file:
        grp = h5_file.create_group("data")
        num_eps = 0
        env_name = None

        for ep_directory in sorted(directory.iterdir()):
            if not ep_directory.is_dir() or ep_directory.name in remove_directories:
                continue
            states = []
            actions = []
            for state_file in sorted(glob(str(ep_directory / "state_*.npz"))):
                payload = np.load(state_file, allow_pickle=True)
                env_name = str(payload["env"])
                states.extend(payload["states"])
                for action_info in payload["action_infos"]:
                    actions.append(action_info["actions"])

            if not states:
                continue
            del states[-1]
            if len(states) != len(actions):
                raise ValueError(f"state/action length mismatch in {ep_directory}: {len(states)} vs {len(actions)}")

            ep_grp = grp.create_group(f"demo_{num_eps}")
            model_xml = (ep_directory / "model.xml").read_text(encoding="utf-8")
            ep_grp.attrs["model_file"] = model_xml
            ep_grp.create_dataset("states", data=np.asarray(states))
            ep_grp.create_dataset("actions", data=np.asarray(actions))
            num_eps += 1

        if num_eps == 0:
            raise RuntimeError(f"No saved demonstrations found in {directory}")

        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info
        grp.attrs["problem_info"] = json.dumps(problem_info)
        grp.attrs["bddl_file_name"] = str(bddl_file)
        grp.attrs["bddl_file_content"] = bddl_file.read_text(encoding="utf-8")

    return raw_hdf5_path


def collect_suite_episode(
    row: dict[str, Any],
    args: argparse.Namespace,
    make_device: Any,
    preflight: Any | None = None,
) -> Path:
    bddl_file = Path(row["bddl_file"])
    output_hdf5 = Path(row["output_hdf5"])
    if output_hdf5.exists() and not args.overwrite:
        print(f"[skip] {row['episode_id']} exists: {output_hdf5}")
        return output_hdf5

    if preflight is not None:
        preflight(args)

    print("")
    print(f"[collect] {row['episode_id']} {row['suite']}/{row['split']}")
    print(f"instruction: {row['instruction']}")
    print(f"bddl: {bddl_file}")
    print(f"output: {output_hdf5}")

    controller_config = load_controller_config(default_controller=args.controller)
    config: dict[str, Any] = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    problem_info = BDDLUtils.get_problem_info(str(bddl_file))
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config

    env = TASK_MAPPING[problem_name](
        bddl_file_name=str(bddl_file),
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    env_info = json.dumps(config)

    timestamp = str(time.time()).replace(".", "_")
    tmp_directory = args.raw_dir / "tmp" / f"{row['episode_id']}_{timestamp}"
    env = DataCollectionWrapper(env, str(tmp_directory))

    device = make_device(env, args)

    remove_directories: list[str] = []
    saving = collect_human_trajectory(
        env,
        device,
        args.arm,
        args.config,
        remove_directories,
        row=row,
        debug_teleop=args.debug_teleop,
        debug_every=args.debug_every,
    )
    if not saving:
        raise RuntimeError(f"{row['episode_id']} was cancelled; no processed HDF5 written")

    raw_hdf5 = Path(row["raw_hdf5"])
    gather_raw_demonstrations_as_hdf5(
        directory=tmp_directory,
        raw_hdf5_path=raw_hdf5,
        env_info=env_info,
        problem_info=problem_info,
        bddl_file=bddl_file,
        remove_directories=remove_directories,
    )
    create_processed_hdf5(
        raw_demo_path=raw_hdf5,
        output_path=output_hdf5,
        use_camera_obs=True,
        use_depth=args.use_depth,
        no_proprio=args.no_proprio,
        cap_index=args.cap_index,
    )

    with h5py.File(output_hdf5, "a") as h5_file:
        h5_file["data"].attrs["vlapb_episode"] = json.dumps(row)
        h5_file["data"].attrs["vlapb_raw_demo"] = str(raw_hdf5)
        h5_file["data"].attrs["vlapb_collection_device"] = args.device
        h5_file["data"].attrs["language_instruction"] = language_instruction
        h5_file["data"].attrs["domain_name"] = domain_name
        h5_file["data"].attrs["vlapb_source"] = "VLAPB_suites"

    if args.validate or args.render_video:
        from validate_teleop_moveit_data import default_video_path, print_validation_result, render_video, validate_hdf5_file

        validation = validate_hdf5_file(
            output_hdf5,
            min_steps=args.validation_min_steps,
            require_single_demo=args.require_single_demo,
        )
        print_validation_result(validation)
        if not validation.ok:
            raise RuntimeError(f"Validation failed for {output_hdf5}")
        if args.render_video:
            preview_path = default_video_path(
                output_hdf5,
                args.preview_dir,
                demo_name=args.preview_demo,
                camera_key=args.preview_camera_key,
            )
            render_video(
                output_hdf5,
                preview_path,
                demo_name=args.preview_demo,
                camera_key=args.preview_camera_key,
                fps=args.preview_fps,
                stride=args.preview_stride,
                overwrite=args.overwrite_preview,
            )
            print(f"[video] {output_hdf5} -> {preview_path}")

    print(f"[done] {row['episode_id']} -> {output_hdf5}")
    return output_hdf5


def collect_suite_data(args: argparse.Namespace, make_device: Any, preflight: Any | None = None) -> None:
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
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        collect_suite_episode(row, args, make_device, preflight)


def smoke_suite_bddl(
    *,
    suites_root: Path = DEFAULT_SUITES_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR / "smoke_renders",
    episode_id: str | None = None,
    suite_name: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    camera_name: str = "agentview",
    image_size: int = 256,
    settle_steps: int = 3,
    max_reset_attempts: int = 25,
) -> None:
    rows = selected_suite_rows(
        suites_root=suites_root,
        output_dir=DEFAULT_OUTPUT_DIR,
        episode_id=episode_id,
        collect_all=episode_id is None,
        suite_name=suite_name,
        split=split,
        limit=limit,
        offset=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str, str]] = []

    for row in rows:
        bddl_file = Path(row["bddl_file"])
        output_path = output_dir / row["suite"] / row["split"] / f"{row['episode_id']}.png"
        print(f"[smoke] {row['episode_id']}: {bddl_file}")
        try:
            render_task_image(
                bddl_file=bddl_file,
                output_path=output_path,
                camera_name=camera_name,
                image_size=image_size,
                settle_steps=settle_steps,
                max_reset_attempts=max_reset_attempts,
            )
            print(f"  ok -> {output_path}")
        except Exception as exc:  # noqa: BLE001 - report all BDDL/env failures.
            failures.append((row["episode_id"], str(bddl_file), repr(exc)))
            print(f"  FAILED: {exc!r}")

    print(f"smoke tested: {len(rows)}")
    print(f"passed: {len(rows) - len(failures)}")
    print(f"failed: {len(failures)}")
    if failures:
        print("failures:")
        for episode_id, bddl_file, error in failures:
            print(f"- {episode_id}: {bddl_file}: {error}")
        raise SystemExit(1)
