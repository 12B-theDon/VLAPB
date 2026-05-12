#!/usr/bin/env python3
"""Sanity-check OpenPI pi0.5 on official LIBERO benchmark tasks.

This loads a local OpenPI checkpoint directly. It does not require the
OpenPI websocket server.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

robosuite_macros_private = types.ModuleType("robosuite.macros_private")
robosuite_macros_private.CACHE_NUMBA = False
sys.modules.setdefault("robosuite.macros_private", robosuite_macros_private)

import numpy as np

try:
    import imageio
except ImportError:  # pragma: no cover
    imageio = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


OPENPI_ROOT = Path("/home/artemis/Documents/openpi")
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
for path in (
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages" / "openpi-client" / "src",
    LIBERO_ROOT / "libero",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from libero import benchmark, get_libero_path  # noqa: E402
from libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi.policies import policy_config as openpi_policy_config  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/pi_checkpoints/openpi-assets/checkpoints/pi05_libero")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "pi05"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class RolloutResult:
    success: bool
    steps: int
    error: str | None
    action_debug: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument("--task-suite-name", default="libero_spatial", choices=sorted(TASK_MAX_STEPS))
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--num-tasks", type=int, default=1, help="Use first N tasks when --task-id is omitted. Use 0 for all.")
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--log-actions", type=int, default=20)
    return parser.parse_args()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> str | None:
    if not frames or imageio is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)
    return str(path)


def selected_task_ids(task_suite: Any, args: argparse.Namespace) -> list[int]:
    if args.task_id is not None:
        return args.task_id
    if args.num_tasks and args.num_tasks > 0:
        return list(range(min(args.num_tasks, task_suite.n_tasks)))
    return list(range(task_suite.n_tasks))


def get_libero_env(task: Any, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def preprocess_image(obs: dict[str, Any], key: str, resize_size: int) -> np.ndarray:
    image = np.ascontiguousarray(obs[key][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))


def build_observation(obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    return {
        "observation/image": preprocess_image(obs, "agentview_image", resize_size),
        "observation/wrist_image": preprocess_image(obs, "robot0_eye_in_hand_image", resize_size),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(task_description),
    }


def load_policy(args: argparse.Namespace):
    train_config = openpi_config.get_config(args.policy_config)
    return openpi_policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device=args.pytorch_device,
    )


def rollout(
    *,
    policy: Any,
    env: Any,
    initial_state: np.ndarray,
    task_description: str,
    max_steps: int,
    args: argparse.Namespace,
) -> tuple[RolloutResult, list[np.ndarray]]:
    obs = env.set_init_state(initial_state)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    frames: list[np.ndarray] = []
    action_debug: list[dict[str, Any]] = []
    done = False
    error = None
    steps = 0

    try:
        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                continue

            element = build_observation(obs, task_description, args.resize_size)
            frames.append(element["observation/image"])
            if not action_plan:
                action_chunk = np.asarray(policy.infer(element)["actions"], dtype=np.float32)
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(f"policy returned {len(action_chunk)} actions, need {args.replan_steps}")
                action_plan.extend(action_chunk[: args.replan_steps])

            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            if len(action_debug) < args.log_actions:
                action_debug.append(
                    {
                        "step": int(t - args.num_steps_wait + 1),
                        "exec": action.round(5).tolist(),
                        "gripper_exec": float(action[-1]),
                    }
                )
            obs, _, done, _ = env.step(action.tolist())
            steps = t - args.num_steps_wait + 1
            if done:
                break
    except Exception as exc:
        error = repr(exc)

    return RolloutResult(bool(done), int(steps), error, action_debug), frames


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    policy = load_policy(args)
    try:
        task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
        run_id = args.run_id or time.strftime("%Y_%m_%d-%H_%M_%S")
        run_dir = args.output_dir / f"pi05_official_libero_{args.task_suite_name}_{run_id}"
        result_path = run_dir / "results.jsonl"
        write_json(run_dir / "run_config.json", jsonable_args(args))
        if result_path.exists():
            result_path.unlink()

        rows: list[dict[str, Any]] = []
        for task_id in tqdm(selected_task_ids(task_suite, args), desc="pi0.5 official LIBERO task"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, args.env_img_res, args.seed)
            try:
                trials = min(args.num_trials_per_task, len(initial_states))
                for episode_idx in tqdm(range(trials), desc=f"task {task_id}", leave=False):
                    env.reset()
                    result, frames = rollout(
                        policy=policy,
                        env=env,
                        initial_state=initial_states[episode_idx],
                        task_description=task_description,
                        max_steps=TASK_MAX_STEPS[args.task_suite_name],
                        args=args,
                    )
                    row = {
                        "model": "pi05",
                        "task_suite_name": args.task_suite_name,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_description": task_description,
                        "success": result.success,
                        "steps": result.steps,
                        "error": result.error,
                        "action_debug": result.action_debug,
                    }
                    if args.save_videos:
                        suffix = "success" if row["success"] else "failure"
                        row["video_path"] = save_video(
                            frames,
                            run_dir / "videos" / f"task{task_id:02d}_ep{episode_idx:02d}_{suffix}.mp4",
                            args.video_fps,
                        )
                    rows.append(row)
                    append_jsonl(result_path, row)
            finally:
                env.close()

        total = len(rows)
        successes = sum(int(row["success"]) for row in rows)
        summary = {"episodes": total, "successes": successes, "success_rate": successes / total if total else 0.0}
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote {run_dir}")
    finally:
        if hasattr(policy, "close"):
            policy.close()


if __name__ == "__main__":
    main()
