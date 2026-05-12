#!/usr/bin/env python3
"""Sanity-check OpenVLA on official LIBERO benchmark tasks.

This intentionally does not use VLAPB BDDL files or prompts. It uses the
official LIBERO task suite, default initial states, and task.language.
"""

from __future__ import annotations

import argparse
import json
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


OPENVLA_ROOT = Path("/home/artemis/Documents/openvla")
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
for path in (OPENVLA_ROOT, LIBERO_ROOT / "libero"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/openvla_checkpoints/openvla-7b-finetuned-libero-10")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "openvla"
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class OpenVLAConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = str(DEFAULT_CHECKPOINT)
    task_suite_name: str = "libero_10"
    unnorm_key: str = "libero_10"
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-suite-name", default="libero_10", choices=sorted(TASK_MAX_STEPS))
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--num-tasks", type=int, default=1, help="Use first N tasks when --task-id is omitted. Use 0 for all.")
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--log-actions", type=int, default=20, help="Number of first policy actions to store per rollout.")
    return parser.parse_args()


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


def load_model(cfg: OpenVLAConfig):
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        raise KeyError(f"Missing unnorm_key={cfg.unnorm_key!r}; available={sorted(model.norm_stats)}")
    return model, get_processor(cfg), get_image_resize_size(cfg)


def rollout(
    *,
    cfg: OpenVLAConfig,
    model: Any,
    processor: Any,
    resize_size: int | tuple[int, int],
    env: Any,
    initial_state: np.ndarray,
    task_description: str,
    max_steps: int,
    num_steps_wait: int,
    log_actions: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    obs = env.set_init_state(initial_state)
    frames: list[np.ndarray] = []
    action_debug: list[dict[str, Any]] = []
    done = False
    error = None
    steps = 0

    try:
        for t in range(max_steps + num_steps_wait):
            if t < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                continue

            image = get_libero_image(obs, resize_size)
            frames.append(image)
            observation = {
                "full_image": image,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            raw_action = np.asarray(get_action(cfg, model, observation, task_description, processor=processor))
            action = normalize_gripper_action(raw_action, binarize=True)
            action = invert_gripper_action(action)
            if len(action_debug) < log_actions:
                action_debug.append(
                    {
                        "step": int(t - num_steps_wait + 1),
                        "raw": raw_action.round(5).tolist(),
                        "exec": np.asarray(action).round(5).tolist(),
                        "gripper_raw": float(raw_action[-1]),
                        "gripper_exec": float(action[-1]),
                    }
                )
            obs, _, done, _ = env.step(action.tolist())
            steps = t - num_steps_wait + 1
            if done:
                break
    except Exception as exc:  # keep batch moving
        error = repr(exc)

    return {"success": bool(done), "steps": int(steps), "error": error, "action_debug": action_debug}, frames


def main() -> None:
    args = parse_args()
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both --load-in-8bit and --load-in-4bit.")
    cfg = OpenVLAConfig(
        pretrained_checkpoint=str(args.checkpoint),
        task_suite_name=args.task_suite_name,
        unnorm_key=args.task_suite_name,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
    )
    np.random.seed(args.seed)
    model, processor, resize_size = load_model(cfg)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    run_id = args.run_id or time.strftime("%Y_%m_%d-%H_%M_%S")
    run_dir = args.output_dir / f"openvla_official_libero_{args.task_suite_name}_{run_id}"
    result_path = run_dir / "results.jsonl"
    write_json(run_dir / "run_config.json", jsonable_args(args))
    if result_path.exists():
        result_path.unlink()

    rows: list[dict[str, Any]] = []
    for task_id in tqdm(selected_task_ids(task_suite, args), desc="OpenVLA official LIBERO task"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        try:
            trials = min(args.num_trials_per_task, len(initial_states))
            for episode_idx in tqdm(range(trials), desc=f"task {task_id}", leave=False):
                env.reset()
                row, frames = rollout(
                    cfg=cfg,
                    model=model,
                    processor=processor,
                    resize_size=resize_size,
                    env=env,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    max_steps=TASK_MAX_STEPS[args.task_suite_name],
                    num_steps_wait=args.num_steps_wait,
                    log_actions=args.log_actions,
                )
                row.update(
                    {
                        "model": "openvla",
                        "task_suite_name": args.task_suite_name,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_description": task_description,
                    }
                )
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


if __name__ == "__main__":
    main()
