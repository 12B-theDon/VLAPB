#!/usr/bin/env python3
"""Evaluate OpenPI pi0.5 checkpoints on VLAPB compare-injection pickup tasks."""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
TOOLS_DIR = ROOT / "tools"
EXAMPLES_DIR = ROOT / "examples"
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
OPENPI_ROOT = Path("/home/artemis/Documents/openpi")

for path in (
    EVAL_DIR,
    TOOLS_DIR,
    EXAMPLES_DIR,
    LIBERO_ROOT / "libero",
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages" / "openpi-client" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robot.libero.libero_utils import quat2axisangle  # noqa: E402
from libero.envs import TASK_MAPPING  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi.policies import policy_config as openpi_policy_config  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402
from openvla_eval_utils import (  # noqa: E402
    DATE_TIME,
    DEFAULT_MANIFEST,
    choose_textual_condition,
    choose_visual_condition,
    compose_multimodal_condition,
    load_condition,
    load_json,
    make_condition_from_scene,
    save_rollout_video,
    write_json,
    write_jsonl,
)
from test_env_reconfiguration import build_env_kwargs, get_action_dim, reset_with_retries  # noqa: E402
from vlapb_eval_common import (  # noqa: E402
    DEFAULT_VLAPB_MANIFEST,
    add_vlapb_selection_args,
    filter_completed_conditions,
    is_vlapb_manifest,
    normalize_modes as normalize_vlapb_modes,
    select_vlapb_conditions,
)


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/pi_checkpoints/openpi-assets/checkpoints/pi05_libero")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pi05_results"
ORDERED_MODES = ["plain", "textual", "visual", "visual_textual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenPI pi0.5 on VLAPB injection comparison pickup tasks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_VLAPB_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument("--modes", nargs="+", default=ORDERED_MODES, choices=["plain", "libero_style", "textual", "visual", "visual_textual"])
    parser.add_argument("--limit-trials", type=int, default=4)
    parser.add_argument("--trial-id", action="append", default=None)
    parser.add_argument("--text-granularity", default="sentence", choices=["words", "sentence", "paragraph"])
    parser.add_argument("--text-selectivity", default="ownership_only", choices=["ownership_only", "ownership_sequence", "full_profile"])
    parser.add_argument("--visual-condition", default="obs_plus_visual_panel", choices=["obs_only", "obs_plus_visual_panel"])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--wrist-camera-name", default="robot0_eye_in_hand")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-videos", default="first-per-mode", choices=["none", "first-per-mode", "all", "successes", "failures", "every-n"])
    parser.add_argument("--video-every", type=int, default=8)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    add_vlapb_selection_args(parser)
    return parser.parse_args()


def normalize_modes(modes: list[str]) -> list[str]:
    normalized = ["plain" if mode == "libero_style" else mode for mode in modes]
    return [mode for mode in ORDERED_MODES if mode in normalized]


def selected_trials(manifest: dict, trial_ids: list[str] | None, limit: int | None) -> list[dict]:
    trials = manifest.get("trials", [])
    if trial_ids:
        trials = [trial for trial in trials if trial["trial_id"] in set(trial_ids)]
    return trials[:limit] if limit is not None and limit > 0 else trials


def make_plain_condition(scene_path: Path) -> dict:
    condition = make_condition_from_scene(scene_path, "libero_style")
    condition["condition_id"] = f"{condition['trial_id']}_plain"
    condition["condition_type"] = "plain"
    return condition


def build_conditions(args: argparse.Namespace, manifest: dict) -> list[dict]:
    if is_vlapb_manifest(manifest):
        args.modes = normalize_vlapb_modes(args.modes)
        return select_vlapb_conditions(manifest, args)
    conditions = []
    modes = normalize_modes(args.modes)
    for trial in selected_trials(manifest, args.trial_id, args.limit_trials):
        trial_id = trial["trial_id"]
        textual = visual = None
        if "textual" in modes or "visual_textual" in modes:
            textual = load_condition(choose_textual_condition(manifest, trial_id, args.text_granularity, args.text_selectivity))
        if "visual" in modes or "visual_textual" in modes:
            visual = load_condition(choose_visual_condition(manifest, trial_id, args.visual_condition))
        if "plain" in modes:
            conditions.append(make_plain_condition(Path(trial["scene_config"])))
        if "textual" in modes:
            conditions.append(textual)
        if "visual" in modes:
            conditions.append(visual)
        if "visual_textual" in modes:
            conditions.append(compose_multimodal_condition(textual, visual))
    return conditions


def prompt_for(condition: dict[str, Any]) -> str:
    return str(condition.get("prompt") or condition.get("task_instruction") or "").strip()


def log_line(log_file, message: str) -> None:
    tqdm.write(message)
    log_file.write(message + "\n")
    log_file.flush()


def prompt_preview(prompt: str, max_chars: int = 120) -> str:
    compact = " ".join(prompt.split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."


def condition_task_info(condition: dict) -> str:
    expected = condition.get("expected", {})
    target = expected.get("object") or condition.get("target_object") or "unknown"
    receptacle = expected.get("target_receptacle", "basket")
    return f"mode={condition['condition_type']} trial={condition['trial_id']} target={target} -> {receptacle}"


def build_pi05_env(bddl_file: Path, camera_name: str, wrist_camera_name: str, image_size: int):
    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    if wrist_camera_name and wrist_camera_name != camera_name:
        env_kwargs["camera_names"] = [camera_name, wrist_camera_name]
    return TASK_MAPPING[problem_name](**env_kwargs)


def preprocess_image(obs: dict[str, Any], image_key: str, resize_size: int) -> np.ndarray:
    image = np.ascontiguousarray(obs[image_key][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))


def first_reference_image(condition: dict[str, Any], resize_size: int) -> np.ndarray | None:
    image_paths = condition.get("input_images", {}).get("image_paths") or []
    if not image_paths:
        return None
    path = Path(image_paths[0])
    if not path.exists():
        return None
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))


def concat_obs_ref_h(obs_image: np.ndarray, ref_image: np.ndarray, resize_size: int) -> np.ndarray:
    combined = np.concatenate([obs_image, ref_image], axis=1)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(combined, resize_size, resize_size))


def pi05_observation(condition: dict[str, Any], obs: dict[str, Any], camera_name: str, wrist_camera_name: str, resize_size: int) -> dict[str, Any]:
    base_image = preprocess_image(obs, f"{camera_name}_image", resize_size)
    wrist_key = f"{wrist_camera_name}_image"
    wrist_image = preprocess_image(obs, wrist_key, resize_size) if wrist_key in obs else np.zeros_like(base_image)
    ref_image = first_reference_image(condition, resize_size)
    visual_strategy = condition.get("visual_strategy")
    if ref_image is not None and condition.get("condition_type") in {"visual", "visual_textual"}:
        if visual_strategy == "obs_ref_h":
            base_image = concat_obs_ref_h(base_image, ref_image, resize_size)
        elif visual_strategy == "ref_as_wrist":
            wrist_image = ref_image
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
        "prompt": prompt_for(condition),
    }


def should_record_video(args: argparse.Namespace, condition: dict, index: int, saved_modes: set[str]) -> bool:
    if args.save_videos == "none":
        return False
    if args.save_videos == "all":
        return True
    if args.save_videos == "first-per-mode":
        return condition["condition_type"] not in saved_modes
    if args.save_videos == "every-n":
        return args.video_every > 0 and (index - 1) % args.video_every == 0
    return args.save_videos in {"successes", "failures"}


def keep_video(args: argparse.Namespace, row: dict) -> bool:
    return not ((args.save_videos == "successes" and not row["success"]) or (args.save_videos == "failures" and row["success"]))


def load_pi05_policy(args: argparse.Namespace, log_file) -> Any:
    log_line(log_file, "Loading PI0.5 policy in the current Python environment:")
    log_line(log_file, f"  policy_config={args.policy_config}")
    log_line(log_file, f"  checkpoint={args.checkpoint}")
    if args.pytorch_device:
        log_line(log_file, f"  pytorch_device={args.pytorch_device}")
    train_config = openpi_config.get_config(args.policy_config)
    return openpi_policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device=args.pytorch_device,
    )


def rollout_pi05_condition(condition: dict[str, Any], policy: Any, args: argparse.Namespace, video_path: Path, record_video: bool) -> dict[str, Any]:
    env = build_pi05_env(Path(condition["bddl_file"]), args.camera_name, args.wrist_camera_name, args.image_size)
    replay_images: list[np.ndarray] = []
    action_plan: collections.deque[np.ndarray] = collections.deque()
    done = False
    steps = 0
    error = None
    try:
        obs = reset_with_retries(env, max_attempts=args.max_reset_attempts)
        dummy_action = np.zeros(get_action_dim(env), dtype=np.float32)
        if dummy_action.shape[0] >= 7:
            dummy_action[-1] = -1.0
        for t in range(args.max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(dummy_action.tolist())
                continue
            element = pi05_observation(condition, obs, args.camera_name, args.wrist_camera_name, args.resize_size)
            if record_video:
                replay_images.append(element["observation/image"])
            if not action_plan:
                action_chunk = np.asarray(policy.infer(element)["actions"], dtype=np.float32)
                action_plan.extend(action_chunk[: args.replan_steps])
            obs, _, done, _ = env.step(action_plan.popleft().tolist())
            steps = t - args.num_steps_wait + 1
            if done:
                break
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    saved_video = save_rollout_video(replay_images, video_path, args.video_fps) if record_video else None
    return {
        "condition_id": condition["condition_id"],
        "condition_type": condition["condition_type"],
        "trial_id": condition["trial_id"],
        "user_id": condition.get("user_id"),
        "prompt": prompt_for(condition),
        "bddl_file": condition["bddl_file"],
        "expected": condition.get("expected", {}),
        "visual_strategy": condition.get("visual_strategy"),
        "success": bool(done),
        "steps": int(steps),
        "error": error,
        "recorded_frames": len(replay_images),
        "video_path": saved_video,
    }


def main() -> None:
    args = parse_args()
    args.modes = normalize_modes(args.modes)
    random.seed(args.seed)
    np.random.seed(args.seed)
    conditions = build_conditions(args, load_json(args.manifest))
    run_dir = args.output_dir / f"pi05_compare_pickup_{args.run_id or DATE_TIME}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(run_dir / "run_config.json", metadata)
    write_json(run_dir / "conditions.json", {"conditions": conditions})
    log_file = (run_dir / "run.log").open("w", encoding="utf-8")
    if args.dry_run:
        log_line(log_file, f"Prepared {len(conditions)} conditions under {run_dir}")
        for condition in conditions:
            log_line(log_file, f"DRY-RUN {condition_task_info(condition)}")
            log_line(log_file, f"  prompt={prompt_preview(prompt_for(condition))}")
        log_file.close()
        return

    policy = None
    try:
        policy = load_pi05_policy(args, log_file)
        result_path = run_dir / "results.jsonl"
        conditions, rows = filter_completed_conditions(conditions, result_path, args.resume)
        saved_video_modes: set[str] = set()
        progress = tqdm(conditions, desc="PI0.5 pickup eval", unit="cond")
        for index, condition in enumerate(progress, start=1):
            progress.set_postfix(mode=condition["condition_type"], trial=condition["trial_id"])
            log_line(log_file, f"[{index}/{len(conditions)}] START {condition['condition_id']}")
            log_line(log_file, f"  task={condition_task_info(condition)}")
            log_line(log_file, f"  prompt={prompt_preview(prompt_for(condition))}")
            record_video = should_record_video(args, condition, index, saved_video_modes)
            row = rollout_pi05_condition(condition, policy, args, run_dir / "videos" / f"{index:04d}_{condition['condition_id']}.mp4", record_video)
            if record_video and not keep_video(args, row):
                if row.get("video_path"):
                    Path(row["video_path"]).unlink(missing_ok=True)
                row["video_path"] = None
                row["recorded_frames"] = 0
            if row.get("video_path"):
                saved_video_modes.add(condition["condition_type"])
            rows.append(row)
            write_jsonl(result_path, rows)
            status = "SUCCESS" if row["success"] else "FAIL"
            suffix = f" error={row['error']}" if row.get("error") else ""
            log_line(log_file, f"[{index}/{len(conditions)}] {status} steps={row['steps']}{suffix}")

        total = len(rows)
        successes = sum(1 for row in rows if row["success"])
        by_mode = {}
        for row in rows:
            stats = by_mode.setdefault(row["condition_type"], {"episodes": 0, "successes": 0})
            stats["episodes"] += 1
            stats["successes"] += int(row["success"])
        for stats in by_mode.values():
            stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
        summary = {"episodes": total, "successes": successes, "success_rate": successes / total if total else 0.0, "by_mode": by_mode}
        write_json(run_dir / "summary.json", summary)
        log_line(log_file, f"Wrote results to {run_dir}")
        log_line(log_file, f"Summary: {summary}")
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
        log_file.close()


if __name__ == "__main__":
    main()
