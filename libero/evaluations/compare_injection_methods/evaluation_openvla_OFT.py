#!/usr/bin/env python3
"""Evaluate OpenVLA-OFT on VLAPB compare-injection pickup conditions.

OpenVLA-OFT supports multiple images in one policy query, so visual injection
can be passed as extra images instead of falling back to a concatenated panel.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
OPENVLA_OFT_ROOT = Path("/home/artemis/Documents/openvla-oft")

for path in (LIBERO_ROOT / "libero", TOOLS_DIR, EXAMPLES_DIR, OPENVLA_OFT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robot.libero.libero_utils import quat2axisangle  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from libero.envs import TASK_MAPPING  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402
from test_env_reconfiguration import (  # noqa: E402
    build_env_kwargs,
    get_action_dim,
    reset_with_retries,
)
from vlapb_eval_common import (  # noqa: E402
    DEFAULT_VLAPB_MANIFEST,
    add_vlapb_selection_args,
    filter_completed_conditions,
    is_vlapb_manifest,
    normalize_modes as normalize_vlapb_modes,
    select_vlapb_conditions,
)


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/openvlaOFT_checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
DEFAULT_EVAL_DIR = VLAPB_LIBERO_ROOT / "evaluations" / "compare_injection_methods"
DEFAULT_MANIFEST = DEFAULT_VLAPB_MANIFEST
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "openvlaOFT_results"
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
ORDERED_MODES = ["plain", "textual", "visual", "visual_textual"]


@dataclass
class OpenVLAOFTEvalConfig:
    pretrained_checkpoint: str
    model_family: str = "openvla"
    task_suite_name: str = "libero_object"
    unnorm_key: str = "libero_object"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVLA-OFT on VLAPB injection comparison pickup tasks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ORDERED_MODES,
        choices=["plain", "libero_style", "textual", "visual", "visual_textual", "both"],
        help="Injection modes. visual/visual_textual use OFT multi-image inputs.",
    )
    parser.add_argument(
        "--visual-source",
        default="all_views",
        choices=["panel", "target_views", "all_views"],
        help="Extra visual inputs for visual/both modes. Default: all object views as separate images.",
    )
    parser.add_argument("--limit-trials", type=int, default=4)
    parser.add_argument("--trial-id", action="append", default=None)
    parser.add_argument("--text-granularity", default="sentence", choices=["words", "sentence", "paragraph"])
    parser.add_argument(
        "--text-selectivity",
        default="ownership_only",
        choices=["ownership_only", "ownership_sequence", "full_profile"],
    )
    parser.add_argument("--visual-condition", default="obs_plus_visual_panel", choices=["obs_only", "obs_plus_visual_panel"])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--wrist-camera-name", default="robot0_eye_in_hand")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--unnorm-key", default="libero_object")
    parser.add_argument("--task-suite-name", default="libero_object")
    parser.add_argument("--num-open-loop-steps", type=int, default=NUM_ACTIONS_CHUNK)
    parser.add_argument(
        "--save-videos",
        default="all",
        choices=["none", "first-per-mode", "all", "successes", "failures", "every-n"],
    )
    parser.add_argument("--video-every", type=int, default=8)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Build condition list but do not load OpenVLA-OFT.")
    add_vlapb_selection_args(parser)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def save_rollout_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> str | None:
    if not frames:
        return None
    try:
        import imageio
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return str(path)


def selected_trials(manifest: dict[str, Any], trial_ids: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    trials = manifest.get("trials", [])
    if trial_ids:
        wanted = set(trial_ids)
        trials = [trial for trial in trials if trial["trial_id"] in wanted]
    if limit is not None and limit > 0:
        trials = trials[:limit]
    return trials


def choose_textual_condition(manifest: dict[str, Any], trial_id: str, granularity: str, selectivity: str) -> Path:
    suffix = f"_text_{granularity}_{selectivity}"
    for item in manifest.get("conditions", []):
        if item.get("trial_id") == trial_id and item.get("condition_id", "").endswith(suffix):
            return Path(item["path"])
    raise FileNotFoundError(f"No textual condition for trial={trial_id}, {granularity=}, {selectivity=}")


def choose_visual_condition(manifest: dict[str, Any], trial_id: str, visual_condition: str) -> Path:
    suffix = f"_visual_{visual_condition}"
    for item in manifest.get("conditions", []):
        if item.get("trial_id") == trial_id and item.get("condition_id", "").endswith(suffix):
            return Path(item["path"])
    raise FileNotFoundError(f"No visual condition for trial={trial_id}, {visual_condition=}")


def load_condition(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    scene_path = Path(payload["scene_config"])
    scene = load_json(scene_path)
    payload["scene"] = scene
    payload["bddl_file"] = scene["bddl_file"]
    return payload


def normalize_modes(modes: list[str]) -> list[str]:
    aliases = {
        "libero_style": "plain",
        "both": "visual_textual",
    }
    normalized = [aliases.get(mode, mode) for mode in modes]
    return [mode for mode in ORDERED_MODES if mode in normalized]


def make_plain_condition_from_scene(scene_path: Path) -> dict[str, Any]:
    scene = load_json(scene_path)
    target_object = str(scene["target_object"]).replace("_", " ")
    task = f"Pick up the {target_object} and put it in the basket."
    return {
        "condition_id": f"{scene['trial_id']}_plain",
        "condition_type": "plain",
        "trial_id": scene["trial_id"],
        "user_id": scene.get("user_id"),
        "task_instruction": task,
        "prompt": task,
        "expected": scene.get("expected", {}),
        "scene_config": str(scene_path),
        "scene": scene,
        "bddl_file": scene["bddl_file"],
        "target_object": scene.get("target_object"),
    }


def condition_prompt(condition: dict[str, Any]) -> str:
    prompt = str(condition.get("prompt") or "").strip()
    task_instruction = str(condition.get("task_instruction") or "").strip()
    return prompt or task_instruction


def compose_visual_textual_condition(textual: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    merged = dict(visual)
    merged["condition_id"] = f"{visual['trial_id']}_visual_textual"
    merged["condition_type"] = "visual_textual"
    merged["task_instruction"] = textual["task_instruction"]
    merged["personalization_injection"] = textual.get("personalization_injection", "")
    merged["prompt"] = condition_prompt(textual)
    merged["textual_condition_id"] = textual["condition_id"]
    merged["visual_condition_id"] = visual["condition_id"]
    return merged


def build_conditions(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if is_vlapb_manifest(manifest):
        args.modes = normalize_vlapb_modes(args.modes)
        return select_vlapb_conditions(manifest, args)
    conditions = []
    modes = normalize_modes(args.modes)
    for trial in selected_trials(manifest, args.trial_id, args.limit_trials):
        trial_id = trial["trial_id"]
        scene_path = Path(trial["scene_config"])
        textual = None
        visual = None
        if "textual" in modes or "visual_textual" in modes:
            textual = load_condition(
                choose_textual_condition(
                    manifest,
                    trial_id,
                    granularity=args.text_granularity,
                    selectivity=args.text_selectivity,
                )
            )
        if "visual" in modes or "visual_textual" in modes:
            visual = load_condition(choose_visual_condition(manifest, trial_id, args.visual_condition))

        if "plain" in modes:
            conditions.append(make_plain_condition_from_scene(scene_path))
        if "textual" in modes:
            assert textual is not None
            condition = dict(textual)
            condition["condition_type"] = "textual"
            conditions.append(condition)
        if "visual" in modes:
            assert visual is not None
            conditions.append(visual)
        if "visual_textual" in modes:
            assert textual is not None and visual is not None
            conditions.append(compose_visual_textual_condition(textual, visual))
    return conditions


def visual_reference_paths(condition: dict[str, Any], visual_source: str) -> list[Path]:
    if condition.get("condition_type") not in {"visual", "visual_textual"}:
        return []
    if condition.get("visual_strategy") == "obs_only":
        return []
    image_paths = condition.get("input_images", {}).get("image_paths") or []
    if image_paths:
        return [Path(path) for path in image_paths]
    if visual_source == "panel":
        panel = condition.get("input_images", {}).get("reference_panel")
        return [Path(panel)] if panel else []

    target_object = condition.get("target_object") or condition.get("expected", {}).get("object")
    paths: list[Path] = []
    for item in condition.get("reference_items", []):
        if visual_source == "target_views" and item.get("object") != target_object:
            continue
        paths.extend(Path(path) for path in item.get("views", []))
    return paths


def load_reference_image(path: Path, resize_size: int | tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    return resize_image_for_policy(array, resize_size)


def extract_env_image(obs: dict[str, Any], camera_name: str, resize_size: int | tuple[int, int]) -> np.ndarray:
    image_key = f"{camera_name}_image"
    if image_key not in obs:
        raise KeyError(f"{image_key} not found in observation keys: {list(obs.keys())}")
    image = obs[image_key][::-1, ::-1]
    return resize_image_for_policy(image, resize_size)


def extract_video_frame(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    image_key = f"{camera_name}_image"
    if image_key not in obs:
        raise KeyError(f"{image_key} not found in observation keys: {list(obs.keys())}")
    return obs[image_key][::-1, ::-1]


def build_policy_observation(
    obs: dict[str, Any],
    camera_name: str,
    wrist_camera_name: str,
    resize_size: int | tuple[int, int],
    reference_images: list[np.ndarray],
) -> dict[str, np.ndarray]:
    observation = {
        "full_image": extract_env_image(obs, camera_name, resize_size),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    wrist_key = f"{wrist_camera_name}_image"
    if wrist_key in obs:
        observation["wrist_image"] = extract_env_image(obs, wrist_camera_name, resize_size)
    for index, image in enumerate(reference_images):
        observation[f"wrist_image_ref_{index:02d}"] = image
    return observation


def build_custom_bddl_env(bddl_file: Path, camera_name: str, wrist_camera_name: str, image_size: int):
    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    if wrist_camera_name and wrist_camera_name != camera_name:
        env_kwargs["camera_names"] = [camera_name, wrist_camera_name]
    return TASK_MAPPING[problem_name](**env_kwargs)


def check_unnorm_key(cfg: OpenVLAOFTEvalConfig, model: Any) -> None:
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        available = ", ".join(sorted(model.norm_stats))
        raise KeyError(f"OpenVLA-OFT unnorm_key={cfg.unnorm_key!r} not found. Available: {available}")


def load_openvla_oft(cfg: OpenVLAOFTEvalConfig):
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    check_unnorm_key(cfg, model)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression or cfg.use_diffusion else None
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=PROPRIO_DIM) if cfg.use_proprio else None
    resize_size = get_image_resize_size(cfg)
    return model, processor, action_head, proprio_projector, resize_size


def set_num_images_for_query(model: Any, cfg: OpenVLAOFTEvalConfig, num_images: int) -> None:
    cfg.num_images_in_input = max(1, num_images)
    if hasattr(model, "vision_backbone") and hasattr(model.vision_backbone, "set_num_images_in_input"):
        model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)


def rollout_openvla_oft_condition(
    condition: dict[str, Any],
    model: Any,
    processor: Any,
    action_head: Any,
    proprio_projector: Any,
    cfg: OpenVLAOFTEvalConfig,
    resize_size: int | tuple[int, int],
    *,
    visual_source: str,
    camera_name: str = "agentview",
    wrist_camera_name: str = "robot0_eye_in_hand",
    image_size: int = 256,
    num_steps_wait: int = 10,
    max_steps: int = 520,
    max_reset_attempts: int = 25,
    record_video: bool = False,
    video_path: Path | None = None,
    video_fps: int = 20,
) -> dict[str, Any]:
    env = build_custom_bddl_env(Path(condition["bddl_file"]), camera_name, wrist_camera_name, image_size)
    replay_images: list[np.ndarray] = []
    prompt = condition_prompt(condition)
    reference_paths = visual_reference_paths(condition, visual_source)
    reference_images = [load_reference_image(path, resize_size) for path in reference_paths if path.exists()]
    num_policy_images = 2 + len(reference_images) if cfg.num_images_in_input >= 2 else 1 + len(reference_images)
    set_num_images_for_query(model, cfg, num_policy_images)
    action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
    done = False
    steps = 0
    error = None
    try:
        obs = reset_with_retries(env, max_attempts=max_reset_attempts)
        dummy_action = np.zeros(get_action_dim(env), dtype=np.float32)
        if dummy_action.shape[0] >= 7:
            dummy_action[-1] = -1.0

        for t in range(max_steps + num_steps_wait):
            if t < num_steps_wait:
                obs, _, done, _ = env.step(dummy_action)
                if record_video:
                    replay_images.append(extract_video_frame(obs, camera_name))
                continue

            policy_obs = build_policy_observation(obs, camera_name, wrist_camera_name, resize_size, reference_images)
            if record_video:
                replay_images.append(extract_video_frame(obs, camera_name))

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    policy_obs,
                    prompt,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())
            steps = t - num_steps_wait + 1
            if done:
                break
    except Exception as exc:  # keep batch evaluation moving.
        error = repr(exc)
    finally:
        env.close()

    saved_video = save_rollout_video(replay_images, video_path, video_fps) if record_video and video_path else None
    return {
        "condition_id": condition["condition_id"],
        "condition_type": condition["condition_type"],
        "trial_id": condition["trial_id"],
        "user_id": condition.get("user_id"),
        "prompt": prompt,
        "bddl_file": condition["bddl_file"],
        "expected": condition.get("expected", {}),
        "visual_source": visual_source,
        "visual_strategy": condition.get("visual_strategy"),
        "reference_image_paths": [str(path) for path in reference_paths],
        "num_images_in_input": num_policy_images,
        "success": bool(done),
        "steps": int(steps),
        "error": error,
        "recorded_frames": len(replay_images),
        "video_path": saved_video,
    }


def prompt_preview(prompt: str, max_chars: int = 120) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def condition_task_info(condition: dict[str, Any]) -> str:
    expected = condition.get("expected", {})
    target = expected.get("object") or condition.get("target_object") or "unknown"
    receptacle = expected.get("target_receptacle", "basket")
    return f"mode={condition['condition_type']} trial={condition['trial_id']} target={target} -> {receptacle}"


def log_line(log_file, message: str) -> None:
    tqdm.write(message)
    log_file.write(message + "\n")
    log_file.flush()


def should_record_video_before_rollout(args: argparse.Namespace, condition: dict[str, Any], index: int, saved_modes: set[str]) -> bool:
    if args.save_videos == "none":
        return False
    if args.save_videos == "all":
        return True
    if args.save_videos == "first-per-mode":
        return condition["condition_type"] not in saved_modes
    if args.save_videos == "every-n":
        return args.video_every > 0 and (index - 1) % args.video_every == 0
    return args.save_videos in {"successes", "failures"}


def keep_video_after_rollout(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    if args.save_videos == "successes":
        return bool(row["success"])
    if args.save_videos == "failures":
        return not bool(row["success"])
    return True


def main() -> None:
    args = parse_args()
    args.modes = normalize_modes(args.modes)
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both --load-in-8bit and --load-in-4bit.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = load_json(args.manifest)
    conditions = build_conditions(args, manifest)
    run_dir = args.output_dir / f"openvla_oft_compare_pickup_{args.run_id or DATE_TIME}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "modes": args.modes,
        "num_conditions": len(conditions),
        "text_granularity": args.text_granularity,
        "text_selectivity": args.text_selectivity,
        "visual_condition": args.visual_condition,
        "visual_source": args.visual_source,
        "wrist_camera_name": args.wrist_camera_name,
        "save_videos": args.save_videos,
        "video_every": args.video_every,
        "video_fps": args.video_fps,
        "dry_run": args.dry_run,
    }
    write_json(run_dir / "run_config.json", metadata)
    write_json(run_dir / "conditions.json", {"conditions": conditions})
    log_file = (run_dir / "run.log").open("w", encoding="utf-8")

    if args.dry_run:
        log_line(log_file, f"Prepared {len(conditions)} conditions under {run_dir}")
        for condition in conditions:
            refs = visual_reference_paths(condition, args.visual_source)
            log_line(log_file, f"DRY-RUN {condition_task_info(condition)} images={2 + len(refs)}")
            log_line(log_file, f"  prompt={prompt_preview(condition_prompt(condition))}")
        log_file.close()
        return

    cfg = OpenVLAOFTEvalConfig(
        pretrained_checkpoint=str(args.checkpoint),
        task_suite_name=args.task_suite_name,
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        num_open_loop_steps=args.num_open_loop_steps,
    )
    model, processor, action_head, proprio_projector, resize_size = load_openvla_oft(cfg)

    log_line(log_file, f"Run directory: {run_dir}")
    log_line(log_file, f"Checkpoint: {args.checkpoint}")
    log_line(log_file, f"Conditions: {len(conditions)}")
    log_line(log_file, f"Visual source: {args.visual_source}")

    result_path = run_dir / "results.jsonl"
    conditions, rows = filter_completed_conditions(conditions, result_path, args.resume)
    saved_video_modes: set[str] = set()
    progress = tqdm(conditions, desc="OpenVLA-OFT pickup eval", unit="cond")
    for index, condition in enumerate(progress, start=1):
        progress.set_postfix(mode=condition["condition_type"], trial=condition["trial_id"])
        log_line(log_file, f"[{index}/{len(conditions)}] START {condition['condition_id']}")
        log_line(log_file, f"  task={condition_task_info(condition)}")
        log_line(log_file, f"  prompt={prompt_preview(condition_prompt(condition))}")
        record_video = should_record_video_before_rollout(args, condition, index, saved_video_modes)
        video_path = run_dir / "videos" / f"{index:04d}_{condition['condition_id']}.mp4"
        row = rollout_openvla_oft_condition(
            condition,
            model,
            processor,
            action_head,
            proprio_projector,
            cfg,
            resize_size,
            visual_source=args.visual_source,
            camera_name=args.camera_name,
            wrist_camera_name=args.wrist_camera_name,
            image_size=args.image_size,
            num_steps_wait=args.num_steps_wait,
            max_steps=args.max_steps,
            max_reset_attempts=args.max_reset_attempts,
            record_video=record_video,
            video_path=video_path,
            video_fps=args.video_fps,
        )
        if record_video and not keep_video_after_rollout(args, row):
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
        log_line(log_file, f"[{index}/{len(conditions)}] {status} steps={row['steps']} images={row['num_images_in_input']}{suffix}")
        if row.get("video_path"):
            log_line(log_file, f"  video={row['video_path']}")

    total = len(rows)
    successes = sum(1 for row in rows if row["success"])
    by_mode = {}
    for row in rows:
        stats = by_mode.setdefault(row["condition_type"], {"episodes": 0, "successes": 0})
        stats["episodes"] += 1
        stats["successes"] += int(row["success"])
    for stats in by_mode.values():
        stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
    summary = {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "by_mode": by_mode,
    }
    write_json(run_dir / "summary.json", summary)
    log_line(log_file, f"Wrote results to {run_dir}")
    log_line(log_file, f"Summary: {summary}")
    log_file.close()


if __name__ == "__main__":
    main()
