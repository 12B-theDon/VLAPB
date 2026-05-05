#!/usr/bin/env python3
"""Reusable OpenVLA helpers for VLAPB/LIBERO evaluation scripts."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image

VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
OPENVLA_ROOT = Path("/home/artemis/Documents/openvla")
EXAMPLES_DIR = VLAPB_LIBERO_ROOT / "examples"

for path in (LIBERO_ROOT / "libero", OPENVLA_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle  # noqa: E402
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
from libero.envs import TASK_MAPPING  # noqa: E402

from test_env_reconfiguration import (  # noqa: E402
    build_env_kwargs,
    get_action_dim,
    reset_with_retries,
)


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/openvla_checkpoints/openvla-7b-finetuned-libero-10")
DEFAULT_EVAL_DIR = VLAPB_LIBERO_ROOT / "evaluations" / "compare_injection_methods"
DEFAULT_MANIFEST = DEFAULT_EVAL_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "openvla_results"
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclass
class OpenVLAEvalConfig:
    pretrained_checkpoint: str = str(DEFAULT_CHECKPOINT)
    model_family: str = "openvla"
    task_suite_name: str = "libero_10"
    unnorm_key: str = "auto"
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def load_openvla(cfg: OpenVLAEvalConfig):
    set_seed(cfg.seed)
    model = get_model(cfg)
    if cfg.unnorm_key == "auto":
        preferred_keys = (
            "libero_10",
            "libero_10_no_noops",
            "libero_object",
            "libero_object_no_noops",
            "bridge_orig",
        )
        cfg.unnorm_key = next((key for key in preferred_keys if key in model.norm_stats), "")
        if not cfg.unnorm_key and model.norm_stats:
            cfg.unnorm_key = sorted(model.norm_stats)[0]
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        available = ", ".join(sorted(model.norm_stats))
        raise KeyError(f"OpenVLA unnorm_key={cfg.unnorm_key!r} not found. Available: {available}")
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    return model, processor, resize_size


def build_custom_bddl_env(bddl_file: Path, camera_name: str, image_size: int):
    env_kwargs, problem_name = build_env_kwargs(bddl_file, camera_name, image_size)
    return TASK_MAPPING[problem_name](**env_kwargs)


def build_policy_observation(obs: dict[str, Any], resize_size: int | tuple[int, int]) -> dict[str, np.ndarray]:
    image = get_libero_image(obs, resize_size)
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return {"full_image": image, "state": state}


def condition_prompt(condition: dict[str, Any]) -> str:
    prompt = str(condition.get("prompt") or "").strip()
    task_instruction = str(condition.get("task_instruction") or "").strip()
    if prompt:
        return prompt
    return task_instruction


def load_condition(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    scene_path = Path(payload["scene_config"])
    scene = load_json(scene_path)
    payload["scene"] = scene
    payload["bddl_file"] = scene["bddl_file"]
    return payload


def make_condition_from_scene(scene_path: Path, mode: str) -> dict[str, Any]:
    scene = load_json(scene_path)
    if mode != "libero_style":
        raise ValueError(f"Unsupported scene-only mode: {mode}")
    target_object = str(scene["target_object"]).replace("_", " ")
    task = f"Pick up the {target_object} and put it in the basket."
    return {
        "condition_id": f"{scene['trial_id']}_{mode}",
        "condition_type": mode,
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


def compose_multimodal_condition(textual: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    merged = dict(visual)
    merged["condition_id"] = f"{visual['trial_id']}_visual_textual"
    merged["condition_type"] = "visual_textual"
    merged["task_instruction"] = textual["task_instruction"]
    merged["personalization_injection"] = textual.get("personalization_injection", "")
    merged["prompt"] = condition_prompt(textual)
    merged["textual_condition_id"] = textual["condition_id"]
    merged["visual_condition_id"] = visual["condition_id"]
    return merged


def visual_policy_image(condition: dict[str, Any], obs: dict[str, Any], resize_size: int | tuple[int, int]) -> np.ndarray:
    current_image = get_libero_image(obs, resize_size)
    if condition.get("visual_strategy") == "obs_only":
        return current_image
    panel_path = condition.get("input_images", {}).get("reference_panel")
    reference_paths = condition.get("input_images", {}).get("image_paths") or []
    reference_path = panel_path or (reference_paths[0] if reference_paths else None)
    use_visual_reference = condition.get("condition_type") in {"visual", "visual_textual"} and reference_path
    if not use_visual_reference or not Path(reference_path).exists():
        return current_image

    if isinstance(resize_size, int):
        target_size = (resize_size, resize_size)
    else:
        target_size = tuple(resize_size)
    obs_image = Image.fromarray(current_image).convert("RGB")
    panel = Image.open(reference_path).convert("RGB").resize(target_size)
    canvas = Image.new("RGB", (target_size[0] * 2, target_size[1]))
    canvas.paste(obs_image, (0, 0))
    canvas.paste(panel, (target_size[0], 0))
    return np.asarray(canvas.resize(target_size))


def rollout_openvla_condition(
    condition: dict[str, Any],
    model: Any,
    processor: Any,
    cfg: OpenVLAEvalConfig,
    resize_size: int | tuple[int, int],
    *,
    camera_name: str = "agentview",
    image_size: int = 256,
    num_steps_wait: int = 10,
    max_steps: int = 520,
    max_reset_attempts: int = 25,
    record_video: bool = False,
    video_path: Path | None = None,
    video_fps: int = 20,
) -> dict[str, Any]:
    env = build_custom_bddl_env(Path(condition["bddl_file"]), camera_name, image_size)
    replay_images: list[np.ndarray] = []
    prompt = condition_prompt(condition)
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
                continue

            policy_obs = build_policy_observation(obs, resize_size)
            policy_obs["full_image"] = visual_policy_image(condition, obs, resize_size)
            if record_video:
                replay_images.append(policy_obs["full_image"])

            action = get_action(cfg, model, policy_obs, prompt, processor=processor)
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
        "visual_strategy": condition.get("visual_strategy"),
        "success": bool(done),
        "steps": int(steps),
        "error": error,
        "recorded_frames": len(replay_images),
        "video_path": saved_video,
    }
