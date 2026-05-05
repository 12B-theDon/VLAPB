#!/usr/bin/env python3
"""Evaluate Octo on compare-injection pickup conditions.

This mirrors evaluation_openvla.py but feeds LIBERO observations through Octo's
native two-view interface:

* observation image_primary: current agentview image
* observation image_wrist: robot0_eye_in_hand image, if enabled
* task goal images: optional single reference image in image_primary/image_wrist
* task language: condition prompt
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

CUDA_NVCC_BIN = Path(sys.executable).resolve().parents[1] / "lib" / "python3.10" / "site-packages" / "nvidia" / "cuda_nvcc" / "bin"
if CUDA_NVCC_BIN.exists():
    os.environ["PATH"] = f"{CUDA_NVCC_BIN}:{os.environ.get('PATH', '')}"

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
EXAMPLES_DIR = ROOT / "examples"
OCTO_ROOT = Path("/home/artemis/Documents/octo")
LIBERO_ROOT = Path("/home/artemis/Documents/LIBERO")
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(OCTO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT / "libero"))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Octo's typing helpers refer to jax.random.KeyArray, which is not present in
# some newer JAX 0.4.x builds. The runtime values are still normal JAX arrays.
if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array

from octo.data.utils.data_utils import NormalizationType  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402
from vlapb_eval_common import (  # noqa: E402
    DEFAULT_VLAPB_MANIFEST,
    add_vlapb_selection_args,
    filter_completed_conditions,
    is_vlapb_manifest,
    normalize_modes as normalize_vlapb_modes,
    select_vlapb_conditions,
)


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/octo_checkpoints/checkpoints/octo-base-1.5")
DEFAULT_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = DEFAULT_VLAPB_MANIFEST
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "octo_results"
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
ORDERED_MODES = ["plain", "textual", "visual", "visual_textual"]


@dataclass
class OctoEvalConfig:
    checkpoint: str
    dataset_key: str = "bridge_dataset"
    normalization_type: str = "normal"
    seed: int = 7
    window_size: int = 2
    exec_horizon: int = 1
    argmax: bool = False
    temperature: float = 1.0
    use_wrist: bool = True
    reference_mode: str = "task_goal"
    reference_source: str = "panel"
    normalize_gripper: bool = True
    invert_gripper: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def condition_prompt(condition: dict[str, Any]) -> str:
    prompt = str(condition.get("prompt") or "").strip()
    task_instruction = str(condition.get("task_instruction") or "").strip()
    return prompt or task_instruction


def load_condition(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    scene_path = Path(payload["scene_config"])
    scene = load_json(scene_path)
    payload["scene"] = scene
    payload["bddl_file"] = scene["bddl_file"]
    return payload


def make_condition_from_scene(scene_path: Path, mode: str) -> dict[str, Any]:
    scene = load_json(scene_path)
    if mode not in {"plain", "libero_style"}:
        raise ValueError(f"Unsupported scene-only mode: {mode}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Octo on VLAPB injection comparison pickup tasks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ORDERED_MODES,
        choices=["plain", "libero_style", "textual", "visual", "visual_textual", "both"],
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
    parser.add_argument(
        "--reference-mode",
        default="task_goal",
        choices=["none", "concat_primary", "task_goal", "wrist_goal"],
        help="How visual reference images are passed to Octo.",
    )
    parser.add_argument(
        "--reference-source",
        default="panel",
        choices=["panel", "target_first_view", "target_all_views_panel", "all_views_panel"],
        help=(
            "Which visual reference to use when --reference-mode uses task images. "
            "Octo accepts one task image per configured camera key, so multi-view sources are packed into one panel."
        ),
    )
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--wrist-camera-name", default="robot0_eye_in_hand")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--wrist-image-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--exec-horizon", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-key", default="bridge_dataset")
    parser.add_argument("--normalization-type", default="normal", choices=["normal", "bounds"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--no-wrist", dest="use_wrist", action="store_false")
    parser.set_defaults(use_wrist=True)
    parser.add_argument("--no-normalize-gripper", dest="normalize_gripper", action="store_false")
    parser.set_defaults(normalize_gripper=True)
    parser.add_argument("--no-invert-gripper", dest="invert_gripper", action="store_false")
    parser.set_defaults(invert_gripper=True)
    parser.add_argument(
        "--save-videos",
        default="first-per-mode",
        choices=["none", "first-per-mode", "all", "successes", "failures", "every-n"],
    )
    parser.add_argument("--video-every", type=int, default=8)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    add_vlapb_selection_args(parser)
    return parser.parse_args()


def normalize_modes(modes: list[str]) -> list[str]:
    aliases = {
        "libero_style": "plain",
        "both": "visual_textual",
    }
    normalized = [aliases.get(mode, mode) for mode in modes]
    return [mode for mode in ORDERED_MODES if mode in normalized]


def selected_trials(manifest: dict, trial_ids: list[str] | None, limit: int | None) -> list[dict]:
    trials = manifest.get("trials", [])
    if trial_ids:
        wanted = set(trial_ids)
        trials = [trial for trial in trials if trial["trial_id"] in wanted]
    if limit is not None and limit > 0:
        trials = trials[:limit]
    return trials


def build_conditions(args: argparse.Namespace, manifest: dict) -> list[dict]:
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
            conditions.append(make_condition_from_scene(scene_path, "plain"))
        if "textual" in modes:
            assert textual is not None
            textual = dict(textual)
            textual["condition_type"] = "textual"
            conditions.append(textual)
        if "visual" in modes:
            assert visual is not None
            conditions.append(visual)
        if "visual_textual" in modes:
            assert textual is not None and visual is not None
            conditions.append(compose_multimodal_condition(textual, visual))
    return conditions


def prompt_preview(prompt: str, max_chars: int = 120) -> str:
    compact = " ".join(prompt.split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."


def condition_task_info(condition: dict) -> str:
    expected = condition.get("expected", {})
    target = expected.get("object") or condition.get("target_object") or "unknown"
    receptacle = expected.get("target_receptacle", "basket")
    return f"mode={condition['condition_type']} trial={condition['trial_id']} target={target} -> {receptacle}"


def log_line(log_file, message: str) -> None:
    tqdm.write(message)
    log_file.write(message + "\n")
    log_file.flush()


def should_record_video_before_rollout(args: argparse.Namespace, condition: dict, index: int, saved_modes: set[str]) -> bool:
    if args.save_videos == "none":
        return False
    if args.save_videos == "all":
        return True
    if args.save_videos == "first-per-mode":
        return condition["condition_type"] not in saved_modes
    if args.save_videos == "every-n":
        return args.video_every > 0 and (index - 1) % args.video_every == 0
    return args.save_videos in {"successes", "failures"}


def keep_video_after_rollout(args: argparse.Namespace, row: dict) -> bool:
    if args.save_videos == "successes":
        return bool(row["success"])
    if args.save_videos == "failures":
        return not bool(row["success"])
    return True


def resize_rgb(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    return np.asarray(Image.fromarray(image).convert("RGB").resize(size))


def visual_policy_image(
    condition: dict[str, Any],
    obs: dict[str, Any],
    resize_size: int | tuple[int, int],
    primary_key: str,
) -> np.ndarray:
    current_image = libero_camera_image(obs, primary_key, resize_size)
    panel_path = condition.get("input_images", {}).get("reference_panel")
    use_visual_panel = condition.get("condition_type") in {"visual", "visual_textual"} and panel_path
    if not use_visual_panel or not Path(panel_path).exists():
        return current_image

    if isinstance(resize_size, int):
        target_size = (resize_size, resize_size)
    else:
        target_size = tuple(resize_size)
    obs_image = Image.fromarray(current_image).convert("RGB")
    panel = Image.open(panel_path).convert("RGB").resize(target_size)
    canvas = Image.new("RGB", (target_size[0] * 2, target_size[1]))
    canvas.paste(obs_image, (0, 0))
    canvas.paste(panel, (target_size[0], 0))
    return np.asarray(canvas.resize(target_size))


def load_rgb(path: Path, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    return np.asarray(Image.open(path).convert("RGB").resize(size))


def make_views_panel(paths: list[Path], size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    images = [Image.open(path).convert("RGB").resize(size) for path in paths if path.exists()]
    if not images:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    canvas = Image.new("RGB", (size[0] * len(images), size[1]))
    for idx, image in enumerate(images):
        canvas.paste(image, (idx * size[0], 0))
    return np.asarray(canvas.resize(size))


def target_reference_paths(condition: dict[str, Any]) -> list[Path]:
    target = condition.get("target_object") or condition.get("expected", {}).get("object")
    for item in condition.get("reference_items", []):
        if item.get("object") == target:
            return [Path(path) for path in item.get("views", [])]
    return []


def all_reference_paths(condition: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for item in condition.get("reference_items", []):
        paths.extend(Path(path) for path in item.get("views", []))
    return paths


def condition_reference_image(condition: dict[str, Any], source: str, size: int | tuple[int, int]) -> np.ndarray | None:
    if condition.get("condition_type") not in {"visual", "visual_textual"}:
        return None
    if condition.get("visual_strategy") == "obs_only":
        return None
    image_paths = condition.get("input_images", {}).get("image_paths") or []
    if image_paths and Path(image_paths[0]).exists():
        return load_rgb(Path(image_paths[0]), size)
    if source == "panel":
        panel = condition.get("input_images", {}).get("reference_panel")
        return load_rgb(Path(panel), size) if panel and Path(panel).exists() else None

    target_paths = target_reference_paths(condition)
    if source == "target_first_view":
        return load_rgb(target_paths[0], size) if target_paths else None
    if source == "target_all_views_panel":
        return make_views_panel(target_paths, size) if target_paths else None
    if source == "all_views_panel":
        paths = all_reference_paths(condition)
        return make_views_panel(paths, size) if paths else None
    raise ValueError(f"Unsupported reference source: {source}")


def build_octo_env(bddl_file: Path, camera_name: str, wrist_camera_name: str, image_size: int):
    import robosuite as suite
    from libero.envs import TASK_MAPPING
    from libero.envs.bddl_utils import get_problem_info

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
    if wrist_camera_name and wrist_camera_name != camera_name:
        env_kwargs["camera_names"] = [camera_name, wrist_camera_name]
    return TASK_MAPPING[problem_name](**env_kwargs)


def get_action_dim(env) -> int:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "action_spec"):
        low, _high = env.action_spec
        return int(np.asarray(low).shape[0])
    raise AttributeError("Could not determine action dimension for environment")


def reset_with_retries(env, max_attempts: int):
    from robosuite.utils.errors import RandomizationError

    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return env.reset()
        except RandomizationError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to reset the environment after {max_attempts} attempts") from last_error


def libero_camera_image(obs: dict[str, Any], key: str, size: int | tuple[int, int], rotate: bool = True) -> np.ndarray:
    image = obs[key]
    if rotate:
        image = image[::-1, ::-1]
    return resize_rgb(image, size)


def make_octo_step_observation(
    obs: dict[str, Any],
    condition: dict[str, Any],
    cfg: OctoEvalConfig,
    *,
    primary_size: int,
    wrist_size: int,
    primary_key: str,
    wrist_key: str,
) -> dict[str, np.ndarray]:
    if cfg.reference_mode == "concat_primary":
        primary = visual_policy_image(condition, obs, primary_size, primary_key)
    else:
        primary = libero_camera_image(obs, primary_key, primary_size)

    step_obs = {"image_primary": primary}
    if cfg.use_wrist:
        if wrist_key in obs:
            step_obs["image_wrist"] = libero_camera_image(obs, wrist_key, wrist_size)
        else:
            step_obs["image_wrist"] = np.zeros((wrist_size, wrist_size, 3), dtype=np.uint8)
    return step_obs


def stack_history(history: deque[dict[str, np.ndarray]], window_size: int, num_obs: int) -> dict[str, np.ndarray]:
    stacked = {key: np.stack([item[key] for item in history]) for key in history[0]}
    pad_length = window_size - min(num_obs, window_size)
    timestep_pad_mask = np.ones(window_size, dtype=bool)
    timestep_pad_mask[:pad_length] = False
    stacked["timestep_pad_mask"] = timestep_pad_mask
    return stacked


def make_octo_task(model: OctoModel, condition: dict[str, Any], cfg: OctoEvalConfig, primary_size: int, wrist_size: int):
    prompt = condition_prompt(condition)
    goals = None
    reference = condition_reference_image(condition, cfg.reference_source, primary_size)
    if reference is not None and cfg.reference_mode in {"task_goal", "wrist_goal"}:
        zero_primary = np.zeros((primary_size, primary_size, 3), dtype=np.uint8)
        zero_wrist = np.zeros((wrist_size, wrist_size, 3), dtype=np.uint8)
        if cfg.reference_mode == "task_goal":
            goals = {
                "image_primary": reference[None],
                "image_wrist": zero_wrist[None],
            }
        else:
            goals = {
                "image_primary": zero_primary[None],
                "image_wrist": resize_rgb(reference, wrist_size)[None],
            }
    task = model.create_tasks(goals=goals, texts=[prompt])
    if goals is not None and "pad_mask_dict" in task:
        task["pad_mask_dict"]["image_primary"] = np.array([cfg.reference_mode == "task_goal"], dtype=bool)
        task["pad_mask_dict"]["image_wrist"] = np.array([cfg.reference_mode == "wrist_goal"], dtype=bool)
    return task


def reference_image_count(condition: dict[str, Any], cfg: OctoEvalConfig) -> int:
    if condition.get("condition_type") not in {"visual", "visual_textual"} or cfg.reference_mode == "none":
        return 0
    if condition.get("visual_strategy") == "obs_only":
        return 0
    image_paths = condition.get("input_images", {}).get("image_paths") or []
    if image_paths:
        return len([path for path in image_paths if Path(path).exists()])
    if cfg.reference_source == "panel":
        panel = condition.get("input_images", {}).get("reference_panel")
        return int(bool(panel and Path(panel).exists()))
    if cfg.reference_source == "target_first_view":
        return int(bool(target_reference_paths(condition)))
    if cfg.reference_source == "target_all_views_panel":
        return len([path for path in target_reference_paths(condition) if path.exists()])
    if cfg.reference_source == "all_views_panel":
        return len([path for path in all_reference_paths(condition) if path.exists()])
    return 0


def normalization_type(name: str):
    if name == "normal":
        return NormalizationType.NORMAL
    if name == "bounds":
        return NormalizationType.BOUNDS
    raise ValueError(f"Unsupported normalization type: {name}")


def sample_octo_actions(
    model: OctoModel,
    observations: dict[str, Any],
    task: dict[str, Any],
    rng,
    unnorm_stats: dict[str, Any],
    cfg: OctoEvalConfig,
):
    kwargs = {
        "rng": rng,
        "unnormalization_statistics": unnorm_stats,
        "argmax": cfg.argmax,
        "temperature": cfg.temperature,
    }
    if cfg.normalization_type != "normal":
        kwargs["normalization_type"] = normalization_type(cfg.normalization_type)
    return model.sample_actions(observations, task, **kwargs)


def postprocess_octo_action(action: np.ndarray, cfg: OctoEvalConfig, action_dim: int) -> np.ndarray:
    action = np.array(action, dtype=np.float32, copy=True)
    if action.ndim > 1:
        action = action[0]
    action = action[:action_dim]
    if action.shape[0] < action_dim:
        action = np.pad(action, (0, action_dim - action.shape[0]))
    if action_dim >= 7 and cfg.normalize_gripper:
        action[-1] = 1.0 if action[-1] > 0.5 else -1.0
    if action_dim >= 7 and cfg.invert_gripper:
        action[-1] = -action[-1]
    return action


def rollout_octo_condition(
    condition: dict[str, Any],
    model: OctoModel,
    cfg: OctoEvalConfig,
    *,
    camera_name: str = "agentview",
    wrist_camera_name: str = "robot0_eye_in_hand",
    image_size: int = 256,
    wrist_image_size: int = 128,
    num_steps_wait: int = 10,
    max_steps: int = 520,
    max_reset_attempts: int = 25,
    record_video: bool = False,
    video_path: Path | None = None,
    video_fps: int = 20,
) -> dict[str, Any]:
    env = build_octo_env(Path(condition["bddl_file"]), camera_name, wrist_camera_name, image_size)
    replay_images: list[np.ndarray] = []
    prompt = condition_prompt(condition)
    num_reference_source_images = reference_image_count(condition, cfg)
    done = False
    steps = 0
    error = None
    rng = jax.random.PRNGKey(cfg.seed)
    primary_key = f"{camera_name}_image"
    wrist_key = f"{wrist_camera_name}_image"

    try:
        obs = reset_with_retries(env, max_attempts=max_reset_attempts)
        action_dim = get_action_dim(env)
        dummy_action = np.zeros(action_dim, dtype=np.float32)
        if action_dim >= 7:
            dummy_action[-1] = -1.0

        first_step = make_octo_step_observation(
            obs,
            condition,
            cfg,
            primary_size=image_size,
            wrist_size=wrist_image_size,
            primary_key=primary_key,
            wrist_key=wrist_key,
        )
        history: deque[dict[str, np.ndarray]] = deque([first_step] * cfg.window_size, maxlen=cfg.window_size)
        num_obs = 1
        task = make_octo_task(model, condition, cfg, image_size, wrist_image_size)
        unnorm_stats = model.dataset_statistics[cfg.dataset_key]["action"]

        for t in range(max_steps + num_steps_wait):
            if t < num_steps_wait:
                obs, _, done, _ = env.step(dummy_action.tolist())
                step_obs = make_octo_step_observation(
                    obs,
                    condition,
                    cfg,
                    primary_size=image_size,
                    wrist_size=wrist_image_size,
                    primary_key=primary_key,
                    wrist_key=wrist_key,
                )
                history.append(step_obs)
                num_obs += 1
                continue

            stacked = stack_history(history, cfg.window_size, num_obs)
            batched = jax.tree_map(lambda x: x[None], stacked)
            if record_video:
                replay_images.append(stacked["image_primary"][-1])

            rng, step_rng = jax.random.split(rng)
            actions = sample_octo_actions(model, batched, task, step_rng, unnorm_stats, cfg)
            actions = np.asarray(actions[0])

            for chunk_idx in range(min(cfg.exec_horizon, len(actions))):
                action = postprocess_octo_action(actions[chunk_idx], cfg, action_dim)
                obs, _, done, _ = env.step(action.tolist())
                steps = t - num_steps_wait + 1 + chunk_idx
                step_obs = make_octo_step_observation(
                    obs,
                    condition,
                    cfg,
                    primary_size=image_size,
                    wrist_size=wrist_image_size,
                    primary_key=primary_key,
                    wrist_key=wrist_key,
                )
                history.append(step_obs)
                num_obs += 1
                if done or steps >= max_steps:
                    break
            if done or steps >= max_steps:
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
        "reference_mode": cfg.reference_mode,
        "reference_source": cfg.reference_source,
        "visual_strategy": condition.get("visual_strategy"),
        "num_reference_source_images": num_reference_source_images,
        "num_task_goal_images": int(num_reference_source_images > 0 and cfg.reference_mode in {"task_goal", "wrist_goal"}),
        "success": bool(done),
        "steps": int(steps),
        "error": error,
        "recorded_frames": len(replay_images),
        "video_path": saved_video,
    }


def main() -> None:
    args = parse_args()
    args.modes = normalize_modes(args.modes)
    manifest = load_json(args.manifest)
    conditions = build_conditions(args, manifest)
    run_dir = args.output_dir / f"octo_compare_pickup_{args.run_id or DATE_TIME}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "modes": args.modes,
        "num_conditions": len(conditions),
        "text_granularity": args.text_granularity,
        "text_selectivity": args.text_selectivity,
        "visual_condition": args.visual_condition,
        "reference_mode": args.reference_mode,
        "reference_source": args.reference_source,
        "dataset_key": args.dataset_key,
        "normalization_type": args.normalization_type,
        "window_size": args.window_size,
        "exec_horizon": args.exec_horizon,
        "use_wrist": args.use_wrist,
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
            cfg = OctoEvalConfig(
                checkpoint=str(args.checkpoint),
                reference_mode=args.reference_mode,
                reference_source=args.reference_source,
            )
            source_count = reference_image_count(condition, cfg)
            goal_count = int(source_count > 0 and args.reference_mode in {"task_goal", "wrist_goal"})
            log_line(log_file, f"DRY-RUN {condition_task_info(condition)} source_images={source_count} task_goal_images={goal_count}")
            log_line(log_file, f"  prompt={prompt_preview(condition.get('prompt', ''))}")
        log_file.close()
        return

    cfg = OctoEvalConfig(
        checkpoint=str(args.checkpoint),
        dataset_key=args.dataset_key,
        normalization_type=args.normalization_type,
        seed=args.seed,
        window_size=args.window_size,
        exec_horizon=args.exec_horizon,
        argmax=args.argmax,
        temperature=args.temperature,
        use_wrist=args.use_wrist,
        reference_mode=args.reference_mode,
        reference_source=args.reference_source,
        normalize_gripper=args.normalize_gripper,
        invert_gripper=args.invert_gripper,
    )
    set_seed(cfg.seed)
    model = OctoModel.load_pretrained(cfg.checkpoint)
    if cfg.dataset_key not in model.dataset_statistics:
        available = ", ".join(sorted(model.dataset_statistics))
        raise KeyError(f"Octo dataset_key={cfg.dataset_key!r} not found. Available: {available}")

    log_line(log_file, f"Run directory: {run_dir}")
    log_line(log_file, f"Checkpoint: {args.checkpoint}")
    log_line(log_file, f"Conditions: {len(conditions)}")
    log_line(log_file, f"Reference mode: {args.reference_mode} ({args.reference_source})")
    log_line(log_file, "Octo task-image support: one image_primary goal and one image_wrist goal; multi-view references are packed into panels.")

    result_path = run_dir / "results.jsonl"
    conditions, rows = filter_completed_conditions(conditions, result_path, args.resume)
    saved_video_modes: set[str] = set()
    progress = tqdm(conditions, desc="Octo pickup eval", unit="cond")
    for index, condition in enumerate(progress, start=1):
        progress.set_postfix(mode=condition["condition_type"], trial=condition["trial_id"])
        log_line(log_file, f"[{index}/{len(conditions)}] START {condition['condition_id']}")
        log_line(log_file, f"  task={condition_task_info(condition)}")
        log_line(log_file, f"  prompt={prompt_preview(condition.get('prompt', ''))}")
        record_video = should_record_video_before_rollout(args, condition, index, saved_video_modes)
        video_path = run_dir / "videos" / f"{index:04d}_{condition['condition_id']}.mp4"
        row = rollout_octo_condition(
            condition,
            model,
            cfg,
            camera_name=args.camera_name,
            wrist_camera_name=args.wrist_camera_name,
            image_size=args.image_size,
            wrist_image_size=args.wrist_image_size,
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
        log_line(
            log_file,
            f"[{index}/{len(conditions)}] {status} steps={row['steps']} "
            f"source_images={row['num_reference_source_images']} task_goal_images={row['num_task_goal_images']}{suffix}",
        )
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
