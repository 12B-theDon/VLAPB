#!/usr/bin/env python3
"""Evaluate OpenPI pi0.5 checkpoints on VLAPB compare-injection pickup tasks."""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import math
import os
import random
import sys
import time
import types
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

from libero.envs import TASK_MAPPING  # noqa: E402
from libero.envs.base_object import OBJECTS_DICT  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi.policies import policy_config as openpi_policy_config  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402
from test_env_reconfiguration import build_env_kwargs, get_action_dim, reset_with_retries  # noqa: E402
from vlapb_eval_common import (  # noqa: E402
    DEFAULT_VLAPB_MANIFEST,
    add_vlapb_selection_args,
    apply_prompt_style_to_conditions,
    filter_conditions_by_plain_success,
    filter_completed_conditions,
    is_vlapb_manifest,
    normalize_modes as normalize_vlapb_modes,
    select_vlapb_conditions,
)


DEFAULT_CHECKPOINT = Path("/home/artemis/libero_data/pi_checkpoints/openpi-assets/checkpoints/pi05_libero")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pi05_results"
ORDERED_MODES = ["plain", "textual", "visual", "visual_textual"]
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


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


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.round(5).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: as_jsonable(item) for key, item in value.items()}
    return value


def patch_libero_object_constructors() -> None:
    """Allow LIBERO fixtures to pass joints=None to fixed scanned objects."""

    def wrapped_object_fn(object_fn):
        def wrapper(*args, **kwargs):
            joints = kwargs.get("joints")
            try:
                return object_fn(*args, **kwargs)
            except TypeError as exc:
                if "unexpected keyword argument 'joints'" not in str(exc):
                    raise
                if not inspect.isclass(object_fn) or len(object_fn.__mro__) < 2:
                    kwargs = dict(kwargs)
                    kwargs.pop("joints", None)
                    return object_fn(*args, **kwargs)
                signature = inspect.signature(object_fn)
                name = kwargs.get("name")
                obj_name = kwargs.get("obj_name")
                if name is None and "name" in signature.parameters:
                    default = signature.parameters["name"].default
                    name = default if default is not inspect.Parameter.empty else None
                if obj_name is None and "obj_name" in signature.parameters:
                    default = signature.parameters["obj_name"].default
                    obj_name = default if default is not inspect.Parameter.empty else name
                if name is None:
                    raise
                if obj_name is None:
                    obj_name = name
                instance = object_fn.__new__(object_fn)
                object_fn.__mro__[1].__init__(instance, name=name, obj_name=obj_name, joints=joints)
                return instance

        return wrapper

    for category_name, object_fn in list(OBJECTS_DICT.items()):
        if getattr(object_fn, "_vlapb_joints_compat", False):
            continue
        wrapper = wrapped_object_fn(object_fn)
        wrapper._vlapb_joints_compat = True
        OBJECTS_DICT[category_name] = wrapper


patch_libero_object_constructors()


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


def choose_legacy_condition(manifest: dict[str, Any], trial_id: str, suffix: str) -> Path:
    for item in manifest.get("conditions", []):
        if item.get("trial_id") == trial_id and item.get("condition_id", "").endswith(suffix):
            return Path(item["path"])
    raise FileNotFoundError(f"No legacy condition for trial={trial_id}, suffix={suffix!r}")


def choose_textual_condition(manifest: dict[str, Any], trial_id: str, granularity: str, selectivity: str) -> Path:
    return choose_legacy_condition(manifest, trial_id, f"_text_{granularity}_{selectivity}")


def choose_visual_condition(manifest: dict[str, Any], trial_id: str, visual_condition: str) -> Path:
    return choose_legacy_condition(manifest, trial_id, f"_visual_{visual_condition}")


def load_condition(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    scene_path = Path(payload["scene_config"])
    scene = load_json(scene_path)
    payload["scene"] = scene
    payload["bddl_file"] = scene["bddl_file"]
    return payload


def make_plain_condition(scene_path: Path) -> dict:
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


def compose_multimodal_condition(textual: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    merged = dict(visual)
    merged["condition_id"] = f"{visual['trial_id']}_visual_textual"
    merged["condition_type"] = "visual_textual"
    merged["task_instruction"] = textual["task_instruction"]
    merged["personalization_injection"] = textual.get("personalization_injection", "")
    merged["prompt"] = prompt_for(textual)
    merged["textual_condition_id"] = textual["condition_id"]
    merged["visual_condition_id"] = visual["condition_id"]
    return merged


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
    receptacle = (
        expected.get("target_receptacle")
        or expected.get("fixture")
        or condition.get("target_receptacle")
        or condition.get("fixture")
        or "unknown"
    )
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


def collect_goal_debug(env: Any) -> dict[str, Any]:
    predicates = []
    object_states = getattr(env, "object_states_dict", {})
    for state in getattr(env, "parsed_problem", {}).get("goal_state", []):
        entry: dict[str, Any] = {"state": list(state), "value": None, "objects": {}}
        try:
            entry["value"] = bool(env._eval_predicate(state))
        except Exception as exc:
            entry["error"] = repr(exc)
        for object_name in state[1:]:
            object_state = object_states.get(object_name)
            if object_state is None:
                continue
            try:
                entry["objects"][object_name] = {
                    "type": getattr(object_state, "object_state_type", None),
                    "pos": as_jsonable(object_state.get_geom_state()["pos"]),
                }
            except Exception as exc:
                entry["objects"][object_name] = {"error": repr(exc)}
        if len(state) == 3:
            predicate_name, object_1_name, object_2_name = state
            object_1_state = object_states.get(object_1_name)
            object_2_state = object_states.get(object_2_name)
            if object_1_state is not None and object_2_state is not None:
                checks = {}
                try:
                    if predicate_name.lower() == "in":
                        checks["contact"] = bool(object_2_state.check_contact(object_1_state))
                        checks["contain"] = bool(object_2_state.check_contain(object_1_state))
                    elif predicate_name.lower() == "on":
                        checks["ontop"] = bool(object_2_state.check_ontop(object_1_state))
                except Exception as exc:
                    checks["error"] = repr(exc)
                if checks:
                    entry["checks"] = checks
        predicates.append(entry)
    return {"check_success": bool(env._check_success()), "predicates": predicates}


def rollout_pi05_condition(condition: dict[str, Any], policy: Any, args: argparse.Namespace, video_path: Path, record_video: bool) -> dict[str, Any]:
    env = None
    replay_images: list[np.ndarray] = []
    action_plan: collections.deque[np.ndarray] = collections.deque()
    done = False
    steps = 0
    error = None
    goal_debug = None
    try:
        env = build_pi05_env(Path(condition["bddl_file"]), args.camera_name, args.wrist_camera_name, args.image_size)
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
        if env is not None:
            try:
                goal_debug = collect_goal_debug(env)
            except Exception as exc:
                goal_debug = {"error": repr(exc)}
        else:
            goal_debug = {"error": "Environment construction failed"}
        if env is not None:
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
        "goal_debug": goal_debug,
        "recorded_frames": len(replay_images),
        "video_path": saved_video,
    }


def main() -> None:
    args = parse_args()
    args.modes = normalize_modes(args.modes)
    random.seed(args.seed)
    np.random.seed(args.seed)
    conditions = build_conditions(args, load_json(args.manifest))
    conditions, plain_success_keys = filter_conditions_by_plain_success(conditions, args.require_plain_success_from)
    conditions = apply_prompt_style_to_conditions(conditions, args.prompt_style)
    run_dir = args.output_dir / f"pi05_compare_pickup_{args.run_id or DATE_TIME}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata["plain_success_episode_count"] = len(plain_success_keys)
    metadata["num_conditions_after_plain_filter"] = len(conditions)
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
