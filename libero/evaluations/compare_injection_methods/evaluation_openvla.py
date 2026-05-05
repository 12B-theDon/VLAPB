#!/usr/bin/env python3
"""Evaluate OpenVLA on four compare-injection pickup conditions.

Conditions:
1. plain: normal task prompt only.
2. textual: textual profile injection + pickup task.
3. visual: visual reference panel fallback + pickup task.
4. visual_textual: visual reference panel fallback + textual profile injection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_PICKUP_CHECKPOINT = Path("/home/artemis/Documents/openvla")

from openvla_eval_utils import (  # noqa: E402
    DATE_TIME,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    OpenVLAEvalConfig,
    choose_textual_condition,
    choose_visual_condition,
    compose_multimodal_condition,
    load_condition,
    load_json,
    load_openvla,
    make_condition_from_scene,
    rollout_openvla_condition,
    write_json,
    write_jsonl,
)
from vlapb_eval_common import (  # noqa: E402
    DEFAULT_VLAPB_MANIFEST,
    add_vlapb_selection_args,
    filter_completed_conditions,
    is_vlapb_manifest,
    normalize_modes as normalize_vlapb_modes,
    result_key,
    select_vlapb_conditions,
)

ORDERED_MODES = ["plain", "textual", "visual", "visual_textual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVLA on VLAPB injection comparison pickup tasks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_VLAPB_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/artemis/libero_data/openvla_checkpoints/openvla-7b"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ORDERED_MODES,
        choices=["plain", "libero_style", "textual", "visual", "visual_textual"],
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
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--max-reset-attempts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--unnorm-key", default="auto")
    parser.add_argument(
        "--save-videos",
        default="all",
        choices=["none", "first-per-mode", "all", "successes", "failures", "every-n"],
        help="Control rollout video saving. Default saves one video per condition type.",
    )
    parser.add_argument("--video-every", type=int, default=8, help="Used when --save-videos every-n.")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Build condition list but do not load OpenVLA or roll out.")
    add_vlapb_selection_args(parser)
    return parser.parse_args()


def normalize_modes(modes: list[str]) -> list[str]:
    normalized = ["plain" if mode == "libero_style" else mode for mode in modes]
    return [mode for mode in ORDERED_MODES if mode in normalized]


def make_plain_condition(scene_path: Path) -> dict:
    condition = make_condition_from_scene(scene_path, "libero_style")
    condition["condition_id"] = f"{condition['trial_id']}_plain"
    condition["condition_type"] = "plain"
    return condition


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
            conditions.append(make_plain_condition(scene_path))
        if "textual" in modes:
            conditions.append(textual)
        if "visual" in modes:
            conditions.append(visual)
        if "visual_textual" in modes:
            conditions.append(compose_multimodal_condition(textual, visual))
    return conditions


def prompt_preview(prompt: str, max_chars: int = 120) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def condition_task_info(condition: dict) -> str:
    expected = condition.get("expected", {})
    target = expected.get("object") or condition.get("target_object") or "unknown"
    receptacle = expected.get("target_receptacle", "basket")
    return (
        f"mode={condition['condition_type']} "
        f"trial={condition['trial_id']} "
        f"target={target} -> {receptacle}"
    )


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
    # success/failure-only modes need the frames available before the outcome is known.
    return args.save_videos in {"successes", "failures"}


def keep_video_after_rollout(args: argparse.Namespace, row: dict) -> bool:
    if args.save_videos == "successes":
        return bool(row["success"])
    if args.save_videos == "failures":
        return not bool(row["success"])
    return True


def main() -> None:
    args = parse_args()
    args.modes = normalize_modes(args.modes)
    manifest = load_json(args.manifest)
    conditions = build_conditions(args, manifest)
    run_dir = args.output_dir / f"openvla_compare_pickup_{args.run_id or DATE_TIME}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "modes": args.modes,
        "num_conditions": len(conditions),
        "text_granularity": args.text_granularity,
        "text_selectivity": args.text_selectivity,
        "visual_condition": args.visual_condition,
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
            log_line(log_file, f"DRY-RUN {condition_task_info(condition)}")
            log_line(log_file, f"  prompt={prompt_preview(condition.get('prompt', ''))}")
        log_file.close()
        return

    cfg = OpenVLAEvalConfig(
        pretrained_checkpoint=str(args.checkpoint),
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
    )
    model, processor, resize_size = load_openvla(cfg)

    log_line(log_file, f"Run directory: {run_dir}")
    log_line(log_file, f"Checkpoint: {args.checkpoint}")
    log_line(log_file, f"Conditions: {len(conditions)}")

    result_path = run_dir / "results.jsonl"
    conditions, rows = filter_completed_conditions(conditions, result_path, args.resume)
    saved_video_modes: set[str] = set()
    progress = tqdm(conditions, desc="OpenVLA pickup eval", unit="cond")
    for index, condition in enumerate(progress, start=1):
        progress.set_postfix(
            mode=condition["condition_type"],
            trial=condition["trial_id"],
        )
        log_line(log_file, f"[{index}/{len(conditions)}] START {condition['condition_id']}")
        log_line(log_file, f"  task={condition_task_info(condition)}")
        log_line(log_file, f"  prompt={prompt_preview(condition.get('prompt', ''))}")
        record_video = should_record_video_before_rollout(args, condition, index, saved_video_modes)
        video_path = run_dir / "videos" / f"{index:04d}_{condition['condition_id']}.mp4"
        row = rollout_openvla_condition(
            condition,
            model,
            processor,
            cfg,
            resize_size,
            camera_name=args.camera_name,
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
        log_line(log_file, f"[{index}/{len(conditions)}] {status} steps={row['steps']}{suffix}")
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
