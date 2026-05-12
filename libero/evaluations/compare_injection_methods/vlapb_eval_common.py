#!/usr/bin/env python3
"""Common helpers for VLAPB zero-shot and fine-tuned evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_VLAPB_MANIFEST = Path(__file__).resolve().parent / "manifest_vlapb.json"
DEFAULT_TEXT_VARIANT = "sentence"
DEFAULT_VISUAL_STRATEGIES = ("obs_only", "obs_ref_h", "ref_as_wrist")
VISUAL_MODES = {"visual", "visual_textual"}
TEXT_VARIANT_ALIASES = {
    "keyword": "keyword",
    "sentence": "sentence",
    "paragraph": "paragraph",
    "one_sentence": "sentence",
    "four_sentence": "paragraph",
    "eight_sentence": "paragraph",
}


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def add_vlapb_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-family", nargs="+", choices=["object", "placement", "sequence"], default=None)
    parser.add_argument("--suite", nargs="+", default=None)
    parser.add_argument("--split", nargs="+", default=None)
    parser.add_argument("--task-type", nargs="+", default=None)
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument("--text-variant", default=DEFAULT_TEXT_VARIANT)
    parser.add_argument("--visual-strategies", nargs="+", choices=list(DEFAULT_VISUAL_STRATEGIES), default=list(DEFAULT_VISUAL_STRATEGIES))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool_arg,
        metavar="{true,false}",
        help="Resume from existing result rows. Accepts --resume, --resume true/false, or --no-resume.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resuming from existing result rows.")
    parser.add_argument("--run-id", default=None, help="Stable run id for resumable long evaluations.")
    parser.add_argument(
        "--prompt-style",
        default="manifest",
        choices=["manifest", "task_only", "libero_simple"],
        help=(
            "Prompt rewrite for text diagnostics. manifest keeps prompts as stored; "
            "task_only removes profile/visual injection text; libero_simple rewrites to a short LIBERO-like command."
        ),
    )
    parser.add_argument(
        "--require-plain-success-from",
        type=Path,
        default=None,
        help=(
            "Optional results.jsonl from a prior plain-only run. When provided, keep only episode/trial ids "
            "whose plain condition succeeded there. This is intended for easy-task injection ablations."
        ),
    )


def is_vlapb_manifest(manifest: dict[str, Any]) -> bool:
    return str(manifest.get("schema_version", "")).startswith("vlapb_eval_manifest")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def result_key(row: dict[str, Any]) -> tuple[str, str | None]:
    return (row["condition_id"], row.get("visual_strategy"))


def filter_completed_conditions(conditions: list[dict[str, Any]], result_path: Path, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(result_path) if resume else []
    done = {result_key(row) for row in rows}
    pending = [condition for condition in conditions if result_key(condition) not in done]
    return pending, rows


def condition_matches(condition: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.task_family and condition.get("task_family") not in set(args.task_family):
        return False
    if args.suite and condition.get("suite") not in set(args.suite):
        return False
    if args.split and condition.get("split") not in set(args.split):
        return False
    if args.task_type and condition.get("task_type") not in set(args.task_type):
        return False
    if args.episode_id and condition.get("episode_id", condition.get("trial_id")) not in set(args.episode_id):
        return False
    return True


def normalize_modes(modes: list[str]) -> list[str]:
    aliases = {"libero_style": "plain", "both": "visual_textual"}
    order = ["plain", "textual", "visual", "visual_textual"]
    selected = {aliases.get(mode, mode) for mode in modes}
    return [mode for mode in order if mode in selected]


def normalize_text_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    return TEXT_VARIANT_ALIASES.get(variant, variant)


def expand_visual_strategies(condition: dict[str, Any], strategies: list[str]) -> list[dict[str, Any]]:
    if condition.get("condition_type") not in VISUAL_MODES:
        item = dict(condition)
        item["visual_strategy"] = None
        return [item]
    expanded = []
    for strategy in strategies:
        item = dict(condition)
        item["visual_strategy"] = strategy
        item["condition_id"] = f"{condition['condition_id']}_{strategy}"
        expanded.append(item)
    return expanded


def select_vlapb_conditions(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = set(normalize_modes(args.modes))
    requested_text_variant = normalize_text_variant(args.text_variant)
    conditions = []
    for condition in manifest.get("conditions", []):
        mode = condition.get("condition_type")
        if mode not in modes:
            continue
        if not condition_matches(condition, args):
            continue
        if mode in {"textual", "visual_textual"}:
            condition_variant = normalize_text_variant(condition.get("text_variant"))
            if condition_variant is not None and condition_variant != requested_text_variant:
                continue
        conditions.extend(expand_visual_strategies(condition, args.visual_strategies))
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        conditions = [condition for index, condition in enumerate(conditions) if index % args.num_shards == args.shard_index]
    if (
        args.limit_trials is not None
        and args.limit_trials > 0
        and not getattr(args, "episode_id", None)
        and not getattr(args, "trial_id", None)
    ):
        conditions = conditions[: args.limit_trials]
    return conditions


def humanize_name(name: Any) -> str:
    return str(name or "").replace("_", " ").strip()


def condition_target_object(condition: dict[str, Any]) -> str:
    expected = condition.get("expected", {})
    return humanize_name(
        expected.get("target_object")
        or expected.get("object")
        or condition.get("target_object")
        or condition.get("object")
        or "object"
    )


def condition_target_receptacle(condition: dict[str, Any]) -> str:
    expected = condition.get("expected", {})
    return humanize_name(
        expected.get("fixture")
        or expected.get("target_receptacle")
        or condition.get("fixture")
        or condition.get("target_receptacle")
        or "basket"
    )


def condition_relation(condition: dict[str, Any]) -> str:
    expected = condition.get("expected", {})
    return str(expected.get("relation") or condition.get("relation") or "").lower()


def libero_simple_prompt(condition: dict[str, Any]) -> str:
    obj = condition_target_object(condition)
    receptacle = condition_target_receptacle(condition)
    relation = condition_relation(condition)
    prep = "on" if relation == "on" else "in"
    return f"Pick up the {obj} and put it {prep} the {receptacle}."


def task_only_prompt(condition: dict[str, Any]) -> str:
    task_instruction = str(condition.get("task_instruction") or "").strip()
    if task_instruction:
        return task_instruction
    prompt = str(condition.get("prompt") or "").strip()
    return prompt.split("\n\n", 1)[0].strip()


def apply_prompt_style(condition: dict[str, Any], prompt_style: str) -> dict[str, Any]:
    if prompt_style == "manifest":
        return condition
    item = dict(condition)
    if prompt_style == "task_only":
        prompt = task_only_prompt(item)
    elif prompt_style == "libero_simple":
        prompt = libero_simple_prompt(item)
    else:
        raise ValueError(f"Unknown prompt_style={prompt_style!r}")
    item["prompt"] = prompt
    item["task_instruction"] = prompt
    item["prompt_style"] = prompt_style
    return item


def apply_prompt_style_to_conditions(conditions: list[dict[str, Any]], prompt_style: str) -> list[dict[str, Any]]:
    return [apply_prompt_style(condition, prompt_style) for condition in conditions]


def condition_episode_key(condition: dict[str, Any]) -> str:
    return str(condition.get("episode_id") or condition.get("trial_id") or condition.get("condition_id"))


def successful_plain_episode_keys(result_path: Path) -> set[str]:
    keys: set[str] = set()
    for row in read_jsonl(result_path):
        if row.get("condition_type") != "plain":
            continue
        if not bool(row.get("success")):
            continue
        key = str(row.get("episode_id") or row.get("trial_id") or row.get("condition_id"))
        keys.add(key)
    return keys


def filter_conditions_by_plain_success(
    conditions: list[dict[str, Any]],
    plain_result_path: Path | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    if plain_result_path is None:
        return conditions, set()
    allowed_keys = successful_plain_episode_keys(plain_result_path)
    filtered = [condition for condition in conditions if condition_episode_key(condition) in allowed_keys]
    return filtered, allowed_keys
