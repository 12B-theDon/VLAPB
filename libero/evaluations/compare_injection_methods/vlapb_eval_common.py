#!/usr/bin/env python3
"""Common helpers for VLAPB zero-shot and fine-tuned evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_VLAPB_MANIFEST = Path(__file__).resolve().parent / "manifest_vlapb.json"
DEFAULT_TEXT_VARIANT = "one_sentence"
DEFAULT_VISUAL_STRATEGIES = ("obs_only", "obs_ref_h", "ref_as_wrist")
VISUAL_MODES = {"visual", "visual_textual"}


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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", default=None, help="Stable run id for resumable long evaluations.")


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
    conditions = []
    for condition in manifest.get("conditions", []):
        mode = condition.get("condition_type")
        if mode not in modes:
            continue
        if mode == "textual" and condition.get("text_variant") != args.text_variant:
            continue
        if not condition_matches(condition, args):
            continue
        conditions.extend(expand_visual_strategies(condition, args.visual_strategies))
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        conditions = [condition for index, condition in enumerate(conditions) if index % args.num_shards == args.shard_index]
    if args.limit_trials is not None and args.limit_trials > 0:
        conditions = conditions[: args.limit_trials]
    return conditions
