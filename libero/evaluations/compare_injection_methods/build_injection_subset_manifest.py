#!/usr/bin/env python3
"""Build an easy injection-comparison manifest subset from a VLAPB manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlapb_eval_common import DEFAULT_VLAPB_MANIFEST, normalize_modes, normalize_text_variant


TEXT_VARIANT_ALIASES = {
    "sentence": "sentence",
    "keyword": "keyword",
    "paragraph": "paragraph",
    "one_sentence": "sentence",
    "four_sentence": "paragraph",
    "eight_sentence": "paragraph",
}


def normalize_legacy_text_variant(raw: Any) -> str | None:
    if raw is None:
        return None
    return TEXT_VARIANT_ALIASES.get(str(raw), str(raw))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an injection-comparison-ready VLAPB manifest subset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_VLAPB_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("plain", "textual", "visual", "visual_textual"),
        choices=["plain", "textual", "visual", "visual_textual", "libero_style", "both"],
    )
    parser.add_argument("--task-family", nargs="+", default=None)
    parser.add_argument("--suite", nargs="+", default=None)
    parser.add_argument("--split", nargs="+", default=None)
    parser.add_argument("--task-type", nargs="+", default=None)
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument("--text-variant", default="sentence")
    parser.add_argument("--require-plain-success-from", type=Path, default=None)
    parser.add_argument("--limit-trials", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Alias for limit-trials.")
    return parser.parse_args()


def condition_matches(condition: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.task_family and condition.get("task_family") not in set(args.task_family):
        return False
    if args.suite and condition.get("suite") not in set(args.suite):
        return False
    if args.split and condition.get("split") not in set(args.split):
        return False
    if args.task_type and condition.get("task_type") not in set(args.task_type):
        return False
    ep = condition.get("episode_id") or condition.get("trial_id")
    if args.episode_id and ep not in set(args.episode_id):
        return False
    return True


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


def successful_plain_keys(result_path: Path) -> set[str]:
    keys: set[str] = set()
    for row in read_jsonl(result_path):
        if row.get("condition_type") != "plain":
            continue
        if not bool(row.get("success")):
            continue
        key = str(row.get("episode_id") or row.get("trial_id") or row.get("condition_id"))
        if key:
            keys.add(key)
    return keys


def condition_key(condition: dict[str, Any]) -> str:
    return str(condition.get("episode_id") or condition.get("trial_id") or condition.get("condition_id"))


def select_conditions_for_episode(
    conditions: list[dict[str, Any]],
    required_modes: set[str],
    requested_text_variant: str | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    selected: list[dict[str, Any]] = []
    available: set[str] = set()
    for condition in conditions:
        mode = condition.get("condition_type")
        if mode == "textual":
            if requested_text_variant is not None and normalize_legacy_text_variant(condition.get("text_variant")) != requested_text_variant:
                continue
            available.add("textual")
            selected.append(condition)
        elif mode == "visual_textual":
            if requested_text_variant is not None and normalize_legacy_text_variant(condition.get("text_variant")) != requested_text_variant:
                continue
            available.add("visual_textual")
            selected.append(condition)
        elif mode in {"plain", "visual"}:
            available.add(mode)
            selected.append(condition)

    if not required_modes.issubset(available):
        return [], set()

    return [condition for condition in selected if condition.get("condition_type") in required_modes], available


def filter_vlapb_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_modes = set(normalize_modes(list(args.modes)))
    plain_success = successful_plain_keys(args.require_plain_success_from) if args.require_plain_success_from else None
    requested_text_variant = normalize_text_variant(args.text_variant)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for condition in manifest.get("conditions", []):
        if not condition_matches(condition, args):
            continue
        grouped[condition_key(condition)].append(condition)

    selected_conditions: list[dict[str, Any]] = []
    dropped_count = 0
    kept_episodes = []
    for episode_id, episode_conditions in grouped.items():
        if plain_success is not None and episode_id not in plain_success:
            dropped_count += 1
            continue
        filtered, available = select_conditions_for_episode(
            episode_conditions, required_modes, requested_text_variant
        )
        if not filtered:
            dropped_count += 1
            continue
        selected_conditions.extend(filtered)
        kept_episodes.append(
            {
                "episode_id": episode_id,
                "kept_modes": sorted(available),
                "condition_count": len(filtered),
            }
        )

    selected_conditions.sort(key=lambda condition: condition.get("condition_id", ""))
    if args.limit is not None and args.limit > 0:
        args.limit_trials = args.limit
    if args.limit_trials is not None and args.limit_trials > 0:
        # keep first N episodes with complete variant coverage
        episode_order = [item["episode_id"] for item in kept_episodes]
        allowed = set(episode_order[: args.limit_trials])
        selected_conditions = [cond for cond in selected_conditions if condition_key(cond) in allowed]
        kept_episodes = [item for item in kept_episodes if item["episode_id"] in allowed]

    summary = {
        "episode_count": len(kept_episodes),
        "condition_count": len(selected_conditions),
        "dropped_count": dropped_count,
        "by_mode": {mode: 0 for mode in sorted(required_modes)},
        "source_manifest": str(args.manifest),
    }
    for item in kept_episodes:
        for mode in item["kept_modes"]:
            if mode in summary["by_mode"]:
                summary["by_mode"][mode] += 1
    for mode in list(summary["by_mode"]):
        if summary["by_mode"][mode] == 0:
            summary["by_mode"].pop(mode)

    return selected_conditions, summary


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    conditions, summary = filter_vlapb_manifest(manifest, args)
    output = {
        "conditions": conditions,
        **({"summary": summary} if summary else {}),
    }
    write_json(args.output, output)
    print(f"Wrote {args.output}")
    print(
        "episodes={episode_count} conditions={condition_count} dropped={dropped_count}".format(
            **summary
        )
    )
    print(f"by_mode={summary['by_mode']}")


if __name__ == "__main__":
    main()
