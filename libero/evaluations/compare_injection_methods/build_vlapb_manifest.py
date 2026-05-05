#!/usr/bin/env python3
"""Build a unified evaluation manifest from VLAPB_suites.

The VLAPB suite generator writes one directory per suite/split with separate
metadata, textual injection, visual injection, and BDDL files. This script
normalizes those files into one manifest that can drive zero-shot and
fine-tuned rollout evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VLAPB_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITES_ROOT = VLAPB_ROOT / "VLAPB_suites"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "manifest_vlapb.json"

SUITE_ALIASES = {
    "object": "belongings",
    "objects": "belongings",
    "belonging": "belongings",
    "belongings": "belongings",
    "placement": "placements",
    "placements": "placements",
    "sequence": "sequences",
    "sequences": "sequences",
}

TASK_FAMILY = {
    "belongings": "object",
    "placements": "placement",
    "sequences": "sequence",
}

TEXT_VARIANT_ORDER = ("one_sentence", "four_sentence", "eight_sentence")
DEFAULT_TEXT_VARIANTS = ("one_sentence",)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_suite(name: str) -> str:
    key = name.lower().strip()
    if key not in SUITE_ALIASES:
        raise ValueError(f"Unknown suite {name!r}; expected one of {sorted(SUITE_ALIASES)}")
    return SUITE_ALIASES[key]


def sibling_input_path(metadata_path: Path, dirname: str) -> Path:
    return metadata_path.parent.parent / dirname / metadata_path.name


def existing_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def compact_object_name(name: str | None) -> str | None:
    return name.replace("_", " ") if isinstance(name, str) else name


def expected_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "suite": metadata.get("suite"),
        "task_family": TASK_FAMILY.get(metadata.get("suite")),
        "split": metadata.get("split"),
        "task_type": metadata.get("task_type"),
        "target_user": metadata.get("target_user"),
        "target_object": metadata.get("target_object"),
        "target_sequence": metadata.get("target_sequence"),
        "fixture": metadata.get("fixture"),
        "fixtures": metadata.get("fixtures"),
        "relation": metadata.get("relation"),
        "preference": metadata.get("preference"),
        "language": metadata.get("language"),
    }
    return {key: value for key, value in expected.items() if value is not None}


def task_labels(metadata: dict[str, Any]) -> dict[str, Any]:
    suite = metadata.get("suite")
    labels = {
        "suite": suite,
        "task_family": TASK_FAMILY.get(suite),
        "split": metadata.get("split"),
        "task_type": metadata.get("task_type"),
        "scene": metadata.get("scene"),
        "table": metadata.get("table"),
        "target_user": metadata.get("target_user"),
        "target_object": metadata.get("target_object"),
    }
    if suite == "placements":
        preference = metadata.get("preference") or {}
        labels.update(
            {
                "placement_label": preference.get("label"),
                "placement_location": preference.get("location"),
                "fixed_type": preference.get("fixed_type"),
            }
        )
    if suite == "sequences":
        labels.update(
            {
                "sequence_strategy": metadata.get("sequence_strategy"),
                "sequence_length": len(metadata.get("target_sequence") or []),
            }
        )
    return {key: value for key, value in labels.items() if value is not None}


def visual_reference_items(visual: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not visual:
        return []
    paths = visual.get("image_paths") or []
    objects = visual.get("object_types") or []
    items = []
    for index, path in enumerate(paths):
        object_name = objects[index] if index < len(objects) else visual.get("target_object")
        items.append(
            {
                "object": object_name,
                "label": compact_object_name(object_name),
                "views": [path],
            }
        )
    return items


def make_condition(
    *,
    episode_id: str,
    condition_type: str,
    prompt: str,
    metadata: dict[str, Any],
    metadata_path: Path,
    text_path: Path | None,
    visual_path: Path | None,
    text_variant: str | None = None,
    textual_injection: str | None = None,
    visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suffix = condition_type if text_variant is None else f"{condition_type}_{text_variant}"
    condition = {
        "condition_id": f"{episode_id}_{suffix}",
        "condition_type": condition_type,
        "episode_id": episode_id,
        "trial_id": episode_id,
        "suite": metadata.get("suite"),
        "task_family": TASK_FAMILY.get(metadata.get("suite")),
        "split": metadata.get("split"),
        "task_type": metadata.get("task_type"),
        "user_id": metadata.get("target_user"),
        "target_object": metadata.get("target_object"),
        "target_sequence": metadata.get("target_sequence"),
        "task_instruction": metadata.get("language"),
        "prompt": prompt,
        "text_variant": text_variant,
        "personalization_injection": textual_injection,
        "bddl_file": metadata.get("bddl_file"),
        "metadata_file": str(metadata_path),
        "text_input_file": str(text_path) if text_path else None,
        "visual_input_file": str(visual_path) if visual_path else None,
        "expected": expected_from_metadata(metadata),
    }
    if visual:
        condition["visual_prompt"] = visual.get("general_input")
        condition["visual_textual_prompt"] = visual.get("visual_textual_input")
        condition["input_images"] = {
            "image_paths": visual.get("image_paths") or [],
            "reference_panel": None,
        }
        condition["reference_items"] = visual_reference_items(visual)
    return {key: value for key, value in condition.items() if value is not None}


def condition_prompt_with_text(task: str, injection: str) -> str:
    return f"{injection}\n\n{task}" if injection else task


def conditions_for_episode(
    metadata: dict[str, Any],
    metadata_path: Path,
    text_path: Path | None,
    text: dict[str, Any] | None,
    visual_path: Path | None,
    visual: dict[str, Any] | None,
    *,
    text_variants: list[str],
) -> list[dict[str, Any]]:
    episode_id = metadata["episode_id"]
    task = metadata.get("language") or ""
    conditions = [
        make_condition(
            episode_id=episode_id,
            condition_type="plain",
            prompt=task,
            metadata=metadata,
            metadata_path=metadata_path,
            text_path=text_path,
            visual_path=visual_path,
        )
    ]

    story_variants = (text or {}).get("story_variants") or {}
    for variant in text_variants:
        if variant not in story_variants:
            continue
        injection = story_variants[variant]
        conditions.append(
            make_condition(
                episode_id=episode_id,
                condition_type="textual",
                prompt=condition_prompt_with_text(task, injection),
                metadata=metadata,
                metadata_path=metadata_path,
                text_path=text_path,
                visual_path=visual_path,
                text_variant=variant,
                textual_injection=injection,
            )
        )

    if visual:
        visual_prompt = visual.get("general_input") or task
        conditions.append(
            make_condition(
                episode_id=episode_id,
                condition_type="visual",
                prompt=visual_prompt,
                metadata=metadata,
                metadata_path=metadata_path,
                text_path=text_path,
                visual_path=visual_path,
                visual=visual,
            )
        )
        conditions.append(
            make_condition(
                episode_id=episode_id,
                condition_type="visual_textual",
                prompt=visual.get("visual_textual_input") or visual_prompt,
                metadata=metadata,
                metadata_path=metadata_path,
                text_path=text_path,
                visual_path=visual_path,
                visual=visual,
            )
        )
    return conditions


def iter_metadata_files(root: Path, suites: set[str]) -> list[Path]:
    paths = []
    for suite in sorted(suites):
        suite_root = root / suite
        if not suite_root.exists():
            continue
        paths.extend(sorted(suite_root.glob("*/metadata/*.json")))
    return paths


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    suites = {normalize_suite(suite) for suite in args.suites}
    text_variants = args.text_variants or list(DEFAULT_TEXT_VARIANTS)

    episodes = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    all_conditions = []
    skipped = []

    for metadata_path in iter_metadata_files(args.suites_root, suites):
        try:
            metadata = load_json(metadata_path)
            episode_id = metadata["episode_id"]
            suite = metadata.get("suite") or metadata_path.parents[2].name
            split = metadata.get("split") or metadata_path.parents[1].name

            text_path_candidate = sibling_input_path(metadata_path, "text_inputs")
            visual_path_candidate = sibling_input_path(metadata_path, "visual_inputs")
            text_path = text_path_candidate if text_path_candidate.exists() else None
            visual_path = visual_path_candidate if visual_path_candidate.exists() else None

            if args.require_text and text_path is None:
                skipped.append({"metadata_file": str(metadata_path), "reason": "missing_text_input"})
                continue
            if args.require_visual and visual_path is None:
                skipped.append({"metadata_file": str(metadata_path), "reason": "missing_visual_input"})
                continue

            text = load_json(text_path) if text_path else None
            visual = load_json(visual_path) if visual_path else None
            episode_conditions = conditions_for_episode(
                metadata,
                metadata_path,
                text_path,
                text,
                visual_path,
                visual,
                text_variants=text_variants,
            )
            episode = {
                "episode_id": episode_id,
                "trial_id": episode_id,
                "suite": suite,
                "task_family": TASK_FAMILY.get(suite),
                "split": split,
                "task_type": metadata.get("task_type"),
                "bddl_file": metadata.get("bddl_file"),
                "metadata_file": str(metadata_path),
                "text_input_file": existing_path(text_path_candidate),
                "visual_input_file": existing_path(visual_path_candidate),
                "has_textual_injection": text_path is not None,
                "has_visual_injection": visual_path is not None,
                "language": metadata.get("language"),
                "labels": task_labels(metadata),
                "expected": expected_from_metadata(metadata),
                "conditions": episode_conditions,
            }
            episodes.append(episode)
            grouped[TASK_FAMILY.get(suite, suite)][split].append(episode)
            all_conditions.extend(episode_conditions)
        except Exception as exc:  # keep builder useful on partially generated suites.
            skipped.append({"metadata_file": str(metadata_path), "reason": repr(exc)})

    if args.limit is not None and args.limit > 0:
        episodes = episodes[: args.limit]
        allowed = {episode["episode_id"] for episode in episodes}
        all_conditions = [condition for condition in all_conditions if condition["episode_id"] in allowed]
        trimmed_grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for episode in episodes:
            trimmed_grouped[episode["task_family"]][episode["split"]].append(episode)
        grouped = trimmed_grouped

    summary = summarize_manifest(episodes, all_conditions, skipped)
    return {
        "schema_version": "vlapb_eval_manifest_v1",
        "suites_root": str(args.suites_root),
        "selected_suites": sorted(suites),
        "text_variants": text_variants,
        "summary": summary,
        "episodes": episodes,
        "conditions": all_conditions,
        "groups": {
            family: {
                split: {
                    "episode_ids": [item["episode_id"] for item in items],
                    "episode_count": len(items),
                    "condition_count": sum(len(item["conditions"]) for item in items),
                }
                for split, items in sorted(splits.items())
            }
            for family, splits in sorted(grouped.items())
        },
        "skipped": skipped,
    }


def summarize_manifest(
    episodes: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    by_family = Counter(episode["task_family"] for episode in episodes)
    by_suite = Counter(episode["suite"] for episode in episodes)
    by_split = Counter(f"{episode['task_family']}/{episode['split']}" for episode in episodes)
    by_task_type = Counter(
        f"{episode['task_family']}/{episode.get('task_type') or episode['split']}" for episode in episodes
    )
    by_condition = Counter(condition["condition_type"] for condition in conditions)
    return {
        "episode_count": len(episodes),
        "condition_count": len(conditions),
        "skipped_count": len(skipped),
        "by_task_family": dict(sorted(by_family.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_task_type": dict(sorted(by_task_type.items())),
        "by_condition_type": dict(sorted(by_condition.items())),
        "with_textual_injection": sum(1 for episode in episodes if episode["has_textual_injection"]),
        "with_visual_injection": sum(1 for episode in episodes if episode["has_visual_injection"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified VLAPB evaluation manifest.")
    parser.add_argument("--suites-root", type=Path, default=DEFAULT_SUITES_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["object", "placement", "sequence"],
        help="Suite groups to include. Aliases: object=belongings, placement=placements, sequence=sequences.",
    )
    parser.add_argument(
        "--text-variants",
        nargs="+",
        default=list(DEFAULT_TEXT_VARIANTS),
        choices=list(TEXT_VARIANT_ORDER),
    )
    parser.add_argument("--require-text", action="store_true")
    parser.add_argument("--require-visual", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional global episode limit for smoke manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    write_json(args.output, manifest)
    summary = manifest["summary"]
    print(f"Wrote {args.output}")
    print(
        "episodes={episode_count} conditions={condition_count} skipped={skipped_count}".format(
            **summary
        )
    )
    print(f"by_task_family={summary['by_task_family']}")
    print(f"by_condition_type={summary['by_condition_type']}")


if __name__ == "__main__":
    main()
