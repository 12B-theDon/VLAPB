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

TEXT_VARIANT_ORDER = ("keyword", "sentence", "paragraph", "one_sentence", "four_sentence", "eight_sentence")
DEFAULT_TEXT_VARIANTS = ("sentence",)
TEXT_VARIANT_ALIASES = {
    "keyword": "keyword",
    "sentence": "sentence",
    "paragraph": "paragraph",
    "one_sentence": "sentence",
    "four_sentence": "paragraph",
    "eight_sentence": "paragraph",
}
SPLIT_ORDER = (
    "type1",
    "type2",
    "type3",
    "type4",
    "adaptability",
    "multiuser",
    "consistency",
)
FULL_DESIGNED_SPLIT_QUOTAS = {
    ("belongings", "type1"): 1,
    ("belongings", "type2"): 1,
    ("belongings", "type3"): 1,
    ("belongings", "adaptability"): 1,
    ("belongings", "multiuser"): 1,
    ("placements", "type1"): 1,
    ("placements", "type2"): 1,
    ("placements", "type3"): 1,
    ("placements", "type4"): 1,
    ("placements", "adaptability"): 1,
    ("placements", "multiuser"): 1,
    ("placements", "consistency"): 1,
    ("sequences", "type1"): 1,
    ("sequences", "type2"): 1,
    ("sequences", "adaptability"): 1,
    ("sequences", "consistency"): 1,
}


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


def object_token(name: str | None) -> str:
    return str(name or "object")


def fixture_token(name: str | None) -> str:
    return str(name or "target")


def placement_preposition(relation: str | None, fixture: str | None = None) -> str:
    relation_text = str(relation or "").lower()
    fixture_text = str(fixture or "").lower()
    if relation_text == "in" or any(token in fixture_text for token in ("basket", "tray", "cabinet", "drawer", "shelf", "microwave")):
        return "in"
    return "on"


def direct_task_input(metadata: dict[str, Any]) -> str:
    suite = metadata.get("suite")
    if suite == "sequences":
        sequence = metadata.get("target_sequence") or metadata.get("graspable_objects") or []
        sequence_text = ", ".join(object_token(item) for item in sequence)
        fixture = fixture_token(metadata.get("fixture"))
        prep = placement_preposition(metadata.get("relation"), fixture)
        return f"Put the objects {prep} the {fixture} in this order: {sequence_text}"

    target = object_token(metadata.get("target_object"))
    if suite == "placements":
        preference = metadata.get("preference") or {}
        label = preference.get("label")
        fixture = fixture_token(preference.get("fixed_type") or metadata.get("fixture"))
        if label:
            label_text = str(label)
            if label_text in {"in", "inside"}:
                return f"Pick the {target} and place it in the {fixture}"
            if label_text in {"on", "top"}:
                return f"Pick the {target} and place it on the {fixture}"
            return f"Pick the {target} and place it at the {label} of the {fixture}"
        return f"Pick the {target} and place it near the {fixture}"

    fixture = fixture_token(metadata.get("fixture"))
    prep = placement_preposition(metadata.get("relation"), fixture)
    return f"Pick the {target} and place it {prep} the {fixture}"


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


def compact_keyword_injection(metadata: dict[str, Any]) -> str:
    user = metadata.get("target_user")
    suite = metadata.get("suite")
    if suite == "belongings" and metadata.get("target_object"):
        return f"{user}: {compact_object_name(metadata['target_object'])}"
    if suite == "placements":
        preference = metadata.get("preference") or {}
        label = preference.get("label")
        fixed_type = preference.get("fixed_type")
        if label and fixed_type:
            return f"{user}: {compact_object_name(label)} {compact_object_name(fixed_type)}"
    if suite == "sequences" and metadata.get("sequence_strategy"):
        return f"{user}: {compact_object_name(metadata['sequence_strategy'])}"
    return f"{user}: profile cue"


def profile_injection_variants(metadata: dict[str, Any], text: dict[str, Any] | None) -> dict[str, str]:
    story_variants = (text or {}).get("story_variants") or {}
    sentence = story_variants.get("one_sentence") or story_variants.get("sentence") or ""
    paragraph = story_variants.get("eight_sentence") or story_variants.get("four_sentence") or sentence
    keyword = story_variants.get("keyword") or compact_keyword_injection(metadata)
    return {
        "keyword": keyword,
        "sentence": sentence,
        "paragraph": paragraph,
    }


def normalize_text_variant(variant: str) -> str:
    if variant not in TEXT_VARIANT_ALIASES:
        raise ValueError(f"Unknown text variant {variant!r}; expected one of {TEXT_VARIANT_ORDER}")
    return TEXT_VARIANT_ALIASES[variant]


def visual_label_injection(visual: dict[str, Any] | None) -> str | None:
    items = visual_reference_items(visual)
    labels = [item["label"] for item in items if item.get("label")]
    if not labels:
        return None
    return "Visual labels: " + ", ".join(labels)


def compose_prompt(
    task_input: str,
    *,
    profile_text: str | None = None,
    profile_text_granularity: str | None = None,
    visual_label: str | None = None,
) -> str:
    parts = [task_input]
    if profile_text:
        label = f"Profile injection ({profile_text_granularity})" if profile_text_granularity else "Profile injection"
        parts.append(f"{label}: {profile_text}")
    if visual_label:
        parts.append(f"Visual injection: {visual_label}")
    return "\n\n".join(parts)


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
    visual_label: str | None = None,
    visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suffix = condition_type if text_variant is None else f"{condition_type}_{text_variant}"
    task_input = direct_task_input(metadata)
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
        "task_instruction": task_input,
        "source_task_instruction": metadata.get("language"),
        "general_task_input": task_input,
        "profile_text_injection": textual_injection,
        "profile_text_granularity": text_variant,
        "visual_label_injection": visual_label,
        "input_format": {
            "general_task_input": task_input,
            "profile_text_injection": textual_injection,
            "profile_text_granularity": text_variant,
            "visual_label_injection": visual_label,
            "has_profile_text": textual_injection is not None,
            "has_visual_label": visual_label is not None,
        },
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
    keep_null_keys = {
        "profile_text_injection",
        "profile_text_granularity",
        "visual_label_injection",
        "text_variant",
        "personalization_injection",
    }
    return {key: value for key, value in condition.items() if value is not None or key in keep_null_keys}


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
    task = direct_task_input(metadata)
    visual_label = visual_label_injection(visual)
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

    injection_variants = profile_injection_variants(metadata, text)
    for variant in text_variants:
        granularity = normalize_text_variant(variant)
        if granularity not in injection_variants or not injection_variants[granularity]:
            continue
        injection = injection_variants[granularity]
        conditions.append(
            make_condition(
                episode_id=episode_id,
                condition_type="textual",
                prompt=compose_prompt(task, profile_text=injection, profile_text_granularity=granularity),
                metadata=metadata,
                metadata_path=metadata_path,
                text_path=text_path,
                visual_path=visual_path,
                text_variant=granularity,
                textual_injection=injection,
            )
        )

    if visual:
        conditions.append(
            make_condition(
                episode_id=episode_id,
                condition_type="visual",
                prompt=compose_prompt(task, visual_label=visual_label),
                metadata=metadata,
                metadata_path=metadata_path,
                text_path=text_path,
                visual_path=visual_path,
                visual_label=visual_label,
                visual=visual,
            )
        )
        for variant in text_variants:
            granularity = normalize_text_variant(variant)
            if granularity not in injection_variants or not injection_variants[granularity]:
                continue
            injection = injection_variants[granularity]
            conditions.append(
                make_condition(
                    episode_id=episode_id,
                    condition_type="visual_textual",
                    prompt=compose_prompt(
                        task,
                        profile_text=injection,
                        profile_text_granularity=granularity,
                        visual_label=visual_label,
                    ),
                    metadata=metadata,
                    metadata_path=metadata_path,
                    text_path=text_path,
                    visual_path=visual_path,
                    text_variant=granularity,
                    textual_injection=injection,
                    visual_label=visual_label,
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
        seen_splits: set[str] = set()
        for split in SPLIT_ORDER:
            metadata_dir = suite_root / split / "metadata"
            if not metadata_dir.exists():
                continue
            seen_splits.add(split)
            paths.extend(sorted(metadata_dir.glob("*.json")))
        for metadata_dir in sorted(suite_root.glob("*/metadata")):
            if metadata_dir.parent.name in seen_splits:
                continue
            paths.extend(sorted(metadata_dir.glob("*.json")))
    return paths


def regroup_episodes(episodes: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        grouped[episode["task_family"]][episode["split"]].append(episode)
    return grouped


def parse_split_quotas(values: list[str] | None, preset: str | None = None) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    if preset == "full-designed":
        quotas.update(FULL_DESIGNED_SPLIT_QUOTAS)
    for value in values or []:
        try:
            suite_split, raw_count = value.split("=", 1)
            suite, split = suite_split.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"--split-quota must be formatted as suite:split=count, got {value!r}") from exc
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"--split-quota count must be non-negative, got {value!r}")
        quotas[(normalize_suite(suite), split)] = count
    return quotas


def trim_episodes(
    episodes: list[dict[str, Any]],
    all_conditions: list[dict[str, Any]],
    *,
    limit: int | None,
    limit_per_suite: int | None,
    split_quotas: dict[tuple[str, str], int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    if split_quotas:
        counts: Counter[tuple[str, str]] = Counter()
        trimmed = []
        for episode in episodes:
            key = (episode["suite"], episode["split"])
            quota = split_quotas.get(key)
            if quota is None or counts[key] >= quota:
                continue
            trimmed.append(episode)
            counts[key] += 1
        episodes = trimmed

    if limit_per_suite is not None and limit_per_suite > 0:
        counts: Counter[str] = Counter()
        trimmed = []
        for episode in episodes:
            suite = episode["suite"]
            if counts[suite] >= limit_per_suite:
                continue
            trimmed.append(episode)
            counts[suite] += 1
        episodes = trimmed

    if limit is not None and limit > 0:
        episodes = episodes[:limit]

    allowed = {episode["episode_id"] for episode in episodes}
    all_conditions = [condition for condition in all_conditions if condition["episode_id"] in allowed]
    grouped = regroup_episodes(episodes)
    return episodes, all_conditions, grouped


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    suites = {normalize_suite(suite) for suite in args.suites}
    text_variants = args.text_variants or list(DEFAULT_TEXT_VARIANTS)
    split_quotas = parse_split_quotas(args.split_quota, args.preset)

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

    episodes, all_conditions, grouped = trim_episodes(
        episodes,
        all_conditions,
        limit=args.limit,
        limit_per_suite=args.limit_per_suite,
        split_quotas=split_quotas,
    )

    summary = summarize_manifest(episodes, all_conditions, skipped)
    return {
        "schema_version": "vlapb_eval_manifest_v1",
        "suites_root": str(args.suites_root),
        "selected_suites": sorted(suites),
        "preset": args.preset,
        "split_quotas": {f"{suite}:{split}": count for (suite, split), count in sorted(split_quotas.items())},
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
    parser.add_argument(
        "--preset",
        choices=("full-designed",),
        default=None,
        help="Use a named episode selection preset. full-designed selects one episode from every designed split.",
    )
    parser.add_argument("--require-text", action="store_true")
    parser.add_argument("--require-visual", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional global episode limit for smoke manifests.")
    parser.add_argument(
        "--limit-per-suite",
        type=int,
        default=None,
        help="Optional per-suite episode limit, useful for balanced smoke manifests such as 5 per suite.",
    )
    parser.add_argument(
        "--split-quota",
        action="append",
        default=None,
        help="Select N episodes from one designed split, formatted as suite:split=count. May be repeated.",
    )
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
