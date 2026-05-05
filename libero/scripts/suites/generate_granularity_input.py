#!/usr/bin/env python3
"""Generate personalized story-style text inputs for VLAPB suites.

This script reads episode metadata under VLAPB_suites plus user profiles from
profiles.json, writes one prompt HTML file, and creates text_inputs/*.json files
containing one-, four-, and eight-sentence personalization stories.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = SCRIPT_DIR.parents[1]
VLAPB_PROJECT_ROOT = VLAPB_LIBERO_ROOT.parent
DEFAULT_PROFILES = VLAPB_LIBERO_ROOT / "profiles" / "profiles.json"
DEFAULT_SUITE_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"
DEFAULT_PROMPT_HTML = SCRIPT_DIR / "granularity_story_prompt.html"

SUITES = ("belongings", "placements", "sequences")
SPLIT_ORDER = (
    "type1",
    "type2",
    "type3",
    "type4",
    "adaptability",
    "multiuser",
    "consistency",
)

PROMPT_TEMPLATE = """Generate three textual injection stories from the given JSON for a personalized VLA benchmark.

The stories are used before task execution. They register user-specific preferences, ownership, object-order rules, and placement habits. They must not be written as task commands.

Generate:
- one_sentence_story: exactly 1 sentence
- four_sentence_story: exactly 4 sentences
- eight_sentence_story: exactly 8 sentences

All three versions must imply the same expected target.
Longer stories may add context or controlled distractors, but must not reveal the target action more directly than the shorter story.
Do not add facts not present in the JSON.
Do not contradict the JSON.
Do not introduce new task-relevant objects.
Do not use imperative robot commands such as "put", "place", "move", or "pick up" as direct instructions.
Do not say "the correct action is" or "the robot should."

Axis rules:
- Sensitivity: express the target user's relevant preference; keep counterfactual variants parallel.
- Adaptability: produce both initial and updated stories; the update must override the old preference.
- Consistency: express a stable multi-factor profile for the same user across multiple tasks.
- Selectivity: include relevant and irrelevant preferences, but keep irrelevant preferences controlled and separable.
- Multi-user: clearly separate preferences by user_id; do not merge users' preferences.
- Entanglement/Disentanglement: keep changed and tested factors independent.

Return valid JSON with:
{
  "task_id": "...",
  "evaluation_axis": "...",
  "user_id": "...",
  "story_variants": {
    "one_sentence": "...",
    "four_sentence": "...",
    "eight_sentence": "..."
  },
  "adaptability_variants": null or {
    "initial": {
      "one_sentence": "...",
      "four_sentence": "...",
      "eight_sentence": "..."
    },
    "updated": {
      "one_sentence": "...",
      "four_sentence": "...",
      "eight_sentence": "..."
    }
  }
}

Input JSON:
{INPUT_JSON}
"""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def object_text(value: Any) -> str:
    if value is None or value == "":
        return "item"
    return str(value).replace("_", " ")


def user_text(value: Any) -> str:
    return str(value) if value else "unknown_user"


def join_items(items: Sequence[Any]) -> str:
    words = [object_text(item) for item in items if item]
    if not words:
        return "no listed items"
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return f"{', '.join(words[:-1])}, and {words[-1]}"


def ordinal(index: int) -> str:
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    return words.get(index, f"{index}th")


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def ensure_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def paragraph(sentences: Sequence[str], expected_count: int) -> str:
    if len(sentences) != expected_count:
        raise ValueError(f"expected {expected_count} sentences, got {len(sentences)}")
    return " ".join(ensure_sentence(sentence) for sentence in sentences)


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    profiles = data.get("profiles", data)
    if isinstance(profiles, dict):
        return {str(user_id): dict(profile) for user_id, profile in profiles.items()}
    return {str(profile["user_id"]): profile for profile in profiles}


def profile_for(profiles: Mapping[str, dict[str, Any]], user_id: Any) -> dict[str, Any]:
    return profiles.get(user_text(user_id), {})


def profile_belongings(profiles: Mapping[str, dict[str, Any]], user_id: Any) -> list[str]:
    return [str(item) for item in profile_for(profiles, user_id).get("belongings", [])]


def profile_placements(profiles: Mapping[str, dict[str, Any]], user_id: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in profile_for(profiles, user_id).get("placements", [])]


def profile_strategy(profiles: Mapping[str, dict[str, Any]], user_id: Any) -> str | None:
    strategy = profile_for(profiles, user_id).get("placement_order_strategy")
    return str(strategy) if strategy else None


def split_limits(root: Path, suites: Iterable[str], respect_summary: bool) -> dict[tuple[str, str], int]:
    if not respect_summary:
        return {}
    limits: dict[tuple[str, str], int] = {}
    for suite in suites:
        summary_path = root / suite / "generation_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        for split, count in (summary.get("planned_counts") or {}).items():
            limits[(suite, split)] = int(count)
    return limits


def metadata_files(root: Path, suites: Iterable[str], respect_summary: bool) -> list[Path]:
    limits = split_limits(root, suites, respect_summary)
    files: list[Path] = []
    for suite in suites:
        suite_root = root / suite
        if not suite_root.exists():
            continue
        seen_splits: set[str] = set()
        for split in SPLIT_ORDER:
            metadata_dir = suite_root / split / "metadata"
            if not metadata_dir.exists():
                continue
            seen_splits.add(split)
            split_files = sorted(metadata_dir.glob("*.json"))
            limit = limits.get((suite, split))
            files.extend(split_files[:limit] if limit is not None else split_files)
        for metadata_dir in sorted(suite_root.glob("*/metadata")):
            split = metadata_dir.parent.name
            if split in seen_splits:
                continue
            split_files = sorted(metadata_dir.glob("*.json"))
            limit = limits.get((suite, split))
            files.extend(split_files[:limit] if limit is not None else split_files)
    return sorted(set(files))


def output_path_for(metadata_path: Path) -> Path:
    return metadata_path.parent.parent / "text_inputs" / metadata_path.name


def suite_axis(metadata: Mapping[str, Any]) -> str:
    split = str(metadata.get("split") or metadata.get("task_type") or "unknown")
    aliases = {
        "type1": "sensitivity",
        "type2": "sensitivity",
        "type3": "selectivity",
        "type4": "entanglement",
        "multiuser": "multi-user",
    }
    return aliases.get(split, split)


def placement_phrase(placement: Mapping[str, Any] | None) -> str:
    placement = placement or {}
    label = placement.get("label")
    fixed_type = placement.get("fixed_type")
    object_type = placement.get("object_type")
    if object_type and label and fixed_type:
        return f"{object_text(object_type)} is associated with the {object_text(label)} area of the {object_text(fixed_type)}"
    if label and fixed_type:
        return f"the {object_text(label)} area of the {object_text(fixed_type)}"
    if fixed_type:
        return f"a familiar area of the {object_text(fixed_type)}"
    return "the recorded preferred area"


def strategy_phrase(strategy: Any) -> str:
    return object_text(strategy) if strategy else "recorded order"


def ordered_sequence(sequence: Sequence[Any]) -> str:
    return ", ".join(f"{ordinal(i)} {object_text(item)}" for i, item in enumerate(sequence, start=1))


def irrelevant_fact(metadata: Mapping[str, Any], profiles: Mapping[str, dict[str, Any]], user_id: str) -> str | None:
    suite = metadata.get("suite")
    belongings = profile_belongings(profiles, user_id)
    placements = profile_placements(profiles, user_id)
    strategy = profile_strategy(profiles, user_id)
    target_objects = set(metadata.get("graspable_objects") or [])
    target_objects.update(metadata.get("target_sequence") or [])
    if metadata.get("target_object"):
        target_objects.add(metadata["target_object"])

    if suite != "belongings":
        for item in belongings:
            if item not in target_objects:
                return f"{user_id} also keeps {object_text(item)} as a personal belonging"
    if suite != "placements":
        for placement in placements:
            obj = placement.get("object_type")
            if obj and obj not in target_objects:
                return f"{user_id} also has a separate habit where {placement_phrase(placement)}"
    if suite != "sequences" and strategy:
        return f"{user_id} also has a separate {strategy_phrase(strategy)} ordering habit"
    return None


def participation_fact(metadata: Mapping[str, Any]) -> str:
    users = metadata.get("participating_users") or []
    if users:
        return f"The participating user_id list contains {', '.join(user_text(user) for user in users)}"
    return f"The target user_id field is {user_text(metadata.get('target_user'))}"


def object_fact(metadata: Mapping[str, Any]) -> str:
    suite = metadata.get("suite")
    if suite == "sequences":
        sequence = metadata.get("target_sequence") or metadata.get("graspable_objects") or []
        return f"The target_sequence field lists {join_items(sequence)}"
    if metadata.get("target_object"):
        return f"The target_object field lists {object_text(metadata.get('target_object'))}"
    objects = metadata.get("graspable_objects") or []
    return f"The graspable_objects field lists {join_items(objects)}"


def suite_fact(metadata: Mapping[str, Any]) -> str:
    return f"The suite field is {metadata.get('suite', 'unknown')} and the split field is {metadata.get('split', 'unknown')}"


def adaptation_user_fact(metadata: Mapping[str, Any]) -> str:
    adaptation = metadata.get("adaptation") or {}
    return f"The adaptation user_id field is {user_text(adaptation.get('user_id') or metadata.get('target_user'))}"


def scene_fact(metadata: Mapping[str, Any]) -> str:
    scene = metadata.get("scene")
    table = metadata.get("table")
    if scene and table:
        return f"The scene field is {scene} and the table field is {table}"
    if scene:
        return f"The scene field is {scene}"
    if table:
        return f"The table field is {table}"
    return f"The task_id field is {metadata.get('episode_id', 'unknown')}"


def facts_for_belongings(metadata: Mapping[str, Any]) -> list[str]:
    ownership = metadata.get("ownership") or {}
    if ownership:
        return [
            f"{user_text(user_id)} treats {join_items(objects)} as belonging to that user"
            for user_id, objects in ownership.items()
        ]
    return [
        f"{user_text(metadata.get('target_user'))} treats {object_text(metadata.get('target_object'))} as belonging to that user"
    ]


def facts_for_placements(metadata: Mapping[str, Any]) -> list[str]:
    user_id = user_text(metadata.get("target_user"))
    preference = metadata.get("preference") or {}
    return [f"{user_id}'s placement habit says {placement_phrase(preference)}"]


def facts_for_sequences(metadata: Mapping[str, Any]) -> list[str]:
    user_id = user_text(metadata.get("target_user"))
    sequence = metadata.get("target_sequence") or metadata.get("graspable_objects") or []
    return [
        f"{user_id}'s ordering habit is {strategy_phrase(metadata.get('sequence_strategy'))}",
        f"{user_id}'s recorded order is {ordered_sequence(sequence)}",
    ]


def relevant_facts(metadata: Mapping[str, Any]) -> list[str]:
    suite = metadata.get("suite")
    if suite == "belongings":
        return facts_for_belongings(metadata)
    if suite == "placements":
        return facts_for_placements(metadata)
    if suite == "sequences":
        return facts_for_sequences(metadata)
    return [f"{user_text(metadata.get('target_user'))} has a recorded personalization profile"]


def consistency_facts(metadata: Mapping[str, Any]) -> list[str]:
    consistency = metadata.get("consistency") or {}
    facts: list[str] = []
    if consistency.get("seen_objects"):
        facts.append(f"seen belongings include {join_items(consistency['seen_objects'])}")
    if consistency.get("unseen_object"):
        facts.append(f"the same profile also covers {object_text(consistency['unseen_object'])}")
    return facts


def story_set(
    metadata: Mapping[str, Any],
    profiles: Mapping[str, dict[str, Any]],
    *,
    override_facts: Sequence[str] | None = None,
    update_note: str | None = None,
) -> dict[str, str]:
    user_id = user_text(metadata.get("target_user"))
    facts = list(override_facts if override_facts is not None else relevant_facts(metadata))
    extra = irrelevant_fact(metadata, profiles, user_id)
    consistency = consistency_facts(metadata)
    is_override = override_facts is not None
    detail = update_note or (facts[1] if len(facts) > 1 else (adaptation_user_fact(metadata) if is_override else object_fact(metadata)))
    context_one = consistency[0] if consistency else scene_fact(metadata)
    context_two = consistency[1] if len(consistency) > 1 else participation_fact(metadata)

    one = ensure_sentence(f"For {user_id}, {'; '.join(facts)}")

    four_sentences = [
        f"The profile record is for {user_id}",
        facts[0],
        detail,
        extra or participation_fact(metadata),
    ]

    eight_sentences = [
        f"The profile record is for {user_id}",
        suite_fact(metadata),
        facts[0],
        detail,
        context_one,
        context_two,
        extra or participation_fact(metadata),
        f"The user_id for this personalization record is {user_id}",
    ]

    return {
        "one_sentence": one,
        "four_sentence": paragraph(four_sentences, 4),
        "eight_sentence": paragraph(eight_sentences, 8),
    }


def adaptability_story_sets(
    metadata: Mapping[str, Any],
    profiles: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, str]] | None:
    adaptation = metadata.get("adaptation")
    if not adaptation:
        return None

    user_id = user_text(adaptation.get("user_id") or metadata.get("target_user"))
    suite = metadata.get("suite")
    if suite == "belongings":
        previous = object_text(adaptation.get("previous_belonging"))
        updated = object_text(adaptation.get("updated_belonging") or metadata.get("target_object"))
        initial_facts = [f"{user_id} previously treated {previous} as the relevant belonging"]
        updated_facts = [f"{user_id} now treats {updated} as the relevant belonging"]
        note = f"The newer belonging record overrides the older {previous} record"
    elif suite == "placements":
        previous = placement_phrase(adaptation.get("previous_placement") or {})
        updated = placement_phrase(adaptation.get("updated_placement") or metadata.get("preference") or {})
        initial_facts = [f"{user_id} previously preferred {previous}"]
        updated_facts = [f"{user_id} now prefers {updated}"]
        note = "The newer placement record overrides the older placement record"
    elif suite == "sequences":
        previous = strategy_phrase(adaptation.get("previous_strategy"))
        updated = strategy_phrase(adaptation.get("updated_strategy") or metadata.get("sequence_strategy"))
        initial_facts = [f"{user_id} previously preferred the {previous} ordering habit"]
        updated_facts = [f"{user_id} now prefers the {updated} ordering habit"]
        note = "The newer ordering record overrides the older ordering record"
    else:
        return None

    initial_metadata = dict(metadata, target_user=user_id)
    updated_metadata = dict(metadata, target_user=user_id)
    return {
        "initial": story_set(initial_metadata, profiles, override_facts=initial_facts),
        "updated": story_set(updated_metadata, profiles, override_facts=updated_facts, update_note=note),
    }


def validate_variants(variants: Mapping[str, str]) -> None:
    expected = {"one_sentence": 1, "four_sentence": 4, "eight_sentence": 8}
    for key, count in expected.items():
        actual = sentence_count(variants[key])
        if actual != count:
            raise ValueError(f"{key} should have {count} sentence(s), got {actual}: {variants[key]}")
    forbidden = re.compile(r"\b(put|place|move|pick up|correct action is|robot should)\b", re.I)
    for key, text in variants.items():
        if forbidden.search(text):
            raise ValueError(f"{key} contains forbidden command wording: {text}")


def build_text_input(metadata: Mapping[str, Any], profiles: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    variants = story_set(metadata, profiles)
    validate_variants(variants)
    adaptability = adaptability_story_sets(metadata, profiles)
    if adaptability:
        validate_variants(adaptability["initial"])
        validate_variants(adaptability["updated"])

    return {
        "task_id": metadata.get("episode_id"),
        "evaluation_axis": suite_axis(metadata),
        "user_id": user_text(metadata.get("target_user")),
        "story_variants": {
            "one_sentence": variants["one_sentence"],
            "four_sentence": variants["four_sentence"],
            "eight_sentence": variants["eight_sentence"],
        },
        "adaptability_variants": adaptability,
    }


def write_prompt_html(path: Path) -> None:
    body = html.escape(PROMPT_TEMPLATE)
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VLAPB Granularity Story Prompt</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; border: 1px solid #d0d7de; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>VLAPB Granularity Story Prompt</h1>
  <pre>{body}</pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    profiles = load_profiles(args.profiles)
    files = metadata_files(args.suite_root, args.suites, args.respect_generation_summary)
    expected_outputs = {output_path_for(path) for path in files}
    counts: Counter[tuple[str, str]] = Counter()
    written = 0
    removed_stale = 0

    if not args.dry_run:
        write_prompt_html(args.prompt_html)

    iterator = tqdm(files, desc="Generating story text inputs", unit="episode") if tqdm else files
    for metadata_path in iterator:
        metadata = read_json(metadata_path)
        out_path = output_path_for(metadata_path)
        counts[(str(metadata.get("suite", "unknown")), str(metadata.get("split", "unknown")))] += 1
        if args.dry_run:
            continue
        if out_path.exists() and not args.overwrite:
            continue
        write_json(out_path, build_text_input(metadata, profiles))
        written += 1

    if not args.dry_run and args.clean_stale:
        for suite in args.suites:
            for text_path in (args.suite_root / suite).glob("*/text_inputs/*.json"):
                if text_path not in expected_outputs:
                    text_path.unlink()
                    removed_stale += 1

    distribution: dict[str, dict[str, int]] = defaultdict(dict)
    for (suite, split), count in sorted(counts.items()):
        distribution[suite][split] = count

    summary = {
        "suite_root": str(args.suite_root),
        "profiles": str(args.profiles),
        "prompt_html": str(args.prompt_html),
        "total_metadata": len(files),
        "written": 0 if args.dry_run else written,
        "removed_stale": 0 if args.dry_run else removed_stale,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "clean_stale": args.clean_stale,
        "respect_generation_summary": args.respect_generation_summary,
        "distribution": distribution,
        "schema": {
            "task_id": "metadata episode_id",
            "evaluation_axis": "split mapped to the benchmark axis name",
            "story_variants": "one-, four-, and eight-sentence personalization stories",
            "adaptability_variants": "initial/updated story sets for adaptation metadata, otherwise null",
        },
    }
    if not args.dry_run:
        write_json(args.suite_root / "granularity_text_input_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--prompt-html", type=Path, default=DEFAULT_PROMPT_HTML)
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--clean-stale", action="store_true", default=True)
    parser.add_argument("--no-clean-stale", dest="clean_stale", action="store_false")
    parser.add_argument(
        "--all-existing-metadata",
        dest="respect_generation_summary",
        action="store_false",
        help="Process every metadata JSON on disk instead of limiting splits by generation_summary.json.",
    )
    parser.set_defaults(respect_generation_summary=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate(args)
    print(f"[INFO] suite root: {summary['suite_root']}")
    print(f"[INFO] profiles: {summary['profiles']}")
    print(f"[INFO] prompt html: {summary['prompt_html']}")
    print(f"[INFO] metadata episodes: {summary['total_metadata']}")
    print(f"[INFO] distribution: {summary['distribution']}")
    if summary["dry_run"]:
        print("[INFO] dry-run: no files written")
    else:
        print(f"[INFO] written text inputs: {summary['written']}")
        print(f"[INFO] removed stale text inputs: {summary['removed_stale']}")
        print(f"[INFO] summary: {args.suite_root / 'granularity_text_input_summary.json'}")


if __name__ == "__main__":
    main()
