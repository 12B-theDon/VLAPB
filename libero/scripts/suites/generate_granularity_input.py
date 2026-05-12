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
import os
import re
import time
import urllib.error
import urllib.request
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
DEFAULT_STORY_TEMPLATE = SCRIPT_DIR / "granularity_story_template.json"
DEFAULT_OPENAI_API_KEY_FILE = SCRIPT_DIR / "openai_api_key.txt"
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

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

PROMPT_TEMPLATE = """Generate three textual injection notes from the given JSON for a personalized VLA benchmark.

The notes are used as profile context after a separate task command. They register user-specific preferences, ownership, object-order rules, and placement habits. They must not be written as task commands.
Write them as concise VLA-friendly profile notes that explain the user's tendency, not like the general task input and not like instructions for an agent.

Generate:
- one_sentence_story: exactly 1 sentence
- four_sentence_story: exactly 4 sentences
- eight_sentence_story: exactly 8 sentences

All three versions must imply the same expected target.
Longer notes may add context or controlled distractors, but must keep the target-relevant cue easy to read.
Do not add facts not present in the JSON.
Do not contradict the JSON.
Do not introduce new task-relevant objects.
Do not restate the general task command; the concrete task input is provided separately.
Do not use imperative robot commands or task verbs such as "put", "place", "move", or "pick up".
Do not say "the correct action is" or "the robot should."
Keep wording short, concrete, and low-fluff; avoid storytelling, emotional language, and vague scene description.
Prefer varied natural wording over repeating the same sentence frame.
Use the profile_story_facts field as the main source; episode_context is only supporting context.
For belongings, infer favored object characteristics from the user's items, such as material, shape, category, container-like form, food-like role, or tabletop use.
For placements, infer spatial organization habits from the preference label and fixture, such as edge anchoring, lower/back storage, visible access, containment, or surface organization.
For sequences, infer ordering tendencies from the strategy, such as scanning left to right, sweeping right to left, nearest-first cleanup, farthest-first planning, alphabetical indexing, or alternating by position.

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


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_story_template(path: Path) -> dict[str, Any]:
    template = read_json(path)
    variants = template.get("story_variants")
    if not isinstance(variants, dict):
        raise ValueError(f"{path} must contain a story_variants object")
    required = {"one_sentence", "four_sentence", "eight_sentence"}
    missing = required - set(variants)
    if missing:
        raise ValueError(f"{path} is missing story_variants keys: {sorted(missing)}")
    if not isinstance(variants["four_sentence"], list) or len(variants["four_sentence"]) != 4:
        raise ValueError(f"{path} story_variants.four_sentence must be a list of 4 templates")
    if not isinstance(variants["eight_sentence"], list) or len(variants["eight_sentence"]) != 8:
        raise ValueError(f"{path} story_variants.eight_sentence must be a list of 8 templates")
    return template


def render_template(value: str, context: Mapping[str, str]) -> str:
    return value.format_map(SafeFormatDict(context))


def read_api_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() not in {"OPENAI_API_KEY", "api_key", "key"}:
                continue
            line = value.strip().strip('"').strip("'")
        return line
    return None


def resolve_openai_api_key(env_name: str, key_file: Path | None) -> str | None:
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    if key_file is not None:
        return read_api_key_file(key_file)
    return None


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
    unique_files = []
    seen_files: set[Path] = set()
    for path in files:
        if path in seen_files:
            continue
        unique_files.append(path)
        seen_files.add(path)
    return unique_files


def output_path_for(metadata_path: Path) -> Path:
    return metadata_path.parent.parent / "text_inputs" / metadata_path.name


def limited_metadata_files(
    files: list[Path],
    max_episodes: int | None,
    max_episodes_per_suite: int | None,
    split_quotas: Mapping[tuple[str, str], int],
    suite_root: Path,
) -> list[Path]:
    if split_quotas:
        counts: Counter[tuple[str, str]] = Counter()
        limited = []
        for path in files:
            try:
                relative = path.relative_to(suite_root)
                suite, split = relative.parts[0], relative.parts[1]
            except (ValueError, IndexError):
                continue
            key = (suite, split)
            quota = split_quotas.get(key)
            if quota is None or counts[key] >= quota:
                continue
            limited.append(path)
            counts[key] += 1
        files = limited

    if max_episodes_per_suite is not None:
        if max_episodes_per_suite < 0:
            raise ValueError("--max-episodes-per-suite must be non-negative")
        if max_episodes_per_suite > 0:
            counts: Counter[str] = Counter()
            limited = []
            for path in files:
                try:
                    suite = path.relative_to(suite_root).parts[0]
                except ValueError:
                    suite = path.parts[-4] if len(path.parts) >= 4 else "unknown"
                if counts[suite] >= max_episodes_per_suite:
                    continue
                limited.append(path)
                counts[suite] += 1
            files = limited
    if max_episodes is None:
        return files
    if max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")
    return files[:max_episodes]


def parse_split_quotas(values: Sequence[str] | None, preset: str | None = None) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    if preset == "full-designed":
        quotas.update(FULL_DESIGNED_SPLIT_QUOTAS)
    valid_suites = set(SUITES)
    for value in values or []:
        try:
            suite_split, raw_count = value.split("=", 1)
            suite, split = suite_split.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"--split-quota must be formatted as suite:split=count, got {value!r}") from exc
        if suite not in valid_suites:
            raise ValueError(f"Unknown suite in --split-quota: {suite!r}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"--split-quota count must be non-negative, got {value!r}")
        quotas[(suite, split)] = count
    return quotas


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


def unique_phrases(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in seen:
            unique.append(phrase)
            seen.add(phrase)
    return unique


def object_characteristics(item: Any) -> list[str]:
    name = str(item or "").lower()
    traits: list[str] = []
    if any(token in name for token in ("bowl", "ramekin", "mug", "plate")):
        traits.extend(["tabletop dishware", "contained serving shapes"])
    if any(token in name for token in ("porcelain", "ceramic", "glazed")):
        traits.extend(["smooth ceramic finishes", "neat finished surfaces"])
    if any(token in name for token in ("black", "white", "red", "yellow")):
        traits.append("clear visual color cues")
    if any(token in name for token in ("sauce", "dressing", "ketchup")):
        traits.extend(["condiment-style items", "bottle-like pantry goods"])
    if any(token in name for token in ("pudding", "cookies", "soup", "cheese", "butter", "milk", "juice")):
        traits.extend(["food-oriented items", "kitchen consumables"])
    if any(token in name for token in ("book",)):
        traits.extend(["flat readable objects", "ordered study materials"])
    if any(token in name for token in ("pan", "pot")):
        traits.extend(["cookware", "handled kitchen tools"])
    if any(token in name for token in ("bottle",)):
        traits.extend(["tall container forms", "drink or pantry containers"])
    if not traits:
        traits.extend(["distinctive household objects", "recognizable personal items"])
    return unique_phrases(traits)


def object_characteristic_phrase(items: Sequence[Any]) -> str:
    traits: list[str] = []
    for item in items:
        traits.extend(object_characteristics(item))
    return join_items(unique_phrases(traits)[:4])


def belonging_tendency(user_id: Any, objects: Sequence[Any]) -> str:
    user = user_text(user_id)
    items = [object_text(item) for item in objects]
    noun = "item" if len(items) == 1 else "items"
    return (
        f"{user}'s belongings suggest a preference for {object_characteristic_phrase(objects)}, "
        f"with {join_items(items)} serving as the recorded {noun}"
    )


def label_tendency(label: Any) -> str:
    text = object_text(label).lower()
    parts = [part for part in re.split(r"[_\s-]+", text) if part]
    tendencies: list[str] = []
    if "left" in parts:
        tendencies.append("left-side anchoring")
    if "right" in parts:
        tendencies.append("right-side anchoring")
    if "front" in parts:
        tendencies.append("front-facing access")
    if "back" in parts:
        tendencies.append("back-row storage")
    if "top" in parts:
        tendencies.append("upper-surface visibility")
    if "bottom" in parts:
        tendencies.append("lower-compartment stability")
    if "middle" in parts or "center" in parts:
        tendencies.append("central grouping")
    if "in" in parts or "inside" in parts:
        tendencies.append("contained organization")
    if "on" in parts:
        tendencies.append("open-surface organization")
    return join_items(unique_phrases(tendencies) or ["a stable recorded zone"])


def fixture_tendency(fixture: Any) -> str:
    text = object_text(fixture).lower()
    if "basket" in text or "tray" in text:
        return "container-based grouping"
    if "cabinet" in text or "shelf" in text:
        return "structured storage"
    if "stove" in text:
        return "flat work-surface access"
    if "microwave" in text:
        return "appliance-adjacent staging"
    return "fixture-specific organization"


def placement_tendency(user_id: Any, placement: Mapping[str, Any] | None) -> str:
    placement = placement or {}
    user = user_text(user_id)
    item = object_text(placement.get("object_type"))
    label = placement.get("label")
    fixture = placement.get("fixed_type")
    return (
        f"{user}'s spatial habit links {item} with {label_tendency(label)} around the "
        f"{object_text(fixture)}, suggesting {fixture_tendency(fixture)}"
    )


def sequence_tendency_text(strategy: Any) -> str:
    key = str(strategy or "").lower()
    tendencies = {
        "left_to_right": "scans a workspace from left to right and favors a clean horizontal sweep",
        "right_to_left": "scans a workspace from right to left and clears items by reversing the usual visual path",
        "alphabetical": "uses name-based indexing and treats alphabetical order as the organizing cue",
        "reverse_alphabetical": "works backward through names and uses reverse alphabetical order as the organizing cue",
        "fixed_near_to_far": "starts with nearby items first and expands outward across the workspace",
        "fixed_far_to_near": "starts with distant items first and finishes with the closest items",
        "odd_positions_first": "prioritizes alternating positions by attending to odd-numbered slots first",
        "even_positions_first": "prioritizes alternating positions by attending to even-numbered slots first",
    }
    return tendencies.get(key, f"uses the {object_text(strategy)} ordering habit")


def sequence_tendency(user_id: Any, strategy: Any, sequence: Sequence[Any]) -> str:
    return (
        f"{user_text(user_id)} {sequence_tendency_text(strategy)}, "
        f"with the profile example ordered as {ordered_sequence(sequence)}"
    )


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
                return belonging_tendency(user_id, [item])
    if suite != "placements":
        for placement in placements:
            obj = placement.get("object_type")
            if obj and obj not in target_objects:
                return placement_tendency(user_id, placement)
    if suite != "sequences" and strategy:
        return f"{user_id} also shows an ordering tendency that {sequence_tendency_text(strategy)}"
    return None


def participation_fact(metadata: Mapping[str, Any]) -> str:
    users = metadata.get("participating_users") or []
    if users:
        return f"The shared scene includes {', '.join(user_text(user) for user in users)}"
    return f"The profile note centers on {user_text(metadata.get('target_user'))}"


def object_fact(metadata: Mapping[str, Any]) -> str:
    suite = metadata.get("suite")
    if suite == "sequences":
        sequence = metadata.get("target_sequence") or metadata.get("graspable_objects") or []
        return f"The profile example uses {join_items(sequence)} to express an ordering tendency"
    if metadata.get("target_object"):
        item = metadata.get("target_object")
        return f"The relevant object carries {object_characteristic_phrase([item])} as profile-level cues"
    objects = metadata.get("graspable_objects") or []
    return f"The available objects express cues such as {object_characteristic_phrase(objects)}"


def suite_fact(metadata: Mapping[str, Any]) -> str:
    return f"This injection belongs to the {metadata.get('suite', 'unknown')} profile axis in the {metadata.get('split', 'unknown')} split"


def adaptation_user_fact(metadata: Mapping[str, Any]) -> str:
    adaptation = metadata.get("adaptation") or {}
    return f"The profile update belongs to {user_text(adaptation.get('user_id') or metadata.get('target_user'))}"


def scene_fact(metadata: Mapping[str, Any]) -> str:
    scene = metadata.get("scene")
    table = metadata.get("table")
    if scene and table:
        return f"The scene context is {scene} with {table}"
    if scene:
        return f"The scene context is {scene}"
    if table:
        return f"The table context is {table}"
    return f"The episode identifier is {metadata.get('episode_id', 'unknown')}"


def facts_for_belongings(metadata: Mapping[str, Any]) -> list[str]:
    ownership = metadata.get("ownership") or {}
    if ownership:
        return [belonging_tendency(user_id, list(objects)) for user_id, objects in ownership.items()]
    return [
        belonging_tendency(metadata.get("target_user"), [metadata.get("target_object")])
    ]


def facts_for_placements(metadata: Mapping[str, Any]) -> list[str]:
    user_id = user_text(metadata.get("target_user"))
    preference = metadata.get("preference") or {}
    return [placement_tendency(user_id, preference)]


def facts_for_sequences(metadata: Mapping[str, Any]) -> list[str]:
    user_id = user_text(metadata.get("target_user"))
    sequence = metadata.get("target_sequence") or metadata.get("graspable_objects") or []
    return [
        sequence_tendency(user_id, metadata.get("sequence_strategy"), sequence),
        f"The sequence record keeps the tendency abstract so the separate task input can provide the concrete action",
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
    story_template: Mapping[str, Any],
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
    context = {
        "user_id": user_id,
        "facts_joined": "; ".join(facts),
        "fact0": facts[0],
        "detail": detail,
        "extra_or_participation": extra or participation_fact(metadata),
        "suite_fact": suite_fact(metadata),
        "context_one": context_one,
        "context_two": context_two,
    }
    variants = story_template["story_variants"]
    one = ensure_sentence(render_template(str(variants["one_sentence"]), context))
    four_sentences = [render_template(str(sentence), context) for sentence in variants["four_sentence"]]
    eight_sentences = [render_template(str(sentence), context) for sentence in variants["eight_sentence"]]

    return {
        "one_sentence": one,
        "four_sentence": paragraph(four_sentences, 4),
        "eight_sentence": paragraph(eight_sentences, 8),
    }


def adaptability_story_sets(
    metadata: Mapping[str, Any],
    profiles: Mapping[str, dict[str, Any]],
    story_template: Mapping[str, Any],
) -> dict[str, dict[str, str]] | None:
    adaptation = metadata.get("adaptation")
    if not adaptation:
        return None

    user_id = user_text(adaptation.get("user_id") or metadata.get("target_user"))
    suite = metadata.get("suite")
    if suite == "belongings":
        previous = object_text(adaptation.get("previous_belonging"))
        updated_raw = adaptation.get("updated_belonging") or metadata.get("target_object")
        updated = object_text(updated_raw)
        initial_facts = [belonging_tendency(user_id, [adaptation.get("previous_belonging")])]
        updated_facts = [belonging_tendency(user_id, [updated_raw])]
        note = f"The updated profile shifts the favored object characteristics from {previous} to {updated}"
    elif suite == "placements":
        previous = placement_phrase(adaptation.get("previous_placement") or {})
        updated_placement = adaptation.get("updated_placement") or metadata.get("preference") or {}
        updated = placement_phrase(updated_placement)
        initial_facts = [placement_tendency(user_id, adaptation.get("previous_placement") or {})]
        updated_facts = [placement_tendency(user_id, updated_placement)]
        note = f"The updated profile changes the spatial tendency from {previous} to {updated}"
    elif suite == "sequences":
        previous = strategy_phrase(adaptation.get("previous_strategy"))
        updated_strategy = adaptation.get("updated_strategy") or metadata.get("sequence_strategy")
        updated = strategy_phrase(updated_strategy)
        sequence = metadata.get("target_sequence") or metadata.get("graspable_objects") or []
        initial_facts = [sequence_tendency(user_id, adaptation.get("previous_strategy"), sequence)]
        updated_facts = [sequence_tendency(user_id, updated_strategy, sequence)]
        note = f"The updated profile changes the ordering tendency from {previous} to {updated}"
    else:
        return None

    initial_metadata = dict(metadata, target_user=user_id)
    updated_metadata = dict(metadata, target_user=user_id)
    return {
        "initial": story_set(initial_metadata, profiles, story_template, override_facts=initial_facts),
        "updated": story_set(updated_metadata, profiles, story_template, override_facts=updated_facts, update_note=note),
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


def validate_text_input_payload(payload: Mapping[str, Any]) -> None:
    variants = payload.get("story_variants")
    if not isinstance(variants, Mapping):
        raise ValueError("GPT payload must contain story_variants")
    validate_variants(variants)
    adaptability = payload.get("adaptability_variants")
    if adaptability is not None:
        if not isinstance(adaptability, Mapping):
            raise ValueError("adaptability_variants must be null or an object")
        validate_variants(adaptability["initial"])
        validate_variants(adaptability["updated"])


def compact_profile_keyword(metadata: Mapping[str, Any]) -> str:
    user = user_text(metadata.get("target_user"))
    suite = metadata.get("suite")
    if suite == "belongings" and metadata.get("target_object"):
        return f"{user}: {object_text(metadata['target_object'])}"
    if suite == "placements":
        preference = metadata.get("preference") or {}
        label = preference.get("label")
        fixed_type = preference.get("fixed_type")
        if label and fixed_type:
            return f"{user}: {object_text(label)} {object_text(fixed_type)}"
    if suite == "sequences" and metadata.get("sequence_strategy"):
        return f"{user}: {object_text(metadata['sequence_strategy'])}"
    return f"{user}: profile cue"


def build_text_input(
    metadata: Mapping[str, Any],
    profiles: Mapping[str, dict[str, Any]],
    story_template: Mapping[str, Any],
) -> dict[str, Any]:
    variants = story_set(metadata, profiles, story_template)
    validate_variants(variants)
    adaptability = adaptability_story_sets(metadata, profiles, story_template)
    if adaptability:
        validate_variants(adaptability["initial"])
        validate_variants(adaptability["updated"])

    return {
        "task_id": metadata.get("episode_id"),
        "evaluation_axis": suite_axis(metadata),
        "user_id": user_text(metadata.get("target_user")),
        "profile_injection_variants": {
            "keyword": compact_profile_keyword(metadata),
            "sentence": variants["one_sentence"],
            "paragraph": variants["eight_sentence"],
        },
        "story_variants": {
            "one_sentence": variants["one_sentence"],
            "four_sentence": variants["four_sentence"],
            "eight_sentence": variants["eight_sentence"],
        },
        "adaptability_variants": adaptability,
    }


def profile_subset(metadata: Mapping[str, Any], profiles: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    user_ids = {user_text(metadata.get("target_user"))}
    user_ids.update(user_text(user) for user in metadata.get("participating_users") or [])
    adaptation = metadata.get("adaptation") or {}
    if adaptation.get("user_id"):
        user_ids.add(user_text(adaptation["user_id"]))
    return {user_id: profile_for(profiles, user_id) for user_id in sorted(user_ids) if profile_for(profiles, user_id)}


def compact_episode_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_id",
        "suite",
        "split",
        "task_type",
        "target_user",
        "participating_users",
        "target_object",
        "target_sequence",
        "ownership",
        "preference",
        "sequence_strategy",
        "fixture",
        "relation",
        "scene",
        "table",
        "adaptation",
    )
    return {key: metadata[key] for key in keys if key in metadata and metadata[key] is not None}


def profile_story_facts(metadata: Mapping[str, Any], profiles: Mapping[str, dict[str, Any]]) -> list[str]:
    user_id = user_text(metadata.get("target_user"))
    facts = list(relevant_facts(metadata))
    extra = irrelevant_fact(metadata, profiles, user_id)
    if extra:
        facts.append(extra)
    facts.append(participation_fact(metadata))
    return facts


def gpt_input_payload(metadata: Mapping[str, Any], profiles: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": metadata.get("episode_id"),
        "suite": metadata.get("suite"),
        "split": metadata.get("split"),
        "evaluation_axis": suite_axis(metadata),
        "target_user": metadata.get("target_user"),
        "suite_description": {
            "belongings": "Describe the user's favored object characteristics inferred from belonging records.",
            "placements": "Describe the user's spatial organization tendency inferred from fixture and region preferences.",
            "sequences": "Describe the user's ordering or scanning tendency inferred from the sequence strategy.",
        }.get(str(metadata.get("suite")), "Use the provided profile facts for the personalized VLAPB task."),
        "injection_scope": (
            "This text is only the personalized injection. The concrete general task input is provided separately, "
            "so do not write an action request."
        ),
        "augmentation_rules": [
            "Turn item ownership into favored characteristics such as material, shape, object category, container form, or tabletop use.",
            "Turn region and fixture preferences into spatial habits such as edge anchoring, lower storage, visible access, or containment.",
            "Turn sequence strategies into ordering habits such as left-to-right scanning, right-to-left clearing, nearest-first cleanup, or alphabetical indexing.",
            "Keep every inferred tendency tied to objects, users, strategies, or regions present in this payload.",
        ],
        "profile_story_facts": profile_story_facts(metadata, profiles),
        "episode_context": compact_episode_context(metadata),
        "profiles": profile_subset(metadata, profiles),
        "required_output": {
            "story_variants": {
                "one_sentence": "exactly 1 sentence",
                "four_sentence": "exactly 4 sentences",
                "eight_sentence": "exactly 8 sentences",
            },
            "adaptability_variants": "null unless metadata contains adaptation, otherwise initial and updated story_variants",
        },
    }


def openai_text_from_response(response: Mapping[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    text = "".join(chunks).strip()
    if not text:
        raise ValueError(f"OpenAI response did not contain output text: {response}")
    return text


def generate_text_input_with_gpt(
    metadata: Mapping[str, Any],
    profiles: Mapping[str, dict[str, Any]],
    *,
    api_key: str,
    model: str,
    max_output_tokens: int,
    request_timeout: float,
    retry_count: int,
    retry_base_seconds: float,
    verbose: bool = False,
) -> dict[str, Any]:
    task_id = metadata.get("episode_id")
    input_payload = gpt_input_payload(metadata, profiles)
    variant_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["one_sentence", "four_sentence", "eight_sentence"],
        "properties": {
            "one_sentence": {"type": "string"},
            "four_sentence": {"type": "string"},
            "eight_sentence": {"type": "string"},
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["task_id", "evaluation_axis", "user_id", "story_variants", "adaptability_variants"],
        "properties": {
            "task_id": {"type": "string"},
            "evaluation_axis": {"type": "string"},
            "user_id": {"type": "string"},
            "story_variants": variant_schema,
            "adaptability_variants": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["initial", "updated"],
                "properties": {
                    "initial": variant_schema,
                    "updated": variant_schema,
                },
            },
        },
    }
    body = {
        "model": model,
        "instructions": (
            "You generate concise augmented personalization text for VLAPB benchmark episodes. "
            "Use only facts present in the input JSON. Do not invent objects, users, fixtures, preferences, or task goals. "
            "Write VLA-friendly injection-level user tendency notes, not the general task input and not robot commands. "
            "Infer favored characteristics from belongings, spatial habits from placement preferences, and scanning or ordering habits from sequence strategies. "
            "Do not use the words put, place, move, pick up, correct action is, or robot should. "
            "Keep the target-relevant cue early, concrete, and easy to parse; avoid storytelling, emotional language, and vague scene filler. "
            "The one_sentence, four_sentence, and eight_sentence variants must imply the same target behavior with increasing context granularity. "
            "For adaptation metadata, produce initial and updated variants; the updated variant must clearly override the initial one."
        ),
        "input": json.dumps(input_payload, indent=2, sort_keys=True),
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "vlapb_granularity_text_input",
                "strict": True,
                "schema": schema,
            }
        },
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            if verbose:
                print(f"[GPT] request {task_id} attempt {attempt + 1}/{retry_count + 1}", flush=True)
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            payload = json.loads(openai_text_from_response(data))
            payload["task_id"] = str(payload.get("task_id") or task_id)
            payload["evaluation_axis"] = str(payload.get("evaluation_axis") or suite_axis(metadata))
            payload["user_id"] = user_text(payload.get("user_id") or metadata.get("target_user"))
            validate_text_input_payload(payload)
            return dict(payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after:
                try:
                    sleep_seconds = float(retry_after)
                except ValueError:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
            else:
                sleep_seconds = retry_base_seconds * (2 ** attempt)
            sleep_seconds = max(sleep_seconds, 0.0)
            if verbose:
                print(f"[GPT] HTTP {exc.code} for {task_id}; retrying in {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            sleep_seconds = max(retry_base_seconds * (2 ** attempt), 0.0)
            if verbose:
                print(
                    f"[GPT] {type(exc).__name__} for {task_id}: {exc}; retrying in {sleep_seconds:.1f}s",
                    flush=True,
                )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"GPT generation failed for {task_id}: {last_error}") from last_error


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
    story_template = load_story_template(args.story_template)
    split_quotas = parse_split_quotas(args.split_quota, args.preset)
    files = limited_metadata_files(
        metadata_files(args.suite_root, args.suites, args.respect_generation_summary),
        args.max_episodes,
        args.max_episodes_per_suite,
        split_quotas,
        args.suite_root,
    )
    expected_outputs = {output_path_for(path) for path in files}
    counts: Counter[tuple[str, str]] = Counter()
    written = 0
    removed_stale = 0
    gpt_failures = 0
    api_key = resolve_openai_api_key(args.openai_api_key_env, args.openai_api_key_file)
    capped_run = args.max_episodes is not None or args.max_episodes_per_suite is not None or bool(split_quotas)
    if args.generation_mode == "gpt" and not args.dry_run and not capped_run and not args.allow_full_gpt:
        raise RuntimeError(
            "Refusing uncapped GPT generation because it can be slow and expensive. "
            "Use --preset full-designed for the 16-episode designed split set, "
            "add --max-episodes/--max-episodes-per-suite/--split-quota, or pass --allow-full-gpt intentionally."
        )
    if args.generation_mode == "gpt" and not args.dry_run and not api_key:
        raise RuntimeError(
            f"--generation-mode gpt requires ${args.openai_api_key_env} "
            f"or --openai-api-key-file {args.openai_api_key_file}"
        )

    if not args.dry_run:
        write_prompt_html(args.prompt_html)
    if args.verbose:
        print(f"[INFO] found {len(files)} metadata episode(s)", flush=True)
        print(f"[INFO] overwrite: {args.overwrite}", flush=True)

    iterator = tqdm(files, desc="Generating story text inputs", unit="episode") if tqdm else files
    for metadata_path in iterator:
        metadata = read_json(metadata_path)
        out_path = output_path_for(metadata_path)
        counts[(str(metadata.get("suite", "unknown")), str(metadata.get("split", "unknown")))] += 1
        if args.dry_run:
            continue
        if out_path.exists() and not args.overwrite:
            if args.verbose:
                print(f"[SKIP] {metadata.get('episode_id')} exists: {out_path}", flush=True)
            continue
        if args.generation_mode == "gpt":
            if args.verbose:
                print(f"[GPT] generating {metadata.get('episode_id')} -> {out_path}", flush=True)
            try:
                payload = generate_text_input_with_gpt(
                    metadata,
                    profiles,
                    api_key=str(api_key),
                    model=args.openai_model,
                    max_output_tokens=args.openai_max_output_tokens,
                    request_timeout=args.openai_timeout,
                    retry_count=args.openai_retries,
                    retry_base_seconds=args.openai_retry_base_seconds,
                    verbose=args.verbose,
                )
            except RuntimeError:
                if not args.gpt_fallback_template:
                    raise
                gpt_failures += 1
                payload = build_text_input(metadata, profiles, story_template)
                payload["generation_fallback"] = "template_after_gpt_failure"
                if args.verbose:
                    print(f"[FALLBACK] template output for {metadata.get('episode_id')}", flush=True)
        else:
            payload = build_text_input(metadata, profiles, story_template)
        write_json(out_path, payload)
        written += 1
        if args.verbose:
            print(f"[WRITE] {out_path}", flush=True)
        if args.generation_mode == "gpt" and args.openai_sleep_between_requests > 0:
            if args.verbose:
                print(f"[WAIT] sleeping {args.openai_sleep_between_requests:.1f}s", flush=True)
            time.sleep(args.openai_sleep_between_requests)

    should_clean_stale = args.clean_stale and (not capped_run or args.clean_stale_with_max_episodes)
    if not args.dry_run and should_clean_stale:
        for suite in args.suites:
            for text_path in (args.suite_root / suite).glob("*/text_inputs/*.json"):
                if text_path not in expected_outputs:
                    text_path.unlink()
                    removed_stale += 1
    elif args.verbose and args.clean_stale and capped_run:
        print("[INFO] skipped stale cleanup because an episode cap is set", flush=True)

    distribution: dict[str, dict[str, int]] = defaultdict(dict)
    for (suite, split), count in sorted(counts.items()):
        distribution[suite][split] = count

    summary = {
        "suite_root": str(args.suite_root),
        "profiles": str(args.profiles),
        "story_template": str(args.story_template),
        "prompt_html": str(args.prompt_html),
        "total_metadata": len(files),
        "written": 0 if args.dry_run else written,
        "removed_stale": 0 if args.dry_run else removed_stale,
        "dry_run": args.dry_run,
        "generation_mode": args.generation_mode,
        "openai_model": args.openai_model if args.generation_mode == "gpt" else None,
        "openai_api_key_file": str(args.openai_api_key_file) if args.generation_mode == "gpt" else None,
        "gpt_failures": gpt_failures,
        "preset": args.preset,
        "max_episodes": args.max_episodes,
        "max_episodes_per_suite": args.max_episodes_per_suite,
        "split_quotas": {f"{suite}:{split}": count for (suite, split), count in sorted(split_quotas.items())},
        "overwrite": args.overwrite,
        "clean_stale": args.clean_stale,
        "clean_stale_with_max_episodes": args.clean_stale_with_max_episodes,
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
    parser.add_argument("--story-template", type=Path, default=DEFAULT_STORY_TEMPLATE)
    parser.add_argument("--prompt-html", type=Path, default=DEFAULT_PROMPT_HTML)
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    parser.add_argument("--generation-mode", choices=("template", "gpt"), default="template")
    parser.add_argument(
        "--preset",
        choices=("full-designed",),
        default=None,
        help="Use a named episode selection preset. full-designed selects one episode from every designed split.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Process only the first N metadata episodes.")
    parser.add_argument(
        "--max-episodes-per-suite",
        type=int,
        default=None,
        help="Process only the first N metadata episodes for each selected suite.",
    )
    parser.add_argument(
        "--split-quota",
        action="append",
        default=None,
        help="Process N episodes from one designed split, formatted as suite:split=count. May be repeated.",
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--openai-api-key-file",
        type=Path,
        default=DEFAULT_OPENAI_API_KEY_FILE,
        help="Local ignored file containing the OpenAI API key, used if the env var is unset.",
    )
    parser.add_argument("--openai-max-output-tokens", type=int, default=1200)
    parser.add_argument("--openai-timeout", type=float, default=60.0)
    parser.add_argument("--openai-retries", type=int, default=5)
    parser.add_argument("--openai-retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--openai-sleep-between-requests", type=float, default=0.0)
    parser.add_argument(
        "--allow-full-gpt",
        action="store_true",
        help="Allow uncapped GPT generation over every selected metadata episode.",
    )
    parser.add_argument(
        "--gpt-fallback-template",
        action="store_true",
        help="If a GPT request fails, write the deterministic template output for that episode instead of stopping.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--clean-stale", action="store_true", default=True)
    parser.add_argument("--no-clean-stale", dest="clean_stale", action="store_false")
    parser.add_argument(
        "--clean-stale-with-max-episodes",
        action="store_true",
        help="Allow stale text-input cleanup during capped --max-episodes runs.",
    )
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
    print(f"[INFO] story template: {summary['story_template']}")
    print(f"[INFO] prompt html: {summary['prompt_html']}")
    print(f"[INFO] generation mode: {summary['generation_mode']}")
    if summary["openai_model"]:
        print(f"[INFO] OpenAI model: {summary['openai_model']}")
    if summary["openai_api_key_file"]:
        print(f"[INFO] OpenAI API key file: {summary['openai_api_key_file']}")
    print(f"[INFO] metadata episodes: {summary['total_metadata']}")
    print(f"[INFO] distribution: {summary['distribution']}")
    if summary["dry_run"]:
        print("[INFO] dry-run: no files written")
    else:
        print(f"[INFO] written text inputs: {summary['written']}")
        if summary["gpt_failures"]:
            print(f"[INFO] GPT failures with template fallback: {summary['gpt_failures']}")
        print(f"[INFO] removed stale text inputs: {summary['removed_stale']}")
        print(f"[INFO] summary: {args.suite_root / 'granularity_text_input_summary.json'}")


if __name__ == "__main__":
    main()
