#!/usr/bin/env python3
"""Generate VLAPB belongings-suite BDDL and episode metadata.

Input/output:
  Read all user profiles from profiles.json and write episode-level .bddl
  plus readable metadata .json files under VLAPB_suites/belongings.

Generation:
  Use every available user and object from profiles.json to generate type1,
  type2, type3, adaptability, and multi-user belongings episodes.
  Each episode contains exactly four graspable objects and one fixed object.

Hyperparameters:
  Fixtures, table/scene variants, spawn radius, random seed, output root, and
  adaptability overlap limit are configurable through CLI arguments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tiny fallback for minimal envs
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable: Iterable[Any] | None = None, **_: Any) -> None:
            self.iterable = iterable

        def __iter__(self) -> Iterator[Any]:
            return iter(self.iterable or ())

        def update(self, _count: int = 1) -> None:
            return None

        def close(self) -> None:
            return None


SCRIPT_DIR = Path(__file__).resolve().parent
VLAPB_LIBERO_ROOT = SCRIPT_DIR.parents[1]
VLAPB_PROJECT_ROOT = VLAPB_LIBERO_ROOT.parent
DEFAULT_PROFILES_PATH = VLAPB_LIBERO_ROOT / "profiles" / "profiles.json"
DEFAULT_OUTPUT_ROOT = VLAPB_PROJECT_ROOT / "VLAPB_suites"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "VLAPB_config.yaml"
DEFAULT_PAIR_STATS_PATH = VLAPB_LIBERO_ROOT / "docs" / "libero_objects_summary.md"

FIXTURE_RELATIONS = {
    "basket": "In",
    "wooden_tray": "In",
    "flat_stove": "On",
    "plate": "On",
}
DEFAULT_FIXTURES = tuple(FIXTURE_RELATIONS)
DEFAULT_SCENE_VARIANTS = (
    ("KITCHEN_SCENE1", "kitchen_table"),
    ("KITCHEN_SCENE5", "kitchen_table"),
    ("LIVING_ROOM_SCENE1", "living_room_table"),
    ("LIVING_ROOM_SCENE6", "living_room_table"),
)
OBJECTS_PER_EPISODE = 4


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    belongings: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class SceneVariant:
    scene: str
    table: str


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    suite: str
    split: str
    task_type: str
    scene: str
    table: str
    fixture: str
    relation: str
    target_user: str
    target_object: str
    graspable_objects: tuple[str, ...]
    ownership: dict[str, list[str]]
    participating_users: tuple[str, ...]
    adaptation: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pair-stats", type=Path, default=DEFAULT_PAIR_STATS_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixtures", nargs="+", default=list(DEFAULT_FIXTURES))
    parser.add_argument("--spawn-radius", type=float, default=0.24)
    parser.add_argument("--object-box-size", type=float, default=0.025)
    parser.add_argument("--fixture-box-size", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adaptability-overlap-limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", help="Reserved for future rendering/simulation checks.")
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="Optional debugging cap. By default, generate every possible episode.",
    )
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logging.warning("config file not found: %s; using CLI/default generation limits", path)
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read VLAPB_config.yaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def get_nested_config(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def read_pair_stats(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        logging.warning("pair stats file not found: %s; using unweighted episode order", path)
        return {}
    text = path.read_text(encoding="utf-8")
    stats: dict[str, dict[str, int]] = {}
    line_re = re.compile(r"^- `([^`]+)`: (.+)$", re.MULTILINE)
    item_re = re.compile(r"`([^`]+)`\((\d+)\)")
    for fixture, payload in line_re.findall(text):
        parsed = {obj: int(count) for obj, count in item_re.findall(payload)}
        if parsed:
            stats[fixture] = parsed
    return stats


def pair_weight(pair_stats: dict[str, dict[str, int]], fixture: str, object_type: str) -> int:
    return pair_stats.get(fixture, {}).get(object_type, 0)


def global_pair_weight(pair_stats: dict[str, dict[str, int]], object_type: str) -> int:
    return sum(stats.get(object_type, 0) for stats in pair_stats.values())


def ordered_user_object_pairs(
    profiles: list[UserProfile],
    fixture: str,
    pair_stats: dict[str, dict[str, int]],
) -> list[tuple[UserProfile, str]]:
    pairs = [(profile, item) for profile in profiles for item in profile.belongings]
    buckets: dict[int, list[tuple[UserProfile, str]]] = defaultdict(list)
    for pair in pairs:
        buckets[global_pair_weight(pair_stats, pair[1])].append(pair)
    for bucket in buckets.values():
        bucket.sort(key=lambda pair: (pair[0].user_id, pair[1]))

    # Mix unseen, rare, and common pairs instead of front-loading frequent ones.
    ordered = []
    weights = sorted(buckets)
    while any(buckets.values()):
        for weight in weights:
            if buckets[weight]:
                ordered.append(buckets[weight].pop(0))
    return ordered


def ordered_items_for_fixture(
    items: Iterable[str],
    fixture: str,
    pair_stats: dict[str, dict[str, int]],
) -> list[str]:
    buckets: dict[int, list[str]] = defaultdict(list)
    for item in sorted(set(items)):
        buckets[global_pair_weight(pair_stats, item)].append(item)
    ordered = []
    weights = sorted(buckets)
    while any(buckets.values()):
        for weight in weights:
            if buckets[weight]:
                ordered.append(buckets[weight].pop(0))
    return ordered


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_profiles(path: Path) -> list[UserProfile]:
    data = read_json(path)
    profiles = []
    for raw_profile in data.get("profiles", []):
        user_id = str(raw_profile["user_id"])
        belongings = tuple(str(item) for item in raw_profile.get("belongings", []))
        if belongings:
            profiles.append(UserProfile(user_id=user_id, belongings=belongings, raw=raw_profile))
    if not profiles:
        raise ValueError(f"No profiles with belongings found in {path}")
    return profiles


def all_profile_objects(profiles: list[UserProfile]) -> tuple[str, ...]:
    objects = []
    seen = set()
    for profile in profiles:
        for item in profile.belongings:
            if item not in seen:
                seen.add(item)
                objects.append(item)
    return tuple(objects)


def ownership_index(profiles: list[UserProfile]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for profile in profiles:
        for item in profile.belongings:
            index[item].append(profile.user_id)
    return {key: sorted(value) for key, value in sorted(index.items())}


def choose_scene_variant(rng: random.Random) -> SceneVariant:
    scene, table = rng.choice(DEFAULT_SCENE_VARIANTS)
    return SceneVariant(scene=scene, table=table)


def choose_fixture(fixtures: tuple[str, ...], rng: random.Random) -> str:
    return rng.choice(fixtures)


def unique_combinations(values: Iterable[str], size: int) -> Iterator[tuple[str, ...]]:
    yield from itertools.combinations(sorted(set(values)), size)


def non_user_objects(user: UserProfile, object_pool: tuple[str, ...]) -> tuple[str, ...]:
    belongings = set(user.belongings)
    return tuple(item for item in object_pool if item not in belongings)


def make_ownership(objects_by_user: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    return {
        user_id: sorted(set(objects))
        for user_id, objects in sorted(objects_by_user.items())
        if set(objects)
    }


def count_iterable(factory: Any) -> int:
    return sum(1 for _ in factory())


def count_unique_assignments(users: tuple[UserProfile, ...]) -> int:
    """Count ways to choose one unique belonging for each user."""
    memo: dict[tuple[int, tuple[str, ...]], int] = {}

    def visit(index: int, used: frozenset[str]) -> int:
        key = (index, tuple(sorted(used)))
        if key in memo:
            return memo[key]
        if index == len(users):
            return 1
        total = 0
        for item in users[index].belongings:
            if item not in used:
                total += visit(index + 1, used | {item})
        memo[key] = total
        return total

    return visit(0, frozenset())


def count_type1_specs(profiles: list[UserProfile], object_pool: tuple[str, ...], fixtures: tuple[str, ...]) -> int:
    total = 0
    for user in profiles:
        distractor_count = len(non_user_objects(user, object_pool))
        total += len(user.belongings) * math.comb(distractor_count, 3) * len(fixtures)
    return total


def count_type2_specs(profiles: list[UserProfile], fixtures: tuple[str, ...]) -> int:
    total = 0
    for users in itertools.combinations(profiles, 4):
        total += count_unique_assignments(tuple(users)) * len(users) * len(fixtures)
    return total


def count_type3_specs(profiles: list[UserProfile], fixtures: tuple[str, ...]) -> int:
    total = 0
    for target_user in profiles:
        others = [profile for profile in profiles if profile.user_id != target_user.user_id]
        for other_users in itertools.combinations(others, 3):
            total += count_unique_assignments((target_user, *other_users)) * len(fixtures)
    return total


def count_adaptability_specs(
    profiles: list[UserProfile],
    object_pool: tuple[str, ...],
    object_owners: dict[str, list[str]],
    fixtures: tuple[str, ...],
    overlap_limit: int,
) -> int:
    total = 0
    for user in profiles:
        original_belongings = set(user.belongings)
        candidates = [
            item
            for item in object_pool
            if item not in original_belongings
            and len([owner for owner in object_owners.get(item, []) if owner != user.user_id]) <= overlap_limit
        ]
        for new_object in candidates:
            distractor_count = len([item for item in object_pool if item != new_object and item not in original_belongings])
            total += len(user.belongings) * math.comb(distractor_count, 3) * len(fixtures)
    return total


def count_multiuser_specs(profiles: list[UserProfile], object_pool: tuple[str, ...], fixtures: tuple[str, ...]) -> int:
    total = 0
    for user_a, user_b in itertools.combinations(profiles, 2):
        excluded = set(user_a.belongings) | set(user_b.belongings)
        distractor_count = len([item for item in object_pool if item not in excluded])
        for target_user, other_user in ((user_a, user_b), (user_b, user_a)):
            unique_pair_count = sum(
                1
                for target_object in target_user.belongings
                for other_object in other_user.belongings
                if target_object != other_object
            )
            total += unique_pair_count * math.comb(distractor_count, 2) * len(fixtures)
    for users in itertools.permutations(profiles, 4):
        total += count_unique_assignments(tuple(users)) * len(fixtures)
    return total


def distribute_episode_budget(
    max_counts: dict[str, int],
    total_budget: int | None,
    split_order: tuple[str, ...],
) -> dict[str, int]:
    if total_budget is None:
        return dict(max_counts)
    if total_budget < 0:
        raise ValueError(f"Episode budget must be non-negative, got {total_budget}")
    base, remainder = divmod(total_budget, len(split_order))
    planned = {}
    for index, split in enumerate(split_order):
        requested = base + (1 if index < remainder else 0)
        planned[split] = min(requested, max_counts.get(split, 0))
    return planned


def build_type1_specs(
    profiles: list[UserProfile],
    object_pool: tuple[str, ...],
    fixtures: tuple[str, ...],
    pair_stats: dict[str, dict[str, int]],
    rng: random.Random,
) -> Iterator[EpisodeSpec]:
    index = 1
    pair_orders = {fixture: ordered_user_object_pairs(profiles, fixture, pair_stats) for fixture in fixtures}
    max_pair_count = max(len(pairs) for pairs in pair_orders.values())
    max_distractor_count = max(
        math.comb(len(non_user_objects(user, object_pool)), 3)
        for pairs in pair_orders.values()
        for user, _target_object in pairs
    )
    for distractor_rank in range(max_distractor_count):
        for pair_rank in range(max_pair_count):
            for fixture in fixtures:
                pairs = pair_orders[fixture]
                if pair_rank >= len(pairs):
                    continue
                user, target_object = pairs[pair_rank]
                distractors_list = list(unique_combinations(non_user_objects(user, object_pool), 3))
                if distractor_rank >= len(distractors_list):
                    continue
                distractors = distractors_list[distractor_rank]
                scene = choose_scene_variant(rng)
                objects = (target_object, *distractors)
                yield make_episode(
                    index=index,
                    split="type1",
                    task_type="type1",
                    scene=scene,
                    fixture=fixture,
                    target_user=user.user_id,
                    target_object=target_object,
                    graspable_objects=objects,
                    ownership=make_ownership({user.user_id: [target_object]}),
                    participating_users=(user.user_id,),
                )
                index += 1


def build_type2_specs(
    profiles: list[UserProfile],
    fixtures: tuple[str, ...],
    pair_stats: dict[str, dict[str, int]],
    rng: random.Random,
) -> Iterator[EpisodeSpec]:
    index = 1
    for users in itertools.combinations(profiles, 4):
        target_orders = {}
        for fixture in fixtures:
            user_ids = {user.user_id for user in users}
            target_orders[fixture] = [pair for pair in ordered_user_object_pairs(profiles, fixture, pair_stats) if pair[0].user_id in user_ids]
        max_target_count = max(len(pairs) for pairs in target_orders.values())
        max_other_count = max(
                len(list(itertools.product(*[user.belongings for user in users if user.user_id != target_orders[fixture][target_rank][0].user_id])))
                for fixture in fixtures
                for target_rank in range(max_target_count)
                if target_rank < len(target_orders[fixture])
            )
        for other_rank in range(max_other_count):
            for target_rank in range(max_target_count):
                for fixture in fixtures:
                    target_pairs = target_orders[fixture]
                    if target_rank >= len(target_pairs):
                        continue
                    target_user, target_object = target_pairs[target_rank]
                    other_users = [user for user in users if user.user_id != target_user.user_id]
                    other_products = list(itertools.product(*[user.belongings for user in other_users]))
                    if other_rank >= len(other_products):
                        continue
                    other_objects = other_products[other_rank]
                    objects = (target_object, *other_objects)
                    if len(set(objects)) != OBJECTS_PER_EPISODE:
                        continue
                    ownership = make_ownership(
                        {
                            target_user.user_id: [target_object],
                            **{user.user_id: [obj] for user, obj in zip(other_users, other_objects)},
                        }
                    )
                    scene = choose_scene_variant(rng)
                    yield make_episode(
                        index=index,
                        split="type2",
                        task_type="type2",
                        scene=scene,
                        fixture=fixture,
                        target_user=target_user.user_id,
                        target_object=target_object,
                        graspable_objects=objects,
                        ownership=ownership,
                        participating_users=tuple(user.user_id for user in users),
                    )
                    index += 1


def build_type3_specs(
    profiles: list[UserProfile],
    fixtures: tuple[str, ...],
    pair_stats: dict[str, dict[str, int]],
    rng: random.Random,
) -> Iterator[EpisodeSpec]:
    index = 1
    for users in itertools.combinations(profiles, 4):
        target_orders = {}
        user_ids = {user.user_id for user in users}
        for fixture in fixtures:
            target_orders[fixture] = [pair for pair in ordered_user_object_pairs(profiles, fixture, pair_stats) if pair[0].user_id in user_ids]
        max_target_count = max(len(pairs) for pairs in target_orders.values())
        max_other_count = max(
            len(list(itertools.product(*[user.belongings for user in users if user.user_id != target_orders[fixture][target_rank][0].user_id])))
            for fixture in fixtures
            for target_rank in range(max_target_count)
            if target_rank < len(target_orders[fixture])
        )
        for other_rank in range(max_other_count):
            for target_rank in range(max_target_count):
                for fixture in fixtures:
                    target_pairs = target_orders[fixture]
                    if target_rank >= len(target_pairs):
                        continue
                    target_user, target_object = target_pairs[target_rank]
                    other_users = [user for user in users if user.user_id != target_user.user_id]
                    other_products = list(itertools.product(*[user.belongings for user in other_users]))
                    if other_rank >= len(other_products):
                        continue
                    other_objects = other_products[other_rank]
                    objects = (target_object, *other_objects)
                    if len(set(objects)) != OBJECTS_PER_EPISODE:
                        continue
                    ownership = make_ownership(
                        {
                            target_user.user_id: [target_object],
                            **{user.user_id: [obj] for user, obj in zip(other_users, other_objects)},
                        }
                    )
                    scene = choose_scene_variant(rng)
                    yield make_episode(
                        index=index,
                        split="type3",
                        task_type="type3",
                        scene=scene,
                        fixture=fixture,
                        target_user=target_user.user_id,
                        target_object=target_object,
                        graspable_objects=objects,
                        ownership=ownership,
                        participating_users=(target_user.user_id, *[u.user_id for u in other_users]),
                    )
                    index += 1


def build_adaptability_specs(
    profiles: list[UserProfile],
    object_pool: tuple[str, ...],
    object_owners: dict[str, list[str]],
    fixtures: tuple[str, ...],
    pair_stats: dict[str, dict[str, int]],
    overlap_limit: int,
    rng: random.Random,
) -> Iterator[EpisodeSpec]:
    index = 1
    for user in profiles:
        original_belongings = set(user.belongings)
        for old_object in user.belongings:
            for distractor_rank in range(math.comb(max(len(object_pool) - len(original_belongings) - 1, 0), 3)):
                for candidate_rank in range(len(object_pool)):
                    for fixture in fixtures:
                        candidates = [
                            item
                            for item in object_pool
                            if item not in original_belongings
                            and len([owner for owner in object_owners.get(item, []) if owner != user.user_id]) <= overlap_limit
                        ]
                        candidates = ordered_items_for_fixture(candidates, fixture, pair_stats)
                        if candidate_rank >= len(candidates):
                            continue
                        new_object = candidates[candidate_rank]
                        distractors_list = list(unique_combinations([item for item in object_pool if item != new_object and item not in original_belongings], 3))
                        if distractor_rank >= len(distractors_list):
                            continue
                        distractors = distractors_list[distractor_rank]
                        objects = (new_object, *distractors)
                        if len(set(objects)) != OBJECTS_PER_EPISODE:
                            continue
                        scene = choose_scene_variant(rng)
                        adaptation = {
                            "episode_number": index,
                            "user_id": user.user_id,
                            "previous_belonging": old_object,
                            "updated_belonging": new_object,
                            "overlapping_other_users": [
                                owner for owner in object_owners.get(new_object, []) if owner != user.user_id
                            ],
                        }
                        yield make_episode(
                            index=index,
                            split="adaptability",
                            task_type="adaptability",
                            scene=scene,
                            fixture=fixture,
                            target_user=user.user_id,
                            target_object=new_object,
                            graspable_objects=objects,
                            ownership=make_ownership({user.user_id: [new_object]}),
                            participating_users=(user.user_id,),
                            adaptation=adaptation,
                        )
                        index += 1


def build_multiuser_specs(
    profiles: list[UserProfile],
    object_pool: tuple[str, ...],
    fixtures: tuple[str, ...],
    pair_stats: dict[str, dict[str, int]],
    rng: random.Random,
) -> Iterator[EpisodeSpec]:
    index = 1
    pair_orders = {fixture: ordered_user_object_pairs(profiles, fixture, pair_stats) for fixture in fixtures}
    max_pair_count = max(len(pairs) for pairs in pair_orders.values())
    for other_user in profiles:
        for other_object in other_user.belongings:
            for distractor_rank in range(math.comb(len(object_pool), 2)):
                for pair_rank in range(max_pair_count):
                    for fixture in fixtures:
                        pairs = pair_orders[fixture]
                        if pair_rank >= len(pairs):
                            continue
                        target_user, target_object = pairs[pair_rank]
                        if other_user.user_id == target_user.user_id or other_object == target_object:
                            continue
                        users = (target_user, other_user)
                        excluded = set(target_user.belongings) | set(other_user.belongings)
                        distractors_list = list(unique_combinations([item for item in object_pool if item not in excluded], 2))
                        if distractor_rank >= len(distractors_list):
                            continue
                        distractors = distractors_list[distractor_rank]
                        objects = (target_object, other_object, *distractors)
                        if len(set(objects)) != OBJECTS_PER_EPISODE:
                            continue
                        scene = choose_scene_variant(rng)
                        yield make_episode(
                            index=index,
                            split="multiuser",
                            task_type="multiuser_type1",
                            scene=scene,
                            fixture=fixture,
                            target_user=target_user.user_id,
                            target_object=target_object,
                            graspable_objects=objects,
                            ownership=make_ownership(
                                {
                                    target_user.user_id: [target_object],
                                    other_user.user_id: [other_object],
                                }
                            ),
                            participating_users=tuple(user.user_id for user in users),
                        )
                        index += 1
        for fixture in fixtures:
            pairs = pair_orders[fixture]
            if pair_rank >= len(pairs):
                continue
            target_user, target_object = pairs[pair_rank]
            other_profiles = [profile for profile in profiles if profile.user_id != target_user.user_id]
            for partner_user, other_user_a, other_user_b in itertools.permutations(other_profiles, 3):
                for partner_object in partner_user.belongings:
                    for other_object_a in other_user_a.belongings:
                        for other_object_b in other_user_b.belongings:
                            objects = (target_object, partner_object, other_object_a, other_object_b)
                            if len(set(objects)) != OBJECTS_PER_EPISODE:
                                continue
                            scene = choose_scene_variant(rng)
                            yield make_episode(
                                index=index,
                                split="multiuser",
                                task_type="multiuser_type2",
                                scene=scene,
                                fixture=fixture,
                                target_user=target_user.user_id,
                                target_object=target_object,
                                graspable_objects=objects,
                                ownership=make_ownership(
                                    {
                                        target_user.user_id: [target_object],
                                        partner_user.user_id: [partner_object],
                                        other_user_a.user_id: [other_object_a],
                                        other_user_b.user_id: [other_object_b],
                                    }
                                ),
                                participating_users=(
                                    target_user.user_id,
                                    partner_user.user_id,
                                    other_user_a.user_id,
                                    other_user_b.user_id,
                                ),
                            )
                            index += 1


def make_episode(
    index: int,
    split: str,
    task_type: str,
    scene: SceneVariant,
    fixture: str,
    target_user: str,
    target_object: str,
    graspable_objects: tuple[str, ...],
    ownership: dict[str, list[str]],
    participating_users: tuple[str, ...],
    adaptation: dict[str, Any] | None = None,
) -> EpisodeSpec:
    episode_id = f"belongings_{split}_{index:06d}"
    return EpisodeSpec(
        episode_id=episode_id,
        suite="belongings",
        split=split,
        task_type=task_type,
        scene=scene.scene,
        table=scene.table,
        fixture=fixture,
        relation=FIXTURE_RELATIONS[fixture],
        target_user=target_user,
        target_object=target_object,
        graspable_objects=tuple(graspable_objects),
        ownership=ownership,
        participating_users=tuple(participating_users),
        adaptation=adaptation,
    )


def object_ids(object_types: tuple[str, ...]) -> list[tuple[str, str]]:
    counts: Counter[str] = Counter()
    ids = []
    for object_type in object_types:
        counts[object_type] += 1
        ids.append((f"{object_type}_{counts[object_type]}", object_type))
    return ids


def grouped_declarations(id_type_pairs: list[tuple[str, str]]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for object_id, object_type in id_type_pairs:
        grouped[object_type].append(object_id)
    return [f"    {' '.join(ids)} - {object_type}" for object_type, ids in sorted(grouped.items())]


def square_range(center_x: float, center_y: float, half_size: float) -> tuple[float, float, float, float]:
    return (
        round(center_x - half_size, 4),
        round(center_y - half_size, 4),
        round(center_x + half_size, 4),
        round(center_y + half_size, 4),
    )


def circular_spawn_centers(
    fixture: str,
    count: int,
    radius: float,
    rng: random.Random,
) -> list[tuple[float, float]]:
    if count != OBJECTS_PER_EPISODE:
        raise ValueError(f"Expected {OBJECTS_PER_EPISODE} graspable objects, got {count}")
    base_angles = [math.radians(angle) for angle in (35, 125, 215, 305)]
    if fixture == "basket":
        # Avoid the back side where the high basket neck can occlude objects.
        base_angles = [math.radians(angle) for angle in (25, 90, 155, 270)]
    rotation = rng.uniform(-0.18, 0.18)
    centers = []
    for angle in base_angles:
        jittered_radius = radius * rng.uniform(0.82, 1.08)
        x = jittered_radius * math.cos(angle + rotation)
        y = jittered_radius * math.sin(angle + rotation)
        centers.append((round(x, 4), round(y, 4)))
    return centers


def fixture_region_name(fixture: str) -> str:
    return f"{fixture}_region"


def goal_region_name(fixture: str) -> str:
    if FIXTURE_RELATIONS[fixture] == "In":
        return "contain_region"
    if fixture == "flat_stove":
        return "cook_region"
    return "top_region"


def render_region(name: str, target: str, ranges: tuple[float, float, float, float] | None = None) -> str:
    if ranges is None:
        return f"""      ({name}
          (:target {target})
      )"""
    x1, y1, x2, y2 = ranges
    return f"""      ({name}
          (:target {target})
          (:ranges (
              ({x1} {y1} {x2} {y2})
            )
          )
      )"""


def language_for(spec: EpisodeSpec) -> str:
    preposition = "in" if spec.relation == "In" else "on"
    return (
        f"Pick the {spec.target_object} belonging to {spec.target_user} "
        f"and place it {preposition} the {spec.fixture}"
    )


def render_bddl(spec: EpisodeSpec, spawn_radius: float, object_box_size: float, fixture_box_size: float) -> str:
    rng = random.Random(f"{spec.episode_id}:{spec.scene}:{spec.fixture}")
    object_id_pairs = object_ids(spec.graspable_objects)
    target_object_id = object_id_pairs[0][0]
    fixture_id = f"{spec.fixture}_1"
    fixture_region = fixture_region_name(spec.fixture)
    goal_region = goal_region_name(spec.fixture)
    spawn_centers = circular_spawn_centers(spec.fixture, len(object_id_pairs), spawn_radius, rng)

    regions = [render_region(fixture_region, spec.table, square_range(0.0, 0.0, fixture_box_size))]
    for idx, (center_x, center_y) in enumerate(spawn_centers):
        regions.append(render_region(f"object_init_region_{idx}", spec.table, square_range(center_x, center_y, object_box_size)))
    regions.append(render_region(goal_region, fixture_id))

    fixture_lines = [f"    {spec.table} - table"]
    object_lines = grouped_declarations(object_id_pairs)
    fixture_lines.append(f"    {fixture_id} - {spec.fixture}")

    init_lines = []
    init_lines.append(f"    (On {fixture_id} {spec.table}_{fixture_region})")
    for idx, (object_id, _object_type) in enumerate(object_id_pairs):
        init_lines.append(f"    (On {object_id} {spec.table}_object_init_region_{idx})")

    if spec.relation == "In" or spec.fixture == "flat_stove":
        goal_target = f"{fixture_id}_{goal_region}"
    else:
        goal_target = fixture_id
    goal_line = f"    (And ({spec.relation} {target_object_id} {goal_target}))"

    return f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language_for(spec)})
    (:regions
{chr(10).join(regions)}
    )

  (:fixtures
{chr(10).join(fixture_lines)}
  )

  (:objects
{chr(10).join(object_lines)}
  )

  (:obj_of_interest
    {target_object_id}
  )

  (:init
{chr(10).join(init_lines)}
  )

  (:goal
{goal_line}
  )

)
"""


def metadata_for(spec: EpisodeSpec, profiles_path: Path, bddl_path: Path, spawn_radius: float) -> dict[str, Any]:
    return {
        **asdict(spec),
        "language": language_for(spec),
        "profile_source": str(profiles_path),
        "bddl_file": str(bddl_path),
        "constraints": {
            "graspable_object_count": OBJECTS_PER_EPISODE,
            "spawn_policy": "randomized circle around the fixed object; basket avoids back-side occlusion",
            "spawn_radius": spawn_radius,
            "fixture_relation": {
                "basket": "In",
                "wooden_tray": "In",
                "flat_stove": "On",
                "plate": "On",
            },
        },
    }


def output_paths(output_root: Path, spec: EpisodeSpec) -> tuple[Path, Path]:
    base = output_root / "belongings" / spec.split
    return (
        base / "bddl_files" / f"{spec.episode_id}.bddl",
        base / "metadata" / f"{spec.episode_id}.json",
    )


def build_factories(
    profiles: list[UserProfile],
    object_pool: tuple[str, ...],
    object_owners: dict[str, list[str]],
    fixtures: tuple[str, ...],
    pair_stats: dict[str, dict[str, int]],
    overlap_limit: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "type1": lambda: build_type1_specs(profiles, object_pool, fixtures, pair_stats, random.Random(seed + 11)),
        "type2": lambda: build_type2_specs(profiles, fixtures, pair_stats, random.Random(seed + 22)),
        "type3": lambda: build_type3_specs(profiles, fixtures, pair_stats, random.Random(seed + 33)),
        "adaptability": lambda: build_adaptability_specs(
            profiles,
            object_pool,
            object_owners,
            fixtures,
            pair_stats,
            overlap_limit,
            random.Random(seed + 44),
        ),
        "multiuser": lambda: build_multiuser_specs(profiles, object_pool, fixtures, pair_stats, random.Random(seed + 55)),
    }


def validate_fixtures(fixtures: list[str]) -> tuple[str, ...]:
    unknown = sorted(set(fixtures) - set(FIXTURE_RELATIONS))
    if unknown:
        raise ValueError(f"Unsupported fixtures: {unknown}. Valid fixtures: {sorted(FIXTURE_RELATIONS)}")
    return tuple(fixtures)


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    config = read_config(args.config)
    pair_stats = read_pair_stats(args.pair_stats)
    fixtures = validate_fixtures(args.fixtures)
    profiles = load_profiles(args.profiles)
    object_pool = all_profile_objects(profiles)
    object_owners = ownership_index(profiles)
    factories = build_factories(
        profiles=profiles,
        object_pool=object_pool,
        object_owners=object_owners,
        fixtures=fixtures,
        pair_stats=pair_stats,
        overlap_limit=args.adaptability_overlap_limit,
        seed=args.seed,
    )

    logging.info("profiles: %d users", len(profiles))
    logging.info("object pool: %d unique objects from profiles.json", len(object_pool))
    logging.info("fixtures: %s", ", ".join(fixtures))
    logging.info("output root: %s", args.output_root)
    logging.info("seed: %d | spawn_radius: %.3f | gpu: %s", args.seed, args.spawn_radius, args.use_gpu)
    logging.info("pair stats: %s", args.pair_stats)
    logging.info("weighted fixtures: %s", {fixture: len(pair_stats.get(fixture, {})) for fixture in fixtures})

    counts = {
        "type1": count_type1_specs(profiles, object_pool, fixtures),
        "type2": count_type2_specs(profiles, fixtures),
        "type3": count_type3_specs(profiles, fixtures),
        "adaptability": count_adaptability_specs(
            profiles,
            object_pool,
            object_owners,
            fixtures,
            args.adaptability_overlap_limit,
        ),
        "multiuser": count_multiuser_specs(profiles, object_pool, fixtures),
    }
    split_order = ("type1", "type2", "type3", "adaptability", "multiuser")
    type_split_order = ("type1", "type2", "type3")
    configured_type_total = get_nested_config(config, ("belongings", "type_episodes"))
    configured_adaptability = get_nested_config(config, ("belongings", "adaptability_episodes"))
    configured_multiuser = get_nested_config(config, ("belongings", "multiuser_episodes"))

    # Backward-compatible fallback for older configs: total_episodes controls only type1-3 now.
    if configured_type_total is None:
        configured_type_total = get_nested_config(config, ("belongings", "total_episodes"))
    if configured_type_total is not None:
        configured_type_total = int(configured_type_total)
    if configured_adaptability is not None:
        configured_adaptability = int(configured_adaptability)
    if configured_multiuser is not None:
        configured_multiuser = int(configured_multiuser)

    type_counts = {split: counts[split] for split in type_split_order}
    planned_counts = distribute_episode_budget(type_counts, configured_type_total, type_split_order)
    planned_counts["adaptability"] = min(
        counts["adaptability"],
        counts["adaptability"] if configured_adaptability is None else configured_adaptability,
    )
    planned_counts["multiuser"] = min(
        counts["multiuser"],
        counts["multiuser"] if configured_multiuser is None else configured_multiuser,
    )
    if args.max_episodes_per_split is not None:
        planned_counts = {name: min(count, args.max_episodes_per_split) for name, count in planned_counts.items()}
    total = sum(counts.values())
    planned_total = sum(planned_counts.values())
    logging.info("episode counts by split: %s", counts)
    logging.info("total episodes to generate: %d", total)
    logging.info("config: %s", args.config)
    if configured_type_total is not None:
        logging.info("configured belongings.type_episodes: %d", configured_type_total)
    if configured_adaptability is not None:
        logging.info("configured belongings.adaptability_episodes: %d", configured_adaptability)
    if configured_multiuser is not None:
        logging.info("configured belongings.multiuser_episodes: %d", configured_multiuser)
    logging.info("planned episodes: %s", planned_counts)
    logging.info("planned total episodes: %d", planned_total)
    if args.max_episodes_per_split is not None:
        logging.info("debug cap enabled: max %d episodes per split", args.max_episodes_per_split)
        logging.info("planned episodes after cap: %s", planned_counts)

    if args.dry_run:
        logging.info("dry run enabled; no files written")
        return

    summary = {
        "suite": "belongings",
        "profiles": str(args.profiles),
        "config": str(args.config),
        "pair_stats": str(args.pair_stats),
        "output_root": str(args.output_root),
        "counts": counts,
        "total": total,
        "planned_counts": planned_counts,
        "planned_total": planned_total,
        "configured_type_episodes": configured_type_total,
        "configured_adaptability_episodes": configured_adaptability,
        "configured_multiuser_episodes": configured_multiuser,
        "hyperparameters": {
            "fixtures": fixtures,
            "scene_variants": [list(item) for item in DEFAULT_SCENE_VARIANTS],
            "spawn_radius": args.spawn_radius,
            "object_box_size": args.object_box_size,
            "fixture_box_size": args.fixture_box_size,
            "seed": args.seed,
            "adaptability_overlap_limit": args.adaptability_overlap_limit,
            "use_gpu": args.use_gpu,
            "max_episodes_per_split": args.max_episodes_per_split,
            "pair_stats_source": str(args.pair_stats),
        },
    }

    progress = tqdm(total=planned_total, desc="Generating belongings episodes", unit="episode")
    written = 0
    for split in split_order:
        split_written = 0
        for spec in factories[split]():
            if split_written >= planned_counts[split]:
                break
            bddl_path, metadata_path = output_paths(args.output_root, spec)
            bddl = render_bddl(spec, args.spawn_radius, args.object_box_size, args.fixture_box_size)
            write_text(bddl_path, bddl)
            write_json(metadata_path, metadata_for(spec, args.profiles, bddl_path, args.spawn_radius))
            written += 1
            split_written += 1
            if args.debug and written <= 5:
                logging.debug("sample episode %s: %s", spec.episode_id, asdict(spec))
            progress.update(1)
    progress.close()

    summary["written"] = written
    write_json(args.output_root / "belongings" / "generation_summary.json", summary)
    logging.info("written episodes: %d", written)
    logging.info("summary: %s", args.output_root / "belongings" / "generation_summary.json")


if __name__ == "__main__":
    main()
