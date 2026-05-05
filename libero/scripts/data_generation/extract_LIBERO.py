#!/usr/bin/env python3
"""Extract LIBERO object-placement metadata into ``libero_objects.json``.

The extractor reads the original LIBERO HDF5 demo files, resolves their BDDL
task files, and preserves enough scene / object / region metadata to regenerate
object placements later.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    h5py = None


VLAPB_LIBERO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_REPO_ROOT = VLAPB_LIBERO_ROOT.parents[1] / "LIBERO"
DEFAULT_DATA_ROOT = Path("/home/artemis/libero_data")
DEFAULT_BDDL_ROOT = LIBERO_REPO_ROOT / "libero" / "libero" / "bddl_files"
DEFAULT_OUTPUT = VLAPB_LIBERO_ROOT / "docs" / "libero_objects.json"
DEFAULT_SUMMARY_OUTPUT = VLAPB_LIBERO_ROOT / "docs" / "libero_objects_summary.md"

DATASETS = {
    "libero_10": DEFAULT_DATA_ROOT / "libero_10",
    "libero_90": DEFAULT_DATA_ROOT / "libero_90",
    "libero_object": DEFAULT_DATA_ROOT / "libero_object",
    "libero_spatial": DEFAULT_DATA_ROOT / "libero_spatial",
}


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract graspable/fixed LIBERO object metadata from HDF5 demos and BDDL files."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bddl-root", type=Path, default=DEFAULT_BDDL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        choices=sorted(DATASETS),
        help="LIBERO dataset folders to scan.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    return parser.parse_args()


def tokenize_sexp(text: str) -> list[str]:
    text = re.sub(r";.*$", "", text, flags=re.MULTILINE)
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexp(text: str) -> list[Any]:
    tokens = tokenize_sexp(text)
    stack: list[list[Any]] = [[]]
    for token in tokens:
        if token == "(":
            child: list[Any] = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("Unexpected closing parenthesis while parsing BDDL")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise ValueError("Unclosed parenthesis while parsing BDDL")
    if len(stack[0]) != 1 or not isinstance(stack[0][0], list):
        raise ValueError("Expected one top-level BDDL expression")
    return stack[0][0]


def section(form: list[Any], name: str) -> list[Any]:
    for item in form:
        if isinstance(item, list) and item and item[0] == name:
            return item[1:]
    return []


def atom_text(items: list[Any]) -> str:
    return " ".join(str(item) for item in items if not isinstance(item, list))


def parse_typed_entities(items: list[Any], category: str) -> dict[str, Entity]:
    entities: dict[str, Entity] = {}
    idx = 0
    while idx < len(items):
        ids: list[str] = []
        while idx < len(items) and items[idx] != "-":
            if isinstance(items[idx], str):
                ids.append(items[idx])
            idx += 1
        if idx >= len(items) or items[idx] != "-":
            break
        idx += 1
        if idx >= len(items) or not isinstance(items[idx], str):
            break
        entity_type = items[idx]
        idx += 1
        for entity_id in ids:
            entities[entity_id] = Entity(entity_id, entity_type, category)
    return entities


def float_or_text(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def parse_regions(items: list[Any]) -> dict[str, dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, list) or not item:
            continue
        name = str(item[0])
        region: dict[str, Any] = {"name": name}
        for child in item[1:]:
            if not isinstance(child, list) or not child:
                continue
            key = str(child[0]).lstrip(":")
            if key == "target" and len(child) > 1:
                region["target"] = child[1]
            elif key in {"ranges", "yaw_rotation"}:
                region[key] = nested_numbers(child[1:])
            else:
                region[key] = child[1:]
        regions[name] = region
    return regions


def nested_numbers(value: Any) -> Any:
    if isinstance(value, list):
        return [nested_numbers(item) for item in value]
    if isinstance(value, str):
        return float_or_text(value)
    return value


def parse_predicates(items: list[Any]) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, list) or not item:
            continue
        if item[0] == "And":
            predicates.extend(parse_predicates(item[1:]))
            continue
        predicates.append(
            {
                "predicate": str(item[0]),
                "args": [str(arg) for arg in item[1:] if isinstance(arg, str)],
            }
        )
    return predicates


def parse_bddl(path: Path) -> dict[str, Any]:
    form = parse_sexp(path.read_text(encoding="utf-8"))
    language = atom_text(section(form, ":language"))
    fixtures = parse_typed_entities(section(form, ":fixtures"), "fixture")
    objects = parse_typed_entities(section(form, ":objects"), "object")
    entities = {**fixtures, **objects}
    regions = parse_regions(section(form, ":regions"))
    init = parse_predicates(section(form, ":init"))
    goal = parse_predicates(section(form, ":goal"))
    obj_of_interest = [str(item) for item in section(form, ":obj_of_interest") if isinstance(item, str)]

    return {
        "problem": form[1][1] if len(form) > 1 and isinstance(form[1], list) and len(form[1]) > 1 else None,
        "language": language,
        "fixtures": fixtures,
        "objects": objects,
        "entities": entities,
        "regions": regions,
        "init": init,
        "goal": goal,
        "obj_of_interest": obj_of_interest,
    }


def split_region_reference(ref: str, regions: dict[str, dict[str, Any]], entities: dict[str, Entity]) -> dict[str, Any]:
    if ref in entities:
        entity = entities[ref]
        return {
            "kind": "object",
            "entity_id": entity.entity_id,
            "type": entity.entity_type,
            "category": entity.category,
        }

    matches = []
    for region_name in regions:
        suffix = f"_{region_name}"
        if ref.endswith(suffix):
            prefix = ref[: -len(suffix)]
            matches.append((len(region_name), prefix, region_name))
    if matches:
        _size, target_id, region_name = sorted(matches, reverse=True)[0]
        target_entity = entities.get(target_id)
        region = regions[region_name]
        return {
            "kind": "region",
            "reference": ref,
            "target_id": target_id,
            "target_type": target_entity.entity_type if target_entity else region.get("target"),
            "target_category": target_entity.category if target_entity else None,
            "region": region,
        }

    return {"kind": "unknown", "reference": ref}


def entity_payload(entity: Entity, couplings: list[str], task_id: str, scene: str | None, table: str | None, location: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "id": entity.entity_id,
        "type": entity.entity_type,
        "category": entity.category,
        "coupling_ids": sorted(couplings),
        "task_id": task_id,
        "scene": scene,
        "table": table,
    }
    if location:
        payload["location"] = location
    return payload


def normalize_scene(stem: str) -> str | None:
    match = re.match(r"(?P<scene>[A-Z]+_ROOM_SCENE\d+|KITCHEN_SCENE\d+|LIVING_ROOM_SCENE\d+|STUDY_SCENE\d+)", stem)
    if match:
        return match.group("scene")
    return None


def strip_demo_suffix(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_demo") else stem


def bddl_candidates(dataset: str, hdf5_path: Path, bddl_root: Path, attrs: dict[str, Any]) -> list[Path]:
    stem = strip_demo_suffix(hdf5_path)
    candidates = [bddl_root / dataset / f"{stem}.bddl"]

    raw_attr = attrs.get("bddl_file_name") or attrs.get("bddl_file")
    if raw_attr:
        attr_path = Path(str(raw_attr))
        candidates.append(bddl_root / attr_path.name if attr_path.suffix == ".bddl" else bddl_root / dataset / f"{stem}.bddl")
        if "bddl_files" in attr_path.parts:
            parts = attr_path.parts
            idx = parts.index("bddl_files")
            if idx + 2 < len(parts):
                candidates.append(bddl_root / parts[idx + 1] / parts[-1])

    for suite_dir in bddl_root.iterdir() if bddl_root.exists() else []:
        if suite_dir.is_dir():
            candidates.append(suite_dir / f"{stem}.bddl")
    return dedupe_paths(candidates)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    output = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def read_hdf5_attrs(path: Path) -> dict[str, Any]:
    if h5py is None:
        raise RuntimeError("h5py is required to read LIBERO HDF5 metadata")
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        attrs = dict(data.attrs) if data is not None else dict(handle.attrs)
    return {key: json_safe(value) for key, value in attrs.items()}


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def init_locations(init_predicates: list[dict[str, Any]], regions: dict[str, dict[str, Any]], entities: dict[str, Entity]) -> dict[str, dict[str, Any]]:
    locations = {}
    for pred in init_predicates:
        args = pred.get("args", [])
        if len(args) < 2:
            continue
        locations[args[0]] = {
            "predicate": pred["predicate"],
            **split_region_reference(args[1], regions, entities),
        }
    return locations


def resolve_fixed_object(target_ref: str, regions: dict[str, dict[str, Any]], entities: dict[str, Entity]) -> Entity | None:
    if target_ref in entities:
        return entities[target_ref]
    target = split_region_reference(target_ref, regions, entities)
    target_id = target.get("target_id")
    if target_id in entities:
        return entities[target_id]
    region = target.get("region")
    if isinstance(region, dict):
        region_target = region.get("target")
        if region_target in entities:
            return entities[region_target]
    return None


def table_for_entity(entity_id: str, locations: dict[str, dict[str, Any]]) -> str | None:
    location = locations.get(entity_id)
    if not location:
        return None
    if location.get("kind") == "region":
        region = location.get("region") or {}
        return region.get("target")
    return location.get("entity_id") or location.get("target_id")


def relation_marks(predicate: str, target_ref: str, fixed: Entity | None) -> list[str]:
    marks = []
    if predicate.lower() == "on" and fixed and fixed.category == "object":
        marks.append("stacking")
    elif predicate.lower() == "on":
        marks.append("surface_placement")
    elif predicate.lower() == "in":
        marks.append("containment")
    if re.search(r"(top|middle|bottom|left|right|front|back)", target_ref):
        marks.append("spatial_detail")
    return marks


def build_task(dataset: str, hdf5_path: Path, bddl_path: Path, attrs: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    task_id = f"{dataset}:{strip_demo_suffix(hdf5_path)}"
    scene = normalize_scene(strip_demo_suffix(hdf5_path))
    entities: dict[str, Entity] = parsed["entities"]
    regions = parsed["regions"]
    locations = init_locations(parsed["init"], regions, entities)
    couplings = []

    for index, pred in enumerate(parsed["goal"]):
        args = pred.get("args", [])
        if len(args) < 2:
            continue
        moved = entities.get(args[0])
        if moved is None:
            continue
        fixed = resolve_fixed_object(args[1], regions, entities)
        coupling_id = f"{task_id}:goal_{index}"
        moved_table = table_for_entity(moved.entity_id, locations)
        fixed_table = table_for_entity(fixed.entity_id, locations) if fixed else None
        couplings.append(
            {
                "id": coupling_id,
                "task_id": task_id,
                "dataset": dataset,
                "scene": scene,
                "table": fixed_table or moved_table,
                "relation": pred["predicate"],
                "marks": relation_marks(pred["predicate"], args[1], fixed),
                "graspable_object": entity_payload(
                    moved,
                    [coupling_id],
                    task_id,
                    scene,
                    moved_table,
                    locations.get(moved.entity_id),
                ),
                "fixed_object": entity_payload(
                    fixed,
                    [coupling_id],
                    task_id,
                    scene,
                    fixed_table,
                    locations.get(fixed.entity_id),
                )
                if fixed
                else {
                    "id": args[1],
                    "type": None,
                    "category": "unknown",
                    "coupling_ids": [coupling_id],
                    "task_id": task_id,
                    "scene": scene,
                    "table": None,
                    "location": split_region_reference(args[1], regions, entities),
                },
                "goal_location": split_region_reference(args[1], regions, entities),
            }
        )

    return {
        "id": task_id,
        "dataset": dataset,
        "scene": scene,
        "language": parsed["language"] or attrs.get("problem_info"),
        "hdf5_file": str(hdf5_path),
        "bddl_file": str(bddl_path),
        "problem": parsed["problem"],
        "hdf5_metadata": attrs,
        "fixtures": [entity_payload(entity, [], task_id, scene, table_for_entity(entity.entity_id, locations), locations.get(entity.entity_id)) for entity in parsed["fixtures"].values()],
        "objects": [entity_payload(entity, [], task_id, scene, table_for_entity(entity.entity_id, locations), locations.get(entity.entity_id)) for entity in parsed["objects"].values()],
        "obj_of_interest": parsed["obj_of_interest"],
        "regions": regions,
        "init": parsed["init"],
        "goal": parsed["goal"],
        "couplings": couplings,
    }


def aggregate_objects(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    graspable: dict[tuple[str, str, str], dict[str, Any]] = {}
    fixed: dict[tuple[str, str, str], dict[str, Any]] = {}
    couplings = []

    for task in tasks:
        for coupling in task["couplings"]:
            couplings.append(coupling)
            for bucket_name, bucket in (("graspable_object", graspable), ("fixed_object", fixed)):
                item = coupling[bucket_name]
                key = (item.get("id"), item.get("task_id"), coupling["id"])
                bucket[key] = item

    return list(graspable.values()), list(fixed.values()), couplings


def scene_area(scene: str | None, dataset: str) -> str:
    if scene:
        if scene.startswith("KITCHEN_"):
            return "kitchen"
        if scene.startswith("STUDY_"):
            return "study"
        if scene.startswith("LIVING_ROOM_"):
            return "living_room"
    return dataset


def display_fixed_type(fixed_type: str | None) -> str:
    aliases = {
        "flat_stove": "flat_oven",
        "wooden_cabinet": "drawer",
        "white_cabinet": "drawer",
        "desk_caddy": "caddy",
        "wooden_two_layer_shelf": "shelf",
        "wooden_tray": "tray",
    }
    return aliases.get(fixed_type or "unknown", fixed_type or "unknown")


def location_labels(coupling: dict[str, Any]) -> list[str]:
    labels = [str(coupling.get("relation", "")).lower()]
    goal_location = coupling.get("goal_location") or {}
    region = goal_location.get("region") or {}
    haystack = " ".join(
        str(part)
        for part in (
            goal_location.get("reference", ""),
            region.get("name", ""),
            region.get("target", ""),
        )
    ).lower()

    for word in ("top", "middle", "bottom", "front", "back", "left", "right", "cook", "contain"):
        if word in haystack:
            labels.append(word)
    if "stacking" in coupling.get("marks", []):
        labels.append("stacking")

    return sorted({label for label in labels if label})


def format_count_cells(counts: Counter[str], total: int) -> tuple[str, str]:
    ordered_labels = [
        "on",
        "in",
        "top",
        "middle",
        "bottom",
        "front",
        "back",
        "left",
        "right",
        "cook",
        "contain",
        "stacking",
    ]
    return (
        str(total),
        ", ".join(f"{label}({counts[label]})" for label in ordered_labels if counts[label]),
    )


def summary_groups(output: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, Any]] = {}
    raw_grouped: dict[str, dict[str, Any]] = {}

    for coupling in output["couplings"]:
        fixed = coupling["fixed_object"]
        fixed_type = fixed.get("type") or fixed.get("id") or "unknown"
        area = scene_area(coupling.get("scene"), coupling.get("dataset", "unknown"))
        labels = location_labels(coupling)

        for groups, name in ((grouped, display_fixed_type(fixed_type)), (raw_grouped, fixed_type)):
            entry = groups.setdefault(name, {"areas": set(), "counts": Counter(), "total": 0})
            entry["areas"].add(area)
            entry["total"] += 1
            for label in labels:
                entry["counts"][label] += 1

    return grouped, raw_grouped


def markdown_count_table(groups: dict[str, dict[str, Any]], name_header: str) -> list[str]:
    lines = [
        f"## {name_header}",
        "",
        "| Fixed item | Areas | Target count | Placement labels |",
        "| --- | --- | ---: | --- |",
    ]
    for name in sorted(groups):
        entry = groups[name]
        areas = ", ".join(sorted(entry["areas"]))
        total, labels = format_count_cells(entry["counts"], entry["total"])
        lines.append(f"| `{name}` | {areas} | {total} | {labels or '-'} |")
    return lines


def location_signature(location: dict[str, Any] | None) -> str:
    location = location or {}
    region = location.get("region") or {}
    payload = {
        "predicate": location.get("predicate"),
        "kind": location.get("kind"),
        "reference": location.get("reference"),
        "target": region.get("target"),
        "ranges": region.get("ranges"),
        "yaw_rotation": region.get("yaw_rotation"),
    }
    return json.dumps(payload, sort_keys=True)


def fixed_location_consistency(output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    for coupling in output["couplings"]:
        fixed = coupling["fixed_object"]
        fixed_type = fixed.get("type") or fixed.get("id") or "unknown"
        context = (coupling.get("scene") or coupling.get("dataset") or "unknown", coupling.get("table") or "none")
        by_type[fixed_type][context].add(location_signature(fixed.get("location")))

    summary = {}
    for fixed_type, contexts in by_type.items():
        inconsistent = {context: len(signatures) for context, signatures in contexts.items() if len(signatures) > 1}
        summary[fixed_type] = {
            "contexts": len(contexts),
            "unique_locations": sum(len(signatures) for signatures in contexts.values()),
            "max_locations_per_context": max(len(signatures) for signatures in contexts.values()),
            "inconsistent_contexts": inconsistent,
        }
    return summary


def markdown_consistency_table(output: dict[str, Any]) -> list[str]:
    consistency = fixed_location_consistency(output)
    lines = [
        "## Table-Bounded Fixed Location Consistency",
        "",
        "A row is **consistent** when the same raw fixed object type has exactly one initial location for each `(scene, table)` context. Different scenes may still have different table-bounded coordinates.",
        "",
        "| Raw fixed type | Contexts | Unique locations | Max per context | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for fixed_type in sorted(consistency):
        item = consistency[fixed_type]
        inconsistent = item["inconsistent_contexts"]
        verdict = "consistent" if not inconsistent else "varies"
        lines.append(
            f"| `{fixed_type}` | {item['contexts']} | {item['unique_locations']} | "
            f"{item['max_locations_per_context']} | {verdict} |"
        )
    return lines


def build_fixed_item_summary(output: dict[str, Any]) -> str:
    grouped, raw_grouped = summary_groups(output)
    lines = [
        "# LIBERO Fixed Item Placement Summary",
        "",
        f"- Tasks: `{output['metadata']['task_count']}`",
        f"- Couplings: `{output['metadata']['coupling_count']}`",
        "",
    ]
    lines.extend(markdown_count_table(grouped, "Grouped Fixed Items"))
    lines.extend([""])
    lines.extend(markdown_count_table(raw_grouped, "Raw Fixed Object Types"))
    lines.extend([""])
    lines.extend(markdown_consistency_table(output))
    return "\n".join(lines) + "\n"


def extract(args: argparse.Namespace) -> dict[str, Any]:
    tasks = []
    missing_bddl = []
    dataset_counts = defaultdict(int)

    for dataset in args.datasets:
        data_dir = args.data_root / dataset
        for hdf5_path in sorted(data_dir.glob("*.hdf5")):
            attrs = read_hdf5_attrs(hdf5_path)
            candidates = bddl_candidates(dataset, hdf5_path, args.bddl_root, attrs)
            bddl_path = next((path for path in candidates if path.exists()), None)
            if bddl_path is None:
                missing_bddl.append({"dataset": dataset, "hdf5_file": str(hdf5_path), "tried": [str(path) for path in candidates]})
                continue
            parsed = parse_bddl(bddl_path)
            tasks.append(build_task(dataset, hdf5_path, bddl_path, attrs, parsed))
            dataset_counts[dataset] += 1

    graspable_objects, fixed_objects, couplings = aggregate_objects(tasks)
    return {
        "metadata": {
            "data_root": str(args.data_root),
            "bddl_root": str(args.bddl_root),
            "datasets": list(args.datasets),
            "task_count": len(tasks),
            "coupling_count": len(couplings),
            "graspable_object_count": len(graspable_objects),
            "fixed_object_count": len(fixed_objects),
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "missing_bddl": missing_bddl,
        },
        "graspable_objects": graspable_objects,
        "fixed_objects": fixed_objects,
        "couplings": couplings,
        "tasks": tasks,
    }


def main() -> None:
    args = parse_args()
    output = extract(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    args.output.write_text(json.dumps(output, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(build_fixed_item_summary(output), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Tasks: {output['metadata']['task_count']}")
    print(f"Couplings: {output['metadata']['coupling_count']}")
    if output["metadata"]["missing_bddl"]:
        print(f"Missing BDDL files: {len(output['metadata']['missing_bddl'])}")


if __name__ == "__main__":
    main()
