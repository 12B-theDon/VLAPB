#!/usr/bin/env python3
"""Visualize profile uniqueness and overlap for VLAPB users."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = SCRIPT_DIR.parent / "profiles" / "profiles.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "profiles"

UNIQUE_COLOR = "#1b8f3a"
TEXT_COLOR = "#111111"
GRID_COLOR = "#e6e8eb"
USER_COLOR = "#2f3a4a"
SHARED_COLORS = [
    "#d95f02",
    "#1b9e77",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#1f78b4",
    "#b15928",
    "#6a3d9a",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#8dd3c7",
    "#bc80bd",
    "#b3de69",
    "#ff7f00",
    "#33a02c",
    "#cab2d6",
]


def user_sort_key(profile: dict) -> int:
    user_id = str(profile.get("user_id", "0"))
    digits = "".join(ch for ch in user_id if ch.isdigit())
    return int(digits or 0)


def load_profiles(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    profiles = payload.get("profiles", [])
    return sorted(profiles, key=user_sort_key)


def humanize(value: object) -> str:
    return str(value).replace("_", " ").replace(".", " / ")


def comma_join(values: Iterable[str]) -> str:
    return ", ".join(values)


def object_color_map(profiles: list[dict]) -> tuple[dict[str, str], Counter]:
    counts = Counter(obj for profile in profiles for obj in profile.get("belongings", []))
    repeated = sorted(obj for obj, count in counts.items() if count > 1)
    colors = {
        obj: SHARED_COLORS[index % len(SHARED_COLORS)]
        for index, obj in enumerate(repeated)
    }
    for obj, count in counts.items():
        if count == 1:
            colors[obj] = UNIQUE_COLOR
    return colors, counts


def setup_axis(ax, title: str, profiles: list[dict], row_height: float) -> None:
    users = [str(profile.get("user_id", index + 1)) for index, profile in enumerate(profiles)]
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=22)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(profiles) - 0.5, -1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(len(profiles)):
        ax.axhline(row + 0.5, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.text(0.02, -0.72, "user", fontsize=10, color="#697386", fontweight="bold")
    ax.text(0.18, -0.72, "profile data", fontsize=10, color="#697386", fontweight="bold")
    for row, user_id in enumerate(users):
        ax.text(
            0.02,
            row,
            humanize(user_id),
            va="center",
            ha="left",
            fontsize=9,
            color=USER_COLOR,
            fontweight="bold",
        )


def save_belongings(profiles: list[dict], output_path: Path) -> None:
    colors, counts = object_color_map(profiles)
    row_height = 0.48
    fig_height = max(5.5, 1.1 + len(profiles) * row_height)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    setup_axis(ax, "Belongings by User", profiles, row_height)

    for row, profile in enumerate(profiles):
        x = 0.18
        belongings = sorted(profile.get("belongings", []))
        for obj in belongings:
            count = counts[obj]
            color = colors[obj] if count > 1 else UNIQUE_COLOR
            label = humanize(obj)
            bbox = None
            if count > 1:
                bbox = {
                    "boxstyle": "round,pad=0.22,rounding_size=0.08",
                    "facecolor": color,
                    "edgecolor": "none",
                    "alpha": 0.18,
                }
            ax.text(
                x,
                row,
                label,
                va="center",
                ha="left",
                fontsize=8.6,
                color=color if count == 1 else TEXT_COLOR,
                bbox=bbox,
            )
            x += min(0.19, 0.038 + len(label) * 0.0062)

    legend_handles = [
        Patch(facecolor=UNIQUE_COLOR, edgecolor="none", label="appears once"),
        Patch(facecolor="#d95f02", alpha=0.18, edgecolor="none", label="repeated object, same color across users"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_key_value_rows(
    profiles: list[dict],
    output_path: Path,
    *,
    title: str,
    value_fn,
) -> None:
    row_height = 0.62
    fig_height = max(5.5, 1.1 + len(profiles) * row_height)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    setup_axis(ax, title, profiles, row_height)

    for row, profile in enumerate(profiles):
        ax.text(
            0.18,
            row,
            value_fn(profile),
            va="center",
            ha="left",
            fontsize=8.4,
            color=TEXT_COLOR,
            wrap=True,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sorted_placements(profile: dict) -> list[dict]:
    return sorted(profile.get("placements", []), key=lambda item: item.get("order", 0))


def placement_text(profile: dict) -> str:
    items = []
    for placement in sorted_placements(profile):
        obj = humanize(placement.get("object_type", "unknown"))
        area = humanize(placement.get("area", "unknown"))
        fixed = humanize(placement.get("fixed_type", "unknown"))
        label = humanize(placement.get("label", "unknown"))
        scene = humanize(placement.get("scene", "unknown"))
        items.append(f"{obj} -> {area}: {fixed} / {label} ({scene})")
    return comma_join(items)


def sequence_text(profile: dict) -> str:
    strategy = humanize(profile.get("placement_order_strategy", "unknown"))
    order = [humanize(item.get("object_type", "unknown")) for item in sorted_placements(profile)]
    return f"strategy: {strategy}; order: {comma_join(order)}"


def save_overlap_heatmap(profiles: list[dict], output_path: Path) -> None:
    users = [str(profile.get("user_id", index + 1)) for index, profile in enumerate(profiles)]
    belongings = [set(profile.get("belongings", [])) for profile in profiles]
    matrix = [
        [len(left & right) for right in belongings]
        for left in belongings
    ]

    fig_size = max(6.5, len(profiles) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=max(4, max(map(max, matrix))))
    ax.set_title("Pairwise Belongings Overlap", loc="left", fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(range(len(users)))
    ax.set_yticks(range(len(users)))
    ax.set_xticklabels([humanize(user) for user in users], fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels([humanize(user) for user in users], fontsize=8)
    ax.set_xlabel("user")
    ax.set_ylabel("user")
    for i in range(len(users)):
        for j in range(len(users)):
            color = "white" if matrix[i][j] >= 3 else "#111111"
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="shared belongings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(profiles: list[dict], output_path: Path) -> None:
    belongings = [set(profile.get("belongings", [])) for profile in profiles]
    object_counts = Counter(obj for group in belongings for obj in group)
    pair_overlaps = []
    for i, left in enumerate(belongings):
        for right in belongings[i + 1 :]:
            pair_overlaps.append(len(left & right))

    placement_signatures = {
        tuple(
            (
                item.get("object_type"),
                item.get("area"),
                item.get("scene"),
                item.get("table"),
                item.get("fixed_type"),
                item.get("label"),
                item.get("location"),
            )
            for item in sorted_placements(profile)
        )
        for profile in profiles
    }
    sequence_signatures = {
        (
            profile.get("placement_order_strategy"),
            tuple(item.get("object_type") for item in sorted_placements(profile)),
        )
        for profile in profiles
    }
    full_signatures = {
        (
            tuple(sorted(profile.get("belongings", []))),
            tuple(
                (
                    item.get("object_type"),
                    item.get("area"),
                    item.get("scene"),
                    item.get("table"),
                    item.get("fixed_type"),
                    item.get("label"),
                    item.get("location"),
                )
                for item in sorted_placements(profile)
            ),
            profile.get("placement_order_strategy"),
            tuple(item.get("object_type") for item in sorted_placements(profile)),
        )
        for profile in profiles
    }

    lines = [
        "VLAPB profile uniqueness summary",
        "",
        f"users: {len(profiles)}",
        f"object universe: {len(object_counts)}",
        f"objects per user: {sorted({len(group) for group in belongings})}",
        f"globally unique objects: {sum(1 for count in object_counts.values() if count == 1)}",
        f"max pairwise belongings overlap: {max(pair_overlaps) if pair_overlaps else 0}",
        f"pairwise overlap histogram: {dict(sorted(Counter(pair_overlaps).items()))}",
        f"unique placement signatures: {len(placement_signatures)} / {len(profiles)}",
        f"unique sequence signatures: {len(sequence_signatures)} / {len(profiles)}",
        f"unique full profile signatures: {len(full_signatures)} / {len(profiles)}",
        "",
        "order strategy counts:",
    ]
    for strategy, count in sorted(Counter(profile.get("placement_order_strategy", "unknown") for profile in profiles).items()):
        lines.append(f"- {strategy}: {count}")
    lines.extend([
        "",
        "area counts:",
    ])
    for area, count in sorted(Counter(area for profile in profiles for area in profile.get("areas", [])).items()):
        lines.append(f"- {area}: {count}")
    lines.extend([
        "",
        "object counts:",
    ])
    for obj, count in sorted(object_counts.items()):
        lines.append(f"- {obj}: {count}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.profile_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_belongings(profiles, args.output_dir / "belongings.png")
    save_key_value_rows(
        profiles,
        args.output_dir / "placement.png",
        title="Placement Preferences by User",
        value_fn=placement_text,
    )
    save_key_value_rows(
        profiles,
        args.output_dir / "sequence.png",
        title="Sequence Preferences by User",
        value_fn=sequence_text,
    )
    save_overlap_heatmap(profiles, args.output_dir / "belongings_overlap.png")
    write_summary(profiles, args.output_dir / "summary.txt")

    print(f"saved profile visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
