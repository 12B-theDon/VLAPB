#!/usr/bin/env python3
"""Register the VLAPB RLDS dataset name in a local OpenVLA-OFT checkout.

OpenVLA-OFT's finetuning script looks up dataset metadata in three registries:

* prismatic/vla/datasets/rlds/oxe/configs.py
* prismatic/vla/datasets/rlds/oxe/transforms.py
* prismatic/vla/datasets/rlds/oxe/mixtures.py

This helper adds an idempotent `vlapb_moveit` entry matching the LIBERO
continuous-action schema produced by convert_teleop_hdf5_to_rlds.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OPENVLA_OFT_ROOT = Path("/home/artemis/Documents/openvla-oft")
DEFAULT_DATASET_NAME = "vlapb_moveit"


def insert_before_marker(path: Path, marker: str, snippet: str, needle: str, dry_run: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    if needle in text:
        print(f"[skip] {path}: already registered")
        return False
    if marker not in text:
        raise ValueError(f"Marker not found in {path}: {marker!r}")
    new_text = text.replace(marker, snippet + marker, 1)
    if dry_run:
        print(f"[dry-run] would update {path}")
    else:
        path.write_text(new_text, encoding="utf-8")
        print(f"[ok] updated {path}")
    return True


def register_dataset(root: Path, dataset_name: str, dry_run: bool) -> None:
    configs = root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "configs.py"
    transforms = root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "transforms.py"
    mixtures = root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "mixtures.py"
    for path in (configs, transforms, mixtures):
        if not path.exists():
            raise FileNotFoundError(path)

    config_snippet = f'''    "{dataset_name}": {{
        "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": "wrist_image"}},
        "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
        "state_obs_keys": ["EEF_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }},
'''
    insert_before_marker(
        configs,
        "    ### ALOHA fine-tuning datasets\n",
        config_snippet,
        f'"{dataset_name}": {{',
        dry_run,
    )

    transform_snippet = f'    "{dataset_name}": libero_dataset_transform,\n'
    insert_before_marker(
        transforms,
        "    ### ALOHA fine-tuning datasets\n",
        transform_snippet,
        f'"{dataset_name}": libero_dataset_transform',
        dry_run,
    )

    mixture_snippet = f'''    "{dataset_name}": [
        ("{dataset_name}", 1.0),
    ],
'''
    insert_before_marker(
        mixtures,
        "    # === ALOHA Fine-Tuning Datasets ===\n",
        mixture_snippet,
        f'"{dataset_name}": [',
        dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-oft-root", type=Path, default=DEFAULT_OPENVLA_OFT_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_dataset(args.openvla_oft_root, args.dataset_name, args.dry_run)


if __name__ == "__main__":
    main()
