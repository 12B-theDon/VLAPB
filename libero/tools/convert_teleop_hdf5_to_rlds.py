#!/usr/bin/env python3
"""Convert VLAPB/LIBERO teleop HDF5 demonstrations into TFDS/RLDS.

The generated dataset is shaped for the OpenVLA/OpenVLA-OFT LIBERO loader:

* observation/image: third-person RGB image
* observation/wrist_image: wrist RGB image
* observation/state: EEF xyz + axis-angle orientation + 2D gripper state
* action: 7D LIBERO action
* language_instruction: task text, repeated per step

Run this from an environment that has h5py, tensorflow, and
tensorflow-datasets installed, typically the OpenVLA-OFT environment.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install h5py in the active environment.") from exc

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install tensorflow and tensorflow-datasets.") from exc


DEFAULT_INPUT_DIR = Path("/home/artemis/Documents/VLAPB/libero/teleop_moveit/hdf5")
DEFAULT_OUTPUT_DIR = Path("/home/artemis/Documents/VLAPB/libero/teleop_moveit/rlds")
DEFAULT_DATASET_NAME = "vlapb_moveit"


def snake_to_camel(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\W]+", value) if part)


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def language_from_file(h5_file: h5py.File, fallback: str) -> str:
    attrs = h5_file["data"].attrs
    language = decode_attr(attrs.get("language_instruction", ""))
    if language:
        return str(language)

    problem_info_raw = decode_attr(attrs.get("problem_info", ""))
    if problem_info_raw:
        try:
            problem_info = json.loads(problem_info_raw)
            language = problem_info.get("language_instruction")
            if language:
                return str(language)
        except json.JSONDecodeError:
            pass

    return fallback.replace("_", " ").removesuffix(" demo").strip()


def ensure_uint8_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image array with shape [T,H,W,3], got {array.shape}")
    return array


def read_demo_arrays(h5_file: h5py.File, demo_name: str) -> dict[str, np.ndarray]:
    demo = h5_file[f"data/{demo_name}"]
    obs = demo["obs"]

    action = np.asarray(demo["actions"], dtype=np.float32)
    image = ensure_uint8_rgb(obs["agentview_rgb"][()])
    wrist_image = ensure_uint8_rgb(obs["eye_in_hand_rgb"][()])
    ee_state = np.asarray(obs["ee_states"], dtype=np.float32)
    gripper_state = np.asarray(obs["gripper_states"], dtype=np.float32)

    if ee_state.ndim != 2 or ee_state.shape[-1] != 6:
        raise ValueError(f"{demo_name}: expected ee_states shape [T,6], got {ee_state.shape}")
    if gripper_state.ndim != 2 or gripper_state.shape[-1] != 2:
        raise ValueError(f"{demo_name}: expected gripper_states shape [T,2], got {gripper_state.shape}")
    if action.ndim != 2 or action.shape[-1] != 7:
        raise ValueError(f"{demo_name}: expected actions shape [T,7], got {action.shape}")

    length = min(len(action), len(image), len(wrist_image), len(ee_state), len(gripper_state))
    if length <= 0:
        raise ValueError(f"{demo_name}: empty trajectory")

    return {
        "action": action[:length],
        "image": image[:length],
        "wrist_image": wrist_image[:length],
        "state": np.concatenate([ee_state[:length], gripper_state[:length]], axis=-1),
    }


def iter_hdf5_files(input_dir: Path, pattern: str) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def make_builder_class(
    *,
    dataset_name: str,
    input_dir: Path,
    pattern: str,
    limit_files: int | None,
    val_fraction: float,
):
    class_name = snake_to_camel(dataset_name)
    hdf5_paths = iter_hdf5_files(input_dir, pattern)
    if limit_files is not None:
        hdf5_paths = hdf5_paths[:limit_files]
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files matched {input_dir}/{pattern}")

    val_count = int(round(len(hdf5_paths) * val_fraction))
    val_paths = hdf5_paths[:val_count]
    train_paths = hdf5_paths[val_count:]
    if not train_paths:
        train_paths, val_paths = hdf5_paths, []

    class VlapbRldsBuilder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "VLAPB teleop HDF5 converted to OpenVLA-OFT-compatible RLDS."}

        def _info(self) -> tfds.core.DatasetInfo:
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": {
                                    "image": tfds.features.Image(shape=(None, None, 3), dtype=np.uint8),
                                    "wrist_image": tfds.features.Image(shape=(None, None, 3), dtype=np.uint8),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                },
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "language_instruction": tfds.features.Text(),
                                "is_first": np.bool_,
                                "is_last": np.bool_,
                                "is_terminal": np.bool_,
                            }
                        ),
                        "episode_metadata": {
                            "file_path": tfds.features.Text(),
                            "demo_name": tfds.features.Text(),
                        },
                    }
                ),
                supervised_keys=None,
                description="VLAPB teleop HDF5 demonstrations converted to RLDS for OpenVLA-OFT.",
                homepage="https://github.com/",
            )

        def _split_generators(self, dl_manager: tfds.download.DownloadManager):
            splits = {
                "train": self._generate_examples(train_paths),
            }
            if val_paths:
                splits["val"] = self._generate_examples(val_paths)
            return splits

        def _generate_examples(self, paths: list[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
            for hdf5_path in paths:
                with h5py.File(hdf5_path, "r") as h5_file:
                    language = language_from_file(h5_file, hdf5_path.stem)
                    for demo_name in sorted(h5_file["data"].keys()):
                        arrays = read_demo_arrays(h5_file, demo_name)
                        length = len(arrays["action"])
                        steps = []
                        for step_index in range(length):
                            steps.append(
                                {
                                    "observation": {
                                        "image": arrays["image"][step_index],
                                        "wrist_image": arrays["wrist_image"][step_index],
                                        "state": arrays["state"][step_index],
                                    },
                                    "action": arrays["action"][step_index],
                                    "language_instruction": language,
                                    "is_first": step_index == 0,
                                    "is_last": step_index == length - 1,
                                    "is_terminal": step_index == length - 1,
                                }
                            )
                        key = f"{hdf5_path.stem}_{demo_name}"
                        yield key, {
                            "steps": steps,
                            "episode_metadata": {
                                "file_path": str(hdf5_path),
                                "demo_name": str(demo_name),
                            },
                        }

    VlapbRldsBuilder.__name__ = class_name
    VlapbRldsBuilder.name = dataset_name
    return VlapbRldsBuilder, len(train_paths), len(val_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--pattern", default="*_demo.hdf5")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-and-prepare-kwargs", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")

    builder_cls, train_count, val_count = make_builder_class(
        dataset_name=args.dataset_name,
        input_dir=args.input_dir,
        pattern=args.pattern,
        limit_files=args.limit_files,
        val_fraction=args.val_fraction,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = builder_cls(data_dir=str(output_dir))

    builder_data_dir = Path(str(builder.data_dir))
    if builder_data_dir.exists() and args.overwrite:
        shutil.rmtree(builder_data_dir)

    kwargs = {}
    if args.download_and_prepare_kwargs:
        kwargs = json.loads(args.download_and_prepare_kwargs)

    print(f"dataset: {args.dataset_name}")
    print(f"input:   {args.input_dir}")
    print(f"output:  {output_dir}")
    print(f"splits:  train_files={train_count} val_files={val_count}")
    builder.download_and_prepare(**kwargs)
    print(f"done:    {builder.data_dir}")


if __name__ == "__main__":
    main()
