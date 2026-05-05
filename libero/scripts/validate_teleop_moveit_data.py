"""Validate and preview VLAPB suite teleop/MoveIt demonstrations.

This script checks the processed LIBERO-style HDF5 files written by
``collect_suite_teleop_xbox.py``. It can be used one file at a time while
collecting keyboard teleop data, or across the whole suite collection output after
collection.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEOP_MOVEIT_DIR = REPO_ROOT / "libero" / "teleop_moveit"
DEFAULT_HDF5_DIR = DEFAULT_TELEOP_MOVEIT_DIR / "hdf5"
DEFAULT_PREVIEW_DIR = DEFAULT_TELEOP_MOVEIT_DIR / "previews"
DEFAULT_REPORT_DIR = DEFAULT_TELEOP_MOVEIT_DIR / "review"

REQUIRED_DEMO_DATASETS = ("actions", "states", "robot_states", "rewards", "dones")
REQUIRED_OBS_DATASETS = ("agentview_rgb", "eye_in_hand_rgb")
OPTIONAL_PROPRIO_DATASETS = ("gripper_states", "joint_states", "ee_states", "ee_pos", "ee_ori")


@dataclass
class ValidationResult:
    path: Path
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    demos: list[dict[str, Any]] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def hdf5_paths_under(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.hdf5"))


def path_matches_episode(path: Path, episode_id: str) -> bool:
    if path.stem == episode_id or path.stem == f"{episode_id}_demo":
        return True
    try:
        with h5py.File(path, "r") as h5_file:
            payload = h5_file["data"].attrs.get("vlapb_episode")
            if payload is None:
                return False
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            row = json.loads(payload)
            return row.get("episode_id") == episode_id
    except Exception:
        return False


def selected_hdf5_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.files:
        paths.extend(Path(file_path) for file_path in args.files)
    if args.episode:
        matches = [path for path in hdf5_paths_under(args.hdf5_dir) if path_matches_episode(path, args.episode)]
        if not matches:
            raise ValueError(f"Episode {args.episode} not found under {args.hdf5_dir}")
        paths.extend(matches)
    if args.all:
        paths.extend(hdf5_paths_under(args.hdf5_dir))

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique_paths.append(resolved)
    if not unique_paths:
        raise ValueError("Provide at least one HDF5 file, --episode EPISODE_ID, or --all.")
    return unique_paths


def h5_attr_to_python(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_demo_names(data_group: h5py.Group) -> list[str]:
    return sorted(name for name in data_group.keys() if name.startswith("demo_"))


def dataset_len(dataset: h5py.Dataset) -> int:
    if dataset.shape == ():
        return 0
    return int(dataset.shape[0])


def validate_rgb_dataset(result: ValidationResult, demo_name: str, obs: h5py.Group, key: str, expected_len: int) -> None:
    if key not in obs:
        result.fail(f"{demo_name}/obs/{key} is missing")
        return
    dataset = obs[key]
    if dataset_len(dataset) != expected_len:
        result.fail(f"{demo_name}/obs/{key} length {dataset_len(dataset)} != actions length {expected_len}")
    if len(dataset.shape) != 4:
        result.fail(f"{demo_name}/obs/{key} should be rank-4 T,H,W,C; got shape {dataset.shape}")
        return
    if dataset.shape[-1] != 3:
        result.fail(f"{demo_name}/obs/{key} should have 3 RGB channels; got shape {dataset.shape}")
    if dataset.dtype != np.uint8:
        result.warn(f"{demo_name}/obs/{key} dtype is {dataset.dtype}, expected uint8")


def validate_hdf5_file(path: Path, *, min_steps: int = 10, require_single_demo: bool = False) -> ValidationResult:
    result = ValidationResult(path=path)
    if not path.exists():
        result.fail("file does not exist")
        return result

    try:
        with h5py.File(path, "r") as h5_file:
            if "data" not in h5_file:
                result.fail("missing top-level data group")
                return result

            data_group = h5_file["data"]
            result.attrs = {key: h5_attr_to_python(value) for key, value in data_group.attrs.items()}
            demo_names = get_demo_names(data_group)
            if not demo_names:
                result.fail("data group has no demo_* groups")
                return result
            if require_single_demo and len(demo_names) != 1:
                result.fail(f"expected exactly one demo, found {len(demo_names)}")

            attr_num_demos = data_group.attrs.get("num_demos")
            if attr_num_demos is not None and int(attr_num_demos) != len(demo_names):
                result.fail(f"data.attrs['num_demos']={attr_num_demos} but found {len(demo_names)} demo groups")

            total_samples = 0
            for demo_name in demo_names:
                demo = data_group[demo_name]
                demo_summary: dict[str, Any] = {"name": demo_name}
                if "obs" not in demo:
                    result.fail(f"{demo_name} missing obs group")
                    continue
                obs = demo["obs"]

                missing = [key for key in REQUIRED_DEMO_DATASETS if key not in demo]
                for key in missing:
                    result.fail(f"{demo_name}/{key} is missing")
                if missing:
                    continue

                actions = demo["actions"]
                n_steps = dataset_len(actions)
                demo_summary["num_samples"] = n_steps
                demo_summary["action_shape"] = tuple(actions.shape)
                total_samples += n_steps

                if n_steps < min_steps:
                    result.warn(f"{demo_name} has only {n_steps} samples; min_steps={min_steps}")

                attr_samples = demo.attrs.get("num_samples")
                if attr_samples is not None and int(attr_samples) != n_steps:
                    result.fail(f"{demo_name}.attrs['num_samples']={attr_samples} but actions length is {n_steps}")

                for key in REQUIRED_DEMO_DATASETS:
                    length = dataset_len(demo[key])
                    if length != n_steps:
                        result.fail(f"{demo_name}/{key} length {length} != actions length {n_steps}")

                if len(actions.shape) != 2:
                    result.fail(f"{demo_name}/actions should be rank-2 T,A; got shape {actions.shape}")
                else:
                    action_array = actions[()]
                    if not np.isfinite(action_array).all():
                        result.fail(f"{demo_name}/actions contains NaN or Inf")
                    if np.allclose(action_array, 0.0):
                        result.fail(f"{demo_name}/actions are all zero")
                    demo_summary["action_abs_mean"] = float(np.mean(np.abs(action_array)))
                    demo_summary["action_abs_max"] = float(np.max(np.abs(action_array)))

                dones = demo["dones"][()]
                rewards = demo["rewards"][()]
                if n_steps > 0:
                    if int(dones[-1]) != 1:
                        result.fail(f"{demo_name}/dones last value should be 1")
                    if np.count_nonzero(dones) != 1:
                        result.warn(f"{demo_name}/dones has {np.count_nonzero(dones)} nonzero values; expected one final done")
                    if int(rewards[-1]) != 1:
                        result.warn(f"{demo_name}/rewards last value is not 1")

                for key in REQUIRED_OBS_DATASETS:
                    validate_rgb_dataset(result, demo_name, obs, key, n_steps)

                for key in OPTIONAL_PROPRIO_DATASETS:
                    if key in obs and dataset_len(obs[key]) != n_steps:
                        result.fail(f"{demo_name}/obs/{key} length {dataset_len(obs[key])} != actions length {n_steps}")

                result.demos.append(demo_summary)

            attr_total = data_group.attrs.get("total")
            if attr_total is not None and int(attr_total) != total_samples:
                result.fail(f"data.attrs['total']={attr_total} but summed samples={total_samples}")

    except OSError as exc:
        result.fail(f"could not open HDF5: {exc}")

    return result


def print_validation_result(result: ValidationResult) -> None:
    status = "OK" if result.ok else "FAILED"
    print(f"[{status}] {result.path}")
    for demo in result.demos:
        print(
            f"  {demo['name']}: samples={demo.get('num_samples')} "
            f"action_shape={demo.get('action_shape')} "
            f"abs_mean={demo.get('action_abs_mean', 0.0):.5f} "
            f"abs_max={demo.get('action_abs_max', 0.0):.5f}"
        )
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for error in result.errors:
        print(f"  error: {error}")


def render_video(
    hdf5_path: Path,
    output_path: Path,
    *,
    demo_name: str = "demo_0",
    camera_key: str = "agentview_rgb",
    fps: int = 20,
    stride: int = 1,
    resize_width: int | None = 512,
    resize_height: int | None = 512,
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for render-video. Install imageio/imageio-ffmpeg if needed.") from exc

    with h5py.File(hdf5_path, "r") as h5_file:
        dataset_path = f"data/{demo_name}/obs/{camera_key}"
        if dataset_path not in h5_file:
            raise KeyError(f"{dataset_path} not found in {hdf5_path}")
        frames = h5_file[dataset_path][()]

    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if (resize_width is None) ^ (resize_height is None):
        raise ValueError("resize_width and resize_height must be set together")
    if resize_width is not None and resize_width < 1:
        raise ValueError("resize_width must be >= 1")
    if resize_height is not None and resize_height < 1:
        raise ValueError("resize_height must be >= 1")
    frames = frames[::stride]
    # robosuite/MuJoCo camera observations are stored in framebuffer row order.
    # Flip only for human-facing preview videos; keep HDF5 data unchanged.
    frames = frames[:, ::-1, :, :]
    if resize_width is not None and resize_height is not None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for preview resizing. Install opencv-python.") from exc
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC))
        frames = np.asarray(resized, dtype=np.uint8)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def default_video_path(hdf5_path: Path, preview_dir: Path, *, demo_name: str = "demo_0", camera_key: str = "agentview_rgb") -> Path:
    suffix = "" if demo_name == "demo_0" else f"_{demo_name}"
    camera_suffix = "" if camera_key == "agentview_rgb" else f"_{camera_key}"
    return preview_dir / f"{hdf5_path.stem}{suffix}{camera_suffix}.mp4"


def row_from_hdf5_attrs(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as h5_file:
            payload = h5_file["data"].attrs.get("vlapb_episode")
            if payload is None:
                return {}
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            return json.loads(payload)
    except Exception:
        return {}


def row_lookup_by_output(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for path in paths:
        row = row_from_hdf5_attrs(path)
        lookup[str(path)] = row
        lookup[path.name] = row
    return lookup


def metadata_for_path(path: Path, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return lookup.get(str(path), lookup.get(path.name, {}))


def write_review_report(
    *,
    results: list[ValidationResult],
    preview_paths: dict[Path, Path],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = row_lookup_by_output([result.path for result in results])
    csv_path = output_dir / "review_report.csv"
    html_path = output_dir / "review_report.html"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "status",
                "episode_id",
                "episode_type",
                "user_id",
                "hdf5",
                "preview",
                "num_demos",
                "num_samples",
                "instruction",
                "expected_behavior",
                "warnings",
                "errors",
            ],
        )
        writer.writeheader()
        for result in results:
            row = metadata_for_path(result.path, lookup)
            writer.writerow(
                {
                    "status": "ok" if result.ok else "failed",
                    "episode_id": row.get("episode_id", ""),
                    "episode_type": row.get("episode_type", ""),
                    "user_id": row.get("user_id", ""),
                    "hdf5": str(result.path),
                    "preview": str(preview_paths.get(result.path, "")),
                    "num_demos": len(result.demos),
                    "num_samples": sum(int(demo.get("num_samples", 0)) for demo in result.demos),
                    "instruction": row.get("instruction", ""),
                    "expected_behavior": row.get("expected_behavior", ""),
                    "warnings": " | ".join(result.warnings),
                    "errors": " | ".join(result.errors),
                }
            )

    cards = []
    for result in results:
        row = metadata_for_path(result.path, lookup)
        preview_path = preview_paths.get(result.path)
        rel_preview = ""
        if preview_path is not None and preview_path.exists():
            try:
                rel_preview = str(preview_path.relative_to(output_dir))
            except ValueError:
                rel_preview = str(preview_path)

        status = "ok" if result.ok else "failed"
        warning_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in result.warnings)
        error_html = "".join(f"<li>{html.escape(error)}</li>" for error in result.errors)
        video_html = (
            f'<video src="{html.escape(rel_preview)}" controls preload="metadata"></video>'
            if rel_preview
            else '<div class="missing">No preview video</div>'
        )
        cards.append(
            f"""
            <article class="card {status}">
              <header>
                <strong>{html.escape(row.get("episode_id", result.path.stem))}</strong>
                <span>{html.escape(row.get("episode_type", ""))}</span>
                <span>{html.escape(row.get("user_id", ""))}</span>
              </header>
              {video_html}
              <p class="instruction">{html.escape(row.get("instruction", ""))}</p>
              <p>{html.escape(row.get("expected_behavior", ""))}</p>
              <p class="path">{html.escape(str(result.path))}</p>
              <ul class="errors">{error_html}</ul>
              <ul class="warnings">{warning_html}</ul>
            </article>
            """
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VLAPB Suite Teleop/MoveIt Data Review</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, sans-serif; background: #f7f7f5; color: #202020; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d8d8d4; border-radius: 8px; padding: 12px; }}
    .card.failed {{ border-color: #b42318; }}
    header {{ display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }}
    header span {{ color: #555; font-size: 13px; }}
    video {{ width: 100%; aspect-ratio: 1 / 1; background: #111; border-radius: 6px; }}
    p {{ margin: 8px 0; line-height: 1.35; }}
    .instruction {{ font-weight: 600; }}
    .path {{ color: #666; font-size: 12px; overflow-wrap: anywhere; }}
    .errors {{ color: #b42318; }}
    .warnings {{ color: #8a5a00; }}
    .missing {{ display: grid; place-items: center; height: 220px; background: #eee; border-radius: 6px; color: #666; }}
  </style>
</head>
<body>
  <h1>VLAPB Suite Teleop/MoveIt Data Review</h1>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path, csv_path


def validate_command(args: argparse.Namespace) -> None:
    results = [
        validate_hdf5_file(path, min_steps=args.min_steps, require_single_demo=args.require_single_demo)
        for path in selected_hdf5_paths(args)
    ]
    for result in results:
        print_validation_result(result)
    if any(not result.ok for result in results):
        raise SystemExit(1)


def render_video_command(args: argparse.Namespace) -> None:
    for path in selected_hdf5_paths(args):
        validation = validate_hdf5_file(path, min_steps=args.min_steps, require_single_demo=False)
        print_validation_result(validation)
        if not validation.ok and not args.allow_invalid:
            raise SystemExit(1)
        output_path = args.output
        if output_path is None:
            output_path = default_video_path(path, args.preview_dir, demo_name=args.demo, camera_key=args.camera_key)
        rendered = render_video(
            path,
            output_path,
            demo_name=args.demo,
            camera_key=args.camera_key,
            fps=args.fps,
            stride=args.stride,
            overwrite=args.overwrite,
        )
        print(f"[video] {path} -> {rendered}")


def review_report_command(args: argparse.Namespace) -> None:
    paths = selected_hdf5_paths(args)
    results = [
        validate_hdf5_file(path, min_steps=args.min_steps, require_single_demo=args.require_single_demo)
        for path in paths
    ]
    preview_paths: dict[Path, Path] = {}
    for result in results:
        print_validation_result(result)
        preview_path = default_video_path(result.path, args.preview_dir, demo_name=args.demo, camera_key=args.camera_key)
        preview_paths[result.path] = preview_path
        if result.ok or args.allow_invalid:
            try:
                render_video(
                    result.path,
                    preview_path,
                    demo_name=args.demo,
                    camera_key=args.camera_key,
                    fps=args.fps,
                    stride=args.stride,
                    overwrite=args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - report preview failures in the review artifact.
                result.warn(f"preview video failed: {exc}")

    html_path, csv_path = write_review_report(
        results=results,
        preview_paths=preview_paths,
        output_dir=args.report_dir,
    )
    print(f"[report] html: {html_path}")
    print(f"[report] csv:  {csv_path}")
    if any(not result.ok for result in results):
        raise SystemExit(1)


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="*", type=Path, help="Processed LIBERO HDF5 files to inspect one by one.")
    parser.add_argument("--hdf5-dir", type=Path, default=DEFAULT_HDF5_DIR)
    parser.add_argument("--episode", type=str, default=None, help="Resolve one episode id from teleop_moveit/hdf5.")
    parser.add_argument("--all", action="store_true", help="Inspect every .hdf5 file under --hdf5-dir.")


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--require-single-demo", action="store_true", help="Fail unless each file contains exactly one demo.")


def add_video_args(parser: argparse.ArgumentParser, *, include_output: bool = True) -> None:
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    if include_output:
        parser.add_argument("--output", type=Path, default=None, help="Explicit mp4 path. Only use with a single input file.")
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--camera-key", type=str, default="agentview_rgb")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true", help="Try rendering even if validation fails.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Check HDF5 structure, lengths, actions, dones, rewards, and observations.")
    add_selection_args(validate)
    add_validation_args(validate)

    video = subparsers.add_parser("render-video", help="Render agentview_rgb or another RGB obs dataset to mp4.")
    add_selection_args(video)
    add_validation_args(video)
    add_video_args(video)

    report = subparsers.add_parser("review-report", help="Validate files, render previews, and write HTML/CSV review pages.")
    add_selection_args(report)
    add_validation_args(report)
    add_video_args(report, include_output=False)
    report.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    args = parser.parse_args()
    if getattr(args, "output", None) is not None:
        selected = selected_hdf5_paths(args)
        if len(selected) != 1:
            parser.error("--output can only be used with exactly one selected HDF5 file")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        validate_command(args)
    elif args.command == "render-video":
        render_video_command(args)
    elif args.command == "review-report":
        review_report_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
