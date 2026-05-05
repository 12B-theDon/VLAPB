#!/usr/bin/env python3
"""Batch-plan VLAPB MoveIt/MTC tasks with a tqdm ETA bar.

Run this inside the MoveIt container after sourcing the workspace, or use the
--source-setup default to let the script source /root/ws_moveit/install/setup.bash
for each ros2 launch invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - container fallback
    tqdm = None


DEFAULT_MOVEIT_DIR = Path("/home/artemis/Documents/VLAPB/libero/teleop_moveit/moveit")
DEFAULT_SOURCE_SETUP = Path("/root/ws_moveit/install/setup.bash")


@dataclass(frozen=True)
class Job:
    episode_id: str
    step_index: int
    total_steps: int
    task_file: Path
    output_trajectory: Path


@dataclass(frozen=True)
class TrajectoryStats:
    point_count: int
    duration: float
    stage_counts: dict[str, int]
    flags: tuple[str, ...]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_pick_x_offset(args: argparse.Namespace, job: Job) -> float:
    offset = float(args.waypoint_pick_x_offset)
    if args.waypoint_max_pick_world_x is None:
        return offset
    try:
        task = read_json(job.task_file)
        pick_x = float(task["steps"][job.step_index]["pick_pose"]["position"]["x"])
    except (KeyError, IndexError, TypeError, ValueError):
        return offset
    capped = float(args.waypoint_max_pick_world_x) - pick_x
    return max(0.0, min(offset, capped))


def episode_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("ep_") and name[3:].isdigit():
        return (int(name[3:]), name)
    return (10**9, name)


def discover_jobs(args: argparse.Namespace) -> list[Job]:
    pattern = "ep_*" if args.ep_only else "*"
    episode_dirs = sorted(
        [
            path
            for path in args.moveit_dir.glob(pattern)
            if path.is_dir() and (path / "mtc_task.json").exists()
        ],
        key=episode_sort_key,
    )
    if args.episode:
        wanted = set(args.episode)
        episode_dirs = [path for path in episode_dirs if path.name in wanted]

    jobs: list[Job] = []
    for episode_dir in episode_dirs:
        task_file = episode_dir / "mtc_task.json"
        if not task_file.exists():
            continue
        steps = read_json(task_file).get("steps", [])
        if not steps:
            continue
        step_indices: Iterable[int]
        if args.all_steps:
            step_indices = range(len(steps))
        else:
            step_indices = [args.step_index]
        for step_index in step_indices:
            output = episode_dir / f"step_{step_index}_trajectory.json"
            if args.skip_existing and output.exists():
                continue
            jobs.append(
                Job(
                    episode_id=episode_dir.name,
                    step_index=step_index,
                    total_steps=len(steps),
                    task_file=task_file,
                    output_trajectory=output,
                )
            )
    return jobs


def ros2_launch_command(args: argparse.Namespace, job: Job) -> list[str]:
    pick_x_offset = effective_pick_x_offset(args, job)
    launch_args = [
        "ros2",
        "launch",
        args.launch_package,
        args.launch_file,
        f"task_file:={job.task_file}",
        f"step_index:={job.step_index}",
        f"execute:={str(args.execute).lower()}",
        f"output_trajectory:={job.output_trajectory}",
        f"planner_mode:={args.planner_mode}",
        f"plan_gripper_motion:={str(args.plan_gripper_motion).lower()}",
        f"use_waypoint_collision_scene:={str(args.use_waypoint_collision_scene).lower()}",
        f"waypoint_hover_z_offset:={args.waypoint_hover_z_offset}",
        f"waypoint_pick_z_offset:={args.waypoint_pick_z_offset}",
        f"waypoint_place_z_offset:={args.waypoint_place_z_offset}",
        f"waypoint_pick_x_offset:={pick_x_offset}",
        f"waypoint_pick_y_offset:={args.waypoint_pick_y_offset}",
        f"waypoint_place_x_offset:={args.waypoint_place_x_offset}",
        f"waypoint_place_y_offset:={args.waypoint_place_y_offset}",
        f"waypoint_pre_grasp_extra_z_offset:={args.waypoint_pre_grasp_extra_z_offset}",
        f"waypoint_post_release_lift_z_offset:={args.waypoint_post_release_lift_z_offset}",
        f"target_grasp_collision_scale:={args.target_grasp_collision_scale}",
        f"waypoint_velocity_scaling:={args.waypoint_velocity_scaling}",
        f"waypoint_acceleration_scaling:={args.waypoint_acceleration_scaling}",
        f"waypoint_point_dt:={args.waypoint_point_dt}",
        f"waypoint_grasp_hold_points:={args.waypoint_grasp_hold_points}",
        f"waypoint_secure_hold_points:={args.waypoint_secure_hold_points}",
        f"waypoint_release_hold_points:={args.waypoint_release_hold_points}",
        f"max_solutions:={args.max_solutions}",
    ]
    if args.source_setup and args.source_setup.exists():
        quoted = " ".join(subprocess.list2cmdline([part]) for part in launch_args)
        return ["bash", "-lc", f"source {subprocess.list2cmdline([str(args.source_setup)])} && {quoted}"]
    return launch_args


def tail(text: str, lines: int) -> str:
    if lines <= 0:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def print_heartbeat(args: argparse.Namespace, label: str, started: float) -> None:
    if args.heartbeat <= 0:
        return
    elapsed = time.monotonic() - started
    if args.timeout > 0:
        remaining = max(0.0, args.timeout - elapsed)
        print(
            f"[running] {label} elapsed={format_duration(elapsed)} "
            f"timeout_left={format_duration(remaining)}",
            flush=True,
        )
    else:
        print(f"[running] {label} elapsed={format_duration(elapsed)}", flush=True)


def trajectory_payload(path: Path) -> dict:
    payload = read_json(path)
    if "points" in payload:
        return payload
    if "trajectory" in payload:
        return payload["trajectory"]
    if "joint_trajectory" in payload:
        return payload["joint_trajectory"]
    return payload


def analyze_trajectory(args: argparse.Namespace, path: Path) -> TrajectoryStats:
    trajectory = trajectory_payload(path)
    points = list(trajectory.get("points", []))
    duration = 0.0
    stage_counts: Counter[str] = Counter()
    for point in points:
        try:
            duration = max(duration, float(point.get("time_from_start", 0.0)))
        except (TypeError, ValueError):
            pass
        stage_counts[str(point.get("stage_name", "unknown"))] += 1

    flags: list[str] = []
    if len(points) < args.min_trajectory_points:
        flags.append(f"too_few_points:{len(points)}<{args.min_trajectory_points}")
    if len(points) > args.max_trajectory_points:
        flags.append(f"too_many_points:{len(points)}>{args.max_trajectory_points}")
    if duration < args.min_trajectory_duration:
        flags.append(f"too_short_duration:{duration:.2f}<{args.min_trajectory_duration:.2f}")
    if duration > args.max_trajectory_duration:
        flags.append(f"too_long_duration:{duration:.2f}>{args.max_trajectory_duration:.2f}")
    return TrajectoryStats(
        point_count=len(points),
        duration=duration,
        stage_counts=dict(stage_counts),
        flags=tuple(flags),
    )


def extract_failure_stage(text: str) -> str | None:
    match = re.search(r"Waypoint planning failed at ([A-Za-z0-9_:-]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"Task planning failed at ([A-Za-z0-9_:-]+)", text)
    if match:
        return match.group(1)
    return None


def log_has_runner_failure(text: str) -> bool:
    failure_patterns = (
        "Waypoint planning failed at ",
        "Task planning failed at ",
        "MoveGroupInterface::plan() failed",
        "process has died",
    )
    return any(pattern in text for pattern in failure_patterns)


def run_job(args: argparse.Namespace, job: Job, label: str) -> tuple[bool, str, float]:
    job.output_trajectory.parent.mkdir(parents=True, exist_ok=True)
    command = ros2_launch_command(args, job)
    started = time.monotonic()
    log_path = job.output_trajectory.with_suffix(".log")
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    next_heartbeat = started + args.heartbeat if args.heartbeat > 0 else float("inf")
    timed_out = False
    try:
        while process.poll() is None:
            now = time.monotonic()
            if args.timeout > 0 and now - started > args.timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGINT)
                break
            if now >= next_heartbeat:
                print_heartbeat(args, label, started)
                next_heartbeat = now + args.heartbeat
            time.sleep(0.25)

        if timed_out:
            deadline = time.monotonic() + 5.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.25)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    finally:
        log_file.close()

    elapsed = time.monotonic() - started
    stdout = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if timed_out:
        return False, f"timeout after {args.timeout:.1f}s {elapsed:.1f}s log={log_path}\n{tail(stdout, args.failure_tail_lines)}", elapsed
    if log_has_runner_failure(stdout):
        if not args.keep_failed_trajectories:
            job.output_trajectory.unlink(missing_ok=True)
        return False, f"runner reported planning failure; launch_exit={process.returncode} {elapsed:.1f}s log={log_path}\n{tail(stdout, args.failure_tail_lines)}", elapsed
    if not args.write_logs and process.returncode == 0 and job.output_trajectory.exists():
        log_path.unlink(missing_ok=True)
    if process.returncode == 0 and job.output_trajectory.exists():
        return True, f"ok {elapsed:.1f}s", elapsed
    if process.returncode == 0:
        return False, f"no trajectory written; launch_exit=0 {elapsed:.1f}s log={log_path}\n{tail(stdout, args.failure_tail_lines)}", elapsed
    return False, f"exit={process.returncode} {elapsed:.1f}s log={log_path}\n{tail(stdout, args.failure_tail_lines)}", elapsed


def progress_iter(jobs: list[Job]):
    if tqdm is not None:
        return tqdm(jobs, total=len(jobs), unit="step", dynamic_ncols=True)
    return jobs


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def runtime_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def build_summary(
    args: argparse.Namespace,
    jobs: list[Job],
    started_all: float,
    successful_runtimes: list[float],
    failed_runtimes: list[float],
    point_counts: list[int],
    trajectory_durations: list[float],
    failure_stages: Counter[str],
    outlier_trajectories: list[dict[str, str | int | float | list[str] | dict[str, int]]],
) -> dict:
    total_elapsed = time.monotonic() - started_all
    return {
        "total_discovered_jobs": len(jobs),
        "attempted_jobs": len(successful_runtimes) + len(failed_runtimes),
        "successful_jobs": len(successful_runtimes),
        "failed_jobs": len(failed_runtimes),
        "outlier_trajectory_count": len(outlier_trajectories),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed": format_duration(total_elapsed),
        "runtime_seconds": {
            "successful": runtime_stats(successful_runtimes),
            "failed": runtime_stats(failed_runtimes),
        },
        "trajectory_points": runtime_stats([float(value) for value in point_counts]),
        "trajectory_duration_seconds": runtime_stats(trajectory_durations),
        "failure_stage_counts": dict(failure_stages),
        "outlier_trajectories": outlier_trajectories,
        "thresholds": {
            "min_trajectory_points": args.min_trajectory_points,
            "max_trajectory_points": args.max_trajectory_points,
            "min_trajectory_duration": args.min_trajectory_duration,
            "max_trajectory_duration": args.max_trajectory_duration,
        },
    }


def write_run_files(
    args: argparse.Namespace,
    jobs: list[Job],
    started_all: float,
    successful_runtimes: list[float],
    failed_runtimes: list[float],
    point_counts: list[int],
    trajectory_durations: list[float],
    failure_stages: Counter[str],
    outlier_trajectories: list[dict[str, str | int | float | list[str] | dict[str, int]]],
    failures: list[dict[str, str | int | float | None]],
) -> tuple[Path, Path, dict]:
    failures_file = args.failures_file or (args.moveit_dir / "planning_failures.json")
    summary_file = args.summary_file or (args.moveit_dir / "planning_summary.json")
    summary = build_summary(
        args,
        jobs,
        started_all,
        successful_runtimes,
        failed_runtimes,
        point_counts,
        trajectory_durations,
        failure_stages,
        outlier_trajectories,
    )
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        failures_file.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return summary_file, failures_file, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moveit-dir", type=Path, default=DEFAULT_MOVEIT_DIR)
    parser.add_argument("--episode", action="append", help="Episode id, e.g. ep_037. Can be repeated.")
    parser.add_argument(
        "--ep-only",
        action="store_true",
        help="Only scan ep_* directories. By default, scan every directory that contains mtc_task.json.",
    )
    parser.add_argument("--all-steps", action="store_true", help="Plan every step in each mtc_task.json.")
    parser.add_argument("--step-index", type=int, default=0, help="Used when --all-steps is not set.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds per episode step. Use <=0 for no timeout.")
    parser.add_argument("--max-solutions", type=int, default=5)
    parser.add_argument("--planner-mode", choices=("waypoint", "mtc"), default="waypoint")
    parser.add_argument("--waypoint-hover-z-offset", type=float, default=0.20)
    parser.add_argument("--waypoint-pick-z-offset", type=float, default=0.060)
    parser.add_argument("--waypoint-place-z-offset", type=float, default=0.22)
    parser.add_argument("--waypoint-pick-x-offset", type=float, default=0.04)
    parser.add_argument(
        "--waypoint-max-pick-world-x",
        type=float,
        default=None,
        help=(
            "Optional safety cap for pick_pose.x + waypoint_pick_x_offset. "
            "Useful for large batches where far-table objects would otherwise "
            "push pre-grasp outside the Panda workspace."
        ),
    )
    parser.add_argument("--waypoint-pick-y-offset", type=float, default=0.0)
    parser.add_argument("--waypoint-place-x-offset", type=float, default=0.0)
    parser.add_argument("--waypoint-place-y-offset", type=float, default=0.0)
    parser.add_argument("--waypoint-pre-grasp-extra-z-offset", type=float, default=0.08)
    parser.add_argument("--waypoint-post-release-lift-z-offset", type=float, default=0.34)
    parser.add_argument("--target-grasp-collision-scale", type=float, default=0.45)
    parser.add_argument("--waypoint-velocity-scaling", type=float, default=0.08)
    parser.add_argument("--waypoint-acceleration-scaling", type=float, default=0.08)
    parser.add_argument("--waypoint-point-dt", type=float, default=0.16)
    parser.add_argument("--waypoint-grasp-hold-points", type=int, default=18)
    parser.add_argument("--waypoint-secure-hold-points", type=int, default=12)
    parser.add_argument("--waypoint-release-hold-points", type=int, default=18)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-gripper-motion", action="store_true")
    parser.add_argument("--use-waypoint-collision-scene", dest="use_waypoint_collision_scene", action="store_true", default=True)
    parser.add_argument("--no-use-waypoint-collision-scene", dest="use_waypoint_collision_scene", action="store_false")
    parser.add_argument("--launch-package", default="vlapb_mtc_runner")
    parser.add_argument("--launch-file", default="vlapb_mtc_runner.launch.py")
    parser.add_argument("--source-setup", type=Path, default=DEFAULT_SOURCE_SETUP)
    parser.add_argument("--write-logs", action="store_true", help="Write .log file for successful jobs too.")
    parser.add_argument("--keep-failed-trajectories", action="store_true", help="Keep partial/stale trajectory JSON files when the runner reports failure.")
    parser.add_argument("--heartbeat", type=float, default=10.0, help="Print per-step elapsed/timeout status every N seconds when tqdm is unavailable.")
    parser.add_argument("--failure-tail-lines", type=int, default=12)
    parser.add_argument("--failures-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--min-trajectory-points", type=int, default=50)
    parser.add_argument("--max-trajectory-points", type=int, default=2000)
    parser.add_argument("--min-trajectory-duration", type=float, default=5.0)
    parser.add_argument("--max-trajectory-duration", type=float, default=180.0)
    parser.add_argument("--summary-every", type=int, default=25, help="Write planning_summary.json every N attempted jobs. Use <=0 to only write at exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = discover_jobs(args)
    if not jobs:
        print("No MoveIt planning jobs found.")
        return

    failures: list[dict[str, str | int | float | None]] = []
    outlier_trajectories: list[dict[str, str | int | float | list[str] | dict[str, int]]] = []
    successful_runtimes: list[float] = []
    failed_runtimes: list[float] = []
    point_counts: list[int] = []
    trajectory_durations: list[float] = []
    failure_stages: Counter[str] = Counter()
    print(f"planning jobs: {len(jobs)} step(s)")
    iterator = progress_iter(jobs)
    started_all = time.monotonic()
    interrupted = False
    try:
        job_iterator = enumerate(iterator, start=1)
        for completed, job in job_iterator:
            label = f"{job.episode_id} step {job.step_index}/{job.total_steps - 1}"
            if tqdm is not None:
                iterator.set_description(label)
            else:
                elapsed = time.monotonic() - started_all
                if completed > 1:
                    rate = elapsed / (completed - 1)
                    eta = rate * (len(jobs) - completed + 1)
                    print(
                        f"[plan {completed}/{len(jobs)}] {label} "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                    )
                else:
                    print(f"[plan {completed}/{len(jobs)}] {label} elapsed=0s eta=calculating")
            ok, message, runtime = run_job(args, job, label)
            if tqdm is not None:
                iterator.set_postfix_str(message.splitlines()[0][:80])
            else:
                elapsed = time.monotonic() - started_all
                rate = elapsed / completed
                eta = rate * (len(jobs) - completed)
                print(
                    f"[done {completed}/{len(jobs)}] {label}: {message.splitlines()[0]} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                )
            if not ok:
                failed_runtimes.append(runtime)
                log_path = job.output_trajectory.with_suffix(".log")
                log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else message
                failure_stage = extract_failure_stage(log_text)
                if failure_stage:
                    failure_stages[failure_stage] += 1
                failure = {
                    "episode_id": job.episode_id,
                    "step_index": job.step_index,
                    "task_file": str(job.task_file),
                    "output_trajectory": str(job.output_trajectory),
                    "runtime_seconds": runtime,
                    "failure_stage": failure_stage,
                    "error": message,
                }
                failures.append(failure)
                print(f"[failed] {label}: {message}")
                if not args.continue_on_error:
                    break
            else:
                successful_runtimes.append(runtime)
                try:
                    stats = analyze_trajectory(args, job.output_trajectory)
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    stats = TrajectoryStats(0, 0.0, {}, (f"analysis_failed:{exc}",))
                point_counts.append(stats.point_count)
                trajectory_durations.append(stats.duration)
                if stats.flags:
                    outlier = {
                        "episode_id": job.episode_id,
                        "step_index": job.step_index,
                        "trajectory": str(job.output_trajectory),
                        "point_count": stats.point_count,
                        "duration_seconds": stats.duration,
                        "stage_counts": stats.stage_counts,
                        "flags": list(stats.flags),
                    }
                    outlier_trajectories.append(outlier)
                    print(
                        f"[outlier] {label}: points={stats.point_count} "
                        f"duration={stats.duration:.2f}s flags={','.join(stats.flags)}",
                        flush=True,
                    )
            if args.summary_every > 0 and completed % args.summary_every == 0:
                summary_file, _, summary = write_run_files(
                    args,
                    jobs,
                    started_all,
                    successful_runtimes,
                    failed_runtimes,
                    point_counts,
                    trajectory_durations,
                    failure_stages,
                    outlier_trajectories,
                    failures,
                )
                print(
                    f"[summary] attempted={summary['attempted_jobs']} ok={summary['successful_jobs']} "
                    f"failed={summary['failed_jobs']} outliers={summary['outlier_trajectory_count']} "
                    f"-> {summary_file}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted: writing summary before exit", flush=True)

    summary_file, failures_file, summary = write_run_files(
        args,
        jobs,
        started_all,
        successful_runtimes,
        failed_runtimes,
        point_counts,
        trajectory_durations,
        failure_stages,
        outlier_trajectories,
        failures,
    )
    print(
        "summary: "
        f"ok={len(successful_runtimes)} failed={len(failed_runtimes)} "
        f"outliers={len(outlier_trajectories)} elapsed={format_duration(total_elapsed)} "
        f"-> {summary_file}",
        flush=True,
    )
    if failures:
        failures_file.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"failures: {len(failures)} -> {failures_file}")
        raise SystemExit(1)
    if interrupted:
        raise SystemExit(130)
    print("all planning jobs completed")


if __name__ == "__main__":
    main()
