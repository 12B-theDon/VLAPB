#!/usr/bin/env python3
"""Launch OpenPI pi0.5 fine-tuning on VLAPB episodes converted to LeRobot."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, extend_command, print_or_run


OPENPI_ROOT = Path("/home/artemis/Documents/openpi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune OpenPI pi0.5 on VLAPB episode data.")
    add_common_args(parser, default_dataset_name="physical-intelligence/libero")
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--exp-name", default="vlapb_pi05")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-norm-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args passed to OpenPI scripts/train.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"}
    if args.compute_norm_stats:
        norm_cmd = ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", args.config_name]
        print_or_run(norm_cmd, cwd=OPENPI_ROOT, dry_run=args.dry_run, env=env)

    cmd = [
        "uv",
        "run",
        "scripts/train.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        f"--data.repo-id={args.dataset_name}",
        f"--checkpoint-base-dir={args.run_root_dir / 'openpi'}",
        f"--num-train-steps={args.max_steps}",
    ]
    if args.batch_size is not None:
        cmd.append(f"--batch-size={args.batch_size}")
    if args.learning_rate is not None:
        cmd.append(f"--lr-schedule.peak-lr={args.learning_rate}")
        cmd.append(f"--lr-schedule.decay-lr={args.learning_rate}")
    if args.save_freq is not None:
        cmd.append(f"--save-interval={args.save_freq}")
    if args.overwrite:
        cmd.append("--overwrite")
    extend_command(cmd, args.extra_args)
    print_or_run(cmd, cwd=OPENPI_ROOT, dry_run=args.dry_run, env=env)


if __name__ == "__main__":
    main()
