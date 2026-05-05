#!/usr/bin/env python3
"""Shared helpers for VLAPB fine-tuning launchers."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


VLAPB_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = VLAPB_ROOT / "VLAPB_suites"
DEFAULT_RUN_ROOT = VLAPB_ROOT / "libero" / "evaluations" / "finetune_models" / "runs"


def add_common_args(parser: argparse.ArgumentParser, *, default_dataset_name: str = "vlapb_episodes") -> None:
    parser.add_argument("--episodes-dir", type=Path, required=True, help="Directory containing generated episode data.")
    parser.add_argument("--dataset-name", default=default_dataset_name, help="Dataset name registered inside the model repo.")
    parser.add_argument("--run-root-dir", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--run-id-note", default="vlapb")
    parser.add_argument("--wandb-project", default="vlapb_finetune")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print command without launching training.")


def extend_command(cmd: list[str], values: list[str] | None) -> list[str]:
    if values:
        cmd.extend(values)
    return cmd


def print_or_run(cmd: list[str], *, cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"cwd: {cwd}")
    print(rendered)
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=True)

