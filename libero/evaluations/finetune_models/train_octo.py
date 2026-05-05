#!/usr/bin/env python3
"""Launch Octo fine-tuning on generated VLAPB episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, extend_command, print_or_run


OCTO_ROOT = Path("/home/artemis/Documents/octo")
DEFAULT_PRETRAINED = "/home/artemis/Documents/octo/checkpoints/octo-base-1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Octo on VLAPB episode data.")
    add_common_args(parser, default_dataset_name="vlapb_episodes")
    parser.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED)
    parser.add_argument("--pretrained-step", type=int, default=None)
    parser.add_argument("--config", default="scripts/configs/finetune_config.py:full,language_conditioned")
    parser.add_argument("--group", default="vlapb")
    parser.add_argument("--name", default="vlapb_octo")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args passed to Octo finetune.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size or 128
    save_interval = args.save_freq or 5_000
    cmd = [
        "python3",
        "scripts/finetune.py",
        f"--name={args.name}",
        f"--config={args.config}",
        f"--config.pretrained_path={args.pretrained_path}",
        f"--config.batch_size={batch_size}",
        f"--config.num_steps={args.max_steps}",
        f"--config.save_interval={save_interval}",
        f"--config.save_dir={args.run_root_dir / 'octo'}",
        f"--config.dataset_kwargs.name={args.dataset_name}",
        f"--config.dataset_kwargs.data_dir={args.episodes_dir}",
        f"--config.wandb.project={args.wandb_project}",
        f"--config.wandb.group={args.group}",
    ]
    if args.pretrained_step is not None:
        cmd.append(f"--config.pretrained_step={args.pretrained_step}")
    if args.learning_rate is not None:
        cmd.append(f"--config.optimizer.learning_rate.peak_value={args.learning_rate}")
    if args.wandb_entity:
        cmd.append(f"--config.wandb.entity={args.wandb_entity}")
    extend_command(cmd, args.extra_args)
    print_or_run(cmd, cwd=OCTO_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
