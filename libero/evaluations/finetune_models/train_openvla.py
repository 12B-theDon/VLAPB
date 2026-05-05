#!/usr/bin/env python3
"""Launch OpenVLA LoRA fine-tuning on generated VLAPB episodes.

The generated episodes should be converted/registered as an RLDS dataset before
running this launcher, because upstream OpenVLA fine-tuning consumes RLDS data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, extend_command, print_or_run


OPENVLA_ROOT = Path("/home/artemis/Documents/openvla")
DEFAULT_BASE_MODEL = "openvla/openvla-7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA on VLAPB episode data.")
    add_common_args(parser)
    parser.add_argument("--vla-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-image-aug", dest="image_aug", action="store_false")
    parser.set_defaults(image_aug=True)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args passed to OpenVLA finetune.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size or 16
    learning_rate = args.learning_rate or 5e-4
    save_steps = args.save_freq or 5_000
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(args.num_gpus),
        "vla-scripts/finetune.py",
        "--vla_path",
        args.vla_path,
        "--data_root_dir",
        str(args.episodes_dir),
        "--dataset_name",
        args.dataset_name,
        "--run_root_dir",
        str(args.run_root_dir / "openvla"),
        "--batch_size",
        str(batch_size),
        "--max_steps",
        str(args.max_steps),
        "--save_steps",
        str(save_steps),
        "--learning_rate",
        str(learning_rate),
        "--shuffle_buffer_size",
        str(args.shuffle_buffer_size),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_dropout",
        str(args.lora_dropout),
        "--run_id_note",
        args.run_id_note,
        "--wandb_project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if not args.image_aug:
        cmd.extend(["--image_aug", "False"])
    extend_command(cmd, args.extra_args)
    print_or_run(cmd, cwd=OPENVLA_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

