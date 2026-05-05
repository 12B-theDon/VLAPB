#!/usr/bin/env python3
"""Launch OpenVLA-OFT fine-tuning on generated VLAPB episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, extend_command, print_or_run


OPENVLA_OFT_ROOT = Path("/home/artemis/Documents/openvla-oft")
DEFAULT_BASE_MODEL = "openvla/openvla-7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA-OFT on VLAPB episode data.")
    add_common_args(parser)
    parser.add_argument("--vla-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-l1-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-diffusion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-image-aug", dest="image_aug", action="store_false")
    parser.set_defaults(image_aug=True)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args passed to OpenVLA-OFT finetune.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size or 8
    learning_rate = args.learning_rate or 5e-4
    save_freq = args.save_freq or 10_000
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
        str(args.run_root_dir / "openvla_oft"),
        "--batch_size",
        str(batch_size),
        "--max_steps",
        str(args.max_steps),
        "--save_freq",
        str(save_freq),
        "--learning_rate",
        str(learning_rate),
        "--shuffle_buffer_size",
        str(args.shuffle_buffer_size),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_dropout",
        str(args.lora_dropout),
        "--num_images_in_input",
        str(args.num_images_in_input),
        "--run_id_note",
        args.run_id_note,
        "--wandb_project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.use_proprio:
        cmd.extend(["--use_proprio", "True"])
    if not args.use_l1_regression:
        cmd.extend(["--use_l1_regression", "False"])
    if args.use_diffusion:
        cmd.extend(["--use_diffusion", "True"])
    if not args.image_aug:
        cmd.extend(["--image_aug", "False"])
    extend_command(cmd, args.extra_args)
    print_or_run(cmd, cwd=OPENVLA_OFT_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

