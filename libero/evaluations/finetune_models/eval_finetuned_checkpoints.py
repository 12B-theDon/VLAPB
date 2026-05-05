#!/usr/bin/env python3
"""Run VLAPB compare-injection evaluation with fine-tuned checkpoints."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1] / "compare_injection_methods"
DEFAULT_CHECKPOINTS = {
    "octo": Path("/home/artemis/Documents/octo"),
    "openvla_oft": Path("/home/artemis/Documents/openvla-oft"),
    "openvla": Path("/home/artemis/Documents/openvla"),
    "pi05": Path("/home/artemis/Documents/openpi"),
}
EVAL_SCRIPTS = {
    "octo": EVAL_ROOT / "evaluation_octo.py",
    "openvla_oft": EVAL_ROOT / "evaluation_openvla_OFT.py",
    "openvla": EVAL_ROOT / "evaluation_openvla.py",
    "pi05": EVAL_ROOT / "evaluation_pi05.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLAPB checkpoints.")
    parser.add_argument("--models", nargs="+", default=list(EVAL_SCRIPTS), choices=list(EVAL_SCRIPTS))
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Common root containing model checkpoint dirs.")
    parser.add_argument("--octo-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["octo"])
    parser.add_argument("--openvla-oft-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["openvla_oft"])
    parser.add_argument("--openvla-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["openvla"])
    parser.add_argument("--pi05-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["pi05"])
    parser.add_argument("--manifest", type=Path, default=EVAL_ROOT / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=EVAL_ROOT / "finetuned_checkpoint_results")
    parser.add_argument("--limit-trials", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=["plain", "textual", "visual", "visual_textual"])
    parser.add_argument("--save-videos", default="first-per-mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args appended to every eval script.")
    return parser.parse_args()


def checkpoint_for(args: argparse.Namespace, model: str) -> Path:
    if args.checkpoint_root is not None:
        names = {
            "octo": "octo",
            "openvla_oft": "openvla-oft",
            "openvla": "openvla",
            "pi05": "openpi",
        }
        return args.checkpoint_root / names[model]
    return {
        "octo": args.octo_checkpoint,
        "openvla_oft": args.openvla_oft_checkpoint,
        "openvla": args.openvla_checkpoint,
        "pi05": args.pi05_checkpoint,
    }[model]


def main() -> None:
    args = parse_args()
    for model in args.models:
        output_dir = args.output_dir / model
        cmd = [
            sys.executable,
            str(EVAL_SCRIPTS[model]),
            "--checkpoint",
            str(checkpoint_for(args, model)),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(output_dir),
            "--limit-trials",
            str(args.limit_trials),
            "--modes",
            *args.modes,
            "--save-videos",
            args.save_videos,
        ]
        cmd.extend(args.extra_args)
        print(" ".join(shlex.quote(part) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(EVAL_ROOT), check=True)


if __name__ == "__main__":
    main()

