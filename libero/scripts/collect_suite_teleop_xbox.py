#!/usr/bin/env python3
"""Xbox controller teleop entrypoint for VLAPB suite demonstrations."""

from __future__ import annotations

import argparse
import select
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import robosuite.utils.transform_utils as T


VLAPB_LIBERO_ROOT = Path("/home/artemis/Documents/VLAPB/libero")
TOOLS_DIR = VLAPB_LIBERO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from teleop_moveit_tools import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUITES_ROOT,
    collect_suite_data,
    create_processed_hdf5,
    smoke_suite_bddl,
)


class LinuxXboxController:
    """Xbox-style teleop device using Linux joystick events."""

    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80
    EVENT_STRUCT = struct.Struct("IhBB")

    def __init__(
        self,
        device_path: str = "/dev/input/js0",
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        deadzone: float = 0.12,
        pos_step: float = 0.004,
        rot_step: float = 0.004,
        gripper_button: int = 0,
        reset_button: int = 6,
    ):
        self.device_path = device_path
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.deadzone = deadzone
        self.pos_step = pos_step
        self.rot_step = rot_step
        self.gripper_button = gripper_button
        self.reset_button = reset_button

        self.axes: dict[int, float] = {}
        self.buttons: dict[int, int] = {}
        self.grasp = False
        self._reset_state = 0
        self._enabled = False
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._lock = threading.Lock()
        self._stop = False

        self._display_controls()
        self._file = open(device_path, "rb", buffering=0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _display_controls() -> None:
        def print_command(control: str, info: str) -> None:
            control += " " * max(1, 24 - len(control))
            print(f"{control}\t{info}")

        print("")
        print_command("Xbox control", "Command")
        print_command("Left stick", "move arm horizontally in x-y plane")
        print_command("Right stick up/down", "move arm vertically")
        print_command("Right stick left/right", "rotate arm about z-axis")
        print_command("LB / RB", "rotate arm about x-axis")
        print_command("D-pad up/down", "rotate arm about y-axis")
        print_command("A", "toggle gripper open/close")
        print_command("Back", "reset/cancel rollout")
        print("")

    def _axis(self, index: int) -> float:
        value = self.axes.get(index, 0.0)
        return 0.0 if abs(value) < self.deadzone else value

    def _reset_internal_state(self) -> None:
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    def start_control(self) -> None:
        with self._lock:
            self._reset_internal_state()
            self._reset_state = 0
            self._enabled = True

    def get_controller_state(self) -> dict[str, Any]:
        with self._lock:
            lx = self._axis(0)
            ly = self._axis(1)
            rx = self._axis(3)
            ry = self._axis(4)
            dpad_x = self._axis(6)
            dpad_y = self._axis(7)
            lb = float(self.buttons.get(4, 0))
            rb = float(self.buttons.get(5, 0))
            grasp = self.grasp
            reset = self._reset_state

        dpos = np.array(
            [
                ly * self.pos_step * self.pos_sensitivity,
                lx * self.pos_step * self.pos_sensitivity,
                -ry * self.pos_step * self.pos_sensitivity,
            ]
        )
        raw_drotation = np.array(
            [
                (rb - lb + dpad_x) * self.rot_step * self.rot_sensitivity,
                -dpad_y * self.rot_step * self.rot_sensitivity,
                rx * self.rot_step * self.rot_sensitivity,
            ]
        )

        roll, pitch, yaw = raw_drotation
        drot1 = T.rotation_matrix(angle=-pitch, direction=[1.0, 0.0, 0.0], point=None)[:3, :3]
        drot2 = T.rotation_matrix(angle=roll, direction=[0.0, 1.0, 0.0], point=None)[:3, :3]
        drot3 = T.rotation_matrix(angle=yaw, direction=[0.0, 0.0, 1.0], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": raw_drotation,
            "grasp": int(grasp),
            "reset": reset,
        }

    def _run(self) -> None:
        while not self._stop:
            try:
                readable, _, _ = select.select([self._file], [], [], 0.1)
                if not readable:
                    continue
                payload = self._file.read(self.EVENT_STRUCT.size)
            except (ValueError, OSError):
                break

            if len(payload) != self.EVENT_STRUCT.size:
                continue

            _event_time, value, event_type, number = self.EVENT_STRUCT.unpack(payload)
            event_type &= ~self.JS_EVENT_INIT

            with self._lock:
                if event_type == self.JS_EVENT_AXIS:
                    self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
                elif event_type == self.JS_EVENT_BUTTON:
                    previous = self.buttons.get(number, 0)
                    pressed = int(value)
                    self.buttons[number] = pressed

                    if self._enabled and number == self.gripper_button and pressed and not previous:
                        self.grasp = not self.grasp

                    if self._enabled and number == self.reset_button and pressed and not previous:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    def close(self) -> None:
        self._stop = True
        try:
            self._file.close()
        except Exception:
            pass

        if self._thread.is_alive():
            self._thread.join(timeout=0.2)


def xbox_preflight(args: argparse.Namespace) -> None:
    if not Path(args.xbox_device).exists():
        raise RuntimeError(
            f"Xbox teleop device does not exist: {args.xbox_device}. "
            "Plug in the controller and check /dev/input/js*."
        )


def make_xbox_device(_env: Any, args: argparse.Namespace) -> LinuxXboxController:
    return LinuxXboxController(
        device_path=args.xbox_device,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        deadzone=args.xbox_deadzone,
        pos_step=args.xbox_pos_step,
        rot_step=args.xbox_rot_step,
        gripper_button=args.xbox_gripper_button,
        reset_button=args.xbox_reset_button,
    )


def add_common_collect_args(collect: argparse.ArgumentParser) -> None:
    collect.add_argument("--suites-root", type=Path, default=DEFAULT_SUITES_ROOT)
    collect.add_argument("--episode", dest="episode_id", type=str, default=None)
    collect.add_argument("--suite", choices=("belongings", "placements", "sequences"), default=None)
    collect.add_argument("--split", type=str, default=None)
    collect.add_argument("--all", dest="collect_all", action="store_true")
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--offset", type=int, default=0)
    collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    collect.add_argument("--raw-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "raw")
    collect.add_argument("--overwrite", action="store_true")
    collect.add_argument("--device", choices=("xbox",), default="xbox")
    collect.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    collect.add_argument("--config", type=str, default="single-arm-opposed")
    collect.add_argument("--arm", type=str, default="right")
    collect.add_argument("--camera", type=str, default="agentview")
    collect.add_argument("--controller", type=str, default="OSC_POSE")
    collect.add_argument("--pos-sensitivity", type=float, default=1.5)
    collect.add_argument("--rot-sensitivity", type=float, default=1.0)
    collect.add_argument("--xbox-device", type=str, default="/dev/input/js0")
    collect.add_argument("--xbox-deadzone", type=float, default=0.12)
    collect.add_argument("--xbox-pos-step", type=float, default=0.004)
    collect.add_argument("--xbox-rot-step", type=float, default=0.004)
    collect.add_argument("--xbox-gripper-button", type=int, default=0)
    collect.add_argument("--xbox-reset-button", type=int, default=6)
    collect.add_argument("--cap-index", type=int, default=5)
    collect.add_argument("--demos-per-episode", type=int, default=1)
    collect.add_argument("--use-depth", action="store_true")
    collect.add_argument("--no-proprio", action="store_true")
    collect.add_argument("--validate", action="store_true")
    collect.add_argument("--render-video", action="store_true")
    collect.add_argument("--preview-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "previews")
    collect.add_argument("--preview-demo", type=str, default="demo_0")
    collect.add_argument("--preview-camera-key", type=str, default="agentview_rgb")
    collect.add_argument("--preview-fps", type=int, default=20)
    collect.add_argument("--preview-stride", type=int, default=1)
    collect.add_argument("--overwrite-preview", action="store_true")
    collect.add_argument("--validation-min-steps", type=int, default=10)
    collect.add_argument("--require-single-demo", action="store_true")
    collect.add_argument("--debug-teleop", action="store_true", help="Print live EEF, target, fixed object, goal, and success state.")
    collect.add_argument("--debug-every", type=int, default=20, help="Print teleop debug state every N environment steps.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-bddl")
    smoke.add_argument("--suites-root", type=Path, default=DEFAULT_SUITES_ROOT)
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "smoke_renders")
    smoke.add_argument("--limit", type=int, default=None)
    smoke.add_argument("--episode", dest="episode_id", type=str, default=None)
    smoke.add_argument("--suite", choices=("belongings", "placements", "sequences"), default=None)
    smoke.add_argument("--split", type=str, default=None)
    smoke.add_argument("--camera-name", type=str, default="agentview")
    smoke.add_argument("--image-size", type=int, default=256)
    smoke.add_argument("--settle-steps", type=int, default=3)
    smoke.add_argument("--max-reset-attempts", type=int, default=25)

    convert = subparsers.add_parser("convert-raw")
    convert.add_argument("--raw-demo", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--no-camera-obs", action="store_true")
    convert.add_argument("--use-depth", action="store_true")
    convert.add_argument("--no-proprio", action="store_true")
    convert.add_argument("--cap-index", type=int, default=5)

    collect = subparsers.add_parser("collect")
    add_common_collect_args(collect)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "smoke-bddl":
        smoke_suite_bddl(
            suites_root=args.suites_root,
            output_dir=args.output_dir,
            limit=args.limit,
            episode_id=args.episode_id,
            suite_name=args.suite,
            split=args.split,
            camera_name=args.camera_name,
            image_size=args.image_size,
            settle_steps=args.settle_steps,
            max_reset_attempts=args.max_reset_attempts,
        )
    elif args.command == "convert-raw":
        output = create_processed_hdf5(
            raw_demo_path=args.raw_demo,
            output_path=args.output,
            use_camera_obs=not args.no_camera_obs,
            use_depth=args.use_depth,
            no_proprio=args.no_proprio,
            cap_index=args.cap_index,
        )
        print(f"saved processed dataset: {output}")
    elif args.command == "collect":
        if args.demos_per_episode != 1:
            raise NotImplementedError("--demos-per-episode is reserved for a later expansion; use 1 for now.")
        collect_suite_data(args, make_xbox_device, xbox_preflight)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
