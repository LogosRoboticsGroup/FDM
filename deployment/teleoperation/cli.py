from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace

from .config import load_config
from .contracts import ArmTarget
from .piper import validate_socketcan_interfaces
from .replay import PiperReplayRuntime, load_replay_episode
from .robot import create_robot
from .runtime import TeleoperationRuntime, synthetic_vr_frames
from .vr.monitor import run_vr_monitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StarVLA dual-arm WebXR teleoperation")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in (
        "validate-config",
        "dry-run",
        "teleoperate",
        "record",
        "replay",
        "probe",
        "ik-preview",
        "vr-monitor",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
    subparsers.choices["dry-run"].add_argument("--synthetic-vr-frames", type=int, default=100)
    subparsers.choices["teleoperate"].add_argument(
        "--no-keyboard",
        action="store_true",
        help="disable keyboard input",
    )
    record = subparsers.choices["record"]
    record.add_argument("--root", help="new or existing LeRobot dataset directory")
    record.add_argument("--repo-id", help="LeRobot dataset repo id stored in metadata")
    record.add_argument("--task", help="natural-language task written on every frame")
    record.add_argument("--fps", type=int, help="recording FPS override")
    record.add_argument(
        "--resume",
        action="store_true",
        help="append episodes to an existing compatible dataset",
    )
    record.add_argument(
        "--no-keyboard",
        action="store_true",
        help="disable keyboard input; episode events must then come from WebXR",
    )
    replay = subparsers.choices["replay"]
    replay.add_argument("--root", help="local LeRobot dataset directory")
    replay.add_argument("--repo-id", help="LeRobot dataset repo id stored in metadata")
    replay.add_argument("--episode", type=int, default=0, help="episode index to validate or replay")
    replay.add_argument(
        "--validate-images",
        action="store_true",
        help="decode every recorded camera frame during validation",
    )
    replay.add_argument(
        "--execute",
        action="store_true",
        help="enable both arms and execute the recorded joint targets",
    )
    replay.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="replay speed multiplier in (0, 1]",
    )
    replay.add_argument(
        "--start-tolerance-deg",
        type=float,
        default=5.0,
        help="maximum first-target distance from configured home for each joint",
    )
    replay.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=10.0,
        help="maximum allowed recorded per-frame joint step",
    )
    replay.add_argument(
        "--no-return-home",
        action="store_true",
        help="leave the arms at the final recorded target after a completed replay",
    )
    subparsers.choices["probe"].add_argument("--samples", type=int, default=20)
    subparsers.choices["ik-preview"].add_argument("--samples", type=int, default=20)
    subparsers.choices["vr-monitor"].add_argument(
        "--output-format",
        choices=("human", "json"),
        default="human",
    )
    subparsers.choices["vr-monitor"].add_argument("--poll-hz", type=float, default=500.0)
    subparsers.choices["vr-monitor"].add_argument(
        "--samples",
        type=int,
        default=0,
        help="stop after N frames; 0 means run until Ctrl-C",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if config.robot_type == "arx_x5" and args.command in {"probe", "ik-preview"}:
        raise ValueError(
            "ARX probe/ik-preview are not supported by this CLI; " "the SDK controller constructor writes to motors"
        )

    if args.command == "record":
        recording = replace(
            config.recording,
            enabled=True,
            root=args.root if args.root is not None else config.recording.root,
            repo_id=args.repo_id if args.repo_id is not None else config.recording.repo_id,
            task=args.task if args.task is not None else config.recording.task,
            fps=args.fps if args.fps is not None else config.recording.fps,
            resume=args.resume or config.recording.resume,
        )
        config = replace(config, recording=recording)
        config.validate()
        if not recording.cameras:
            logging.getLogger(__name__).warning(
                "Recording has no cameras configured; the dataset will contain only low-dimensional "
                "state/action features."
            )

    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2))
        return 0

    if args.command == "replay":
        root = args.root if args.root is not None else config.recording.root
        repo_id = args.repo_id if args.repo_id is not None else config.recording.repo_id
        if root is None or not root.strip():
            raise ValueError("replay requires --root or recording.root")
        if not repo_id.strip():
            raise ValueError("replay repo id must not be empty")
        episode = load_replay_episode(
            root=root,
            repo_id=repo_id,
            episode_index=args.episode,
            validate_images=args.validate_images,
        )
        episode.validate_for_execution(
            config.arms,
            start_tolerance_deg=args.start_tolerance_deg,
            max_joint_step_deg=args.max_joint_step_deg,
            robot_type=config.robot_type,
        )
        runtime = PiperReplayRuntime(config, episode, speed=args.speed)
        if not args.execute:
            try:
                runtime.validate_joint_limits()
            finally:
                runtime.close()
            print(json.dumps(episode.summary(executed=False), indent=2))
            return 0
        if not config.runtime.hardware_access or not config.runtime.motion_enabled:
            raise ValueError("replay --execute requires hardware_access=true and motion_enabled=true")
        completed = False
        try:
            runtime.open()
            runtime.run()
            completed = True
        except KeyboardInterrupt:
            logging.getLogger(__name__).warning(
                "Replay interrupted; stopping command transmission without automatically going home"
            )
        finally:
            runtime.close(return_home=completed and not args.no_return_home)
        print(json.dumps(episode.summary(executed=completed), indent=2))
        return 0

    if args.command == "dry-run":
        if config.runtime.hardware_access or config.runtime.motion_enabled:
            raise ValueError("dry-run requires hardware_access=false and motion_enabled=false")
        runtime = TeleoperationRuntime(config, start_vr_server=False, keyboard=False)
        try:
            runtime.open()
            cycles = runtime.run(
                max_cycles=args.synthetic_vr_frames,
                frames=synthetic_vr_frames(args.synthetic_vr_frames),
            )
        finally:
            runtime.close()
        print(
            json.dumps(
                {
                    "cycles": cycles,
                    "hardware_access": False,
                    "motion_enabled": False,
                    "recording_enabled": config.recording.enabled,
                    "piper_sdk_imported": "piper_sdk" in sys.modules,
                    "arx_sdk_imported": "arx5_interface" in sys.modules,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "vr-monitor":
        run_vr_monitor(
            config,
            output_format=args.output_format,
            poll_hz=args.poll_hz,
            max_frames=args.samples or None,
        )
        return 0

    if args.command == "probe":
        if not config.runtime.hardware_access or config.runtime.motion_enabled:
            raise ValueError("probe requires hardware_access=true and motion_enabled=false")
        robot = create_robot(config)
        try:
            robot.open_read_only()
            for _ in range(args.samples):
                print(robot.read_observation())
        finally:
            robot.close()
        return 0

    if args.command == "ik-preview":
        if not config.runtime.hardware_access or config.runtime.motion_enabled:
            raise ValueError("ik-preview requires hardware_access=true and motion_enabled=false")
        if any(arm.control_backend != "host_ik" for arm in config.arms.values()):
            raise ValueError("ik-preview requires host_ik on both arms")
        robot = create_robot(config)
        try:
            robot.start_kinematics()
            robot.open_read_only()
            for _ in range(args.samples):
                observation = robot.read_observation()
                targets = {
                    side: ArmTarget(
                        position_m=current.position_m,
                        quaternion_wxyz=current.quaternion_wxyz,
                        gripper_open_fraction=current.gripper_open_fraction,
                    )
                    for side, current in observation.items()
                }
                _, joints, plan = robot.plan_targets_read_only(targets)
                print(
                    json.dumps(
                        {
                            "feedback_joints": {side: state.angles_rad for side, state in joints.items()},
                            "solved_joints": plan.joints,
                        }
                    )
                )
                time.sleep(0.05)
        finally:
            robot.close()
        return 0

    if config.runtime.hardware_access and not config.runtime.motion_enabled:
        raise ValueError("read-only hardware access is available only through probe or ik-preview")
    if config.runtime.hardware_access:
        try:
            validate_socketcan_interfaces(
                [arm.can_interface for arm in config.arms.values()],
                require_error_active=True,
            )
        except ConnectionError as exc:
            logging.getLogger(__name__).error("Piper CAN preflight failed: %s", exc)
            return 2
    runtime = TeleoperationRuntime(config, keyboard=not args.no_keyboard)
    try:
        runtime.open()
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
