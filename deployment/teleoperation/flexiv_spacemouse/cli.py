from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace

from .config import load_flexiv_spacemouse_config
from .replay import FlexivReplayRuntime, load_flexiv_replay_episode
from .robot import FlexivRobot
from .runtime import FlexivSpaceMouseRuntime
from .spacemouse import SpaceMouseReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-Flexiv SpaceMouse teleoperation recorder")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "record", "replay", "spacemouse-monitor", "probe"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
    record = subparsers.choices["record"]
    record.add_argument("--root", help="new or existing LeRobot dataset directory")
    record.add_argument("--repo-id", help="LeRobot dataset repo id stored in metadata")
    record.add_argument("--task", help="natural-language task written on every frame")
    record.add_argument("--fps", type=int, help="recording FPS override")
    record.add_argument("--resume", action="store_true", help="append compatible episodes")
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
        help="enable the Flexiv arm and execute the recorded joint trajectory",
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
        help="leave the arm at the final recorded target after a completed replay",
    )
    monitor = subparsers.choices["spacemouse-monitor"]
    monitor.add_argument("--samples", type=int, default=0, help="stop after N reports; 0 means Ctrl-C")
    monitor.add_argument("--poll-hz", type=float, default=20.0)
    subparsers.choices["probe"].add_argument("--samples", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_flexiv_spacemouse_config(args.config)

    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2))
        return 0

    if args.command == "spacemouse-monitor":
        if args.poll_hz <= 0:
            raise ValueError("--poll-hz must be positive")
        reader = SpaceMouseReader(config.spacemouse)
        count = 0
        try:
            reader.open()
            while not args.samples or count < args.samples:
                state = reader.snapshot()
                print(
                    json.dumps(
                        {
                            "translation": [round(value, 4) for value in state.translation],
                            "rotation": [round(value, 4) for value in state.rotation],
                            "buttons": sorted(state.buttons),
                            "device": str(reader.device_path),
                        }
                    ),
                    flush=True,
                )
                count += 1
                time.sleep(1.0 / args.poll_hz)
        except KeyboardInterrupt:
            pass
        finally:
            reader.close()
        return 0

    if args.command == "probe":
        if not config.runtime.hardware_access or config.runtime.motion_enabled:
            raise ValueError("probe requires hardware_access=true and motion_enabled=false")
        robot = FlexivRobot(config.robot)
        try:
            robot.open(enable_motion=False)
            for _ in range(args.samples):
                print(json.dumps(asdict(robot.read_observation())))
                time.sleep(0.05)
        finally:
            robot.close()
        return 0

    if args.command == "replay":
        root = args.root if args.root is not None else config.recording.root
        repo_id = args.repo_id if args.repo_id is not None else config.recording.repo_id
        if root is None or not root.strip():
            raise ValueError("replay requires --root or recording.root")
        if not repo_id.strip():
            raise ValueError("replay repo id must not be empty")
        episode = load_flexiv_replay_episode(
            root=root,
            repo_id=repo_id,
            episode_index=args.episode,
            validate_images=args.validate_images,
        )
        episode.validate_for_execution(
            config.robot,
            start_tolerance_deg=args.start_tolerance_deg,
            max_joint_step_deg=args.max_joint_step_deg,
        )
        if not args.execute:
            print(json.dumps(episode.summary(executed=False), indent=2))
            return 0
        if not config.runtime.hardware_access or not config.runtime.motion_enabled:
            raise ValueError(
                "replay --execute requires hardware_access=true and motion_enabled=true"
            )
        runtime = FlexivReplayRuntime(config, episode, speed=args.speed)
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
    if config.runtime.hardware_access and not config.runtime.motion_enabled:
        raise ValueError("record with hardware_access=true requires motion_enabled=true")
    if not recording.cameras:
        logging.getLogger(__name__).warning(
            "No cameras are configured; the dataset will contain only robot state/action features"
        )
    runtime = FlexivSpaceMouseRuntime(config)
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
