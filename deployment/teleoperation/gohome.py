from __future__ import annotations

import argparse
import json
import logging
import math

from .config import load_config
from .piper import DualPiper

DEFAULT_HOME_TIMEOUT_S = 15.0
DEFAULT_HOME_TOLERANCE_DEG = 1.0


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move both configured Piper arms to their configured home joints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timeout-s", type=_positive_finite_float, default=DEFAULT_HOME_TIMEOUT_S)
    parser.add_argument("--tolerance-deg", type=_positive_finite_float, default=DEFAULT_HOME_TOLERANCE_DEG)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if config.robot_type != "piper":
        raise ValueError("use the ARX teleoperation home controls for ARX configs")
    if not config.runtime.hardware_access or not config.runtime.motion_enabled:
        raise ValueError("go-home requires hardware_access=true and motion_enabled=true")

    robot = DualPiper(config.arms)
    try:
        robot.open_read_only()
        robot.prepare_motion()
        final_feedback = robot.go_home(tolerance_deg=args.tolerance_deg, timeout_s=args.timeout_s)
    except KeyboardInterrupt:
        return 130
    finally:
        robot.close()

    print(
        json.dumps(
            {
                "status": "home_reached",
                "target_joint_degrees": {side: list(arm.home_joints_deg) for side, arm in config.arms.items()},
                "target_gripper_open_fraction": {
                    side: arm.home_gripper_open_fraction for side, arm in config.arms.items()
                },
                "final_joint_degrees": {
                    side: list(feedback.angles_degrees) for side, feedback in final_feedback.items()
                },
                "tolerance_degrees": args.tolerance_deg,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
