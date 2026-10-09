"""ARX entry point sharing the dual-arm inference protocol and controls."""

from .piper_cli import main as dual_arm_main


def main(argv: list[str] | None = None) -> int:
    return dual_arm_main(argv, expected_robot_type="arx_x5")


if __name__ == "__main__":
    raise SystemExit(main())
