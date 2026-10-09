from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpaceMouseState:
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    buttons: frozenset[int] = frozenset()
    received_monotonic_s: float = 0.0
    connected: bool = False

    @property
    def axes(self) -> tuple[float, float, float, float, float, float]:
        return (*self.translation, *self.rotation)


@dataclass(frozen=True)
class FlexivObservation:
    tcp_pose_wxyz: tuple[float, float, float, float, float, float, float]
    joint_positions_rad: tuple[float, float, float, float, float, float, float]
    gripper_width_m: float


@dataclass(frozen=True)
class FlexivTarget:
    tcp_pose_wxyz: tuple[float, float, float, float, float, float, float]
    gripper_width_m: float | None = None


@dataclass(frozen=True)
class FlexivCommandResult:
    observation: FlexivObservation
    target: FlexivTarget | None = None
    pose_written: bool = False
    gripper_written: bool = False
    control_mode_recovered: bool = False

    @property
    def hardware_write(self) -> bool:
        return self.pose_written or self.gripper_written
