from __future__ import annotations

from dataclasses import dataclass, field

SIDES = ("left", "right")


@dataclass(frozen=True)
class VRHand:
    tracked: bool = False
    position_m: tuple[float, float, float] | None = None
    quaternion_xyzw: tuple[float, float, float, float] | None = None
    trigger: float = 0.0
    squeeze: float = 0.0


@dataclass(frozen=True)
class VRFrame:
    session_id: str
    seq: int
    client_timestamp_ms: float
    received_monotonic_s: float
    hands: dict[str, VRHand] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmObservation:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    gripper_open_fraction: float
    joint_positions_rad: tuple[float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class ArmTarget:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    gripper_open_fraction: float | None = None


@dataclass(frozen=True)
class ArmCommand:
    position_mm: tuple[float, float, float]
    rotation_vector_rad: tuple[float, float, float]
    gripper_open_fraction: float | None = None


@dataclass(frozen=True)
class JointState:
    timestamp_s: float
    angles_rad: tuple[float, float, float, float, float, float]
    feedback_hz: float = 0.0


@dataclass(frozen=True)
class IKRequest:
    request_id: int
    target: ArmCommand
    seed_joints_rad: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class IKResult:
    request_id: int
    target: ArmCommand
    joints_rad: tuple[float, float, float, float, float, float] | None


@dataclass(frozen=True)
class AppliedArmCommand:
    command: ArmCommand
    pose_written: bool
    gripper_written: bool
    joint_target_rad: tuple[float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class CommandResult:
    observation: dict[str, ArmObservation]
    applied: dict[str, AppliedArmCommand] = field(default_factory=dict)

    @property
    def hardware_write(self) -> bool:
        return any(item.pose_written or item.gripper_written for item in self.applied.values())
