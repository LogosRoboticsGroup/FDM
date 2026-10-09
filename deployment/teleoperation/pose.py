from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .config import VRConfig
from .contracts import ArmCommand, ArmObservation, ArmTarget, VRHand


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def normalize_xyzw(quaternion: tuple[float, ...]) -> np.ndarray:
    quat = _finite(np.asarray(quaternion), "quaternion")
    if quat.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("quaternion norm is zero")
    return quat / norm


def xyzw_to_wxyz(quaternion: tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, z, w = normalize_xyzw(quaternion)
    return float(w), float(x), float(y), float(z)


def wxyz_to_xyzw(quaternion: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = _finite(np.asarray(quaternion), "quaternion")
    return normalize_xyzw((x, y, z, w))


def wxyz_to_rotation_6d(quaternion: tuple[float, ...]) -> np.ndarray:
    """Encode a quaternion as the first two columns of its rotation matrix."""
    matrix = Rotation.from_quat(wxyz_to_xyzw(quaternion)).as_matrix()
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rotation_6d_to_wxyz(values: np.ndarray) -> tuple[float, float, float, float]:
    """Project a continuous 6D rotation representation onto SO(3)."""
    values = _finite(np.asarray(values), "rotation_6d")
    if values.shape != (6,):
        raise ValueError("rotation_6d must have shape (6,)")

    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("rotation_6d first axis is degenerate")
    first = first / first_norm

    second = values[3:] - float(np.dot(first, values[3:])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ValueError("rotation_6d axes are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    x, y, z, w = Rotation.from_matrix(np.column_stack((first, second, third))).as_quat()
    return float(w), float(x), float(y), float(z)


def axis_angle_to_rpy_degrees(rotation_vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in Rotation.from_rotvec(rotation_vector).as_euler("xyz", degrees=True))


def rpy_degrees_to_axis_angle(rpy_degrees: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in Rotation.from_euler("xyz", rpy_degrees, degrees=True).as_rotvec())


def j6_to_tcp(position_mm: np.ndarray, rotation_vector: np.ndarray, offset_mm: tuple[float, ...]) -> np.ndarray:
    return np.asarray(position_mm) + Rotation.from_rotvec(rotation_vector).apply(offset_mm)


def tcp_to_j6(position_mm: np.ndarray, rotation_vector: np.ndarray, offset_mm: tuple[float, ...]) -> np.ndarray:
    return np.asarray(position_mm) - Rotation.from_rotvec(rotation_vector).apply(offset_mm)


def observation_to_command(
    observation: ArmObservation,
    *,
    gripper_open_fraction: float | None,
) -> ArmCommand:
    rotation = Rotation.from_quat(wxyz_to_xyzw(observation.quaternion_wxyz)).as_rotvec()
    return ArmCommand(
        position_mm=tuple(value * 1000.0 for value in observation.position_m),
        rotation_vector_rad=tuple(float(value) for value in rotation),
        gripper_open_fraction=gripper_open_fraction,
    )


class PoseFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._state: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def update(self, side: str, position: np.ndarray, quaternion_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        previous = self._state.get(side)
        if previous is None:
            result = position.copy(), quaternion_xyzw.copy()
        else:
            prev_position, prev_quaternion = previous
            position = prev_position * (1 - self.alpha) + position * self.alpha
            rotations = Rotation.from_quat(np.stack([prev_quaternion, quaternion_xyzw]))
            quaternion_xyzw = Slerp([0, 1], rotations)(self.alpha).as_quat()
            result = position, quaternion_xyzw
        self._state[side] = result
        return result

    def reset(self, side: str | None = None) -> None:
        if side is None:
            self._state.clear()
        else:
            self._state.pop(side, None)


@dataclass
class _Anchor:
    vr_position: np.ndarray
    vr_rotation: Rotation
    robot_position: np.ndarray
    robot_rotation: Rotation


class RelativePoseMapper:
    def __init__(self, config: VRConfig):
        self.config = config
        self._anchors: dict[str, _Anchor] = {}
        self._filter = PoseFilter(config.smoothing_alpha)
        self._vr_to_robot = Rotation.from_matrix(np.asarray(config.vr_to_robot, dtype=np.float64))
        self._grip_rotation = Rotation.from_euler("x", config.grip_pitch_deg, degrees=True)

    def reset(self, side: str | None = None) -> None:
        if side is None:
            self._anchors.clear()
        else:
            self._anchors.pop(side, None)
        self._filter.reset(side)

    def map(
        self,
        side: str,
        hand: VRHand,
        observation: ArmObservation,
    ) -> ArmTarget | None:
        if not hand.tracked or hand.position_m is None or hand.quaternion_xyzw is None:
            self.reset(side)
            return None

        position = _finite(np.asarray(hand.position_m), "VR position")
        quaternion = normalize_xyzw(hand.quaternion_xyzw)
        position, quaternion = self._filter.update(side, position, quaternion)
        vr_position = self._vr_to_robot.apply(position)
        vr_rotation = self._vr_to_robot * self._grip_rotation * Rotation.from_quat(quaternion) * self._vr_to_robot.inv()
        robot_rotation = Rotation.from_quat(wxyz_to_xyzw(observation.quaternion_wxyz))

        anchor = self._anchors.get(side)
        if anchor is None:
            self._anchors[side] = _Anchor(
                vr_position=vr_position,
                vr_rotation=vr_rotation,
                robot_position=np.asarray(observation.position_m),
                robot_rotation=robot_rotation,
            )
            return None

        delta_position = (vr_position - anchor.vr_position) * np.asarray(self.config.position_scale)
        delta_rotation = vr_rotation * anchor.vr_rotation.inv()
        scaled_rotation = Rotation.from_rotvec(delta_rotation.as_rotvec() * self.config.rotation_scale)
        target_rotation = scaled_rotation * anchor.robot_rotation
        target_position = anchor.robot_position + delta_position
        x, y, z, w = target_rotation.as_quat()
        return ArmTarget(
            position_m=tuple(float(value) for value in target_position),
            quaternion_wxyz=(float(w), float(x), float(y), float(z)),
            gripper_open_fraction=1.0 - hand.squeeze,
        )
