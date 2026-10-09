from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .config import FlexivRobotConfig, SpaceMouseConfig
from .contracts import FlexivObservation, FlexivTarget, SpaceMouseState


class SpaceMousePoseMapper:
    def __init__(self, mouse_config: SpaceMouseConfig, robot_config: FlexivRobotConfig):
        self.mouse_config = mouse_config
        self.robot_config = robot_config
        self._translation_to_world = np.asarray(mouse_config.translation_to_world, dtype=np.float64)
        self._rotation_to_world = np.asarray(mouse_config.rotation_to_world, dtype=np.float64)
        self._workspace_min = np.asarray(robot_config.workspace_min_m, dtype=np.float64)
        self._workspace_max = np.asarray(robot_config.workspace_max_m, dtype=np.float64)
        self._active = False
        self._filtered_axes = np.zeros(6, dtype=np.float64)
        self._target_position: np.ndarray | None = None
        self._target_rotation: Rotation | None = None

    @property
    def active(self) -> bool:
        return self._active

    def reset(self, observation: FlexivObservation | None = None) -> None:
        self._active = False
        self._filtered_axes.fill(0.0)
        if observation is None:
            self._target_position = None
            self._target_rotation = None
        else:
            pose = observation.tcp_pose_wxyz
            self._target_position = np.asarray(pose[:3], dtype=np.float64)
            self._target_rotation = Rotation.from_quat((pose[4], pose[5], pose[6], pose[3]))

    def is_neutral(self, state: SpaceMouseState) -> bool:
        return max(abs(value) for value in state.axes) <= self.mouse_config.release_threshold

    def update_activity(self, state: SpaceMouseState) -> bool:
        level = max(abs(value) for value in state.axes)
        if self._active:
            self._active = level > self.mouse_config.release_threshold
        else:
            self._active = level >= self.mouse_config.activation_threshold
        if not self._active:
            self._filtered_axes.fill(0.0)
        return self._active

    def map(
        self,
        state: SpaceMouseState,
        observation: FlexivObservation,
        dt_s: float,
        *,
        gripper_width_m: float | None = None,
    ) -> FlexivTarget | None:
        if not self.update_activity(state):
            return None
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        dt_s = min(dt_s, 0.05)

        raw = np.asarray(state.axes, dtype=np.float64)
        deadzone = self.mouse_config.deadzone
        axes = np.sign(raw) * np.maximum(0.0, np.abs(raw) - deadzone) / (1.0 - deadzone)
        alpha = self.mouse_config.smoothing_alpha
        self._filtered_axes = (1.0 - alpha) * self._filtered_axes + alpha * axes

        if self._target_position is None or self._target_rotation is None:
            self.reset(observation)
            self._active = True
        assert self._target_position is not None and self._target_rotation is not None

        linear_velocity = (
            self._translation_to_world @ self._filtered_axes[:3]
        ) * self.mouse_config.translation_speed_m_s
        angular_velocity = (
            self._rotation_to_world @ self._filtered_axes[3:]
        ) * self.mouse_config.rotation_speed_rad_s
        candidate_position = np.clip(
            self._target_position + linear_velocity * dt_s,
            self._workspace_min,
            self._workspace_max,
        )
        delta_rotation = Rotation.from_rotvec(angular_velocity * dt_s)
        candidate_rotation = (
            self._target_rotation * delta_rotation
            if self.mouse_config.rotation_frame == "tool"
            else delta_rotation * self._target_rotation
        )

        feedback_position = np.asarray(observation.tcp_pose_wxyz[:3], dtype=np.float64)
        lead = candidate_position - feedback_position
        lead_norm = float(np.linalg.norm(lead))
        if lead_norm > self.robot_config.max_position_lead_m:
            candidate_position = (
                feedback_position + lead * self.robot_config.max_position_lead_m / lead_norm
            )

        feedback_pose = observation.tcp_pose_wxyz
        feedback_rotation = Rotation.from_quat(
            (feedback_pose[4], feedback_pose[5], feedback_pose[6], feedback_pose[3])
        )
        rotation_lead = candidate_rotation * feedback_rotation.inv()
        rotation_vector = rotation_lead.as_rotvec()
        rotation_lead_norm = float(np.linalg.norm(rotation_vector))
        if rotation_lead_norm > self.robot_config.max_rotation_lead_rad:
            candidate_rotation = Rotation.from_rotvec(
                rotation_vector * self.robot_config.max_rotation_lead_rad / rotation_lead_norm
            ) * feedback_rotation

        self._target_position = candidate_position
        self._target_rotation = candidate_rotation
        qx, qy, qz, qw = candidate_rotation.as_quat()
        return FlexivTarget(
            tcp_pose_wxyz=(
                *(float(value) for value in candidate_position),
                float(qw),
                float(qx),
                float(qy),
                float(qz),
            ),
            gripper_width_m=gripper_width_m,
        )
