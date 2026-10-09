from __future__ import annotations

from typing import Any

import numpy as np

from ..config import RecordingConfig
from ..pose import wxyz_to_rotation_6d
from ..recording.lerobot_recorder import LeRobotRecorder
from .config import DEFAULT_GRIPPER_SCALE
from .contracts import FlexivCommandResult, FlexivObservation

EEF_NAMES = ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper")
JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper")


def _joint_vector(observation: FlexivObservation, gripper_scale: float = DEFAULT_GRIPPER_SCALE) -> np.ndarray:
    return np.asarray(
        (*observation.joint_positions_rad, observation.gripper_width_m * gripper_scale),
        dtype=np.float32,
    )


def _eef_vector(
    pose_wxyz: tuple[float, ...],
    gripper_width_m: float,
    gripper_scale: float = DEFAULT_GRIPPER_SCALE,
) -> np.ndarray:
    position = pose_wxyz[:3]
    rotation_6d = wxyz_to_rotation_6d(pose_wxyz[3:])
    return np.asarray((*position, *rotation_6d, gripper_width_m * gripper_scale), dtype=np.float32)


def build_flexiv_frame(
    result: FlexivCommandResult,
    images: dict[str, np.ndarray],
    *,
    next_joint_action: np.ndarray | None = None,
    gripper_scale: float = DEFAULT_GRIPPER_SCALE,
) -> dict[str, np.ndarray]:
    observation = result.observation
    state_joint = _joint_vector(observation, gripper_scale)
    action_joint = state_joint.copy()
    if next_joint_action is not None:
        action_joint = np.asarray(next_joint_action, dtype=np.float32).copy()
        action_joint[-1] *= gripper_scale
    target = result.target
    action_pose = observation.tcp_pose_wxyz if target is None else target.tcp_pose_wxyz
    action_gripper = (
        observation.gripper_width_m if target is None or target.gripper_width_m is None else target.gripper_width_m
    )
    frame = {
        "observation.state.eef_pose": _eef_vector(
            observation.tcp_pose_wxyz,
            observation.gripper_width_m,
            gripper_scale,
        ),
        "observation.state.joint_position": state_joint,
        "actions.eef_pose": _eef_vector(action_pose, action_gripper, gripper_scale),
        "actions.joint_position": action_joint,
        "observation.state": state_joint.copy(),
        "action": action_joint.copy(),
    }
    frame.update({f"observation.images.{name}": image for name, image in images.items()})
    return frame


class FlexivLeRobotRecorder(LeRobotRecorder):
    """LeRobot writer matching the repository's existing single-Flexiv schema.

    Joint actions are delayed by one sample: frame t receives the observed joint
    state from frame t+1. This matches convert_flexiv_to_lerobot.py and avoids
    labelling Cartesian-driven motion with the same-frame feedback.
    """

    ROBOT_TYPE = "flexiv_spacemouse"

    def __init__(
        self,
        config: RecordingConfig,
        execution_mode: str,
        *,
        gripper_scale: float = DEFAULT_GRIPPER_SCALE,
    ) -> None:
        self._pending_frame: dict[str, np.ndarray] | None = None
        self.gripper_scale = gripper_scale
        super().__init__(config, execution_mode)

    def start_episode(self) -> None:
        super().start_episode()
        self._pending_frame = None

    def add_frame(self, result: FlexivCommandResult, images: dict[str, np.ndarray]) -> None:
        self._raise_if_failed()
        if not self._active:
            return
        current_joint = _joint_vector(result.observation, self.gripper_scale)
        if self._pending_frame is not None:
            self._pending_frame["actions.joint_position"] = current_joint.copy()
            self._pending_frame["action"] = current_joint.copy()
            self._enqueue_frame(self._pending_frame)
        self._pending_frame = build_flexiv_frame(result, images, gripper_scale=self.gripper_scale)

    def finish_episode(self, save: bool = True) -> None:
        if self._active and self._pending_frame is not None:
            # The terminal sample has no t+1 observation, so it holds its own state.
            self._enqueue_frame(self._pending_frame)
            self._pending_frame = None
        try:
            super().finish_episode(save=save)
        finally:
            self._pending_frame = None

    def _features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.state.eef_pose": {
                "dtype": "float32",
                "shape": (10,),
                "names": [list(EEF_NAMES)],
            },
            "observation.state.joint_position": {
                "dtype": "float32",
                "shape": (8,),
                "names": [list(JOINT_NAMES)],
            },
            "actions.eef_pose": {
                "dtype": "float32",
                "shape": (10,),
                "names": [list(EEF_NAMES)],
            },
            "actions.joint_position": {
                "dtype": "float32",
                "shape": (8,),
                "names": [list(JOINT_NAMES)],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [list(JOINT_NAMES)],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": [list(JOINT_NAMES)],
            },
        }
        for name, camera in self.config.cameras.items():
            features[f"observation.images.{name}"] = {
                "dtype": "video" if self.config.video else "image",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _teleoperation_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "hardware_executed": self.execution_mode == "hardware",
            "video_encoding_workers": self.config.video_encoding_workers,
            "episode_start_semantics": "first SpaceMouse displacement after home and a neutral sample",
            "episode_finish_semantics": "Left Arrow discards; Right Arrow saves; capture stops before home motion",
            "joint_state_order": list(JOINT_NAMES),
            "eef_state_order": list(EEF_NAMES),
            "joint_unit": "rad; gripper width in m multiplied by gripper_scale",
            "gripper_scale": self.gripper_scale,
            "translation_unit": "m",
            "rotation_unit": "continuous 6D (first two rotation-matrix columns)",
            "joint_action_semantics": "next recorded joint observation (one-sample shift); terminal sample holds",
            "eef_action_semantics": "SpaceMouse Cartesian target; inactive frames hold feedback pose",
        }
