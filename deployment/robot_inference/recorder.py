from __future__ import annotations

import time
from typing import Any

import numpy as np

from deployment.teleoperation.config import RecordingConfig
from deployment.teleoperation.flexiv_spacemouse.recorder import (
    EEF_NAMES as FLEXIV_EEF_NAMES,
)
from deployment.teleoperation.recording.lerobot_recorder import EEF_NAMES, JOINT_NAMES, LeRobotRecorder


class InferenceLeRobotRecorder(LeRobotRecorder):
    """Record live observations and held hardware targets on a continuous timeline."""

    CAPTURE_VERSION = 2

    def __init__(
        self,
        config: RecordingConfig,
        *,
        robot_type: str,
        action_space: str,
        gripper_scale: float = 1.0,
    ) -> None:
        self.ROBOT_TYPE = f"{robot_type}_inference"
        self.action_space = action_space
        self.is_dual_arm = robot_type in {"dual_piper", "dual_arx_x5"}
        self.is_flexiv = robot_type == "flexiv"
        self.gripper_scale = gripper_scale
        self._started_at = 0.0
        super().__init__(config, execution_mode="hardware")

    def start_episode(self) -> None:
        super().start_episode()
        self._started_at = time.monotonic()

    def add_frame(self, result: dict[str, np.ndarray], images: dict[str, np.ndarray]) -> None:
        self._raise_if_failed()
        if not self._active:
            return
        frame = {name: np.asarray(value).copy() for name, value in result.items()}
        if self.is_flexiv:
            suffix = "eef_pose" if self.action_space == "cartesian" else "joint_position"
            for name in (f"observation.state.{suffix}", f"actions.{suffix}"):
                frame[name][-1] *= self.gripper_scale
        if self.is_dual_arm:
            suffix = "eef" if self.action_space == "cartesian" else "joint"
            frame["observation.state"] = frame[f"state.{suffix}"].copy()
            frame["action"] = frame[f"action.{suffix}"].copy()
        elif self.is_flexiv and self.action_space == "cartesian":
            frame["observation.state"] = frame["observation.state.eef_pose"].copy()
            frame["action"] = frame["actions.eef_pose"].copy()
        else:
            frame["observation.state"] = frame["observation.state.joint_position"].copy()
            frame["action"] = frame["actions.joint_position"].copy()
        frame["observation.elapsed_s"] = np.asarray([time.monotonic() - self._started_at], dtype=np.float32)
        frame.update({name: image.copy() for name, image in images.items()})
        self._enqueue_frame(frame)

    def _features(self) -> dict[str, dict[str, Any]]:
        if self.is_dual_arm:
            fields = {
                "state.joint": JOINT_NAMES,
                "action.joint": JOINT_NAMES,
                "state.eef": EEF_NAMES,
                "action.eef": EEF_NAMES,
                "action.executed": ("left", "right"),
            }
            names = EEF_NAMES if self.action_space == "cartesian" else JOINT_NAMES
        elif self.is_flexiv and self.action_space == "cartesian":
            names = FLEXIV_EEF_NAMES
            fields = {"observation.state.eef_pose": names, "actions.eef_pose": names}
        else:
            names = (*[f"joint{index}_rad" for index in range(1, 8)], "gripper_width_m")
            fields = {"observation.state.joint_position": names, "actions.joint_position": names}
        fields.update({"observation.state": names, "action": names, "observation.elapsed_s": ("seconds",)})
        features = {
            key: {"dtype": "float32", "shape": (len(names),), "names": list(names)} for key, names in fields.items()
        }
        for name, camera in self.config.cameras.items():
            features[f"observation.images.{name}"] = {
                "dtype": "video" if self.config.video else "image",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _teleoperation_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "execution_mode": "hardware",
            "hardware_executed": True,
            "source": "synchronous_policy_inference",
            "inference_capture_version": self.CAPTURE_VERSION,
            "action_space": self.action_space,
            "state_semantics": "live feedback at each periodic recording sample",
            "action_semantics": (
                "last dispatched robot targets held between writes; initial hold uses episode-start feedback"
            ),
            "capture_semantics": (
                "continuous fixed-fps sampling including inference waits; pause and home motion excluded"
            ),
            "timestamp_semantics": (
                "LeRobot time uses sample index/fps; observation.elapsed_s stores actual capture time"
            ),
            "episode_start_semantics": "Enter or left-controller X after hardware readiness",
            "episode_finish_semantics": "Right/VR A save and home; Left/VR B discard and home; q/Esc quit",
            "joint_unit": (
                "rad; dual-arm gripper is normalized"
                if self.is_dual_arm
                else "rad; Flexiv gripper width in m multiplied by gripper_scale"
            ),
            "translation_unit": "mm" if self.is_dual_arm else "m",
            "rotation_unit": (
                "continuous_6d" if self.is_flexiv and self.action_space == "cartesian" else "axis_angle_rad"
            ),
        }
        if self.is_flexiv:
            metadata["gripper_scale"] = self.gripper_scale
        return metadata

    def _validate_resumed_dataset(self, dataset: Any, expected_features: dict[str, dict[str, Any]]) -> None:
        super()._validate_resumed_dataset(dataset, expected_features)
        capture_version = dataset.meta.info.get("teleoperation", {}).get("inference_capture_version")
        if capture_version != self.CAPTURE_VERSION:
            raise ValueError(
                "cannot resume an inference dataset with different capture timing; use a new recording.root"
            )
