from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.teleoperation.contracts import (
    SIDES,
    AppliedArmCommand,
    ArmCommand,
    ArmObservation,
    ArmTarget,
    CommandResult,
)
from deployment.teleoperation.pose import wxyz_to_xyzw, xyzw_to_wxyz
from deployment.teleoperation.recording.cameras import CameraHub
from deployment.teleoperation.recording.lerobot_recorder import build_frame
from deployment.teleoperation.robot import create_robot

from .client import StarVLAZmqClient
from .config import DualPiperInferenceConfig, validate_policy_metadata
from .synchronous import SynchronousInferenceRuntime

logger = logging.getLogger(__name__)
PIPER_DIM_PER_ARM = 7


class DualPiperInferenceRuntime(SynchronousInferenceRuntime):
    """Closed-loop StarVLA inference for a left/right Piper or ARX X5 pair.

    The client sends the recorder's 14-D EEF or joint state according to
    ``inference.action_space``. Returned chunks are dispatched through host IK
    or directly through the selected robot joint controller, respectively.
    """

    ROBOT_TYPE = "dual_piper"

    def __init__(
        self,
        config: DualPiperInferenceConfig,
        *,
        robot: Any | None = None,
        cameras: Any | None = None,
        client: Any | None = None,
        controls: Any | None = None,
        recorder: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        if config.robot_type == "arx_x5":
            self.ROBOT_TYPE = "dual_arx_x5"
        policy = config.inference
        self.robot = robot if robot is not None else create_robot(config)
        self.cameras = cameras if cameras is not None else CameraHub(config.cameras)
        self.client = (
            client
            if client is not None
            else StarVLAZmqClient(
                policy.server,
                camera_order=policy.camera_order,
                jpeg_quality=policy.jpeg_quality,
                recv_timeout_ms=policy.recv_timeout_ms,
                send_timeout_ms=policy.send_timeout_ms,
                max_retries=policy.max_retries,
            )
        )
        self._client_open = False
        self._cameras_open = False
        self._robot_open = False
        self._init_interaction(controls, recorder)
        self._reset_action_history()

    def open(self) -> dict[str, Any]:
        """Validate the server and cameras before enabling either arm."""

        if self._client_open or self._cameras_open or self._robot_open:
            raise RuntimeError("dual-arm inference runtime is already open")
        if not self.config.runtime.hardware_access:
            raise RuntimeError("runtime.hardware_access must be true for dual-arm inference")
        if not self.config.runtime.motion_enabled:
            raise RuntimeError("runtime.motion_enabled must be true for dual-arm inference")

        self._client_open = True
        self.client.connect()
        metadata = self.client.metadata()
        validate_policy_metadata(self.config.inference, metadata)

        self._cameras_open = True
        self.cameras.connect()
        self._read_rgb_images()

        # Host kinematics reproduces the tool-frame observation used during
        # recording. It is required for both supported policy action spaces.
        self._robot_open = True
        self.robot.start_kinematics()
        self.robot.open_read_only()
        self.robot.prepare_motion()
        policy = self.config.inference
        if policy.home_on_start:
            logger.info("Moving both arms to configured inference home")
            self.robot.go_home(
                tolerance_deg=policy.home_tolerance_deg,
                timeout_s=policy.home_timeout_s,
            )
        self.client.reset()
        logger.info(
            "Dual-arm inference ready: server=%s stat_key=%s action_space=%s " "cameras=%s fps=%.3f",
            policy.server,
            policy.stat_key,
            policy.action_space,
            list(policy.camera_order),
            policy.fps,
        )
        return metadata

    def _state_vector(self, observation: Mapping[str, ArmObservation]) -> np.ndarray:
        return _piper_state_vector(observation, action_space=self.config.inference.action_space)

    def _reset_action_history(self) -> None:
        self._held_joints = {}
        self._held_grippers = {}
        self._held_eef = {}

    def _stationary_frame(self, observation: Mapping[str, ArmObservation]) -> dict[str, np.ndarray]:
        joints = _piper_state_vector(observation, action_space="joint_position")
        eef = _piper_state_vector(observation)
        return {
            "state.joint": joints,
            "action.joint": joints.copy(),
            "state.eef": eef,
            "action.eef": eef.copy(),
            "action.executed": np.zeros(2, dtype=np.float32),
        }

    def _validate_action_array(self, actions: np.ndarray) -> None:
        policy = self.config.inference
        if actions.ndim != 2 or actions.shape[1] != policy.action_dim:
            raise RuntimeError(f"policy actions must have shape (T,{policy.action_dim}), got {actions.shape}")
        if len(actions) == 0 or not np.all(np.isfinite(actions)):
            raise RuntimeError("policy returned empty, NaN, or Inf actions")
        grippers = actions[:, (PIPER_DIM_PER_ARM - 1) :: PIPER_DIM_PER_ARM]
        if np.any(grippers < 0.0) or np.any(grippers > 1.0):
            raise RuntimeError(
                "policy gripper target is outside normalized [0, 1] bounds: "
                f"range=[{float(grippers.min()):.6f}, {float(grippers.max()):.6f}]"
            )

    def _execute_action(
        self, action: np.ndarray, observation: Mapping[str, ArmObservation]
    ) -> dict[str, np.ndarray] | None:
        if self.config.inference.action_space == "cartesian":
            result = self._execute_cartesian_action(action, observation)
        else:
            result = self._execute_joint_action(action, observation)
        live_eef = _piper_state_vector(observation).reshape(len(SIDES), PIPER_DIM_PER_ARM)
        for index, side in enumerate(SIDES):
            live = observation[side]
            self._held_joints.setdefault(side, live.joint_positions_rad)
            self._held_grippers.setdefault(side, live.gripper_open_fraction)
            self._held_eef.setdefault(side, live_eef[index].copy())
        if not result.hardware_write:
            return None
        frame = build_frame(
            result,
            {},
            last_joint_targets_rad=self._held_joints,
            last_gripper_targets=self._held_grippers,
        )
        eef = frame["action.eef"].reshape(2, 7)
        for index, side in enumerate(SIDES):
            applied = result.applied.get(side)
            if applied is not None and applied.pose_written:
                self._held_eef[side] = eef[index].copy()
            self._held_eef[side][6] = self._held_grippers[side]
            eef[index] = self._held_eef[side]
        frame["action.executed"] = np.asarray(
            [
                side in result.applied and (result.applied[side].pose_written or result.applied[side].gripper_written)
                for side in SIDES
            ],
            dtype=np.float32,
        )
        return frame

    def _execute_cartesian_action(self, action: np.ndarray, observation: Mapping[str, ArmObservation]) -> CommandResult:
        shaped = action.reshape(len(SIDES), PIPER_DIM_PER_ARM)
        targets = {side: _arm_target(shaped[index]) for index, side in enumerate(SIDES)}
        return self.robot.apply_targets(targets, observation)

    def _execute_joint_action(self, action: np.ndarray, observation: Mapping[str, ArmObservation]) -> CommandResult:
        shaped = action.reshape(len(SIDES), PIPER_DIM_PER_ARM)
        joints = {side: tuple(float(value) for value in shaped[index, :6]) for index, side in enumerate(SIDES)}
        grippers = {side: float(shaped[index, 6]) for index, side in enumerate(SIDES)}
        actual = self.robot.apply_joint_targets(joints, grippers)
        applied = {}
        for side, (sent_joints, sent_gripper) in actual.items():
            # FK uses the submitted joint target, not the model-requested EEF.
            command = self.robot.tool_pose_from_joints(side, sent_joints)
            applied[side] = AppliedArmCommand(
                command=ArmCommand(command.position_mm, command.rotation_vector_rad, sent_gripper),
                pose_written=True,
                gripper_written=sent_gripper is not None,
                joint_target_rad=sent_joints,
            )
        return CommandResult(observation=dict(observation), applied=applied)


def _piper_state_vector(observation: Mapping[str, ArmObservation], *, action_space: str = "cartesian") -> np.ndarray:
    """Build the recorder-compatible left-then-right EEF or joint state."""

    _require_sides(observation, "Piper observation")
    values: list[float] = []
    for side in SIDES:
        arm = observation[side]
        if action_space == "joint_position":
            if arm.joint_positions_rad is None:
                raise RuntimeError(f"{side} Piper observation is missing joint feedback")
            values.extend(arm.joint_positions_rad)
        else:
            rotation = Rotation.from_quat(wxyz_to_xyzw(arm.quaternion_wxyz)).as_rotvec()
            values.extend(
                (
                    *(float(value) * 1000.0 for value in arm.position_m),
                    *(float(value) for value in rotation),
                )
            )
        values.append(float(arm.gripper_open_fraction))
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (14,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"invalid dual Piper state vector: shape={state.shape}, values={state}")
    return state


def _arm_target(values: np.ndarray) -> ArmTarget:
    quaternion_xyzw = tuple(float(value) for value in Rotation.from_rotvec(values[3:6]).as_quat())
    return ArmTarget(
        position_m=tuple(float(value) / 1000.0 for value in values[:3]),
        quaternion_wxyz=xyzw_to_wxyz(quaternion_xyzw),
        gripper_open_fraction=float(values[6]),
    )


def _require_sides(values: Mapping[str, Any], description: str) -> None:
    if set(values) != set(SIDES):
        raise RuntimeError(f"{description} must contain exactly {list(SIDES)}, got {sorted(values)}")
