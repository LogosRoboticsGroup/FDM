from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.teleoperation.flexiv_spacemouse.contracts import FlexivObservation, FlexivTarget
from deployment.teleoperation.flexiv_spacemouse.robot import FlexivRobot
from deployment.teleoperation.pose import rotation_6d_to_wxyz, wxyz_to_rotation_6d, wxyz_to_xyzw
from deployment.teleoperation.recording.cameras import CameraHub

from .client import StarVLAZmqClient
from .config import FlexivInferenceConfig, resolve_flexiv_policy_metadata
from .synchronous import SynchronousInferenceRuntime

logger = logging.getLogger(__name__)


class FlexivInferenceRuntime(SynchronousInferenceRuntime):
    """Closed-loop StarVLA client for a single Flexiv arm.

    This runtime is fully contained in the StarVLA codebase.  ``InferSystem`` is
    not imported or required; only its small ZMQ/msgpack wire format is retained
    so the existing StarVLA model server remains usable.
    """

    ROBOT_TYPE = "flexiv"

    def __init__(
        self,
        config: FlexivInferenceConfig,
        *,
        robot: Any | None = None,
        cameras: Any | None = None,
        client: Any | None = None,
        controls: Any | None = None,
        recorder: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        policy = config.inference
        self.robot = robot if robot is not None else FlexivRobot(config.robot)
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

    def open(self) -> dict[str, Any]:
        """Validate the remote policy and cameras before enabling robot motion."""

        if self._robot_open:
            raise RuntimeError("Flexiv inference runtime is already open")
        if not self.config.runtime.hardware_access:
            raise RuntimeError("runtime.hardware_access must be true for Flexiv inference")
        if not self.config.runtime.motion_enabled:
            raise RuntimeError("runtime.motion_enabled must be true for Flexiv inference")

        # Server and camera failures are intentionally discovered before the arm
        # is enabled or sent to home.
        self._client_open = True
        self.client.connect()
        metadata = self.client.metadata()
        resolved_policy = resolve_flexiv_policy_metadata(self.config.inference, metadata)
        self.config = replace(self.config, inference=resolved_policy)
        self.config.validate()

        self._cameras_open = True
        self.cameras.connect()
        self._read_rgb_images()

        self._robot_open = True
        self.robot.open(enable_motion=True)
        policy = self.config.inference
        if policy.home_on_start:
            logger.info("Moving Flexiv to configured inference home")
            self.robot.go_home(
                tolerance_deg=policy.home_tolerance_deg,
                timeout_s=policy.home_timeout_s,
            )
        self._prepare_motion()
        self.client.reset()
        logger.info(
            "Flexiv inference ready: server=%s stat_key=%s action_space=%s cameras=%s fps=%.3f",
            policy.server,
            policy.stat_key,
            policy.action_space,
            list(policy.camera_order),
            policy.fps,
        )
        return metadata

    def _go_home(self) -> None:
        super()._go_home()
        self._prepare_motion()

    def _prepare_motion(self) -> None:
        if self.config.inference.action_space == "cartesian":
            self.robot.prepare_cartesian_motion()
        else:
            self.robot.prepare_joint_motion()

    def _state_vector(self, observation: FlexivObservation) -> np.ndarray:
        if self.config.inference.action_space == "cartesian":
            state = _eef_state_vector(observation)
        else:
            state = _state_vector(observation)
        state[-1] *= self._model_gripper_scale
        return state

    def _stationary_frame(self, observation: FlexivObservation) -> dict[str, np.ndarray]:
        if self.config.inference.action_space == "cartesian":
            state = _eef_state_vector(observation)
            return {"observation.state.eef_pose": state, "actions.eef_pose": state.copy()}
        state = _state_vector(observation)
        return {"observation.state.joint_position": state, "actions.joint_position": state.copy()}

    def _execute_action(
        self,
        action: np.ndarray,
        observation: FlexivObservation,
    ) -> dict[str, np.ndarray] | None:
        if self.config.inference.action_space == "cartesian":
            target = _eef_target(action)
            self._validate_cartesian_target(target, observation)
            result = self.robot.apply_target(target, observation)
            if result.control_mode_recovered or not result.pose_written:
                return None
            gripper_width_m = float(action[-1])
            if not math.isclose(gripper_width_m, observation.gripper_width_m, abs_tol=1e-5):
                self.robot.move_gripper(gripper_width_m)
            return {
                "observation.state.eef_pose": _eef_state_vector(observation),
                "actions.eef_pose": _eef_vector(target.tcp_pose_wxyz, gripper_width_m),
            }

        joints = tuple(float(value) for value in action[:7])
        actual = self.robot.apply_joint_target(joints, float(action[7]))
        return {
            "observation.state.joint_position": _state_vector(observation),
            "actions.joint_position": np.asarray(actual, dtype=np.float32),
        }

    def _validate_action_array(self, actions: np.ndarray) -> None:
        policy = self.config.inference
        if policy.action_dim is None or policy.action_space == "auto":
            raise RuntimeError("Flexiv policy metadata has not been resolved")
        if actions.ndim != 2 or actions.shape[1] != policy.action_dim:
            raise RuntimeError(f"policy actions must have shape (T,{policy.action_dim}), got {actions.shape}")
        if len(actions) == 0 or not np.all(np.isfinite(actions)):
            raise RuntimeError("policy returned empty, NaN, or Inf actions")

        gripper = self.config.robot.gripper
        widths = actions[:, -1]
        if np.any(widths < gripper.close_width_m) or np.any(widths > gripper.open_width_m):
            raise RuntimeError(
                "policy gripper target is outside configured bounds: "
                f"range=[{float(widths.min()):.6f}, {float(widths.max()):.6f}], "
                f"allowed=[{gripper.close_width_m:.6f}, {gripper.open_width_m:.6f}]"
            )

        if policy.action_space == "cartesian":
            try:
                for action in actions:
                    rotation_6d_to_wxyz(action[3:9])
            except ValueError as exc:
                raise RuntimeError("policy returned an invalid Cartesian rotation") from exc

    def _prepare_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = actions.copy()
        if actions.ndim == 2 and actions.shape[1] > 0:
            actions[:, -1] /= self._model_gripper_scale
        return actions

    @property
    def _model_gripper_scale(self) -> float:
        return self.config.robot.gripper.gripper_scale

    def _validate_action_sequence(self, actions: np.ndarray) -> None:
        if self.config.inference.action_space != "cartesian" or len(actions) <= 1:
            return

        position_steps = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
        max_position_step = float(position_steps.max())
        position_limit = self.config.robot.max_position_lead_m
        if max_position_step > position_limit:
            raise RuntimeError(
                f"policy Cartesian position step {max_position_step:.6f}m exceeds limit {position_limit:.6f}m"
            )

        rotations = [Rotation.from_quat(wxyz_to_xyzw(rotation_6d_to_wxyz(action[3:9]))) for action in actions]
        max_rotation_step = max(
            float((rotations[index - 1].inv() * rotations[index]).magnitude()) for index in range(1, len(rotations))
        )
        rotation_limit = self.config.robot.max_rotation_lead_rad
        if max_rotation_step > rotation_limit:
            raise RuntimeError(
                f"policy Cartesian rotation step {max_rotation_step:.6f}rad exceeds limit {rotation_limit:.6f}rad"
            )

    def _validate_cartesian_target(self, target: FlexivTarget, observation: FlexivObservation) -> None:
        position = np.asarray(target.tcp_pose_wxyz[:3], dtype=np.float64)
        workspace_min = np.asarray(self.config.robot.workspace_min_m, dtype=np.float64)
        workspace_max = np.asarray(self.config.robot.workspace_max_m, dtype=np.float64)
        if np.any(position < workspace_min) or np.any(position > workspace_max):
            raise RuntimeError(
                f"policy Cartesian target {position.tolist()} is outside workspace "
                f"[{workspace_min.tolist()}, {workspace_max.tolist()}]"
            )

        live_position = np.asarray(observation.tcp_pose_wxyz[:3], dtype=np.float64)
        position_lead = float(np.linalg.norm(position - live_position))
        if position_lead > self.config.robot.max_position_lead_m:
            raise RuntimeError(
                f"policy Cartesian target is {position_lead:.6f}m from live feedback; "
                f"limit={self.config.robot.max_position_lead_m:.6f}m"
            )

        target_rotation = Rotation.from_quat(wxyz_to_xyzw(target.tcp_pose_wxyz[3:]))
        live_rotation = Rotation.from_quat(wxyz_to_xyzw(observation.tcp_pose_wxyz[3:]))
        rotation_lead = float((live_rotation.inv() * target_rotation).magnitude())
        if rotation_lead > self.config.robot.max_rotation_lead_rad:
            raise RuntimeError(
                f"policy Cartesian rotation is {rotation_lead:.6f}rad from live feedback; "
                f"limit={self.config.robot.max_rotation_lead_rad:.6f}rad"
            )


def _state_vector(observation: FlexivObservation) -> np.ndarray:
    state = np.asarray(
        (*observation.joint_positions_rad, observation.gripper_width_m),
        dtype=np.float32,
    )
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"invalid Flexiv state vector: shape={state.shape}, values={state}")
    return state


def _eef_state_vector(observation: FlexivObservation) -> np.ndarray:
    state = _eef_vector(observation.tcp_pose_wxyz, observation.gripper_width_m)
    if state.shape != (10,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"invalid Flexiv EEF state vector: shape={state.shape}, values={state}")
    return state


def _eef_vector(pose_wxyz: tuple[float, ...], gripper_width_m: float) -> np.ndarray:
    return np.asarray(
        (*pose_wxyz[:3], *wxyz_to_rotation_6d(pose_wxyz[3:]), gripper_width_m),
        dtype=np.float32,
    )


def _eef_target(action: np.ndarray) -> FlexivTarget:
    return FlexivTarget(
        tcp_pose_wxyz=tuple(float(value) for value in (*action[:3], *rotation_6d_to_wxyz(action[3:9]))),
        gripper_width_m=float(action[9]),
    )
