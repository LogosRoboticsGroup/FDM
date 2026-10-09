from __future__ import annotations

import logging
import math
import time
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ArmConfig
from ..contracts import AppliedArmCommand, ArmCommand, ArmObservation, CommandResult
from ..pose import observation_to_command, wxyz_to_xyzw, xyzw_to_wxyz

logger = logging.getLogger(__name__)


class ArxKinematics:
    """Convert SDK XYZ metres / RPY radians to the shared TCP pose contract."""

    def __init__(self, sdk, robot_config, config: ArmConfig):
        self.solver = sdk.Arx5Solver(
            robot_config.urdf_path,
            6,
            robot_config.joint_pos_min,
            robot_config.joint_pos_max,
            robot_config.base_link_name,
            robot_config.eef_link_name,
            robot_config.gravity_vector,
        )
        tool = config.tool_from_j6_xyz_mm_rpy_deg
        self.offset = np.asarray(tool[:3]) / 1000.0
        self.rotation = Rotation.from_euler("xyz", tool[3:], degrees=True)

    def tool_pose_from_joints(self, joints) -> ArmCommand:
        pose = self.solver.forward_kinematics(np.asarray(joints, dtype=float))
        rotation = Rotation.from_euler("xyz", pose[3:])
        position = np.asarray(pose[:3]) + rotation.apply(self.offset)
        return ArmCommand(tuple(position * 1000), tuple((rotation * self.rotation).as_rotvec()))

    def solve(self, target, seed):
        rotation = Rotation.from_quat(wxyz_to_xyzw(target.quaternion_wxyz)) * self.rotation.inv()
        position = np.asarray(target.position_m) - rotation.apply(self.offset)
        pose = np.concatenate((position, rotation.as_euler("xyz")))
        status, joints = self.solver.inverse_kinematics(pose, np.asarray(seed, dtype=float))
        return tuple(float(value) for value in joints) if status == 0 else None


class DualArx:
    """Two independent X5 controllers; construction never opens CAN.

    The SDK constructor writes motor commands. Only prepare_motion creates it.
    Closing puts both arms in damping before releasing the SDK controllers.
    """

    def __init__(self, arms: dict[str, ArmConfig], *, sdk=None):
        self.config = dict(arms)
        self.sdk = sdk
        self.models = {}
        self.robot_configs = {}
        self.controllers = {}
        self._held_joints = {}
        self._held_grippers = {}
        self._gripper_contact_width = {}
        self._ik_status = {}

    def start_kinematics(self):
        if self.models:
            return
        if self.sdk is None:
            import arx5_interface

            self.sdk = arx5_interface
        try:
            for side, config in self.config.items():
                robot = self.sdk.RobotConfigFactory.get_instance().get_config("X5")
                robot.gripper_width = config.gripper_max_width_m
                robot.gripper_vel_max = min(robot.gripper_vel_max, config.arx.gripper_velocity_limit_m_s)
                if config.arx.gripper_open_readout is not None:
                    robot.gripper_open_readout = config.arx.gripper_open_readout
                if config.ik_urdf_path is not None:
                    robot.urdf_path = config.ik_urdf_path
                self.robot_configs[side] = robot
                self.models[side] = ArxKinematics(self.sdk, robot, config)
                self._validate_joints(side, np.radians(config.home_joints_deg))
        except BaseException:
            self.models.clear()
            self.robot_configs.clear()
            raise

    def open_read_only(self):
        # Lifecycle hook shared with Piper. No controller, CAN access, or writes.
        # The CLI rejects ARX probe/ik-preview instead of claiming SDK read-only support.
        pass

    def prepare_motion(self):
        if self.controllers:
            raise RuntimeError("ARX controllers are already open")
        self.start_kinematics()
        try:
            for side, config in self.config.items():
                controller_config = self.sdk.ControllerConfigFactory.get_instance().get_config("joint_controller", 6)
                controller_config.background_send_recv = True
                controller = self.sdk.Arx5JointController(
                    self.robot_configs[side], controller_config, config.can_interface
                )
                self.controllers[side] = controller
                state = self._state(side)
                self._held_joints[side] = tuple(state.pos())
                self._held_grippers[side] = float(state.gripper_pos)
                self._send(side, self._held_joints[side], None)
                gain = self.sdk.Gain(6)
                gain.kp()[:] = controller_config.default_kp
                gain.kd()[:] = controller_config.default_kd
                if config.gripper_enabled:
                    gain.gripper_kp = 2.0
                    gain.gripper_kd = controller_config.default_gripper_kd
                controller.set_gain(gain)
        except BaseException:
            controller = None
            self.close()
            raise

    def _state(self, side):
        controller = self.controllers[side]
        state = controller.get_joint_state()
        age = controller.get_timestamp() - state.timestamp
        if not math.isfinite(age) or not 0 <= age <= self.config[side].arx.feedback_timeout_s:
            raise RuntimeError(f"{side} ARX feedback is stale ({age:.3f}s)")
        values = (*state.pos(), state.gripper_pos)
        if len(values) != 7 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{side} ARX returned invalid feedback")
        return state

    def read_observation(self):
        if set(self.controllers) != set(self.config):
            raise RuntimeError("ARX observation requires prepare_motion; SDK has no read-only connection")
        observations = {}
        for side, config in self.config.items():
            state = self._state(side)
            joints = tuple(float(value) for value in state.pos())
            command = self.tool_pose_from_joints(side, joints)
            observations[side] = ArmObservation(
                tuple(value / 1000.0 for value in command.position_mm),
                xyzw_to_wxyz(tuple(Rotation.from_rotvec(command.rotation_vector_rad).as_quat())),
                float(np.clip(state.gripper_pos / config.gripper_max_width_m, 0, 1)),
                joints,
            )
        return observations

    def tool_pose_from_joints(self, side, joints):
        return self.models[side].tool_pose_from_joints(joints)

    def _validate_joints(self, side, joints):
        values = np.asarray(joints, dtype=float)
        robot = self.robot_configs[side]
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{side} ARX joint target must contain six finite radians")
        if np.any(values < robot.joint_pos_min) or np.any(values > robot.joint_pos_max):
            raise ValueError(f"{side} ARX joint target exceeds SDK joint limits")
        return tuple(float(value) for value in values)

    def _validate_gripper(self, side, fraction):
        if fraction is not None and (not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError(f"{side} ARX gripper target must be in [0, 1]")

    def _send(self, side, joints, fraction):
        joints = self._validate_joints(side, joints)
        self._validate_gripper(side, fraction)
        config = self.config[side]
        written = fraction is not None and config.gripper_enabled
        width = fraction * config.gripper_max_width_m if written else self._held_grippers[side]
        if config.gripper_enabled:
            guarded = self._guard_gripper(side, width)
            written = written or guarded != width
            width = guarded
        command = self.sdk.JointState(6)
        command.pos()[:] = joints
        command.gripper_pos = width
        self.controllers[side].set_joint_cmd(command)
        self._held_joints[side] = joints
        self._held_grippers[side] = width
        return joints, width / config.gripper_max_width_m if written else None

    def _guard_gripper(self, side, width):
        """Latch a contact width until an explicit opening target releases it."""
        state = self._state(side)
        torque = float(state.gripper_torque)
        if not math.isfinite(torque):
            raise RuntimeError(f"{side} ARX returned invalid gripper torque")
        contact = self._gripper_contact_width.get(side)
        if contact is not None:
            if width > contact + 0.001:
                del self._gripper_contact_width[side]
            else:
                width = max(width, contact)
        # Torque magnitude is independent of the calibrated encoder direction.
        if width < state.gripper_pos and abs(torque) >= self.robot_configs[side].gripper_torque_max * 0.5:
            width = float(np.clip(state.gripper_pos + 0.0015, 0, self.config[side].gripper_max_width_m))
            if contact is None:
                logger.warning(
                    "%s gripper contact: torque=%.3f, measured_width=%.4fm, relief_target=%.4fm",
                    side,
                    torque,
                    state.gripper_pos,
                    width,
                )
            self._gripper_contact_width[side] = width
        # Bound error against feedback, not the previous target: repeated close
        # requests must not accumulate position error against a blocked object.
        minimum = np.clip(
            state.gripper_pos - self.config[side].arx.gripper_max_closing_error_m,
            0,
            self.config[side].gripper_max_width_m,
        )
        return max(width, float(minimum))

    def apply_targets(self, targets, observation, gripper_targets=None):
        unknown = (set(targets) | set(gripper_targets or {})) - set(self.config)
        if unknown:
            raise ValueError(f"unknown ARX sides: {sorted(unknown)}")
        planned = {}
        for side in self.config:
            target = targets.get(side)
            gripper = (
                gripper_targets.get(side)
                if gripper_targets is not None
                else target.gripper_open_fraction if target is not None else None
            )
            self._validate_gripper(side, gripper)
            joints = None
            if target is not None:
                values = (*target.position_m, *target.quaternion_wxyz)
                if not np.all(np.isfinite(values)):
                    raise ValueError("ARX Cartesian targets must be finite")
                joints = self.models[side].solve(target, observation[side].joint_positions_rad)
                status = "solved" if joints is not None else "failed"
                if joints is not None:
                    try:
                        joints = self._validate_joints(side, joints)
                    except ValueError as exc:
                        status = "invalid_solution"
                        if self._ik_status.get(side) != status:
                            logger.warning("%s; keeping last valid pose. IK joints (rad): %s", exc, joints)
                        joints = None
                self._ik_status[side] = status
            planned[side] = joints, gripper
        applied = {}
        for side, (joints, gripper) in planned.items():
            pose_written = joints is not None
            if not pose_written and (gripper is None or not self.config[side].gripper_enabled):
                if not self.config[side].gripper_enabled:
                    continue
                held = self._held_grippers[side]
                guarded = self._guard_gripper(side, held)
                if guarded == held:
                    continue
                gripper = guarded / self.config[side].gripper_max_width_m
            sent, sent_gripper = self._send(side, joints or self._held_joints[side], gripper)
            command = (
                self.tool_pose_from_joints(side, sent)
                if pose_written
                else observation_to_command(observation[side], gripper_open_fraction=sent_gripper)
            )
            applied[side] = AppliedArmCommand(
                replace(command, gripper_open_fraction=sent_gripper),
                pose_written,
                sent_gripper is not None,
                sent if pose_written else None,
            )
        return CommandResult(observation, applied)

    def apply_joint_targets(self, targets_rad, gripper_targets):
        unknown = (set(targets_rad) | set(gripper_targets)) - set(self.config)
        if unknown:
            raise ValueError(f"unknown ARX sides: {sorted(unknown)}")
        # Validate both arms before writing either one.
        for side, joints in targets_rad.items():
            self._validate_joints(side, joints)
            self._validate_gripper(side, gripper_targets.get(side))
        return {side: self._send(side, joints, gripper_targets.get(side)) for side, joints in targets_rad.items()}

    def go_home(self, *, tolerance_deg, timeout_s):
        if not all(math.isfinite(v) and v > 0 for v in (tolerance_deg, timeout_s)):
            raise ValueError("home tolerance and timeout must be positive and finite")
        initial = self.read_observation()
        targets = {side: np.radians(config.home_joints_deg) for side, config in self.config.items()}
        started = time.monotonic()
        cutoffs = {side: self._state(side).timestamp for side in self.config}
        while time.monotonic() - started < timeout_s:
            elapsed = time.monotonic() - started
            for side, config in self.config.items():
                alpha = min(1.0, elapsed / config.arx.home_duration_s)
                alpha = alpha * alpha * (3 - 2 * alpha)
                start = np.asarray(initial[side].joint_positions_rad)
                self._send(side, start + alpha * (targets[side] - start), config.home_gripper_open_fraction)
            states = {side: self._state(side) for side in self.config}
            if all(
                elapsed >= self.config[side].arx.home_duration_s
                and state.timestamp > cutoffs[side]
                and np.max(np.abs(np.degrees(state.pos() - targets[side]))) <= tolerance_deg
                for side, state in states.items()
            ):
                return states
            time.sleep(0.02)
        raise TimeoutError("timed out waiting for both ARX arms to reach home")

    def latency_diagnostics(self):
        return {"ik_status": dict(self._ik_status)}

    def close(self):
        # Explicitly damp even when an exception traceback retains a controller.
        # Damp both arms before either SDK destructor waits for its sender thread.
        for side, controller in self.controllers.items():
            try:
                controller.set_to_damping()
            except Exception:
                logger.exception("Failed to put %s ARX into damping", side)
        self.controllers.clear()
        self._held_joints.clear()
        self._held_grippers.clear()
        self._gripper_contact_width.clear()
        self.models.clear()
        self.robot_configs.clear()
