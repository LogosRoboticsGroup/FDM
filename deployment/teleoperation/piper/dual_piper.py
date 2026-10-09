from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from scipy.spatial.transform import Rotation

from ..config import ArmConfig
from ..contracts import AppliedArmCommand, ArmObservation, ArmTarget, CommandResult, JointState
from ..pose import observation_to_command, wxyz_to_xyzw
from .host_ik import HostIKCoordinator, HostIKPlan
from .sdk_adapter import JointFeedback, PiperArmAdapter, PiperFault

logger = logging.getLogger(__name__)
HOME_POLL_INTERVAL_S = 0.05
HOME_COMMAND_INTERVAL_S = 0.5


class DualPiper:
    def __init__(
        self,
        arms: dict[str, ArmConfig],
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = dict(arms)
        self.arms = {side: PiperArmAdapter(side, config, sdk_factory=sdk_factory) for side, config in arms.items()}
        self.host_ik: HostIKCoordinator | None = None
        self._last_request_lag: dict[str, int] = {}
        self._last_ik_status: dict[str, str] = {}
        self._last_joint_error_deg: dict[str, float] = {}
        self._last_feedback_hz: dict[str, float] = {}
        self._last_position_error_mm: dict[str, float] = {}
        self._last_rotation_error_deg: dict[str, float] = {}
        self._last_joint_read_ms = 0.0
        self._last_ik_plan_ms = 0.0
        self._last_dispatch_ms = 0.0
        self._command_times: dict[str, deque[float]] = {side: deque(maxlen=128) for side in arms}
        self._last_command_gap_ms: dict[str, float] = {}

    def start_kinematics(self) -> None:
        host_sides = [side for side, config in self.config.items() if config.control_backend == "host_ik"]
        if not host_sides:
            return
        if len(host_sides) != len(self.config):
            raise ValueError("mixed host-IK and firmware backends are not supported")
        self.host_ik = HostIKCoordinator(self.config)
        self.host_ik.start()

    def open_read_only(self) -> None:
        opened = []
        try:
            for arm in self.arms.values():
                arm.open_read_only()
                opened.append(arm)
        except BaseException:
            for arm in reversed(opened):
                arm.close()
            raise

    def prepare_motion(self) -> None:
        for arm in self.arms.values():
            arm.prepare_motion()

    def read_joint_states(self) -> dict[str, JointState]:
        return {side: arm.read_joint_state() for side, arm in self.arms.items()}

    def read_observation(self) -> dict[str, ArmObservation]:
        if self.host_ik is None:
            return {side: arm.read_observation() for side, arm in self.arms.items()}
        joints = self.read_joint_states()
        return {
            side: self.host_ik.observation(side, joints[side], arm.read_observation().gripper_open_fraction)
            for side, arm in self.arms.items()
        }

    def plan_targets_read_only(
        self,
        targets: dict[str, ArmTarget | None],
    ) -> tuple[dict[str, ArmObservation], dict[str, JointState], HostIKPlan]:
        if self.host_ik is None:
            raise RuntimeError("read-only IK planning requires host_ik")
        joints = self.read_joint_states()
        observations = {
            side: self.host_ik.observation(side, joints[side], arm.read_observation().gripper_open_fraction)
            for side, arm in self.arms.items()
        }
        return observations, joints, self.host_ik.plan(targets, joints)

    def apply_targets(
        self,
        targets: dict[str, ArmTarget | None],
        observation: dict[str, ArmObservation],
        gripper_targets: dict[str, float | None] | None = None,
    ) -> CommandResult:
        if self.host_ik is not None:
            return self._apply_host_ik_targets(targets, observation, gripper_targets)
        applied = {
            side: arm.apply_target(
                replace(target, gripper_open_fraction=None) if gripper_targets is not None else target
            )
            for side, arm in self.arms.items()
            if (target := targets.get(side)) is not None
        }
        if gripper_targets is not None:
            self._apply_independent_gripper_targets(
                applied,
                gripper_targets,
                observation,
            )
        return CommandResult(observation=observation, applied=applied)

    def apply_joint_targets(
        self,
        targets_rad: dict[str, Sequence[float]],
        gripper_targets: dict[str, float | None],
    ) -> dict[
        str,
        tuple[
            tuple[float, float, float, float, float, float],
            float | None,
        ],
    ]:
        unknown = (set(targets_rad) | set(gripper_targets)) - set(self.arms)
        if unknown:
            raise ValueError(f"unknown Piper sides: {sorted(unknown)}")
        result = {}
        for side in self.arms:
            joints = targets_rad.get(side)
            if joints is None:
                continue
            actual = self.arms[side].send_joint_target_rad(joints)
            gripper, _ = self.arms[side].send_gripper_fraction(gripper_targets.get(side))
            result[side] = (actual, gripper)
        return result

    def tool_pose_from_joints(self, side, joints):
        return self.host_ik.models[side].tool_pose_from_joints(joints)

    def latency_diagnostics(self) -> dict[str, object]:
        return {
            "ik_request_lag": dict(self._last_request_lag),
            "ik_status": dict(self._last_ik_status),
            "joint_error_deg": dict(self._last_joint_error_deg),
            "position_error_mm": dict(self._last_position_error_mm),
            "rotation_error_deg": dict(self._last_rotation_error_deg),
            "feedback_hz": dict(self._last_feedback_hz),
            "joint_read_ms": self._last_joint_read_ms,
            "ik_plan_ms": self._last_ik_plan_ms,
            "dispatch_ms": self._last_dispatch_ms,
            "command_hz": {side: self._command_hz(side) for side in self.arms},
            "command_gap_ms": dict(self._last_command_gap_ms),
            "ik_solver": (
                diagnostics()
                if self.host_ik is not None
                and callable(
                    diagnostics := getattr(
                        self.host_ik,
                        "solver_diagnostics",
                        None,
                    )
                )
                else {}
            ),
            "gripper": {
                side: diagnostics() if callable(diagnostics := getattr(arm, "gripper_diagnostics", None)) else {}
                for side, arm in self.arms.items()
            },
            "arm_status": {
                side: diagnostics() if callable(diagnostics := getattr(arm, "arm_status_diagnostics", None)) else {}
                for side, arm in self.arms.items()
            },
        }

    def go_home(self, *, tolerance_deg: float, timeout_s: float) -> dict[str, JointFeedback]:
        if not math.isfinite(tolerance_deg) or tolerance_deg <= 0.0:
            raise ValueError("home tolerance must be a positive finite value")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("home timeout must be a positive finite value")

        deadline = time.monotonic() + timeout_s
        self._send_home_targets()

        # GetArmJointMsgs() is an SDK-owned aggregate whose initial values are
        # all zero. Snapshot it after the command and require a later aggregate
        # update so a zero home cannot be accepted from cached startup data.
        feedback_cutoffs = {side: arm.read_joint_feedback().timestamp_s for side, arm in self.arms.items()}
        next_command_at = time.monotonic() + HOME_COMMAND_INTERVAL_S
        while True:
            now = time.monotonic()
            if now >= next_command_at:
                self._send_home_targets()
                next_command_at = now + HOME_COMMAND_INTERVAL_S

            feedback = {side: arm.read_joint_feedback() for side, arm in self.arms.items()}
            errors = {
                side: max(
                    abs(value - target)
                    for value, target in zip(
                        feedback[side].angles_degrees,
                        self.config[side].home_joints_deg,
                        strict=True,
                    )
                )
                for side in self.arms
            }
            fresh_complete = {
                side: item.is_complete and item.timestamp_s > feedback_cutoffs[side] for side, item in feedback.items()
            }
            if all(fresh_complete[side] and errors[side] <= tolerance_deg for side in self.arms):
                return feedback
            if time.monotonic() >= deadline:
                details = ", ".join(
                    f"{side}={errors[side]:.2f}deg"
                    + ("" if fresh_complete[side] else " (waiting for fresh complete feedback)")
                    for side in self.arms
                )
                raise PiperFault(f"timed out waiting for both Piper arms to reach home ({details})")
            time.sleep(HOME_POLL_INTERVAL_S)

    def _send_home_targets(self) -> None:
        for side, arm in self.arms.items():
            config = self.config[side]
            arm.send_joint_target(
                config.home_joints_deg,
                move_speed_percent=config.home_move_speed_percent,
            )
            arm.send_gripper_fraction(config.home_gripper_open_fraction)

    def close(self) -> None:
        for arm in reversed(tuple(self.arms.values())):
            try:
                arm.close()
            except Exception:
                logger.exception("Failed to close %s Piper", arm.side)
        if self.host_ik is not None:
            self.host_ik.close()
            self.host_ik = None

    def _apply_host_ik_targets(
        self,
        targets: dict[str, ArmTarget | None],
        observation: dict[str, ArmObservation],
        gripper_targets: dict[str, float | None] | None,
    ) -> CommandResult:
        assert self.host_ik is not None
        joint_read_started = time.perf_counter()
        joint_states = self.read_joint_states()
        joint_read_finished = time.perf_counter()
        ik_plan_started = joint_read_finished
        plan = self.host_ik.plan(targets, joint_states)
        ik_plan_finished = time.perf_counter()
        self._last_joint_read_ms = (joint_read_finished - joint_read_started) * 1000.0
        self._last_ik_plan_ms = (ik_plan_finished - ik_plan_started) * 1000.0
        self._last_request_lag = dict(plan.request_lag)
        self._last_ik_status = dict(plan.status)
        self._last_feedback_hz = {side: state.feedback_hz for side, state in joint_states.items()}
        self._last_position_error_mm = {}
        self._last_rotation_error_deg = {}
        for side, target in targets.items():
            if target is None:
                continue
            actual = observation[side]
            self._last_position_error_mm[side] = math.dist(target.position_m, actual.position_m) * 1000.0
            target_rotation = Rotation.from_quat(wxyz_to_xyzw(target.quaternion_wxyz))
            actual_rotation = Rotation.from_quat(wxyz_to_xyzw(actual.quaternion_wxyz))
            self._last_rotation_error_deg[side] = math.degrees((target_rotation * actual_rotation.inv()).magnitude())
        self._last_joint_error_deg = {
            side: max(
                math.degrees(abs(target - actual))
                for target, actual in zip(joints, joint_states[side].angles_rad, strict=True)
            )
            for side, joints in plan.joints.items()
        }
        applied = {}
        dispatch_started = time.perf_counter()
        for side, joints in plan.joints.items():
            actual = self.arms[side].send_joint_target_rad(joints)
            self._record_joint_command(side)
            gripper_target = (
                plan.commands[side].gripper_open_fraction if gripper_targets is None else gripper_targets.get(side)
            )
            gripper, gripper_written = self.arms[side].send_gripper_fraction(
                gripper_target,
                observation[side].gripper_open_fraction,
            )
            command = self.host_ik.models[side].tool_pose_from_joints(actual)
            applied[side] = AppliedArmCommand(
                command=replace(command, gripper_open_fraction=gripper),
                pose_written=True,
                gripper_written=gripper_written,
                joint_target_rad=actual,
            )
        self._last_dispatch_ms = (time.perf_counter() - dispatch_started) * 1000.0
        if gripper_targets is not None:
            self._apply_independent_gripper_targets(
                applied,
                gripper_targets,
                observation,
            )
        return CommandResult(observation=observation, applied=applied)

    def _apply_independent_gripper_targets(
        self,
        applied: dict[str, AppliedArmCommand],
        gripper_targets: dict[str, float | None],
        observation: dict[str, ArmObservation],
    ) -> None:
        unknown = set(gripper_targets) - set(self.arms)
        if unknown:
            raise ValueError(f"unknown Piper sides: {sorted(unknown)}")
        for side, target in gripper_targets.items():
            if target is None:
                continue
            existing = applied.get(side)
            if existing is not None and existing.gripper_written:
                continue
            gripper, gripper_written = self.arms[side].send_gripper_fraction(
                target,
                observation[side].gripper_open_fraction,
            )
            if not gripper_written:
                continue
            if existing is not None:
                applied[side] = replace(
                    existing,
                    command=replace(
                        existing.command,
                        gripper_open_fraction=gripper,
                    ),
                    gripper_written=True,
                )
            else:
                applied[side] = AppliedArmCommand(
                    command=observation_to_command(
                        observation[side],
                        gripper_open_fraction=gripper,
                    ),
                    pose_written=False,
                    gripper_written=True,
                )

    def _record_joint_command(self, side: str) -> None:
        now = time.monotonic()
        times = self._command_times[side]
        if times:
            self._last_command_gap_ms[side] = (
                max(
                    0.0,
                    now - times[-1],
                )
                * 1000.0
            )
        times.append(now)
        cutoff = now - 1.0
        while len(times) > 1 and times[0] < cutoff:
            times.popleft()

    def _command_hz(self, side: str) -> float | None:
        times = self._command_times[side]
        if len(times) < 2:
            return None
        duration_s = times[-1] - times[0]
        if duration_s <= 0.0:
            return None
        return (len(times) - 1) / duration_s
