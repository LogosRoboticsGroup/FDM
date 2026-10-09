from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ArmConfig
from ..contracts import AppliedArmCommand, ArmCommand, ArmObservation, ArmTarget, JointState
from ..pose import (
    axis_angle_to_rpy_degrees,
    j6_to_tcp,
    rpy_degrees_to_axis_angle,
    tcp_to_j6,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

GRIPPER_TARGET_GAP_FRACTION = 0.02
GRIPPER_PROGRESS_EPSILON_FRACTION = 0.002
GRIPPER_RELEASE_EDGE_FRACTION = 0.02
GRIPPER_REQUEST_DIRECTION_EPSILON_FRACTION = 1e-4
GRIPPER_DRIVER_ENABLE_BIT = 1 << 6
GRIPPER_FAULT_BITS = 0x3F
GRIPPER_RECOVERY_RETRY_INTERVAL_S = 0.5
JOINT_FEEDBACK_MAX_AGE_S = 0.25
JOINT_FEEDBACK_READY_TIMEOUT_S = 2.0
ARM_STATUS_NAMES = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_solution",
    0x03: "singularity",
    0x04: "target_angle_over_limit",
    0x05: "joint_communication_error",
    0x06: "joint_brake_not_released",
    0x07: "collision",
    0x08: "teaching_overspeed",
    0x09: "joint_status_error",
    0x0A: "other_error",
    0x0E: "controller_over_temperature",
    0x0F: "release_resistor_over_temperature",
}

logger = logging.getLogger(__name__)


class PiperFault(RuntimeError):
    pass


@dataclass(frozen=True)
class JointFeedback:
    timestamp_s: float
    angles_degrees: tuple[float, float, float, float, float, float]
    feedback_hz: float = 0.0

    @property
    def is_complete(self) -> bool:
        # Piper's aggregate joint feedback rate is non-zero only when all
        # three joint-pair CAN packets (1-2, 3-4, and 5-6) are arriving.
        return (
            math.isfinite(self.timestamp_s)
            and self.timestamp_s > 0.0
            and math.isfinite(self.feedback_hz)
            and self.feedback_hz > 0.0
        )


@dataclass(frozen=True)
class SocketCANStatus:
    interface: str
    is_up: bool
    state: str
    bitrate: int | None
    restarts: int = 0
    error_warning: int = 0
    error_passive: int = 0
    bus_off: int = 0
    rx_dropped: int = 0


def _read_socketcan_status(can_interface: str) -> SocketCANStatus:
    try:
        result = subprocess.run(
            [
                "ip",
                "-details",
                "-statistics",
                "-json",
                "link",
                "show",
                "dev",
                can_interface,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except FileNotFoundError as exc:
        raise ConnectionError(
            "cannot inspect SocketCAN because the 'ip' command is unavailable"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ConnectionError(
            f"timed out while inspecting SocketCAN interface {can_interface}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "interface not found"
        raise ConnectionError(
            f"cannot inspect SocketCAN interface {can_interface}: {detail}"
        )
    try:
        records = json.loads(result.stdout)
        record = records[0]
        info_data = record["linkinfo"]["info_data"]
        state = str(info_data["state"]).upper()
        bitrate_value = info_data.get("bittiming", {}).get("bitrate")
        bitrate = None if bitrate_value is None else int(bitrate_value)
        flags = {str(flag).upper() for flag in record.get("flags", ())}
        xstats = record.get("linkinfo", {}).get("info_xstats", {})
        rx_stats = record.get("stats64", {}).get("rx", {})
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConnectionError(
            f"cannot parse SocketCAN status for {can_interface}"
        ) from exc
    return SocketCANStatus(
        interface=can_interface,
        is_up="UP" in flags,
        state=state,
        bitrate=bitrate,
        restarts=int(xstats.get("restarts", 0)),
        error_warning=int(xstats.get("error_warning", 0)),
        error_passive=int(xstats.get("error_passive", 0)),
        bus_off=int(xstats.get("bus_off", 0)),
        rx_dropped=int(rx_stats.get("dropped", 0)),
    )


def _validate_socketcan_ready(
    can_interface: str,
    *,
    require_error_active: bool = False,
) -> SocketCANStatus:
    status = _read_socketcan_status(can_interface)
    if not status.is_up:
        raise ConnectionError(
            f"SocketCAN interface {can_interface} is DOWN; configure and bring it up "
            "before starting StarVLA"
        )
    if status.state in {"BUS-OFF", "STOPPED"}:
        raise ConnectionError(
            f"SocketCAN interface {can_interface} is {status.state}; no Piper commands "
            "will be transmitted. Check arm power, CAN wiring and termination, then "
            "recover the interface outside StarVLA before restarting"
        )
    if require_error_active and status.state != "ERROR-ACTIVE":
        raise ConnectionError(
            f"SocketCAN interface {can_interface} is {status.state}, not ERROR-ACTIVE; "
            "refusing to start Piper motion because a degraded CAN bus can make "
            "teleoperation lag or lose commands. Recover the interface and verify "
            "its state before restarting"
        )
    if status.state != "ERROR-ACTIVE":
        logger.warning(
            "SocketCAN interface %s is degraded: state=%s bitrate=%s",
            can_interface,
            status.state,
            status.bitrate,
        )
    else:
        logger.info(
            "SocketCAN interface ready: interface=%s state=%s bitrate=%s "
            "restarts=%d error_warning=%d error_passive=%d bus_off=%d rx_dropped=%d",
            can_interface,
            status.state,
            status.bitrate,
            status.restarts,
            status.error_warning,
            status.error_passive,
            status.bus_off,
            status.rx_dropped,
        )
    return status


def validate_socketcan_interfaces(
    interfaces: Sequence[str],
    *,
    require_error_active: bool = False,
) -> dict[str, SocketCANStatus]:
    return {
        can_interface: _validate_socketcan_ready(
            can_interface,
            require_error_active=require_error_active,
        )
        for can_interface in interfaces
    }


def _sdk_factory(can_interface: str) -> Any:
    from piper_sdk import C_PiperInterface_V2

    return C_PiperInterface_V2(can_interface, False)


class PiperArmAdapter:
    """Thin Piper SDK adapter with no import-time hardware access."""

    def __init__(
        self,
        side: str,
        config: ArmConfig,
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.side = side
        self.config = config
        self._uses_default_sdk_factory = sdk_factory is None
        self._sdk_factory = sdk_factory or _sdk_factory
        self._sdk: Any | None = None
        self._connected = False
        self._hardware_lock_file: Any | None = None
        self._socketcan_status_at_open: SocketCANStatus | None = None
        self._last_motion_profile: tuple[int, int] | None = None
        self._last_gripper_request: float | None = None
        self._last_gripper_command: float | None = None
        self._last_gripper_feedback: float | None = None
        self._last_gripper_feedback_timestamp_s: float | None = None
        self._last_gripper_feedback_hz: float | None = None
        self._last_gripper_feedback_effort: float | None = None
        self._last_gripper_feedback_status_code: int | None = None
        self._gripper_stall_started_at: float | None = None
        self._gripper_progress_reference: float | None = None
        self._gripper_hold_fraction: float | None = None
        self._gripper_release_clear_armed = False
        self._gripper_release_reference: float | None = None
        self._gripper_stall_events = 0
        self._gripper_recovery_events = 0
        self._last_gripper_enable_recovery_at: float | None = None
        self._gripper_enable_recovery_attempts = 0
        self._last_arm_fault_signature: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def open_read_only(self) -> None:
        if self._connected:
            return
        if self._uses_default_sdk_factory:
            self._socketcan_status_at_open = _validate_socketcan_ready(
                self.config.can_interface
            )
        self._acquire_hardware_lock()
        try:
            sdk = self._sdk_factory(self.config.can_interface)
            if sdk.ConnectPort(piper_init=False) is False:
                raise ConnectionError(f"failed to open {self.config.can_interface}")
            self._sdk = sdk
            self._connected = True
            if self.config.control_backend == "host_ik":
                self._wait_for_joint_feedback()
        except BaseException:
            self.close()
            raise

    def _wait_for_joint_feedback(self) -> None:
        deadline = time.monotonic() + JOINT_FEEDBACK_READY_TIMEOUT_S
        while True:
            try:
                # Initial zero-Hz SDK buckets are expected while its 100 ms
                # FPS counter starts. Keep startup waiting separate from the
                # runtime recovery path so normal initialization is not logged
                # as a feedback dropout.
                self._read_joint_state_once()
                return
            except PiperFault as exc:
                if time.monotonic() >= deadline:
                    raise PiperFault(
                        f"{self.side} Piper joint feedback did not become ready"
                    ) from exc
            time.sleep(0.01)

    def prepare_motion(self) -> None:
        sdk = self._require_sdk()
        sdk.MasterSlaveConfig(0xFC, 0, 0, 0)
        sdk.EnableArm(7)

    def read_observation(self) -> ArmObservation:
        sdk = self._require_sdk()
        return self._observation_from_messages(sdk.GetArmEndPoseMsgs(), sdk.GetArmGripperMsgs())

    def read_joint_feedback(self) -> JointFeedback:
        message = self._require_sdk().GetArmJointMsgs()
        state = message.joint_state
        angles = tuple(float(getattr(state, f"joint_{index}")) / 1000.0 for index in range(1, 7))
        return JointFeedback(
            timestamp_s=float(getattr(message, "time_stamp", 0.0)),
            angles_degrees=angles,
            feedback_hz=float(getattr(message, "Hz", 0.0)),
        )

    def read_joint_state(self) -> JointState:
        try:
            return self._read_joint_state_once()
        except PiperFault as exc:
            if not self._uses_default_sdk_factory:
                raise
            try:
                status = _read_socketcan_status(self.config.can_interface)
            except ConnectionError as status_exc:
                raise PiperFault(
                    f"{exc}; refusing to wait and resume motion from stale input; "
                    f"SocketCAN inspection also failed: {status_exc}"
                ) from exc
            baseline = self._socketcan_status_at_open
            deltas = (
                ""
                if baseline is None
                else (
                    f", delta_restarts={status.restarts - baseline.restarts}"
                    f", delta_error_warning="
                    f"{status.error_warning - baseline.error_warning}"
                    f", delta_error_passive="
                    f"{status.error_passive - baseline.error_passive}"
                    f", delta_bus_off={status.bus_off - baseline.bus_off}"
                    f", delta_rx_dropped="
                    f"{status.rx_dropped - baseline.rx_dropped}"
                )
            )
            raise PiperFault(
                f"{exc}; refusing to wait and resume motion from stale input. "
                f"SocketCAN interface={status.interface}, state={status.state}, "
                f"restarts={status.restarts}, error_warning={status.error_warning}, "
                f"error_passive={status.error_passive}, bus_off={status.bus_off}, "
                f"rx_dropped={status.rx_dropped}{deltas}"
            ) from exc

    def _read_joint_state_once(self) -> JointState:
        feedback = self.read_joint_feedback()
        age_s = (
            time.time() - feedback.timestamp_s
            if math.isfinite(feedback.timestamp_s) and feedback.timestamp_s > 0.0
            else math.inf
        )
        if not feedback.is_complete:
            raise PiperFault(
                f"{self.side} Piper joint feedback is incomplete "
                f"(timestamp={feedback.timestamp_s:.6f}, "
                f"Hz={feedback.feedback_hz:.3f}, age={age_s:.3f}s)"
            )
        # The SDK aggregate can keep returning cached joint angles after CAN
        # feedback stops. Its timestamp is the wall-clock receive time, so Hz
        # alone is not sufficient to prove that the seed used by host IK is
        # current.
        if not 0.0 <= age_s <= JOINT_FEEDBACK_MAX_AGE_S:
            raise PiperFault(
                f"{self.side} Piper joint feedback is stale "
                f"(timestamp={feedback.timestamp_s:.6f}, "
                f"Hz={feedback.feedback_hz:.3f}, age={age_s:.3f}s)"
            )
        return JointState(
            timestamp_s=feedback.timestamp_s,
            angles_rad=tuple(math.radians(value) for value in feedback.angles_degrees),
            feedback_hz=feedback.feedback_hz,
        )

    @staticmethod
    def quantize_joint_target_rad(
        joints_rad: Sequence[float],
    ) -> tuple[tuple[float, float, float, float, float, float], tuple[int, int, int, int, int, int]]:
        values = tuple(float(value) for value in joints_rad)
        if len(values) != 6 or not all(map(math.isfinite, values)):
            raise ValueError("Piper joint target must contain six finite angles")
        raw = tuple(round(math.degrees(value) * 1000.0) for value in values)
        quantized = tuple(math.radians(value / 1000.0) for value in raw)
        return quantized, raw

    def send_joint_target_rad(
        self,
        joints_rad: Sequence[float],
        *,
        move_speed_percent: int | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        quantized, raw = self.quantize_joint_target_rad(joints_rad)
        self._set_mode(
            move_mode=0x01,
            move_speed_percent=move_speed_percent,
        )
        self._require_sdk().JointCtrl(*raw)
        return quantized

    def send_joint_target(
        self,
        joints_degrees: Sequence[float],
        *,
        move_speed_percent: int | None = None,
    ) -> None:
        values = tuple(float(value) for value in joints_degrees)
        if len(values) != 6 or not all(map(math.isfinite, values)):
            raise ValueError("Piper joint target must contain six finite angles")
        self.send_joint_target_rad(
            tuple(math.radians(value) for value in values),
            move_speed_percent=move_speed_percent,
        )

    def send_gripper_fraction(
        self,
        value: float | None,
        observed_fraction: float | None = None,
    ) -> tuple[float | None, bool]:
        if not self.config.gripper_enabled or value is None:
            return None, False
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Piper gripper target must be finite and in [0, 1]")
        if observed_fraction is not None:
            observed_fraction = float(observed_fraction)
            if not math.isfinite(observed_fraction):
                observed_fraction = None
            else:
                observed_fraction = min(1.0, max(0.0, observed_fraction))

        now = time.monotonic()
        opening_request = (
            self._last_gripper_request is not None
            and value
            > self._last_gripper_request
            + GRIPPER_REQUEST_DIRECTION_EPSILON_FRACTION
        )
        opening_edge = (
            self._gripper_release_clear_armed
            and opening_request
            and (
                (
                    self._gripper_release_reference is not None
                    and value >= self._gripper_release_reference + GRIPPER_RELEASE_EDGE_FRACTION
                )
                or (
                    self._gripper_hold_fraction is not None
                    and value >= self._gripper_hold_fraction + GRIPPER_RELEASE_EDGE_FRACTION
                )
            )
        )
        if opening_edge:
            self._gripper_hold_fraction = None
            self._reset_gripper_stall_tracking()
            self._gripper_release_clear_armed = False
            self._gripper_release_reference = None
            self._gripper_recovery_events += 1

        effective = value
        stall_transition = False
        if self._gripper_hold_fraction is not None and not opening_edge:
            effective = self._gripper_hold_fraction
        elif observed_fraction is not None:
            closing_gap = observed_fraction - value
            if opening_request or closing_gap <= GRIPPER_TARGET_GAP_FRACTION:
                self._reset_gripper_stall_tracking()
            else:
                if (
                    self._gripper_stall_started_at is None
                    or self._gripper_progress_reference is None
                ):
                    self._gripper_stall_started_at = now
                    self._gripper_progress_reference = observed_fraction
                elif (
                    observed_fraction
                    <= self._gripper_progress_reference
                    - GRIPPER_PROGRESS_EPSILON_FRACTION
                ):
                    # Measure progress cumulatively over the configured timeout
                    # instead of requiring a minimum displacement every 60 Hz
                    # control cycle. Slow but real closing motion must not be
                    # mistaken for a stall.
                    self._gripper_stall_started_at = now
                    self._gripper_progress_reference = observed_fraction
                elif now - self._gripper_stall_started_at >= self.config.gripper_stall_timeout_s:
                    self._gripper_hold_fraction = max(
                        value,
                        observed_fraction - self.config.gripper_stall_hold_offset_fraction,
                    )
                    effective = self._gripper_hold_fraction
                    stall_transition = True
                    self._gripper_release_clear_armed = True
                    self._gripper_release_reference = value
                    self._gripper_stall_events += 1
                    logger.warning(
                        "%s Piper gripper stopped making closing progress; "
                        "switching from requested=%.3f feedback=%.3f to low-preload hold=%.3f",
                        self.side,
                        value,
                        observed_fraction,
                        effective,
                    )

        feedback_live = (
            self._last_gripper_feedback_hz is not None
            and self._last_gripper_feedback_hz > 0.0
        )
        feedback_status_code = self._last_gripper_feedback_status_code
        feedback_needs_recovery = (
            feedback_live
            and feedback_status_code is not None
            and (
                not feedback_status_code & GRIPPER_DRIVER_ENABLE_BIT
                or bool(feedback_status_code & GRIPPER_FAULT_BITS)
            )
        )
        feedback_recovery_due = (
            feedback_needs_recovery
            and (
                self._last_gripper_enable_recovery_at is None
                or now - self._last_gripper_enable_recovery_at
                >= GRIPPER_RECOVERY_RETRY_INTERVAL_S
            )
        )
        recover_driver = (
            self._last_gripper_command is None
            or opening_edge
            or stall_transition
            or feedback_recovery_due
        )
        raw = round(effective * self.config.gripper_max_width_m * 1_000_000.0)
        sdk = self._require_sdk()
        if recover_driver:
            # Use the atomic enable+clear-error command. On the deployed
            # V1.8-9 controller, sending 0x02 and 0x01 back-to-back can be
            # applied as a lasting disable after contact/stall recovery,
            # dropping a grasped object. A single 0x03 keeps the driver enabled
            # while clearing the same error latch.
            sdk.GripperCtrl(raw, self.config.gripper_effort, 0x03, 0)
            self._last_gripper_enable_recovery_at = now
            self._gripper_enable_recovery_attempts += 1
            if feedback_recovery_due:
                logger.warning(
                    "%s Piper gripper feedback reports disabled/faulted "
                    "(status=0x%02X at %.1f Hz); sent atomic enable+clear-error",
                    self.side,
                    feedback_status_code,
                    self._last_gripper_feedback_hz,
                )
        else:
            sdk.GripperCtrl(raw, self.config.gripper_effort, 0x01, 0)
        if opening_edge:
            logger.info(
                "%s Piper gripper release edge detected; sent atomic "
                "enable+clear-error target=%.3f",
                self.side,
                effective,
            )
        self._last_gripper_request = value
        self._last_gripper_command = raw / (
            self.config.gripper_max_width_m * 1_000_000.0
        )
        if observed_fraction is not None:
            self._last_gripper_feedback = observed_fraction
        return self._last_gripper_command, True

    def gripper_diagnostics(self) -> dict[str, float | int | None]:
        return {
            "requested": self._last_gripper_request,
            "commanded": self._last_gripper_command,
            "feedback": self._last_gripper_feedback,
            "feedback_timestamp_s": self._last_gripper_feedback_timestamp_s,
            "feedback_hz": self._last_gripper_feedback_hz,
            "feedback_effort": self._last_gripper_feedback_effort,
            "feedback_status_code": self._last_gripper_feedback_status_code,
            "feedback_enabled": (
                None
                if self._last_gripper_feedback_status_code is None
                else bool(
                    self._last_gripper_feedback_status_code
                    & GRIPPER_DRIVER_ENABLE_BIT
                )
            ),
            "feedback_fault_bits": (
                None
                if self._last_gripper_feedback_status_code is None
                else self._last_gripper_feedback_status_code & GRIPPER_FAULT_BITS
            ),
            "hold": self._gripper_hold_fraction,
            "stall_events": self._gripper_stall_events,
            "recovery_events": self._gripper_recovery_events,
            "enable_recovery_attempts": self._gripper_enable_recovery_attempts,
        }

    def arm_status_diagnostics(self) -> dict[str, object]:
        message = self._require_sdk().GetArmStatus()
        status = message.arm_status
        status_code = int(status.arm_status)
        error_status = getattr(status, "err_status", None)
        angle_limit_joints = tuple(
            index
            for index in range(1, 7)
            if error_status is not None
            and bool(getattr(error_status, f"joint_{index}_angle_limit", False))
        )
        communication_error_joints = tuple(
            index
            for index in range(1, 7)
            if error_status is not None
            and bool(
                getattr(
                    error_status,
                    f"communication_status_joint_{index}",
                    False,
                )
            )
        )
        fault_signature = (
            status_code,
            angle_limit_joints,
            communication_error_joints,
        )
        has_fault = (
            status_code not in {0x00, 0x0B, 0x0C, 0x0D}
            or bool(angle_limit_joints)
            or bool(communication_error_joints)
        )
        if has_fault and fault_signature != self._last_arm_fault_signature:
            logger.warning(
                "%s Piper controller status code=0x%02X (%s) "
                "angle_limit_joints=%s communication_error_joints=%s",
                self.side,
                status_code,
                ARM_STATUS_NAMES.get(status_code, "unknown"),
                angle_limit_joints,
                communication_error_joints,
            )
        elif not has_fault and self._last_arm_fault_signature is not None:
            logger.info("%s Piper controller status returned to normal", self.side)
        self._last_arm_fault_signature = fault_signature if has_fault else None
        return {
            "timestamp_s": float(getattr(message, "time_stamp", 0.0)),
            "feedback_hz": float(getattr(message, "Hz", 0.0)),
            "status_code": status_code,
            "status": ARM_STATUS_NAMES.get(status_code, "unknown"),
            "control_mode": int(status.ctrl_mode),
            "move_mode": int(status.mode_feed),
            "motion_status": int(status.motion_status),
            "angle_limit_joints": angle_limit_joints,
            "communication_error_joints": communication_error_joints,
        }

    def apply_target(self, target: ArmTarget) -> AppliedArmCommand:
        if self.config.control_backend == "host_ik":
            raise PiperFault("host-IK targets must be dispatched through JointCtrl")
        position = np.asarray(target.position_m, dtype=np.float64) * 1000.0
        rotation = Rotation.from_quat(wxyz_to_xyzw(target.quaternion_wxyz)).as_rotvec()
        if not np.all(np.isfinite((*position, *rotation))):
            raise ValueError("Piper target must contain finite values")

        j6_position = tcp_to_j6(position, rotation, self.config.tcp_offset_mm)
        rpy_degrees = axis_angle_to_rpy_degrees(tuple(rotation))
        self._set_mode(move_mode=0x00)
        self._require_sdk().EndPoseCtrl(*(round(value * 1000.0) for value in (*j6_position, *rpy_degrees)))
        gripper, gripper_written = self.send_gripper_fraction(target.gripper_open_fraction)
        return AppliedArmCommand(
            command=ArmCommand(
                position_mm=tuple(float(value) for value in position),
                rotation_vector_rad=tuple(float(value) for value in rotation),
                gripper_open_fraction=gripper,
            ),
            pose_written=True,
            gripper_written=gripper_written,
        )

    def close(self) -> None:
        sdk = self._sdk
        self._connected = False
        self._last_motion_profile = None
        self._last_gripper_request = None
        self._last_gripper_command = None
        self._last_gripper_feedback = None
        self._last_gripper_feedback_timestamp_s = None
        self._last_gripper_feedback_hz = None
        self._last_gripper_feedback_effort = None
        self._last_gripper_feedback_status_code = None
        self._reset_gripper_stall_tracking()
        self._gripper_hold_fraction = None
        self._gripper_release_clear_armed = False
        self._gripper_release_reference = None
        self._last_gripper_enable_recovery_at = None
        self._gripper_enable_recovery_attempts = 0
        self._last_arm_fault_signature = None
        self._socketcan_status_at_open = None
        self._sdk = None
        try:
            if sdk is not None:
                sdk.DisconnectPort()
        finally:
            self._release_hardware_lock()

    def _observation_from_messages(self, end_pose_message: Any, gripper_message: Any) -> ArmObservation:
        pose = end_pose_message.end_pose
        position_mm = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1000.0
        rpy_degrees = (
            float(pose.RX_axis) / 1000.0,
            float(pose.RY_axis) / 1000.0,
            float(pose.RZ_axis) / 1000.0,
        )
        rotation_vector = np.asarray(rpy_degrees_to_axis_angle(rpy_degrees))
        tcp_mm = j6_to_tcp(position_mm, rotation_vector, self.config.tcp_offset_mm)
        quaternion_wxyz = xyzw_to_wxyz(tuple(Rotation.from_rotvec(rotation_vector).as_quat()))
        gripper_state = gripper_message.gripper_state
        gripper = (
            float(gripper_state.grippers_angle)
            / (self.config.gripper_max_width_m * 1_000_000.0)
        )
        self._last_gripper_feedback = gripper
        self._last_gripper_feedback_timestamp_s = float(
            getattr(gripper_message, "time_stamp", 0.0)
        )
        self._last_gripper_feedback_hz = float(getattr(gripper_message, "Hz", 0.0))
        self._last_gripper_feedback_effort = float(
            getattr(gripper_state, "grippers_effort", 0.0)
        )
        self._last_gripper_feedback_status_code = int(
            getattr(gripper_state, "status_code", 0)
        )
        return ArmObservation(
            position_m=tuple(float(value) / 1000.0 for value in tcp_mm),
            quaternion_wxyz=quaternion_wxyz,
            gripper_open_fraction=gripper,
        )

    def _set_mode(
        self,
        move_mode: int,
        *,
        move_speed_percent: int | None = None,
    ) -> None:
        speed_percent = (
            self.config.move_speed_percent
            if move_speed_percent is None
            else int(move_speed_percent)
        )
        if not 1 <= speed_percent <= 100:
            raise ValueError("Piper move speed percent must be in [1, 100]")
        profile = (move_mode, speed_percent)
        if self._last_motion_profile == profile:
            return
        self._require_sdk().ModeCtrl(0x01, move_mode, speed_percent, 0x00)
        self._last_motion_profile = profile

    def _require_sdk(self) -> Any:
        if not self._connected or self._sdk is None:
            raise ConnectionError(f"{self.side} Piper is not connected")
        return self._sdk

    def _reset_gripper_stall_tracking(self) -> None:
        self._gripper_stall_started_at = None
        self._gripper_progress_reference = None

    def _acquire_hardware_lock(self) -> None:
        if not self._uses_default_sdk_factory or self._hardware_lock_file is not None:
            return
        safe_interface = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.can_interface)
        lock_path = Path("/tmp") / f"starvla-piper-{safe_interface}.lock"
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "unknown owner"
            lock_file.close()
            raise ConnectionError(
                f"{self.config.can_interface} is already owned by another StarVLA Piper "
                f"process ({owner}); stop the existing teleoperate/record/probe/gohome "
                "process before starting another one"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            f"pid={os.getpid()} side={self.side} can={self.config.can_interface}\n"
        )
        lock_file.flush()
        self._hardware_lock_file = lock_file

    def _release_hardware_lock(self) -> None:
        lock_file = self._hardware_lock_file
        self._hardware_lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
