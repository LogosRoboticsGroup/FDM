from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Any

from .config import FlexivRobotConfig
from .contracts import FlexivCommandResult, FlexivObservation, FlexivTarget

logger = logging.getLogger(__name__)

GRIPPER_HOME_TOLERANCE_M = 0.002
GRIPPER_MAX_WIDTH_SAFETY_RATIO = 0.98


def _load_flexivrdk():
    try:
        import flexivrdk
    except ImportError as exc:
        raise RuntimeError(
            "Flexiv hardware access requires flexivrdk==1.8.0; install the SDK in this environment"
        ) from exc
    version = getattr(flexivrdk, "__version__", "unknown")
    major = int(str(version).split(".", 1)[0]) if str(version).split(".", 1)[0].isdigit() else None
    if major is not None and major != 1:
        raise RuntimeError(f"this teleoperator requires Flexiv RDK 1.x; found {version}")
    return flexivrdk


class FlexivRobot:
    def __init__(self, config: FlexivRobotConfig):
        self.config = config
        self._rdk = None
        self._robot: Any | None = None
        self._gripper: Any | None = None
        self._motion_enabled = False
        self._motion_prepared = False
        self._joint_motion_prepared = False
        self._joint_limits_rad: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        self._gripper_target_m: float | None = None
        self._gripper_width_m = 0.0
        self._gripper_min_width_m = self.config.gripper.close_width_m
        self._gripper_max_width_m = self.config.gripper.open_width_m
        self._gripper_velocity_m_s = self.config.gripper.velocity_m_s
        self._gripper_force_n = self.config.gripper.max_force_n
        self._gripper_error: BaseException | None = None
        self._gripper_lock = threading.Lock()
        self._gripper_condition = threading.Condition()
        self._gripper_commands: deque[tuple[str, float]] = deque()
        self._gripper_stop = threading.Event()
        self._gripper_thread: threading.Thread | None = None

    def open(self, *, enable_motion: bool) -> None:
        if self._robot is not None:
            return
        self._rdk = _load_flexivrdk()
        self._robot = self._rdk.Robot(
            self.config.serial_number,
            list(self.config.network_interface_whitelist),
        )
        info = self._robot.info()
        if int(info.DoF) != 7:
            raise RuntimeError(f"expected a 7-DoF Flexiv arm, connected robot reports DoF={info.DoF}")
        home = [math.radians(value) for value in self.config.home_joints_deg]
        if len(info.q_min) == 7 and len(info.q_max) == 7:
            self._joint_limits_rad = (
                tuple(float(value) for value in info.q_min),
                tuple(float(value) for value in info.q_max),
            )
            violations = [
                index + 1
                for index, (target, lower, upper) in enumerate(zip(home, info.q_min, info.q_max, strict=True))
                if not float(lower) <= target <= float(upper)
            ]
            if violations:
                raise ValueError(f"configured Flexiv home joints violate robot limits at joints {violations}")
        logger.info(
            "Connected Flexiv: serial=%s model=%s software=%s",
            info.serial_num,
            info.model_name,
            info.software_ver,
        )
        if not enable_motion:
            return
        if self._robot.fault():
            if not self.config.clear_fault_on_start:
                raise RuntimeError("Flexiv is in fault state and clear_fault_on_start=false")
            logger.warning("Flexiv is in fault state; requesting ClearFault")
            if not self._robot.ClearFault():
                raise RuntimeError("Flexiv fault could not be cleared")
        self._robot.Enable()
        deadline = time.monotonic() + 30.0
        while not self._robot.operational():
            if time.monotonic() >= deadline:
                raise TimeoutError("Flexiv did not become operational within 30 seconds")
            time.sleep(0.1)
        self._motion_enabled = True
        if self.config.gripper.enabled:
            self._gripper = self._rdk.Gripper(self._robot)
            self._gripper.Enable(self.config.gripper.device_name)
            tool = self._rdk.Tool(self._robot)
            tool.Switch(self.config.gripper.tool_name or self.config.gripper.device_name)
            if self.config.gripper.initialize_on_start:
                self._initialize_gripper()
            self._configure_gripper_limits()
            self._update_gripper_width()
            with self._gripper_lock:
                self._gripper_target_m = self._gripper_width_m
            self._start_gripper_worker()
            logger.info("Flexiv gripper enabled: %s", self.config.gripper.device_name)

    def _configure_gripper_limits(self) -> None:
        assert self._gripper is not None
        params = self._gripper.params()
        device_min_width_m = float(params.min_width)
        device_max_width_m = float(params.max_width)
        device_min_velocity_m_s = float(params.min_vel)
        device_max_velocity_m_s = float(params.max_vel)
        device_min_force_n = float(params.min_force)
        device_max_force_n = float(params.max_force)
        values = (
            device_min_width_m,
            device_max_width_m,
            device_min_velocity_m_s,
            device_max_velocity_m_s,
            device_min_force_n,
            device_max_force_n,
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Flexiv gripper reported non-finite parameters: {values}")
        if (
            device_min_width_m < 0
            or device_max_width_m <= device_min_width_m
            or device_min_velocity_m_s < 0
            or device_max_velocity_m_s <= 0
            or device_min_velocity_m_s > device_max_velocity_m_s
            or device_max_force_n <= 0
            or device_min_force_n > device_max_force_n
        ):
            raise RuntimeError(f"Flexiv gripper reported invalid parameters: {values}")

        configured = self.config.gripper
        self._gripper_min_width_m = max(configured.close_width_m, device_min_width_m)
        # GN01 reports 100 mm as its nominal maximum but its controller can reject a
        # command exactly on that boundary. Keep the same 2% endpoint margin used by
        # the known-good Flexiv collector path.
        device_safe_max_width_m = device_min_width_m + GRIPPER_MAX_WIDTH_SAFETY_RATIO * (
            device_max_width_m - device_min_width_m
        )
        self._gripper_max_width_m = min(configured.open_width_m, device_safe_max_width_m)
        if self._gripper_min_width_m >= self._gripper_max_width_m:
            raise ValueError(
                "configured gripper width range does not overlap device range: "
                f"configured=[{configured.close_width_m:.4f}, {configured.open_width_m:.4f}]m "
                f"device=[{device_min_width_m:.4f}, {device_max_width_m:.4f}]m"
            )
        self._gripper_velocity_m_s = min(
            device_max_velocity_m_s,
            max(device_min_velocity_m_s, configured.velocity_m_s),
        )
        self._gripper_force_n = min(
            device_max_force_n,
            max(device_min_force_n, configured.max_force_n),
        )
        if (
            self._gripper_min_width_m != configured.close_width_m
            or self._gripper_max_width_m != configured.open_width_m
            or self._gripper_velocity_m_s != configured.velocity_m_s
            or self._gripper_force_n != configured.max_force_n
        ):
            logger.warning(
                "Configured gripper command range was limited by device parameters: "
                "width=[%.4f, %.4f]m velocity=%.4fm/s force=%.1fN",
                self._gripper_min_width_m,
                self._gripper_max_width_m,
                self._gripper_velocity_m_s,
                self._gripper_force_n,
            )
        logger.info(
            "Flexiv gripper parameters: device_width=[%.4f, %.4f]m "
            "device_velocity=[%.4f, %.4f]m/s device_force=[%.1f, %.1f]N "
            "effective_width=[%.4f, %.4f]m velocity=%.4fm/s force=%.1fN",
            device_min_width_m,
            device_max_width_m,
            device_min_velocity_m_s,
            device_max_velocity_m_s,
            device_min_force_n,
            device_max_force_n,
            self._gripper_min_width_m,
            self._gripper_max_width_m,
            self._gripper_velocity_m_s,
            self._gripper_force_n,
        )

    def gripper_width_limits(self) -> tuple[float, float]:
        return self._gripper_min_width_m, self._gripper_max_width_m

    def _clamp_gripper_width(self, width_m: float) -> float:
        if not math.isfinite(width_m):
            raise ValueError(f"Flexiv gripper target width must be finite, got {width_m}")
        return min(self._gripper_max_width_m, max(self._gripper_min_width_m, width_m))

    def _initialize_gripper(self) -> None:
        assert self._gripper is not None
        self._gripper.Init()
        time.sleep(0.5)
        deadline = time.monotonic() + self.config.gripper.initialization_timeout_s
        stable_since: float | None = None
        while time.monotonic() < deadline:
            if self._gripper.states().is_moving:
                stable_since = None
            elif stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.5:
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"Flexiv gripper initialization timed out after " f"{self.config.gripper.initialization_timeout_s:.1f}s"
        )

    def _update_gripper_width(self) -> None:
        assert self._gripper is not None
        width_m = float(self._gripper.states().width)
        if not math.isfinite(width_m):
            raise RuntimeError(f"Flexiv gripper reported a non-finite width: {width_m}")
        with self._gripper_lock:
            self._gripper_width_m = width_m

    def _start_gripper_worker(self) -> None:
        self._gripper_stop.clear()
        with self._gripper_lock:
            self._gripper_error = None
        with self._gripper_condition:
            self._gripper_commands.clear()
        self._gripper_thread = threading.Thread(
            target=self._gripper_worker,
            name="flexiv-gripper-io",
            daemon=True,
        )
        self._gripper_thread.start()

    def _gripper_worker(self) -> None:
        assert self._gripper is not None
        next_poll_at = time.monotonic()
        while not self._gripper_stop.is_set():
            with self._gripper_condition:
                if not self._gripper_commands:
                    self._gripper_condition.wait(timeout=0.05)
                command = self._gripper_commands.popleft() if self._gripper_commands else None
            if self._gripper_stop.is_set():
                return
            try:
                if command is not None:
                    self._execute_gripper_command(*command)
                if time.monotonic() >= next_poll_at:
                    self._update_gripper_width()
                    next_poll_at = time.monotonic() + 0.1
            except Exception as exc:
                with self._gripper_lock:
                    self._gripper_error = exc
                logger.exception("Flexiv asynchronous gripper I/O failed")
                self._gripper_stop.set()
                return

    def _execute_gripper_command(self, kind: str, target_m: float) -> None:
        assert self._gripper is not None
        if kind not in {"move", "continuous_move", "stop_at"}:
            raise ValueError(f"unknown gripper command: {kind}")
        if kind == "stop_at":
            self._gripper.Stop()
            self._update_gripper_width()
            with self._gripper_lock:
                self._gripper_target_m = self._gripper_width_m
            return
        target_m = self._clamp_gripper_width(target_m)
        if kind == "continuous_move" or target_m > 0.001:
            self._gripper.Move(target_m, self._gripper_velocity_m_s, self._gripper_force_n)
        else:
            self._gripper.Grasp(self._gripper_force_n)

    def _queue_gripper_command(
        self,
        kind: str,
        target_m: float,
        *,
        coalesce_moves: bool = False,
    ) -> None:
        with self._gripper_lock:
            error = self._gripper_error
        if error is not None:
            raise RuntimeError("Flexiv gripper worker is unavailable after an I/O failure") from error
        with self._gripper_condition:
            if coalesce_moves:
                while self._gripper_commands and self._gripper_commands[-1][0] == "move":
                    self._gripper_commands.pop()
            self._gripper_commands.append((kind, target_m))
            self._gripper_condition.notify_all()

    def _stop_gripper_worker(self) -> None:
        self._gripper_stop.set()
        with self._gripper_condition:
            self._gripper_condition.notify_all()
        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)
            if self._gripper_thread.is_alive():
                logger.warning("Flexiv gripper worker did not stop before close")
            self._gripper_thread = None

    def read_observation(self) -> FlexivObservation:
        if self._robot is None:
            raise RuntimeError("Flexiv robot is not open")
        states = self._robot.states()
        pose = tuple(float(value) for value in states.tcp_pose)
        joints = tuple(float(value) for value in states.q)
        if len(pose) != 7 or len(joints) != 7:
            raise RuntimeError(f"unexpected Flexiv state dimensions: tcp_pose={len(pose)} q={len(joints)}")
        with self._gripper_lock:
            gripper_error = self._gripper_error
            gripper_width = self._gripper_width_m
        if gripper_error is not None:
            raise RuntimeError("Flexiv gripper worker failed") from gripper_error
        return FlexivObservation(pose, joints, gripper_width)

    def go_home(self, *, tolerance_deg: float, timeout_s: float) -> FlexivObservation:
        if self._robot is None or self._rdk is None:
            raise RuntimeError("Flexiv robot is not open")
        if not self._motion_enabled:
            raise RuntimeError("Flexiv motion was not enabled")
        home = [math.radians(value) for value in self.config.home_joints_deg]
        gripper_home_m = self._gripper_max_width_m if self.config.gripper.enabled else None
        if gripper_home_m is not None:
            # The gripper worker can open in parallel with the arm's joint-space home motion.
            self.move_gripper(gripper_home_m)
        self._motion_prepared = False
        self._joint_motion_prepared = False
        deadline = time.monotonic() + timeout_s
        self._robot.SwitchMode(self._rdk.Mode.NRT_JOINT_POSITION)
        self._robot.SendJointPosition(
            home,
            [0.0] * 7,
            list(self.config.home_max_velocity_rad_s),
            list(self.config.home_max_acceleration_rad_s2),
        )
        tolerance_rad = math.radians(tolerance_deg)
        while True:
            observation = self.read_observation()
            joint_error = max(
                abs(current - target) for current, target in zip(observation.joint_positions_rad, home, strict=True)
            )
            gripper_error = 0.0 if gripper_home_m is None else abs(observation.gripper_width_m - gripper_home_m)
            if joint_error <= tolerance_rad and gripper_error <= GRIPPER_HOME_TOLERANCE_M:
                return observation
            if self._robot.fault() or not self._robot.operational():
                raise RuntimeError("Flexiv became non-operational while moving home")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Flexiv home timed out after {timeout_s:.1f}s; "
                    f"max joint error={math.degrees(joint_error):.3f}deg; "
                    f"gripper error={gripper_error * 1000.0:.1f}mm"
                )
            time.sleep(0.02)

    def prepare_joint_motion(self) -> None:
        if self._robot is None or self._rdk is None:
            raise RuntimeError("Flexiv robot is not open")
        if not self._motion_enabled:
            raise RuntimeError("Flexiv motion was not enabled")
        self._robot.SwitchMode(self._rdk.Mode.NRT_JOINT_POSITION)
        self._joint_motion_prepared = True
        self._motion_prepared = False

    def apply_joint_target(
        self,
        joints_rad: tuple[float, float, float, float, float, float, float],
        gripper_width_m: float,
    ) -> tuple[float, ...]:
        if self._robot is None:
            raise RuntimeError("Flexiv robot is not open")
        if len(joints_rad) != 7 or not all(math.isfinite(value) for value in joints_rad):
            raise ValueError("Flexiv replay joint target must contain seven finite values")
        if not math.isfinite(gripper_width_m):
            raise ValueError("Flexiv replay gripper target must be finite")
        if self._joint_limits_rad is not None:
            lower, upper = self._joint_limits_rad
            violations = [
                index + 1
                for index, (target, minimum, maximum) in enumerate(zip(joints_rad, lower, upper, strict=True))
                if not minimum <= target <= maximum
            ]
            if violations:
                raise ValueError(f"Flexiv replay target violates robot limits at joints {violations}")
        if not self._joint_motion_prepared:
            self.prepare_joint_motion()
        self._robot.SendJointPosition(
            list(joints_rad),
            [0.0] * 7,
            list(self.config.home_max_velocity_rad_s),
            list(self.config.home_max_acceleration_rad_s2),
        )
        if self.config.gripper.enabled and (
            self._gripper_target_m is None or abs(gripper_width_m - self._gripper_target_m) > 1e-5
        ):
            self.move_gripper(gripper_width_m)
        with self._gripper_lock:
            gripper_target = self._gripper_target_m
            if gripper_target is None:
                gripper_target = self._gripper_width_m
        return (*joints_rad, gripper_target)

    def prepare_cartesian_motion(self) -> None:
        if self._robot is None or self._rdk is None:
            raise RuntimeError("Flexiv robot is not open")
        if not self._motion_enabled:
            raise RuntimeError("Flexiv motion was not enabled")
        self._robot.SwitchMode(self._rdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        observation = self.read_observation()
        set_nullspace = getattr(self._robot, "SetNullSpacePosture", None)
        if callable(set_nullspace):
            set_nullspace(list(observation.joint_positions_rad))
        self._robot.SendCartesianMotionForce(
            list(observation.tcp_pose_wxyz),
            [0.0] * 6,
            [0.0] * 6,
            self.config.max_linear_velocity_m_s,
            self.config.max_angular_velocity_rad_s,
            self.config.max_linear_acceleration_m_s2,
            self.config.max_angular_acceleration_rad_s2,
        )
        self._motion_prepared = True
        self._joint_motion_prepared = False

    def apply_target(
        self,
        target: FlexivTarget | None,
        observation: FlexivObservation,
    ) -> FlexivCommandResult:
        if target is None:
            return FlexivCommandResult(observation=observation)
        if self._robot is None:
            raise RuntimeError("Flexiv robot is not open")
        recovered = self._recover_cartesian_mode_if_needed()
        if recovered:
            return FlexivCommandResult(observation=observation, control_mode_recovered=True)
        if not self._motion_prepared:
            self.prepare_cartesian_motion()
        try:
            self._robot.SendCartesianMotionForce(
                list(target.tcp_pose_wxyz),
                [0.0] * 6,
                [0.0] * 6,
                self.config.max_linear_velocity_m_s,
                self.config.max_angular_velocity_rad_s,
                self.config.max_linear_acceleration_m_s2,
                self.config.max_angular_acceleration_rad_s2,
            )
        except Exception as exc:
            self._motion_prepared = False
            fault, operational, mode = self._control_status()
            expected = self._rdk.Mode.NRT_CARTESIAN_MOTION_FORCE
            if not fault and operational and mode != expected:
                logger.warning(
                    "Flexiv left Cartesian mode during command; anchoring at feedback pose " "before resuming: mode=%s",
                    mode,
                )
                self.prepare_cartesian_motion()
                return FlexivCommandResult(observation=observation, control_mode_recovered=True)
            raise RuntimeError(
                "Flexiv Cartesian command failed: " f"fault={fault} operational={operational} mode={mode}"
            ) from exc
        return FlexivCommandResult(
            observation=observation,
            target=target,
            pose_written=True,
        )

    def _control_status(self) -> tuple[bool, bool, Any]:
        assert self._robot is not None
        return bool(self._robot.fault()), bool(self._robot.operational()), self._robot.mode()

    def _recover_cartesian_mode_if_needed(self) -> bool:
        assert self._robot is not None and self._rdk is not None
        fault, operational, mode = self._control_status()
        if fault or not operational:
            self._motion_prepared = False
            raise RuntimeError(
                "Flexiv is not ready for Cartesian motion: " f"fault={fault} operational={operational} mode={mode}"
            )
        expected = self._rdk.Mode.NRT_CARTESIAN_MOTION_FORCE
        if mode == expected:
            return False
        logger.warning(
            "Flexiv control mode changed unexpectedly; anchoring at feedback pose before resuming: "
            "actual=%s expected=%s",
            mode,
            expected,
        )
        self._motion_prepared = False
        self.prepare_cartesian_motion()
        return True

    def move_gripper(self, width_m: float) -> None:
        if self._gripper is None:
            return
        width_m = self._clamp_gripper_width(width_m)
        with self._gripper_lock:
            self._gripper_target_m = width_m
        self._queue_gripper_command("move", width_m, coalesce_moves=True)

    def start_gripper_motion(self, width_m: float) -> None:
        if self._gripper is None:
            return
        width_m = self._clamp_gripper_width(width_m)
        with self._gripper_lock:
            self._gripper_target_m = width_m
        self._queue_gripper_command("continuous_move", width_m)

    def stop_gripper_at(self, width_m: float) -> None:
        if self._gripper is None:
            return
        width_m = self._clamp_gripper_width(width_m)
        with self._gripper_lock:
            self._gripper_target_m = width_m
        self._queue_gripper_command("stop_at", width_m)

    def close(self) -> None:
        self._stop_gripper_worker()
        if self._gripper is not None:
            try:
                self._gripper.Stop()
            except Exception:
                logger.exception("Failed to stop Flexiv gripper during close")
            self._gripper = None
        if self._robot is not None and self._motion_enabled:
            try:
                self._robot.Stop()
            except Exception:
                logger.exception("Failed to stop Flexiv robot during close")
        self._robot = None
        self._motion_enabled = False
        self._motion_prepared = False
        self._joint_motion_prepared = False
        self._joint_limits_rad = None


class PreviewFlexivRobot:
    def __init__(self, config: FlexivRobotConfig):
        self.config = config
        self.observation = FlexivObservation(
            tcp_pose_wxyz=(0.45, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
            joint_positions_rad=tuple(math.radians(value) for value in config.home_joints_deg),
            gripper_width_m=config.gripper.open_width_m if config.gripper.enabled else 0.0,
        )

    def open(self, *, enable_motion: bool) -> None:
        return None

    def read_observation(self) -> FlexivObservation:
        return self.observation

    def go_home(self, *, tolerance_deg: float, timeout_s: float) -> FlexivObservation:
        self.observation = FlexivObservation(
            tcp_pose_wxyz=self.observation.tcp_pose_wxyz,
            joint_positions_rad=tuple(math.radians(value) for value in self.config.home_joints_deg),
            gripper_width_m=(
                self.config.gripper.open_width_m if self.config.gripper.enabled else self.observation.gripper_width_m
            ),
        )
        return self.observation

    def prepare_cartesian_motion(self) -> None:
        return None

    def gripper_width_limits(self) -> tuple[float, float]:
        return self.config.gripper.close_width_m, self.config.gripper.open_width_m

    def prepare_joint_motion(self) -> None:
        return None

    def apply_joint_target(
        self,
        joints_rad: tuple[float, float, float, float, float, float, float],
        gripper_width_m: float,
    ) -> tuple[float, ...]:
        self.observation = FlexivObservation(
            tcp_pose_wxyz=self.observation.tcp_pose_wxyz,
            joint_positions_rad=joints_rad,
            gripper_width_m=gripper_width_m,
        )
        return (*joints_rad, gripper_width_m)

    def apply_target(
        self,
        target: FlexivTarget | None,
        observation: FlexivObservation,
    ) -> FlexivCommandResult:
        if target is None:
            return FlexivCommandResult(observation=observation)
        self.observation = FlexivObservation(
            tcp_pose_wxyz=target.tcp_pose_wxyz,
            joint_positions_rad=observation.joint_positions_rad,
            gripper_width_m=(
                target.gripper_width_m if target.gripper_width_m is not None else observation.gripper_width_m
            ),
        )
        return FlexivCommandResult(observation=observation, target=target)

    def move_gripper(self, width_m: float) -> None:
        self.observation = FlexivObservation(
            tcp_pose_wxyz=self.observation.tcp_pose_wxyz,
            joint_positions_rad=self.observation.joint_positions_rad,
            gripper_width_m=width_m,
        )

    def start_gripper_motion(self, width_m: float) -> None:
        return None

    def stop_gripper_at(self, width_m: float) -> None:
        self.move_gripper(width_m)

    def close(self) -> None:
        return None
