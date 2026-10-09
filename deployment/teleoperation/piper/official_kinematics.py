"""Host-side Piper-X forward and inverse kinematics."""

from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ArmCommand, IKRequest, IKResult

_STOP = object()


@dataclass(frozen=True)
class PiperIKSolution:
    joints_rad: tuple[float, float, float, float, float, float]


def packaged_piper_x_paths() -> tuple[Path, Path]:
    package_root = Path(__file__).with_name("assets")
    urdf = package_root / "agx_arm_description/agx_arm_urdf/piper_x/urdf/piper_x_description.urdf"
    return urdf, package_root


def resolve_piper_x_paths(urdf_path: str | None, package_dir: str | None) -> tuple[Path, Path]:
    default_urdf, default_package = packaged_piper_x_paths()
    urdf = Path(urdf_path).expanduser().resolve() if urdf_path else default_urdf.resolve()
    package = Path(package_dir).expanduser().resolve() if package_dir else default_package.resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"Piper IK URDF does not exist: {urdf}")
    if not package.is_dir():
        raise FileNotFoundError(f"Piper IK package directory does not exist: {package}")
    return urdf, package


class PiperKinematicsModel:
    def __init__(
        self,
        urdf_path: str | None,
        package_dir: str | None,
        tool_from_j6_xyz_mm_rpy_deg: Sequence[float],
    ) -> None:
        import pinocchio as pin

        self.pin = pin
        resolved_urdf, resolved_package = resolve_piper_x_paths(urdf_path, package_dir)
        self.urdf_path = str(resolved_urdf)
        self.package_dir = str(resolved_package)
        model = pin.buildModelFromUrdf(self.urdf_path)
        if model.nq != 6:
            raise ValueError(f"Expected a six-axis Piper model, got nq={model.nq}")
        self.robot = pin.RobotWrapper(model=model)
        self.lower_limits_rad = np.asarray(
            model.lowerPositionLimit,
            dtype=np.float64,
        ).copy()
        self.upper_limits_rad = np.asarray(
            model.upperPositionLimit,
            dtype=np.float64,
        ).copy()
        if (
            self.lower_limits_rad.shape != (6,)
            or self.upper_limits_rad.shape != (6,)
            or not np.all(np.isfinite((*self.lower_limits_rad, *self.upper_limits_rad)))
            or np.any(self.lower_limits_rad > self.upper_limits_rad)
        ):
            raise ValueError("Piper URDF joint limits do not define a valid six-axis range")

        transform = np.asarray(tuple(tool_from_j6_xyz_mm_rpy_deg), dtype=np.float64)
        if transform.shape != (6,) or not np.all(np.isfinite(transform)):
            raise ValueError("Piper tool transform must contain six finite values")
        placement = pin.SE3(
            pin.rpy.rpyToMatrix(*(math.radians(float(value)) for value in transform[3:])),
            transform[:3] / 1000.0,
        )
        joint6_id = model.getJointId("joint6")
        if joint6_id == 0:
            raise ValueError("Piper model does not contain joint6")
        self.tool_frame_id = model.addFrame(pin.Frame("starvla_tool", joint6_id, placement, pin.FrameType.OP_FRAME))
        self.data = model.createData()

    @staticmethod
    def joint_vector(joints_rad: Sequence[float]) -> np.ndarray:
        joints = np.asarray(tuple(joints_rad), dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("Piper joints must contain six finite radians")
        return joints

    def tool_pose_from_joints(self, joints_rad: Sequence[float]) -> ArmCommand:
        joints = self.joint_vector(joints_rad)
        self.pin.framesForwardKinematics(self.robot.model, self.data, joints)
        pose = self.data.oMf[self.tool_frame_id]
        return ArmCommand(
            position_mm=tuple(float(value) * 1000.0 for value in pose.translation),
            rotation_vector_rad=tuple(float(value) for value in self.pin.log3(pose.rotation)),
        )

    def target_se3(self, target: ArmCommand) -> Any:
        position = np.asarray(target.position_mm, dtype=np.float64)
        rotation = np.asarray(target.rotation_vector_rad, dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3,) or not np.all(np.isfinite((*position, *rotation))):
            raise ValueError("Piper IK target must contain finite XYZ and rotation-vector values")
        return self.pin.SE3(self.pin.exp3(rotation), position / 1000.0)


class OfficialPiperIK:
    def __init__(
        self,
        urdf_path: str | None,
        package_dir: str | None,
        tool_from_j6_xyz_mm_rpy_deg: Sequence[float],
    ) -> None:
        import casadi
        import pinocchio.casadi as cpin

        self.casadi = casadi
        self.cpin = cpin
        self.model = PiperKinematicsModel(urdf_path, package_dir, tool_from_j6_xyz_mm_rpy_deg)
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        casadi = self.casadi
        cmodel = self.cpin.Model(self.model.robot.model)
        cdata = cmodel.createData()
        joints = casadi.SX.sym("q", 6, 1)
        target = casadi.SX.sym("target", 4, 4)
        self.cpin.framesForwardKinematics(cmodel, cdata, joints)
        error = casadi.Function(
            "starvla_piper_ik_error",
            [joints, target],
            [self.cpin.log6(cdata.oMf[self.model.tool_frame_id].inverse() * self.cpin.SE3(target)).vector],
        )
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(6)
        self.param_target = self.opti.parameter(4, 4)
        self.param_seed = self.opti.parameter(6)
        error_vector = error(self.var_q, self.param_target)
        self.opti.minimize(casadi.sumsqr(error_vector) + 0.01 * casadi.sumsqr(self.var_q - self.param_seed))
        self.opti.solver("ipopt", {"ipopt": {"print_level": 0}, "print_time": False})

    def solve(self, target: ArmCommand, seed_joints_rad: Sequence[float]) -> PiperIKSolution | None:
        seed = self.model.joint_vector(seed_joints_rad)
        self.opti.set_initial(self.var_q, seed)
        self.opti.set_value(self.param_seed, seed)
        self.opti.set_value(self.param_target, self.model.target_se3(target).homogeneous)
        try:
            joints = np.asarray(self.opti.solve_limited().value(self.var_q), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            return None
        return PiperIKSolution(tuple(float(value) for value in joints))


class RealtimePiperIK:
    """Bounded-iteration damped least-squares IK for streaming teleoperation."""

    def __init__(
        self,
        urdf_path: str | None,
        package_dir: str | None,
        tool_from_j6_xyz_mm_rpy_deg: Sequence[float],
        *,
        max_iterations: int = 40,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.7,
        max_joint_step_rad: float = 0.15,
    ) -> None:
        self.model = PiperKinematicsModel(urdf_path, package_dir, tool_from_j6_xyz_mm_rpy_deg)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.step_size = step_size
        self.max_joint_step_rad = max_joint_step_rad
        self.last_failure_reason: str | None = None
        self.last_iterations = 0
        self.last_error_norm: float | None = None
        self.last_min_singular_value: float | None = None
        self.last_condition_number: float | None = None

    def solve(self, target: ArmCommand, seed_joints_rad: Sequence[float]) -> PiperIKSolution | None:
        pin = self.model.pin
        robot_model = self.model.robot.model
        joints = self.model.joint_vector(seed_joints_rad).copy()
        desired = self.model.target_se3(target)
        identity = np.eye(6, dtype=np.float64)
        self.last_iterations = 0
        self.last_error_norm = None
        self.last_min_singular_value = None
        self.last_condition_number = None
        last_jacobian: np.ndarray | None = None

        for iteration in range(self.max_iterations):
            self.last_iterations = iteration + 1
            pin.framesForwardKinematics(robot_model, self.model.data, joints)
            current = self.model.data.oMf[self.model.tool_frame_id]
            error = np.asarray(pin.log6(current.actInv(desired)).vector, dtype=np.float64).reshape(6)
            if not np.all(np.isfinite(error)):
                return self._reject("nonfinite_error")
            self.last_error_norm = float(np.linalg.norm(error))
            if self.last_error_norm < self.tolerance:
                if last_jacobian is None:
                    last_jacobian = np.asarray(
                        pin.computeFrameJacobian(
                            robot_model,
                            self.model.data,
                            joints,
                            self.model.tool_frame_id,
                            pin.ReferenceFrame.LOCAL,
                        ),
                        dtype=np.float64,
                    )
                self._update_jacobian_diagnostics(last_jacobian)
                self.last_failure_reason = None
                return PiperIKSolution(tuple(float(value) for value in joints))

            jacobian = np.asarray(
                pin.computeFrameJacobian(
                    robot_model,
                    self.model.data,
                    joints,
                    self.model.tool_frame_id,
                    pin.ReferenceFrame.LOCAL,
                ),
                dtype=np.float64,
            )
            last_jacobian = jacobian
            try:
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + self.damping * identity,
                    error,
                )
            except np.linalg.LinAlgError:
                return self._reject("singular_jacobian")
            if not np.all(np.isfinite(delta)):
                return self._reject("nonfinite_step")
            largest_step = float(np.max(np.abs(delta)))
            if largest_step > self.max_joint_step_rad:
                delta *= self.max_joint_step_rad / largest_step
            joints = pin.integrate(robot_model, joints, delta * self.step_size)

        if last_jacobian is not None:
            self._update_jacobian_diagnostics(last_jacobian)
        return self._reject("unreachable")

    def diagnostics(self) -> dict[str, float | int | str | None]:
        return {
            "iterations": self.last_iterations,
            "error_norm": self.last_error_norm,
            "min_singular_value": self.last_min_singular_value,
            "condition_number": self.last_condition_number,
            "failure": self.last_failure_reason,
        }

    def _reject(self, reason: str) -> None:
        self.last_failure_reason = reason
        return None

    def _update_jacobian_diagnostics(self, jacobian: np.ndarray) -> None:
        try:
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
        except np.linalg.LinAlgError:
            self.last_min_singular_value = None
            self.last_condition_number = None
            return
        self.last_min_singular_value = float(singular_values[-1])
        self.last_condition_number = (
            math.inf
            if self.last_min_singular_value <= 0.0
            else float(singular_values[0] / self.last_min_singular_value)
        )


class OfficialPiperIKThread:
    def __init__(
        self,
        *,
        name: str,
        solver_factory: Callable[[], Any],
        startup_timeout_s: float,
    ) -> None:
        self.name = name
        self._solver_factory = solver_factory
        self._requests: queue.Queue[IKRequest | object] = queue.Queue(maxsize=1)
        self._results: queue.Queue[IKResult] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(target=self._run, name=f"{name}-realtime-ik", daemon=True)
        self._thread.start()
        if not self._ready.wait(startup_timeout_s):
            self.close()
            raise TimeoutError(f"Timed out waiting {startup_timeout_s:.1f}s for {name} IK worker")
        if self._error is not None:
            self.close()
            raise RuntimeError(f"Failed to initialize {name} IK worker:\n{self._error}")

    def submit(self, request: IKRequest) -> None:
        self._raise_if_failed()
        self._replace(self._requests, request)

    def poll_latest(
        self,
        *,
        min_request_id: int | None = None,
        timeout_s: float = 0.0,
    ) -> IKResult | None:
        self._raise_if_failed()
        if timeout_s < 0.0 or not math.isfinite(timeout_s):
            raise ValueError("IK result timeout must be non-negative and finite")
        deadline = time.monotonic() + timeout_s
        latest = None
        while True:
            while True:
                try:
                    latest = self._results.get_nowait()
                except queue.Empty:
                    break
            if latest is not None and (
                min_request_id is None or latest.request_id >= min_request_id
            ):
                return latest
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return latest
            try:
                latest = self._results.get(timeout=remaining)
            except queue.Empty:
                return latest

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._replace(self._requests, _STOP)
        self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        try:
            solver = self._solver_factory()
            self._ready.set()
            while not self._stop.is_set():
                request = self._requests.get()
                if request is _STOP:
                    return
                while True:
                    try:
                        newer = self._requests.get_nowait()
                    except queue.Empty:
                        break
                    if newer is _STOP:
                        return
                    request = newer
                assert isinstance(request, IKRequest)
                solution = solver.solve(request.target, request.seed_joints_rad)
                self._replace(
                    self._results,
                    IKResult(
                        request_id=request.request_id,
                        target=request.target,
                        joints_rad=None if solution is None else solution.joints_rad,
                    ),
                )
        except BaseException:
            self._error = traceback.format_exc()
            self._ready.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self.name} IK worker failed:\n{self._error}")
        if not self._thread.is_alive():
            raise RuntimeError(f"{self.name} IK worker is not running")

    @staticmethod
    def _replace(target_queue: queue.Queue[Any], item: Any) -> None:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        target_queue.put_nowait(item)
