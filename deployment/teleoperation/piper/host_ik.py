from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ArmConfig
from ..contracts import ArmCommand, ArmObservation, ArmTarget, IKRequest, JointState
from ..pose import wxyz_to_xyzw
from .official_kinematics import OfficialPiperIKThread, PiperKinematicsModel, RealtimePiperIK

logger = logging.getLogger(__name__)
IK_CONDITION_WARNING_THRESHOLD = 25.0


class HostIKFault(RuntimeError):
    pass


@dataclass(frozen=True)
class HostIKPlan:
    commands: dict[str, ArmCommand]
    joints: dict[str, tuple[float, float, float, float, float, float]]
    request_ids: dict[str, int] = field(default_factory=dict)
    request_lag: dict[str, int] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)


class HostIKCoordinator:
    def __init__(self, arms: dict[str, ArmConfig]) -> None:
        self.config = dict(arms)
        self.models: dict[str, PiperKinematicsModel] = {}
        self.solvers: dict[str, RealtimePiperIK] = {}
        self.workers: dict[str, OfficialPiperIKThread] = {}
        self._next_request_id = dict.fromkeys(arms, 0)
        self._last_inline_failure: dict[str, str | None] = dict.fromkeys(arms)
        self._inline_singularity_warning = dict.fromkeys(arms, False)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        try:
            for side, config in self.config.items():
                self.models[side] = PiperKinematicsModel(
                    config.ik_urdf_path,
                    config.ik_package_dir,
                    config.tool_from_j6_xyz_mm_rpy_deg,
                )
                if config.ik_execution_mode == "inline":
                    self.solvers[side] = RealtimePiperIK(
                        config.ik_urdf_path,
                        config.ik_package_dir,
                        config.tool_from_j6_xyz_mm_rpy_deg,
                    )
                else:
                    self.workers[side] = OfficialPiperIKThread(
                        name=side,
                        startup_timeout_s=config.ik_worker_startup_timeout_s,
                        solver_factory=lambda config=config: RealtimePiperIK(
                            config.ik_urdf_path,
                            config.ik_package_dir,
                            config.tool_from_j6_xyz_mm_rpy_deg,
                        ),
                    )
        except BaseException:
            self.close()
            raise
        self._started = True

    def close(self) -> None:
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()
        self.solvers.clear()
        self.models.clear()
        self._last_inline_failure = dict.fromkeys(self.config)
        self._inline_singularity_warning = dict.fromkeys(self.config, False)
        self._started = False

    def observation(self, side: str, joints: JointState, gripper_open_fraction: float) -> ArmObservation:
        command = self.models[side].tool_pose_from_joints(joints.angles_rad)
        x, y, z, w = Rotation.from_rotvec(command.rotation_vector_rad).as_quat()
        return ArmObservation(
            position_m=tuple(value / 1000.0 for value in command.position_mm),
            quaternion_wxyz=(float(w), float(x), float(y), float(z)),
            gripper_open_fraction=gripper_open_fraction,
            joint_positions_rad=joints.angles_rad,
        )

    def plan(
        self,
        targets: dict[str, ArmTarget | None],
        joints: dict[str, JointState],
    ) -> HostIKPlan:
        if not self._started:
            raise HostIKFault("host IK is not started")
        expected_request_ids = {}
        requested_commands = {}
        for side, target in targets.items():
            if target is None:
                continue
            self._next_request_id[side] += 1
            expected_request_ids[side] = self._next_request_id[side]
            seed_joints = self._joint_values(joints[side])
            command = self._command(target)
            requested_commands[side] = command
            worker = self.workers.get(side)
            if worker is not None:
                worker.submit(
                    IKRequest(
                        request_id=self._next_request_id[side],
                        target=command,
                        seed_joints_rad=seed_joints,
                    )
                )

        commands = {}
        solved_joints = {}
        request_ids = {}
        request_lag = {}
        status = {}
        for side, solver in self.solvers.items():
            expected_request_id = expected_request_ids.get(side)
            if expected_request_id is None:
                status[side] = "inactive"
                continue
            solution = solver.solve(
                requested_commands[side],
                self._joint_values(joints[side]),
            )
            if solution is None:
                failure = getattr(solver, "last_failure_reason", None) or "failed"
                status[side] = failure
                self._report_inline_failure(side, failure)
                continue
            self._report_inline_failure(side, None)
            commands[side] = requested_commands[side]
            solved_joints[side] = solution.joints_rad
            request_ids[side] = expected_request_id
            request_lag[side] = 0
            condition_number = getattr(solver, "last_condition_number", None)
            near_singularity = (
                condition_number is not None
                and np.isfinite(condition_number)
                and condition_number >= IK_CONDITION_WARNING_THRESHOLD
            )
            self._report_inline_singularity(
                side,
                near_singularity=near_singularity,
                condition_number=condition_number,
                joints_rad=solution.joints_rad,
            )
            status[side] = "near_singularity" if near_singularity else "current"

        active_wait_ms = [
            self.config[side].ik_result_wait_ms
            for side in expected_request_ids
            if side in self.workers
        ]
        result_deadline = time.monotonic() + (
            max(active_wait_ms, default=0.0) / 1000.0
        )
        for side, worker in self.workers.items():
            expected_request_id = expected_request_ids.get(side)
            if expected_request_id is None:
                worker.poll_latest()
                status[side] = "inactive"
                continue
            result = worker.poll_latest(
                min_request_id=expected_request_id,
                timeout_s=max(0.0, result_deadline - time.monotonic()),
            )
            if result is None:
                status[side] = "pending"
                continue
            if result.joints_rad is None:
                status[side] = (
                    "failed"
                    if result.request_id >= expected_request_id
                    else "pending"
                )
                continue
            commands[side] = result.target
            solved_joints[side] = result.joints_rad
            request_ids[side] = result.request_id
            request_lag[side] = max(0, self._next_request_id[side] - result.request_id)
            status[side] = "current" if request_lag[side] == 0 else "stale"
        return HostIKPlan(
            commands=commands,
            joints=solved_joints,
            request_ids=request_ids,
            request_lag=request_lag,
            status=status,
        )

    def _report_inline_failure(self, side: str, failure: str | None) -> None:
        previous = self._last_inline_failure.get(side)
        if failure == previous:
            return
        self._last_inline_failure[side] = failure
        if failure is not None:
            logger.warning(
                "%s Piper IK rejected the target (%s); no new joint command "
                "was sent for this arm",
                side,
                failure,
            )
        elif previous is not None:
            logger.info(
                "%s Piper IK recovered after %s; joint command transmission resumed",
                side,
                previous,
            )

    def solver_diagnostics(self) -> dict[str, dict[str, object]]:
        return {
            side: diagnostics()
            for side, solver in self.solvers.items()
            if callable(diagnostics := getattr(solver, "diagnostics", None))
        }

    def _report_inline_singularity(
        self,
        side: str,
        *,
        near_singularity: bool,
        condition_number: float | None,
        joints_rad: tuple[float, float, float, float, float, float],
    ) -> None:
        previous = self._inline_singularity_warning.get(side, False)
        if near_singularity == previous:
            return
        self._inline_singularity_warning[side] = near_singularity
        if near_singularity:
            logger.warning(
                "%s Piper IK is near a singular configuration "
                "(condition=%.2f, joint5=%.2fdeg); wrist joint targets can "
                "become highly sensitive to small hand motions",
                side,
                condition_number,
                math.degrees(joints_rad[4]),
            )
        else:
            logger.info("%s Piper IK moved away from the singular configuration", side)

    @staticmethod
    def _command(target: ArmTarget) -> ArmCommand:
        rotation = Rotation.from_quat(wxyz_to_xyzw(target.quaternion_wxyz)).as_rotvec()
        position = np.asarray(target.position_m, dtype=np.float64) * 1000.0
        if not np.all(np.isfinite((*position, *rotation))):
            raise HostIKFault("target contains nonfinite values")
        return ArmCommand(
            position_mm=tuple(float(value) for value in position),
            rotation_vector_rad=tuple(float(value) for value in rotation),
            gripper_open_fraction=target.gripper_open_fraction,
        )

    @staticmethod
    def _joint_values(state: JointState) -> tuple[float, float, float, float, float, float]:
        values = tuple(float(value) for value in state.angles_rad)
        if len(values) != 6 or not all(np.isfinite(values)):
            raise HostIKFault("joint feedback must contain six finite radians")
        return values
