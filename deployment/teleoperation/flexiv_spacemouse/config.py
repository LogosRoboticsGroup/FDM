from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np

from ..config import CameraConfig, RecordingConfig, RuntimeConfig

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

DEFAULT_GRIPPER_SCALE = 10.0


@dataclass(frozen=True)
class SpaceMouseConfig:
    device_path: str | None = None
    name_contains: str = "SpaceMouse"
    deadzone: float = 0.08
    activation_threshold: float = 0.12
    release_threshold: float = 0.06
    translation_speed_m_s: float = 0.10
    rotation_speed_rad_s: float = 0.12
    smoothing_alpha: float = 0.35
    rotation_frame: str = "tool"
    # Mapping calibrated by the existing SpaceMouse controller on this workstation.
    translation_to_world: Matrix3 = (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    rotation_to_world: Matrix3 = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class FlexivGripperConfig:
    enabled: bool = False
    device_name: str = ""
    tool_name: str | None = None
    initialize_on_start: bool = True
    initialization_timeout_s: float = 20.0
    open_button: int = 1
    close_button: int = 0
    open_width_m: float = 0.1
    close_width_m: float = 0.0
    step_width_m: float = 0.005
    velocity_m_s: float = 0.1
    max_force_n: float = 20.0
    gripper_scale: float = DEFAULT_GRIPPER_SCALE


@dataclass(frozen=True)
class FlexivRobotConfig:
    serial_number: str = "REPLACE_WITH_FLEXIV_SERIAL"
    network_interface_whitelist: tuple[str, ...] = ()
    clear_fault_on_start: bool = True
    home_joints_deg: tuple[float, float, float, float, float, float, float] = (0.0,) * 7
    home_max_velocity_rad_s: tuple[float, float, float, float, float, float, float] = (0.5,) * 7
    home_max_acceleration_rad_s2: tuple[float, float, float, float, float, float, float] = (1.0,) * 7
    max_linear_velocity_m_s: float = 0.10
    max_angular_velocity_rad_s: float = 0.30
    max_linear_acceleration_m_s2: float = 0.5
    max_angular_acceleration_rad_s2: float = 1.0
    workspace_min_m: tuple[float, float, float] = (0.05, -0.8, 0.02)
    workspace_max_m: tuple[float, float, float] = (1.0, 0.8, 1.2)
    max_position_lead_m: float = 0.06
    max_rotation_lead_rad: float = 0.35
    gripper: FlexivGripperConfig = field(default_factory=FlexivGripperConfig)


@dataclass(frozen=True)
class FlexivSpaceMouseConfig:
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(control_hz=60, hardware_access=False, motion_enabled=False)
    )
    spacemouse: SpaceMouseConfig = field(default_factory=SpaceMouseConfig)
    robot: FlexivRobotConfig = field(default_factory=FlexivRobotConfig)
    recording: RecordingConfig = field(
        default_factory=lambda: RecordingConfig(
            root="examples/Flexiv/recordings/flexiv_spacemouse",
            repo_id="local/flexiv_spacemouse",
            task="Flexiv SpaceMouse teleoperation",
        )
    )

    def validate(self) -> None:
        if self.runtime.control_hz <= 0:
            raise ValueError("runtime.control_hz must be positive")
        if self.runtime.motion_enabled and not self.runtime.hardware_access:
            raise ValueError("motion_enabled requires hardware_access")
        if self.runtime.hardware_access and not self.robot.serial_number.strip():
            raise ValueError("robot.serial_number is required for hardware access")
        self._validate_spacemouse()
        self._validate_robot()
        self._validate_recording()

    def _validate_spacemouse(self) -> None:
        mouse = self.spacemouse
        if mouse.device_path is not None and not mouse.device_path.strip():
            raise ValueError("spacemouse.device_path must not be empty")
        if not mouse.name_contains.strip():
            raise ValueError("spacemouse.name_contains must not be empty")
        if not 0.0 <= mouse.deadzone < 1.0:
            raise ValueError("spacemouse.deadzone must be in [0, 1)")
        if not 0.0 <= mouse.release_threshold < mouse.activation_threshold <= 1.0:
            raise ValueError(
                "spacemouse thresholds must satisfy 0 <= release_threshold < "
                "activation_threshold <= 1"
            )
        if mouse.deadzone >= mouse.activation_threshold:
            raise ValueError("spacemouse.deadzone must be smaller than activation_threshold")
        for name in ("translation_speed_m_s", "rotation_speed_rad_s"):
            if not math.isfinite(value := getattr(mouse, name)) or value <= 0:
                raise ValueError(f"spacemouse.{name} must be positive and finite")
        if not 0.0 < mouse.smoothing_alpha <= 1.0:
            raise ValueError("spacemouse.smoothing_alpha must be in (0, 1]")
        if mouse.rotation_frame not in {"tool", "world"}:
            raise ValueError("spacemouse.rotation_frame must be tool or world")
        self._validate_axis_matrix("translation_to_world", mouse.translation_to_world)
        self._validate_axis_matrix("rotation_to_world", mouse.rotation_to_world)

    @staticmethod
    def _validate_axis_matrix(name: str, values: Matrix3) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"spacemouse.{name} must be a finite 3x3 matrix")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or not math.isclose(
            float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6
        ):
            raise ValueError(f"spacemouse.{name} must be a proper orthonormal rotation matrix")

    def _validate_robot(self) -> None:
        robot = self.robot
        for name in (
            "home_joints_deg",
            "home_max_velocity_rad_s",
            "home_max_acceleration_rad_s2",
        ):
            values = getattr(robot, name)
            if len(values) != 7 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"robot.{name} must contain seven finite values")
        if not all(value > 0 for value in robot.home_max_velocity_rad_s):
            raise ValueError("robot.home_max_velocity_rad_s values must be positive")
        if not all(value > 0 for value in robot.home_max_acceleration_rad_s2):
            raise ValueError("robot.home_max_acceleration_rad_s2 values must be positive")
        for name in (
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_linear_acceleration_m_s2",
            "max_angular_acceleration_rad_s2",
            "max_position_lead_m",
            "max_rotation_lead_rad",
        ):
            if not math.isfinite(value := getattr(robot, name)) or value <= 0:
                raise ValueError(f"robot.{name} must be positive and finite")
        lower = np.asarray(robot.workspace_min_m, dtype=np.float64)
        upper = np.asarray(robot.workspace_max_m, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite([lower, upper])):
            raise ValueError("robot workspace bounds must contain three finite values")
        if np.any(lower >= upper):
            raise ValueError("robot.workspace_min_m must be smaller than workspace_max_m")
        gripper = robot.gripper
        if gripper.enabled and not gripper.device_name.strip():
            raise ValueError("robot.gripper.device_name is required when the gripper is enabled")
        if gripper.tool_name is not None and not gripper.tool_name.strip():
            raise ValueError("robot.gripper.tool_name must not be empty")
        if not math.isfinite(gripper.initialization_timeout_s) or gripper.initialization_timeout_s <= 0:
            raise ValueError("robot.gripper.initialization_timeout_s must be positive and finite")
        if min(gripper.open_button, gripper.close_button) < 0:
            raise ValueError("robot.gripper button indices must be non-negative")
        if gripper.open_button == gripper.close_button:
            raise ValueError("robot.gripper open_button and close_button must differ")
        if not 0 <= gripper.close_width_m < gripper.open_width_m:
            raise ValueError("robot.gripper widths must satisfy 0 <= close < open")
        if (
            not math.isfinite(gripper.step_width_m)
            or not 0 < gripper.step_width_m <= gripper.open_width_m - gripper.close_width_m
        ):
            raise ValueError("robot.gripper.step_width_m must be positive and no larger than the width range")
        if gripper.velocity_m_s <= 0 or gripper.max_force_n <= 0:
            raise ValueError("robot.gripper velocity and max force must be positive")
        if not math.isfinite(gripper.gripper_scale) or gripper.gripper_scale <= 0:
            raise ValueError("robot.gripper.gripper_scale must be positive and finite")

    def _validate_recording(self) -> None:
        recording = self.recording
        if not math.isfinite(recording.home_timeout_s) or recording.home_timeout_s <= 0:
            raise ValueError("recording.home_timeout_s must be positive and finite")
        if not math.isfinite(recording.home_tolerance_deg) or recording.home_tolerance_deg <= 0:
            raise ValueError("recording.home_tolerance_deg must be positive and finite")
        if not recording.enabled:
            return
        if recording.root is None or not recording.root.strip():
            raise ValueError("recording.root is required when recording is enabled")
        if not recording.repo_id.strip() or not recording.task.strip():
            raise ValueError("recording.repo_id and recording.task must not be empty")
        for name in (
            "fps",
            "async_queue_size",
            "image_writer_threads_per_camera",
            "video_encoding_workers",
        ):
            if getattr(recording, name) <= 0:
                raise ValueError(f"recording.{name} must be positive")
        for name in ("episode_finalize_timeout_s", "close_timeout_s"):
            if not math.isfinite(value := getattr(recording, name)) or value <= 0:
                raise ValueError(f"recording.{name} must be positive and finite")
        if not self.runtime.hardware_access and not recording.allow_preview:
            raise ValueError("preview recording requires recording.allow_preview=true")
        for name, camera in recording.cameras.items():
            self._validate_camera(name, camera)

    @staticmethod
    def _validate_camera(name: str, camera: CameraConfig) -> None:
        if not name.strip():
            raise ValueError("recording camera names must not be empty")
        if camera.type not in {"opencv", "intelrealsense"}:
            raise ValueError(f"recording.cameras.{name}.type must be opencv or intelrealsense")
        if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError(f"recording.cameras.{name} width, height and fps must be positive")
        if camera.color_mode not in {"rgb", "bgr"}:
            raise ValueError(f"recording.cameras.{name}.color_mode must be rgb or bgr")
        if camera.fourcc is not None and (len(camera.fourcc) != 4 or not camera.fourcc.isascii()):
            raise ValueError(f"recording.cameras.{name}.fourcc must be four ASCII characters")
        if camera.type == "opencv" and camera.index_or_path is None:
            raise ValueError(f"recording.cameras.{name}.index_or_path is required for opencv")
        if camera.type == "intelrealsense" and not camera.serial_number_or_name:
            raise ValueError(
                f"recording.cameras.{name}.serial_number_or_name is required for intelrealsense"
            )


def load_flexiv_spacemouse_config(path: str | Path) -> FlexivSpaceMouseConfig:
    with Path(path).open() as stream:
        config = draccus.load(FlexivSpaceMouseConfig, stream)
    config.validate()
    return config
