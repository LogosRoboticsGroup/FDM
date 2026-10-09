from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import yaml

from .contracts import SIDES

ToolTransform = tuple[float, float, float, float, float, float]
CONTROL_BACKENDS = {"host_ik", "firmware"}
IK_EXECUTION_MODES = {"inline", "threaded"}


@dataclass(frozen=True)
class RuntimeConfig:
    control_hz: int = 60
    hardware_access: bool = False
    motion_enabled: bool = False
    log_every: int = 60


@dataclass(frozen=True)
class VRConfig:
    bind_host: str = "127.0.0.1"
    https_port: int = 8443
    websocket_port: int = 8442
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    allowed_origins: tuple[str, ...] = ()
    grip_pitch_deg: float = -45.0
    position_scale: tuple[float, float, float] = (2.0, 2.0, 2.0)
    rotation_scale: float = 1.0
    smoothing_alpha: float = 0.4
    stale_timeout_s: float = 0.25
    vr_to_robot: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )


@dataclass(frozen=True)
class ArxSDKConfig:
    """Overrides for real-stanford/arx5-sdk; lengths are in metres."""

    gripper_open_readout: float | None = None
    gripper_velocity_limit_m_s: float = 0.04
    gripper_max_closing_error_m: float = 0.004
    home_duration_s: float = 5.0
    feedback_timeout_s: float = 0.25


@dataclass(frozen=True)
class ArmConfig:
    can_interface: str
    hand: str
    control_backend: str = "host_ik"
    tcp_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_from_j6_xyz_mm_rpy_deg: ToolTransform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ik_urdf_path: str | None = None
    ik_package_dir: str | None = None
    ik_worker_startup_timeout_s: float = 30.0
    ik_execution_mode: str = "threaded"
    ik_result_wait_ms: float = 2.0
    move_speed_percent: int = 10
    home_move_speed_percent: int | None = None
    home_joints_deg: tuple[float, float, float, float, float, float] = (0.0,) * 6
    home_gripper_open_fraction: float | None = None
    gripper_enabled: bool = False
    gripper_effort: int = 1000
    gripper_max_width_m: float = 0.068
    gripper_stall_timeout_s: float = 1.5
    gripper_stall_hold_offset_fraction: float = 0.01
    arx: ArxSDKConfig | None = None


@dataclass(frozen=True)
class CameraConfig:
    type: str
    index_or_path: int | str | None = None
    serial_number_or_name: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    color_mode: str = "rgb"
    fourcc: str | None = None


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False
    root: str | None = None
    repo_id: str = "local/piper_vr"
    task: str = "Piper VR teleoperation"
    fps: int = 30
    video: bool = True
    resume: bool = False
    allow_preview: bool = False
    async_queue_size: int = 4
    image_writer_threads_per_camera: int = 4
    video_encoding_workers: int = 1
    episode_finalize_timeout_s: float = 120.0
    close_timeout_s: float = 120.0
    home_timeout_s: float = 60.0
    home_tolerance_deg: float = 1.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class TeleoperationConfig:
    robot_type: str = "piper"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vr: VRConfig = field(default_factory=VRConfig)
    arms: dict[str, ArmConfig] = field(
        default_factory=lambda: {
            "left": ArmConfig(can_interface="can_left", hand="left"),
            "right": ArmConfig(can_interface="can_right", hand="right"),
        }
    )
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def validate(self) -> None:
        if self.robot_type not in {"piper", "arx_x5"}:
            raise ValueError("robot_type must be piper or arx_x5")
        if self.runtime.control_hz <= 0:
            raise ValueError("runtime.control_hz must be positive")
        if self.runtime.motion_enabled and not self.runtime.hardware_access:
            raise ValueError("motion_enabled requires hardware_access")
        if not 0 < self.vr.smoothing_alpha <= 1:
            raise ValueError("vr.smoothing_alpha must be in (0, 1]")
        if not math.isfinite(self.vr.stale_timeout_s) or self.vr.stale_timeout_s <= 0:
            raise ValueError("vr.stale_timeout_s must be positive and finite")
        if set(self.arms) != set(SIDES):
            raise ValueError("arms must contain exactly left and right")
        if len({arm.can_interface for arm in self.arms.values()}) != len(self.arms):
            raise ValueError("Piper arms must use distinct CAN interfaces")
        if {arm.hand for arm in self.arms.values()} != set(SIDES):
            raise ValueError("left and right VR hands must each map to one arm")
        for side, arm in self.arms.items():
            self._validate_arm(side, arm)
            if self.robot_type == "arx_x5":
                if arm.arx is None or arm.control_backend != "host_ik":
                    raise ValueError(f"arms.{side}: ARX requires arx settings and control_backend=host_ik")
                if arm.ik_execution_mode != "inline":
                    raise ValueError(f"arms.{side}: ARX uses ik_execution_mode=inline")
                if any(arm.tcp_offset_mm) or arm.ik_package_dir is not None:
                    raise ValueError("ARX uses tool_from_j6_xyz_mm_rpy_deg and SDK URDF paths")
                for name in (
                    "home_duration_s",
                    "feedback_timeout_s",
                    "gripper_velocity_limit_m_s",
                    "gripper_max_closing_error_m",
                ):
                    value = getattr(arm.arx, name)
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"arms.{side}.arx.{name} must be positive and finite")
                readout = arm.arx.gripper_open_readout
                if readout is not None and (not math.isfinite(readout) or readout == 0):
                    raise ValueError(f"arms.{side}.arx.gripper_open_readout must be finite and nonzero")
            elif arm.arx is not None:
                raise ValueError("arx settings require robot_type=arx_x5")
        if not math.isfinite(self.recording.home_timeout_s) or self.recording.home_timeout_s <= 0:
            raise ValueError("recording.home_timeout_s must be positive and finite")
        if not math.isfinite(self.recording.home_tolerance_deg) or self.recording.home_tolerance_deg <= 0:
            raise ValueError("recording.home_tolerance_deg must be positive and finite")
        if self.recording.enabled:
            if any(arm.control_backend == "firmware" for arm in self.arms.values()):
                raise ValueError(
                    "recording requires control_backend='host_ik' for both arms; "
                    "firmware does not provide executed joint targets"
                )
            if self.recording.root is None or not self.recording.root.strip():
                raise ValueError("recording.root is required when recording is enabled")
            if not self.recording.repo_id.strip():
                raise ValueError("recording.repo_id must not be empty")
            if not self.recording.task.strip():
                raise ValueError("recording.task must not be empty")
            if self.recording.fps <= 0:
                raise ValueError("recording.fps must be positive")
            if self.recording.async_queue_size <= 0:
                raise ValueError("recording.async_queue_size must be positive")
            if self.recording.image_writer_threads_per_camera <= 0:
                raise ValueError("recording.image_writer_threads_per_camera must be positive")
            if self.recording.video_encoding_workers <= 0:
                raise ValueError("recording.video_encoding_workers must be positive")
            if (
                not math.isfinite(self.recording.episode_finalize_timeout_s)
                or self.recording.episode_finalize_timeout_s <= 0
            ):
                raise ValueError("recording.episode_finalize_timeout_s must be positive and finite")
            if not math.isfinite(self.recording.close_timeout_s) or self.recording.close_timeout_s <= 0:
                raise ValueError("recording.close_timeout_s must be positive and finite")
            if not self.runtime.hardware_access and not self.recording.allow_preview:
                raise ValueError("preview recording requires recording.allow_preview=true")
            for name, camera in self.recording.cameras.items():
                self._validate_camera(name, camera)

    @staticmethod
    def _validate_arm(side: str, arm: ArmConfig) -> None:
        if arm.control_backend not in CONTROL_BACKENDS:
            raise ValueError(f"arms.{side}.control_backend must be one of {sorted(CONTROL_BACKENDS)}")
        if not 1 <= arm.move_speed_percent <= 100:
            raise ValueError(f"arms.{side}.move_speed_percent must be in [1, 100]")
        if arm.home_move_speed_percent is not None and not (1 <= arm.home_move_speed_percent <= 100):
            raise ValueError(f"arms.{side}.home_move_speed_percent must be null or in [1, 100]")
        if len(arm.tcp_offset_mm) != 3 or not all(math.isfinite(value) for value in arm.tcp_offset_mm):
            raise ValueError(f"arms.{side}.tcp_offset_mm must contain three finite values")
        if len(arm.tool_from_j6_xyz_mm_rpy_deg) != 6 or not all(
            math.isfinite(value) for value in arm.tool_from_j6_xyz_mm_rpy_deg
        ):
            raise ValueError(f"arms.{side}.tool_from_j6_xyz_mm_rpy_deg must contain six finite values")
        if len(arm.home_joints_deg) != 6 or not all(math.isfinite(value) for value in arm.home_joints_deg):
            raise ValueError(f"arms.{side}.home_joints_deg must contain six finite values")
        if arm.home_gripper_open_fraction is not None and (
            not math.isfinite(arm.home_gripper_open_fraction) or not 0.0 <= arm.home_gripper_open_fraction <= 1.0
        ):
            raise ValueError(f"arms.{side}.home_gripper_open_fraction must be null or in [0, 1]")
        if arm.home_gripper_open_fraction is not None and not arm.gripper_enabled:
            raise ValueError(f"arms.{side}.home_gripper_open_fraction requires gripper_enabled=true")
        if not math.isfinite(arm.ik_worker_startup_timeout_s) or arm.ik_worker_startup_timeout_s <= 0:
            raise ValueError(f"arms.{side}.ik_worker_startup_timeout_s must be positive and finite")
        if arm.ik_execution_mode not in IK_EXECUTION_MODES:
            raise ValueError(f"arms.{side}.ik_execution_mode must be one of " f"{sorted(IK_EXECUTION_MODES)}")
        if not math.isfinite(arm.ik_result_wait_ms) or arm.ik_result_wait_ms < 0:
            raise ValueError(f"arms.{side}.ik_result_wait_ms must be non-negative and finite")
        if arm.ik_urdf_path is not None and not arm.ik_urdf_path.strip():
            raise ValueError(f"arms.{side}.ik_urdf_path must not be empty")
        if arm.ik_package_dir is not None and not arm.ik_package_dir.strip():
            raise ValueError(f"arms.{side}.ik_package_dir must not be empty")
        if not math.isfinite(arm.gripper_max_width_m) or arm.gripper_max_width_m <= 0:
            raise ValueError(f"arms.{side}.gripper_max_width_m must be positive and finite")
        if arm.gripper_enabled and not 0 <= arm.gripper_effort <= 5000:
            raise ValueError(f"arms.{side}.gripper_effort must be in [0, 5000]")
        if not math.isfinite(arm.gripper_stall_timeout_s) or arm.gripper_stall_timeout_s <= 0:
            raise ValueError(f"arms.{side}.gripper_stall_timeout_s must be positive and finite")
        if (
            not math.isfinite(arm.gripper_stall_hold_offset_fraction)
            or not 0.0 <= arm.gripper_stall_hold_offset_fraction <= 0.1
        ):
            raise ValueError(f"arms.{side}.gripper_stall_hold_offset_fraction must be in [0, 0.1]")

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
            raise ValueError(f"recording.cameras.{name}.serial_number_or_name is required for intelrealsense")


def load_config(path: str | Path) -> TeleoperationConfig:
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if data.get("robot_type") == "arx_x5":
        data.pop("inference", None)
        data.pop("controls", None)
    config = draccus.decode(TeleoperationConfig, data)
    config.validate()
    return config
