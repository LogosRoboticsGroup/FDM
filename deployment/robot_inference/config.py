from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

import draccus
import yaml

from deployment.teleoperation.config import ArmConfig, CameraConfig, RecordingConfig, TeleoperationConfig, VRConfig
from deployment.teleoperation.contracts import SIDES
from deployment.teleoperation.flexiv_spacemouse.config import (
    FlexivRobotConfig,
    FlexivSpaceMouseConfig,
)

FLEXIV_POLICY_DIMS = {
    "joint_position": (8, 8),
    "cartesian": (10, 10),
}


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    """Hardware gates for real-robot inference.

    Both flags must be enabled before the CLI will accept ``--execute``.
    Keeping them separate makes a checked-in configuration safe by default.
    """

    hardware_access: bool = False
    motion_enabled: bool = False


@dataclass(frozen=True)
class PolicyClientConfig:
    server: str = "127.0.0.1:5555"
    stat_key: str | None = None
    prompt: str = ""
    camera_order: tuple[str, ...] = ()
    fps: float = 30.0
    n_execute: int = 5
    jpeg_quality: int = 90
    recv_timeout_ms: int = 60_000
    send_timeout_ms: int = 5_000
    max_retries: int = 3
    state_dim: int | None = None
    action_dim: int | None = None
    action_space: str = "auto"
    strict_server_metadata: bool = True
    home_on_start: bool = True
    home_tolerance_deg: float = 1.0
    home_timeout_s: float = 60.0
    log_every_chunks: int = 1


class PolicyMetadataConfig(Protocol):
    """Fields required to validate an InferSystem-compatible endpoint."""

    camera_order: tuple[str, ...]
    state_dim: int | None
    action_dim: int | None
    action_space: str
    stat_key: str | None
    strict_server_metadata: bool


@dataclass(frozen=True)
class PiperPolicyClientConfig:
    """Policy and execution settings for two Piper arms.

    State and action follow the matching recorder fields, left then right:
    ``cartesian`` uses ``state.eef`` / ``action.eef`` with
    ``[x_mm, y_mm, z_mm, rotvec_xyz_rad, gripper_open]`` per arm;
    ``joint_position`` uses ``state.joint`` / ``action.joint`` with
    ``[q1..q6_rad, gripper_open]`` per arm.
    """

    server: str = "127.0.0.1:5555"
    stat_key: str | None = None
    prompt: str = ""
    camera_order: tuple[str, ...] = ()
    fps: float = 30.0
    n_execute: int = 5
    jpeg_quality: int = 90
    recv_timeout_ms: int = 60_000
    send_timeout_ms: int = 5_000
    max_retries: int = 3
    state_dim: int = 14
    action_dim: int = 14
    action_space: str = "cartesian"
    strict_server_metadata: bool = True
    home_on_start: bool = True
    home_tolerance_deg: float = 1.0
    home_timeout_s: float = 60.0
    log_every_chunks: int = 1


@dataclass(frozen=True)
class FlexivInferenceConfig:
    """Configuration for joint or Cartesian inference on one Flexiv arm."""

    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    robot: FlexivRobotConfig = field(default_factory=FlexivRobotConfig)
    inference: PolicyClientConfig = field(default_factory=PolicyClientConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    controls: str = "keyboard"
    vr: VRConfig = field(default_factory=VRConfig)
    recording: RecordingConfig = field(
        default_factory=lambda: RecordingConfig(
            enabled=True, root="results/flexiv_inference", repo_id="local/flexiv_inference"
        )
    )

    def validate(self) -> None:
        _validate_interaction(self)
        runtime = self.runtime
        policy = self.inference

        if runtime.motion_enabled and not runtime.hardware_access:
            raise ValueError("runtime.motion_enabled requires runtime.hardware_access")
        # Reuse the hardware, limit, home, and gripper checks used by Flexiv
        # teleoperation instead of maintaining a second set of robot semantics.
        FlexivSpaceMouseConfig(robot=self.robot).validate()
        if not policy.server.strip():
            raise ValueError("inference.server must not be empty")
        _validate_server_address(policy.server)
        if policy.stat_key is not None and not policy.stat_key.strip():
            raise ValueError("inference.stat_key must not be empty when provided")
        if not policy.prompt.strip():
            raise ValueError("inference.prompt must not be empty")
        if not math.isfinite(policy.fps) or policy.fps <= 0:
            raise ValueError("inference.fps must be positive and finite")
        if policy.n_execute <= 0:
            raise ValueError("inference.n_execute must be positive")
        if not 1 <= policy.jpeg_quality <= 100:
            raise ValueError("inference.jpeg_quality must be in [1, 100]")
        if min(policy.recv_timeout_ms, policy.send_timeout_ms, policy.max_retries) <= 0:
            raise ValueError("inference timeouts and max_retries must be positive")

        if policy.action_space not in {"auto", *FLEXIV_POLICY_DIMS}:
            raise ValueError("inference.action_space must be auto, cartesian, or joint_position")
        for name in ("state_dim", "action_dim"):
            value = getattr(policy, name)
            if value is not None and value <= 0:
                raise ValueError(f"inference.{name} must be positive when provided")
        if policy.action_space != "auto":
            expected_dims = FLEXIV_POLICY_DIMS[policy.action_space]
            configured_dims = (policy.state_dim, policy.action_dim)
            if any(value is not None for value in configured_dims) and configured_dims != expected_dims:
                raise ValueError(
                    f"Flexiv {policy.action_space} inference requires "
                    f"state_dim={expected_dims[0]} and action_dim={expected_dims[1]}"
                )
        if not self.robot.gripper.enabled:
            raise ValueError(
                "Flexiv inference requires robot.gripper.enabled=true so the final "
                "state/action dimension has the training-time meaning"
            )

        order = policy.camera_order
        if not order:
            raise ValueError("inference.camera_order must contain at least one camera")
        if len(set(order)) != len(order):
            raise ValueError("inference.camera_order contains duplicate camera names")
        missing = sorted(set(order) - set(self.cameras))
        unexpected = sorted(set(self.cameras) - set(order))
        if missing or unexpected:
            raise ValueError(
                "inference.camera_order must exactly match cameras; " f"missing={missing}, unexpected={unexpected}"
            )
        for name, camera in self.cameras.items():
            _validate_camera(name, camera)

        for name in ("home_tolerance_deg", "home_timeout_s"):
            value = getattr(policy, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"inference.{name} must be positive and finite")
        if policy.log_every_chunks < 0:
            raise ValueError("inference.log_every_chunks must be non-negative")


@dataclass(frozen=True)
class DualPiperInferenceConfig:
    """Standalone synchronous inference for a pair of Piper or ARX X5 arms.

    Both arms retain the ``host_ik`` configuration used while recording so
    observations provide matching joint feedback and tool-frame poses.
    """

    robot_type: str = "piper"
    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    arms: dict[str, ArmConfig] = field(
        default_factory=lambda: {
            "left": ArmConfig(can_interface="can_left", hand="left"),
            "right": ArmConfig(can_interface="can_right", hand="right"),
        }
    )
    inference: PiperPolicyClientConfig = field(default_factory=PiperPolicyClientConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    controls: str = "vr"
    vr: VRConfig = field(default_factory=VRConfig)
    recording: RecordingConfig = field(
        default_factory=lambda: RecordingConfig(
            enabled=True, root="results/piper_inference", repo_id="local/piper_inference"
        )
    )

    def validate(self) -> None:
        _validate_interaction(self)
        runtime = self.runtime
        policy = self.inference

        if runtime.motion_enabled and not runtime.hardware_access:
            raise ValueError("runtime.motion_enabled requires runtime.hardware_access")
        # Reuse all Piper CAN, home, tool-frame, IK, and gripper checks.
        TeleoperationConfig(robot_type=self.robot_type, arms=self.arms).validate()
        if set(self.arms) != set(SIDES):
            raise ValueError("arms must contain exactly left and right")
        for side, arm in self.arms.items():
            if arm.control_backend != "host_ik":
                raise ValueError(f"arms.{side}.control_backend must be host_ik for inference state compatibility")
            if not arm.gripper_enabled:
                raise ValueError(f"arms.{side}.gripper_enabled must be true for the 14-D inference schema")

        if not policy.server.strip():
            raise ValueError("inference.server must not be empty")
        _validate_server_address(policy.server)
        if policy.stat_key is not None and not policy.stat_key.strip():
            raise ValueError("inference.stat_key must not be empty when provided")
        if not policy.prompt.strip():
            raise ValueError("inference.prompt must not be empty")
        if not math.isfinite(policy.fps) or policy.fps <= 0:
            raise ValueError("inference.fps must be positive and finite")
        if policy.n_execute <= 0:
            raise ValueError("inference.n_execute must be positive")
        if not 1 <= policy.jpeg_quality <= 100:
            raise ValueError("inference.jpeg_quality must be in [1, 100]")
        if min(policy.recv_timeout_ms, policy.send_timeout_ms, policy.max_retries) <= 0:
            raise ValueError("inference timeouts and max_retries must be positive")
        if policy.state_dim != 14 or policy.action_dim != 14:
            raise ValueError(
                "dual Piper inference requires state_dim=14 and action_dim=14 " "(left 7-D followed by right 7-D)"
            )
        if policy.action_space not in {"cartesian", "joint_position"}:
            raise ValueError("inference.action_space must be cartesian or joint_position")

        order = policy.camera_order
        if not order:
            raise ValueError("inference.camera_order must contain at least one camera")
        if len(set(order)) != len(order):
            raise ValueError("inference.camera_order contains duplicate camera names")
        missing = sorted(set(order) - set(self.cameras))
        unexpected = sorted(set(self.cameras) - set(order))
        if missing or unexpected:
            raise ValueError(
                "inference.camera_order must exactly match cameras; " f"missing={missing}, unexpected={unexpected}"
            )
        for name, camera in self.cameras.items():
            _validate_camera(name, camera)

        for name in ("home_tolerance_deg", "home_timeout_s"):
            value = getattr(policy, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"inference.{name} must be positive and finite")
        if policy.log_every_chunks < 0:
            raise ValueError("inference.log_every_chunks must be non-negative")


def inference_recording_config(config: FlexivInferenceConfig | DualPiperInferenceConfig) -> RecordingConfig:
    """Use the policy cadence, task and already-open RGB cameras for recording."""
    return replace(
        config.recording,
        fps=int(config.inference.fps),
        task=config.inference.prompt,
        cameras={
            name.removeprefix("observation.images."): replace(camera, color_mode="rgb")
            for name, camera in config.cameras.items()
        },
    )


def _validate_interaction(config: FlexivInferenceConfig | DualPiperInferenceConfig) -> None:
    if config.controls not in {"keyboard", "vr"}:
        raise ValueError("controls must be keyboard or vr")
    if config.controls == "vr":
        vr = config.vr
        if bool(vr.tls_cert_path) != bool(vr.tls_key_path):
            raise ValueError("both vr.tls_cert_path and vr.tls_key_path are required")
        if any(not 1 <= port <= 65535 for port in (vr.https_port, vr.websocket_port)):
            raise ValueError("VR ports must be in [1, 65535]")
        if vr.https_port == vr.websocket_port:
            raise ValueError("VR HTTP and WebSocket ports must differ")
    if config.recording.enabled:
        fps = config.inference.fps
        if not math.isfinite(fps) or fps <= 0 or not float(fps).is_integer():
            raise ValueError("LeRobot recording requires a positive integer inference.fps")
        if config.recording.cameras:
            raise ValueError("configure inference cameras at the top level, not recording.cameras")
        if any(not name.startswith("observation.images.") or name == "observation.images." for name in config.cameras):
            raise ValueError("recorded camera names must use observation.images.<name>")
        # Reuse the recorder's existing validation, including worker/timeout settings.
        recording = inference_recording_config(config)
        TeleoperationConfig(recording=replace(recording, allow_preview=True)).validate()


def load_flexiv_inference_config(
    path: str | Path, *, inference_overrides: Mapping[str, Any] | None = None
) -> FlexivInferenceConfig:
    with Path(path).open() as stream:
        config = draccus.load(FlexivInferenceConfig, stream)
    if inference_overrides:
        config = replace(config, inference=replace(config.inference, **inference_overrides))
        if "camera_order" in inference_overrides:
            # Explicit camera selection leaves unused devices disconnected.
            # Unknown names remain in camera_order so validation rejects them.
            config = replace(
                config,
                cameras={
                    name: camera for name, camera in config.cameras.items() if name in config.inference.camera_order
                },
            )
    config.validate()
    return config


def load_dual_piper_inference_config(
    path: str | Path,
    *,
    inference_overrides: Mapping[str, Any] | None = None,
    controls_override: str | None = None,
) -> DualPiperInferenceConfig:
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if data.get("robot_type") == "arx_x5" and "cameras" in data.get("recording", {}):
        # Unified ARX config: validate the shared hardware/recording section first.
        from deployment.teleoperation.config import load_config

        shared = load_config(path)
        data["runtime"] = {
            "hardware_access": shared.runtime.hardware_access,
            "motion_enabled": shared.runtime.motion_enabled,
        }
        data["cameras"] = {
            f"observation.images.{name}": camera for name, camera in data["recording"].pop("cameras").items()
        }
    config = draccus.decode(DualPiperInferenceConfig, data)
    if controls_override is not None:
        config = replace(config, controls=controls_override)
    if inference_overrides:
        config = replace(config, inference=replace(config.inference, **inference_overrides))
        if "camera_order" in inference_overrides:
            # Explicit camera selection leaves unused devices disconnected.
            # Unknown names remain in camera_order so validation rejects them.
            config = replace(
                config,
                cameras={
                    name: camera for name, camera in config.cameras.items() if name in config.inference.camera_order
                },
            )
    config.validate()
    return config


def validate_policy_metadata(
    policy: PolicyMetadataConfig,
    metadata: Mapping[str, Any],
) -> None:
    """Validate the model endpoint without importing camera or robot runtimes."""

    required = ("camera_order", "state_dim", "action_dim")
    missing = [name for name in required if name not in metadata]
    if policy.strict_server_metadata and missing:
        raise RuntimeError(f"policy server metadata is missing required fields: {missing}")

    if "camera_order" in metadata:
        actual_order = tuple(str(name) for name in metadata["camera_order"])
        if actual_order != policy.camera_order:
            raise RuntimeError(
                "policy camera order does not match the robot client: "
                f"server={actual_order}, client={policy.camera_order}"
            )
    expected_action_space = getattr(policy, "action_space", None)
    if expected_action_space != "auto":
        for name, expected in (("state_dim", policy.state_dim), ("action_dim", policy.action_dim)):
            if expected is not None and name in metadata and int(metadata[name]) != expected:
                raise RuntimeError(f"policy {name} mismatch: server={metadata[name]}, client={expected}")
    if (
        expected_action_space not in {None, "auto"}
        and "action_space" in metadata
        and metadata["action_space"] != expected_action_space
    ):
        raise RuntimeError(
            "policy action_space mismatch: " f"server={metadata['action_space']!r}, client={expected_action_space!r}"
        )
    if policy.stat_key is not None and metadata.get("stat_key") != policy.stat_key:
        raise RuntimeError(
            "policy stat_key mismatch: " f"server={metadata.get('stat_key')!r}, client={policy.stat_key!r}"
        )


def resolve_flexiv_policy_metadata(
    policy: PolicyClientConfig,
    metadata: Mapping[str, Any],
) -> PolicyClientConfig:
    """Resolve the Flexiv control schema from server metadata before hardware access."""
    validate_policy_metadata(policy, metadata)
    missing = [name for name in ("state_dim", "action_dim", "action_space") if name not in metadata]
    if missing:
        raise RuntimeError(f"Flexiv policy metadata is missing required fields: {missing}")
    action_space = metadata.get("action_space")
    if action_space not in FLEXIV_POLICY_DIMS:
        raise RuntimeError(
            f"Flexiv policy metadata must provide action_space as cartesian or joint_position, got {action_space!r}"
        )

    state_dim = int(metadata["state_dim"])
    action_dim = int(metadata["action_dim"])
    expected_dims = FLEXIV_POLICY_DIMS[action_space]
    if (state_dim, action_dim) != expected_dims:
        raise RuntimeError(
            f"Flexiv policy {action_space} metadata requires dimensions {expected_dims}, got {(state_dim, action_dim)}"
        )

    stat_key = metadata.get("stat_key", policy.stat_key)
    return replace(
        policy,
        stat_key=None if stat_key is None else str(stat_key),
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=str(action_space),
    )


def _validate_server_address(value: str) -> None:
    address = value.removeprefix("tcp://")
    host, separator, port_text = address.rpartition(":")
    if not separator or not host or not port_text.isdigit():
        raise ValueError("inference.server must be HOST:PORT or tcp://HOST:PORT, " f"got {value!r}")
    port = int(port_text)
    if not 1 <= port <= 65_535:
        raise ValueError("inference.server port must be in [1, 65535]")


def _validate_camera(name: str, camera: CameraConfig) -> None:
    if not name.strip():
        raise ValueError("camera names must not be empty")
    if camera.type not in {"opencv", "intelrealsense"}:
        raise ValueError(f"cameras.{name}.type must be opencv or intelrealsense")
    if min(camera.width, camera.height, camera.fps) <= 0:
        raise ValueError(f"cameras.{name} width, height and fps must be positive")
    if camera.color_mode not in {"rgb", "bgr"}:
        raise ValueError(f"cameras.{name}.color_mode must be rgb or bgr")
    if camera.fourcc is not None and (len(camera.fourcc) != 4 or not camera.fourcc.isascii()):
        raise ValueError(f"cameras.{name}.fourcc must be four ASCII characters")
    if camera.type == "opencv" and camera.index_or_path is None:
        raise ValueError(f"cameras.{name}.index_or_path is required for opencv")
    if camera.type == "intelrealsense" and not camera.serial_number_or_name:
        raise ValueError(f"cameras.{name}.serial_number_or_name is required for intelrealsense")
