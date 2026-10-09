from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArmConfig, TeleoperationConfig
from .contracts import SIDES
from .recording.lerobot_recorder import EEF_NAMES, JOINT_NAMES
from .robot import create_robot

logger = logging.getLogger(__name__)
MASK_FEATURES = (
    "observation.active_mask",
    "observation.hardware_write_mask",
    "observation.joint_target_valid",
)


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayFrame:
    frame_index: int
    timestamp_s: float
    joint_targets_rad: dict[str, tuple[float, float, float, float, float, float]]
    gripper_targets: dict[str, float]


@dataclass(frozen=True)
class ReplayEpisode:
    root: Path
    repo_id: str
    episode_index: int
    fps: int
    robot_type: str
    execution_mode: str | None
    hardware_executed: bool
    joint_target_mode: str
    joint_target_hold_semantics: str | None
    tasks: tuple[str, ...]
    camera_keys: tuple[str, ...]
    images_validated: bool
    frames: tuple[ReplayFrame, ...]
    command_counts: dict[str, int]
    unreplayable_active_counts: dict[str, int]
    max_joint_step_deg: dict[str, float]

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps

    def summary(self, *, executed: bool = False) -> dict[str, object]:
        return {
            "root": str(self.root),
            "repo_id": self.repo_id,
            "episode": self.episode_index,
            "frames": len(self.frames),
            "fps": self.fps,
            "duration_s": self.duration_s,
            "robot_type": self.robot_type,
            "execution_mode": self.execution_mode,
            "hardware_executed": self.hardware_executed,
            "joint_target_mode": self.joint_target_mode,
            "joint_target_hold_semantics": self.joint_target_hold_semantics,
            "tasks": list(self.tasks),
            "camera_keys": list(self.camera_keys),
            "images_validated": self.images_validated,
            "command_counts": dict(self.command_counts),
            "unreplayable_active_counts": dict(self.unreplayable_active_counts),
            "max_joint_step_deg": dict(self.max_joint_step_deg),
            "executed": executed,
        }

    def validate_for_execution(
        self,
        arms: dict[str, ArmConfig],
        *,
        start_tolerance_deg: float,
        max_joint_step_deg: float,
        robot_type: str = "piper",
    ) -> None:
        expected = {"piper": "dual_piper_vr", "arx_x5": "dual_arx_x5_vr"}.get(robot_type)
        if expected is None or self.robot_type != expected:
            raise ReplayError(f"cannot replay robot_type={self.robot_type!r}; expected {expected!r}")
        if self.execution_mode != "hardware" or not self.hardware_executed:
            raise ReplayError("refusing hardware replay because the dataset was not recorded in hardware mode")
        if self.joint_target_mode == "held" and not self.joint_target_hold_semantics:
            raise ReplayError("dataset does not declare held per-arm joint target replay semantics")
        if self.joint_target_mode not in {"held", "masked"}:
            raise ReplayError(f"unsupported joint target mode: {self.joint_target_mode!r}")
        if not math.isfinite(start_tolerance_deg) or start_tolerance_deg <= 0:
            raise ValueError("start_tolerance_deg must be positive and finite")
        if not math.isfinite(max_joint_step_deg) or max_joint_step_deg <= 0:
            raise ValueError("max_joint_step_deg must be positive and finite")

        if not self.frames:
            raise ReplayError("episode contains no joint target commands")
        if self.joint_target_mode == "held":
            for frame in self.frames:
                if set(frame.joint_targets_rad) != set(SIDES):
                    raise ReplayError(f"frame {frame.frame_index} does not contain held joint targets for both arms")
        else:
            if sum(self.command_counts.values()) == 0:
                raise ReplayError("episode contains no valid joint target commands")
            unreplayable = {side: count for side, count in self.unreplayable_active_counts.items() if count}
            if unreplayable:
                raise ReplayError(
                    "episode contains active Cartesian actions without replayable joint targets: " f"{unreplayable}"
                )

        excessive_steps = {side: value for side, value in self.max_joint_step_deg.items() if value > max_joint_step_deg}
        if excessive_steps:
            raise ReplayError(f"recorded joint step exceeds {max_joint_step_deg:.3f}deg: {excessive_steps}")

        first_targets = self._first_joint_targets()
        start_errors = {}
        for side, target in first_targets.items():
            if side not in arms:
                raise ReplayError(f"dataset contains an unknown arm: {side}")
            home = arms[side].home_joints_deg
            error = max(abs(math.degrees(actual) - expected) for actual, expected in zip(target, home, strict=True))
            if error > start_tolerance_deg:
                start_errors[side] = error
        if start_errors:
            raise ReplayError(
                "first recorded joint target is not close to configured home "
                f"(tolerance={start_tolerance_deg:.3f}deg, errors={start_errors})"
            )

    def _first_joint_targets(
        self,
    ) -> dict[str, tuple[float, float, float, float, float, float]]:
        first_targets = {}
        for frame in self.frames:
            for side, target in frame.joint_targets_rad.items():
                first_targets.setdefault(side, target)
        return first_targets


def load_replay_episode(
    *,
    root: str | Path,
    repo_id: str,
    episode_index: int,
    validate_images: bool = False,
) -> ReplayEpisode:
    root = Path(root)
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if not (root / "meta" / "info.json").is_file():
        raise ReplayError(f"not a local LeRobot dataset: {root}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=[episode_index],
            download_videos=False,
        )
    except BaseException as exc:
        raise ReplayError(f"failed to load episode {episode_index} from LeRobot dataset {root}: {exc}") from exc

    if "action.joint" in dataset.features:
        vector_features = {
            "state.eef": EEF_NAMES,
            "action.eef": EEF_NAMES,
            "state.joint": JOINT_NAMES,
            "action.joint": JOINT_NAMES,
        }
        joint_action_key = "action.joint"
    else:
        vector_features = {
            "observation.state": EEF_NAMES,
            "action": EEF_NAMES,
            "action.joint_target": JOINT_NAMES,
        }
        joint_action_key = "action.joint_target"
    for key, names in vector_features.items():
        _validate_feature(dataset, key, shape=(14,), names=names)
    present_masks = set(MASK_FEATURES) & set(dataset.features)
    if present_masks and present_masks != set(MASK_FEATURES):
        raise ReplayError("dataset contains only part of the legacy replay masks: " f"present={sorted(present_masks)}")
    joint_target_mode = "masked" if present_masks else "held"
    if joint_target_mode == "masked":
        _validate_feature(
            dataset,
            "observation.active_mask",
            shape=(2,),
            names=SIDES,
        )
        _validate_feature(
            dataset,
            "observation.hardware_write_mask",
            shape=(2,),
            names=SIDES,
        )
        _validate_feature(
            dataset,
            "observation.joint_target_valid",
            shape=(2,),
            names=SIDES,
        )

    columns = [
        *vector_features,
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    ]
    if joint_target_mode == "masked":
        columns.extend(MASK_FEATURES)
    missing_columns = sorted(set(columns) - set(dataset.hf_dataset.column_names))
    if missing_columns:
        raise ReplayError(f"episode is missing required columns: {missing_columns}")
    records = dataset.hf_dataset.select_columns(columns)
    if len(records) == 0:
        raise ReplayError(f"episode {episode_index} contains no frames")

    max_steps = {side: 0.0 for side in SIDES}
    command_counts = {side: 0 for side in SIDES}
    unreplayable_counts = {side: 0 for side in SIDES}
    previous_targets: dict[str, tuple[float, float, float, float, float, float]] = {}
    tasks = []
    frames = []

    for local_index in range(len(records)):
        record = records[local_index]
        frame_index = _as_int_scalar(record["frame_index"], "frame_index")
        if frame_index != local_index:
            raise ReplayError(f"episode frame_index is not contiguous: expected {local_index}, got {frame_index}")
        recorded_episode = _as_int_scalar(record["episode_index"], "episode_index")
        if recorded_episode != episode_index:
            raise ReplayError(f"episode_index mismatch: expected {episode_index}, got {recorded_episode}")
        timestamp_s = _as_float_scalar(record["timestamp"], "timestamp")
        if not math.isfinite(timestamp_s):
            raise ReplayError(f"frame {frame_index} has a non-finite timestamp")

        task_index = _as_int_scalar(record["task_index"], "task_index")
        try:
            task = str(dataset.meta.tasks[task_index])
        except KeyError as exc:
            raise ReplayError(f"frame {frame_index} refers to missing task_index={task_index}") from exc
        if task not in tasks:
            tasks.append(task)

        vectors = {key: _as_vector(record[key], key, 14) for key in vector_features}
        joint_action = vectors[joint_action_key]
        if joint_target_mode == "masked":
            active_mask = _as_binary_mask(
                record["observation.active_mask"],
                "observation.active_mask",
                frame_index,
            )
            write_mask = _as_binary_mask(
                record["observation.hardware_write_mask"],
                "observation.hardware_write_mask",
                frame_index,
            )
            valid_mask = _as_binary_mask(
                record["observation.joint_target_valid"],
                "observation.joint_target_valid",
                frame_index,
            )
        else:
            active_mask = write_mask = valid_mask = (True, True)

        joint_targets = {}
        gripper_targets = {}
        for side_index, side in enumerate(SIDES):
            if joint_target_mode == "masked":
                if active_mask[side_index] and not valid_mask[side_index]:
                    unreplayable_counts[side] += 1
                if valid_mask[side_index] and not active_mask[side_index]:
                    raise ReplayError(f"frame {frame_index} side={side} has a joint target but is not active")
                if valid_mask[side_index] and not write_mask[side_index]:
                    raise ReplayError(f"frame {frame_index} side={side} has a joint target without a hardware write")
                if not valid_mask[side_index]:
                    continue

            start = side_index * 7
            joints = tuple(float(value) for value in joint_action[start : start + 6])
            gripper = float(joint_action[start + 6])
            if not 0.0 <= gripper <= 1.0:
                raise ReplayError(f"frame {frame_index} side={side} has gripper target outside [0, 1]: " f"{gripper}")
            joint_targets[side] = joints
            gripper_targets[side] = gripper
            command_counts[side] += 1

            previous = previous_targets.get(side)
            if previous is not None:
                step = max(math.degrees(abs(current - old)) for current, old in zip(joints, previous, strict=True))
                max_steps[side] = max(max_steps[side], step)
            previous_targets[side] = joints

        frames.append(
            ReplayFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                joint_targets_rad=joint_targets,
                gripper_targets=gripper_targets,
            )
        )

    camera_keys = tuple(dataset.meta.camera_keys)
    if validate_images:
        _validate_episode_images(dataset, camera_keys)

    teleoperation = dataset.meta.info.get("teleoperation", {})
    return ReplayEpisode(
        root=root,
        repo_id=repo_id,
        episode_index=episode_index,
        fps=int(dataset.fps),
        robot_type=str(dataset.meta.robot_type),
        execution_mode=teleoperation.get("execution_mode"),
        hardware_executed=teleoperation.get("hardware_executed") is True,
        joint_target_mode=joint_target_mode,
        joint_target_hold_semantics=teleoperation.get("joint_target_hold_semantics"),
        tasks=tuple(tasks),
        camera_keys=camera_keys,
        images_validated=validate_images,
        frames=tuple(frames),
        command_counts=command_counts,
        unreplayable_active_counts=unreplayable_counts,
        max_joint_step_deg=max_steps,
    )


class PiperReplayRuntime:
    def __init__(
        self,
        config: TeleoperationConfig,
        episode: ReplayEpisode,
        *,
        speed: float = 1.0,
        robot: Any | None = None,
    ) -> None:
        if not math.isfinite(speed) or not 0 < speed <= 1.0:
            raise ValueError("replay speed must be finite and in (0, 1]")
        self.config = config
        self.episode = episode
        self.speed = speed
        expected = {"piper": "dual_piper_vr", "arx_x5": "dual_arx_x5_vr"}.get(config.robot_type)
        if expected is None or episode.robot_type != expected:
            raise ReplayError("replay dataset robot type does not match hardware configuration")
        self.robot = robot if robot is not None else create_robot(config)
        self._opened = False
        self._closed = False

    def validate_joint_limits(self) -> None:
        if self.config.robot_type == "arx_x5":
            self.robot.start_kinematics()
            for frame in self.episode.frames:
                for side, joints in frame.joint_targets_rad.items():
                    try:
                        self.robot._validate_joints(side, joints)
                    except ValueError as exc:
                        raise ReplayError(f"frame {frame.frame_index}: {exc}") from exc

    def open(self) -> None:
        if self._opened:
            return
        if not self.config.runtime.hardware_access or not self.config.runtime.motion_enabled:
            raise ReplayError("hardware replay requires hardware_access=true and motion_enabled=true")
        self.validate_joint_limits()
        self.robot.open_read_only()
        self._opened = True
        self.robot.prepare_motion()
        logger.info(
            "Replay startup; moving both arms to configured home joints " "(tolerance=%.3fdeg timeout=%.1fs)",
            self.config.recording.home_tolerance_deg,
            self.config.recording.home_timeout_s,
        )
        self.robot.go_home(
            tolerance_deg=self.config.recording.home_tolerance_deg,
            timeout_s=self.config.recording.home_timeout_s,
        )
        logger.info("Both arms reached configured home joints")

    def run(self) -> int:
        if not self._opened or self._closed:
            raise RuntimeError("replay runtime is not open")
        period_s = 1.0 / (self.episode.fps * self.speed)
        started = time.perf_counter()
        total = len(self.episode.frames)
        for offset, frame in enumerate(self.episode.frames):
            self.robot.apply_joint_targets(
                frame.joint_targets_rad,
                frame.gripper_targets,
            )
            if self.config.runtime.log_every and (
                (offset + 1) % self.config.runtime.log_every == 0 or offset + 1 == total
            ):
                logger.info(
                    "Replay episode=%d frame=%d/%d commanded_arms=%s",
                    self.episode.episode_index,
                    offset + 1,
                    total,
                    sorted(frame.joint_targets_rad),
                )
            deadline = started + (offset + 1) * period_s
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        return total

    def close(self, *, return_home: bool = False) -> None:
        if self._closed:
            return
        try:
            if return_home and self._opened:
                logger.info("Replay complete; returning both arms to configured home joints")
                self.robot.go_home(
                    tolerance_deg=self.config.recording.home_tolerance_deg,
                    timeout_s=self.config.recording.home_timeout_s,
                )
                logger.info("Both arms reached configured home joints")
        finally:
            self.robot.close()
            self._closed = True


def _validate_feature(dataset: Any, name: str, *, shape: tuple[int, ...], names: tuple[str, ...]) -> None:
    actual = dataset.features.get(name)
    if actual is None:
        raise ReplayError(f"dataset is missing required feature {name!r}")
    if tuple(actual.get("shape", ())) != shape or tuple(actual.get("names") or ()) != names:
        raise ReplayError(
            f"incompatible feature {name!r}: expected shape={shape}, names={list(names)}, " f"got {actual}"
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_vector(value: Any, name: str, size: int) -> np.ndarray:
    array = _as_numpy(value).astype(np.float64, copy=False).reshape(-1)
    if array.shape != (size,):
        raise ReplayError(f"{name} must contain {size} values, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ReplayError(f"{name} contains non-finite values")
    return array


def _as_binary_mask(value: Any, name: str, frame_index: int) -> tuple[bool, bool]:
    array = _as_vector(value, name, len(SIDES))
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-6) or not np.all((rounded == 0) | (rounded == 1)):
        raise ReplayError(f"frame {frame_index} has non-binary {name}: {array.tolist()}")
    return tuple(bool(item) for item in rounded)


def _as_int_scalar(value: Any, name: str) -> int:
    array = _as_numpy(value).reshape(-1)
    if array.size != 1:
        raise ReplayError(f"{name} must be a scalar")
    result = float(array[0])
    if not math.isfinite(result) or result != round(result):
        raise ReplayError(f"{name} must be a finite integer, got {result}")
    return int(result)


def _as_float_scalar(value: Any, name: str) -> float:
    array = _as_numpy(value).reshape(-1)
    if array.size != 1:
        raise ReplayError(f"{name} must be a scalar")
    try:
        return float(array[0])
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"{name} must be numeric") from exc


def _validate_episode_images(dataset: Any, camera_keys: tuple[str, ...]) -> None:
    for frame_index in range(dataset.num_frames):
        try:
            item = dataset[frame_index]
        except BaseException as exc:
            raise ReplayError(f"failed to decode images for frame {frame_index}: {exc}") from exc
        for key in camera_keys:
            if key not in item:
                raise ReplayError(f"decoded frame {frame_index} is missing camera {key!r}")
            image = _as_numpy(item[key])
            expected = tuple(dataset.features[key]["shape"])
            channel_first = (expected[2], expected[0], expected[1])
            if image.shape not in {expected, channel_first}:
                raise ReplayError(
                    f"camera {key!r} frame {frame_index} has shape={image.shape}; "
                    f"expected {expected} or {channel_first}"
                )
            if np.issubdtype(image.dtype, np.floating) and not np.all(np.isfinite(image)):
                raise ReplayError(f"camera {key!r} frame {frame_index} contains non-finite pixels")
