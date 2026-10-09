from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FlexivRobotConfig, FlexivSpaceMouseConfig
from .recorder import JOINT_NAMES
from .robot import FlexivRobot

logger = logging.getLogger(__name__)


class FlexivReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class FlexivReplayFrame:
    frame_index: int
    timestamp_s: float
    joint_target_rad: tuple[float, float, float, float, float, float, float]
    gripper_target_m: float


@dataclass(frozen=True)
class FlexivReplayEpisode:
    root: Path
    repo_id: str
    episode_index: int
    fps: int
    robot_type: str
    execution_mode: str | None
    hardware_executed: bool
    joint_action_semantics: str | None
    tasks: tuple[str, ...]
    camera_keys: tuple[str, ...]
    images_validated: bool
    incomplete_marker_present: bool
    frames: tuple[FlexivReplayFrame, ...]
    max_joint_step_deg: float
    max_gripper_step_m: float
    max_timestamp_error_s: float

    @property
    def duration_s(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp_s - self.frames[0].timestamp_s

    def summary(self, *, executed: bool = False) -> dict[str, object]:
        first = self.frames[0] if self.frames else None
        last = self.frames[-1] if self.frames else None
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
            "joint_action_semantics": self.joint_action_semantics,
            "tasks": list(self.tasks),
            "camera_keys": list(self.camera_keys),
            "images_validated": self.images_validated,
            "incomplete_marker_present": self.incomplete_marker_present,
            "max_joint_step_deg": self.max_joint_step_deg,
            "max_gripper_step_m": self.max_gripper_step_m,
            "max_timestamp_error_s": self.max_timestamp_error_s,
            "first_gripper_target_m": (
                None if first is None else first.gripper_target_m
            ),
            "last_gripper_target_m": None if last is None else last.gripper_target_m,
            "executed": executed,
        }

    def validate_for_execution(
        self,
        robot: FlexivRobotConfig,
        *,
        start_tolerance_deg: float,
        max_joint_step_deg: float,
    ) -> None:
        if self.robot_type != "flexiv_spacemouse":
            raise FlexivReplayError(
                f"cannot replay robot_type={self.robot_type!r}; expected 'flexiv_spacemouse'"
            )
        if self.execution_mode != "hardware" or not self.hardware_executed:
            raise FlexivReplayError(
                "refusing hardware replay because the dataset was not recorded in hardware mode"
            )
        if not self.joint_action_semantics or "next recorded joint observation" not in (
            self.joint_action_semantics
        ):
            raise FlexivReplayError(
                "dataset does not declare the expected one-sample-shifted joint action semantics"
            )
        if not math.isfinite(start_tolerance_deg) or start_tolerance_deg <= 0:
            raise ValueError("start_tolerance_deg must be positive and finite")
        if not math.isfinite(max_joint_step_deg) or max_joint_step_deg <= 0:
            raise ValueError("max_joint_step_deg must be positive and finite")
        if not self.frames:
            raise FlexivReplayError("episode contains no joint targets")
        if self.max_joint_step_deg > max_joint_step_deg:
            raise FlexivReplayError(
                f"recorded joint step {self.max_joint_step_deg:.3f}deg exceeds "
                f"{max_joint_step_deg:.3f}deg"
            )

        first_target = self.frames[0].joint_target_rad
        start_error = max(
            abs(math.degrees(actual) - expected)
            for actual, expected in zip(first_target, robot.home_joints_deg, strict=True)
        )
        if start_error > start_tolerance_deg:
            raise FlexivReplayError(
                "first recorded joint target is not close to configured home "
                f"(tolerance={start_tolerance_deg:.3f}deg, error={start_error:.3f}deg)"
            )

        gripper_targets = [frame.gripper_target_m for frame in self.frames]
        if robot.gripper.enabled:
            lower = robot.gripper.close_width_m
            upper = robot.gripper.open_width_m
            violations = [
                (frame.frame_index, frame.gripper_target_m)
                for frame in self.frames
                if not lower - 1e-6 <= frame.gripper_target_m <= upper + 1e-6
            ]
            if violations:
                frame_index, target = violations[0]
                raise FlexivReplayError(
                    f"frame {frame_index} gripper target {target:.6f}m is outside configured "
                    f"range [{lower:.6f}, {upper:.6f}]m"
                )
        elif max(gripper_targets) - min(gripper_targets) > 1e-4:
            raise FlexivReplayError(
                "episode contains gripper motion but the configured Flexiv gripper is disabled"
            )


def load_flexiv_replay_episode(
    *,
    root: str | Path,
    repo_id: str,
    episode_index: int,
    validate_images: bool = False,
) -> FlexivReplayEpisode:
    root = Path(root)
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FlexivReplayError(f"not a local LeRobot dataset: {root}")

    info = _read_json(info_path)
    fps = _positive_int(info.get("fps"), "meta/info.json fps")
    chunks_size = _positive_int(info.get("chunks_size"), "meta/info.json chunks_size")
    total_episodes = _positive_int(
        info.get("total_episodes"), "meta/info.json total_episodes"
    )
    if episode_index >= total_episodes:
        raise FlexivReplayError(
            f"episode {episode_index} is outside dataset range [0, {total_episodes - 1}]"
        )

    features = info.get("features")
    if not isinstance(features, dict):
        raise FlexivReplayError("meta/info.json features must be an object")
    for name in ("action", "actions.joint_position", "observation.state"):
        _validate_joint_feature(features, name)

    data_template = info.get("data_path")
    if not isinstance(data_template, str) or not data_template:
        raise FlexivReplayError("meta/info.json data_path must be a non-empty string")
    data_path = root / _format_episode_path(
        data_template,
        episode_index=episode_index,
        chunks_size=chunks_size,
    )
    if not data_path.is_file():
        raise FlexivReplayError(f"episode parquet is missing: {data_path}")

    episode_metadata = _find_jsonl_record(
        root / "meta" / "episodes.jsonl", "episode_index", episode_index
    )
    task_records = _read_jsonl(root / "meta" / "tasks.jsonl")
    tasks_by_index = {
        _as_int(record.get("task_index"), "task_index"): str(record.get("task", ""))
        for record in task_records
    }

    try:
        import pyarrow.parquet as parquet

        table = parquet.read_table(data_path)
    except Exception as exc:
        raise FlexivReplayError(f"failed to read episode parquet {data_path}: {exc}") from exc

    required_columns = {
        "action",
        "actions.joint_position",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    }
    missing = sorted(required_columns - set(table.column_names))
    if missing:
        raise FlexivReplayError(f"episode is missing required columns: {missing}")
    frame_count = table.num_rows
    if frame_count <= 0:
        raise FlexivReplayError(f"episode {episode_index} contains no frames")
    declared_length = _positive_int(episode_metadata.get("length"), "episode length")
    if frame_count != declared_length:
        raise FlexivReplayError(
            f"episode length mismatch: metadata={declared_length}, parquet={frame_count}"
        )

    action = _matrix_column(table, "action", frame_count, 8)
    joint_action = _matrix_column(table, "actions.joint_position", frame_count, 8)
    _matrix_column(table, "observation.state", frame_count, 8)
    if not np.allclose(action, joint_action, rtol=0.0, atol=1e-7):
        difference = float(np.max(np.abs(action - joint_action)))
        raise FlexivReplayError(
            "action and actions.joint_position aliases disagree "
            f"(maximum absolute difference={difference})"
        )

    teleoperation = info.get("teleoperation", {})
    if not isinstance(teleoperation, dict):
        teleoperation = {}
    gripper_scale = float(teleoperation.get("gripper_scale", 1.0))
    action = action.copy()
    action[:, 7] /= gripper_scale

    timestamps = _float_column(table, "timestamp")
    frame_indices = _integer_column(table, "frame_index")
    episode_indices = _integer_column(table, "episode_index")
    task_indices = _integer_column(table, "task_index")
    expected_indices = np.arange(frame_count, dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        mismatch = int(np.flatnonzero(frame_indices != expected_indices)[0])
        raise FlexivReplayError(
            "episode frame_index is not contiguous: "
            f"expected {mismatch}, got {int(frame_indices[mismatch])}"
        )
    if np.any(episode_indices != episode_index):
        mismatch = int(np.flatnonzero(episode_indices != episode_index)[0])
        raise FlexivReplayError(
            f"frame {mismatch} episode_index mismatch: expected {episode_index}, "
            f"got {int(episode_indices[mismatch])}"
        )
    if not np.all(np.isfinite(timestamps)):
        raise FlexivReplayError("episode contains non-finite timestamps")
    timestamp_steps = np.diff(timestamps)
    if np.any(timestamp_steps <= 0):
        mismatch = int(np.flatnonzero(timestamp_steps <= 0)[0] + 1)
        raise FlexivReplayError(
            f"timestamps are not strictly increasing at frame {mismatch}"
        )
    expected_period_s = 1.0 / fps
    max_timestamp_error_s = (
        0.0
        if timestamp_steps.size == 0
        else float(np.max(np.abs(timestamp_steps - expected_period_s)))
    )
    if max_timestamp_error_s > max(1e-3, expected_period_s * 0.25):
        raise FlexivReplayError(
            "recorded timestamp cadence is incompatible with dataset fps: "
            f"fps={fps}, max_error={max_timestamp_error_s:.6f}s"
        )

    tasks = []
    for frame_index, task_index in enumerate(task_indices):
        task = tasks_by_index.get(int(task_index))
        if task is None:
            raise FlexivReplayError(
                f"frame {frame_index} refers to missing task_index={int(task_index)}"
            )
        if task not in tasks:
            tasks.append(task)
    declared_tasks = episode_metadata.get("tasks")
    if isinstance(declared_tasks, list) and set(map(str, declared_tasks)) != set(tasks):
        raise FlexivReplayError(
            f"episode task metadata disagrees with parquet task indices: {declared_tasks} vs {tasks}"
        )

    frames = tuple(
        FlexivReplayFrame(
            frame_index=frame_index,
            timestamp_s=float(timestamps[frame_index]),
            joint_target_rad=tuple(float(value) for value in action[frame_index, :7]),
            gripper_target_m=float(action[frame_index, 7]),
        )
        for frame_index in range(frame_count)
    )
    max_joint_step = (
        0.0
        if frame_count < 2
        else float(np.max(np.abs(np.degrees(np.diff(action[:, :7], axis=0)))))
    )
    max_gripper_step = (
        0.0
        if frame_count < 2
        else float(np.max(np.abs(np.diff(action[:, 7]))))
    )

    camera_keys = tuple(
        name
        for name, feature in features.items()
        if name.startswith("observation.images.")
        and isinstance(feature, dict)
        and feature.get("dtype") in {"video", "image"}
    )
    if validate_images:
        _validate_episode_videos(
            root=root,
            info=info,
            features=features,
            camera_keys=camera_keys,
            episode_index=episode_index,
            chunks_size=chunks_size,
            expected_frames=frame_count,
        )

    return FlexivReplayEpisode(
        root=root,
        repo_id=repo_id,
        episode_index=episode_index,
        fps=fps,
        robot_type=str(info.get("robot_type", "")),
        execution_mode=teleoperation.get("execution_mode"),
        hardware_executed=teleoperation.get("hardware_executed") is True,
        joint_action_semantics=teleoperation.get("joint_action_semantics"),
        tasks=tuple(tasks),
        camera_keys=camera_keys,
        images_validated=validate_images,
        incomplete_marker_present=(root / ".incomplete_episode.json").is_file(),
        frames=frames,
        max_joint_step_deg=max_joint_step,
        max_gripper_step_m=max_gripper_step,
        max_timestamp_error_s=max_timestamp_error_s,
    )


class FlexivReplayRuntime:
    def __init__(
        self,
        config: FlexivSpaceMouseConfig,
        episode: FlexivReplayEpisode,
        *,
        speed: float = 1.0,
        robot: Any | None = None,
    ) -> None:
        if not math.isfinite(speed) or not 0 < speed <= 1.0:
            raise ValueError("replay speed must be finite and in (0, 1]")
        self.config = config
        self.episode = episode
        self.speed = speed
        self.robot = robot if robot is not None else FlexivRobot(config.robot)
        self._opened = False
        self._closed = False

    def open(self) -> None:
        if self._opened:
            return
        self.robot.open(enable_motion=True)
        self._opened = True
        logger.info(
            "Replay startup; moving Flexiv to configured home joints "
            "(tolerance=%.3fdeg timeout=%.1fs)",
            self.config.recording.home_tolerance_deg,
            self.config.recording.home_timeout_s,
        )
        self.robot.go_home(
            tolerance_deg=self.config.recording.home_tolerance_deg,
            timeout_s=self.config.recording.home_timeout_s,
        )
        self.robot.prepare_joint_motion()
        logger.info("Flexiv reached configured home joints and is ready to replay")

    def run(self) -> int:
        if not self._opened or self._closed:
            raise RuntimeError("replay runtime is not open")
        started = time.perf_counter()
        first_timestamp = self.episode.frames[0].timestamp_s
        total = len(self.episode.frames)
        for offset, frame in enumerate(self.episode.frames):
            deadline = started + (frame.timestamp_s - first_timestamp) / self.speed
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            self.robot.apply_joint_target(
                frame.joint_target_rad,
                frame.gripper_target_m,
            )
            if self.config.runtime.log_every and (
                (offset + 1) % self.config.runtime.log_every == 0 or offset + 1 == total
            ):
                logger.info(
                    "Replay episode=%d frame=%d/%d gripper_target_m=%.5f",
                    self.episode.episode_index,
                    offset + 1,
                    total,
                    frame.gripper_target_m,
                )
        return total

    def close(self, *, return_home: bool = False) -> None:
        if self._closed:
            return
        try:
            if return_home and self._opened:
                logger.info("Replay complete; returning Flexiv to configured home joints")
                self.robot.go_home(
                    tolerance_deg=self.config.recording.home_tolerance_deg,
                    timeout_s=self.config.recording.home_timeout_s,
                )
                logger.info("Flexiv reached configured home joints")
        finally:
            self.robot.close()
            self._closed = True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FlexivReplayError(f"failed to read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FlexivReplayError(f"JSON metadata must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FlexivReplayError(f"dataset metadata is missing: {path}")
    records = []
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FlexivReplayError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise FlexivReplayError(f"failed to read JSONL metadata {path}: {exc}") from exc
    return records


def _find_jsonl_record(path: Path, key: str, expected: int) -> dict[str, Any]:
    for record in _read_jsonl(path):
        if _as_int(record.get(key), key) == expected:
            return record
    raise FlexivReplayError(f"{path} does not contain {key}={expected}")


def _positive_int(value: Any, name: str) -> int:
    result = _as_int(value, name)
    if result <= 0:
        raise FlexivReplayError(f"{name} must be positive")
    return result


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise FlexivReplayError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FlexivReplayError(f"{name} must be an integer") from exc
    if result != value:
        raise FlexivReplayError(f"{name} must be an integer")
    return result


def _validate_joint_feature(features: dict[str, Any], name: str) -> None:
    feature = features.get(name)
    if not isinstance(feature, dict):
        raise FlexivReplayError(f"dataset is missing required feature {name!r}")
    shape = tuple(feature.get("shape", ()))
    names = feature.get("names")
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if shape != (8,) or tuple(names or ()) != JOINT_NAMES:
        raise FlexivReplayError(
            f"incompatible feature {name!r}: expected shape=(8,), "
            f"names={list(JOINT_NAMES)}, got {feature}"
        )


def _format_episode_path(template: str, *, episode_index: int, chunks_size: int, **values: Any) -> str:
    try:
        return template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            **values,
        )
    except (KeyError, ValueError) as exc:
        raise FlexivReplayError(f"invalid dataset path template {template!r}: {exc}") from exc


def _matrix_column(table: Any, name: str, rows: int, columns: int) -> np.ndarray:
    try:
        array = np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FlexivReplayError(f"failed to decode column {name!r}: {exc}") from exc
    if array.shape != (rows, columns):
        raise FlexivReplayError(
            f"column {name!r} must have shape {(rows, columns)}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FlexivReplayError(f"column {name!r} contains non-finite values")
    return array


def _float_column(table: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FlexivReplayError(f"failed to decode column {name!r}: {exc}") from exc


def _integer_column(table: Any, name: str) -> np.ndarray:
    values = _float_column(table, name)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise FlexivReplayError(f"column {name!r} must contain finite integers")
    return values.astype(np.int64)


def _validate_episode_videos(
    *,
    root: Path,
    info: dict[str, Any],
    features: dict[str, Any],
    camera_keys: tuple[str, ...],
    episode_index: int,
    chunks_size: int,
    expected_frames: int,
) -> None:
    video_template = info.get("video_path")
    if camera_keys and (not isinstance(video_template, str) or not video_template):
        raise FlexivReplayError("meta/info.json video_path must be set for camera validation")
    try:
        import av
    except ImportError as exc:
        raise FlexivReplayError("image validation requires the PyAV package") from exc

    for camera_key in camera_keys:
        feature = features[camera_key]
        if feature.get("dtype") != "video":
            raise FlexivReplayError(
                f"--validate-images does not support non-video feature {camera_key!r}"
            )
        shape = tuple(feature.get("shape", ()))
        if len(shape) != 3:
            raise FlexivReplayError(f"camera feature {camera_key!r} has invalid shape {shape}")
        video_path = root / _format_episode_path(
            video_template,
            episode_index=episode_index,
            chunks_size=chunks_size,
            video_key=camera_key,
        )
        if not video_path.is_file():
            raise FlexivReplayError(f"episode video is missing: {video_path}")
        try:
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise FlexivReplayError(f"video contains no video stream: {video_path}")
                stream = container.streams.video[0]
                decoded = 0
                for frame in container.decode(stream):
                    if (frame.height, frame.width) != shape[:2]:
                        raise FlexivReplayError(
                            f"camera {camera_key!r} frame {decoded} has size "
                            f"{frame.width}x{frame.height}; expected {shape[1]}x{shape[0]}"
                        )
                    decoded += 1
        except FlexivReplayError:
            raise
        except Exception as exc:
            raise FlexivReplayError(f"failed to decode video {video_path}: {exc}") from exc
        if decoded != expected_frames:
            raise FlexivReplayError(
                f"camera {camera_key!r} frame count mismatch: "
                f"expected {expected_frames}, decoded {decoded}"
            )
