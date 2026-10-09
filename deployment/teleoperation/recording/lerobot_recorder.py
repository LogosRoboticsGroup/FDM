from __future__ import annotations

import inspect
import json
import logging
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import RecordingConfig
from ..contracts import SIDES, ArmObservation, CommandResult
from ..pose import wxyz_to_xyzw
from .base import Recorder, RecorderError, RecorderTimeoutError

EEF_NAMES = tuple(
    f"{side}.{name}" for side in SIDES for name in ("x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad", "gripper_open")
)
JOINT_NAMES = tuple(
    f"{side}.{name}" for side in SIDES for name in (*[f"joint{index}_rad" for index in range(1, 7)], "gripper_open")
)
STANDARD_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

logger = logging.getLogger(__name__)


def _remove_encoded_episode_images(dataset: Any, episode_index: int) -> None:
    """Remove LeRobot's temporary PNG tree after every MP4 is present."""
    video_keys = tuple(dataset.meta.video_keys)
    missing_videos = [
        dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        for key in video_keys
        if not (dataset.root / dataset.meta.get_video_file_path(episode_index, key)).is_file()
    ]
    if missing_videos:
        missing = ", ".join(str(path) for path in missing_videos)
        raise FileNotFoundError(
            "refusing to remove temporary camera images because encoded videos " f"are missing: {missing}"
        )

    images_root = dataset.root / "images"
    if images_root.exists():
        shutil.rmtree(images_root)
        logger.info(
            "Removed temporary LeRobot images after video encoding: episode=%d root=%s",
            episode_index,
            images_root,
        )


def _configure_parallel_video_encoding(dataset: Any, max_workers: int) -> None:
    video_keys = tuple(dataset.meta.video_keys)
    workers = min(max_workers, len(video_keys))
    if workers <= 1:
        return

    def encode_episode_videos(current_dataset: Any, episode_index: int) -> None:
        from lerobot.datasets.utils import write_info
        from lerobot.datasets.video_utils import encode_video_frames

        def encode_video_key(key: str) -> None:
            video_path = current_dataset.root / current_dataset.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                return
            image_dir = current_dataset._get_image_file_path(
                episode_index=episode_index,
                image_key=key,
                frame_index=0,
            ).parent
            frame_count = sum(1 for _ in image_dir.glob("frame_*.png"))
            started_at = time.perf_counter()
            logger.info(
                "Video encoding started: episode=%d camera=%s frames=%d",
                episode_index,
                key,
                frame_count,
            )
            # Passing log_level=None avoids changing PyAV's process-global log
            # callback concurrently from multiple camera encoder threads.
            encode_video_frames(
                image_dir,
                video_path,
                current_dataset.fps,
                log_level=None,
                overwrite=True,
            )
            shutil.rmtree(image_dir)
            logger.info(
                "Video encoding completed: episode=%d camera=%s frames=%d " "elapsed_s=%.3f size_mib=%.1f",
                episode_index,
                key,
                frame_count,
                time.perf_counter() - started_at,
                video_path.stat().st_size / (1024 * 1024),
            )

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="lerobot-video",
        ) as executor:
            list(executor.map(encode_video_key, video_keys))

        if video_keys and episode_index == 0:
            current_dataset.meta.update_video_info()
            write_info(current_dataset.meta.info, current_dataset.meta.root)

    dataset.encode_episode_videos = MethodType(encode_episode_videos, dataset)
    logger.info(
        "Enabled parallel LeRobot video encoding: cameras=%d workers=%d",
        len(video_keys),
        workers,
    )


def _observation_vector(observation: ArmObservation) -> np.ndarray:
    rotation = Rotation.from_quat(wxyz_to_xyzw(observation.quaternion_wxyz)).as_rotvec()
    return np.asarray(
        [
            *(value * 1000.0 for value in observation.position_m),
            *rotation,
            observation.gripper_open_fraction,
        ],
        dtype=np.float32,
    )


def _update_held_commands(
    result: CommandResult,
    joint_holds: dict[str, tuple[float, float, float, float, float, float]],
    gripper_holds: dict[str, float],
) -> None:
    for side in SIDES:
        applied = result.applied.get(side)
        if applied is not None and applied.joint_target_rad is not None:
            joint_holds[side] = applied.joint_target_rad
        if applied is not None and applied.gripper_written and applied.command.gripper_open_fraction is not None:
            gripper_holds[side] = applied.command.gripper_open_fraction
        gripper_holds.setdefault(side, result.observation[side].gripper_open_fraction)


def build_frame(
    result: CommandResult,
    images: dict[str, np.ndarray],
    *,
    last_joint_targets_rad: dict[str, tuple[float, float, float, float, float, float]] | None = None,
    last_gripper_targets: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    eef_states = {side: _observation_vector(result.observation[side]) for side in SIDES}
    eef_actions = []
    joint_states = []
    joint_targets = []
    joint_holds = {} if last_joint_targets_rad is None else last_joint_targets_rad
    gripper_holds = {} if last_gripper_targets is None else last_gripper_targets
    _update_held_commands(result, joint_holds, gripper_holds)
    for side in SIDES:
        observation = result.observation[side]
        if observation.joint_positions_rad is None:
            raise RecorderError(f"{side} joint feedback is required to record state.joint")
        joint_states.extend((*observation.joint_positions_rad, observation.gripper_open_fraction))
        applied = result.applied.get(side)
        held_joints = joint_holds.get(side, (0.0,) * 6)
        held_gripper = gripper_holds[side]
        joint_targets.extend((*held_joints, held_gripper))
        if applied is None:
            eef_actions.append(eef_states[side])
        else:
            command = applied.command
            eef_actions.append(
                np.asarray(
                    [
                        *command.position_mm,
                        *command.rotation_vector_rad,
                        (
                            command.gripper_open_fraction
                            if command.gripper_open_fraction is not None
                            else observation.gripper_open_fraction
                        ),
                    ],
                    dtype=np.float32,
                )
            )
    frame = {
        "state.joint": np.asarray(joint_states, dtype=np.float32),
        "action.joint": np.asarray(joint_targets, dtype=np.float32),
        "state.eef": np.concatenate([eef_states[side] for side in SIDES]),
        "action.eef": np.concatenate(eef_actions),
    }
    frame.update({f"observation.images.{name}": image for name, image in images.items()})
    return frame


@dataclass
class _Request:
    kind: str
    payload: Any = None
    done: threading.Event | None = None


class LeRobotRecorder(Recorder):
    """Owns LeRobotDataset on a writer thread so the control loop never writes files."""

    ROBOT_TYPE = "dual_piper_vr"

    def __init__(
        self,
        config: RecordingConfig,
        execution_mode: str,
        home_joint_targets_rad: dict[str, tuple[float, float, float, float, float, float]] | None = None,
        robot_type: str | None = None,
    ) -> None:
        if not config.enabled or config.root is None:
            raise ValueError("LeRobotRecorder requires enabled recording and a root")
        if robot_type is not None:
            self.ROBOT_TYPE = robot_type
        self.config = config
        self.root = Path(config.root)
        self.execution_mode = execution_mode
        self.home_joint_targets_rad = dict(home_joint_targets_rad or {})
        self._last_joint_targets_rad = dict(self.home_joint_targets_rad)
        self._last_gripper_targets: dict[str, float] = {}
        self._queue: queue.Queue[_Request] = queue.Queue(maxsize=config.async_queue_size)
        self._error: BaseException | None = None
        self._active = False
        self._frames = 0
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="lerobot-recorder", daemon=True)
        self._thread.start()
        try:
            self._request("init", wait=True)
        except BaseException:
            try:
                self._request(
                    "close",
                    wait=True,
                    allow_failed=True,
                    timeout_s=self.config.close_timeout_s,
                )
            except RecorderError:
                pass
            self._thread.join(timeout=self.config.close_timeout_s)
            raise

    def start_episode(self) -> None:
        self._raise_if_failed()
        if self._active:
            raise RecorderError("an episode is already active")
        self._active = True
        self._frames = 0
        self._last_joint_targets_rad = dict(self.home_joint_targets_rad)
        self._last_gripper_targets = {}
        logger.info("Recording episode started: root=%s task=%r", self.root, self.config.task)

    def observe_commands(self, result: CommandResult) -> None:
        _update_held_commands(result, self._last_joint_targets_rad, self._last_gripper_targets)

    def add_frame(self, result: CommandResult, images: dict[str, np.ndarray]) -> None:
        self._raise_if_failed()
        if not self._active:
            return
        frame = build_frame(
            result,
            images,
            last_joint_targets_rad=self._last_joint_targets_rad,
            last_gripper_targets=self._last_gripper_targets,
        )
        self._enqueue_frame(frame)

    def _enqueue_frame(self, frame: dict[str, np.ndarray]) -> None:
        try:
            self._queue.put_nowait(_Request("frame", frame))
        except queue.Full as exc:
            raise RecorderError("recording queue is full") from exc
        self._frames += 1

    def finish_episode(self, save: bool = True) -> None:
        if not self._active:
            return
        kind = "save" if save and self._frames else "discard"
        frames = self._frames
        started_at = time.perf_counter()
        logger.info(
            "Finalizing recording episode: action=%s frames=%d root=%s",
            kind,
            frames,
            self.root,
        )
        if not frames:
            # LeRobot creates episode_buffer lazily on the first add_frame(). There
            # is nothing to clear for an episode stopped before its first frame.
            self._active = False
            self._frames = 0
            logger.info(
                "Recording episode discarded: frames=0 elapsed_s=%.3f root=%s",
                time.perf_counter() - started_at,
                self.root,
            )
            return
        try:
            # Saving an episode may include MP4 encoding. Wait until LeRobot has
            # committed the episode before accepting frames for the next one;
            # otherwise a short bounded frame queue can overflow behind the
            # encoder and stop recording.
            self._request(
                kind,
                wait=True,
                timeout_s=self.config.episode_finalize_timeout_s,
                progress_interval_s=30.0,
            )
        finally:
            self._active = False
            self._frames = 0
        if kind == "save" or frames:
            self._clear_incomplete_marker()
        logger.info(
            "Recording episode %s: frames=%d elapsed_s=%.3f root=%s",
            "saved" if kind == "save" else "discarded",
            frames,
            time.perf_counter() - started_at,
            self.root,
        )

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        if self._active:
            discarded_frames = self._frames
            try:
                self.finish_episode(save=False)
            except RecorderError:
                self._active = False
            if discarded_frames:
                self._mark_incomplete(discarded_frames)

        deadline = time.monotonic() + self.config.close_timeout_s
        close_error: RecorderError | None = None
        try:
            self._request(
                "close",
                wait=True,
                allow_failed=True,
                timeout_s=self.config.close_timeout_s,
            )
        except RecorderError as exc:
            close_error = exc
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise RecorderError("recording worker did not stop") from close_error
        self._closed = True
        if close_error is not None:
            raise close_error
        self._raise_if_failed()

    def _request(
        self,
        kind: str,
        payload: Any = None,
        wait: bool = False,
        allow_failed: bool = False,
        timeout_s: float | None = None,
        progress_interval_s: float | None = None,
    ) -> None:
        if not allow_failed:
            self._raise_if_failed()
        done = threading.Event() if wait else None
        try:
            self._queue.put(_Request(kind, payload, done), timeout=1.0 if timeout_s is None else timeout_s)
        except queue.Full as exc:
            raise RecorderError("recording queue is full") from exc
        if done is not None:
            started_at = time.monotonic()
            deadline = None if timeout_s is None else started_at + timeout_s
            while True:
                if deadline is None:
                    wait_s = progress_interval_s
                else:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        if done.wait(0):
                            break
                        detail = (
                            "the background writer may still be finalizing the episode"
                            if kind in {"save", "discard"}
                            else "the background worker may still be processing the request"
                        )
                        raise RecorderTimeoutError(
                            f"recording {kind} request timed out after {timeout_s:.1f}s; " f"{detail}"
                        )
                    wait_s = remaining_s
                    if progress_interval_s is not None:
                        wait_s = min(wait_s, progress_interval_s)

                if done.wait(wait_s):
                    break
                if not allow_failed:
                    self._raise_if_failed()
                if deadline is not None and time.monotonic() >= deadline:
                    continue
                if progress_interval_s is not None:
                    logger.warning(
                        "Still waiting for recording %s request: elapsed_s=%.1f timeout_s=%s",
                        kind,
                        time.monotonic() - started_at,
                        "none" if timeout_s is None else f"{timeout_s:.1f}",
                    )
            if not allow_failed:
                self._raise_if_failed()

    def _worker(self) -> None:
        dataset = None
        while True:
            request = self._queue.get()
            closing = request.kind == "close"
            try:
                if request.kind == "init":
                    dataset = self._create_dataset()
                elif request.kind == "frame":
                    frame = dict(request.payload)
                    for name, camera in self.config.cameras.items():
                        if camera.color_mode == "bgr":
                            key = f"observation.images.{name}"
                            frame[key] = frame[key][..., ::-1].copy()
                    try:
                        add_frame_parameters = inspect.signature(dataset.add_frame).parameters
                    except (TypeError, ValueError):
                        add_frame_parameters = {"task": None}
                    if "task" in add_frame_parameters:
                        dataset.add_frame(frame, task=self.config.task)
                    else:
                        # Some LeRobot 0.3.x collection forks consume task from
                        # the frame instead of accepting a keyword argument.
                        frame["task"] = self.config.task
                        dataset.add_frame(frame)
                elif request.kind == "save":
                    episode_index = dataset.num_episodes if self.config.video else None
                    image_writer = getattr(dataset, "image_writer", None)
                    image_queue = getattr(image_writer, "queue", None)
                    pending_writes = None
                    if image_queue is not None:
                        try:
                            pending_writes = image_queue.qsize()
                        except (AttributeError, NotImplementedError):
                            pass
                    image_flush_started_at = time.perf_counter()
                    logger.info(
                        "Episode save stage started: stage=flush_images pending_writes=%s",
                        "unknown" if pending_writes is None else pending_writes,
                    )
                    dataset._wait_image_writer()
                    logger.info(
                        "Episode save stage completed: stage=flush_images elapsed_s=%.3f",
                        time.perf_counter() - image_flush_started_at,
                    )
                    commit_started_at = time.perf_counter()
                    logger.info("Episode save stage started: stage=parquet_video_metadata")
                    dataset.save_episode()
                    if self.config.video:
                        assert episode_index is not None
                        _remove_encoded_episode_images(dataset, episode_index)
                    logger.info(
                        "Episode save stage completed: stage=parquet_video_metadata " "elapsed_s=%.3f",
                        time.perf_counter() - commit_started_at,
                    )
                elif request.kind == "discard":
                    # LeRobot writes camera PNGs on its own asynchronous worker
                    # pool. Wait before deleting the episode image directory;
                    # otherwise a writer can recreate a file while
                    # clear_episode_buffer() is removing the same directory.
                    dataset._wait_image_writer()
                    dataset.clear_episode_buffer()
                elif request.kind == "close" and getattr(dataset, "image_writer", None) is not None:
                    dataset.stop_image_writer()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                if request.done is not None:
                    request.done.set()
                self._queue.task_done()
            if closing:
                return

    def _create_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import write_info

        features = self._features()
        if self.config.resume:
            if not self.root.is_dir():
                raise FileNotFoundError(f"cannot resume missing LeRobot dataset: {self.root}")
            dataset = LeRobotDataset(
                repo_id=self.config.repo_id,
                root=self.root,
            )
            self._validate_resumed_dataset(dataset, features)
            self._reject_stale_resume_episode_artifacts(dataset)
            logger.info(
                "Resuming LeRobot dataset: root=%s episodes=%d frames=%d",
                self.root,
                dataset.num_episodes,
                dataset.num_frames,
            )
        else:
            if self.root.exists():
                raise FileExistsError(
                    f"recording root already exists: {self.root}; " "choose a new root or set recording.resume=true"
                )
            dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                root=self.root,
                fps=self.config.fps,
                robot_type=self.ROBOT_TYPE,
                features=features,
                use_videos=self.config.video,
            )
            logger.info("Created LeRobot dataset: root=%s", self.root)

        teleoperation = dataset.meta.info.get("teleoperation")
        if teleoperation is None:
            teleoperation = {}
            dataset.meta.info["teleoperation"] = teleoperation
        teleoperation.update(self._teleoperation_metadata())
        write_info(dataset.meta.info, dataset.meta.root)
        if self.config.video:
            _configure_parallel_video_encoding(
                dataset,
                self.config.video_encoding_workers,
            )
        if self.config.cameras:
            dataset.start_image_writer(
                num_threads=self.config.image_writer_threads_per_camera * len(self.config.cameras)
            )
        return dataset

    def _teleoperation_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "hardware_executed": self.execution_mode == "hardware",
            "video_encoding_workers": self.config.video_encoding_workers,
            "episode_start_semantics": (
                "first active controller trigger or left-controller X after arming"
                if self.ROBOT_TYPE.startswith("dual_arx")
                else "explicit left-controller X button"
            ),
            "eef_state_action_order": list(EEF_NAMES),
            "joint_state_action_order": list(JOINT_NAMES),
            "joint_state_semantics": "measured joint feedback in radians",
            "translation_unit": "mm",
            "rotation_unit": "axis_angle_rad",
            "inactive_action_semantics": "Cartesian action holds current observation",
            "joint_target_semantics": (
                "joint position target sent to ARX SDK in radians, before SDK internal limiting"
                if self.ROBOT_TYPE.startswith("dual_arx")
                else "exact SDK-millidegree-quantized JointCtrl target in radians"
            ),
            "joint_target_hold_semantics": "last sent per-arm target, initialized from configured home joints",
        }

    def _reject_stale_resume_episode_artifacts(self, dataset: Any) -> None:
        episode_index = dataset.num_episodes
        stale: list[str] = []
        camera_keys = tuple(getattr(dataset.meta, "camera_keys", dataset.meta.video_keys))
        for key in camera_keys:
            image_dir = dataset._get_image_file_path(
                episode_index=episode_index,
                image_key=key,
                frame_index=0,
            ).parent
            png_count = sum(1 for _ in image_dir.glob("frame_*.png"))
            if png_count:
                stale.append(f"{image_dir} ({png_count} PNGs)")
        for key in dataset.meta.video_keys:
            video_path = self.root / dataset.meta.get_video_file_path(episode_index, key)
            if video_path.exists():
                stale.append(str(video_path))
        data_path = self.root / dataset.meta.get_data_file_path(ep_index=episode_index)
        if data_path.exists():
            stale.append(str(data_path))
        if stale:
            details = ", ".join(stale)
            raise ValueError(
                "cannot resume safely because unfinished artifacts exist for the next "
                f"episode {episode_index}: {details}. Back up or remove these artifacts "
                "before resuming; otherwise old PNG frames can be mixed into the new video"
            )

    def _features(self) -> dict[str, dict[str, object]]:
        features: dict[str, dict[str, object]] = {
            key: {
                "dtype": "float32",
                "shape": (14,),
                "names": list(names),
            }
            for key, names in (
                ("state.joint", JOINT_NAMES),
                ("action.joint", JOINT_NAMES),
                ("state.eef", EEF_NAMES),
                ("action.eef", EEF_NAMES),
            )
        }
        for name, camera in self.config.cameras.items():
            features[f"observation.images.{name}"] = {
                "dtype": "video" if self.config.video else "image",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _validate_resumed_dataset(self, dataset, expected_features: dict[str, dict[str, object]]) -> None:
        if dataset.fps != self.config.fps:
            raise ValueError(f"cannot resume dataset with fps={dataset.fps}; configured fps={self.config.fps}")
        if dataset.meta.robot_type != self.ROBOT_TYPE:
            raise ValueError(
                "cannot resume dataset with robot_type=" f"{dataset.meta.robot_type!r}; expected {self.ROBOT_TYPE!r}"
            )

        actual_custom_features = set(dataset.features) - STANDARD_FEATURES
        if actual_custom_features != set(expected_features):
            missing = sorted(set(expected_features) - actual_custom_features)
            extra = sorted(actual_custom_features - set(expected_features))
            raise ValueError(f"cannot resume dataset with a different feature set; missing={missing}, extra={extra}")
        for name, expected in expected_features.items():
            actual = dataset.features[name]
            if (
                actual.get("dtype") != expected["dtype"]
                or tuple(actual.get("shape", ())) != tuple(expected["shape"])
                or actual.get("names") != expected["names"]
            ):
                raise ValueError(
                    f"cannot resume dataset with incompatible feature {name!r}: " f"actual={actual}, expected={expected}"
                )

        teleoperation = dataset.meta.info.get("teleoperation", {})
        previous_mode = teleoperation.get("execution_mode")
        if previous_mode is not None and previous_mode != self.execution_mode:
            raise ValueError(
                f"cannot mix {previous_mode!r} and {self.execution_mode!r} execution modes " "in one LeRobot dataset"
            )

    def _mark_incomplete(self, frames: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / ".incomplete_episode.json"
        marker.write_text(json.dumps({"frames": frames, "task": self.config.task}, indent=2))

    def _clear_incomplete_marker(self) -> None:
        marker = self.root / ".incomplete_episode.json"
        marker.unlink(missing_ok=True)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RecorderError(f"recording worker failed: {self._error}") from self._error
