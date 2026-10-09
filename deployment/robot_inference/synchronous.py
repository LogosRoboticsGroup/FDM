from __future__ import annotations

import logging
import threading
import time
from contextlib import ExitStack
from typing import Any, Mapping

import numpy as np

from .client import PolicyReply, PolicyRequestCancelled
from .config import inference_recording_config
from .controls import InferenceControls
from .recorder import InferenceLeRobotRecorder

logger = logging.getLogger(__name__)


class SynchronousInferenceRuntime:
    """Shared operator, episode and synchronous dispatch lifecycle for all arms."""

    def _init_interaction(self, controls: Any | None, recorder: Any | None) -> None:
        self.controls = controls if controls is not None else InferenceControls(self.config.controls, self.config.vr)
        self.recorder = recorder
        self._episode_active = False
        self._stop = False
        self._ready_at = 0.0
        self._robot_lock = threading.Lock()
        self._capture_stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._capture_error: BaseException | None = None
        self._recorded_actions: dict[str, np.ndarray] = {}

    def run(self) -> None:
        if not self._robot_open:
            raise RuntimeError("open() must be called before run()")
        if self.recorder is None and self.config.recording.enabled:
            self.recorder = InferenceLeRobotRecorder(
                inference_recording_config(self.config),
                robot_type=self.ROBOT_TYPE,
                action_space=getattr(self.config.inference, "action_space", "joint_position"),
                gripper_scale=self._model_gripper_scale,
            )
        self._ready_at = time.monotonic()
        self.controls.start()
        logger.info("Hardware ready; %s to start inference; q/Esc to quit", self._start_key)
        chunks = 0
        policy = self.config.inference
        while not self._stop:
            self._poll_events()
            if self._stop:
                break
            if not self._episode_active:
                time.sleep(0.01)
                continue
            try:
                reply, executed, latency = self._execute_chunk()
            except PolicyRequestCancelled:
                continue
            chunks += 1
            if policy.log_every_chunks and chunks % policy.log_every_chunks == 0:
                logger.info(
                    "%s synchronous inference chunk=%d returned=%d executed=%d roundtrip_ms=%.3f server_ms=%s",
                    self.ROBOT_TYPE,
                    chunks,
                    len(reply.actions),
                    executed,
                    latency * 1000.0,
                    reply.infer_time_ms,
                )

    @property
    def _start_key(self) -> str:
        return "left-controller X" if self.config.controls == "vr" else "Enter"

    def _poll_events(self) -> None:
        self._raise_capture_error()
        events = self.controls.drain()
        if any(event.kind == "quit" for event in events):
            self._stop = True
            self._episode_active = False
            self._stop_capture()
            return
        events = [event for event in events if event.received_monotonic_s >= self._ready_at]
        if any(event.kind in self.controls.STOP_EVENTS for event in events):
            events = [event for event in events if event.kind != "start_episode"]
        for event in events:
            if event.received_monotonic_s < self._ready_at:
                continue
            if event.kind == "start_episode":
                if self._episode_active:
                    continue
                try:
                    self.client.reset(cancelled=self.controls.interrupted)
                except PolicyRequestCancelled:
                    return
                if self.controls.interrupted():
                    return
                self._reset_action_history()
                if self.recorder is not None:
                    self.recorder.start_episode()
                self._episode_active = True
                self._start_capture()
                logger.info("Inference episode started")
            elif event.kind in self.controls.STOP_EVENTS:
                self._episode_active = False
                self._stop_capture()
                save = event.kind in {"save_episode", "save_episode_and_home"}
                home = event.kind.endswith("_and_home")
                # Stop capture and home before waiting for video encoding or
                # image cleanup. Neither operation runs on a robot writer thread.
                try:
                    if home:
                        self._go_home()
                finally:
                    if self.recorder is not None:
                        self.recorder.finish_episode(save=save)
                self._ready_at = time.monotonic()
                logger.info("Episode %s; waiting for %s", "saved" if save else "discarded", self._start_key)

    def _go_home(self) -> None:
        policy = self.config.inference
        self.robot.go_home(tolerance_deg=policy.home_tolerance_deg, timeout_s=policy.home_timeout_s)

    def _reset_action_history(self) -> None:
        """Hardware adapters may reset held-command tracking at episode boundaries."""

    def _start_capture(self) -> None:
        if self.recorder is None:
            return
        self._recorded_actions = {}
        self._capture_error = None
        self._capture_stop.clear()
        # Capture the initial hold before waiting for the first policy response.
        self._capture_frame()
        self._capture_thread = threading.Thread(target=self._capture_loop, name="inference-capture", daemon=True)
        self._capture_thread.start()

    def _stop_capture(self) -> None:
        self._capture_stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join()
            self._capture_thread = None
        self._raise_capture_error()

    def _capture_loop(self) -> None:
        period = 1.0 / self.config.inference.fps
        next_frame_at = time.monotonic() + period
        try:
            while not self._capture_stop.wait(max(0.0, next_frame_at - time.monotonic())):
                self._capture_frame()
                next_frame_at += period
                # An overdue sample must not cause a burst of duplicate frames.
                if next_frame_at < time.monotonic():
                    next_frame_at = time.monotonic() + period
        except BaseException as exc:
            self._capture_error = exc
            self._capture_stop.set()

    def _capture_frame(self) -> None:
        with self._robot_lock:
            if self._capture_stop.is_set():
                return
            observation = self.robot.read_observation()
            images = self._read_rgb_images()
            frame = self._stationary_frame(observation)
            if not self._recorded_actions:
                self._recorded_actions = {key: value.copy() for key, value in frame.items() if key.startswith("action")}
            frame.update(self._recorded_actions)
            self.recorder.add_frame(frame, images)
            if "action.executed" in self._recorded_actions:
                self._recorded_actions["action.executed"] = np.zeros(2, dtype=np.float32)

    def _raise_capture_error(self) -> None:
        if self._capture_error is not None:
            raise RuntimeError("inference recording capture failed") from self._capture_error

    def _interrupted(self) -> bool:
        self._raise_capture_error()
        return self.controls.interrupted()

    def close(self) -> None:
        self._stop = True
        self._episode_active = False
        # ExitStack runs all cleanup stages even after a device or writer error.
        with ExitStack() as stack:
            if self.recorder is not None:
                stack.callback(self.recorder.close)
            stack.callback(self.controls.close)
            for flag, resource in (
                ("_client_open", self.client),
                ("_cameras_open", self.cameras),
                ("_robot_open", self.robot),
            ):
                if getattr(self, flag):
                    setattr(self, flag, False)
                    stack.callback(resource.close)
            # Join before closing either the robot, cameras or recorder.
            stack.callback(self._stop_capture)

    def _execute_chunk(self) -> tuple[PolicyReply, int, float]:
        if not self._robot_open:
            raise RuntimeError("open() must be called before executing inference")
        self._check_interrupted()
        policy = self.config.inference
        with self._robot_lock:
            observation = self.robot.read_observation()
        state = self._state_vector(observation)
        images = self._read_rgb_images()
        self._check_interrupted()
        started_at = time.perf_counter()
        reply = self.client.predict(
            images,
            state,
            prompt=policy.prompt,
            fps=policy.fps,
            stat_key=policy.stat_key,
            cancelled=self._interrupted,
        )
        latency = time.perf_counter() - started_at
        self._check_interrupted()
        actions = self._prepare_policy_actions(reply.actions)
        self._validate_action_array(actions)
        selected = actions[: min(policy.n_execute, len(actions))]
        self._validate_action_sequence(selected)
        executed = 0
        next_action_at = time.perf_counter()
        for action in selected:
            if self._interrupted():
                break
            # Serialize SDK reads and writes with the independent recording
            # sampler; the network request never holds this lock.
            with self._robot_lock:
                observation = self.robot.read_observation()
                if self._interrupted():
                    break
                frame = self._execute_action(action, observation)
                if frame is not None:
                    executed += 1
                    for key, value in frame.items():
                        if key.startswith("action"):
                            if key == "action.executed" and key in self._recorded_actions:
                                value = np.maximum(value, self._recorded_actions[key])
                            self._recorded_actions[key] = value.copy()
            next_action_at += 1.0 / policy.fps
            while (remaining := next_action_at - time.perf_counter()) > 0:
                if self._interrupted():
                    break
                time.sleep(min(0.01, remaining))
        return reply, executed, latency

    def _check_interrupted(self) -> None:
        if self._interrupted():
            raise PolicyRequestCancelled()

    def _validate_action_sequence(self, actions: np.ndarray) -> None:
        pass

    def _prepare_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        return actions

    @property
    def _model_gripper_scale(self) -> float:
        return 1.0

    def _read_rgb_images(self) -> dict[str, np.ndarray]:
        raw: Mapping[str, np.ndarray] = self.cameras.read()
        expected = set(self.config.inference.camera_order)
        if set(raw) != expected:
            raise RuntimeError(
                f"camera hub returned an unexpected camera set: expected={sorted(expected)}, actual={sorted(raw)}"
            )
        return {
            name: (
                np.asarray(raw[name])[..., ::-1].copy()
                if self.config.cameras[name].color_mode == "bgr"
                else np.asarray(raw[name])
            )
            for name in self.config.inference.camera_order
        }
