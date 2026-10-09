from __future__ import annotations

import logging
import math
import time
from dataclasses import replace

from ..control import EventQueue
from ..recording import NullRecorder, Recorder, RecorderError, RecorderTimeoutError
from ..recording.cameras import CameraHub
from .config import FlexivSpaceMouseConfig
from .contracts import FlexivCommandResult
from .keyboard import RecordingKeyboardController
from .mapper import SpaceMousePoseMapper
from .recorder import FlexivLeRobotRecorder
from .robot import FlexivRobot, PreviewFlexivRobot
from .spacemouse import SpaceMouseReader

logger = logging.getLogger(__name__)


class FlexivSpaceMouseRuntime:
    def __init__(
        self,
        config: FlexivSpaceMouseConfig,
        *,
        keyboard: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.events = EventQueue()
        self.keyboard = RecordingKeyboardController(self.events) if keyboard else None
        self.spacemouse = SpaceMouseReader(config.spacemouse)
        self.robot = (
            FlexivRobot(config.robot)
            if config.runtime.hardware_access
            else PreviewFlexivRobot(config.robot)
        )
        self.mapper = SpaceMousePoseMapper(config.spacemouse, config.robot)
        self.cameras = CameraHub(config.recording.cameras) if config.recording.enabled else None
        self.recorder: Recorder = (
            FlexivLeRobotRecorder(
                config.recording,
                execution_mode="hardware" if config.runtime.motion_enabled else "preview",
                gripper_scale=config.robot.gripper.gripper_scale,
            )
            if config.recording.enabled
            else NullRecorder()
        )
        self._episode_active = False
        self._ready_after_neutral = False
        self._stop = False
        self._recording_failed = False
        self._gripper_target_m: float | None = None
        self._gripper_held_button: int | None = None
        self._gripper_hold_start_m: float | None = None
        self._gripper_hold_elapsed_s = 0.0
        self._last_log: dict[str, object] = {}

    def open(self) -> None:
        if self.keyboard is not None:
            self.keyboard.start()
        self.spacemouse.open()
        self.robot.open(enable_motion=self.config.runtime.motion_enabled)
        self._move_home("Recording startup")
        if self.cameras is not None:
            self.cameras.connect()
        logger.info(
            "Recording armed; center the SpaceMouse, then move any axis to start the first episode"
        )

    def run(self, max_cycles: int | None = None) -> int:
        control_period = 1.0 / self.config.runtime.control_hz
        record_period = 1.0 / self.config.recording.fps if self.config.recording.enabled else None
        next_record_at = time.perf_counter()
        previous_step_at = time.perf_counter()
        cycles = 0
        while not self._stop and (max_cycles is None or cycles < max_cycles):
            started = time.perf_counter()
            self._poll_events()
            if self._stop:
                break
            dt_s = max(1e-6, started - previous_step_at)
            previous_step_at = started
            result = self._step(dt_s)
            if (
                record_period is not None
                and self._episode_active
                and not self._recording_failed
                and started >= next_record_at
            ):
                self._record(result)
                next_record_at = max(next_record_at + record_period, started + record_period)
            cycles += 1
            if self.config.runtime.log_every and cycles % self.config.runtime.log_every == 0:
                logger.info(
                    "flexiv-spacemouse cycles=%d write=%s episode_active=%s ready=%s state=%s",
                    cycles,
                    result.hardware_write,
                    self._episode_active,
                    self._ready_after_neutral,
                    self._last_log,
                )
            remaining = control_period - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        return cycles

    def close(self) -> None:
        self._stop = True
        if self.keyboard is not None:
            self.keyboard.close()
        self.spacemouse.close()
        try:
            self.recorder.close()
        finally:
            try:
                if self.cameras is not None:
                    self.cameras.close()
            finally:
                self.robot.close()

    def _step(self, dt_s: float) -> FlexivCommandResult:
        observation = self.robot.read_observation()
        state = self.spacemouse.snapshot()
        if not state.connected:
            raise RuntimeError("SpaceMouse disconnected")
        if self._gripper_target_m is None:
            self._gripper_target_m = observation.gripper_width_m
        gripper_written = self._handle_gripper_buttons(
            state.buttons,
            dt_s,
            observation.gripper_width_m,
        )

        if not self._ready_after_neutral:
            self.mapper.reset(observation)
            if self.mapper.is_neutral(state):
                self._ready_after_neutral = True
                logger.info("SpaceMouse is neutral; motion and automatic episode start are enabled")
            result = FlexivCommandResult(
                observation=observation,
                gripper_written=gripper_written,
            )
        else:
            target = self.mapper.map(
                state,
                observation,
                dt_s,
                gripper_width_m=self._gripper_target_m,
            )
            if target is not None and self.config.recording.enabled and not self._episode_active:
                self.recorder.start_episode()
                self._episode_active = True
                logger.info("SpaceMouse moved; recording episode started")
            result = self.robot.apply_target(target, observation)
            if result.control_mode_recovered:
                self._ready_after_neutral = False
                self.mapper.reset(observation)
                logger.warning(
                    "Flexiv Cartesian mode recovered at the feedback pose; "
                    "center the SpaceMouse before motion resumes"
                )
            if gripper_written:
                result = replace(result, gripper_written=True)

        self._last_log = {
            "axes": tuple(round(value, 3) for value in state.axes),
            "buttons": sorted(state.buttons),
            "motion_active": self.mapper.active,
        }
        return result

    def _handle_gripper_buttons(
        self,
        buttons: frozenset[int],
        dt_s: float,
        feedback_width_m: float,
    ) -> bool:
        gripper = self.config.robot.gripper
        if not gripper.enabled:
            return False

        close_width_m, open_width_m = self._gripper_width_limits()

        open_held = gripper.open_button in buttons
        close_held = gripper.close_button in buttons
        if open_held == close_held:
            stopped = self._finish_gripper_hold(feedback_width_m)
            if not stopped and math.isfinite(feedback_width_m):
                self._gripper_target_m = min(
                    open_width_m,
                    max(close_width_m, feedback_width_m),
                )
            return stopped

        held_button = gripper.open_button if open_held else gripper.close_button
        if held_button != self._gripper_held_button:
            if self._gripper_held_button is not None:
                self._finish_gripper_hold(feedback_width_m)
            assert self._gripper_target_m is not None
            self._gripper_held_button = held_button
            self._gripper_hold_start_m = self._gripper_target_m
            self._gripper_hold_elapsed_s = 0.0
            boundary_m = open_width_m if open_held else close_width_m
            if boundary_m == self._gripper_hold_start_m:
                self._reset_gripper_hold()
                return False
            # A single far target lets the gripper's own velocity controller move smoothly;
            # elapsed time below tracks where to settle when the button is released.
            self.robot.start_gripper_motion(boundary_m)
        else:
            self._gripper_hold_elapsed_s += dt_s

        assert self._gripper_hold_start_m is not None
        distance_m = max(gripper.step_width_m, gripper.velocity_m_s * self._gripper_hold_elapsed_s)
        direction = 1 if open_held else -1
        self._gripper_target_m = min(
            open_width_m,
            max(close_width_m, self._gripper_hold_start_m + direction * distance_m),
        )
        return self._gripper_hold_elapsed_s == 0.0

    def _finish_gripper_hold(self, feedback_width_m: float) -> bool:
        if self._gripper_held_button is None:
            return False
        assert self._gripper_target_m is not None
        gripper = self.config.robot.gripper
        opening = self._gripper_held_button == gripper.open_button
        estimated_target_m = self._gripper_target_m
        close_width_m, open_width_m = self._gripper_width_limits()
        if not math.isfinite(feedback_width_m):
            raise RuntimeError(f"Flexiv gripper returned non-finite width: {feedback_width_m}")
        feedback_width_m = min(open_width_m, max(close_width_m, feedback_width_m))
        self.robot.stop_gripper_at(feedback_width_m)
        self._gripper_target_m = feedback_width_m
        logger.info(
            "SpaceMouse gripper %s hold finished: duration=%.3fs "
            "feedback=%.1fmm estimated=%.1fmm",
            "open" if opening else "close",
            self._gripper_hold_elapsed_s,
            feedback_width_m * 1000.0,
            estimated_target_m * 1000.0,
        )
        self._reset_gripper_hold()
        return True

    def _gripper_width_limits(self) -> tuple[float, float]:
        configured = self.config.robot.gripper
        get_limits = getattr(self.robot, "gripper_width_limits", None)
        if not callable(get_limits):
            return configured.close_width_m, configured.open_width_m
        close_width_m, open_width_m = get_limits()
        if not 0 <= close_width_m < open_width_m:
            raise RuntimeError(
                f"robot returned invalid gripper limits: [{close_width_m}, {open_width_m}]"
            )
        return close_width_m, open_width_m

    def _reset_gripper_hold(self) -> None:
        self._gripper_held_button = None
        self._gripper_hold_start_m = None
        self._gripper_hold_elapsed_s = 0.0

    def _poll_events(self) -> None:
        for event in self.events.drain():
            if event.kind == "quit":
                self._stop = True
            elif event.kind == "save_episode_and_home":
                self._finish_episode_and_home(save=True)
            elif event.kind == "discard_episode_and_home":
                self._finish_episode_and_home(save=False)

    def _finish_episode_and_home(self, *, save: bool) -> None:
        result = "saved" if save else "discarded"
        self._episode_active = False
        self._ready_after_neutral = False
        self.mapper.reset()
        home_failed = False
        try:
            self._move_home(
                f"{'Right' if save else 'Left'} Arrow pressed; capture stopped before episode is {result}"
            )
        except Exception:
            home_failed = True
            logger.exception("Flexiv go-home failed after episode capture stopped")
        try:
            self.recorder.finish_episode(save=save)
        except RecorderTimeoutError:
            self._recording_timeout()
            return
        except RecorderError:
            self._recording_failure()
            return
        if home_failed:
            self._stop = True
            return
        logger.info(
            "Episode %s after returning home; center the SpaceMouse, then move it to start the next episode",
            result,
        )

    def _move_home(self, reason: str) -> None:
        self._ready_after_neutral = False
        self._reset_gripper_hold()
        self.mapper.reset()
        if self.config.runtime.motion_enabled:
            logger.info(
                "%s; moving Flexiv to configured home joints (tolerance=%.3fdeg timeout=%.1fs)",
                reason,
                self.config.recording.home_tolerance_deg,
                self.config.recording.home_timeout_s,
            )
            observation = self.robot.go_home(
                tolerance_deg=self.config.recording.home_tolerance_deg,
                timeout_s=self.config.recording.home_timeout_s,
            )
            self.robot.prepare_cartesian_motion()
            logger.info("Flexiv reached configured home joints")
        else:
            observation = self.robot.read_observation()
        if self.config.robot.gripper.enabled:
            _, self._gripper_target_m = self._gripper_width_limits()
        self.mapper.reset(observation)

    def _record(self, result: FlexivCommandResult) -> None:
        try:
            images = self.cameras.read() if self.cameras is not None else {}
            self.recorder.add_frame(result, images)
        except Exception:
            self._recording_failure()

    def _recording_failure(self) -> None:
        if self._recording_failed:
            return
        self._recording_failed = True
        self._episode_active = False
        self._stop = True
        logger.exception("Recording stopped after a writer or camera failure")

    def _recording_timeout(self) -> None:
        if self._recording_failed:
            return
        self._recording_failed = True
        self._episode_active = False
        self._stop = True
        logger.exception(
            "Recording episode finalization exceeded its timeout; stopping without starting another episode"
        )
