from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable

from scipy.spatial.transform import Rotation

from .config import TeleoperationConfig
from .contracts import (
    SIDES,
    AppliedArmCommand,
    ArmCommand,
    ArmObservation,
    ArmTarget,
    CommandResult,
    VRFrame,
    VRHand,
)
from .control import EventQueue, KeyboardController
from .pose import RelativePoseMapper, observation_to_command, wxyz_to_xyzw
from .recording import (
    LeRobotRecorder,
    NullRecorder,
    Recorder,
    RecorderError,
    RecorderTimeoutError,
)
from .recording.cameras import CameraHub
from .robot import create_robot
from .vr import VRState, WebXRServer

logger = logging.getLogger(__name__)
TRIGGER_THRESHOLD = 0.5
TRIGGER_RELEASE_THRESHOLD = 0.2


class PreviewRobot:
    def __init__(self) -> None:
        self.observation = {
            side: ArmObservation(
                position_m=(0.3, 0.15 if side == "left" else -0.15, 0.3),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                gripper_open_fraction=1.0,
                joint_positions_rad=(0.0,) * 6,
            )
            for side in SIDES
        }

    def read_observation(self) -> dict[str, ArmObservation]:
        return dict(self.observation)

    def apply_targets(
        self,
        targets: dict[str, ArmTarget | None],
        observation: dict[str, ArmObservation],
        gripper_targets: dict[str, float | None] | None = None,
    ) -> CommandResult:
        applied = {}
        for side, target in targets.items():
            if target is None:
                continue
            rotation = Rotation.from_quat(wxyz_to_xyzw(target.quaternion_wxyz)).as_rotvec()
            command = ArmCommand(
                position_mm=tuple(value * 1000.0 for value in target.position_m),
                rotation_vector_rad=tuple(float(value) for value in rotation),
                gripper_open_fraction=target.gripper_open_fraction,
            )
            applied[side] = AppliedArmCommand(command, False, False)
            self.observation[side] = ArmObservation(
                position_m=target.position_m,
                quaternion_wxyz=target.quaternion_wxyz,
                gripper_open_fraction=(
                    gripper_targets.get(side)
                    if gripper_targets is not None and gripper_targets.get(side) is not None
                    else (
                        target.gripper_open_fraction
                        if target.gripper_open_fraction is not None
                        else observation[side].gripper_open_fraction
                    )
                ),
                joint_positions_rad=observation[side].joint_positions_rad,
            )
        if gripper_targets is not None:
            for side, gripper_target in gripper_targets.items():
                if gripper_target is None:
                    continue
                current = self.observation[side]
                self.observation[side] = ArmObservation(
                    position_m=current.position_m,
                    quaternion_wxyz=current.quaternion_wxyz,
                    gripper_open_fraction=gripper_target,
                    joint_positions_rad=current.joint_positions_rad,
                )
                existing = applied.get(side)
                if existing is None:
                    applied[side] = AppliedArmCommand(
                        command=observation_to_command(
                            observation[side],
                            gripper_open_fraction=gripper_target,
                        ),
                        pose_written=False,
                        gripper_written=True,
                    )
                else:
                    applied[side] = AppliedArmCommand(
                        command=ArmCommand(
                            position_mm=existing.command.position_mm,
                            rotation_vector_rad=existing.command.rotation_vector_rad,
                            gripper_open_fraction=gripper_target,
                        ),
                        pose_written=existing.pose_written,
                        gripper_written=True,
                        joint_target_rad=existing.joint_target_rad,
                    )
        return CommandResult(observation=observation, applied=applied)

    def go_home(self, *, tolerance_deg: float, timeout_s: float) -> dict:
        return {}

    def close(self) -> None:
        return None


class TeleoperationRuntime:
    def __init__(
        self,
        config: TeleoperationConfig,
        *,
        start_vr_server: bool = True,
        keyboard: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.vr_state = VRState()
        self.vr_server = WebXRServer(config.vr, self.vr_state) if start_vr_server else None
        self.events = EventQueue()
        self.keyboard = KeyboardController(self.events) if keyboard else None
        self.robot = create_robot(config) if config.runtime.hardware_access else PreviewRobot()
        self.mapper = RelativePoseMapper(config.vr)
        self.cameras = CameraHub(config.recording.cameras) if config.recording.enabled else None
        self.recorder: Recorder = (
            LeRobotRecorder(
                config.recording,
                execution_mode="hardware" if config.runtime.motion_enabled else "preview",
                robot_type="dual_arx_x5_vr" if config.robot_type == "arx_x5" else "dual_piper_vr",
                home_joint_targets_rad={
                    side: tuple(math.radians(value) for value in arm.home_joints_deg)
                    for side, arm in config.arms.items()
                },
            )
            if config.recording.enabled
            else NullRecorder()
        )
        self._recording_failed = False
        self._episode_active = False
        self._accept_new_episodes = config.recording.enabled
        self._episode_started_at: float | None = None
        self._episode_armed_at: float | None = None
        self._teleop_trigger_active = dict.fromkeys(SIDES, False)
        self._stop = False
        self._last_latency: dict[str, object] = {}
        self._deadline_misses = 0
        self._interval_deadline_misses = 0
        self._interval_max_cycle_work_ms = 0.0
        self._interval_max_record_ms = 0.0

    def open(self) -> None:
        if self.keyboard is not None:
            self.keyboard.start()
        if self.config.runtime.hardware_access and self.config.runtime.motion_enabled:
            self.robot.start_kinematics()
        if self.vr_server is not None:
            self.vr_server.start()
        if self.config.runtime.hardware_access:
            self.robot.open_read_only()
            if self.config.runtime.motion_enabled:
                self.robot.prepare_motion()
        if self.config.recording.enabled:
            self._move_home("Recording startup")
        if self.cameras is not None:
            self.cameras.connect()
        if self.config.recording.enabled:
            self._arm_episode_start()
            logger.info("Recording armed; use X (or an ARX controller trigger) to start the first episode")

    def run(self, max_cycles: int | None = None, frames: Iterable[VRFrame] | None = None) -> int:
        frame_iter = iter(frames) if frames is not None else None
        control_period = 1.0 / self.config.runtime.control_hz
        record_period = 1.0 / self.config.recording.fps if self.config.recording.enabled else None
        next_record_at = time.perf_counter()
        vr_activity_generation = self.vr_state.activity_generation()
        cycles = 0
        while not self._stop and (max_cycles is None or cycles < max_cycles):
            started = time.perf_counter()
            if frame_iter is not None:
                try:
                    self.vr_state.update(next(frame_iter))
                except StopIteration:
                    frame_iter = None
            self._poll_events()
            if self._stop:
                break
            result = self._step()
            self.recorder.observe_commands(result)
            record_ms = 0.0
            if (
                record_period is not None
                and self._episode_active
                and not self._recording_failed
                and started >= next_record_at
            ):
                record_started = time.perf_counter()
                self._record(result)
                record_ms = (time.perf_counter() - record_started) * 1000.0
                next_record_at = max(next_record_at + record_period, started + record_period)
            work_finished = time.perf_counter()
            work_s = work_finished - started
            deadline_overrun_ms = max(0.0, work_s - control_period) * 1000.0
            if deadline_overrun_ms > 0.0:
                self._deadline_misses += 1
                self._interval_deadline_misses += 1
            self._interval_max_cycle_work_ms = max(
                self._interval_max_cycle_work_ms,
                work_s * 1000.0,
            )
            self._interval_max_record_ms = max(self._interval_max_record_ms, record_ms)
            self._last_latency.update(
                {
                    "record_ms": record_ms,
                    "cycle_work_ms": work_s * 1000.0,
                    "deadline_overrun_ms": deadline_overrun_ms,
                    "deadline_misses": self._deadline_misses,
                    "interval_deadline_misses": self._interval_deadline_misses,
                    "interval_max_cycle_work_ms": self._interval_max_cycle_work_ms,
                    "interval_max_record_ms": self._interval_max_record_ms,
                }
            )
            cycles += 1
            if self.config.runtime.log_every and cycles % self.config.runtime.log_every == 0:
                self._log_latency(cycles, result)
            remaining = control_period - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            if frame_iter is None and not self._stop:
                # After enforcing the configured maximum command rate, wait
                # for a fresh WebXR frame when none arrived during the cycle.
                # This phase-locks a 90 Hz control loop to a 72/90 Hz headset
                # instead of repeatedly sampling it on an unrelated timer.
                vr_activity_generation = self.vr_state.wait_for_activity(
                    vr_activity_generation,
                    control_period,
                )
        return cycles

    def close(self) -> None:
        self._stop = True
        if self.keyboard is not None:
            self.keyboard.close()
        if self.vr_server is not None:
            self.vr_server.stop()
        try:
            self.recorder.close()
        finally:
            if self.cameras is not None:
                self.cameras.close()
            self.robot.close()

    def _step(self) -> CommandResult:
        step_started = time.perf_counter()
        read_started = step_started
        observation = self.robot.read_observation()
        read_finished = time.perf_counter()
        frame = self.vr_state.latest()
        vr_receive = self.vr_state.receive_diagnostics()
        vr_age_ms = None if frame is None else max(0.0, time.monotonic() - frame.received_monotonic_s) * 1000.0
        vr_stale = frame is not None and vr_age_ms > self.config.vr.stale_timeout_s * 1000.0
        if frame is None or vr_stale:
            self.mapper.reset()
            self._teleop_trigger_active = dict.fromkeys(SIDES, False)
            result = CommandResult(observation=observation)
            self._last_latency = {
                "vr_seq": None if frame is None else frame.seq,
                "vr_age_ms": vr_age_ms,
                "vr_stale": vr_stale,
                "vr_hands": {},
                "vr_receive": vr_receive,
                "read_ms": (read_finished - read_started) * 1000.0,
                "map_ms": 0.0,
                "apply_ms": 0.0,
                "step_ms": (time.perf_counter() - step_started) * 1000.0,
                "robot": self._robot_latency_diagnostics(),
            }
            return result

        map_started = time.perf_counter()
        hands = {side: frame.hands.get(arm.hand, VRHand()) for side, arm in self.config.arms.items()}
        trigger_active = {side: self._update_teleop_trigger(side, hand) for side, hand in hands.items()}
        if self.config.robot_type == "arx_x5" and self._episode_armed_at is not None and any(trigger_active.values()):
            self._start_episode_from_controller()
        targets = {}
        gripper_targets = {}
        vr_hands = {}
        for side, hand in hands.items():
            if trigger_active[side]:
                targets[side] = self.mapper.map(side, hand, observation[side])
            else:
                self.mapper.reset(side)
                targets[side] = None
            gripper_targets[side] = 1.0 - hand.squeeze if trigger_active[side] else None
            vr_hands[side] = {
                "tracked": hand.tracked,
                "trigger": round(hand.trigger, 3),
                "squeeze": round(hand.squeeze, 3),
                "active": trigger_active[side],
                "target": targets[side] is not None,
                "gripper_target": None if gripper_targets[side] is None else round(float(gripper_targets[side]), 3),
            }
        map_finished = time.perf_counter()
        result = self.robot.apply_targets(
            targets,
            observation,
            gripper_targets=gripper_targets,
        )
        apply_finished = time.perf_counter()
        self._last_latency = {
            "vr_seq": frame.seq,
            "vr_age_ms": vr_age_ms,
            "vr_stale": False,
            "vr_hands": vr_hands,
            "vr_receive": vr_receive,
            "read_ms": (read_finished - read_started) * 1000.0,
            "map_ms": (map_finished - map_started) * 1000.0,
            "apply_ms": (apply_finished - map_finished) * 1000.0,
            "step_ms": (apply_finished - step_started) * 1000.0,
            "robot": self._robot_latency_diagnostics(),
        }
        return result

    def _robot_latency_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.robot, "latency_diagnostics", None)
        return diagnostics() if callable(diagnostics) else {}

    def _log_latency(self, cycles: int, result: CommandResult) -> None:
        values = self._last_latency
        logger.info(
            "teleop cycles=%d write=%s vr_seq=%s vr_age_ms=%s vr_stale=%s vr_hands=%s "
            "vr_receive=%s "
            "read_ms=%.2f map_ms=%.2f "
            "apply_ms=%.2f step_ms=%.2f record_ms=%.2f cycle_work_ms=%.2f "
            "interval_max_record_ms=%.2f interval_max_cycle_work_ms=%.2f "
            "deadline_overrun_ms=%.2f interval_deadline_misses=%d deadline_misses=%d robot=%s",
            cycles,
            result.hardware_write,
            values.get("vr_seq"),
            "--" if values.get("vr_age_ms") is None else f"{float(values['vr_age_ms']):.2f}",
            bool(values.get("vr_stale", False)),
            values.get("vr_hands", {}),
            values.get("vr_receive", {}),
            float(values.get("read_ms", 0.0)),
            float(values.get("map_ms", 0.0)),
            float(values.get("apply_ms", 0.0)),
            float(values.get("step_ms", 0.0)),
            float(values.get("record_ms", 0.0)),
            float(values.get("cycle_work_ms", 0.0)),
            float(values.get("interval_max_record_ms", 0.0)),
            float(values.get("interval_max_cycle_work_ms", 0.0)),
            float(values.get("deadline_overrun_ms", 0.0)),
            int(values.get("interval_deadline_misses", 0)),
            int(values.get("deadline_misses", 0)),
            values.get("robot", {}),
        )
        self._interval_deadline_misses = 0
        self._interval_max_cycle_work_ms = 0.0
        self._interval_max_record_ms = 0.0

    def _update_teleop_trigger(self, side: str, hand: VRHand) -> bool:
        if not hand.tracked:
            self._teleop_trigger_active[side] = False
            return False
        active = self._teleop_trigger_active[side]
        if active:
            active = hand.trigger > TRIGGER_RELEASE_THRESHOLD
        else:
            active = hand.trigger >= TRIGGER_THRESHOLD
        self._teleop_trigger_active[side] = active
        return active

    def _poll_events(self) -> None:
        for event in self.vr_state.drain_events():
            self._handle_event(
                event.kind,
                received_monotonic_s=event.received_monotonic_s,
            )
        for event in self.events.drain():
            self._handle_event(event.kind)

    def _handle_event(
        self,
        kind: str,
        *,
        received_monotonic_s: float | None = None,
    ) -> None:
        if kind == "quit":
            self._stop = True
            return
        if kind == "save_episode_and_home" and not self.config.recording.enabled:
            self._go_home_from_controller()
            return
        if (
            kind
            in {
                "start_episode",
                "save_episode",
                "save_episode_and_home",
                "discard_episode_and_home",
                "rerecord_episode",
                "stop_recording",
            }
            and not self.config.recording.enabled
        ):
            logger.debug("Ignoring recording event while recording is disabled: %s", kind)
            return
        if self._recording_failed:
            return
        if kind == "start_episode":
            self._start_episode_from_controller(
                received_monotonic_s=received_monotonic_s,
            )
        elif kind == "save_episode_and_home":
            self._save_episode_and_home()
        elif kind in {"discard_episode_and_home", "rerecord_episode"}:
            self._discard_episode_and_home()
        elif kind in {"save_episode", "stop_recording"}:
            try:
                self._episode_active = False
                self._episode_armed_at = None
                self.recorder.finish_episode(save=True)
                self._episode_started_at = None
                self.mapper.reset()
                if kind == "stop_recording":
                    self._accept_new_episodes = False
                    logger.info("Recording stopped; episode start control will no longer start episodes")
                else:
                    self._arm_episode_start()
                    logger.info("Episode saved; use X (or an ARX controller trigger) to start recording")
            except RecorderTimeoutError:
                self._recording_timeout()
            except RecorderError:
                self._recording_failure()

    def _save_episode_and_home(self) -> None:
        self._finish_episode_and_home(save=True)

    def _discard_episode_and_home(self) -> None:
        self._finish_episode_and_home(save=False)

    def _start_episode_from_controller(
        self,
        *,
        received_monotonic_s: float | None = None,
    ) -> None:
        if not self._accept_new_episodes:
            logger.warning("Ignoring episode start command: recording is not accepting new episodes")
            return
        if self._episode_active:
            logger.warning("Ignoring episode start command: an episode is already active")
            return
        if self._episode_armed_at is None:
            logger.warning("Ignoring episode start command: recording is not armed yet")
            return
        if received_monotonic_s is not None and received_monotonic_s < self._episode_armed_at:
            logger.warning(
                "Ignoring stale episode start command received before recording " "was armed: event_age_at_arm_s=%.3f",
                self._episode_armed_at - received_monotonic_s,
            )
            return
        try:
            self.mapper.reset()
            self.recorder.start_episode()
            self._episode_active = True
            self._episode_started_at = time.monotonic()
            self._episode_armed_at = None
            logger.info(
                "Episode start requested; recording episode started: hands=%s",
                self._trigger_diagnostics(),
            )
        except RecorderError:
            self._recording_failure()

    def _go_home_from_controller(self) -> None:
        if not self._triggers_released():
            logger.warning("Ignoring right-controller A go-home command: release both triggers before going home")
            return
        try:
            self._move_home("Right-controller A pressed")
        except Exception:
            self._stop = True
            logger.exception("Controller go-home failed; stopping teleoperation")

    def _finish_episode_and_home(self, *, save: bool) -> None:
        action = "saving" if save else "discarding"
        if not self._triggers_released():
            logger.warning(
                "Ignoring controller episode command: release both triggers before %s and going home",
                action,
            )
            return

        # Stop accepting frames immediately, but do not synchronously finalize
        # the LeRobot episode yet. Finalization can wait for pending PNG writes
        # and encode one MP4 per camera, which is much slower for long episodes.
        # Moving home first makes the A/B command responsive while keeping the
        # home motion outside the recorded episode.
        self._episode_active = False
        self._episode_armed_at = None
        capture_elapsed_s = (
            None if self._episode_started_at is None else max(0.0, time.monotonic() - self._episode_started_at)
        )
        logger.info(
            "Episode capture stopped: elapsed_s=%s",
            "unknown" if capture_elapsed_s is None else f"{capture_elapsed_s:.3f}",
        )
        self._episode_started_at = None
        result = "saved" if save else "discarded"
        home_failed = False
        try:
            self._move_home(f"Episode capture stopped; moving home before it is {result}")
        except Exception:
            home_failed = True
            logger.exception(
                "Go-home failed after episode capture stopped; attempting to %s the episode "
                "before stopping teleoperation",
                "save" if save else "discard",
            )

        try:
            logger.info(
                "Both arms are no longer being teleoperated; %s recorded episode now",
                "saving" if save else "discarding",
            )
            self.recorder.finish_episode(save=save)
        except RecorderTimeoutError:
            self._recording_timeout()
            if home_failed:
                self._stop = True
            return
        except RecorderError:
            self._recording_failure()
            if home_failed:
                self._stop = True
            return

        if home_failed:
            self._stop = True
            logger.error(
                "Episode was %s, but go-home failed; stopping teleoperation without arming " "the next episode",
                result,
            )
            return

        self._arm_episode_start()
        logger.info(
            "Episode %s after returning home; recording armed. Use X (or an ARX controller trigger) "
            "to start the next episode",
            result,
        )

    def _move_home(self, reason: str) -> None:
        self.mapper.reset()
        self._teleop_trigger_active = dict.fromkeys(SIDES, False)
        try:
            if self.config.runtime.hardware_access and self.config.runtime.motion_enabled:
                logger.info(
                    "%s; moving both arms to configured home joints " "(tolerance=%.3fdeg timeout=%.1fs)",
                    reason,
                    self.config.recording.home_tolerance_deg,
                    self.config.recording.home_timeout_s,
                )
                self.robot.go_home(
                    tolerance_deg=self.config.recording.home_tolerance_deg,
                    timeout_s=self.config.recording.home_timeout_s,
                )
                logger.info("Both arms reached configured home joints")
        finally:
            self.mapper.reset()
            self._teleop_trigger_active = dict.fromkeys(SIDES, False)

    def _arm_episode_start(self) -> None:
        self._episode_active = False
        self._episode_started_at = None
        self._episode_armed_at = time.monotonic()

    def _trigger_diagnostics(self) -> dict[str, dict[str, object]]:
        frame = self.vr_state.latest()
        if frame is None:
            return {}
        return {
            side: {
                "tracked": hand.tracked,
                "trigger": round(hand.trigger, 3),
                "squeeze": round(hand.squeeze, 3),
            }
            for side, arm in self.config.arms.items()
            for hand in [frame.hands.get(arm.hand, VRHand())]
        }

    def _triggers_released(self) -> bool:
        frame = self.vr_state.latest()
        if frame is None:
            return False
        if time.monotonic() - frame.received_monotonic_s > self.config.vr.stale_timeout_s:
            return False
        return all(
            not (hand := frame.hands.get(arm.hand, VRHand())).tracked or hand.trigger <= TRIGGER_RELEASE_THRESHOLD
            for arm in self.config.arms.values()
        )

    def _record(self, result: CommandResult) -> None:
        try:
            images = self.cameras.read() if self.cameras is not None else {}
            self.recorder.add_frame(result, images)
        except Exception:
            self._recording_failure()

    def _recording_failure(self) -> None:
        if self._recording_failed:
            return
        self._recording_failed = True
        self._accept_new_episodes = False
        self._episode_active = False
        self._episode_started_at = None
        self._episode_armed_at = None
        logger.exception("Recording stopped after a writer failure")

    def _recording_timeout(self) -> None:
        if self._recording_failed:
            return
        self._recording_failed = True
        self._accept_new_episodes = False
        self._episode_active = False
        self._episode_started_at = None
        self._episode_armed_at = None
        self._stop = True
        logger.exception(
            "Recording episode finalization exceeded its timeout; the background writer may "
            "still be saving. Stopping teleoperation without starting another episode"
        )


def synthetic_vr_frames(count: int) -> Iterable[VRFrame]:
    for seq in range(count):
        offset = seq * 0.0005
        hand = VRHand(
            tracked=True,
            position_m=(offset, 1.2, -0.3),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            trigger=1.0,
            squeeze=0.2,
        )
        yield VRFrame(
            session_id="synthetic",
            seq=seq,
            client_timestamp_ms=float(seq),
            received_monotonic_s=time.monotonic(),
            hands={"left": hand, "right": hand},
        )
