from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from ..contracts import SIDES, VRFrame, VRHand

ALLOWED_EVENTS = {
    "start_episode",
    "save_episode",
    "save_episode_and_home",
    "discard_episode_and_home",
    "rerecord_episode",
    "stop_recording",
}


@dataclass(frozen=True)
class VREvent:
    kind: str
    received_monotonic_s: float


def _numbers(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _unit_value(payload: dict[str, Any], name: str) -> float:
    value = float(payload.get(name, 0.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _hand(payload: Any) -> VRHand:
    if not isinstance(payload, dict) or not payload.get("tracked", False):
        return VRHand()
    return VRHand(
        tracked=True,
        position_m=_numbers(payload.get("position_m"), 3, "position_m"),  # type: ignore[arg-type]
        quaternion_xyzw=_numbers(payload.get("quaternion_xyzw"), 4, "quaternion_xyzw"),  # type: ignore[arg-type]
        trigger=_unit_value(payload, "trigger"),
        squeeze=_unit_value(payload, "squeeze"),
    )


def parse_vr_message(
    message: str | bytes | dict[str, Any],
    received_monotonic_s: float | None = None,
) -> VRFrame:
    raw = message if isinstance(message, dict) else json.loads(message)
    if not isinstance(raw, dict):
        raise ValueError("VR message must be a mapping")
    if raw.get("protocol") != "starvla.vr" or raw.get("version") != 1:
        raise ValueError("unsupported VR protocol")
    session_id = str(raw.get("session_id", "")).strip()
    if not session_id:
        raise ValueError("session_id is required")
    seq = int(raw.get("seq", -1))
    if seq < 0:
        raise ValueError("seq must be non-negative")
    timestamp = float(raw.get("client_timestamp_ms", 0.0))
    if not math.isfinite(timestamp):
        raise ValueError("client_timestamp_ms must be finite")
    hands = raw.get("hands", {})
    if not isinstance(hands, dict):
        raise ValueError("hands must be a mapping")
    return VRFrame(
        session_id=session_id,
        seq=seq,
        client_timestamp_ms=timestamp,
        received_monotonic_s=time.monotonic() if received_monotonic_s is None else received_monotonic_s,
        hands={side: _hand(hands.get(side)) for side in SIDES},
    )


class VRState:
    """Thread-safe latest-frame store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._activity = threading.Condition(self._lock)
        self._activity_generation = 0
        self._frame: VRFrame | None = None
        self._events: deque[VREvent] = deque()
        self._receive_session_id: str | None = None
        self._receive_last_seq: int | None = None
        self._receive_last_s: float | None = None
        self._receive_times: deque[float] = deque()
        self._receive_gap_ms: float | None = None
        self._receive_skipped_frames = 0
        self._receive_rejected_frames = 0

    def update(self, frame: VRFrame) -> bool:
        with self._activity:
            current = self._frame
            if current is not None and frame.session_id == current.session_id and frame.seq <= current.seq:
                self._receive_rejected_frames += 1
                return False
            self._track_receive(frame)
            self._frame = frame
            self._activity_generation += 1
            self._activity.notify_all()
            return True

    def latest(self) -> VRFrame | None:
        with self._lock:
            return self._frame

    def disconnect(self, session_id: str, last_seq: int | None = None) -> None:
        with self._activity:
            if (
                self._frame is not None
                and self._frame.session_id == session_id
                and (last_seq is None or self._frame.seq <= last_seq)
            ):
                self._frame = None
                self._activity_generation += 1
                self._activity.notify_all()

    def push_event(self, kind: str, received_monotonic_s: float | None = None) -> None:
        if kind not in ALLOWED_EVENTS:
            raise ValueError(f"unsupported VR event: {kind}")
        event = VREvent(kind, time.monotonic() if received_monotonic_s is None else received_monotonic_s)
        with self._activity:
            self._events.append(event)
            self._activity_generation += 1
            self._activity.notify_all()

    def drain_events(self) -> list[VREvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def activity_generation(self) -> int:
        with self._lock:
            return self._activity_generation

    def wait_for_activity(self, generation: int, timeout_s: float) -> int:
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("VR activity timeout must be non-negative and finite")
        with self._activity:
            self._activity.wait_for(
                lambda: self._activity_generation != generation,
                timeout=timeout_s,
            )
            return self._activity_generation

    def receive_diagnostics(self) -> dict[str, object]:
        with self._lock:
            receive_hz = None
            if len(self._receive_times) > 1:
                duration_s = self._receive_times[-1] - self._receive_times[0]
                if duration_s > 0.0:
                    receive_hz = (len(self._receive_times) - 1) / duration_s
            return {
                "session_id": self._receive_session_id,
                "receive_hz": receive_hz,
                "gap_ms": self._receive_gap_ms,
                "skipped_frames": self._receive_skipped_frames,
                "rejected_frames": self._receive_rejected_frames,
            }

    def _track_receive(self, frame: VRFrame) -> None:
        if frame.session_id != self._receive_session_id:
            self._receive_session_id = frame.session_id
            self._receive_last_seq = None
            self._receive_last_s = None
            self._receive_times.clear()
            self._receive_gap_ms = None
            self._receive_skipped_frames = 0
            self._receive_rejected_frames = 0

        if self._receive_last_seq is not None:
            self._receive_skipped_frames += max(
                0,
                frame.seq - self._receive_last_seq - 1,
            )
        self._receive_gap_ms = (
            None
            if self._receive_last_s is None
            else max(0.0, frame.received_monotonic_s - self._receive_last_s) * 1000.0
        )
        self._receive_last_seq = frame.seq
        self._receive_last_s = frame.received_monotonic_s
        self._receive_times.append(frame.received_monotonic_s)
        cutoff_s = frame.received_monotonic_s - 1.0
        while len(self._receive_times) > 1 and self._receive_times[0] < cutoff_s:
            self._receive_times.popleft()
