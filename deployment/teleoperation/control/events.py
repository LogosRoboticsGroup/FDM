from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OperatorEvent:
    kind: str
    source: str
    received_monotonic_s: float = field(default_factory=time.monotonic)


class EventQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[OperatorEvent] = deque()

    def push(self, kind: str, source: str) -> None:
        with self._lock:
            self._events.append(OperatorEvent(kind, source))

    def drain(self) -> list[OperatorEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events
