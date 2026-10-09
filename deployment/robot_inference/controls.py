from __future__ import annotations

from typing import ClassVar

from deployment.teleoperation.config import VRConfig
from deployment.teleoperation.control import EventQueue, KeyboardController, OperatorEvent


class InferenceKeyboardController(KeyboardController):
    KEYS: ClassVar[dict[str, str]] = {
        **KeyboardController.KEYS,
        "\r": "start_episode",
        "\n": "start_episode",
        "\x1b[D": "discard_episode_and_home",
        "\x1bOD": "discard_episode_and_home",
        "\x1b[C": "save_episode_and_home",
        "\x1bOC": "save_episode_and_home",
    }
    HELP = "Enter=start Left=discard+home Right=save+home q/Esc=quit"
    FLUSH_ON_START = True


class InferenceControls:
    """Adapt keyboard and existing WebXR events to the same episode controls."""

    STOP_EVENTS: ClassVar[set[str]] = {"quit", "save_episode", "save_episode_and_home", "discard_episode_and_home"}

    def __init__(self, mode: str, vr_config: VRConfig, *, keyboard: bool = True, start_vr_server: bool = True) -> None:
        self.events = EventQueue()
        keyboard_type = InferenceKeyboardController if mode == "keyboard" else KeyboardController
        self.keyboard = keyboard_type(self.events) if keyboard else None
        self.vr_state = None
        self.vr_server = None
        self._pending: list[OperatorEvent] = []
        if mode == "vr":
            from deployment.teleoperation.vr import VRState, WebXRServer

            self.vr_state = VRState()
            if start_vr_server:
                self.vr_server = WebXRServer(vr_config, self.vr_state)

    def start(self) -> None:
        if self.vr_server is not None:
            self.vr_server.start()
        if self.keyboard is not None:
            self.keyboard.start()

    def close(self) -> None:
        try:
            if self.keyboard is not None:
                self.keyboard.close()
        finally:
            if self.vr_server is not None:
                self.vr_server.stop()

    def _collect(self) -> None:
        self._pending.extend(self.events.drain())
        if self.vr_state is not None:
            self._pending.extend(
                OperatorEvent(event.kind, "vr", event.received_monotonic_s)
                for event in self.vr_state.drain_events()
                if event.kind in {"start_episode", "save_episode_and_home", "discard_episode_and_home"}
            )

    def interrupted(self) -> bool:
        self._collect()
        return any(event.kind in self.STOP_EVENTS for event in self._pending)

    def drain(self) -> list[OperatorEvent]:
        self._collect()
        events, self._pending = self._pending, []
        return sorted(events, key=lambda event: event.received_monotonic_s)
