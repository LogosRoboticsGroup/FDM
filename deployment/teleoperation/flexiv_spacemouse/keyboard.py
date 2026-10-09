from __future__ import annotations

from typing import ClassVar

from ..control.keyboard import KeyboardController


class RecordingKeyboardController(KeyboardController):
    """SpaceMouse recording bindings on the shared terminal/pynput listener."""

    KEYS: ClassVar[dict[str, str]] = {
        **KeyboardController.KEYS,
        "\x1b[D": "discard_episode_and_home",
        "\x1b[C": "save_episode_and_home",
        "\x1bOD": "discard_episode_and_home",
        "\x1bOC": "save_episode_and_home",
    }
    HELP = "Left=discard+home Right=save+home q/Esc=quit"
