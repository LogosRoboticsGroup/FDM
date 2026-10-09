from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import tty
from typing import ClassVar

from .events import EventQueue

logger = logging.getLogger(__name__)


class KeyboardController:
    KEYS: ClassVar[dict[str, str]] = {
        "q": "quit",
        "\x1b": "quit",
    }
    HELP = "q/Esc=quit"
    FLUSH_ON_START = False

    def __init__(self, events: EventQueue):
        self.events = events
        self._listener = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._terminal_fd: int | None = None
        self._terminal_attrs: list | None = None

    def start(self) -> None:
        if sys.stdin.isatty():
            self._start_terminal()
            return

        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError("keyboard input requires an interactive TTY or pynput") from exc

        def on_press(key):
            special = {
                keyboard.Key.esc: "\x1b",
                keyboard.Key.enter: "\r",
                keyboard.Key.left: "\x1b[D",
                keyboard.Key.right: "\x1b[C",
            }
            self._handle_key(special.get(key, getattr(key, "char", None)))

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        logger.info("Keyboard: %s", self.HELP)

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=1.0)
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._terminal_fd is not None and self._terminal_attrs is not None:
            termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_attrs)
        self._terminal_fd = None
        self._terminal_attrs = None

    def _start_terminal(self) -> None:
        self._stop.clear()
        self._terminal_fd = sys.stdin.fileno()
        self._terminal_attrs = termios.tcgetattr(self._terminal_fd)
        if self.FLUSH_ON_START:
            termios.tcflush(self._terminal_fd, termios.TCIFLUSH)
        tty.setcbreak(self._terminal_fd)
        self._thread = threading.Thread(target=self._read_terminal, name="teleop-keyboard", daemon=True)
        self._thread.start()
        logger.info("Keyboard: %s", self.HELP)

    def _read_terminal(self) -> None:
        assert self._terminal_fd is not None
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._terminal_fd], [], [], 0.1)
                if not readable:
                    continue
                data = os.read(self._terminal_fd, 1)
                if data == b"\x1b":
                    data += self._read_escape_suffix()
            except OSError:
                break
            if not data:
                break
            self._handle_bytes(data)

    def _read_escape_suffix(self) -> bytes:
        assert self._terminal_fd is not None
        suffix = b""
        for _ in range(2):
            readable, _, _ = select.select([self._terminal_fd], [], [], 0.03)
            if not readable:
                break
            suffix += os.read(self._terminal_fd, 1)
        return suffix

    def _handle_bytes(self, data: bytes) -> None:
        self._handle_key(data.decode(errors="ignore"))

    def _handle_key(self, char: str | None) -> None:
        if char in self.KEYS:
            self.events.push(self.KEYS[char], "keyboard")
