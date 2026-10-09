from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..contracts import CommandResult


class RecorderError(RuntimeError):
    pass


class RecorderTimeoutError(RecorderError):
    """A recorder request exceeded its wait limit but may still be running."""


class Recorder(ABC):
    def observe_commands(self, result: CommandResult) -> None:
        """Track held commands independently of the recording cadence."""

    @abstractmethod
    def start_episode(self) -> None: ...

    @abstractmethod
    def add_frame(self, result: CommandResult, images: dict[str, np.ndarray]) -> None: ...

    @abstractmethod
    def finish_episode(self, save: bool = True) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class NullRecorder(Recorder):
    def start_episode(self) -> None:
        return None

    def add_frame(self, result: CommandResult, images: dict[str, np.ndarray]) -> None:
        return None

    def finish_episode(self, save: bool = True) -> None:
        return None

    def close(self) -> None:
        return None
