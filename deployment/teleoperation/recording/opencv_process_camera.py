from __future__ import annotations

import logging
import math
import multiprocessing as mp
import platform
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CameraConfig

logger = logging.getLogger(__name__)
OPENCV_OPEN_TIMEOUT_S = 5.0
OPENCV_STOP_TIMEOUT_S = 2.0
OPENCV_ASYNC_READ_TIMEOUT_MS = 200.0
OPENCV_STARTUP_READ_ATTEMPTS = 30
OPENCV_STARTUP_RETRY_DELAY_S = 0.05
OPENCV_MAX_CONSECUTIVE_READ_FAILURES = 3


def _configure_opencv_capture(
    capture: Any,
    cv2_module: Any,
    *,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None,
) -> None:
    """Configure a V4L2 stream before its first read.

    UVC cameras choose their transport format while the stream is configured.
    Set the compressed format first, matching Collector's working camera
    path, so width/FPS negotiation does not silently settle on uncompressed YUYV.
    """
    if fourcc is not None:
        capture.set(cv2_module.CAP_PROP_FOURCC, cv2_module.VideoWriter_fourcc(*fourcc))
    capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2_module.CAP_PROP_FPS, fps)


def _fourcc_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * offset)) & 0xFF) for offset in range(4)).rstrip("\x00")


def _publish_frame(
    frame: np.ndarray,
    *,
    color_mode: str,
    expected_shape: tuple[int, int, int],
    frame_buffer: Any,
    frame_lock: Any,
    frame_event: Any,
    cv2_module: Any,
) -> None:
    if frame.shape != expected_shape:
        raise RuntimeError(
            f"OpenCV camera returned shape={frame.shape}, expected={expected_shape}"
        )
    if frame.dtype != np.uint8:
        raise RuntimeError(f"OpenCV camera returned dtype={frame.dtype}, expected=uint8")
    output = (
        cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        if color_mode == "rgb"
        else frame
    )
    with frame_lock:
        shared = np.frombuffer(frame_buffer, dtype=np.uint8).reshape(expected_shape)
        np.copyto(shared, output)
        frame_event.set()


def _opencv_capture_worker(
    *,
    source: int | str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    fourcc: str | None,
    frame_buffer: Any,
    frame_lock: Any,
    frame_event: Any,
    stop_event: Any,
    status_queue: Any,
) -> None:
    capture = None
    try:
        import cv2

        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open OpenCV camera {source}")

        _configure_opencv_capture(
            capture,
            cv2,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )

        expected_shape = (height, width, 3)
        first_frame = None
        for _ in range(OPENCV_STARTUP_READ_ATTEMPTS):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                first_frame = candidate
                break
            if stop_event.wait(OPENCV_STARTUP_RETRY_DELAY_S):
                return
        if first_frame is None:
            raise RuntimeError(f"OpenCV camera {source} opened but did not produce a frame")

        _publish_frame(
            first_frame,
            color_mode=color_mode,
            expected_shape=expected_shape,
            frame_buffer=frame_buffer,
            frame_lock=frame_lock,
            frame_event=frame_event,
            cv2_module=cv2,
        )
        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        actual_fourcc = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
        status_queue.put(
            ("ready", actual_width, actual_height, actual_fps, actual_fourcc)
        )

        consecutive_failures = 0
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= OPENCV_MAX_CONSECUTIVE_READ_FAILURES:
                    raise RuntimeError(
                        f"OpenCV camera {source} failed "
                        f"{consecutive_failures} consecutive reads"
                    )
                continue
            consecutive_failures = 0
            _publish_frame(
                frame,
                color_mode=color_mode,
                expected_shape=expected_shape,
                frame_buffer=frame_buffer,
                frame_lock=frame_lock,
                frame_event=frame_event,
                cv2_module=cv2,
            )
    except BaseException as exc:
        status_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if capture is not None:
            capture.release()


class ProcessOpenCVCamera:
    """OpenCV camera isolated in a killable process with a latest-frame slot."""

    def __init__(self, config: CameraConfig):
        if config.index_or_path is None:
            raise ValueError("OpenCV camera requires index_or_path")
        self.config = config
        source = config.index_or_path
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        elif isinstance(source, (str, Path)):
            # Collector resolves stable /dev/v4l/by-path links to their concrete
            # /dev/videoN node before opening with CAP_V4L2. Do the same here.
            source = str(Path(source).resolve())
        self.source: int | str = source
        self._context = mp.get_context("spawn")
        self._process: Any | None = None
        self._frame_buffer: Any | None = None
        self._frame_lock: Any | None = None
        self._frame_event: Any | None = None
        self._stop_event: Any | None = None
        self._status_queue: Any | None = None
        self._connected = False
        self._last_error: str | None = None

    def __str__(self) -> str:
        return f"ProcessOpenCVCamera({self.source})"

    @property
    def is_connected(self) -> bool:
        # Keep this true until disconnect() performs parent-side cleanup, even
        # when the worker has exited after reporting a capture error.
        return self._connected

    def connect(self) -> None:
        if self._connected or self._process is not None:
            raise RuntimeError(f"{self} is already connected")

        frame_size = self.config.width * self.config.height * 3
        self._frame_buffer = self._context.RawArray("B", frame_size)
        self._frame_lock = self._context.Lock()
        self._frame_event = self._context.Event()
        self._stop_event = self._context.Event()
        self._status_queue = self._context.Queue()
        self._last_error = None
        self._process = self._context.Process(
            target=_opencv_capture_worker,
            kwargs={
                "source": self.source,
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "color_mode": self.config.color_mode,
                "fourcc": self.config.fourcc,
                "frame_buffer": self._frame_buffer,
                "frame_lock": self._frame_lock,
                "frame_event": self._frame_event,
                "stop_event": self._stop_event,
                "status_queue": self._status_queue,
            },
            daemon=True,
            name=f"opencv-camera-{self.source}",
        )
        self._process.start()

        deadline = time.monotonic() + OPENCV_OPEN_TIMEOUT_S
        while time.monotonic() < deadline:
            message = self._next_status(timeout_s=min(0.1, deadline - time.monotonic()))
            if message is None:
                if not self._process.is_alive():
                    break
                continue
            if message[0] == "error":
                self._last_error = str(message[1])
                self._cleanup_process()
                raise RuntimeError(f"failed to connect {self}: {self._last_error}")
            if message[0] == "ready":
                _, actual_width, actual_height, actual_fps, actual_fourcc = message
                self._connected = True
                if (actual_width, actual_height) != (self.config.width, self.config.height):
                    self.disconnect()
                    raise RuntimeError(
                        f"{self} negotiated {actual_width}x{actual_height}, expected "
                        f"{self.config.width}x{self.config.height}"
                    )
                if self.config.fourcc is not None and actual_fourcc != self.config.fourcc:
                    self.disconnect()
                    raise RuntimeError(
                        f"{self} negotiated fourcc={actual_fourcc!r}, "
                        f"expected={self.config.fourcc!r}"
                    )
                if actual_fps <= 0 or not math.isclose(actual_fps, self.config.fps, rel_tol=0.01):
                    self.disconnect()
                    raise RuntimeError(
                        f"{self} negotiated fps={actual_fps:.3f}, expected={self.config.fps}"
                    )
                logger.info(
                    "%s connected: %dx%d@%.1ffps fourcc=%s worker_pid=%s",
                    self,
                    actual_width,
                    actual_height,
                    actual_fps,
                    actual_fourcc or "unknown",
                    self._process.pid,
                )
                return

        self._cleanup_process()
        detail = f": {self._last_error}" if self._last_error else ""
        raise TimeoutError(
            f"timed out waiting {OPENCV_OPEN_TIMEOUT_S:.1f}s for the first frame from "
            f"{self}; ensure no other process owns the device{detail}"
        )

    def async_read(self, timeout_ms: float = OPENCV_ASYNC_READ_TIMEOUT_MS) -> np.ndarray:
        if not self._connected or self._process is None:
            raise RuntimeError(f"{self} is not connected")
        if not self._process.is_alive():
            self._consume_status_errors()
            detail = f": {self._last_error}" if self._last_error else ""
            raise RuntimeError(f"capture process for {self} exited{detail}")
        assert self._frame_event is not None
        if not self._frame_event.wait(timeout_ms / 1000.0):
            self._consume_status_errors()
            if not self._process.is_alive():
                detail = f": {self._last_error}" if self._last_error else ""
                raise RuntimeError(f"capture process for {self} exited{detail}")
            raise TimeoutError(f"timed out waiting {timeout_ms:.0f}ms for a frame from {self}")

        assert self._frame_lock is not None
        assert self._frame_buffer is not None
        with self._frame_lock:
            self._frame_event.clear()
            shape = (self.config.height, self.config.width, 3)
            return np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(shape).copy()

    def disconnect(self) -> None:
        if self._process is None and not self._connected:
            return
        self._cleanup_process()

    def _next_status(self, *, timeout_s: float) -> tuple[Any, ...] | None:
        assert self._status_queue is not None
        try:
            return self._status_queue.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None

    def _consume_status_errors(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                return
            if message and message[0] == "error":
                self._last_error = str(message[1])

    def _cleanup_process(self) -> None:
        process = self._process
        if self._stop_event is not None:
            self._stop_event.set()
        if process is not None:
            process.join(timeout=OPENCV_STOP_TIMEOUT_S)
            if process.is_alive():
                logger.warning("%s read remained blocked during shutdown; terminating worker", self)
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        self._consume_status_errors()
        if self._status_queue is not None:
            self._status_queue.close()
        self._process = None
        self._frame_buffer = None
        self._frame_lock = None
        self._frame_event = None
        self._stop_event = None
        self._status_queue = None
        self._connected = False
        if process is not None:
            logger.info("%s disconnected", self)
