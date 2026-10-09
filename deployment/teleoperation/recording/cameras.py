from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import CameraConfig
from .opencv_process_camera import ProcessOpenCVCamera

logger = logging.getLogger(__name__)
CAMERA_READY_TIMEOUT_S = 2.0
CAMERA_STALE_TIMEOUT_S = 1.0
OPENCV_CONNECT_ATTEMPTS = 3
OPENCV_CONNECT_RETRY_DELAY_S = 0.5


class CameraHub:
    def __init__(self, configs: dict[str, CameraConfig]):
        self.configs = configs
        self.cameras = self._make_cameras(configs) if configs else {}
        self._frames: dict[str, np.ndarray] = {}
        self._frame_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._capture_error: BaseException | None = None
        self._last_frame_monotonic_s: float | None = None

    @staticmethod
    def _make_cameras(configs: dict[str, CameraConfig]):
        from lerobot.cameras.configs import ColorMode
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.cameras.utils import make_cameras_from_configs

        opencv_cameras = {}
        lerobot_configs = {}
        for name, config in configs.items():
            common = dict(
                fps=config.fps,
                width=config.width,
                height=config.height,
                color_mode=ColorMode(config.color_mode),
            )
            if config.type == "opencv":
                opencv_cameras[name] = ProcessOpenCVCamera(config)
            elif config.type == "intelrealsense":
                if not config.serial_number_or_name:
                    raise ValueError(f"camera {name} requires serial_number_or_name")
                lerobot_configs[name] = RealSenseCameraConfig(
                    serial_number_or_name=config.serial_number_or_name,
                    **common,
                )
            else:
                raise ValueError(f"unsupported camera type: {config.type}")
        lerobot_cameras = make_cameras_from_configs(lerobot_configs)
        return {
            name: opencv_cameras[name] if name in opencv_cameras else lerobot_cameras[name]
            for name in configs
        }

    def connect(self) -> None:
        connected = []
        try:
            for camera in self.cameras.values():
                self._connect_camera(camera)
                connected.append(camera)
            if self.cameras:
                self._start_capture_thread()
        except BaseException:
            self._stop_capture_thread()
            for camera in reversed(connected):
                camera.disconnect()
            raise

    @staticmethod
    def _connect_camera(camera) -> None:
        attempts = OPENCV_CONNECT_ATTEMPTS if isinstance(camera, ProcessOpenCVCamera) else 1
        for attempt in range(1, attempts + 1):
            try:
                camera.connect()
                return
            except BaseException:
                if attempt == attempts:
                    raise
                logger.warning(
                    "OpenCV camera connection attempt %d/%d failed; releasing and retrying: %s",
                    attempt,
                    attempts,
                    camera,
                    exc_info=True,
                )
                time.sleep(OPENCV_CONNECT_RETRY_DELAY_S)

    def read(self) -> dict[str, np.ndarray]:
        if not self.cameras:
            return {}
        self._raise_if_capture_failed()
        with self._frame_lock:
            if set(self._frames) != set(self.cameras):
                raise RuntimeError("camera capture has not produced a complete frame set")
            assert self._last_frame_monotonic_s is not None
            age_s = time.monotonic() - self._last_frame_monotonic_s
            if age_s > CAMERA_STALE_TIMEOUT_S:
                raise RuntimeError(
                    f"latest complete camera frame set is stale: age={age_s:.3f}s"
                )
            return dict(self._frames)

    def close(self) -> None:
        self._stop_capture_thread()
        for camera in reversed(tuple(self.cameras.values())):
            if camera.is_connected:
                camera.disconnect()

    def _start_capture_thread(self) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        self._frames = {}
        self._capture_error = None
        self._last_frame_monotonic_s = None
        self._ready.clear()
        self._stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-hub-capture",
            daemon=True,
        )
        self._capture_thread.start()
        if not self._ready.wait(CAMERA_READY_TIMEOUT_S):
            raise TimeoutError(
                f"timed out waiting {CAMERA_READY_TIMEOUT_S:.1f}s for the first camera frame set"
            )
        self._raise_if_capture_failed()

    def _stop_capture_thread(self) -> None:
        self._stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=CAMERA_READY_TIMEOUT_S)
            if self._capture_thread.is_alive():
                logger.warning("Camera capture thread did not stop before disconnect")
            self._capture_thread = None

    def _capture_loop(self) -> None:
        consecutive_timeouts = 0
        try:
            while not self._stop.is_set():
                try:
                    frames = {
                        name: np.asarray(camera.async_read()).copy()
                        for name, camera in self.cameras.items()
                    }
                except TimeoutError as exc:
                    if self._stop.is_set():
                        return
                    consecutive_timeouts += 1
                    if consecutive_timeouts == 1 or consecutive_timeouts % 30 == 0:
                        logger.warning(
                            "Camera capture timed out; keeping the last complete frame set "
                            "and retrying (consecutive_timeouts=%d): %s",
                            consecutive_timeouts,
                            exc,
                        )
                    continue
                if consecutive_timeouts:
                    logger.info(
                        "Camera capture recovered after %d consecutive timeouts",
                        consecutive_timeouts,
                    )
                    consecutive_timeouts = 0
                with self._frame_lock:
                    self._frames = frames
                    self._last_frame_monotonic_s = time.monotonic()
                self._ready.set()
        except BaseException as exc:
            if not self._stop.is_set():
                self._capture_error = exc
                logger.exception("Camera capture thread failed")
            self._ready.set()

    def _raise_if_capture_failed(self) -> None:
        if self._capture_error is not None:
            raise RuntimeError(f"camera capture failed: {self._capture_error}") from self._capture_error
