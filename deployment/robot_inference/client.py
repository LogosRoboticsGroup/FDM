from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Mapping

import msgpack
import numpy as np
import zmq
from PIL import Image

logger = logging.getLogger(__name__)


class PolicyClientError(RuntimeError):
    """Raised when an InferSystem-compatible request or response is invalid."""


class PolicyRequestCancelled(Exception):
    """An operator stopped a synchronous request before any further dispatch."""


@dataclass(frozen=True)
class PolicyReply:
    actions: np.ndarray
    infer_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StarVLAZmqClient:
    """Robot-side client for StarVLA's ZMQ/msgpack policy endpoint."""

    def __init__(
        self,
        server: str,
        *,
        camera_order: tuple[str, ...],
        jpeg_quality: int = 90,
        recv_timeout_ms: int = 60_000,
        send_timeout_ms: int = 5_000,
        max_retries: int = 3,
    ) -> None:
        self.server = _normalize_server(server)
        self.camera_order = tuple(camera_order)
        self.jpeg_quality = int(jpeg_quality)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.send_timeout_ms = int(send_timeout_ms)
        self.max_retries = int(max_retries)
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    @property
    def is_connected(self) -> bool:
        return self._context is not None and self._socket is not None

    def connect(self) -> None:
        if self.is_connected:
            return
        self._context = zmq.Context()
        self._create_socket()
        logger.info("Connected StarVLA policy client to %s", self.server)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def metadata(self) -> dict[str, Any]:
        response = self._request({"cmd": "metadata"})
        metadata = response.get("metadata")
        if not isinstance(metadata, Mapping):
            raise PolicyClientError("metadata response does not contain a mapping")
        return dict(metadata)

    def reset(self, *, cancelled: Callable[[], bool] | None = None) -> None:
        self._request({"cmd": "reset"}, cancelled=cancelled)

    def predict(
        self,
        images_rgb: Mapping[str, np.ndarray],
        state: np.ndarray | list[float] | tuple[float, ...],
        *,
        prompt: str,
        fps: float,
        stat_key: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> PolicyReply:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim != 1 or not np.all(np.isfinite(state_array)):
            raise PolicyClientError(f"state must be one finite vector, got shape={state_array.shape}")
        missing = [name for name in self.camera_order if name not in images_rgb]
        unexpected = [name for name in images_rgb if name not in set(self.camera_order)]
        if missing or unexpected:
            raise PolicyClientError(f"camera set mismatch: missing={missing}, unexpected={unexpected}")

        payload: dict[str, Any] = {
            "cmd": "predict",
            "state": state_array.tolist(),
            "prompt": prompt,
            "fps": float(fps),
        }
        if stat_key is not None:
            payload["stat_key"] = stat_key
        for name in self.camera_order:
            payload[name] = _encode_rgb_jpeg(images_rgb[name], self.jpeg_quality)

        response = self._request(payload, cancelled=cancelled)
        try:
            actions = np.asarray(response["actions"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyClientError("predict response has invalid or missing actions") from exc
        if actions.ndim != 2 or actions.shape[0] == 0 or not np.all(np.isfinite(actions)):
            raise PolicyClientError(f"actions must be a non-empty finite (T,D) array, got shape={actions.shape}")
        infer_time = response.get("infer_time_ms")
        if infer_time is not None:
            infer_time = float(infer_time)
        metadata = response.get("metadata")
        return PolicyReply(
            actions=actions,
            infer_time_ms=infer_time,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def _create_socket(self) -> None:
        if self._context is None:
            raise RuntimeError("client context has not been created")
        if self._socket is not None:
            self._socket.close()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.server)

    def _wait_socket(self, flag: int, timeout_ms: int, cancelled: Callable[[], bool] | None) -> None:
        assert self._socket is not None
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            if cancelled is not None and cancelled():
                raise PolicyRequestCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise zmq.Again()
            if self._socket.poll(timeout=min(50, max(1, int(remaining * 1000))), flags=flag):
                return

    def _request(self, payload: dict[str, Any], *, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("StarVLA policy client is not connected")
        last_error: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                assert self._socket is not None
                packed = msgpack.packb(payload, use_bin_type=True)
                self._wait_socket(zmq.POLLOUT, self.send_timeout_ms, cancelled)
                self._socket.send(packed, flags=zmq.NOBLOCK)
                self._wait_socket(zmq.POLLIN, self.recv_timeout_ms, cancelled)
                response = msgpack.unpackb(self._socket.recv(flags=zmq.NOBLOCK), raw=False, strict_map_key=False)
                if not isinstance(response, dict):
                    raise PolicyClientError(f"server response must be a mapping, got {type(response).__name__}")
                if response.get("status") != "ok":
                    raise PolicyClientError(
                        f"server rejected {payload.get('cmd', 'predict')!r}: "
                        f"{response.get('message', 'unknown error')}"
                    )
                return response
            except PolicyRequestCancelled:
                # REQ sockets cannot send again before receiving. Reconnect so
                # the late response to a cancelled chunk can never be reused.
                self._create_socket()
                raise
            except PolicyClientError:
                raise
            except (zmq.Again, zmq.ZMQError) as exc:
                last_error = exc
                logger.warning(
                    "StarVLA policy request failed (%d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                self._create_socket()
        raise PolicyClientError(f"request failed after {self.max_retries} attempts: {last_error}")


def _normalize_server(value: str) -> str:
    value = value.strip()
    return value if value.startswith("tcp://") else f"tcp://{value}"


def _encode_rgb_jpeg(image: np.ndarray, quality: int) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PolicyClientError(f"camera image must have shape (H,W,3), got {array.shape}")
    if array.dtype != np.uint8:
        if not np.all(np.isfinite(array)):
            raise PolicyClientError("camera image contains NaN or Inf")
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(np.ascontiguousarray(array), mode="RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
    )
    return buffer.getvalue()
