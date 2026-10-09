"""InferSystem protocol adapters for StarVLA deployment.

InferSystem's robot-side client sends a compact ZMQ/msgpack request:

    {"cmd": "predict", "state": [...], "prompt": "...", "<camera>": jpeg_bytes}

StarVLA's policy runtime expects the same normalized payload shape used by the
websocket server:

    {"batch_images": [[...]], "instructions": [...], "state": [...], "view_mask": [...]}
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


RESERVED_REQUEST_KEYS = {
    "cmd",
    "state",
    "prompt",
    "extra",
    "stat_key",
    "fps",
    "view_mask",
    "request_id",
    "step_idx",
    "actions",
    "delay",
}

PASSTHROUGH_KEYS = {
    "actions",
    "delay",
    "do_sample",
    "ee",
    "fps",
    "mode",
    "num_ddim_steps",
    "q",
    "request_id",
    "step_idx",
    "use_ddim",
    "view_mask",
}


class InferSystemProtocolError(ValueError):
    """Raised when an InferSystem request cannot be converted safely."""


@dataclass(frozen=True)
class InferSystemAdapterConfig:
    """Options controlling InferSystem -> StarVLA request conversion."""

    camera_order: tuple[str, ...]
    stat_key: str | None = None
    strict_cameras: bool = True
    image_color: str = "rgb"
    image_resize_size: tuple[int, int] | None = None
    default_fps: float | None = None

    def __post_init__(self) -> None:
        image_color = str(self.image_color).lower()
        if image_color not in {"rgb", "bgr"}:
            raise ValueError(f"image_color must be 'rgb' or 'bgr', got {self.image_color!r}.")
        resize_size = self.image_resize_size
        if resize_size is not None:
            if len(resize_size) != 2:
                raise ValueError(f"image_resize_size must be (height, width), got {resize_size!r}.")
            resize_size = (int(resize_size[0]), int(resize_size[1]))
            if resize_size[0] <= 0 or resize_size[1] <= 0:
                raise ValueError(f"image_resize_size must contain positive integers, got {resize_size!r}.")
        object.__setattr__(self, "image_color", image_color)
        object.__setattr__(self, "camera_order", tuple(self.camera_order))
        object.__setattr__(self, "image_resize_size", resize_size)


def parse_camera_order(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse comma-separated camera order values."""

    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    order = tuple(str(part).strip() for part in parts if str(part).strip())
    if len(set(order)) != len(order):
        raise InferSystemProtocolError(f"camera_order contains duplicate entries: {order!r}.")
    return order


def decode_image_bytes(
    buf: bytes | bytearray,
    *,
    image_color: str = "rgb",
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode image bytes to a uint8 HWC array.

    JPEG bytes produced by OpenCV encode valid image colors; decoding through
    PIL returns RGB, which is the convention used by StarVLA image preprocessors.
    """

    try:
        image = Image.open(BytesIO(bytes(buf))).convert("RGB")
        if resize_size is not None:
            target_h, target_w = int(resize_size[0]), int(resize_size[1])
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    except Exception as exc:  # pragma: no cover - exact PIL exception varies
        raise InferSystemProtocolError(f"failed to decode image bytes: {exc}") from exc

    arr = np.asarray(image, dtype=np.uint8)
    if image_color == "bgr":
        arr = arr[..., ::-1].copy()
    return arr


def _decode_camera_images(
    msg: Mapping[str, Any],
    *,
    image_color: str,
    resize_size: tuple[int, int] | None,
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for key, value in msg.items():
        if key in RESERVED_REQUEST_KEYS:
            continue
        if not isinstance(value, (bytes, bytearray)):
            continue
        images[str(key)] = decode_image_bytes(value, image_color=image_color, resize_size=resize_size)
    return images


def _select_images(
    decoded_images: Mapping[str, np.ndarray],
    *,
    camera_order: Sequence[str],
    strict_cameras: bool,
) -> tuple[list[np.ndarray], list[bool], tuple[str, ...]]:
    if camera_order:
        order = tuple(camera_order)
    else:
        order = tuple(sorted(decoded_images))

    if not order:
        raise InferSystemProtocolError("no camera images found and camera_order is empty.")

    missing = [name for name in order if name not in decoded_images]
    unexpected = [name for name in decoded_images if name not in set(order)]
    if strict_cameras and missing:
        raise InferSystemProtocolError(f"missing camera image(s): {missing}; expected order={order}.")
    if strict_cameras and unexpected:
        raise InferSystemProtocolError(f"unexpected camera image(s): {unexpected}; expected order={order}.")

    selected: list[np.ndarray] = []
    view_mask: list[bool] = []
    for name in order:
        if name not in decoded_images:
            raise InferSystemProtocolError(f"missing camera image {name!r}; expected order={order}.")
        selected.append(decoded_images[name])
        view_mask.append(True)
    return selected, view_mask, order


def _coerce_state(value: Any) -> np.ndarray:
    state = np.asarray([] if value is None else value, dtype=np.float32)
    if state.ndim == 0:
        raise InferSystemProtocolError("state must be a 1-D or 2-D numeric array, got scalar.")
    if state.ndim > 2:
        raise InferSystemProtocolError(f"state must be a 1-D or 2-D numeric array, got shape {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise InferSystemProtocolError("state contains NaN or Inf.")
    return state


def _extra_dict(msg: Mapping[str, Any]) -> dict[str, Any]:
    extra = msg.get("extra")
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise InferSystemProtocolError(f"extra must be a mapping when present, got {type(extra)!r}.")
    return dict(extra)


def infersystem_to_starvla_payload(
    msg: Mapping[str, Any],
    config: InferSystemAdapterConfig,
) -> dict[str, Any]:
    """Convert one InferSystem predict request to a StarVLA policy payload."""

    if not isinstance(msg, Mapping):
        raise InferSystemProtocolError(f"request must be a mapping, got {type(msg)!r}.")

    cmd = str(msg.get("cmd", "predict"))
    if cmd != "predict":
        raise InferSystemProtocolError(f"expected cmd='predict', got {cmd!r}.")

    extra = _extra_dict(msg)
    decoded_images = _decode_camera_images(
        msg,
        image_color=config.image_color,
        resize_size=config.image_resize_size,
    )
    images, view_mask, _resolved_order = _select_images(
        decoded_images,
        camera_order=config.camera_order,
        strict_cameras=config.strict_cameras,
    )

    prompt = msg.get("prompt", extra.get("prompt", ""))
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise InferSystemProtocolError(f"prompt must be a string, got {type(prompt)!r}.")

    payload: dict[str, Any] = {
        "batch_images": [images],
        "instructions": [prompt],
        "state": [_coerce_state(msg.get("state", extra.get("state", [])))],
        "view_mask": [extra.get("view_mask", msg.get("view_mask", view_mask))],
    }

    stat_key = msg.get("stat_key", extra.get("stat_key", config.stat_key))
    if stat_key is not None:
        payload["stat_key"] = str(stat_key)

    fps = msg.get("fps", extra.get("fps", config.default_fps))
    if fps is not None:
        payload["fps"] = [float(fps)]

    for key in PASSTHROUGH_KEYS:
        if key in payload or key in {"fps", "view_mask"}:
            continue
        if key in msg:
            payload[key] = msg[key]
        elif key in extra:
            payload[key] = extra[key]

    return payload


def extract_action_chunk(output: Mapping[str, Any]) -> np.ndarray:
    """Extract a single action chunk from StarVLA policy output."""

    if "actions" in output:
        actions = np.asarray(output["actions"], dtype=np.float32)
    elif "normalized_actions" in output:
        actions = np.asarray(output["normalized_actions"], dtype=np.float32)
    else:
        raise InferSystemProtocolError(f"policy output has no actions key; available keys={list(output.keys())}.")

    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise InferSystemProtocolError(f"expected batch size 1 in actions, got shape {actions.shape}.")
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim != 2:
        raise InferSystemProtocolError(f"expected actions shape (T,D), (1,T,D), or (D,), got {actions.shape}.")
    return np.asarray(actions, dtype=np.float32)


def msgpack_safe(value: Any) -> Any:
    """Convert numpy-rich values into plain msgpack-compatible Python values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): msgpack_safe(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [msgpack_safe(item) for item in value]
    if isinstance(value, list):
        return [msgpack_safe(item) for item in value]
    return value
