# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Serve StarVLA checkpoints through the InferSystem ZMQ protocol."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import time
from typing import Any

import msgpack
import numpy as np
import torch
import zmq

from deployment.model_server.infersystem_protocol import (
    InferSystemAdapterConfig,
    InferSystemProtocolError,
    decode_image_bytes,
    extract_action_chunk,
    infersystem_to_starvla_payload,
    msgpack_safe,
    parse_camera_order,
)
from deployment.model_server.inference_loader import load_framework_for_inference


logger = logging.getLogger(__name__)


class InferSystemPolicyServer:
    """ZMQ REP server that adapts InferSystem requests to StarVLA policies."""

    def __init__(
        self,
        policy,
        *,
        bind: str,
        adapter_config: InferSystemAdapterConfig,
        log_timing_every: int = 0,
        log_actions_every: int = 0,
        save_generated_video_dir: str | None = None,
        save_generated_video_every: int = 1,
        save_generated_video_fps: float | None = None,
    ) -> None:
        self._policy = policy
        self._bind = bind
        self._adapter_config = adapter_config
        self._log_timing_every = int(log_timing_every)
        self._log_actions_every = max(0, int(log_actions_every))
        self._infer_count = 0
        self._save_generated_video_dir = Path(save_generated_video_dir) if save_generated_video_dir else None
        self._save_generated_video_every = max(1, int(save_generated_video_every))
        self._save_generated_video_fps = None if save_generated_video_fps is None else float(save_generated_video_fps)
        if self._save_generated_video_dir is not None:
            self._save_generated_video_dir.mkdir(parents=True, exist_ok=True)
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def run(self) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind)
        logger.info("InferSystem policy server listening on %s", self._bind)

        try:
            while True:
                raw = self._socket.recv()
                reply = self._handle_raw_request(raw)
                self._socket.send(msgpack.packb(msgpack_safe(reply), use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("InferSystem policy server interrupted; shutting down.")
        finally:
            self.close()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def _handle_raw_request(self, raw: bytes) -> dict[str, Any]:
        try:
            msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except Exception as exc:
            logger.exception("Failed to decode InferSystem msgpack request.")
            return self._error(f"msgpack decode failed: {exc}")

        if not isinstance(msg, dict):
            return self._error(f"request must be a dict, got {type(msg)!r}.")

        cmd = str(msg.get("cmd", "predict"))
        if cmd == "predict":
            return self._handle_predict(msg)
        if cmd == "reset":
            return self._handle_reset(msg)
        if cmd == "metadata":
            return {"status": "ok", "metadata": self._metadata()}
        return self._error(f"unknown cmd: {cmd}")

    def _handle_predict(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = infersystem_to_starvla_payload(msg, self._adapter_config)
            output, timing = self._run_infer_path(payload)
            actions = extract_action_chunk(output)
            save_stem = self._build_save_stem(msg, self._infer_count + 1)
            input_frame_paths = self._maybe_save_input_frames(
                msg, save_stem, self._infer_count + 1, payload["batch_images"][0]
            )
            generated_video_path = self._maybe_save_generated_video(output, msg, save_stem, self._infer_count + 1)
        except InferSystemProtocolError as exc:
            logger.warning("InferSystem request rejected: %s", exc)
            return self._error(str(exc))
        except Exception as exc:
            logger.exception("StarVLA inference failed.")
            return self._error(str(exc))

        self._infer_count += 1
        if self._log_timing_every > 0 and self._infer_count % self._log_timing_every == 0:
            logger.info(
                "TIMING infer_count=%d preprocess_ms=%.3f predict_ms=%.3f postprocess_ms=%.3f infer_time_ms=%.3f",
                self._infer_count,
                float(timing["preprocess_ms"]),
                float(timing["predict_ms"]),
                float(timing["postprocess_ms"]),
                float(timing["infer_time_ms"]),
            )
        if self._log_actions_every > 0 and self._infer_count % self._log_actions_every == 0:
            logger.info(
                "ACTION infer_count=%d shape=%s values=%s",
                self._infer_count,
                tuple(actions.shape),
                np.array2string(actions, precision=6, separator=", ", threshold=20000, max_line_width=200),
            )

        metadata = {
            "action_shape": list(actions.shape),
            "stat_key": payload.get("stat_key", getattr(self._policy, "stat_key", None)),
            "camera_order": list(self._adapter_config.camera_order),
        }
        for name, path in input_frame_paths.items():
            metadata[f"{name}_input_frame_path"] = str(path)
        if generated_video_path is not None:
            metadata["generated_video_path"] = str(generated_video_path)

        return {
            "status": "ok",
            "actions": actions,
            "infer_time_ms": timing["infer_time_ms"],
            "timing": timing,
            "metadata": metadata,
        }

    def _run_infer_path(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        total_started_at = time.perf_counter()

        preprocess_started_at = time.perf_counter()
        payload_preprocessed = self._policy.preprocess(payload, stat_key=payload.get("stat_key"))
        preprocess_ms = self._elapsed_ms(preprocess_started_at)
        if self._save_generated_video_dir is not None:
            payload_preprocessed["return_video"] = True

        predict_started_at = time.perf_counter()
        with torch.inference_mode():
            output = self._policy.predict_action(**payload_preprocessed)
        predict_ms = self._elapsed_ms(predict_started_at)

        postprocess_started_at = time.perf_counter()
        output = self._policy.postprocess(output, payload, stat_key=payload.get("stat_key"))
        postprocess_ms = self._elapsed_ms(postprocess_started_at)

        timing = {
            "preprocess_ms": round(preprocess_ms, 3),
            "predict_ms": round(predict_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "infer_time_ms": round(self._elapsed_ms(total_started_at), 3),
        }
        return output, timing

    def _maybe_save_generated_video(
        self,
        output: dict[str, Any],
        msg: dict[str, Any],
        save_stem: str,
        infer_index: int,
    ) -> Path | None:
        if not self._should_save_debug_artifacts(infer_index):
            return None
        video = output.get("video")
        if video is None:
            logger.warning("Generated-video saving is enabled, but policy output has no `video` field.")
            return None

        try:
            import imageio.v2 as imageio

            frames = self._video_to_uint8_frames(video)
            if not frames:
                raise ValueError("decoded generated video contains no frames.")
            fps = self._resolve_save_video_fps(msg)
            path = self._save_generated_video_dir / f"{save_stem}_generated.mp4"
            imageio.mimwrite(str(path), frames, fps=fps, quality=8, macro_block_size=1)
            logger.info("Saved generated video to %s (frames=%d, fps=%.3f)", path, len(frames), fps)
            return path
        except Exception:
            logger.exception("Failed to save generated video; continuing inference.")
            return None

    def _maybe_save_input_frames(
        self,
        msg: dict[str, Any],
        save_stem: str,
        infer_index: int,
        model_images: list[np.ndarray] | None = None,
    ) -> dict[str, Path]:
        if not self._should_save_debug_artifacts(infer_index):
            return {}

        try:
            import imageio.v2 as imageio

            raw_frame = self._concat_request_frames(msg, resize_size=None)
            raw_path = self._save_generated_video_dir / f"{save_stem}_input_raw.png"
            imageio.imwrite(str(raw_path), np.ascontiguousarray(raw_frame))

            paths = {"raw": raw_path}
            if self._adapter_config.image_resize_size is not None:
                # 用 payload 里的图像存 model 视图,与模型实际输入保持一致。
                if model_images is not None:
                    model_frame = np.concatenate(model_images, axis=1)
                else:
                    model_frame = self._concat_request_frames(msg, resize_size=self._adapter_config.image_resize_size)
                model_path = self._save_generated_video_dir / f"{save_stem}_input_model.png"
                imageio.imwrite(str(model_path), np.ascontiguousarray(model_frame))
                paths["model"] = model_path

            logger.info("Saved input frame(s) to %s (cameras=%s)", self._save_generated_video_dir, list(self._adapter_config.camera_order))
            return paths
        except Exception:
            logger.exception("Failed to save input frames; continuing inference.")
            return {}

    def _concat_request_frames(self, msg: dict[str, Any], *, resize_size: tuple[int, int] | None) -> np.ndarray:
        frames = []
        for camera_name in self._adapter_config.camera_order:
            value = msg.get(camera_name)
            if not isinstance(value, (bytes, bytearray)):
                raise InferSystemProtocolError(f"cannot save input frame; missing camera image {camera_name!r}.")
            frames.append(decode_image_bytes(value, image_color="rgb", resize_size=resize_size))
        return np.concatenate(frames, axis=1)

    def _should_save_debug_artifacts(self, infer_index: int) -> bool:
        return (
            self._save_generated_video_dir is not None
            and infer_index % self._save_generated_video_every == 0
        )

    def _build_save_stem(self, msg: dict[str, Any], infer_index: int) -> str:
        request_id = self._safe_filename_suffix(msg.get("request_id"))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}_{infer_index:06d}{request_id}"

    def _resolve_save_video_fps(self, msg: dict[str, Any]) -> float:
        if self._save_generated_video_fps is not None:
            return max(float(self._save_generated_video_fps), 0.001)

        extra = msg.get("extra") if isinstance(msg.get("extra"), dict) else {}
        fps_value = msg.get("fps", extra.get("fps", self._adapter_config.default_fps))
        if isinstance(fps_value, (list, tuple)):
            fps_value = fps_value[0] if fps_value else None
        try:
            fps = float(fps_value)
        except (TypeError, ValueError):
            fps = 0.0

        stride = 1
        try:
            stride = int(getattr(self._policy.config.datasets.vla_data, "future_frame_stride", 1))
        except Exception:
            stride = 1
        if fps > 0:
            return max(fps / max(stride, 1), 0.001)
        return 8.0

    @staticmethod
    def _video_to_uint8_frames(video: Any) -> list[np.ndarray]:
        if torch.is_tensor(video):
            arr = video.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            arr = np.asarray(video, dtype=np.float32)

        if arr.ndim == 6:
            # [B,V,C,T,H,W] or [B,V,T,H,W,C]; save the first batch with views tiled horizontally.
            per_view = [InferSystemPolicyServer._video_to_uint8_frames(view) for view in arr[0]]
            frame_count = min(len(view_frames) for view_frames in per_view)
            return [np.concatenate([view_frames[i] for view_frames in per_view], axis=1) for i in range(frame_count)]
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim != 4:
            raise ValueError(f"expected generated video shape [B,C,T,H,W], [C,T,H,W], or [T,H,W,C], got {arr.shape}.")

        if arr.shape[0] in {1, 3, 4}:  # [C,T,H,W]
            arr = np.transpose(arr, (1, 2, 3, 0))
        elif arr.shape[1] in {1, 3, 4}:  # [T,C,H,W]
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"cannot infer channel dimension for generated video shape {arr.shape}.")

        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        return [np.ascontiguousarray(frame) for frame in arr]

    @staticmethod
    def _safe_filename_suffix(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:48]
        return f"_{safe}" if safe else ""

    def _handle_reset(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            if hasattr(self._policy, "handle_request"):
                self._policy.handle_request("reset", msg)
            elif hasattr(self._policy, "reset"):
                self._policy.reset()
        except Exception as exc:
            logger.exception("Policy reset failed.")
            return self._error(str(exc))
        return {"status": "ok"}

    def _metadata(self) -> dict[str, Any]:
        if hasattr(self._policy, "get_metadata"):
            metadata = dict(self._policy.get_metadata())
        else:
            metadata = {}
        data_config = getattr(self._policy, "data_config", None)
        state_ids = getattr(data_config, "state_ids", ())
        state_dim = len(state_ids) if state_ids else getattr(data_config, "state_pad_size", None)
        action_dim = getattr(data_config, "transformed_action_dim", None)
        if action_dim is None:
            action_ids = getattr(data_config, "action_ids", ())
            action_dim = len(action_ids) if action_ids else getattr(data_config, "action_pad_size", None)
        if state_dim is not None:
            metadata.setdefault("state_dim", int(state_dim))
        if action_dim is not None:
            metadata.setdefault("action_dim", int(action_dim))
        action_space = getattr(data_config, "action_space", None)
        if action_space is not None:
            metadata.setdefault("action_space", str(action_space))
        metadata.update(
            {
                "env": "starvla_infersystem_zmq_server",
                "camera_order": list(self._adapter_config.camera_order),
                "stat_key": self._adapter_config.stat_key or metadata.get("stat_key"),
            }
        )
        return metadata

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {"status": "error", "message": message}


def _build_policy(args: argparse.Namespace):
    policy = load_framework_for_inference(args.ckpt_path)
    policy.set_dataconfig(args.stat_key)
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()
    if args.compile:
        logger.info("Compiling policy with framework defaults.")
        policy.compile()
        logger.info("Policy compile done.")
    return policy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StarVLA server for the InferSystem ZMQ/msgpack protocol.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="StarVLA checkpoint path.")
    parser.add_argument("--bind", type=str, default="tcp://*:5555", help="ZMQ REP bind address.")
    parser.add_argument(
        "--stat_key",
        type=str,
        default=None,
        help="Optional assertion for the checkpoint's sole dataset statistics key.",
    )
    parser.add_argument(
        "--camera_order",
        type=str,
        required=True,
        help="Comma-separated camera order matching InferSystem inference.enabled_cameras.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--use_bf16", action="store_true", help="Load policy in bfloat16.")
    parser.add_argument("--compile", action="store_true", help="Run policy.compile() before serving.")
    parser.add_argument("--log_timing_every", type=int, default=0, help="Log inference timing every N predict requests.")
    parser.add_argument("--log_actions_every", type=int, default=0, help="Log returned action chunk every N predict requests.")
    parser.add_argument(
        "--strict_cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject missing or unexpected camera frames.",
    )
    parser.add_argument("--image_color", choices=("rgb", "bgr"), default="rgb", help="Image array color order for StarVLA.")
    parser.add_argument(
        "--pre_resize_image_size",
        type=str,
        default="auto",
        help="Resize each decoded InferSystem camera image to H,W before BaseDataConfig preprocessing. Use 'auto', 'none', or e.g. '224,224'.",
    )
    parser.add_argument("--default_fps", type=float, default=None, help="Optional FPS passed to StarVLA when client omits it.")
    parser.add_argument(
        "--save_generated_video_dir",
        type=str,
        default=None,
        help="Optional directory for mp4 videos decoded from the same joint denoising pass as action inference.",
    )
    parser.add_argument(
        "--save_generated_video_every",
        type=int,
        default=1,
        help="When saving generated videos, save every N predict requests.",
    )
    parser.add_argument(
        "--save_generated_video_fps",
        type=float,
        default=None,
        help="FPS for saved generated videos. Defaults to request fps divided by future_frame_stride when available.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def _parse_hw_size(value: str | None) -> tuple[int, int] | str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "off", "false", "0"}:
        return None
    if text == "auto":
        return "auto"
    normalized = text.replace("x", ",").replace(" ", ",")
    parts = [part for part in normalized.split(",") if part]
    if len(parts) != 2:
        raise ValueError(f"Expected --pre_resize_image_size as 'H,W', 'auto', or 'none', got {value!r}.")
    size = (int(parts[0]), int(parts[1]))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"--pre_resize_image_size must be positive, got {size!r}.")
    return size


def _resolve_pre_resize_image_size(args: argparse.Namespace, policy) -> tuple[int, int] | None:
    requested = _parse_hw_size(args.pre_resize_image_size)
    if requested != "auto":
        return requested

    data_config = getattr(policy, "data_config", None)
    image_size = getattr(data_config, "image_size", None)
    if image_size is None:
        try:
            image_size = policy.config.datasets.vla_data.image_size
        except Exception:
            image_size = None
    if image_size is None:
        return None

    size = tuple(int(dim) for dim in image_size)
    if len(size) != 2:
        raise ValueError(f"Cannot resolve auto pre-resize image size from {image_size!r}.")
    return size


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    camera_order = parse_camera_order(args.camera_order)
    if not camera_order:
        raise ValueError("--camera_order must contain at least one camera name.")

    policy = _build_policy(args)
    pre_resize_image_size = _resolve_pre_resize_image_size(args, policy)
    adapter_config = InferSystemAdapterConfig(
        camera_order=camera_order,
        stat_key=policy.stat_key,
        strict_cameras=bool(args.strict_cameras),
        image_color=args.image_color,
        image_resize_size=pre_resize_image_size,
        default_fps=args.default_fps,
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(
        "Starting InferSystem policy server on %s (host=%s, ip=%s, pre_resize_image_size=%s), metadata=%s",
        args.bind,
        hostname,
        local_ip,
        pre_resize_image_size,
        policy.get_metadata() if hasattr(policy, "get_metadata") else {},
    )

    server = InferSystemPolicyServer(
        policy=policy,
        bind=args.bind,
        adapter_config=adapter_config,
        log_timing_every=args.log_timing_every,
        log_actions_every=args.log_actions_every,
        save_generated_video_dir=args.save_generated_video_dir,
        save_generated_video_every=args.save_generated_video_every,
        save_generated_video_fps=args.save_generated_video_fps,
    )
    server.run()


if __name__ == "__main__":
    main(build_argparser().parse_args())
