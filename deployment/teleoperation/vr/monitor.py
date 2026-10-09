from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import TextIO

from ..config import TeleoperationConfig, load_config
from ..contracts import SIDES, VRFrame, VRHand
from .protocol import VRState
from .server import WebXRServer


@dataclass(frozen=True)
class FrameMetrics:
    gap_ms: float | None
    receive_hz: float | None
    skipped_frames: int
    total_skipped_frames: int


class FrameRateTracker:
    """Track receive cadence without buffering VR poses."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._session_id: str | None = None
        self._last_seq: int | None = None
        self._last_received_s: float | None = None
        self._received_times: deque[float] = deque()
        self._total_skipped = 0

    def update(self, frame: VRFrame) -> FrameMetrics:
        if frame.session_id != self._session_id:
            self._session_id = frame.session_id
            self._last_seq = None
            self._last_received_s = None
            self._received_times.clear()
            self._total_skipped = 0

        skipped = 0 if self._last_seq is None else max(0, frame.seq - self._last_seq - 1)
        gap_ms = (
            None
            if self._last_received_s is None
            else max(0.0, frame.received_monotonic_s - self._last_received_s) * 1000.0
        )
        self._total_skipped += skipped
        self._last_seq = frame.seq
        self._last_received_s = frame.received_monotonic_s

        self._received_times.append(frame.received_monotonic_s)
        cutoff = frame.received_monotonic_s - self.window_s
        while len(self._received_times) > 1 and self._received_times[0] < cutoff:
            self._received_times.popleft()
        duration = self._received_times[-1] - self._received_times[0]
        receive_hz = (len(self._received_times) - 1) / duration if duration > 0.0 else None
        return FrameMetrics(gap_ms, receive_hz, skipped, self._total_skipped)


def _hand_record(hand: VRHand) -> dict[str, object]:
    return {
        "tracked": hand.tracked,
        "xyz_m": hand.position_m,
        "trigger": hand.trigger,
        "squeeze": hand.squeeze,
    }


def frame_record(frame: VRFrame, metrics: FrameMetrics) -> dict[str, object]:
    return {
        "session_id": frame.session_id,
        "seq": frame.seq,
        "client_timestamp_ms": frame.client_timestamp_ms,
        "received_monotonic_s": frame.received_monotonic_s,
        "receive_hz": metrics.receive_hz,
        "gap_ms": metrics.gap_ms,
        "skipped_frames": metrics.skipped_frames,
        "total_skipped_frames": metrics.total_skipped_frames,
        "hands": {side: _hand_record(frame.hands.get(side, VRHand())) for side in SIDES},
    }


def _format_hand(label: str, hand: VRHand) -> str:
    if not hand.tracked or hand.position_m is None:
        return f"{label}=untracked"
    x, y, z = hand.position_m
    return (
        f"{label}=({x:+.4f}, {y:+.4f}, {z:+.4f})m "
        f"trigger={hand.trigger:.2f} squeeze={hand.squeeze:.2f}"
    )


def format_frame(frame: VRFrame, metrics: FrameMetrics) -> str:
    rate = "--" if metrics.receive_hz is None else f"{metrics.receive_hz:5.1f}"
    gap = "--" if metrics.gap_ms is None else f"{metrics.gap_ms:5.1f}"
    hands = " | ".join(
        _format_hand("L" if side == "left" else "R", frame.hands.get(side, VRHand())) for side in SIDES
    )
    return (
        f"seq={frame.seq:7d} rx={rate}Hz gap={gap}ms "
        f"skip={metrics.skipped_frames}/{metrics.total_skipped_frames} | {hands}"
    )


def run_vr_monitor(
    config: TeleoperationConfig,
    *,
    output_format: str = "human",
    poll_hz: float = 500.0,
    max_frames: int | None = None,
    output: TextIO | None = None,
) -> int:
    """Run only the WebXR communication path; never open Piper CAN devices."""

    config.validate()
    if output_format not in {"human", "json"}:
        raise ValueError("output_format must be 'human' or 'json'")
    if poll_hz <= 0.0:
        raise ValueError("poll_hz must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    output = output or sys.stdout
    state = VRState()
    server = WebXRServer(config.vr, state)
    tracker = FrameRateTracker()
    last_frame_key: tuple[str, int] | None = None
    printed = 0
    period_s = 1.0 / poll_hz

    scheme = "https" if config.vr.tls_cert_path else "http"

    server.start()
    print("VR monitor only: Piper SDK/CAN/motion are not started.", file=sys.stderr, flush=True)
    print(
        f"Open {scheme}://{config.vr.bind_host}:{config.vr.https_port}/ in the VR browser.",
        file=sys.stderr,
        flush=True,
    )
    try:
        while max_frames is None or printed < max_frames:
            frame = state.latest()
            if frame is not None:
                frame_key = (frame.session_id, frame.seq)
                if frame_key != last_frame_key:
                    metrics = tracker.update(frame)
                    if output_format == "json":
                        line = json.dumps(frame_record(frame, metrics), separators=(",", ":"))
                    else:
                        line = format_frame(frame, metrics)
                    print(line, file=output, flush=True)
                    last_frame_key = frame_key
                    printed += 1
                    continue
            time.sleep(period_s)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return printed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receive and print raw StarVLA WebXR controller XYZ poses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-format", choices=("human", "json"), default="human")
    parser.add_argument("--poll-hz", type=float, default=500.0)
    parser.add_argument("--samples", type=int, default=0, help="stop after N frames; 0 means run until Ctrl-C")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    run_vr_monitor(
        config,
        output_format=args.output_format,
        poll_hz=args.poll_hz,
        max_frames=args.samples or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
