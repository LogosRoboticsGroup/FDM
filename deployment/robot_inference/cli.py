from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from .config import load_flexiv_inference_config, resolve_flexiv_policy_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StarVLA client-side inference for one Flexiv arm")
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config")
    metadata = commands.add_parser("server-metadata")
    run = commands.add_parser("run")
    for command in (validate, metadata, run):
        command.add_argument("--config", required=True)
        command.add_argument("--server", help="override inference.server (HOST:PORT)")
        command.add_argument("--stat-key", "--stat_key", help="override inference.stat_key")
        command.add_argument("--prompt", help="override inference.prompt")
        command.add_argument(
            "--camera-order",
            "--camera_order",
            type=lambda value: tuple(part.strip() for part in value.split(",") if part.strip()),
            help="comma-separated camera names; enable only these cameras from the YAML config",
        )
        command.add_argument(
            "--recv-timeout-ms", "--recv_timeout_ms", type=int, help="override inference.recv_timeout_ms"
        )
        command.add_argument("--n-execute", "--n_execute", type=int, help="override inference.n_execute")
    run.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement before enabling and commanding the Flexiv arm",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = {
        name: getattr(args, name)
        for name in ("server", "stat_key", "prompt", "camera_order", "recv_timeout_ms", "n_execute")
        if getattr(args, name) is not None
    }
    config = load_flexiv_inference_config(args.config, inference_overrides=overrides)

    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2))
        return 0

    if args.command == "server-metadata":
        from .client import StarVLAZmqClient

        policy = config.inference
        client = StarVLAZmqClient(
            policy.server,
            camera_order=policy.camera_order,
            jpeg_quality=policy.jpeg_quality,
            recv_timeout_ms=policy.recv_timeout_ms,
            send_timeout_ms=policy.send_timeout_ms,
            max_retries=policy.max_retries,
        )
        try:
            client.connect()
            metadata = client.metadata()
            resolve_flexiv_policy_metadata(policy, metadata)
            print(json.dumps(metadata, indent=2))
        finally:
            client.close()
        return 0

    if not args.execute:
        raise ValueError("Flexiv inference requires the explicit --execute acknowledgement")
    if not config.runtime.hardware_access or not config.runtime.motion_enabled:
        raise ValueError("run --execute requires runtime.hardware_access=true and runtime.motion_enabled=true")
    from .runtime import FlexivInferenceRuntime

    runtime = FlexivInferenceRuntime(config)
    try:
        metadata = runtime.open()
        logging.getLogger(__name__).info("Validated policy metadata: %s", metadata)
        runtime.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning(
            "Inference interrupted; stopping command transmission without returning home"
        )
    finally:
        runtime.close()
    return 0
