from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .config import CameraConfig
from .recording.cameras import CameraHub

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/Piper/configs/piper_vr_dual.yaml"
CAPTURE_DIR = Path(__file__).resolve().with_name("capture")


def _load_camera_configs(config_path: str | Path) -> dict[str, CameraConfig]:
    """Load only recording.cameras, without decoding other config fields."""
    path = Path(config_path)
    with path.open() as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, Mapping):
        raise ValueError(f"camera config must be a YAML mapping: {path}")

    recording = payload.get("recording")
    if not isinstance(recording, Mapping) or "cameras" not in recording:
        raise ValueError(f"no recording.cameras mapping in {path}")
    raw_cameras = recording["cameras"]
    location = "recording.cameras"

    if not isinstance(raw_cameras, Mapping) or not raw_cameras:
        raise ValueError(f"{location} must be a non-empty mapping in {path}")

    camera_configs: dict[str, CameraConfig] = {}
    for name, values in raw_cameras.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location} camera names must be non-empty strings")
        if not isinstance(values, Mapping):
            raise ValueError(f"{location}.{name} must be a mapping")
        try:
            config = CameraConfig(**dict(values))
        except TypeError as exc:
            raise ValueError(f"invalid fields in {location}.{name}: {exc}") from exc
        if config.type not in {"opencv", "intelrealsense"}:
            raise ValueError(f"{location}.{name}.type must be opencv or intelrealsense")
        if config.width <= 0 or config.height <= 0 or config.fps <= 0:
            raise ValueError(f"{location}.{name} width, height and fps must be positive")
        if config.color_mode not in {"rgb", "bgr"}:
            raise ValueError(f"{location}.{name}.color_mode must be rgb or bgr")
        if config.fourcc is not None and (len(config.fourcc) != 4 or not config.fourcc.isascii()):
            raise ValueError(f"{location}.{name}.fourcc must be four ASCII characters")
        if config.type == "opencv" and config.index_or_path is None:
            raise ValueError(f"{location}.{name}.index_or_path is required for opencv")
        if config.type == "intelrealsense" and not config.serial_number_or_name:
            raise ValueError(f"{location}.{name}.serial_number_or_name is required for intelrealsense")
        camera_configs[name] = config
    return camera_configs


def _capture_filename(name: str, config: CameraConfig) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    identifier = config.serial_number_or_name
    if identifier:
        safe_identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier.strip())
        if safe_identifier:
            return f"{safe_name}_{safe_identifier}.png"
    return f"{safe_name}.png"


def _save_rgb_png(image: np.ndarray, config: CameraConfig, output_path: Path) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 color image, got shape={array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if config.color_mode == "bgr":
        array = array[..., ::-1]
    Image.fromarray(array, mode="RGB").save(output_path, format="PNG")


def capture(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    camera_configs = _load_camera_configs(config_path)

    destinations: dict[str, tuple[CameraConfig, Path]] = {}
    output_paths: set[Path] = set()
    for name, camera_config in camera_configs.items():
        output_path = CAPTURE_DIR / _capture_filename(name, camera_config)
        if output_path in output_paths:
            raise ValueError(f"camera names produce a duplicate capture path: {output_path}")
        output_paths.add(output_path)
        destinations[name] = (
            camera_config,
            output_path,
        )

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    hub = CameraHub(camera_configs)
    try:
        hub.connect()
        images = hub.read()
        saved = []
        for name, image in images.items():
            camera_config, output_path = destinations[name]
            _save_rgb_png(image, camera_config, output_path)
            saved.append(output_path)
            print(f"saved {name}: {output_path} shape={tuple(image.shape)}")
        return saved
    finally:
        hub.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one PNG from each configured recording.cameras entry")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"teleoperation YAML config (default: {DEFAULT_CONFIG})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
