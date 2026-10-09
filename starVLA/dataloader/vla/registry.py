"""VLA registries loaded from examples/**/train_files/data_registry/data_config.py.

Benchmark modules export ROBOT_TYPE_CONFIG_MAP and/or DATASET_NAMED_MIXTURES.
They use the same BaseDataConfig and dictionary mixture schema as the core loader.
Names must be unique across all benchmarks; imports never silently
override an existing registration. Keep these modules free of simulator imports.
"""

import importlib
from pathlib import Path

from starVLA.dataloader.vla.data_config import BaseDataConfig

ROBOT_TYPE_CONFIG_MAP = {}
DATASET_NAMED_MIXTURES = {}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DISCOVERED = False


def _find_registry_dirs():
    return sorted(
        path
        for path in (_REPO_ROOT / "examples").glob("**/train_files/data_registry")
        if (path / "data_config.py").is_file() and "sdk_tools" not in path.relative_to(_REPO_ROOT).parts
    )


def _merge_registry(target, incoming, sources, source):
    for name in incoming:
        if name in target:
            raise ValueError(f"Duplicate VLA registration {name!r}: {sources[name]} and {source}")
    target.update(incoming)
    sources.update(dict.fromkeys(incoming, source))


def discover_and_merge():
    """Load benchmark registrations once, publishing only a complete valid registry."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    configs = {}
    mixtures = {}
    config_sources = {}
    mixture_sources = {}
    for path in _find_registry_dirs():
        # Canonical module names also allow DataConfig instances to be pickled by
        # DataLoader workers using spawn, and avoid loading an example twice.
        module_name = ".".join((*path.relative_to(_REPO_ROOT).parts, "data_config"))
        module = importlib.import_module(module_name)
        source = str(path / "data_config.py")
        _merge_registry(configs, getattr(module, "ROBOT_TYPE_CONFIG_MAP", {}), config_sources, source)
        _merge_registry(mixtures, getattr(module, "DATASET_NAMED_MIXTURES", {}), mixture_sources, source)

    for name, config in configs.items():
        if not isinstance(config, BaseDataConfig):
            raise TypeError(f"VLA data_type {name!r} in {config_sources[name]} must use BaseDataConfig")
    for name, mixture in mixtures.items():
        if not isinstance(mixture, dict) or not mixture:
            raise ValueError(f"VLA mixture {name!r} in {mixture_sources[name]} must be a non-empty dictionary")
        for dataset_name, spec in mixture.items():
            data_type = spec["data_type"]
            if data_type not in configs:
                raise ValueError(f"VLA mixture {name!r}, dataset {dataset_name!r}: unknown data_type {data_type!r}")

    ROBOT_TYPE_CONFIG_MAP.update(configs)
    DATASET_NAMED_MIXTURES.update(mixtures)
    _DISCOVERED = True


def video_keys(data_mix):
    """Camera keys of the longest dataset spec in a named mixture."""
    mixture = DATASET_NAMED_MIXTURES[str(data_mix)]
    return max(
        (
            (
                spec["video_keys"]
                if spec.get("video_keys") is not None
                else ROBOT_TYPE_CONFIG_MAP[spec["data_type"]].video_keys
            )
            for spec in mixture.values()
        ),
        key=len,
    )


def num_video_keys(data_mix):
    """Maximum camera count in a named mixture, matching the dataset builder."""
    return len(video_keys(data_mix))


discover_and_merge()
