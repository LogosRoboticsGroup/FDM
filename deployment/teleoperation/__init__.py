"""Piper + WebXR teleoperation runtime.

The package keeps configuration and dry-run imports free of hardware side effects.
"""

from .config import TeleoperationConfig, load_config
from .contracts import ArmObservation, ArmTarget, VRFrame, VRHand

__all__ = [
    "ArmObservation",
    "ArmTarget",
    "TeleoperationConfig",
    "VRFrame",
    "VRHand",
    "load_config",
]
