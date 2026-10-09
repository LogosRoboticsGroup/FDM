from __future__ import annotations


def create_robot(config):
    """Select hardware without importing either vendor SDK until startup."""
    if config.robot_type == "arx_x5":
        from .arx import DualArx

        return DualArx(config.arms)
    from .piper import DualPiper

    return DualPiper(config.arms)
