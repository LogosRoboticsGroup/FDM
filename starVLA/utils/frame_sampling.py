def get_num_future_frames(action_horizon: int, future_frame_stride: int) -> int:
    """Derive the future frame count, excluding the current observation."""
    action_horizon = int(action_horizon)
    future_frame_stride = int(future_frame_stride)
    if action_horizon <= 0 or future_frame_stride <= 0:
        raise ValueError("action_horizon and future_frame_stride must be positive.")
    num_future_frames, remainder = divmod(action_horizon, future_frame_stride)
    if remainder:
        raise ValueError(
            f"action_horizon ({action_horizon}) must be divisible by future_frame_stride ({future_frame_stride})."
        )
    return num_future_frames
