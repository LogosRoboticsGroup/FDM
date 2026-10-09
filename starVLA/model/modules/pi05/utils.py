# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small compiled operators used by the Pi0.5 model."""

import torch
import torch.nn.functional as F


@torch.compile
def gelu_glu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """Fused GELU-GLU activation: ``gelu(gate_input) * value_input``."""
    return F.gelu(gate_input) * value_input
