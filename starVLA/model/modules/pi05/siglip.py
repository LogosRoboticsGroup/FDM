# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""JAX-aligned SigLIP vision encoder used by Pi0.5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class Encoder1DBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = MlpBlock(dim, mlp_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dtype = self.norm1.weight.dtype
        y = self.norm1(x.to(norm_dtype))
        residual, _ = self.attn(y, y, y)
        x = x + self.dropout1(residual)
        return x + self.dropout2(self.mlp(self.norm2(x.to(norm_dtype))))


class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [Encoder1DBlock(dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_use_reentrant = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
            else:
                x = layer(x)
        return self.norm(x.to(self.norm.weight.dtype))


class SigLIPViT(nn.Module):
    """SigLIP ViT with the same parameter layout as RLinf/OpenPI."""

    def __init__(
        self,
        variant: str = "So400m/14",
        num_classes: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        params = _decode_variant(variant)
        self.width = params["width"]
        self.patch_size = params["patch_size"]
        self.dtype_mm = dtype

        self.stem = nn.Conv2d(
            3,
            self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        num_patches = (224 // self.patch_size[0]) * (224 // self.patch_size[1])
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, self.width))
        self.encoder = Encoder(
            self.width,
            params["depth"],
            params["num_heads"],
            params["mlp_dim"],
        )
        self.head = nn.Linear(self.width, num_classes)
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, None]:
        x = image.permute(0, 3, 1, 2).float()
        x = F.conv2d(
            x,
            self.stem.weight.float(),
            self.stem.bias.float(),
            stride=self.stem.stride,
        )
        batch, channels, height, width = x.shape
        x = x.reshape(batch, channels, height * width).permute(0, 2, 1)
        x = (x + self.pos_embedding.float()).to(self.dtype_mm)
        return self.head(self.encoder(x)), None


def _decode_variant(variant: str) -> dict[str, int | tuple[int, int]]:
    name, patch = variant.split("/")
    values = {
        "mu": (32, 1, 128, 2),
        "So400m": (1152, 27, 4304, 16),
    }
    if name not in values:
        raise ValueError(f"Unsupported Pi0.5 SigLIP variant: {variant}")
    width, depth, mlp_dim, num_heads = values[name]
    patch_size = (int(patch), int(patch))
    return {
        "width": width,
        "depth": depth,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "patch_size": patch_size,
    }
