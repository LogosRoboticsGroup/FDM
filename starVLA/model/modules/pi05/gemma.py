# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""JAX-aligned dual-expert Gemma used by Pi0.5.

This is the Pi0.5-only subset of RLinf's PyTorch implementation.  The visual
language expert uses ordinary RMSNorm while the action expert uses adaptive
RMSNorm conditioned on the flow-matching timestep.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .utils import gelu_glu

PALIGEMMA_VOCAB_SIZE = 257_152
Variant = Literal["dummy", "gemma_300m", "gemma_2b"]


@dataclasses.dataclass(frozen=True)
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_config(variant: Variant) -> Config:
    if variant == "dummy":
        return Config(64, 4, 128, 8, 1, 16)
    if variant == "gemma_300m":
        return Config(1024, 18, 4096, 8, 1, 256)
    if variant == "gemma_2b":
        return Config(2048, 18, 16_384, 8, 1, 256)
    raise ValueError(f"Unknown Gemma variant: {variant}")


class RMSNorm(nn.Module):
    """Gemma RMSNorm, optionally adaptive for the Pi0.5 action expert."""

    def __init__(self, dim: int, adaptive: bool = False):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.ada_modulation = nn.Linear(dim, dim * 3)
            nn.init.zeros_(self.ada_modulation.weight)
            nn.init.zeros_(self.ada_modulation.bias)
        else:
            self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + 1e-6)

        if not self.adaptive:
            return (normed * (1.0 + self.scale.float())).to(dtype), None

        modulation = self.ada_modulation(cond.to(self.ada_modulation.weight.dtype))
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        if x.ndim == 3 and modulation.ndim == 2:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
            gate = gate.unsqueeze(-2)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens) * math.sqrt(self.embed_dim)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class Attention(nn.Module):
    """Grouped-query attention shared across the VLM and action experts."""

    def __init__(self, configs: Sequence[Config]):
        super().__init__()
        self.expert_configs = configs
        self.num_heads = configs[0].num_heads
        self.num_kv_heads = configs[0].num_kv_heads
        self.head_dim = configs[0].head_dim

        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.v_proj = nn.ModuleList()
        self.o_proj = nn.ModuleList()
        for config in configs:
            if config.num_kv_heads == config.num_heads:
                self.q_proj.append(nn.Linear(config.width, 3 * config.num_heads * config.head_dim, bias=False))
                self.k_proj.append(None)
                self.v_proj.append(None)
            else:
                self.q_proj.append(nn.Linear(config.width, config.num_heads * config.head_dim, bias=False))
                self.k_proj.append(nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False))
                self.v_proj.append(nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False))
            self.o_proj.append(nn.Linear(config.num_heads * config.head_dim, config.width, bias=False))
        self._init_weights()

    def _init_weights(self) -> None:
        for index, config in enumerate(self.expert_configs):
            nn.init.normal_(self.q_proj[index].weight, std=1.0 / math.sqrt(config.width))
            if self.k_proj[index] is not None:
                nn.init.normal_(self.k_proj[index].weight, std=1.0 / math.sqrt(config.width))
                nn.init.normal_(self.v_proj[index].weight, std=1.0 / math.sqrt(config.width))
            nn.init.normal_(
                self.o_proj[index].weight,
                std=1.0 / math.sqrt(config.num_heads * config.head_dim),
            )

    def forward(
        self,
        xs: list[torch.Tensor | None],
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        dtype = next(x.dtype for x in xs if x is not None)
        q_parts: list[torch.Tensor] = []
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []

        for index, x in enumerate(xs):
            if x is None:
                continue
            batch, length, _ = x.shape
            if self.k_proj[index] is None:
                qkv = self.q_proj[index](x).reshape(batch, length, 3, self.num_heads, self.head_dim)
                q, k, v = qkv.unbind(dim=2)
            else:
                q = self.q_proj[index](x).reshape(batch, length, self.num_heads, self.head_dim)
                k = self.k_proj[index](x).reshape(batch, length, self.num_kv_heads, self.head_dim)
                v = self.v_proj[index](x).reshape(batch, length, self.num_kv_heads, self.head_dim)
            q_parts.append(q)
            k_parts.append(k)
            v_parts.append(v)

        q = _apply_rope(torch.cat(q_parts, dim=1), positions=positions)
        k = _apply_rope(torch.cat(k_parts, dim=1), positions=positions)
        v = torch.cat(v_parts, dim=1)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        new_kv_cache = (k, v)

        q = q * (self.head_dim**-0.5)
        num_groups = self.num_heads // self.num_kv_heads
        q = q.reshape(q.shape[0], q.shape[1], self.num_kv_heads, num_groups, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.num_kv_heads, self.head_dim)
        logits = torch.einsum("BTKGH,BSKH->BKGTS", q.float(), k.float())

        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, 0:1]
        mask = attn_mask[:, :, None].expand_as(logits).bool()
        logits = torch.where(mask, logits, -2.3819763e38)
        probs = F.softmax(logits, dim=-1).to(dtype)
        encoded = torch.einsum("BKGTS,BSKH->BTKGH", probs, v.to(dtype))
        encoded = encoded.reshape(encoded.shape[0], encoded.shape[1], self.num_heads, self.head_dim)

        outputs: list[torch.Tensor | None] = []
        start = 0
        for index, x in enumerate(xs):
            if x is None:
                outputs.append(None)
                continue
            end = start + x.shape[1]
            expert_out = encoded[:, start:end].reshape(x.shape[0], x.shape[1], -1)
            outputs.append(self.o_proj[index](expert_out))
            start = end
        return outputs, new_kv_cache


class FeedForward(nn.Module):
    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.w_gating = nn.Parameter(torch.empty(2, features, hidden_dim))
        self.w_linear = nn.Parameter(torch.empty(hidden_dim, features))
        nn.init.normal_(self.w_gating[0], std=1.0 / math.sqrt(features))
        nn.init.normal_(self.w_gating[1], std=1.0 / math.sqrt(features))
        nn.init.normal_(self.w_linear, std=1.0 / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        gate = torch.matmul(x, self.w_gating[0].to(dtype))
        value = torch.matmul(x, self.w_gating[1].to(dtype))
        return torch.matmul(gelu_glu(gate, value), self.w_linear.to(dtype))


class Block(nn.Module):
    def __init__(self, configs: Sequence[Config], adarms: Sequence[bool]):
        super().__init__()
        self.configs = configs
        self.attn = Attention(configs)
        self.pre_attention_norms = nn.ModuleList(
            [RMSNorm(config.width, adaptive=adarms[i]) for i, config in enumerate(configs)]
        )
        self.pre_ffw_norms = nn.ModuleList(
            [RMSNorm(config.width, adaptive=adarms[i]) for i, config in enumerate(configs)]
        )
        self.mlps = nn.ModuleList([FeedForward(config.width, config.mlp_dim) for config in configs])

    def forward(
        self,
        xs: list[torch.Tensor | None],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        adarms_cond: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        pre_attn: list[torch.Tensor | None] = []
        gates: list[torch.Tensor | None] = []
        for index, x in enumerate(xs):
            if x is None:
                pre_attn.append(None)
                gates.append(None)
            else:
                normed, gate = self.pre_attention_norms[index](x, adarms_cond[index])
                pre_attn.append(normed)
                gates.append(gate)

        post_attn, kv_cache = self.attn(pre_attn, positions, attn_mask, kv_cache)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]

        post_ffn: list[torch.Tensor | None] = []
        gates = []
        for index, x in enumerate(xs):
            if x is None:
                post_ffn.append(None)
                gates.append(None)
            else:
                normed, gate = self.pre_ffw_norms[index](x, adarms_cond[index])
                post_ffn.append(self.mlps[index](normed))
                gates.append(gate)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_ffn, gates, strict=True)]
        return xs, kv_cache


class Module(nn.Module):
    """Two-expert Gemma stack with per-layer KV caches."""

    def __init__(
        self,
        configs: Sequence[Config],
        embed_dtype: torch.dtype,
        vocab_size: int = PALIGEMMA_VOCAB_SIZE,
    ):
        super().__init__()
        self.configs = configs
        self.embed_dtype = embed_dtype
        self.embedder = Embedder(vocab_size=vocab_size, embed_dim=configs[0].width)
        self.layers = nn.ModuleList([Block(configs, adarms=[False, True]) for _ in range(configs[0].depth)])
        self.final_norms = nn.ModuleList(
            [RMSNorm(config.width, adaptive=index == 1) for index, config in enumerate(configs)]
        )
        self.gradient_checkpointing = False
        self.gradient_checkpointing_use_reentrant = False

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedder.encode(tokens).to(self.embed_dtype)

    def forward(
        self,
        embedded: Sequence[torch.Tensor | None],
        positions: torch.Tensor,
        mask: torch.Tensor,
        adarms_cond: Sequence[torch.Tensor | None] | None = None,
        *,
        kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if adarms_cond is None:
            adarms_cond = [None, None]
        xs = [x.to(self.embed_dtype) if x is not None else None for x in embedded]
        mask = mask.unsqueeze(1)
        layer_caches = [None] * len(self.layers) if kv_cache is None else list(kv_cache)
        new_caches = []
        for index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                xs, new_cache = torch.utils.checkpoint.checkpoint(
                    layer,
                    xs,
                    layer_caches[index],
                    positions,
                    mask,
                    list(adarms_cond),
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
            else:
                xs, new_cache = layer(xs, layer_caches[index], positions, mask, list(adarms_cond))
            new_caches.append(new_cache)

        outputs: list[torch.Tensor | None] = []
        for index, x in enumerate(xs):
            if x is None:
                outputs.append(None)
            else:
                outputs.append(self.final_norms[index](x, adarms_cond[index])[0])
        return outputs, tuple(new_caches)


@torch.compile
def _apply_rope(x: torch.Tensor, *, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to an attention query or key tensor."""
    head_dim = x.shape[-1]
    exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32, device=x.device)
    radians = positions[..., None].float() / (max_wavelength**exponents)[None, None]
    sin, cos = torch.sin(radians[..., None, :]), torch.cos(radians[..., None, :])
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


@torch.compile
def _fused_gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fuse ``x + y * gate`` into a single compiled kernel."""
    return x + y * gate


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    if x is None:
        return None
    if gate is None:
        return x + y
    return _fused_gated_residual(x, y, gate)
