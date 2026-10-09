# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Checkpoint loading for RLinf and legacy OpenPI PyTorch Pi0.5 layouts."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


_LEGACY_PREFIX = "paligemma_with_expert."


def load_pi05_weights(model, checkpoint: str | Path) -> None:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = path / "model.safetensors"
    state_dict = load_file(str(path), device="cpu")
    if any(key.startswith(_LEGACY_PREFIX) for key in state_dict):
        state_dict = convert_openpi_state_dict(
            state_dict,
            vision_depth=len(model.img.encoder.layers),
            llm_depth=len(model.llm.layers),
        )
    model.load_state_dict(state_dict, strict=True)


def convert_openpi_state_dict(
    source: dict[str, torch.Tensor], *, vision_depth: int, llm_depth: int
) -> dict[str, torch.Tensor]:
    """Convert the Transformers-based OpenPI Pi0.5 checkpoint in memory."""
    target: dict[str, torch.Tensor] = {}
    vision = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    _copy_linear(source, target, vision + "embeddings.patch_embedding", "img.stem")
    position_key = vision + "embeddings.position_embedding.weight"
    if position_key in source:
        position = source[position_key]
        target["img.pos_embedding"] = position.unsqueeze(0) if position.ndim == 2 else position

    for layer_index in range(vision_depth):
        old = f"{vision}encoder.layers.{layer_index}."
        new = f"img.encoder.layers.{layer_index}."
        _copy_linear(source, target, old + "layer_norm1", new + "norm1")
        _copy_linear(source, target, old + "layer_norm2", new + "norm2")
        weights = [
            source[old + f"self_attn.{name}.weight"]
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        biases = [
            source[old + f"self_attn.{name}.bias"]
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        target[new + "attn.in_proj_weight"] = torch.cat(weights)
        target[new + "attn.in_proj_bias"] = torch.cat(biases)
        _copy_linear(source, target, old + "self_attn.out_proj", new + "attn.out_proj")
        _copy_linear(source, target, old + "mlp.fc1", new + "mlp.fc1")
        _copy_linear(source, target, old + "mlp.fc2", new + "mlp.fc2")

    _copy_linear(source, target, vision + "post_layernorm", "img.encoder.norm")
    _copy_linear(
        source,
        target,
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear",
        "img.head",
    )

    pali = "paligemma_with_expert.paligemma.model.language_model."
    action = "paligemma_with_expert.gemma_expert.model."
    for layer_index in range(llm_depth):
        new = f"llm.layers.{layer_index}."
        for expert_index, old_root in enumerate((pali, action)):
            old = f"{old_root}layers.{layer_index}."
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                target[f"{new}attn.{projection}.{expert_index}.weight"] = source[
                    f"{old}self_attn.{projection}.weight"
                ]
            target[f"{new}mlps.{expert_index}.w_gating"] = torch.stack(
                [
                    source[old + "mlp.gate_proj.weight"].T.contiguous(),
                    source[old + "mlp.up_proj.weight"].T.contiguous(),
                ]
            )
            target[f"{new}mlps.{expert_index}.w_linear"] = source[
                old + "mlp.down_proj.weight"
            ].T.contiguous()

        pali_old = f"{pali}layers.{layer_index}."
        target[new + "pre_attention_norms.0.scale"] = source[
            pali_old + "input_layernorm.weight"
        ]
        target[new + "pre_ffw_norms.0.scale"] = source[
            pali_old + "post_attention_layernorm.weight"
        ]
        action_old = f"{action}layers.{layer_index}."
        _copy_linear(
            source,
            target,
            action_old + "input_layernorm.dense",
            new + "pre_attention_norms.1.ada_modulation",
        )
        _copy_linear(
            source,
            target,
            action_old + "post_attention_layernorm.dense",
            new + "pre_ffw_norms.1.ada_modulation",
        )

    target["llm.final_norms.0.scale"] = source[pali + "norm.weight"]
    _copy_linear(
        source,
        target,
        action + "norm.dense",
        "llm.final_norms.1.ada_modulation",
    )
    embedding_key = "paligemma_with_expert.paligemma.lm_head.weight"
    target["llm.embedder.embedding.weight"] = source[embedding_key]

    for key, value in source.items():
        if key.startswith(("action_in_proj", "action_out_proj", "time_mlp_")):
            target[key] = value
    return target


def _copy_linear(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    old: str,
    new: str,
) -> None:
    for suffix in (".weight", ".bias"):
        key = old + suffix
        if key in source:
            target[new + suffix] = source[key]
