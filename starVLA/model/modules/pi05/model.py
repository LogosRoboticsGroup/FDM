# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pi0.5 flow-matching model, aligned with RLinf's PyTorch implementation."""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gemma, siglip


@dataclasses.dataclass
class Observation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Pi05Config:
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    dtype: str = "bfloat16"
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    vision_variant: str = "So400m/14"
    vocab_size: int = gemma.PALIGEMMA_VOCAB_SIZE
    rtc_training_max_delay: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.rtc_training_max_delay < self.action_horizon:
            raise ValueError("rtc_training_max_delay must satisfy 0 <= delay < action_horizon")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    block = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = block[:, None, :] <= block[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return attn_mask & valid_mask


def posemb_sincos(
    position: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    fraction = torch.linspace(
        0.0,
        1.0,
        embedding_dim // 2,
        dtype=torch.float32,
        device=position.device,
    )
    period = min_period * (max_period / min_period) ** fraction
    radians = torch.einsum("...i,j->...ij", position.float(), 1.0 / period * 2 * torch.pi)
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1).to(position.dtype)


class Pi05(nn.Module):
    """Pi0.5 only: discrete state prefix plus an adaRMS action expert."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.max_token_len = config.max_token_len
        self.embed_dtype = config.torch_dtype

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)
        self.llm = gemma.Module(
            [paligemma_config, action_expert_config],
            embed_dtype=self.embed_dtype,
            vocab_size=config.vocab_size,
        )
        self.img = siglip.SigLIPViT(
            variant=config.vision_variant,
            num_classes=paligemma_config.width,
            dtype=self.embed_dtype,
        )

        width = action_expert_config.width
        self.action_in_proj = nn.Linear(config.action_dim, width)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)
        self.action_out_proj = nn.Linear(width, config.action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (
            self.action_in_proj,
            self.time_mlp_in,
            self.time_mlp_out,
            self.action_out_proj,
        ):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def embed_prefix(self, observation: Observation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = []
        input_masks = []
        for name, image in observation.images.items():
            image_tokens, _ = self.img(image)
            tokens.append(image_tokens)
            input_masks.append(
                observation.image_masks[name][:, None].expand(image_tokens.shape[0], image_tokens.shape[1])
            )

        language_tokens = self.llm.embed(observation.tokenized_prompt)
        tokens.append(language_tokens)
        input_masks.append(observation.tokenized_prompt_mask)
        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_masks, dim=1)
        ar_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_actions = noisy_actions.to(self.embed_dtype)
        timestep = timestep.to(self.embed_dtype)
        action_tokens = self.action_in_proj(noisy_actions)
        time_embedding = posemb_sincos(timestep, self.action_in_proj.out_features)
        time_embedding = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_embedding))))
        input_mask = torch.ones(action_tokens.shape[:2], dtype=torch.bool, device=action_tokens.device)
        ar_mask = torch.zeros(action_tokens.shape[1], dtype=torch.bool, device=action_tokens.device)
        ar_mask[0] = True
        return action_tokens, input_mask, ar_mask, time_embedding

    def compute_loss(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        action_tokens: torch.Tensor | None = None,
        action_token_mask: torch.Tensor | None = None,
        subtask_tokens: torch.Tensor | None = None,
        subtask_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return unreduced flow loss, plus the RTC mask and token losses when enabled."""
        actions = actions.to(self.embed_dtype)
        if noise is None:
            noise = torch.randn(
                actions.shape,
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
        noise = noise.to(actions)
        if time is None:
            time = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((actions.shape[0],))
            time = time * 0.999 + 0.001
        time = time.to(actions)

        rtc_mask = None
        if self.config.rtc_training_max_delay:
            delays = torch.randint(
                self.config.rtc_training_max_delay + 1,
                (actions.shape[0],),
                device=actions.device,
                generator=generator,
            )
            rtc_mask = torch.arange(actions.shape[1], device=actions.device)[None] < delays[:, None]
            time = time[:, None].expand_as(rtc_mask).masked_fill(rtc_mask, 0)
        time_expanded = time[:, None, None] if rtc_mask is None else time[..., None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(x_t, time)
        context_mask = prefix_mask
        if subtask_tokens is not None:
            prefix, prefix_mask, prefix_ar = self._append_subtask(
                prefix, prefix_mask, prefix_ar, subtask_tokens, subtask_token_mask
            )
        action_context_mask = prefix_mask
        action_start = prefix.shape[1]
        if action_tokens is not None:
            prefix = torch.cat([prefix, self.llm.embed(action_tokens)], dim=1)
            prefix_mask = torch.cat([prefix_mask, action_token_mask], dim=1)
            prefix_ar = torch.cat([prefix_ar, torch.ones_like(action_tokens[0], dtype=torch.bool)])
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar, suffix_ar])
        attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1
        if action_tokens is not None:
            # Flow sees context and subtasks, but never teacher-forced FAST actions.
            attention_mask[:, prefix.shape[1] :, action_start : prefix.shape[1]] = False
            # Exclude the entire FAST segment (including its terminator/EOS) from
            # expert positions, matching inference's Context + Subtask prefix.
            positions[:, prefix.shape[1] :] -= action_token_mask.sum(dim=1, keepdim=True)
        outputs, _ = self.llm(
            [prefix, suffix],
            positions=positions,
            mask=attention_mask,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(outputs[1][:, -self.action_horizon :])
        flow_loss = torch.mean(torch.square(velocity - target_velocity), dim=-1)
        losses = {"flow_loss": flow_loss}
        if rtc_mask is not None:
            losses["flow_loss_mask"] = ~rtc_mask
        if subtask_tokens is not None:
            losses["subtask_ce_loss"] = self._token_loss(
                outputs[0][:, :action_start], context_mask, subtask_tokens, subtask_token_mask
            )
        if action_tokens is not None:
            losses["action_ce_loss"] = self._token_loss(
                outputs[0], action_context_mask, action_tokens, action_token_mask
            )
        return losses

    def _append_subtask(
        self,
        prefix: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_ar: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append the causal language segment used by both training and flow inference."""
        return (
            torch.cat([prefix, self.llm.embed(tokens)], dim=1),
            torch.cat([prefix_mask, token_mask], dim=1),
            torch.cat([prefix_ar, torch.ones_like(tokens[0], dtype=torch.bool)]),
        )

    @torch.compiler.disable
    def _token_loss(
        self,
        output: torch.Tensor,
        context_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Bridge padding before each target segment for next-token prediction.
        # Keep value-dependent CE chunk counts outside the compiled backbone.
        context_length = context_mask.shape[1]
        last_context = torch.arange(context_length, device=output.device)[None].expand_as(context_mask)
        last_context = last_context.masked_fill(~context_mask, -1).amax(dim=1)
        batch_indices = torch.arange(output.shape[0], device=output.device)
        predictors = torch.cat([output[batch_indices, last_context][:, None], output[:, context_length:-1]], dim=1)
        return self._token_ce(predictors[token_mask], tokens[token_mask])

    def _token_ce(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_sum = hidden.sum().float() * 0
        for start in range(0, hidden.shape[0], 128):
            ce_sum = ce_sum + torch.utils.checkpoint.checkpoint(
                self._action_token_ce, hidden[start : start + 128], labels[start : start + 128], use_reentrant=False
            )
        return ce_sum / max(labels.numel(), 1)

    def _action_token_ce(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.llm.embedder.decode(hidden).float(), labels, reduction="sum")

    @torch.no_grad()
    def generate_subtask_tokens(self, observation: Observation, *, eos_id: int, max_tokens: int) -> torch.Tensor:
        """Greedy language decoding with a cached bidirectional image/task prefix."""
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        outputs, cache = self.llm(
            [prefix, None],
            positions=prefix_mask.int().cumsum(dim=1) - 1,
            mask=make_attn_mask(prefix_mask, prefix_ar),
        )
        last = torch.arange(prefix.shape[1], device=prefix.device)[None].expand_as(prefix_mask)
        last = last.masked_fill(~prefix_mask, -1).amax(dim=1)
        batch = torch.arange(prefix.shape[0], device=prefix.device)
        hidden = outputs[0][batch, last]
        finished = torch.zeros(prefix.shape[0], dtype=torch.bool, device=prefix.device)
        generated = []
        for _ in range(max_tokens):
            token = self.llm.embedder.decode(hidden).argmax(dim=-1)
            token = torch.where(finished, eos_id, token)
            generated.append(token)
            finished = finished | (token == eos_id)
            if bool(finished.all()) or len(generated) == max_tokens:
                break
            positions = prefix_mask.sum(dim=1, keepdim=True)
            prefix_mask = torch.cat([prefix_mask, torch.ones_like(finished[:, None])], dim=1)
            outputs, cache = self.llm(
                [self.llm.embed(token[:, None]), None],
                positions=positions,
                mask=prefix_mask[:, None, :],
                kv_cache=cache,
            )
            hidden = outputs[0][:, -1]
        return torch.stack(generated, dim=1)

    def build_prefix_cache(
        self,
        observation: Observation,
        subtask_tokens: torch.Tensor | None = None,
        subtask_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        if subtask_tokens is not None:
            prefix, prefix_mask, prefix_ar = self._append_subtask(
                prefix, prefix_mask, prefix_ar, subtask_tokens, subtask_token_mask
            )
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        _, kv_cache = self.llm(
            [prefix, None],
            positions=positions,
            mask=make_attn_mask(prefix_mask, prefix_ar),
        )
        return prefix_mask, kv_cache

    def denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_mask: torch.Tensor,
        kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        suffix, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_attention = make_attn_mask(suffix_mask, suffix_ar)
        prefix_attention = prefix_mask[:, None, :].expand(prefix_mask.shape[0], suffix.shape[1], prefix_mask.shape[1])
        attention_mask = torch.cat([prefix_attention, suffix_attention], dim=-1)
        positions = prefix_mask.sum(dim=-1, keepdim=True) + torch.cumsum(suffix_mask.int(), dim=1) - 1
        suffix_out = self.llm(
            [None, suffix],
            positions=positions,
            mask=attention_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )[0][1]
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        subtask_tokens: torch.Tensor | None = None,
        subtask_token_mask: torch.Tensor | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        inference_delay: int = 0,
    ) -> torch.Tensor:
        if not 0 <= inference_delay < self.action_horizon:
            raise ValueError("inference_delay must satisfy 0 <= delay < action_horizon")
        batch = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            noise = torch.randn(
                batch,
                self.action_horizon,
                self.action_dim,
                device=device,
                generator=generator,
            )
        x_t = noise
        hard_prefix = None
        if inference_delay:
            if prev_chunk_left_over is None:
                raise ValueError("prev_chunk_left_over is required for a nonzero inference_delay")
            previous = torch.as_tensor(prev_chunk_left_over, device=device, dtype=x_t.dtype)
            if previous.ndim == 2:
                previous = previous.unsqueeze(0)
            if (
                previous.ndim != 3
                or previous.shape[0] != batch
                or previous.shape[1] < inference_delay
                or not 0 < previous.shape[2] <= self.action_dim
            ):
                raise ValueError("RTC prefix must have shape (B, T >= inference_delay, A <= action_dim)")
            previous = previous[:, :inference_delay]
            if not torch.isfinite(previous).all():
                raise ValueError("RTC prefix contains NaN or Inf")
            hard_prefix = F.pad(previous, (0, self.action_dim - previous.shape[2]))
            x_t = torch.cat([hard_prefix, x_t[:, inference_delay:]], dim=1)
        prefix_mask, kv_cache = self.build_prefix_cache(observation, subtask_tokens, subtask_token_mask)
        dt = -1.0 / num_steps
        for step in range(num_steps):
            timestep = torch.full((batch,), 1.0 + step * dt, device=device, dtype=torch.float32)
            if hard_prefix is not None:
                timestep = timestep[:, None].expand(batch, self.action_horizon).clone()
                timestep[:, :inference_delay] = 0
            x_t = x_t + dt * self.denoise_step(x_t, timestep, prefix_mask, kv_cache)
            if hard_prefix is not None:
                x_t = torch.cat([hard_prefix, x_t[:, inference_delay:]], dim=1)
        return x_t

    def forward(
        self,
        observation: Observation,
        actions: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.compute_loss(observation, actions, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None) -> None:
        use_reentrant = (gradient_checkpointing_kwargs or {}).get("use_reentrant", False)
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self) -> None:
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
