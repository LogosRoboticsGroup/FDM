"""Pi0.5 with a randomly initialized, action-sized future-latent expert."""

import torch
from torch import nn
from torch.nn import functional as F

from . import gemma
from .model import Pi05, make_attn_mask, posemb_sincos


class Pi05Causal(Pi05):
    def add_generation_expert(self, latent_channels=48, scale_factor=4):
        """Call after loading the unmodified Pi0.5 checkpoint, preserving its keys."""
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError("scale_factor must be a positive integer")
        self.generation_scale_factor = scale_factor
        config = self.llm.configs[1]
        for layer in self.llm.layers:
            expert = gemma.Block([config], adarms=[True])
            for name in ("pre_attention_norms", "pre_ffw_norms", "mlps"):
                getattr(layer, name).append(getattr(expert, name)[0])
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(layer.attn, name).append(getattr(expert.attn, name)[0])
            layer.configs = [*layer.configs, config]
            layer.attn.expert_configs = layer.configs
        self.llm.configs = [*self.llm.configs, config]
        self.llm.final_norms.append(gemma.RMSNorm(config.width, adaptive=True))
        self.generation_in_proj = nn.Conv2d(latent_channels, config.width, kernel_size=1)
        self.generation_downsample = nn.Conv2d(config.width, config.width, kernel_size=scale_factor, stride=scale_factor)
        self.generation_upsample = nn.ConvTranspose2d(
            config.width, config.width, kernel_size=scale_factor, stride=scale_factor
        )
        self.generation_out_norm = nn.LayerNorm(config.width)
        self.generation_out_proj = nn.Linear(config.width, latent_channels)
        self.generation_time_mlp = nn.Sequential(
            nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width), nn.SiLU()
        )
        self.to(dtype=self.embed_dtype)

    def predict_latents(self, observation, context, view_mask, actions, time):
        """Joint pass: prefix -> actions -> generation; context is [B,V,2,C,H,W]."""
        batch, views, frames, channels, height, width = context.shape
        if height % self.generation_scale_factor or width % self.generation_scale_factor:
            raise ValueError("generation scale_factor must divide the VAE latent grid")
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, action_cond = self.embed_suffix(actions, time)
        generation = context.reshape(batch * views * frames, channels, height, width).to(self.embed_dtype)
        generation = self.generation_downsample(self.generation_in_proj(generation))
        pooled_height, pooled_width = generation.shape[-2:]
        generation = generation.permute(0, 2, 3, 1).reshape(batch, -1, generation.shape[1])
        generation_mask = view_mask[:, :, None].expand(-1, -1, frames * pooled_height * pooled_width).reshape(batch, -1)
        generation_ar = torch.zeros(generation.shape[1], dtype=torch.bool, device=generation.device)
        generation_ar[0] = True
        mask = torch.cat([prefix_mask, suffix_mask, generation_mask], dim=1)
        ar = torch.cat([prefix_ar, suffix_ar, generation_ar])
        # Direct latent regression has no generation diffusion time. Use t=0 for adaRMS.
        generation_cond = self.generation_time_mlp(
            posemb_sincos(torch.zeros_like(time).to(self.embed_dtype), generation.shape[-1])
        )
        outputs, _ = self.llm(
            [prefix, suffix, generation],
            positions=mask.int().cumsum(dim=1) - 1,
            mask=make_attn_mask(mask, ar),
            adarms_cond=[None, action_cond, generation_cond],
        )
        generation = outputs[2].reshape(batch, views, frames, pooled_height, pooled_width, -1).mean(dim=2)
        generation = generation.reshape(batch * views, pooled_height, pooled_width, -1).permute(0, 3, 1, 2)
        generation = self.generation_upsample(generation).permute(0, 2, 3, 1)
        latents = self.generation_out_proj(self.generation_out_norm(generation))
        latents = latents.reshape(batch, views, height, width, channels).permute(0, 1, 4, 2, 3)
        return self.action_out_proj(outputs[1]), latents

    def compute_loss(self, observation, actions, *, context, target, view_mask, noise=None, time=None, generator=None):
        actions = actions.to(self.embed_dtype)
        if noise is None:
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
        else:
            noise = noise.to(actions)
        if time is None:
            time = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)) * 0.999 + 0.001
        time = time.to(actions)
        noisy_actions = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        velocity, predicted = self.predict_latents(observation, context, view_mask, noisy_actions, time)
        action_loss = (velocity.float() - (noise - actions).float()).square().mean(dim=-1)
        per_view = F.mse_loss(predicted.float(), target.float(), reduction="none").mean(dim=(2, 3, 4))
        generation_loss = (per_view * view_mask).sum() / view_mask.sum().clamp_min(1)
        return action_loss, generation_loss
