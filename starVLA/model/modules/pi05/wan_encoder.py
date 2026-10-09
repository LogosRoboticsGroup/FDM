"""Frozen Wan2.2 image codec: independent T=1 encoding avoids temporal mixing."""

from pathlib import Path

import torch
from torch import nn


class Wan22ImageCodec(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from starVLA.model.modules.wan_video.wan_vae import LATENTS_MEAN_48CH, LATENTS_STD_48CH, Wan2_2_VAE_

        self.model = Wan2_2_VAE_(z_dim=48, dim=160)
        self.register_buffer("mean", torch.tensor(LATENTS_MEAN_48CH), persistent=False)
        self.register_buffer("inv_std", torch.tensor(LATENTS_STD_48CH).reciprocal(), persistent=False)
        path = Path(checkpoint).expanduser()
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            weights = load_file(str(path))
        else:
            weights = torch.load(path, map_location="cpu", weights_only=True)
        weights = weights.get("model_state", weights)
        weights = {key.removeprefix("model."): value for key, value in weights.items()}
        self.model.load_state_dict(weights, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, images):
        """[N,3,H,W] in [-1,1] -> [N,48,H/16,W/16]."""
        images = images.to(device=self.mean.device, dtype=next(self.model.parameters()).dtype)
        return self.model.encode(images[:, :, None], [self.mean, self.inv_std])[:, :, 0]

    @torch.no_grad()
    def decode(self, latents):
        latents = latents.to(device=self.mean.device, dtype=next(self.model.parameters()).dtype)
        return torch.cat(
            [self.model.decode(latent[None, :, None], [self.mean, self.inv_std]) for latent in latents]
        ).clamp(-1, 1)
