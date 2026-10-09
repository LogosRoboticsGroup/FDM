"""Pi0.5 causal action and future-frame prediction with a frozen Wan2.2 VAE."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.pi05.causal import Pi05Causal
from starVLA.model.modules.pi05.wan_encoder import Wan22ImageCodec
from starVLA.model.tools import FRAMEWORK_REGISTRY

from .Pi05 import Pi05DefaultConfig, Pi05Framework, _image_to_hwc


@dataclass
class Pi05CausalDefaultConfig(Pi05DefaultConfig):
    name: str = "Pi05Causal"
    model: dict = field(default_factory=lambda: {**Pi05DefaultConfig().model, "action_ce_weight": 0.0})
    generation_model: dict = field(
        default_factory=lambda: {"vae_path": None, "image_size": 256, "scale_factor": 4, "loss_weight": 1.0}
    )


@FRAMEWORK_REGISTRY.register("Pi05Causal")
class Pi05CausalFramework(Pi05Framework):
    def __init__(self, config=None, **kwargs):
        config = merge_framework_config(Pi05CausalDefaultConfig, config)
        model_cfg = config.framework.model
        if (
            model_cfg.action_ce_weight
            or model_cfg.subtask_enabled
            or config.framework.action_model.rtc_training_max_delay
        ):
            raise ValueError(
                "Pi05Causal training supports flow and generation losses only; disable FAST CE, subtask and RTC"
            )
        super().__init__(config, **kwargs)
        generation_cfg = self.config.framework.generation_model
        if not 1 <= self.num_images <= 3:
            raise ValueError("Pi05Causal requires 1 to 3 camera slots")
        self.generation_image_size = int(generation_cfg.image_size)
        if self.generation_image_size <= 0 or self.generation_image_size % 16:
            raise ValueError("generation_model.image_size must be a positive multiple of 16")
        if not generation_cfg.vae_path:
            raise ValueError("Set framework.generation_model.vae_path to the Wan2.2 VAE checkpoint")
        # The original two experts have already loaded strictly from the base checkpoint.
        scale_factor = generation_cfg.scale_factor
        if not isinstance(scale_factor, int) or scale_factor <= 0 or (self.generation_image_size // 16) % scale_factor:
            raise ValueError("generation_model.scale_factor must be a positive integer dividing image_size / 16")
        self.pi05.add_generation_expert(scale_factor=scale_factor)
        self.vae = Wan22ImageCodec(generation_cfg.vae_path)
        self.generation_loss_weight = float(generation_cfg.loss_weight)

    def _build_model(self, config):
        return Pi05Causal(config)

    def _prepare_batch_images(self, batch_images):
        """Select the future GT frame for video evaluation from past/current/future inputs."""
        video = torch.stack([torch.stack([torch.as_tensor(view) for view in views]) for views in batch_images])
        if video.ndim != 6 or video.shape[2:4] != (3, 3):
            raise ValueError("Pi05Causal video evaluation requires [B,V,3,3,H,W] past/current/future images")
        return video[:, :, :, 2:3]

    def _prepare_frames(self, examples, training):
        """Accept per-view [C,T,H,W] in past/current/future order, or separate image keys."""
        if isinstance(examples, dict):
            examples = [examples]
        current_examples, batches, masks = [], [], []
        for example in examples:
            views = example["image"]
            if not isinstance(views, (list, tuple)):
                views = [views]
            if not 1 <= len(views) <= self.num_images:
                raise ValueError("Number of views must fit the configured camera slots")
            mask = torch.as_tensor(example.get("view_mask", [True] * len(views)), dtype=torch.bool)
            if mask.shape != (len(views),) or not mask.any():
                raise ValueError("view_mask must match the views and include a valid camera")
            sequences, current = [], []
            for index, view in enumerate(views):
                if getattr(view, "ndim", None) == 4:
                    if view.shape[0] != 3 or view.shape[1] not in ((3,) if training else (2, 3)):
                        raise ValueError("Expected [3,T,H,W], with T=3 for training or T=2/3 for inference")
                    frames = [view[:, t] for t in range(3 if training else 2)]
                else:
                    past = example.get("past_image")
                    if past is None:
                        raise ValueError("Supply past_image (o_{t-15}) or temporal images in past/current order")
                    if not isinstance(past, (list, tuple)):
                        past = [past]
                    frames = [past[index], view]
                    if training:
                        future = example.get("future_image")
                        if future is None:
                            raise ValueError("Training requires future_image (o_{t+15})")
                        if not isinstance(future, (list, tuple)):
                            future = [future]
                        frames.append(future[index])
                current.append(frames[1])
                # Keep native resolution until resizing for the VAE.
                sequence = torch.stack([_image_to_hwc(frame, size=self.generation_image_size) for frame in frames])
                sequences.append(sequence.permute(0, 3, 1, 2))
            sequences.extend(torch.zeros_like(sequences[0]) for _ in range(self.num_images - len(views)))
            batches.append(torch.stack(sequences))
            masks.append(F.pad(mask, (0, self.num_images - len(views)), value=False))
            current_examples.append({**example, "image": current})
        return current_examples, torch.stack(batches).to(self.device), torch.stack(masks).to(self.device)

    def _prepare_observation(self, examples):
        # Action-only inference uses o_t and never sends the future target into SigLIP.
        if isinstance(examples, dict):
            examples = [examples]
        current = []
        for example in examples:
            views = example["image"]
            if not isinstance(views, (list, tuple)):
                views = [views]
            current.append(
                {**example, "image": [view[:, 1] if getattr(view, "ndim", None) == 4 else view for view in views]}
            )
        return super()._prepare_observation(current)

    def _encode_frames(self, frames):
        # Batch all camera views at each time while keeping independent T=1 encoding.
        batch, views = frames.shape[:2]
        latents = []
        for frame in frames.unbind(dim=2):
            encoded = self.vae.encode(frame.flatten(0, 1))
            latents.append(encoded.reshape(batch, views, *encoded.shape[1:]))
        return torch.stack(latents, dim=2)

    def forward(self, examples, **kwargs):
        current, frames, view_mask = self._prepare_frames(examples, training=True)
        observation = self._prepare_observation(current)
        latents = self._encode_frames(frames)
        action_loss, generation_loss = self.pi05(
            observation,
            self._prepare_actions(current),
            context=latents[:, :, :2],
            target=latents[:, :, 2],
            view_mask=view_mask,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
            generator=kwargs.get("generator"),
        )
        if "action_mask" in current[0]:
            mask = torch.stack([torch.as_tensor(item["action_mask"]) for item in current])
            mask = mask[:, -self.action_horizon :].any(dim=-1).to(action_loss.device)
            action_loss = (action_loss * mask).sum() / mask.sum().clamp_min(1)
        else:
            action_loss = action_loss.mean()
        generation_loss = generation_loss * self.generation_loss_weight
        return {
            "total_loss": action_loss + generation_loss,
            "action_loss": action_loss,
            "generation_loss": generation_loss,
        }

    def compute_loss(self, tag, batch, loss_scale=None):
        if not self.supports_training_tag(tag):
            return None
        scale = (loss_scale or {}).get(tag, 1.0)
        return {key: value * scale for key, value in self(batch).items() if key != "total_loss"}

    @torch.inference_mode()
    def predict_video(self, examples=None, *, actions=None, noise=None, num_steps=None, seed=None, **kwargs):
        if kwargs.get("num_video_frames", 1) != 1:
            raise ValueError("Pi05Causal predicts one future frame per view")
        if examples is None:
            images, instructions = kwargs["batch_images"], kwargs["instructions"]
            examples = [{"image": views, "lang": text} for views, text in zip(images, instructions, strict=True)]
            for key in ("state", "view_mask", "past_image"):
                value = kwargs.get(key, kwargs.get("proprio") if key == "state" else None)
                if value is not None:
                    for index, example in enumerate(examples):
                        example[key] = value[index]
        current, frames, view_mask = self._prepare_frames(examples, training=False)
        observation = self._prepare_observation(current)
        steps = int(num_steps if num_steps is not None else kwargs.get("num_inference_steps", self.num_inference_steps))
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        generator = kwargs.get("generator")
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        was_training = self.training
        self.eval()
        try:
            if actions is None:
                actions = self.pi05.sample_actions(observation, num_steps=steps, noise=noise, generator=generator)
            else:
                actions = torch.as_tensor(actions, device=self.device, dtype=self.pi05.embed_dtype)
                if (
                    actions.ndim != 3
                    or actions.shape[0] != len(current)
                    or actions.shape[1] < self.action_horizon
                    or actions.shape[2] != self.action_dim
                ):
                    raise ValueError("Expected normalized actions [B,T,action_dim] with T >= action_horizon")
                actions = F.pad(actions[:, -self.action_horizon :], (0, self.model_action_dim - self.action_dim))
            _, predicted = self.pi05.predict_latents(
                observation,
                self._encode_frames(frames),
                view_mask,
                actions,
                torch.zeros(len(current), device=self.device),
            )
            video = self.vae.decode(predicted.flatten(0, 1))
            video = (video.reshape(len(current), self.num_images, *video.shape[1:]).float() * 0.5 + 0.5).clamp(0, 1)
        finally:
            self.train(was_training)
        return {"video": video, "normalized_actions": actions[..., : self.action_dim].float().cpu().numpy()}
