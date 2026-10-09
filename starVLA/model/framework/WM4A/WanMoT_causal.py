from typing import Any, List, Optional

import torch

from starVLA.model.tools import FRAMEWORK_REGISTRY

from .WanMoT import WanMoT


@FRAMEWORK_REGISTRY.register("WanMoTCausal")
class WanMoTCausal(WanMoT):
    """Future-video tokens attend to the noisy action stream during training.

    Video-to-video attention follows video_attention_mask_mode in the config.
    Actions read the initial frame and all actions; future video reads actions.
    With first_frame_causal, the initial frame reads only itself and future
    video attention is bidirectional. Inference accepts fixed actions or
    jointly denoises both streams.
    """

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = super()._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        first_frame_end = min(video_tokens_per_frame, video_seq_len)
        mask[first_frame_end:video_seq_len, video_seq_len:] = True
        return mask

    @torch.inference_mode()
    def predict_video(
        self,
        batch_images: Any,
        view_mask: Optional[Any],
        instructions: List[str],
        fps: Optional[List[float]] = None,
        context: Optional[Any] = None,
        context_mask: Optional[Any] = None,
        actions: Optional[Any] = None,
        state: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        """Jointly denoise video/actions, or condition video on supplied actions.

        Without actions, both streams start from noise and update at each step
        using their respective scheduler timesteps. Supplied normalized actions
        [B,T,D] stay fixed at t_action=0; use their final action_horizon steps,
        matching training. Both modes use the causal MoT attention.
        """
        self.eval()
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype
        if state is None:
            state = kwargs.get("proprio", None)

        default_num_video_frames = self.num_future_frames + 1
        num_video_frames = int(kwargs.get("num_video_frames", default_num_video_frames))
        if num_video_frames <= 1 or num_video_frames % 4 != 1:
            raise ValueError(f"WanMoTCausal video inference requires T > 1 and T % 4 == 1, got T={num_video_frames}.")

        images = self._prepare_batch_images(batch_images)[:, :, :, :1, :, :]
        images = self._resize_views_if_needed(images)
        batch_size, n_view, _, _, _, _ = images.shape
        view_mask_tensor = self._prepare_view_mask(view_mask, batch_size, n_view, images.device)
        images = self._normalize_rgb_views_for_vae(images)
        video = self._concat_views(images, view_mask_tensor)
        video = self._resize_video_for_vae(video).to(device=device, dtype=dtype, non_blocking=True)

        context, context_mask = self._prepare_runtime_text_context_with_cache(
            context,
            context_mask,
            instructions,
            batch_size,
            device,
            dtype,
            "predict_video",
        )
        joint_denoising = actions is None
        if not joint_denoising:
            actions = self._batch_to_tensor(actions, device=device, dtype=dtype)
            if (
                actions.ndim != 3
                or actions.shape[0] != batch_size
                or actions.shape[1] < self.action_horizon
                or actions.shape[2] != self.action_dim
            ):
                raise ValueError(
                    f"WanMoTCausal.predict_video requires normalized actions [B,T,D] with "
                    f"B={batch_size}, T >= {self.action_horizon}, D={self.action_dim}; got {tuple(actions.shape)}."
                )
            actions = actions[:, -self.action_horizon :, :]
        state_tensor = self._batch_to_tensor(state, device=device, dtype=dtype)
        context, context_mask = self._append_proprio_to_context(context, context_mask, state_tensor)

        first_frame_latents = self._encode_first_frame_latents(video)
        latent_t = (num_video_frames - 1) // int(getattr(self.vae, "temporal_downsample_factor", 4)) + 1
        seed = kwargs.get("seed", None)
        generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(
            (batch_size, first_frame_latents.shape[1], latent_t, *first_frame_latents.shape[-2:]),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents[:, :, :1] = first_frame_latents
        schedule_kwargs = dict(
            num_inference_steps=int(
                kwargs.get(
                    "num_inference_steps",
                    getattr(
                        self.video_config, "num_inference_steps", getattr(self.action_config, "num_inference_steps", 10)
                    ),
                )
            ),
            device=device,
            dtype=dtype,
            shift_override=kwargs.get("sigma_shift", None),
        )
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(**schedule_kwargs)
        if joint_denoising:
            actions = torch.randn(
                (batch_size, self.action_horizon, self.action_dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(**schedule_kwargs)
        timestep_action = torch.zeros((batch_size,), device=device, dtype=dtype)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        for step_index, (step_t, step_delta) in enumerate(zip(infer_timesteps, infer_deltas, strict=True)):
            if joint_denoising:
                timestep_action = action_timesteps[step_index].expand(batch_size).to(device=device, dtype=dtype)
            with self._autocast_context(dtype=dtype):
                video_pre = self.video_expert.pre_dit(
                    x=latents,
                    timestep=step_t.expand(batch_size).to(device=device, dtype=dtype),
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                action_pre = self.action_expert.pre_dit(
                    action_tokens=actions,
                    timestep=timestep_action,
                    context=context,
                    context_mask=context_mask,
                )
                tokens_out = self._run_mot(video_pre, action_pre)
                pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
                if joint_denoising:
                    pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
            if joint_denoising:
                actions = self.infer_action_scheduler.step(pred_action, action_deltas[step_index], actions)
            latents = self.infer_video_scheduler.step(pred_video, step_delta, latents)
            latents[:, :, :1] = first_frame_latents

        video = (self._decode_video_latents(latents) * 0.5 + 0.5).clamp_(0, 1)
        return {
            "video": video,
            "normalized_actions": actions.detach().to(device="cpu", dtype=torch.float32).numpy(),
        }
