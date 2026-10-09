import argparse
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import functional as _F
from transformers import AutoTokenizer

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.wan_mot.action_expert import ActionExpert
from starVLA.model.modules.wan_mot.mot import MoT
from starVLA.model.modules.wan_mot.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from starVLA.model.modules.wan_mot.wan_video_expert import WanVideoExpert
from starVLA.model.modules.wan_video.wan_dit import WanModelStateDictConverter
from starVLA.model.modules.wan_video.wan_text_encoder import WanTextEncoder
from starVLA.model.modules.wan_video.wan_vae import Wan2_2_VAE
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.utils.frame_sampling import get_num_future_frames
from starVLA.utils.text_embedding_cache import format_text_prompt, maybe_load_text_embedding_cache

logger = initialize_overwatch(__name__)


def _image_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.array(value, copy=True))
    else:
        tensor = _F.pil_to_tensor(value)
    if tensor.ndim >= 3 and tensor.shape[-1] in (1, 3, 4) and tensor.shape[-3] not in (1, 3, 4):
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 4:
            tensor = tensor.permute(3, 0, 1, 2)
    return tensor


def _to_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _parse_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value).lower()
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


@FRAMEWORK_REGISTRY.register("WanMoT")
class WanMoT(baseframework):
    """Minimal WanMoT framework for starVLA training and action inference."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.config = config
        self.framework_config = config.framework
        self.video_config = config.framework.video_model
        self.action_config = config.framework.action_model
        self.torch_dtype = _parse_dtype(getattr(self.framework_config, "torch_dtype", "bfloat16"))
        self.skip_dit_load_from_pretrain = bool(getattr(self.framework_config, "skip_dit_load_from_pretrain", False))
        self.concat_multi_camera = str(getattr(self.framework_config, "concat_multi_camera", "horizontal"))
        self.view_image_size = getattr(self.framework_config, "view_image_size", None)
        self.view_image_size = None if self.view_image_size is None else tuple(self.view_image_size)
        self.image_size = self._resolve_image_size()
        self.action_horizon = int(self.action_config.action_horizon)
        self.action_dim = int(self.action_config.action_dim)
        self.proprio_dim = getattr(self.action_config, "state_dim", None)
        self.proprio_dim = None if self.proprio_dim is None else int(self.proprio_dim)
        self.text_len = int(getattr(self.video_config, "tokenizer_max_len", getattr(self.video_config, "text_len", 128)))
        self.load_text_encoder = bool(getattr(self.video_config, "load_text_encoder", True))
        self.prompt_template = getattr(self.video_config, "prompt_template", None)
        self.prefer_cached_text_context = bool(getattr(self.video_config, "prefer_cached_text_context", False))
        self.text_mask_padding_as_valid = bool(getattr(self.video_config, "text_mask_padding_as_valid", False))
        self.enable_prompt_cache = bool(getattr(self.video_config, "enable_prompt_cache", True))
        self._cached_prompt_key = None
        self._cached_text_cond = None

        loss_cfg = getattr(self.framework_config, "loss", {})
        self.loss_lambda_video = float(getattr(loss_cfg, "lambda_video", 1.0))
        self.loss_lambda_action = float(getattr(loss_cfg, "lambda_action", 1.0))
        action_supervision_cfg = _to_plain_dict(getattr(self.action_config, "supervision", {}))
        self.action_supervision_target = str(action_supervision_cfg.get("target", "velocity")).lower()
        self.action_loss_weight_mode = str(action_supervision_cfg.get("weight_mode", "scheduler")).lower()

        video_model_config = _to_plain_dict(getattr(self.video_config, "config"))
        action_model_config = _to_plain_dict(getattr(self.action_config, "config"))
        enable_gradient_checkpointing = bool(getattr(self.config.trainer, "enable_gradient_checkpointing", False))
        video_model_config["use_gradient_checkpointing"] = enable_gradient_checkpointing
        action_model_config["use_gradient_checkpointing"] = enable_gradient_checkpointing

        self.text_dim = int(video_model_config["text_dim"])

        self.video_expert = self._build_video_expert(video_model_config).to(dtype=self.torch_dtype)
        self._load_video_expert_if_needed()
        self.action_expert = ActionExpert.from_pretrained(
            action_dit_config=action_model_config,
            action_dit_pretrained_path=getattr(self.action_config, "model_path", None),
            skip_dit_load_from_pretrain=bool(
                getattr(
                    self.action_config,
                    "skip_load_from_pretrain",
                    self.skip_dit_load_from_pretrain,
                )
            ),
            device="cpu",
            torch_dtype=self.torch_dtype,
        )
        if int(self.action_expert.num_heads) != int(self.video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(self.action_expert.attn_head_dim) != int(self.video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(self.action_expert.blocks)) != int(len(self.video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        self.mot = MoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
        self.dit = self.mot

        self.vae = self._build_vae()
        self.text_encoder, self.tokenizer = self._build_text_components()
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(self.torch_dtype)
        else:
            self.proprio_encoder = None

        video_scheduler_cfg = getattr(self.video_config, "scheduler", {})
        action_scheduler_cfg = getattr(self.action_config, "scheduler", {})
        self.train_video_scheduler = self._build_scheduler(video_scheduler_cfg, train=True)
        self.infer_video_scheduler = self._build_scheduler(video_scheduler_cfg, train=False)
        self.train_action_scheduler = self._build_scheduler(action_scheduler_cfg, train=True)
        self.infer_action_scheduler = self._build_scheduler(action_scheduler_cfg, train=False)

    @property
    def num_future_frames(self) -> int:
        return get_num_future_frames(
            self.action_horizon, getattr(self.config.datasets.vla_data, "future_frame_stride", 1)
        )

    def _build_video_expert(self, video_model_config: dict) -> WanVideoExpert:
        return WanVideoExpert(**video_model_config)

    def _resolve_image_size(self) -> Tuple[int, int]:
        """VAE resolution from the view layout; Robotwin uses FastWAM's fixed 384x320 canvas."""
        from starVLA.dataloader.vla.registry import num_video_keys

        view_h, view_w = map(int, self.view_image_size)
        n_view = num_video_keys(self.config.datasets.vla_data.data_mix)
        mode = self.concat_multi_camera
        if mode == "robotwin":
            if n_view != 3:
                raise ValueError(f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {n_view}")
            # FastWAM: main camera 256x320 above two 128x160 wrist cameras.
            return 384, 320
        if mode == "vertical":
            return view_h * n_view, view_w
        if mode == "tri_view":
            return view_h + view_h // 2, view_w
        if mode == "first":
            return view_h, view_w
        if mode == "horizontal":
            return self._horizontal_canvas_size()
        raise ValueError(f"Unsupported WanMoT concat_multi_camera={mode!r}")

    def _horizontal_canvas_size(self) -> Tuple[int, int]:
        from starVLA.dataloader.vla.registry import num_video_keys

        view_h, view_w = map(int, self.view_image_size)
        return view_h, view_w * num_video_keys(self.config.datasets.vla_data.data_mix)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def compile(self):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version.")
        # Gradient checkpointing recomputes activations in backward, which cannot be
        # captured by CUDA graphs ("reduce-overhead"). The checkpoint calls already use
        # `use_reentrant=False`, so Dynamo can trace them under the default mode — pick
        # that mode when checkpointing is on so compile and checkpointing coexist.
        checkpointing = bool(getattr(self.mot, "enable_gradient_checkpointing", False))
        mode = None if checkpointing else "reduce-overhead"
        fullgraph = False
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        self.vae = torch.compile(self.vae, mode=mode, fullgraph=fullgraph)
        self.mot = torch.compile(self.mot, mode=mode, fullgraph=fullgraph)
        self.dit = self.mot

    def _before_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "mot" not in state_dict:
            return state_dict
        converted = {f"mot.{key}": value for key, value in state_dict["mot"].items()}
        if "proprio_encoder" in state_dict:
            converted.update({f"proprio_encoder.{key}": value for key, value in state_dict["proprio_encoder"].items()})
        self._loaded_WanMoT_release_step = state_dict.get("step", None)
        return converted

    def _after_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if hasattr(self, "_loaded_WanMoT_release_step"):
            logger.info("Loaded WanMoT release checkpoint step=%s", self._loaded_WanMoT_release_step)

    def _load_video_expert_if_needed(self) -> None:
        if self.skip_dit_load_from_pretrain:
            logger.info(
                "Skipping WanMoT video DiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "expecting checkpoint override."
            )
            return
        path = getattr(self.video_config, "dit_path", None)
        if not path:
            raise ValueError("framework.video_model.dit_path is required unless skip_dit_load_from_pretrain=true.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"WanMoT video DiT path does not exist: {path}")
        state_dict = WanModelStateDictConverter.load_state_dict(path)
        has_cross_view = True
        if hasattr(self.video_expert, "state_dict_has_cross_view_weights"):
            has_cross_view = self.video_expert.state_dict_has_cross_view_weights(state_dict)
        missing, unexpected = self.video_expert.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("WanMoT video_expert missing keys: %s", missing[:20])
        if unexpected:
            logger.warning("WanMoT video_expert unexpected keys: %s", unexpected[:20])
        copy_cross_view = bool(getattr(self.video_config, "copy_cross_view_from_self_attn", True))
        force_copy_cross_view = bool(getattr(self.video_config, "force_copy_cross_view_from_self_attn", False))
        if (
            hasattr(self.video_expert, "init_cross_view_from_self_attn")
            and copy_cross_view
            and (force_copy_cross_view or not has_cross_view)
        ):
            self.video_expert.init_cross_view_from_self_attn()

    def _build_vae(self) -> nn.Module:
        z_dim = int(getattr(self.video_config, "z_dim", 48))
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2_2_VAE construction currently requires CUDA.")
        vae = Wan2_2_VAE(
            z_dim=z_dim,
            dim=int(getattr(self.video_config, "vae_dim", 160)),
        )
        vae_path = getattr(self.video_config, "vae_path", None)
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"WanMoT VAE path does not exist: {vae_path}")
        vae_sd = WanModelStateDictConverter.load_state_dict(vae_path)
        if "model_state" in vae_sd:
            vae_sd = {"model." + key: value for key, value in vae_sd["model_state"].items()}
        elif not any(key.startswith("model.") for key in vae_sd.keys()):
            vae_sd = {"model." + key: value for key, value in vae_sd.items()}
        vae.load_state_dict(vae_sd, strict=False)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        return vae

    def _load_tokenizer(self, tokenizer_path: str):
        if os.path.exists(tokenizer_path):
            return AutoTokenizer.from_pretrained(tokenizer_path)
        if os.path.isabs(tokenizer_path) or os.path.sep in tokenizer_path:
            raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _build_text_components(self):
        if not self.load_text_encoder:
            logger.info(
                "Skipping WanMoT text encoder/tokenizer load (`load_text_encoder=False`); "
                "training must provide cached `context/context_mask`."
            )
            return None, None
        text_encoder = WanTextEncoder(**_to_plain_dict(getattr(self.video_config, "text_encoder_kwargs", {})))
        text_encoder_path = getattr(self.video_config, "text_encoder_path", None)
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"WanMoT text encoder path does not exist: {text_encoder_path}")
        text_encoder.load_state_dict(WanModelStateDictConverter.load_state_dict(text_encoder_path), strict=False)
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
        tokenizer = self._load_tokenizer(str(getattr(self.video_config, "tokenizer_path", "google/umt5-xxl")))
        return text_encoder, tokenizer

    @staticmethod
    def _build_scheduler(scheduler_cfg: Any, *, train: bool) -> WanContinuousFlowMatchScheduler:
        cfg = _to_plain_dict(scheduler_cfg)
        shift_key = "train_shift" if train else "infer_shift"
        return WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(cfg.get("num_train_timesteps", 1000)),
            shift=float(cfg.get(shift_key, cfg.get("shift", 5.0))),
            sample_mode=str(cfg.get("sample_mode", "shift")) if train else "shift",
            noise_beta_alpha=float(cfg.get("noise_beta_alpha", 1.5)),
            noise_beta_beta=float(cfg.get("noise_beta_beta", 1.0)),
            noise_s=float(cfg.get("noise_s", 0.999)),
        )

    def _autocast_context(self, dtype: Optional[torch.dtype] = None):
        device = next(self.video_expert.parameters()).device
        if device.type != "cuda":
            return nullcontext()
        if dtype == torch.float32:
            return torch.autocast("cuda", enabled=False)
        if dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    def _batch_to_tensor(
        self,
        value: Any,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, list):
            value = torch.stack([item if torch.is_tensor(item) else torch.as_tensor(item) for item in value])
        elif not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if device is None and dtype is None:
            return value
        to_kwargs = {}
        if device is not None:
            to_kwargs["device"] = device
            to_kwargs["non_blocking"] = True
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        return value.to(**to_kwargs)

    def _prepare_batch_images(self, batch_images: Any) -> torch.Tensor:
        if isinstance(batch_images, torch.Tensor):
            images = batch_images.detach().clone()
        elif isinstance(batch_images, np.ndarray):
            images = torch.as_tensor(np.array(batch_images, copy=True))
        else:
            images = torch.stack(
                [torch.stack([_image_to_tensor(image) for image in images_i], dim=0) for images_i in batch_images],
                dim=0,
            )

        if images.ndim == 5 and images.shape[-1] in (1, 3, 4) and images.shape[2] not in (1, 3, 4):
            images = images.permute(0, 1, 4, 2, 3)
        elif images.ndim == 6 and images.shape[-1] in (1, 3, 4) and images.shape[2] not in (1, 3, 4):
            images = images.permute(0, 1, 5, 2, 3, 4)

        if images.ndim == 5:
            images = images.unsqueeze(3)
        if images.ndim != 6:
            raise ValueError(f"Expected images shape [B,V,C,H,W] or [B,V,C,T,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"Expected RGB images with channel dim 3, got {tuple(images.shape)}")
        return images.contiguous()

    def _resize_views_if_needed(self, images: torch.Tensor) -> torch.Tensor:
        if self.view_image_size is None:
            return images
        target_h, target_w = int(self.view_image_size[0]), int(self.view_image_size[1])
        if self.concat_multi_camera == "robotwin":
            # Match FastWAM's ToTensor -> Resize, including raw uint8 inference inputs.
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            return self._resize_rgb_frames(images, (target_h, target_w))
        if images.shape[-2:] == (target_h, target_w):
            return images
        batch_size, n_view, channels, frames, _, _ = images.shape
        flat = images.reshape(batch_size * n_view, channels, frames, images.shape[-2], images.shape[-1]).float()
        flat = F.interpolate(
            flat,
            size=(frames, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        return flat.reshape(batch_size, n_view, channels, frames, target_h, target_w)

    @staticmethod
    def _resize_rgb_frames(images: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Resize spatial dimensions independently for every RGB channel and frame."""
        shape = images.shape
        flat = images.reshape(-1, 1, *shape[-2:]).float()
        flat = _F.resize(flat, list(size), interpolation=_F.InterpolationMode.BILINEAR, antialias=True)
        return flat.reshape(*shape[:-2], *size)

    def _prepare_view_mask(self, view_mask: Any, batch_size: int, n_view: int, device: torch.device) -> torch.Tensor:
        if view_mask is None:
            return torch.ones(batch_size, n_view, dtype=torch.bool, device=device)
        masks = []
        for mask in view_mask:
            if torch.is_tensor(mask):
                mask_tensor = mask.detach().clone().reshape(-1)
            else:
                mask_tensor = torch.as_tensor(np.array(mask, copy=True)).reshape(-1)
            masks.append(mask_tensor)
        mask = torch.stack(masks, dim=0).to(device=device, dtype=torch.bool)
        if mask.shape != (batch_size, n_view):
            raise ValueError(f"Expected view_mask shape {(batch_size, n_view)}, got {tuple(mask.shape)}")
        return mask

    def _concat_views(self, images: torch.Tensor, view_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError(f"`images` must be [B,V,C,T,H,W], got {tuple(images.shape)}")
        batch_size, n_view, _, _, _, _ = images.shape
        if view_mask is None:
            view_mask = torch.ones(batch_size, n_view, dtype=torch.bool, device=images.device)
        if view_mask.shape != (batch_size, n_view):
            raise ValueError(f"Expected view_mask shape {(batch_size, n_view)}, got {tuple(view_mask.shape)}")
        mode = self.concat_multi_camera
        if mode == "robotwin":
            if n_view != 3:
                raise ValueError(f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {n_view}")
            cam_top = self._resize_rgb_frames(images[:, 0], (256, 320))
            cam_left = self._resize_rgb_frames(images[:, 1], (128, 160))
            cam_right = self._resize_rgb_frames(images[:, 2], (128, 160))
            return torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)
        if mode == "first":
            return images[:, 0]
        if mode == "horizontal":
            return torch.cat([images[:, index] for index in range(n_view)], dim=-1)
        if mode == "vertical":
            return torch.cat([images[:, index] for index in range(n_view)], dim=-2)
        if mode == "tri_view":
            if n_view != 3:
                raise ValueError(f"`concat_multi_camera='tri_view'` requires exactly 3 views, got {n_view}.")
            _, _, _, frames, height, width = images.shape
            if height % 2 != 0 or width % 2 != 0:
                raise ValueError(f"`tri_view` requires even view height/width, got {(height, width)}.")
            cam_top = images[:, 0]
            bottom_size = (frames, height // 2, width // 2)
            cam_left = F.interpolate(
                images[:, 1].float(),
                size=bottom_size,
                mode="trilinear",
                align_corners=False,
            )
            cam_right = F.interpolate(
                images[:, 2].float(),
                size=bottom_size,
                mode="trilinear",
                align_corners=False,
            )
            return torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)
        raise ValueError(f"Unsupported WanMoT concat_multi_camera={mode!r}")

    def _normalize_rgb_views_for_vae(self, images: torch.Tensor) -> torch.Tensor:
        if getattr(self, "stat_key", None) == "libero":
            parameter = next(self.video_expert.parameters())
            images = images.to(device=parameter.device, dtype=parameter.dtype)
            return images * (2.0 / 255.0) - 1.0
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0
        return images * 2.0 - 1.0

    def _resize_video_for_vae(self, video: torch.Tensor) -> torch.Tensor:
        target_h, target_w = int(self.image_size[0]), int(self.image_size[1])
        if video.shape[-2:] != (target_h, target_w):
            video = F.interpolate(
                video,
                size=(video.shape[2], target_h, target_w),
                mode="trilinear",
                align_corners=False,
            )
        if target_h % 16 != 0 or target_w % 16 != 0:
            raise ValueError(f"WanMoT image_size must be multiples of 16, got {(target_h, target_w)}")
        return video

    def _encode_text(
        self, prompts: Sequence[str], device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prompts)
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "WanMoT prompt encoding requires loaded text encoder/tokenizer. "
                "Set `framework.video_model.load_text_encoder=true` or provide cached `context/context_mask`."
            )
        tokens = self.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device=device, dtype=torch.bool)
        with torch.inference_mode():
            context = self.text_encoder(input_ids, mask=mask).to(dtype=dtype)
        context = context.masked_fill(~mask[:, :, None], 0)
        if self.text_mask_padding_as_valid:
            mask = torch.ones_like(mask)
        return context, mask

    def _format_prompt(self, instruction: str) -> str:
        return format_text_prompt(str(instruction))

    def _encode_text_cached(
        self, prompts: Sequence[str], device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_key = tuple(prompts)
        if self.enable_prompt_cache and self._cached_prompt_key == prompt_key and self._cached_text_cond is not None:
            context, mask = self._cached_text_cond
            return context.to(device=device, dtype=dtype), mask.to(device=device)
        context, mask = self._encode_text(prompts, device, dtype)
        if self.enable_prompt_cache:
            self._cached_prompt_key = prompt_key
            self._cached_text_cond = (context.detach().cpu(), mask.detach().cpu())
        return context, mask

    def _prepare_cached_text_context(
        self,
        examples: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = torch.stack([self._batch_to_tensor(example["context"]) for example in examples])
        context_mask = torch.stack([self._batch_to_tensor(example["context_mask"]) for example in examples])
        if context.ndim != 3:
            raise ValueError(f"Cached `context` must be [B,L,D], got {tuple(context.shape)}")
        if context_mask.ndim != 2:
            raise ValueError(f"Cached `context_mask` must be [B,L], got {tuple(context_mask.shape)}")
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                "Cached `context/context_mask` shape mismatch: "
                f"context={tuple(context.shape)}, context_mask={tuple(context_mask.shape)}"
            )
        if context.shape[-1] != self.text_dim:
            raise ValueError(f"Cached `context` dim must be {self.text_dim}, got {context.shape[-1]}")
        context = context.to(device=device, dtype=dtype, non_blocking=True)
        context_mask = context_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        context = context.masked_fill(~context_mask[:, :, None], 0)
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def _prepare_text_context(
        self,
        examples: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        has_context = ["context" in example for example in examples]
        has_context_mask = ["context_mask" in example for example in examples]
        if any(context_present != mask_present for context_present, mask_present in zip(has_context, has_context_mask)):
            raise ValueError("WanMoT cached text inputs require both `context` and `context_mask` in each sample.")
        if (self.enable_prompt_cache or self.prefer_cached_text_context) and any(has_context):
            if not all(has_context):
                raise ValueError("WanMoT cached text inputs must be present for every sample in the batch.")
            return self._prepare_cached_text_context(examples, device, dtype)
        prompts = [self._format_prompt(example["lang"]) for example in examples]
        return self._encode_text_cached(prompts, device, dtype)

    def _prepare_runtime_text_context(
        self,
        context: Optional[Any],
        context_mask: Optional[Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        caller: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if (context is None) != (context_mask is None):
            raise ValueError(f"WanMoT.{caller} cached text inputs require both `context` and `context_mask`.")
        if context is None:
            return None
        context = self._batch_to_tensor(context, device=device, dtype=dtype)
        context_mask = self._batch_to_tensor(context_mask, device=device, dtype=torch.bool)
        if context.ndim != 3:
            raise ValueError(f"Cached `context` must be [B,L,D], got {tuple(context.shape)}")
        if context_mask.ndim != 2:
            raise ValueError(f"Cached `context_mask` must be [B,L], got {tuple(context_mask.shape)}")
        if context.shape[0] != batch_size:
            raise ValueError(f"Cached `context` batch size must be {batch_size}, got {context.shape[0]}")
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                "Cached `context/context_mask` shape mismatch: "
                f"context={tuple(context.shape)}, context_mask={tuple(context_mask.shape)}"
            )
        if context.shape[-1] != self.text_dim:
            raise ValueError(f"Cached `context` dim must be {self.text_dim}, got {context.shape[-1]}")
        context = context.masked_fill(~context_mask[:, :, None], 0)
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def _load_runtime_text_context_from_cache(
        self,
        instructions: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
        caller: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        vla_data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        cache_dir = getattr(vla_data_cfg, "text_embedding_cache_dir", None)
        if not cache_dir:
            return None
        context_len = int(getattr(vla_data_cfg, "text_context_len", self.text_len))
        encoder_id = str(getattr(vla_data_cfg, "text_cache_encoder_id", "wan22ti2v5b"))
        require_cache = bool(getattr(vla_data_cfg, "require_text_embedding_cache", False))
        contexts = []
        masks = []
        for instruction in instructions:
            cached_text = maybe_load_text_embedding_cache(
                cache_dir,
                str(instruction),
                context_len=context_len,
                encoder_id=encoder_id,
                required=require_cache,
            )
            if cached_text is None:
                return None
            context, mask = cached_text
            contexts.append(context)
            masks.append(mask)
        return self._prepare_runtime_text_context(
            torch.stack(contexts, dim=0),
            torch.stack(masks, dim=0),
            len(contexts),
            device,
            dtype,
            caller,
        )

    def _prepare_runtime_text_context_with_cache(
        self,
        context: Optional[Any],
        context_mask: Optional[Any],
        instructions: Sequence[str],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        caller: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_context = self._prepare_runtime_text_context(
            context,
            context_mask,
            batch_size,
            device,
            dtype,
            caller,
        )
        if text_context is not None:
            return text_context
        if len(instructions) != batch_size:
            raise ValueError(f"WanMoT.{caller} expected {batch_size} instruction(s), got {len(instructions)}.")
        cached_context = self._load_runtime_text_context_from_cache(instructions, device, dtype, caller)
        if cached_context is not None:
            return cached_context
        prompts = [self._format_prompt(instruction) for instruction in instructions]
        return self._encode_text(prompts, device, dtype)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("WanMoT state_dim is enabled but no `state`/`proprio` input was provided.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be [B,D] or [B,T,D], got {tuple(proprio.shape)}")
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[-1]}")
        proprio_token = self.proprio_encoder(proprio.to(device=context.device, dtype=context.dtype)).unsqueeze(1)
        proprio_mask = torch.ones(context.shape[0], 1, dtype=torch.bool, device=context.device)
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    @torch.no_grad()
    def _run_vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        out_device = video.device
        out_dtype = video.dtype
        try:
            vae_param = next(self.vae.parameters())
            video = video.to(device=vae_param.device, dtype=vae_param.dtype)
        except StopIteration:
            pass
        if hasattr(self.vae, "single_encode"):
            latents = self.vae.single_encode(video)
        else:
            latents = self.vae.encode(video)
        return latents.to(device=out_device, dtype=out_dtype)

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        return self._run_vae_encode(video)

    @torch.no_grad()
    def _encode_first_frame_latents(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            video = video.unsqueeze(2)
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError(f"`video` must be [B,3,1,H,W] or [B,3,H,W], got {tuple(video.shape)}")
        return self._encode_video_latents(video)

    @torch.no_grad()
    def _run_vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        out_device = latents.device
        out_dtype = latents.dtype
        try:
            vae_param = next(self.vae.parameters())
            latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        except StopIteration:
            pass
        if hasattr(self.vae, "single_decode"):
            video = self.vae.single_decode(latents)
        else:
            video = self.vae.decode(latents)
        return video.to(device=out_device, dtype=out_dtype)

    @torch.no_grad()
    def _decode_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self._run_vae_decode(latents)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, : min(video_tokens_per_frame, video_seq_len)] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        temporal_factor = int(getattr(self.vae, "temporal_downsample_factor", 4))
        tail_is_pad = image_is_pad[:, 1:]
        if tail_is_pad.shape[1] % temporal_factor != 0:
            raise ValueError(
                f"Cannot align image_is_pad shape {tuple(image_is_pad.shape)} with temporal factor {temporal_factor}."
            )
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        video_is_pad = (
            torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
            if include_initial_video_step
            else latent_tail_is_pad
        )
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(f"Video-loss mask mismatch: mask={video_is_pad.shape[1]} loss={video_loss_token.shape[1]}")
        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        return (video_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _compute_action_loss_per_sample(
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_mask is None:
            return action_loss.mean(dim=(1, 2))
        if action_mask.ndim == 3:
            if action_mask.shape != action_loss.shape:
                raise ValueError(
                    "`action_mask` shape must match action loss shape for per-dim masking: "
                    f"mask={tuple(action_mask.shape)}, loss={tuple(action_loss.shape)}"
                )
            valid = action_mask.to(device=action_loss.device, dtype=action_loss.dtype)
            return (action_loss * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1.0)
        elif action_mask.ndim == 2:
            action_loss_token = action_loss.mean(dim=2)
            valid = action_mask
        else:
            raise ValueError(f"`action_mask` must be [B,T,D] or [B,T], got {tuple(action_mask.shape)}")
        if valid.shape != action_loss_token.shape:
            raise ValueError(
                "`action_mask` shape must match action timestep loss shape: "
                f"mask={tuple(valid.shape)}, loss={tuple(action_loss_token.shape)}"
            )
        valid = valid.to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        return (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _run_mot(
        self,
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        return self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        del kwargs
        if examples is None:
            raise ValueError("WanMoT.forward requires `examples`.")
        if not bool(getattr(self.config.datasets.vla_data, "use_future_frames", False)):
            raise ValueError("WanMoT training requires datasets.vla_data.use_future_frames=true.")
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        images = self._prepare_batch_images([example["image"] for example in examples])
        images = self._resize_views_if_needed(images)
        batch_size, n_view, _, num_frames, _, _ = images.shape
        if num_frames <= 1 or num_frames % 4 != 1:
            raise ValueError(f"WanMoT video frames must satisfy T > 1 and T % 4 == 1, got T={num_frames}.")
        view_mask = self._prepare_view_mask(
            [example["view_mask"] for example in examples] if "view_mask" in examples[0] else None,
            batch_size,
            n_view,
            images.device,
        )
        images = self._normalize_rgb_views_for_vae(images)
        video = self._concat_views(images, view_mask)
        video = self._resize_video_for_vae(video).to(device=device, dtype=dtype, non_blocking=True)

        context, context_mask = self._prepare_text_context(examples, device, dtype)
        state = None
        if "state" in examples[0]:
            state = torch.stack([self._batch_to_tensor(example["state"]) for example in examples])
            state = state.to(device=device, dtype=dtype, non_blocking=True)
        context, context_mask = self._append_proprio_to_context(context, context_mask, state)

        actions = torch.stack([self._batch_to_tensor(example["actions"]) for example in examples])
        actions = actions[:, -self.action_horizon :, :].to(device=device, dtype=dtype, non_blocking=True)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {actions.shape[-1]}.")
        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = torch.stack([self._batch_to_tensor(example["action_mask"]) for example in examples])
            action_mask = action_mask[:, -self.action_horizon :, :].to(device=device, dtype=torch.bool)

        image_is_pad = None
        if "image_is_pad" in examples[0]:
            image_is_pad = torch.stack([self._batch_to_tensor(example["image_is_pad"]) for example in examples])
            image_is_pad = image_is_pad[:, :num_frames].to(device=device, dtype=torch.bool)

        with torch.no_grad():
            input_latents = self._encode_video_latents(video)
        input_latents = input_latents.to(device=device, dtype=dtype)
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(batch_size, device=device, dtype=dtype)
        noisy_video = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        first_frame_latents = input_latents[:, :, :1]
        noisy_video[:, :, :1] = first_frame_latents

        noise_action = torch.randn_like(actions)
        timestep_action = self.train_action_scheduler.sample_training_t(batch_size, device=device, dtype=dtype)
        noisy_action = self.train_action_scheduler.add_noise(actions, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(actions, noise_action, timestep_action)

        with self._autocast_context(dtype=dtype):
            video_pre = self.video_expert.pre_dit(
                x=noisy_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
                ),  # 第 0 帧 timestamp 应该为 0
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            tokens_out = self._run_mot(video_pre, action_pre)
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action_velocity = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = False
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        if self.action_supervision_target == "action_space":
            pred_action_clean = self.train_action_scheduler.recover_sample_from_velocity(
                noisy_samples=noisy_action,
                pred_velocity=pred_action_velocity,
                timestep=timestep_action,
            )
            loss_action_per_sample = self._compute_action_loss_per_sample(
                pred_action_clean,
                actions,
                action_mask,
            )
        else:
            loss_action_per_sample = self._compute_action_loss_per_sample(
                pred_action_velocity,
                target_action,
                action_mask,
            )
        if self.action_loss_weight_mode == "none":
            action_weight = torch.ones_like(loss_action_per_sample)
        else:
            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                device=loss_action_per_sample.device,
                dtype=loss_action_per_sample.dtype,
            )
        loss_action = (loss_action_per_sample * action_weight).mean()
        total_loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return {
            "total_loss": total_loss,
            "action_loss": float(loss_action.detach().item()),
            "video_loss": float(loss_video.detach().item()),
        }

    @torch.inference_mode()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: List[dict],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        with self._autocast_context(dtype=latents_action.dtype):
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            return self.action_expert.post_dit(action_tokens, action_pre)

    def preprocess(self, data: dict, stat_key=None, inplace=False):
        # FastWAM's RoboTwin evaluator resizes raw cameras directly to their
        # canvas slots. Training's per-view resize would otherwise run first.
        effective_key = stat_key or getattr(self, "stat_key", None)
        if effective_key != "robotwin" or self.concat_multi_camera != "robotwin":
            return super().preprocess(data, stat_key=stat_key, inplace=inplace)
        data_processed = data if inplace else data.copy()
        images = data_processed.pop("batch_images", None)
        data_processed = super().preprocess(data_processed, stat_key=stat_key, inplace=True)
        if images is not None:
            data_processed["batch_images"] = images
        return data_processed

    def _prepare_robotwin_eval_video(self, batch_images, *, device, dtype):
        """Match FastWAM's PIL bilinear resize and bf16 pixel normalization."""
        videos = []
        for images in batch_images:
            if len(images) != 3:
                raise ValueError("RoboTwin evaluation requires head/left/right cameras")
            resized = []
            for image, size in zip(images, ((320, 256), (160, 128), (160, 128)), strict=True):
                image = _F.to_pil_image(_image_to_tensor(image))
                resized.append(np.asarray(image.resize(size, resample=Image.Resampling.BILINEAR)))
            canvas = np.concatenate([resized[0], np.concatenate(resized[1:], axis=1)], axis=0)
            videos.append(torch.from_numpy(canvas).permute(2, 0, 1))
        video = torch.stack(videos).unsqueeze(2).to(device=device, dtype=dtype)
        return video * (2.0 / 255.0) - 1.0

    def _sample_inference_noise(self, shape, *, device, dtype, seed):
        reference_noise = getattr(self, "stat_key", None) in ("libero", "robotwin")
        rand_device = "cpu" if reference_noise else device
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(int(seed))
        noise_dtype = torch.float32 if reference_noise else dtype
        return torch.randn(shape, device=rand_device, dtype=noise_dtype, generator=generator).to(
            device=device, dtype=dtype
        )

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Any,
        view_mask: Optional[Any],
        instructions: List[str],
        fps: Optional[List[float]] = None,
        state: Optional[Any] = None,
        actions: Optional[Any] = None,
        data_id: Optional[Any] = None,
        context: Optional[Any] = None,
        context_mask: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        del fps, data_id
        if actions is not None:
            raise ValueError("WanMoT.predict_action does not support action prefixes in the first implementation.")
        self.eval()
        action_horizon = int(kwargs.get("action_horizon", self.action_horizon))
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        if getattr(self, "stat_key", None) == "robotwin" and self.concat_multi_camera == "robotwin":
            video = self._prepare_robotwin_eval_video(batch_images, device=device, dtype=dtype)
            batch_size = video.shape[0]
        else:
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
            "predict_action",
        )
        state_tensor = self._batch_to_tensor(state, device=device, dtype=dtype)
        context, context_mask = self._append_proprio_to_context(context, context_mask, state_tensor)

        first_frame_latents = self._encode_first_frame_latents(video)
        timestep_video = torch.zeros((batch_size,), device=device, dtype=dtype)
        with self._autocast_context(dtype=dtype):
            video_pre = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_horizon,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
        seed = kwargs.get("seed", getattr(getattr(self.action_config, "predict", None), "seed", None))
        latents_action = self._sample_inference_noise(
            (batch_size, action_horizon, self.action_dim),
            device=device,
            dtype=dtype,
            seed=seed,
        )
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(
                kwargs.get("num_inference_steps", getattr(self.action_config, "num_inference_steps", 10))
            ),
            device=device,
            dtype=dtype,
            shift_override=kwargs.get("sigma_shift", None),
        )
        for step_t_action, step_delta_action in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t_action.expand(batch_size).to(device=device, dtype=dtype)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        return {"normalized_actions": latents_action.detach().to(device="cpu", dtype=torch.float32).numpy()}

    @torch.inference_mode()
    def predict_video(
        self,
        batch_images: Any,
        view_mask: Optional[Any],
        instructions: List[str],
        fps: Optional[List[float]] = None,
        context: Optional[Any] = None,
        context_mask: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        del fps
        self.eval()
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

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

        default_num_video_frames = self.num_future_frames + 1
        num_video_frames = int(kwargs.get("num_video_frames", default_num_video_frames))
        if num_video_frames <= 1 or num_video_frames % 4 != 1:
            raise ValueError(f"WanMoT video inference requires T > 1 and T % 4 == 1, got T={num_video_frames}.")

        first_frame_latents = self._encode_first_frame_latents(video)
        latent_t = (num_video_frames - 1) // int(getattr(self.vae, "temporal_downsample_factor", 4)) + 1
        latent_h = video.shape[-2] // int(getattr(self.vae, "upsampling_factor", 16))
        latent_w = video.shape[-1] // int(getattr(self.vae, "upsampling_factor", 16))

        seed = kwargs.get("seed", None)
        generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(
            (batch_size, int(getattr(self.vae, "z_dim", self.video_config.z_dim)), latent_t, latent_h, latent_w),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents[:, :, :1] = first_frame_latents

        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
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
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep = step_t.expand(batch_size).to(device=device, dtype=dtype)
            with self._autocast_context(dtype=dtype):
                pred_video = self.video_expert(
                    x=latents,
                    timestep=timestep,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
            latents = self.infer_video_scheduler.step(pred_video, step_delta, latents)
            latents[:, :, :1] = first_frame_latents

        video = self._decode_video_latents(latents)
        video = (video * 0.5 + 0.5).clamp_(0, 1)
        return {"video": video}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/vla/starvla_wam.yaml",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    OmegaConf.resolve(cfg)
    model = WanMoT(cfg)
    print("constructed WanMoT")
    print(
        "action_dim",
        model.action_dim,
        "state_dim",
        model.proprio_dim,
        "text_dim",
        model.text_dim,
    )
    print(
        "video_layers",
        len(model.video_expert.blocks),
        "action_layers",
        len(model.action_expert.blocks),
    )
    print(
        "text_encoder_loaded",
        model.text_encoder is not None,
        "tokenizer_loaded",
        model.tokenizer is not None,
    )
