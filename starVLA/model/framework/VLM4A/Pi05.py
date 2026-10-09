# Copyright 2025 starVLA community and 2026 The RLinf Authors.
# Licensed under the MIT License and Apache License 2.0, respectively.

"""Pi0.5 framework adapter for starVLA's existing VLA batch contract."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.pi05 import Observation, Pi05, Pi05Config
from starVLA.model.modules.pi05.checkpoint import load_pi05_weights
from starVLA.model.modules.pi05.subtask import resolve_subtask
from starVLA.model.modules.pi05.tokenizer import Pi05Tokenizer
from starVLA.model.tools import FRAMEWORK_REGISTRY

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass
class Pi05DefaultConfig:
    name: str = "Pi05"
    model: dict = field(
        default_factory=lambda: {
            "model_path": None,
            "tokenizer_path": None,
            "dtype": "bfloat16",
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "vision_variant": "So400m/14",
            "max_token_len": 200,
            "num_images": 3,
            "fast_tokenizer_path": "physical-intelligence/fast",
            "fast_max_tokens": None,
            "action_ce_weight": 0.01,
            "subtask_enabled": False,
            "subtask_ce_weight": 1.0,
            "subtask_max_tokens": 64,
            "infer_mode": "flow",
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 7,
            "state_dim": 7,
            "model_action_dim": 32,
            "action_horizon": 10,
            "num_inference_steps": 10,
            "rtc_training_max_delay": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("Pi05")
class Pi05Framework(baseframework):
    """Pi0.5 model with no Pi0 compatibility branches."""

    def __init__(self, config: Any = None, **kwargs) -> None:
        del kwargs
        super().__init__()
        self.config = merge_framework_config(Pi05DefaultConfig, config)
        model_cfg = self.config.framework.model
        action_cfg = self.config.framework.action_model

        self.action_dim = int(action_cfg.action_dim)
        self.model_action_dim = int(action_cfg.model_action_dim)
        self.action_horizon = int(action_cfg.action_horizon)
        self.num_inference_steps = int(action_cfg.num_inference_steps)
        self.num_images = int(model_cfg.num_images)
        datasets_cfg = self._get_config_value(self.config, "datasets")
        vla_data_cfg = self._get_config_value(datasets_cfg, "vla_data")
        self.disable_state = bool(self._get_config_value(vla_data_cfg, "disable_state", True))

        core_config = Pi05Config(
            action_dim=self.model_action_dim,
            action_horizon=self.action_horizon,
            max_token_len=int(model_cfg.max_token_len),
            dtype=str(model_cfg.dtype),
            paligemma_variant=str(model_cfg.paligemma_variant),
            action_expert_variant=str(model_cfg.action_expert_variant),
            vision_variant=str(model_cfg.vision_variant),
            rtc_training_max_delay=int(action_cfg.rtc_training_max_delay),
        )
        self.pi05 = self._build_model(core_config)
        if model_cfg.model_path:
            load_pi05_weights(self.pi05, model_cfg.model_path)
        self.pi05.to(dtype=core_config.torch_dtype)
        trainer_cfg = getattr(self.config, "trainer", None)
        if bool(getattr(trainer_cfg, "enable_gradient_checkpointing", False)):
            self.pi05.gradient_checkpointing_enable({"use_reentrant": False})
        self._tokenizer_path = model_cfg.tokenizer_path
        self._tokenizer = None
        self._fast_processor = None
        self.action_ce_weight = float(model_cfg.action_ce_weight)
        self.fast_max_tokens = None if model_cfg.fast_max_tokens is None else int(model_cfg.fast_max_tokens)
        if self.fast_max_tokens is not None and self.fast_max_tokens < 1:
            raise ValueError("fast_max_tokens must be positive or null")
        self.subtask_enabled = bool(model_cfg.subtask_enabled)
        self.subtask_ce_weight = float(model_cfg.subtask_ce_weight)
        self.subtask_max_tokens = int(model_cfg.subtask_max_tokens)
        if self.subtask_max_tokens < 1:
            raise ValueError("subtask_max_tokens must be positive")

    def _build_model(self, config: Pi05Config) -> Pi05:
        return Pi05(config)

    @property
    def device(self) -> torch.device:
        return next(self.pi05.parameters()).device

    def _get_tokenizer(self) -> Pi05Tokenizer:
        if self._tokenizer is None:
            tokenizer_path = self._tokenizer_path or os.environ.get("OPENPI_TOKENIZER_PATH")
            if tokenizer_path is None:
                raise ValueError(
                    "Set framework.model.tokenizer_path or OPENPI_TOKENIZER_PATH " "to paligemma_tokenizer.model."
                )
            self._tokenizer = Pi05Tokenizer(tokenizer_path, max_length=self.pi05.max_token_len)
        return self._tokenizer

    def compile(self) -> None:
        """Compile the tensor-only Pi0.5 core before distributed wrapping."""
        # Token CE also uses eager checkpoint recomputation and variable-length
        # tensors across the compile boundary; avoid private CUDA-graph pools.
        checkpointing = bool(self.pi05.llm.gradient_checkpointing or self.pi05.img.encoder.gradient_checkpointing)
        mode = None if checkpointing or self.action_ce_weight or self.subtask_enabled else "reduce-overhead"
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        self.pi05 = torch.compile(self.pi05, mode=mode, fullgraph=False)

    def _prepare_observation(self, examples: list[dict]) -> Observation:
        images: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for view_index, key in enumerate(_IMAGE_KEYS[: self.num_images]):
            view_tensors = []
            view_masks = []
            for example in examples:
                example_images = example["image"]
                if not isinstance(example_images, (list, tuple)):
                    example_images = [example_images]
                if view_index < len(example_images):
                    view_tensors.append(_image_to_hwc(example_images[view_index]))
                    example_mask = example.get("view_mask")
                    view_masks.append(True if example_mask is None else bool(example_mask[view_index]))
                else:
                    view_tensors.append(torch.zeros(224, 224, 3))
                    view_masks.append(False)
            images[key] = torch.stack(view_tensors).to(self.device)
            masks[key] = torch.tensor(view_masks, dtype=torch.bool, device=self.device)

        raw_states = []
        tokens = []
        token_masks = []
        tokenizer = self._get_tokenizer()
        for example in examples:
            state = example.get("state")
            if state is None:
                state = torch.zeros(self.action_dim, dtype=torch.float32)
            state = torch.as_tensor(state, dtype=torch.float32).reshape(-1).cpu()
            raw_states.append(state)
            token_state = None if self.disable_state else state.numpy()
            if example.get("_generate_subtask", False):
                token_ids, token_mask = tokenizer.tokenize(example["lang"], token_state, generate_subtask=True)
            else:
                token_ids, token_mask = tokenizer.tokenize(example["lang"], token_state)
            tokens.append(torch.from_numpy(token_ids))
            token_masks.append(torch.from_numpy(token_mask))

        state = torch.stack(
            [
                F.pad(
                    item[: self.model_action_dim],
                    (0, max(0, self.model_action_dim - item.numel())),
                )
                for item in raw_states
            ]
        ).to(self.device)
        return Observation(
            images=images,
            image_masks=masks,
            state=state,
            tokenized_prompt=torch.stack(tokens).to(self.device),
            tokenized_prompt_mask=torch.stack(token_masks).to(self.device),
        )

    def _prepare_actions(self, examples: list[dict]) -> torch.Tensor:
        actions = [example.get("actions", example.get("action")) for example in examples]
        actions = torch.stack([torch.as_tensor(action) for action in actions]).float()
        actions = actions[:, -self.action_horizon :, : self.action_dim]
        return F.pad(actions, (0, self.model_action_dim - actions.shape[-1])).to(self.device)

    def forward(self, examples: list[dict], **kwargs) -> dict[str, torch.Tensor]:
        targets = [resolve_subtask(example) for example in examples] if self.subtask_enabled else None
        if targets is not None:
            examples = [
                dict(example, _generate_subtask=target is not None)
                for example, target in zip(examples, targets, strict=True)
            ]
        observation = self._prepare_observation(examples)
        actions = self._prepare_actions(examples)
        token_kwargs = {}
        if self.action_ce_weight:
            if self._fast_processor is None:
                from starVLA.model.modules.action_model.fast_ActionHeader import _load_fast_processor

                self._fast_processor = _load_fast_processor(self.config.framework.model.fast_tokenizer_path)
            token_ids, token_mask = self._get_tokenizer().tokenize_actions(
                actions[:, :, : self.action_dim].detach().float().cpu().numpy(),
                self._fast_processor,
                max_length=self.fast_max_tokens,
            )
            token_kwargs.update(
                action_tokens=torch.from_numpy(token_ids).to(self.device),
                action_token_mask=torch.from_numpy(token_mask).to(self.device),
            )
        if targets is not None:
            token_ids = torch.zeros(len(examples), self.subtask_max_tokens, dtype=torch.long)
            token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
            for index, target in enumerate(targets):
                if target is not None:
                    sequence = self._get_tokenizer().tokenize_subtask(target, self.subtask_max_tokens)
                    token_ids[index, : len(sequence)] = torch.tensor(sequence)
                    token_mask[index, : len(sequence)] = True
            token_kwargs.update(subtask_tokens=token_ids.to(self.device), subtask_token_mask=token_mask.to(self.device))
        # Calling the module (rather than compute_loss directly) enters the
        # OptimizedModule.forward installed by compile().
        losses = self.pi05(
            observation,
            actions,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
            generator=kwargs.get("generator"),
            **token_kwargs,
        )
        loss = losses["flow_loss"]
        flow_mask = losses.get("flow_loss_mask")
        action_ce_loss = losses.get("action_ce_loss", loss.new_zeros(()))
        subtask_ce_loss = losses.get("subtask_ce_loss", loss.new_zeros(()))
        if "action_mask" in examples[0]:
            temporal_mask = (
                torch.stack([torch.as_tensor(example["action_mask"]) for example in examples])[:, -self.action_horizon :]
                .any(dim=-1)
                .to(loss.device)
            )
            flow_mask = temporal_mask if flow_mask is None else flow_mask & temporal_mask
        if flow_mask is None:
            action_loss = loss.mean()
        else:
            action_loss = (loss * flow_mask).sum() / flow_mask.sum().clamp_min(1)
        return {
            "total_loss": (
                action_loss + self.action_ce_weight * action_ce_loss + self.subtask_ce_weight * subtask_ce_loss
            ),
            "action_loss": action_loss,
            "action_ce_loss": action_ce_loss,
            "subtask_ce_loss": subtask_ce_loss,
        }

    def preprocess(self, data: dict, stat_key=None, inplace=False):
        """Adapt the shared client's environment-space RTC prefix after action transforms."""
        processed = super().preprocess(data, stat_key=stat_key, inplace=inplace)
        if "actions" in processed:
            processed["prev_chunk_left_over"] = processed.pop("actions")
        if "delay" in processed:
            processed["inference_delay"] = processed.pop("delay")
        return processed

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] | dict | None = None, **kwargs) -> dict[str, Any]:
        if examples is None:
            batch_images = kwargs.pop("batch_images")
            instructions = kwargs.pop("instructions")
            states = kwargs.pop("state", None)
            view_masks = kwargs.pop("view_mask", None)
            examples = []
            for index, (images, instruction) in enumerate(zip(batch_images, instructions, strict=True)):
                example = {"image": images, "lang": instruction}
                if states is not None:
                    example["state"] = states[index]
                if view_masks is not None:
                    example["view_mask"] = view_masks[index]
                examples.append(example)
        if not isinstance(examples, list):
            examples = [examples]
        mode = kwargs.get("infer_mode", self.config.framework.model.infer_mode)
        if mode not in ("flow", "flow_with_subtask"):
            raise ValueError(f"Unsupported Pi05 infer_mode: {mode}")
        subtasks = None
        token_kwargs = {}
        if mode == "flow_with_subtask":
            examples = [dict(example, _generate_subtask=True) for example in examples]
        observation = self._prepare_observation(examples)
        if mode == "flow_with_subtask":
            tokenizer = self._get_tokenizer()
            eos_id = tokenizer.tokenizer.eos_id()
            tokens = self.pi05.generate_subtask_tokens(observation, eos_id=eos_id, max_tokens=self.subtask_max_tokens)
            # Include the first EOS, excluding padding EOS emitted for already finished rows.
            eos = tokens == eos_id
            token_mask = eos.int().cumsum(dim=1) - eos.int() == 0
            token_kwargs.update(subtask_tokens=tokens, subtask_token_mask=token_mask)
            subtasks = tokenizer.decode_subtasks(tokens.cpu().numpy())
        actions = self.pi05.sample_actions(
            observation,
            **token_kwargs,
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            inference_delay=kwargs.get("inference_delay", 0),
            num_steps=int(kwargs.get("num_steps", self.num_inference_steps)),
            noise=kwargs.get("noise"),
            generator=kwargs.get("generator"),
        )
        result = {"normalized_actions": actions[..., : self.action_dim].float().cpu().numpy()}
        if subtasks is not None:
            result["subtask"] = subtasks
        return result

    @torch.inference_mode()
    def predict_subtask(self, examples: list[dict]) -> list[str]:
        observation = self._prepare_observation([dict(example, _generate_subtask=True) for example in examples])
        tokenizer = self._get_tokenizer()
        tokens = self.pi05.generate_subtask_tokens(
            observation, eos_id=tokenizer.tokenizer.eos_id(), max_tokens=self.subtask_max_tokens
        )
        return tokenizer.decode_subtasks(tokens.cpu().numpy())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.pi05.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.pi05.gradient_checkpointing_disable()


def _image_to_hwc(image: Image.Image | np.ndarray | torch.Tensor, size: int = 224) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    image = torch.as_tensor(image)
    if image.ndim == 4:
        image = image[:, 0]
    if image.ndim != 3:
        raise ValueError(f"Pi0.5 expects image [C,H,W] or [H,W,C], got {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    else:
        image = image[..., :3]
    image = image.float()
    if image.shape[:2] != (size, size):
        image = F.interpolate(
            image.permute(2, 0, 1)[None],
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[
            0
        ].permute(1, 2, 0)
    if image.max() > 1.5:
        image = image / 255.0
    if image.min() >= 0:
        image = image * 2.0 - 1.0
    return image
