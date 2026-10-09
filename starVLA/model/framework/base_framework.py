"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import copy
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.dataloader.vla.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
from starVLA.model.tools import FRAMEWORK_REGISTRY, FrameworkTools, auto_get_trainable_modules
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_FRAMEWORKS_IMPORTED = False


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    _SKIP = {"__init__", "base_framework", "share_tools"}
    framework_dir = Path(__file__).resolve().parent

    # Scan top-level modules (backwards compat)
    for _, module_name, is_pkg in pkgutil.iter_modules([str(framework_dir)]):
        if module_name in _SKIP:
            continue
        if is_pkg:
            # Scan sub-packages (VLM4A/, WM4A/, etc.)
            sub_dir = framework_dir / module_name
            for _, sub_name, _ in pkgutil.iter_modules([str(sub_dir)]):
                if sub_name.startswith("_"):
                    continue
                importlib.import_module(f"starVLA.model.framework.{module_name}.{sub_name}")
        else:
            importlib.import_module(f"starVLA.model.framework.{module_name}")

    _FRAMEWORKS_IMPORTED = True


def build_framework(cfg): # The single entry point for building different model frameworks
    """
    Build a framework model from config.
    Args:
        cfg: Config object containing `cfg.framework.name`.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError("Missing `cfg.framework.name`. The framework API now only accepts `framework.name`.")

    _auto_import_framework_modules()

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(
            f"Framework `{framework_id}` is not implemented. Available frameworks: {available}"
        )

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO @JinhuiYE find sometime to merge yaml config with transformer config


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """

        super().__init__(hf_config)

    def compile(self):
        pass

    def reset(self):
        pass

    def _after_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        del state_dict

    def _before_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return state_dict

    # ------------------------------------------------------------------
    # Soft-constraint interface: subclasses should override these.
    # Default implementations raise NotImplementedError so that IDE
    # tooling (e.g. pylance, mypy) flags missing overrides, while
    # still allowing PreTrainedModel instantiation (no ABC).
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires at least:
                - image: List[PIL.Image]
                - lang: str
                - action: np.ndarray shaped [T, action_dim]

        Returns:
            dict: Must contain ``"action_loss"`` (torch.Tensor scalar).
                  May contain extra keys for logging (e.g. ``"kl_loss"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference: predict future actions from observations.

        Args:
            examples: Same schema as *forward* (minus ``action`` which is optional).
            **kwargs: Framework-specific inference options (e.g. ``use_ddim``).

        Returns:
            dict: Must contain ``"normalized_actions"`` (np.ndarray [B, T, action_dim]).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    # ------------------------------------------------------------------
    # Unified loss interface for Trainer
    # ------------------------------------------------------------------

    def supports_training_tag(self, tag: str) -> bool:
        """Return whether this framework can consume batches for *tag*."""
        if tag == "vla":
            return type(self).forward is not baseframework.forward
        if tag == "vlm":
            return hasattr(self, "qwen_vl_interface") or type(self).forward_vlm is not baseframework.forward_vlm
        return False

    def compute_loss(self, tag: str, batch, loss_scale: dict = None) -> Dict[str, torch.Tensor] | None:
        """Unified forward entry-point: route to the right forward by *tag*.

        The trainer calls ``model.compute_loss(tag, batch)`` for every
        ``(tag, batch)`` pair produced by :class:`DataLoaderManager`.
        The model internally dispatches:

        - ``"vla"`` → ``self.forward(batch)``
        - ``"vlm"`` → ``self.forward_vlm(batch)``

        Subclasses can override this to add more tags (e.g. ``"world"``).

        Args:
            tag: dataset type tag (``"vla"``, ``"vlm"``, …)
            batch: the batch produced by the corresponding DataLoader.
            loss_scale: ``{"vla": 1.0, "vlm": 0.1}`` per-tag loss multiplier.
                        Defaults to 1.0 for unspecified tags.

        Returns:
            dict[str, Tensor] | None: keyed losses (e.g. ``{"action_loss": ...}``).
                Returns ``None`` when this framework does not support the
                incoming dataloader tag so the trainer can ``continue``.
        """
        if not self.supports_training_tag(tag):
            return None

        scale = (loss_scale or {}).get(tag, 1.0)

        if tag == "vla":
            out = self.forward(batch)
        elif tag == "vlm":
            out = self.forward_vlm(batch)
        else:
            return None

        # Apply loss scale and filter to Tensor values only
        return {k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)}

    def forward_vlm(self, batch) -> Dict[str, torch.Tensor]:
        """VLM forward pass (default implementation).

        Delegates to ``self.qwen_vl_interface(**batch)`` which is present on
        every framework subclass that uses a Qwen VL backbone.

        Subclasses may override to add custom VLM logic.

        Args:
            batch: dict produced by the VLM dataloader.

        Returns:
            dict: Must contain ``"vlm_loss"`` (torch.Tensor scalar).
        """
        if not hasattr(self, "qwen_vl_interface"):
            raise NotImplementedError(
                f"{type(self).__name__} has no `qwen_vl_interface`. "
                "Override forward_vlm() to support VLM training."
            )
        out = self.qwen_vl_interface(**batch)
        return {"vlm_loss": out.loss}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> "baseframework":
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Dot-path config overrides applied before framework construction.
                Pass a mapping as config_overrides for multiple overrides.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

        config = dict_to_namespace(model_config)
        config_overrides = kwargs.pop("config_overrides", None)
        overrides = {
            "framework.skip_dit_load_from_pretrain": True,
            "framework.action_model.skip_load_from_pretrain": True,
        }
        if config_overrides is not None:
            overrides.update(dict(config_overrides))
        overrides.update(kwargs)
        for key, value in overrides.items():
            OmegaConf.update(config, str(key), value, merge=True)

        model_config = config
        model_config.trainer.pretrained_checkpoint = None
        
        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        if pretrained_checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            model_state_dict = load_file(str(pretrained_checkpoint))
        else:
            try:
                model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
        model_state_dict = FrameworkModel._before_load_state_dict(model_state_dict)
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            missing_keys, unexpected_keys = FrameworkModel.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e
        FrameworkModel._after_load_state_dict(model_state_dict)

        # **ensure model is on GPU**
        FrameworkModel = FrameworkModel
        return FrameworkModel

    def get_action_stats(self, stat_key=None, norm_stats=None):
        """
        Retrieve raw action normalization statistics.
        """
        if norm_stats is None:
            norm_stats = self.norm_stats
        stat_key = self._check_stat_key(norm_stats, stat_key)
        return norm_stats[stat_key]["action"]

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        keys = auto_get_trainable_modules(self, max_depth=max_depth)
        return keys

    @staticmethod
    def _check_stat_key(norm_stats, stat_key):
        """Return the checkpoint's sole dataset-statistics key."""
        keys = tuple(norm_stats)
        if len(keys) != 1:
            raise ValueError(f"Expected exactly one dataset statistics key, got {list(keys)}")

        resolved_key = keys[0]
        if stat_key is not None and stat_key != resolved_key:
            raise ValueError(
                f"Requested stat_key={stat_key!r} does not match the checkpoint statistics key {resolved_key!r}"
            )
        return resolved_key

    def parameters(self, recurse: bool = True):
        for _name, param in self.named_parameters(recurse=recurse):
            if not param.requires_grad:
                continue
            yield param

    @staticmethod
    def _get_config_value(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        getter = getattr(obj, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                pass
        return getattr(obj, key, default)

    def _resolve_data_config_key(self, stat_key: str) -> str:
        if stat_key not in ROBOT_TYPE_CONFIG_MAP:
            raise NotImplementedError(
                f"No data_config is registered for dataset statistics key {stat_key!r}. "
                f"Available: {list(ROBOT_TYPE_CONFIG_MAP)}"
            )
        return stat_key

    def set_dataconfig(self, stat_key=None):
        stat_key = self._check_stat_key(self.norm_stats, stat_key)
        self.stat_key = stat_key

        data_config_key = self._resolve_data_config_key(stat_key)
        self.data_config = copy.deepcopy(ROBOT_TYPE_CONFIG_MAP[data_config_key])

        datasets_cfg = self._get_config_value(getattr(self, "config", None), "datasets")
        vla_data_cfg = self._get_config_value(datasets_cfg, "vla_data")
        self.data_config.set_normalization_mode(self._get_config_value(vla_data_cfg, "normalization_mode", "min_max"))
        if hasattr(self.data_config, "set_image_size") and vla_data_cfg is not None:
            self.data_config.resize_method = str(
                self._get_config_value(vla_data_cfg, "resize_method", "center_crop_resize")
            )
            image_size = self._get_config_value(vla_data_cfg, "image_size")
            if image_size is not None:
                self.data_config.set_image_size(image_size)

            disable_state = self._get_config_value(vla_data_cfg, "disable_state", None)
            if disable_state is not None:
                self.data_config.disable_state = bool(disable_state)

        framework_cfg = self._get_config_value(getattr(self, "config", None), "framework")
        action_model_cfg = self._get_config_value(framework_cfg, "action_model")
        if action_model_cfg is None:
            action_model_cfg = self._get_config_value(framework_cfg, "pat_mlp_model")
        if action_model_cfg is None:
            raise AttributeError("framework.action_model or framework.pat_mlp_model is required for data_config sizing.")

        self.data_config.state_pad_size = int(self._get_config_value(action_model_cfg, "state_dim"))
        self.data_config.action_pad_size = int(self._get_config_value(action_model_cfg, "action_dim"))

    def preprocess(self, data: dict, stat_key=None, inplace=False):
        if inplace:
            data_processed = data
        else:
            data_processed = data.copy()
        if not hasattr(self, "data_config"):
            self.set_dataconfig(stat_key)
        elif stat_key is not None and stat_key != getattr(self, "stat_key", None):
            self.set_dataconfig(stat_key)
        if stat_key is None:
            stat_key = self.stat_key

        data_processed = self.data_config.input_transform(data_processed)
        data_processed = self.data_config.normalize_data(data_processed, self.norm_stats[stat_key])
        data_processed = self.data_config.pad_data(data_processed)
        return data_processed

    def postprocess(self, data: dict, input_data: dict, stat_key=None, inplace=False):
        if inplace:
            data_processed = data
        else:
            data_processed = data.copy()
        stat_key = self.stat_key if stat_key is None else stat_key
        data_processed = self.data_config.unpad_data(data_processed)
        data_processed = self.data_config.unnormalize_data(data_processed, self.norm_stats[stat_key])
        data_processed = self.data_config.output_transform(data_processed, input_data)
        return data_processed

    def _resolve_action_chunk_size(self) -> int:
        framework_cfg = self._get_config_value(getattr(self, "config", None), "framework")
        action_model_cfg = self._get_config_value(framework_cfg, "action_model")
        if action_model_cfg is None:
            action_model_cfg = self._get_config_value(framework_cfg, "pat_mlp_model")
        action_horizon = self._get_config_value(action_model_cfg, "action_horizon")
        if action_horizon is not None:
            return int(action_horizon)
        future_action_window_size = self._get_config_value(action_model_cfg, "future_action_window_size")
        if future_action_window_size is not None:
            return int(future_action_window_size) + 1
        return int(getattr(self, "action_horizon", 1))

    def get_metadata(self) -> dict:
        stat_key = str(getattr(self, "stat_key", self._check_stat_key(self.norm_stats, None)))
        state_ids = getattr(self.data_config, "state_ids", ())
        state_dim = len(state_ids) if state_ids else int(self.data_config.state_pad_size)
        action_dim = getattr(self.data_config, "transformed_action_dim", None)
        if action_dim is None:
            action_ids = getattr(self.data_config, "action_ids", ())
            action_dim = len(action_ids) if action_ids else int(self.data_config.action_pad_size)
        metadata = {
            "stat_key": stat_key,
            "action_chunk_size": self._resolve_action_chunk_size(),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
        }
        action_space = getattr(self.data_config, "action_space", None)
        if action_space is not None:
            metadata["action_space"] = str(action_space)
        return metadata
