"""RoboTwin absolute joint actions for the LeRobot V2 dataloader.

Keep the Parquet column order: left joints (6), left gripper, right joints (6),
right gripper. The old modality loader concatenated both arms before grippers;
its packed tensors/statistics must not be reused without reordering.

Chunk length is supplied by ``datasets.vla_data.action_horizon``. The RoboTwin
training YAMLs link it to ``framework.action_model.action_horizon`` so a CLI
override of the latter sets the model and dataset chunk lengths together.
"""

import copy
from typing import ClassVar

import numpy as np

from starVLA.dataloader.vla.data_config.data_config_base import BaseDataConfig


class AgilexDataConfig(BaseDataConfig):
    """Three cameras and 14 joint/gripper targets; z-score follows FastWAM."""

    statistics_cache_key = "robotwin_abs_qpos"
    video_keys: ClassVar[list[str]] = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    state_ids: ClassVar[list[int]] = list(range(14))
    action_ids: ClassVar[list[int]] = list(range(14))
    action_origin_dim = 14
    state_pad_size = 14
    action_pad_size = 14
    gripper_state_ids: ClassVar[list[int]] = [6, 13]
    gripper_action_ids: ClassVar[list[int]] = [6, 13]
    disable_state = True
    binary_threshold = 0.5

    def set_normalization_mode(self, mode):
        super().set_normalization_mode(mode)
        # FastWAM normalizes grippers with the same statistics as joint targets.
        self.binarize_gripper = mode != "z_score"

    def _normalize(self, values, stats):
        if self.normalization_mode == "z_score":
            mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
            return np.clip((values - mean) / (std + 1e-8), -5.0, 5.0)
        return super()._normalize(values, stats)

    def _unnormalize(self, values, stats):
        if self.normalization_mode == "z_score":
            mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
            return values * (std + 1e-8) + mean
        return super()._unnormalize(values, stats)

    def normalize_data(self, data, stats):
        # The legacy binary transform thresholds raw gripper values, independent
        # of per-task min/max statistics (including constant-gripper episodes).
        grippers = []
        if self.binarize_gripper:
            for key, ids in (("state", self.gripper_state_ids), ("actions", self.gripper_action_ids)):
                if data.get(key) is not None and (key != "state" or not self.disable_state):
                    grippers.append((key, ids, np.asarray(data[key])[..., ids] > self.binary_threshold))
        data = super().normalize_data(data, stats)
        for key, ids, values in grippers:
            data[key][..., ids] = values
        return data

    def unnormalize_data(self, data, stats):
        data = super().unnormalize_data(data, stats)
        if self.binarize_gripper:
            grippers = np.asarray(data["normalized_actions"])[..., self.gripper_action_ids]
            data["actions"][..., self.gripper_action_ids] = grippers > self.binary_threshold
        return data


ROBOTWIN_3_VIEW_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


ROBOTWIN_DATASETS = {
    "robotwin": {
        "data_root": "playground/Datasets/RoboTwin",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "robotwin",
        "video_keys": list(ROBOTWIN_3_VIEW_KEYS),
    }
}


ROBOT_TYPE_CONFIG_MAP = {"robotwin": AgilexDataConfig()}

DATASET_NAMED_MIXTURES = {"robotwin_all": copy.deepcopy(ROBOTWIN_DATASETS)}
