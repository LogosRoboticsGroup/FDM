# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO data layout and mixtures for the LeRobot V2 loader."""

import copy
from typing import ClassVar

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.vla.data_config.data_config_base import BaseDataConfig


class LiberoDataConfig(BaseDataConfig):
    video_keys: ClassVar[list[str]] = ["observation.images.image", "observation.images.wrist_image"]
    state_ids: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5, 6, 7]
    action_ids: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5, 6]
    action_origin_dim = 7
    gripper_state_ids: ClassVar[list[int]] = [6, 7]
    gripper_action_ids: ClassVar[list[int]] = [6]
    disable_state = True

    def input_transform(self, data):
        # Evaluation uses continuous proprio and min/max normalization, as in FastWAM.
        # Training continues to use input_transform_dataloader and the checkpoint config.
        self.inference_mode = True
        self.disable_state = False
        self.binarize_gripper = False
        self.image_size = (224, 224)
        return super().input_transform(data)

    def resize_image(self, img):
        if not getattr(self, "inference_mode", False):
            return super().resize_image(img)
        image = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img, dtype=np.uint8))
        height, width = self.image_size
        src_w, src_h = image.size
        scale = max(width / src_w, height / src_h)
        image = image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.Resampling.BILINEAR)
        left, top = max((image.width - width) // 2, 0), max((image.height - height) // 2, 0)
        image = image.crop((left, top, left + width, top + height))
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)

    @staticmethod
    def _eval_affine(stats):
        low, high = np.asarray(stats["min"], dtype=np.float32), np.asarray(stats["max"], dtype=np.float32)
        span = high - low
        constant = span < 1e-4
        scale = 2.0 / np.where(constant, 2.0, span)
        offset = np.where(constant, -low, -1.0 - scale * low)
        return scale, offset

    def _normalize(self, values, stats):
        if not getattr(self, "inference_mode", False):
            return super()._normalize(values, stats)
        scale, offset = self._eval_affine(stats)
        return np.clip(np.asarray(values, dtype=np.float32) * scale + offset, -5.0, 5.0)

    def _unnormalize(self, values, stats):
        if not getattr(self, "inference_mode", False):
            return super()._unnormalize(values, stats)
        scale, offset = self._eval_affine(stats)
        return (np.asarray(values, dtype=np.float32) - offset) / scale


LIBERO_NO_NOOPS_DATASETS = {
    "libero_object_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_goal_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_spatial_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_10_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
}


LIBERO_NO_NOOPS_DATASETS_MULTI = {
    "libero_ipec": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    }
}


ROBOT_TYPE_CONFIG_MAP = {"libero": LiberoDataConfig()}

DATASET_NAMED_MIXTURES = {
    "libero_all": copy.deepcopy(LIBERO_NO_NOOPS_DATASETS),
    "libero_all_multi": copy.deepcopy(LIBERO_NO_NOOPS_DATASETS_MULTI),
}
