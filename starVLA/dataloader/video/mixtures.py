import copy

def _select(registry, dataset_names):
    return {dataset_name: copy.deepcopy(registry[dataset_name]) for dataset_name in dataset_names}

LIBERO_NO_NOOPS_DATASETS = {
    "libero_object_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_video",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_goal_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_video",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_spatial_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_video",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_10_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_video",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
}


DATASET_NAMED_MIXTURES = {
    "libero_all": _select(LIBERO_NO_NOOPS_DATASETS, tuple(LIBERO_NO_NOOPS_DATASETS)),
}


def num_video_keys(data_mix):
    """Number of camera/video keys declared by a named data mix (raises if unknown)."""
    mixture = DATASET_NAMED_MIXTURES[str(data_mix)]
    return max(len(spec["video_keys"]) for spec in mixture.values())