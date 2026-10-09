from starVLA.dataloader.video.data_config.data_config_base import VideoBaseDataConfig
from starVLA.dataloader.video.data_config.data_config import (
    LIBERO_DataConfig,
)

VIDEO_TYPE_CONFIG_MAP = {
    "libero": LIBERO_DataConfig(),
}

__all__ = ["VideoBaseDataConfig", "VIDEO_TYPE_CONFIG_MAP"]
