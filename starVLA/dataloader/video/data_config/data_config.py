from starVLA.dataloader.video.data_config.data_config_base import VideoBaseDataConfig


class LIBERO_DataConfig(VideoBaseDataConfig):
    video_keys = ("observation.images.image", "observation.images.wrist_image")
