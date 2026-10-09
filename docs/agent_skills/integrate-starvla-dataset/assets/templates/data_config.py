"""<<TODO_BENCH>> data adapter and mixtures for the current LeRobot V2 loader.

Place at examples/<<TODO_BENCH>>/train_files/data_registry/data_config.py.
The VLA registry discovers this module. Names must be globally unique.
Keep simulator / evaluation dependencies out of this module.
"""

from typing import ClassVar

from starVLA.dataloader.vla.data_config.data_config_base import BaseDataConfig


class MyRobotDataConfig(BaseDataConfig):
    # Match the Parquet columns and camera fields written by your converter.
    video_keys: ClassVar[list[str]] = ["observation.images.<<TODO_CAM_1>>"]
    state_key = "observation.state"
    action_key = "action"
    state_ids: ClassVar[list[int]] = list(range(7))  # TODO: select the actual columns
    action_ids: ClassVar[list[int]] = list(range(7))
    action_origin_dim = 7
    state_pad_size = 7
    action_pad_size = 7
    gripper_state_ids: ClassVar[list[int]] = [6]
    gripper_action_ids: ClassVar[list[int]] = [6]
    # Use BaseRelativeDataConfig for actions relative to the current state.
    # action_horizon and normalization_mode come from datasets.vla_data in YAML.


ROBOT_TYPE_CONFIG_MAP = {"<<TODO_robot_type_string>>": MyRobotDataConfig()}

DATASET_NAMED_MIXTURES = {
    "<<TODO_mixture_name>>": {
        "<<TODO_dataset_name>>": {
            "data_root": "playground/Datasets/<<TODO_DATASET_DIR>>",
            "data_weight": 1.0,
            "data_class": "lerobot_vla",  # multi_lerobot_vla pools datasets below a root
            "data_type": "<<TODO_robot_type_string>>",
            "video_keys": list(MyRobotDataConfig.video_keys),
        },
    },
}
