"""ARX VR recordings use the Piper EEF and joint data adapters unchanged."""

import os

DATASET_NAMED_MIXTURES = {
    name: {
        name: {
            "data_root": os.environ.get("ARX_DATA_ROOT", "results/arx_vr"),
            "data_weight": 1.0,
            "data_class": "lerobot_vla",
            "data_type": data_type,
        },
    }
    for name, data_type in (("arx_vr", "piper"), ("arx_vr_joint", "piper_joint"))
}
