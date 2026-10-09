# Policy evaluation protocol

Simulation and robot clients communicate with the model server through WebSocket and MessagePack. The current server applies the checkpoint's preprocessing and postprocessing around model inference.

```text
Environment observations
  → benchmark/robot client
  → WebSocket policy server
  → framework.preprocess
  → framework.predict_action
  → framework.postprocess
  → environment-coordinate actions
  → client action queue / environment controller
```

## Start the server

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/run/checkpoints/steps_2000_pytorch_model.pt \
  --port 10093 \
  --use_bf16
```

Keep `config.yaml` and `dataset_statistics.json` in the run directory above `checkpoints/`. The server selects the checkpoint's data configuration; an explicit `--stat_key` must match its statistics key.

## Client contract

Use [WebsocketClientPolicy](../deployment/model_server/tools/websocket_policy_client.py) for transport. A batch-size-one request has the following form; `images`, `instruction`, and `state` are supplied by the environment:

```python
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=10093)
try:
    metadata = client.get_server_metadata()
    response = client.infer({
        "batch_images": [[np.asarray(image, dtype=np.uint8) for image in images]],
        "instructions": [instruction],
        "state": [np.asarray(state, dtype=np.float32).reshape(1, -1)],
        "stat_key": metadata["stat_key"],
    })
    if not response.get("ok", False):
        raise RuntimeError(response.get("error"))
    actions = np.asarray(response["data"]["actions"])[0]
finally:
    client.close()
```

Images are RGB arrays in the camera order used during training. State and actions must use the checkpoint's embodiment convention. `data.actions` is already postprocessed into environment coordinates; do not unnormalize it again. Direct framework `predict_action` calls return `normalized_actions` and bypass the server's transform pipeline.

For chunk execution and asynchronous inference, use [M1Inference](../deployment/model_server/inferencer.py), as used by the current [LIBERO evaluator](LIBERO/eval_files/eval_libero.py). It manages the pending action queue; reset it at episode boundaries. Pi05 RTC additionally sends the remaining action prefix and its delay.

Environment adapters remain responsible for camera acquisition, robot/simulator action conventions, timing, and reset behavior. Follow the [LIBERO](LIBERO/README.md), [RoboTwin](Robotwin/README.md), [RoboCasa365](Robocasa_365/README.md), or relevant robot guide for the full loop.
