from collections import deque
from typing import Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        image_size: Optional[list[int]] = None,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        replan_steps: int = 24,
        num_inference_steps: Optional[int] = None,
        seed: int = 42,
        action_horizon: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        sigma_shift: Optional[float] = None,
    ) -> None:

        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.replan_steps = replan_steps
        self.inference_kwargs = {"seed": seed}
        for key, value in (
            ("num_inference_steps", num_inference_steps),
            ("action_horizon", action_horizon),
            ("num_video_frames", num_video_frames),
        ):
            if value is not None:
                if value <= 0:
                    raise ValueError(f"{key} must be positive")
                self.inference_kwargs[key] = value
        if sigma_shift is not None:
            self.inference_kwargs["sigma_shift"] = sigma_shift
        self.pending_actions = deque()
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.action_mode = action_mode
        if action_mode not in ("abs", "delta", "rel"):
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        self.task_description = None
        self.initial_state = None
        self.prev_action = None
        print(f"RoboTwin server metadata: {self.client.get_server_metadata()}")

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.pending_actions.clear()
        self.initial_state = None
        self.prev_action = None
        self.client.reset(instruction=task_description)

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        del step
        if example is not None and example.get("lang") != self.task_description:
            self.reset(example.get("lang"))
        if self.pending_actions:
            return self.pending_actions.popleft()
        if example is None:
            raise ValueError("Observation is required when the action queue is empty")

        state = example.get("state")
        if self.action_mode in ("delta", "rel") and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state")
            self.initial_state = np.asarray(state).copy()

        payload = {
            "batch_images": [[self._resize_image(image) for image in example["image"]]],
            "instructions": [self.task_description],
            "view_mask": [[True, True, True]],
            "state": [state],
            **self.inference_kwargs,
        }
        if self.unnorm_key is not None:
            payload["stat_key"] = self.unnorm_key
        response = self.client.predict_action(payload)
        actions = np.asarray(response["data"]["actions"][0], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 14:
            raise ValueError(f"Expected a nonempty [T, 14] action chunk, got {actions.shape}")
        actions = actions[: self.replan_steps]
        if self.action_mode == "delta":
            actions = self._delta_to_absolute(actions, state)
            self.prev_action = actions[-1].copy()
        elif self.action_mode == "rel":
            actions = self._rel_to_absolute(actions)
        # Current RoboTwin data configs preserve [left arm, left gripper, right arm, right gripper].
        self.pending_actions.extend(actions)
        return self.pending_actions.popleft()

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.image_size is not None:
            image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        replan_steps=int(usr_args.get("replan_steps", 24)),
        num_inference_steps=usr_args.get("num_inference_steps"),
        seed=int(usr_args.get("seed", 42)),
        action_horizon=usr_args.get("action_horizon"),
        num_video_frames=usr_args.get("num_video_frames"),
        sigma_shift=usr_args.get("sigma_shift"),
        image_size=usr_args.get("image_size"),
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):  # noqa: A001 - RoboTwin policy entrypoint
    if not model.should_request_observation():
        TASK_ENV.take_action(model.step(None), action_type="qpos")
        return

    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action, action_type="qpos")
