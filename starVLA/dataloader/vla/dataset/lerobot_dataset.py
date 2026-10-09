import copy
import os
import traceback
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image
import hashlib
import inspect

import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from starVLA.utils.system_utils import zero_rank_print
from starVLA.utils.atomic_json import atomic_dump_json
from starVLA.utils.data_utils import load_jsonl, OnlineStats
from starVLA.utils.video_utils import decode_video_frames
from starVLA.utils.frame_sampling import get_num_future_frames
from starVLA.dataloader.vla.data_config import BaseDataConfig
from starVLA.dataloader.vla.dataset.subtask import attach_subtask
from starVLA.dataloader.vla.dataset.base_vla_dataset import (
    StepIndexedVLADatasetMixin,
    build_static_keep_steps,
    select_static_filter_signal,
    split_episode_entries,
)
from starVLA.utils.text_embedding_cache import (
    DEFAULT_CONTEXT_LEN,
    DEFAULT_ENCODER_ID,
    maybe_load_text_embedding_cache,
)

import decord

def _process_episode(args):
    idx, episode_data_entry, action_horizon, data_config, keep_steps = args
    
    parquet_path = episode_data_entry[1]
    total_frames = episode_data_entry[3]

    dataframe = pd.read_parquet(parquet_path)
    total_frames = min(total_frames, len(dataframe))
    
    state_data = np.stack(dataframe[data_config.state_key or 'observation.state'].to_numpy())[:total_frames]
    action_data = np.stack(dataframe[data_config.action_key or 'action'].to_numpy())[:total_frames]
    if keep_steps is not None:
        keep_steps = np.asarray(keep_steps, dtype=np.int64)
        state_data = state_data[keep_steps]
        action_data = action_data[keep_steps]
        total_frames = keep_steps.size
    
    data = {
        "actions": action_data,
        "state": state_data,
    }

    states = []
    actions = []
    for step_idx in range(total_frames):
        data_transformed = data_config.input_transform_dataloader(data, step_idx, action_horizon, inplace=False)
        if 'state' in data_transformed and data_transformed['state'] is not None:
            states.append(data_transformed['state'])
        actions.append(data_transformed['actions'])

    return states, actions


class LeRobotV2Dataset(StepIndexedVLADatasetMixin, Dataset):
    def __init__(
        self,
        config,
        data_root: str,
        data_name: str,
        data_type: str,
        data_config: BaseDataConfig,
        video_keys: list = None,
        mode: str = "train",
        **kwargs,
    ):
        zero_rank_print(f"Loading dataset {data_root} (LeRobot)...")
        self.config = config
        self.data_root = data_root
        self.data_name = data_name
        self.data_config = copy.deepcopy(data_config)
        self.data_config.set_normalization_mode(getattr(config, "normalization_mode", "min_max"))
        self.data_config.disable_state = getattr(config, "disable_state", True)
        self.data_type = data_type
        self.mode = mode
        self.filter_static_steps = bool(getattr(config, 'filter_static_steps', False)) and self.mode == "train"
        self.static_filter_key = str(getattr(config, 'static_filter_key', 'action'))
        self.static_filter_eps = float(getattr(config, 'static_filter_eps', 1e-4))
        self.text_embedding_cache_dir = getattr(config, 'text_embedding_cache_dir', None)
        self.text_context_len = int(getattr(config, 'text_context_len', DEFAULT_CONTEXT_LEN))
        self.text_cache_encoder_id = str(getattr(config, 'text_cache_encoder_id', DEFAULT_ENCODER_ID))
        self.require_text_embedding_cache = bool(getattr(config, 'require_text_embedding_cache', False))
        self.text_context_prompt_template = getattr(config, 'text_context_prompt_template', None)

        # Allow overriding video_keys from mixture spec
        self.video_keys = video_keys if video_keys is not None else data_config.video_keys
        self.data_hash_name = data_name + '_' + hashlib.sha1(data_root.encode("utf-8")).hexdigest()[:6]
        statistics_cache_key = getattr(data_config, "statistics_cache_key", data_type)
        if data_config.state_key:
            statistics_cache_key += f"_{data_config.state_key}"
        if data_config.action_key:
            statistics_cache_key += f"_{data_config.action_key}"
        if self.data_config.normalization_mode == "q99":
            statistics_cache_key += "_q99"
        split_strategy = getattr(config, "split_strategy", "none")
        if split_strategy == "episode_ratio":
            statistics_cache_key = (
                f"{statistics_cache_key}_split_{split_strategy}"
                f"_eval_{float(getattr(config, 'eval_ratio', 0.0)):g}"
                f"_seed_{getattr(config, 'split_seed', 42)}"
            )
        if bool(getattr(config, 'filter_static_steps', False)):
            statistics_cache_key = f"{statistics_cache_key}_static_{self.static_filter_key}_eps_{self.static_filter_eps:g}"
        self.statistics_hash_name = self.data_hash_name + '_' + hashlib.sha1(statistics_cache_key.encode("utf-8")).hexdigest()[:6]

        dataset_info_cache_path = os.path.join("playground/meta_data", self.data_hash_name + "_vla_dataset_info_cache.json")
        if os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            self.dataset = {}
            # into the meta folder
            meta_folder = os.path.join(data_root, "meta")

            tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
            task_index_task_str = load_jsonl(tasks_jsonl)
            task_index_task_str_dict = {}
            for item in task_index_task_str:
                task_index_task_str_dict[item['task_index']] = item['task']

            with open(os.path.join(meta_folder, "info.json"), "r") as f:
                metainfo = json.load(f)
                # total_chunks = metainfo["total_chunks"]
                chunks_size = metainfo["chunks_size"]
                features = metainfo["features"]
                # Read path templates from info.json (support lerobot v2.0 and v2.1 variations)
                data_path_template = metainfo.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
                video_path_template = metainfo.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
                assert all([k in features and features[k]['dtype']=='video' for k in self.video_keys]), f"Some video keys {self.video_keys} not in features {features.keys()}"
                
            episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
            episodes_data = load_jsonl(episodes_jsonl)

            self.dataset['data'] = []

            def _process_episode(episode_data):
                episode_index = episode_data['episode_index']
                tasks = episode_data['tasks']
                meta_length = int(episode_data["length"])
                episode_chunk = int(episode_index // chunks_size)
                video_path = os.path.join(data_root, video_path_template.format(
                    episode_chunk=episode_chunk, episode_index=episode_index, video_key="{}"))
                try:
                    video_frame_counts = [
                        len(decord.VideoReader(video_path.format(cam), num_threads=1))
                        for cam in self.video_keys
                    ]
                    if any(frame_count != meta_length for frame_count in video_frame_counts):
                        return None
                    parquet_path = os.path.join(data_root, data_path_template.format(
                        episode_chunk=episode_chunk, episode_index=episode_index))
                    if not os.path.exists(parquet_path):
                        return None
                    data = pd.read_parquet(parquet_path)
                    lang = task_index_task_str_dict.get(data['task_index'][0], tasks[0] if tasks else "")
                    if not lang:
                        return None
                    return [video_path, parquet_path, lang, len(data)]
                except:
                    return None

            with ThreadPoolExecutor(max_workers=32) as ex:
                for info in tqdm(ex.map(_process_episode, episodes_data), total=len(episodes_data)):
                    if info is not None:
                        self.dataset['data'].append(info)

            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            atomic_dump_json(self.dataset, dataset_info_cache_path)

        self.dataset["data"] = split_episode_entries(
            self.dataset["data"],
            config=self.config,
            data_name=self.data_name,
            mode=self.mode,
        )
        self.total_episodes = len(self.dataset['data'])
        if self.total_episodes == 0:
            raise ValueError(f"[{self.data_name}] No VLA episodes remain after applying mode={self.mode!r} split.")
        episode_lengths = [sample[-1] for sample in self.dataset['data']]
        self.episode_keep_steps = self._build_episode_keep_steps() if self.filter_static_steps else None
        self.initialize_step_indexing(
            episode_lengths,
            mode=self.mode,
            episode_step_indices=self.episode_keep_steps,
        )
        zero_rank_print(f"[{os.path.basename(data_root)}] Total episodes: {self.total_episodes}, Total steps: {self.total_steps}")
        self.compute_statistics()

        self.image_size = config.image_size

        example_data = self.dataset['data'][0]
        video_reader = decord.VideoReader(example_data[0].format(self.video_keys[0]))
        h, w = video_reader[0].shape[:2]
        scale = min(self.image_size) / min(h, w)
        self.decord_resize_shape = (int(scale*h), int(scale*w))
        self.data_config.resize_method = str(getattr(config, "resize_method", "center_crop_resize"))
        if self.data_config.resize_method == "resize":
            # Decode at native resolution; torchvision performs the antialiased resize once.
            self.decord_resize_shape = (-1, -1)
        self.origin_fps = video_reader.get_avg_fps()

        self.jitter_transform = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1)
    
        # Some specialized datasets own image transforms directly.
        if hasattr(self.data_config, "set_image_size"):
            self.data_config.set_image_size(self.image_size)
        self.data_config.state_pad_size = self.config.state_pad_size
        self.data_config.action_pad_size = self.config.action_pad_size

        self.use_future_frames = self.config.use_future_frames
        self.image_frame_offsets = getattr(config, "image_frame_offsets", None)
        if self.image_frame_offsets is not None:
            if not self.image_frame_offsets or any(int(x) != x for x in self.image_frame_offsets):
                raise ValueError("image_frame_offsets must be a nonempty list of integer raw-frame offsets")
            self.image_frame_offsets = np.asarray(self.image_frame_offsets, dtype=np.int64)
        self.future_frame_stride = int(getattr(config, "future_frame_stride", 1))
        self.num_future_frames = (
            get_num_future_frames(config.action_horizon, self.future_frame_stride) if self.use_future_frames else 0
        )
        self.use_decord = getattr(config, 'use_decord', True)

    @property
    def data_key(self):
        return  f"{self.data_type}"

    def _get_selected_video_keys(self):
        return list(self.video_keys)

    def _build_episode_keep_steps(self):
        zero_rank_print(
            f"Building static-step filter from {self.static_filter_key} "
            f"with static_filter_eps={self.static_filter_eps:g}..."
        )
        keep_steps_by_episode = [None] * self.total_episodes

        def _process(idx):
            parquet_path = self.dataset["data"][idx][1]
            total_frames = int(self.dataset["data"][idx][3])
            dataframe = pd.read_parquet(parquet_path)
            total_frames_local = min(total_frames, len(dataframe))
            state_data = np.stack(dataframe[self.data_config.state_key or 'observation.state'].to_numpy())[
                :total_frames_local
            ]
            action_data = np.stack(dataframe[self.data_config.action_key or 'action'].to_numpy())[:total_frames_local]
            signal = select_static_filter_signal(
                self.data_config,
                state=state_data,
                action=action_data,
                filter_key=self.static_filter_key,
            )
            return idx, build_static_keep_steps(signal, self.static_filter_eps)

        num_workers = min(self.total_episodes, min(32, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_process, idx) for idx in range(self.total_episodes)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Static-step filter"):
                idx, keep_steps = fut.result()
                keep_steps_by_episode[idx] = keep_steps

        raw_steps = sum(int(sample[-1]) for sample in self.dataset["data"])
        kept_steps = sum(int(steps.size) for steps in keep_steps_by_episode)
        zero_rank_print(
            f"Static-step filter kept {kept_steps}/{raw_steps} steps "
            f"({kept_steps / max(raw_steps, 1):.4f})."
        )
        return keep_steps_by_episode
    
    def compute_statistics(self):
        statistic_path = os.path.join("playground/meta_data", self.statistics_hash_name + "_vla_dataset_statistic.json")
        if os.path.exists(statistic_path):
            zero_rank_print(f"Load Dataset Statistics from {statistic_path}")
            with open(statistic_path, "r") as f:
                self.dataset_statistics = json.load(f)
            if self.data_config.disable_state or "state" in self.dataset_statistics:
                return
            zero_rank_print("State statistics missing from cache, recomputing...")
        
        zero_rank_print("Computing dataset statistics...")
        state_stats = None
        action_stats = None
        num_workers = min(self.total_episodes, min(8, os.cpu_count()))

        args_iter = [
            (
                idx,
                self.dataset["data"][idx],
                self.config.action_horizon,
                self.data_config,
                None if self.episode_keep_steps is None else self.episode_keep_steps[idx],
            )
            for idx in range(self.total_episodes)
        ]
        
        # for args in tqdm(args_iter):
        #     states, actions = _process_episode(args)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_process_episode, args) for args in args_iter]

            for fut in tqdm(as_completed(futures), total=len(futures)):
                states, actions = fut.result()

                if action_stats is None:
                    action_stats = OnlineStats(
                        actions[0].shape[1], compute_quantiles=self.data_config.normalization_mode == "q99"
                    )
                action_stats.update_batch(np.concatenate(actions, axis=0))

                if len(states) > 0 and states[0] is not None:
                    if state_stats is None:
                        state_stats = OnlineStats(
                            states[0].shape[-1], compute_quantiles=self.data_config.normalization_mode == "q99"
                        )
                    if states[0].shape[-1] == state_stats.mean.shape[-1]:
                        state_stats.update_batch(np.concatenate(states, axis=0))
                    
        action_res = action_stats.finalize()
        self.dataset_statistics = {
            "action": {key: value.tolist() if value is not None else None for key, value in action_res.items()},
        }
        if state_stats is not None:
            state_res = state_stats.finalize()
            self.dataset_statistics.update(
                {
                    "state": {key: value.tolist() if value is not None else None for key, value in state_res.items()},
                }
            )
        atomic_dump_json(self.dataset_statistics, statistic_path)
        zero_rank_print(f"Saved dataset statistics to {statistic_path}")

    def save_dataset_statistics(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dataset_statistics = {self.data_name: self.dataset_statistics}
        atomic_dump_json(dataset_statistics, save_path)

    def _decode_frames(self, video_path, indexes, indexes_s):
        """Read video frames. Returns (N, C, H, W) tensor."""
        if self.use_decord:
            h, w = self.decord_resize_shape
            vr = decord.VideoReader(video_path, height=h, width=w, num_threads=1)
            batch = vr.get_batch(indexes.tolist())
            if not isinstance(batch, torch.Tensor):
                batch = torch.from_numpy(batch.asnumpy())
            batch = batch.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            return batch
        else:
            return decode_video_frames(video_path, indexes_s, tolerance_s=1e-4)

    def _get_image_indexes(self, episode_idx, step_idx, total_frames):
        # Explicit context offsets refer to raw episode frames, even after idle-step filtering.
        offsets = getattr(self, "image_frame_offsets", None)
        if offsets is not None:
            return np.clip(step_idx + offsets, 0, total_frames - 1)
        if self.episode_step_indices is None:
            if not self.use_future_frames:
                return np.array([step_idx], dtype=np.int64)
            indexes = np.array(
                [step_idx + i * self.future_frame_stride for i in range(self.num_future_frames + 1)],
                dtype=np.int64,
            )
            return np.clip(indexes, a_min=0, a_max=total_frames-1)
    
        keep_steps = self.episode_step_indices[int(episode_idx)]
        compact_step_idx = self.get_compact_step_index(episode_idx, step_idx)

        if not self.use_future_frames:
            compact_indexes = np.array([compact_step_idx], dtype=np.int64)
        else:
            compact_indexes = np.array(
                [compact_step_idx + i * self.future_frame_stride for i in range(self.num_future_frames + 1)],
                dtype=np.int64,
            )
            compact_indexes = np.clip(compact_indexes, a_min=0, a_max=keep_steps.size - 1)
        return keep_steps[compact_indexes]
    
    def get_sample_at_raw_step(self, episode_idx, step_idx):
        data = self.dataset["data"][episode_idx]
        video_path, parquet_path, prompt, total_frames = data
        dataframe = pd.read_parquet(parquet_path)
        total_frames = min(total_frames, len(dataframe))
        if step_idx < 0 or step_idx >= total_frames:
            raise IndexError(
                f"step_idx={step_idx} out of range for episode_idx={episode_idx} with total_frames={total_frames}"
            )
        
        indexes = self._get_image_indexes(episode_idx, step_idx, total_frames)
        indexes_s = (indexes / self.origin_fps).tolist()

        video_keys = self._get_selected_video_keys()

        images = []
        view_masks = []
        for video_key in video_keys:
            video_file = video_path.format(video_key)
            frame = self._decode_frames(video_file, indexes, indexes_s)
            if not self.use_future_frames and self.image_frame_offsets is None:
                frame = frame[0]  # Get first (and only) frame # [1, C, H, W] -> [C, H, W]
            else:
                frame = frame.permute(1, 0, 2, 3) # [T, C, H, W] -> [C, T, H, W]
            images.append(frame)
            view_masks.append(True)
        
        state_data = np.stack(dataframe[self.data_config.state_key or 'observation.state'].to_numpy())
        action_data = np.stack(dataframe[self.data_config.action_key or 'action'].to_numpy())
        transform_step_idx = step_idx
        if self.episode_step_indices is not None:
            keep_steps = self.episode_step_indices[episode_idx]
            transform_step_idx = self.get_compact_step_index(episode_idx, step_idx)
            state_data = state_data[keep_steps]
            action_data = action_data[keep_steps]
        
        input_data = {
            "image": images,
            "view_mask": view_masks,
            "lang": prompt,
            "fps": self.origin_fps,
            "actions": action_data,
            "state": state_data,
            "image_video_keys": video_keys,
        }

        data_unnormalized = self.data_config.input_transform_dataloader(input_data, transform_step_idx, self.config.action_horizon)
        data = self.data_config.normalize_data(data_unnormalized, self.dataset_statistics)

        data = self.data_config.pad_data(data)
        attach_subtask(data, dataframe, step_idx, self.config)
        if self.text_embedding_cache_dir:
            cached_text = maybe_load_text_embedding_cache(
                self.text_embedding_cache_dir,
                prompt,
                context_len=self.text_context_len,
                encoder_id=self.text_cache_encoder_id,
                required=self.require_text_embedding_cache,
                prompt_template=self.text_context_prompt_template,
            )
            if cached_text is not None:
                context, context_mask = cached_text
                data["context"] = context
                data["context_mask"] = context_mask
        return data
