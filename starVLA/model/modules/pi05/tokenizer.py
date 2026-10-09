# Copyright 2024 Physical Intelligence and 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""PaliGemma prompt tokenizer with Pi0.5 discrete-state formatting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sentencepiece

logger = logging.getLogger(__name__)


class Pi05Tokenizer:
    def __init__(self, model_path: str | Path, max_length: int = 200):
        self.max_length = max_length
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(Path(model_path).expanduser()))

    def tokenize(
        self, prompt: str, state: np.ndarray | None, *, generate_subtask: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if generate_subtask:
            prompt = f"Task: {cleaned}"
            if state is not None:
                bins = np.linspace(-1, 1, 257)[:-1]
                discrete_state = np.digitize(np.asarray(state), bins=bins) - 1
                prompt += ", State: " + " ".join(map(str, discrete_state.tolist()))
            tokens = self.tokenizer.encode(prompt + "\nPredict the next action in language.\nSubtask: ", add_bos=True)
            if len(tokens) > self.max_length:
                raise ValueError("Subtask prompt exceeds max_token_len; increase the configured length.")
        elif state is None:
            tokens = self.tokenizer.encode(cleaned, add_bos=True)
            tokens += self.tokenizer.encode("\n")
        else:
            bins = np.linspace(-1, 1, 257)[:-1]
            discrete_state = np.digitize(np.asarray(state), bins=bins) - 1
            state_text = " ".join(map(str, discrete_state.tolist()))
            full_prompt = f"Task: {cleaned}, State: {state_text};\nAction: "
            tokens = self.tokenizer.encode(full_prompt, add_bos=True)
        if len(tokens) > self.max_length:
            logger.warning(
                "Pi0.5 prompt has %d tokens; truncating to %d.",
                len(tokens),
                self.max_length,
            )
        tokens = tokens[: self.max_length]
        mask = np.zeros(self.max_length, dtype=np.bool_)
        mask[: len(tokens)] = True
        padded = np.zeros(self.max_length, dtype=np.int64)
        padded[: len(tokens)] = tokens
        return padded, mask

    def tokenize_subtask(self, text: str, max_length: int) -> list[int]:
        tokens = self.tokenizer.encode(text.strip(), add_eos=True)
        if len(tokens) > max_length:
            raise ValueError("Subtask target exceeds subtask_max_tokens; increase the configured length.")
        return tokens

    def decode_subtasks(self, tokens: np.ndarray) -> list[str]:
        eos = self.tokenizer.eos_id()
        sequences = tokens.tolist()
        return [self.tokenizer.decode(ids[: ids.index(eos)] if eos in ids else ids).strip() for ids in sequences]

    def tokenize_actions(
        self, actions: np.ndarray, processor, *, max_length: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode normalized actions; optional fixed padding includes the suffix and never truncates targets."""
        sequences = processor(actions)
        token_max = self.tokenizer.vocab_size() - 1 - 128
        end_tokens = self.tokenizer.encode("|", add_eos=True)
        sequences = [(token_max - np.asarray(sequence, dtype=np.int64)).tolist() + end_tokens for sequence in sequences]
        length = max(map(len, sequences))
        if max_length is not None:
            if length > max_length:
                raise ValueError(
                    f"FAST target has {length} tokens including the suffix, exceeding fast_max_tokens={max_length}; "
                    "increase the configured length."
                )
            length = max_length
        tokens = np.zeros((len(sequences), length), dtype=np.int64)
        mask = np.zeros_like(tokens, dtype=np.bool_)
        for index, sequence in enumerate(sequences):
            tokens[index, : len(sequence)] = sequence
            mask[index, : len(sequence)] = True
        return tokens, mask
