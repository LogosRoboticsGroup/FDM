"""Resolve epoch-based optimizer and warmup step budgets."""

from math import ceil


def resolve_training_steps(cfg, num_samples: int, num_processes: int, gradient_accumulation_steps: int):
    """Fill null step budgets after the training split and world size are known."""
    trainer = cfg.trainer
    if trainer.max_train_steps is None:
        batch_size = int(cfg.datasets.vla_data.per_device_batch_size)
        epochs = int(trainer.epochs)
        if min(num_samples, batch_size, num_processes, gradient_accumulation_steps, epochs) <= 0:
            raise ValueError(
                "Epoch-based training requires positive data size, batch size, world size, accumulation and epochs"
            )
        micro_steps = ceil(num_samples / (batch_size * num_processes))
        trainer.max_train_steps = ceil(micro_steps / gradient_accumulation_steps) * epochs
    if int(trainer.max_train_steps) <= 0:
        raise ValueError("max_train_steps must be positive")
    if trainer.num_warmup_steps is None:
        ratio = float(trainer.get("warmup_ratio", 0.05))
        if not 0 <= ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        trainer.num_warmup_steps = min(int(trainer.max_train_steps * ratio), trainer.max_train_steps - 1)
