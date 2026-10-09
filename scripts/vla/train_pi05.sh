#!/bin/bash
# Multi-GPU Pi0.5 LIBERO fine-tuning script (DeepSpeed ZeRO-2)
# Usage: PI05_MODEL_PATH=/path/to/pi05_base_pytorch_new \
#   bash scripts/vla/train_pi05.sh [data_mix] [run_id_suffix] [extra_args...]
# Example: PI05_MODEL_PATH=/path/to/pi05_base_pytorch_new \
#   bash scripts/vla/train_pi05.sh libero_all_multi debug --trainer.max_train_steps 1000

set -eo pipefail

source /inspire/ssd/project/robot3d/mazipei-253107140027/miniconda3/etc/profile.d/conda.sh
conda activate /inspire/ssd/project/robot3d/mazipei-253107140027/miniconda3/envs/starVLA

###########################################################################################
# === Configuration ===
Framework_name=Pi05
freeze_module_list=''
config_yaml=starVLA/config/training/vla/starvla_pi05.yaml
run_root_dir=./results/Checkpoints/vla
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
# RLinf/OpenPI_RLinf checkpoint directory containing model.safetensors.
pi05_model_path=${PI05_MODEL_PATH:-playground/Pretrained_models/pi05_base_pytorch}
pi05_tokenizer_path=${PI05_TOKENIZER_PATH:-/inspire/ssd/project/robot3d/mazipei-253107140027/.cache/openpi/big_vision/paligemma_tokenizer.model}
###########################################################################################

data_mix=${1:-libero_all_multi}

# $2 is treated as run_id_suffix only if it does NOT contain '=' or start with '--'.
if [ -n "${2:-}" ] && [[ "$2" != *"="* ]] && [[ "$2" != --* ]]; then
  run_id_suffix=$2
  shift 2
elif [ -n "${1:-}" ]; then
  run_id_suffix=""
  shift 1
else
  run_id_suffix=""
fi

# Remaining args are command-line config overrides.
EXTRA_ARGS=("$@")

NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

DISTRIBUTED_ARGS="--num_machines ${NNODES} --num_processes ${NUM_PROCESSES}"
if [ "${NNODES}" -gt 1 ]; then
  DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

date_prefix=$(date +%m%d)
if [ -n "${RUN_ID:-}" ]; then
  run_id="${RUN_ID}"
elif [ -z "${run_id_suffix}" ]; then
  run_id="${date_prefix}_${Framework_name}_${data_mix}"
else
  run_id="${date_prefix}_${Framework_name}_${data_mix}_${run_id_suffix}"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
echo "Logging to ${output_dir}/train.log"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  ${DISTRIBUTED_ARGS} \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.model.model_path "${pi05_model_path}" \
  --framework.model.tokenizer_path "${pi05_tokenizer_path}" \
  --framework.action_model.model_action_dim 32 \
  --framework.action_model.action_horizon 50 \
  --framework.action_model.num_inference_steps 10 \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.action_horizon 50 \
  --datasets.vla_data.image_size '[224,224]' \
  --datasets.vla_data.disable_state false \
  --datasets.vla_data.use_future_frames false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.eval_per_device_batch_size 4 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.enable_gradient_checkpointing false \
  --trainer.global_batch_size 64 \
  --trainer.max_train_steps 30000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 2000 \
  --trainer.learning_rate.base 2.5e-5 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.scheduler_specific_kwargs.min_lr 0.0 \
  --trainer.gradient_clipping 1.0 \
  --trainer.optimizer.betas '[0.9,0.95]' \
  --trainer.optimizer.eps 1.0e-8 \
  --trainer.optimizer.weight_decay 1.0e-10 \
  --trainer.resume_from_checkpoint latest \
  --trainer.enable_compile true \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${output_dir}/train.log"
