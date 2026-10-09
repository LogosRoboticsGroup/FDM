#!/bin/bash
# Multi-GPU WanMoTCausal ARX training script (DeepSpeed ZeRO-2)
# Usage: bash examples/ARX/train_files/train_wanmot-causal.sh [data_mix] [run_id_suffix] [extra_args...]
# Example: bash examples/ARX/train_files/train_wanmot-causal.sh badminton debug --trainer.max_train_steps 1000
# Set TEXT_EMBEDDING_CACHE_DIR to override the benchmark's local text cache.

set -eo pipefail

# Run with the StarVLA training environment already activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

###########################################################################################
# === Configuration ===
Framework_name=WanMoTCausal
config_yaml=starVLA/config/training/vla/starvla_wanmot_causal.yaml
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints/vla}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
###########################################################################################

data_mix=badminton
run_id_suffix=""
if [ -n "${1:-}" ] && [[ "$1" != --* && "$1" != *"="* ]]; then
  data_mix=$1
  shift
fi
if [ -n "${1:-}" ] && [[ "$1" != --* && "$1" != *"="* ]]; then
  run_id_suffix=$1
  shift
fi

# Remaining args are command-line config overrides.
EXTRA_ARGS=("$@")

NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

DISTRIBUTED_ARGS=(--num_machines "$NNODES" --num_processes "$NUM_PROCESSES")
if [ "${NNODES}" -gt 1 ]; then
  DISTRIBUTED_ARGS+=(--machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT")
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
cp "$SCRIPT_DIR/${BASH_SOURCE[0]##*/}" "${output_dir}/"
echo "Logging to ${output_dir}/train.log"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  "${DISTRIBUTED_ARGS[@]}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --framework.view_image_size '[240,320]' \
  --framework.concat_multi_camera robotwin \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --datasets.vla_data.text_embedding_cache_dir "${TEXT_EMBEDDING_CACHE_DIR:-playground/cache/text_embeds/arx}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.global_batch_size 64 \
  --trainer.max_train_steps 30000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 2000 \
  --trainer.enable_compile true \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${output_dir}/train.log"
