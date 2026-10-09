#!/bin/bash
# Multi-GPU Pi0.5 causal ARX fine-tuning script (DeepSpeed ZeRO-2)
# Usage: PI05_MODEL_PATH=/path/to/pi05_base_pytorch_new \
#   bash examples/ARX/train_files/train_pi05-causal.sh [data_mix] [run_id_suffix] [extra_args...]
# Example: PI05_MODEL_PATH=/path/to/pi05_base_pytorch_new \
#   bash examples/ARX/train_files/train_pi05-causal.sh arx_vr debug --trainer.max_train_steps 1000

set -eo pipefail

# Run with the StarVLA training environment already activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

###########################################################################################
# === Configuration ===
Framework_name=Pi05Causal
config_yaml=starVLA/config/training/vla/starvla_pi05_causal.yaml
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints/vla}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
###########################################################################################

data_mix=arx_vr
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
  --framework.action_model.action_horizon 50 \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.disable_state true \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.global_batch_size 64 \
  --trainer.max_train_steps 30000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${output_dir}/train.log"
