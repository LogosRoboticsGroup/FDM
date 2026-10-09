#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_python="${LIBERO_PYTHON:-${LIBERO_python:-python}}"
starVLA_python="${STARVLA_PYTHON:-${starVLA_python:-python}}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"



##### === variables for which evaluation to setup ===
your_ckpt=$1 # results/Checkpoints/.../steps_20000_pytorch_model.pt
task_suite_name=$2 # align with your model | libero_goal
gpu_id=$3   # GPU id to use (e.g. 0, 1, 2, ...)
base_port=$4 # unique port for this eval instance
seed=${5:-${SEED:-42}}
##### === variables for which evaluation to setup ===

num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
host="127.0.0.1"

STARVLA_PYTHON="${starVLA_python}" GPU_ID="$gpu_id" PORT="$base_port" \
    bash examples/LIBERO/eval_files/run_policy_server.sh "$your_ckpt" &


# Get the server PID
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT


# Extract model_root from your_ckpt
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
log_path="${model_root}/logs/${task_suite_name}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"



extra_args=()
if [ -n "${TASK_ID:-}" ]; then
    extra_args+=(--args.task-id "$TASK_ID")
    video_out_path="${video_out_path}/task_${TASK_ID}"
    folder_name="${folder_name}_task_${TASK_ID}"
fi
if [ -n "${PREDICTION_HORIZON:-}" ]; then
    extra_args+=(--args.prediction-horizon "$PREDICTION_HORIZON")
fi
if [ -n "${SIGMA_SHIFT:-}" ]; then
    extra_args+=(--args.sigma-shift "$SIGMA_SHIFT")
fi
if [ "${BINARIZE_GRIPPER:-1}" = "0" ]; then
    extra_args+=(--args.no-binarize-gripper)
fi
if [ "${SAVE_VIDEO:-1}" = "0" ]; then
    extra_args+=(--args.no-save-video)
fi

"${LIBERO_python}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.seed "$seed" \
    --args.action-horizon "${ACTION_HORIZON:-10}" \
    --args.num-steps-wait "${NUM_STEPS_WAIT:-30}" \
    --args.num-inference-steps "${NUM_INFERENCE_STEPS:-10}" \
    "${extra_args[@]}" \
    --args.out-path "$video_out_path"  \
    2>&1 | tee ${log_path}/${folder_name}.log

echo "Evaluation completed. Videos saved to ${video_out_path}, logs saved to ${log_path}/${folder_name}.log"

if [ -n "$server_pid" ]; then
    echo "Killing server process with PID: $server_pid"
    kill $server_pid
else
    echo "No server process found to kill."
fi
