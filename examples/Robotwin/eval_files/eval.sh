#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/data/gaoning/code_repos/RoboTwin}"
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
    echo "ROBOTWIN_PATH does not exist: ${ROBOTWIN_PATH}" >&2
    exit 1
fi

robotwin_eval_script="${ROBOTWIN_PATH}/script/eval_policy.py"
if [[ ! -f "${robotwin_eval_script}" ]]; then
    echo "RoboTwin eval entry does not exist: ${robotwin_eval_script}" >&2
    exit 1
fi

policy_name="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-42}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${8:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

# Support both upstream overrides and the older StarVLA argparse patch.
checkpoint_args=()
if grep -Eq "add_argument\([\"']--policy_ckpt_path" "${robotwin_eval_script}"; then
    checkpoint_args=(--policy_ckpt_path "${policy_ckpt_path}")
fi

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/robotwin_deploy_policy.XXXXXX.yml")"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

sed \
    -e "s/^host:.*/host: \"${policy_host}\"/" \
    -e "s/^port:.*/port: ${policy_port}/" \
    -e "s/^seed:.*/seed: ${seed}/" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

# Do not silently ignore requested evaluator features in an older RoboTwin checkout.
"${robotwin_python}" - "${runtime_deploy_policy}" "${robotwin_eval_script}" <<'PYCONFIG'
import ast
from pathlib import Path
import sys

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
source = ast.parse(Path(sys.argv[2]).read_text())
keys = {node.value for node in ast.walk(source) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
if config.get("skip_get_obs_within_replan", False) and "skip_get_obs_within_replan" not in keys:
    raise SystemExit("RoboTwin evaluator lacks skip_get_obs_within_replan support. "
                     "Use the FastWAM evaluator patch, or explicitly disable this option in the deploy template.")
if config.get("eval_num_episodes", 100) != 100 and "eval_num_episodes" not in keys:
    raise SystemExit("RoboTwin evaluator fixes the episode count at 100; eval_num_episodes would be ignored.")
print("RoboTwin deploy config:", config, flush=True)
PYCONFIG

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH="${SCRIPT_DIR}"
STARVLA_PATH="${REPO_ROOT}"

export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${STARVLA_PATH}:${PYTHONPATH}"
export PYTHONPATH="${EVAL_FILES_PATH}:${PYTHONPATH}"

cd "${ROBOTWIN_PATH}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"

PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" script/eval_policy.py --config "${runtime_deploy_policy}" \
    "${checkpoint_args[@]}" \
    --overrides \
    --policy_ckpt_path "${policy_ckpt_path}" \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
