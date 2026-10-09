#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONFIG=${CONFIG:-examples/ARX/configs/arx_vr_dual.yaml}
echo "ARX config: $CONFIG" >&2

# Use the unified YAML defaults unless an override was explicitly supplied.
ARGS=()
if [[ -n "${SERVER_IP:-}" || -n "${PORT:-}" ]]; then
  ARGS+=(--server "${SERVER_IP:-127.0.0.1}:${PORT:-5555}")
fi
[[ -z "${CONTROLS:-}" ]] || ARGS+=(--controls "$CONTROLS")
[[ -z "${STAT_KEY:-}" ]] || ARGS+=(--stat_key "$STAT_KEY")
[[ -z "${PROMPT:-}" ]] || ARGS+=(--prompt "$PROMPT")
[[ -z "${CAMERA_ORDER:-}" ]] || ARGS+=(--camera_order "$CAMERA_ORDER")
[[ -z "${RECV_TIMEOUT_MS:-}" ]] || ARGS+=(--recv_timeout_ms "$RECV_TIMEOUT_MS")
[[ -z "${N_EXECUTE:-}" ]] || ARGS+=(--n_execute "$N_EXECUTE")

exec python -u -m deployment.robot_inference.arx_cli run \
  --config "$CONFIG" --execute "${ARGS[@]}" "$@"
