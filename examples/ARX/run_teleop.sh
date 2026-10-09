#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${CONFIG:-$ROOT/examples/ARX/configs/arx_vr_dual.yaml}"
echo "ARX config: $CONFIG" >&2
exec python -m deployment.teleoperation.cli teleoperate --config "$CONFIG" "$@"
