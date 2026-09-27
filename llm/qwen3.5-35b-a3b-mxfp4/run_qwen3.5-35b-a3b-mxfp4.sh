#!/usr/bin/env bash
# amd/Qwen3.5-35B-A3B-MXFP4 (MoE MXFP4) SGLang server on a single gfx1151 (Radeon 8060S, 96GB VRAM carve-out).
# Needs patches/quark-mxfp4-moe-w4a16-triton.patch; best batched throughput.
#
# Usage:
#   bash run_qwen3.5-35b-a3b-mxfp4.sh
#   PORT=30000 CTX=8192 bash run_qwen3.5-35b-a3b-mxfp4.sh
#   MODEL=/path/to/checkpoint bash run_qwen3.5-35b-a3b-mxfp4.sh
#
# Measurements and notes: see README.md in this folder.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="${MODELS_DIR:-$(cd ../.. && pwd)/models}"
MODEL="${MODEL:-${MODELS_DIR}/amd/Qwen3.5-35B-A3B-MXFP4}"

# Without the W4A16 path, MXFP4 MoE on gfx1151 either crashes or returns garbage.
if ! python3 - <<'PY'
import inspect, sys
from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4_moe as m
sys.exit(0 if "_use_triton_w4a16" in inspect.getsource(m) else 1)
PY
then
  echo "ERROR: this sglang build lacks the W4A16 MXFP4 MoE path. Apply" >&2
  echo "       patches/quark-mxfp4-moe-w4a16-triton.patch (sgl-project/sglang#41389) first." >&2
  exit 1
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=FAST
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-30000}" \
  --attention-backend triton \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
  --context-length "${CTX:-8192}" \
  --max-running-requests "${MAX_REQ:-16}" \
  --trust-remote-code \
  ${EXTRA_ARGS:-} \
  2>&1 | tee "qwen3.5-35b-a3b-mxfp4_$(date +%Y%m%d_%H%M%S).log"
