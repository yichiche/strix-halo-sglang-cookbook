#!/usr/bin/env bash
# Launch an SGLang LLM server on a single gfx1151 (Radeon 8060S, 96GB VRAM carve-out).
#
# Models (pick with LLM_MODEL):
#   qwen3.5-35b-a3b         Qwen/Qwen3.5-35B-A3B            MoE BF16   (default)
#   qwen3.5-35b-a3b-mxfp4   amd/Qwen3.5-35B-A3B-MXFP4       MoE MXFP4  (needs patches/quark-mxfp4-moe-w4a16-triton.patch)
#   qwen3.8-27b             Qwen/Qwen3.8-27B                dense BF16
#   qwen3.5-35b-a3b-fp8     Qwen/Qwen3.5-35B-A3B-FP8        MoE FP8    (accurate but ~4x slower than BF16)
#
# Usage:
#   bash run_llm_server.sh
#   LLM_MODEL=qwen3.8-27b PORT=30000 bash run_llm_server.sh
#   MODEL=/path/to/checkpoint bash run_llm_server.sh
#
# Measurements, known issues and unsupported checkpoints: see README.md in this folder.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="${MODELS_DIR:-$(cd .. && pwd)/models}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-35b-a3b}"

case "${LLM_MODEL}" in
  qwen3.5-35b-a3b)       MODEL="${MODEL:-${MODELS_DIR}/Qwen/Qwen3.5-35B-A3B}" ;;
  qwen3.5-35b-a3b-mxfp4) MODEL="${MODEL:-${MODELS_DIR}/amd/Qwen3.5-35B-A3B-MXFP4}" ;;
  qwen3.8-27b)           MODEL="${MODEL:-${MODELS_DIR}/Qwen/Qwen3.8-27B}" ;;
  qwen3.5-35b-a3b-fp8)   MODEL="${MODEL:-${MODELS_DIR}/Qwen/Qwen3.5-35B-A3B-FP8}" ;;
  *)
    echo "Unknown LLM_MODEL=${LLM_MODEL}" >&2
    exit 1
    ;;
esac

if [[ "${LLM_MODEL}" == "qwen3.5-35b-a3b-mxfp4" ]] && ! python3 - <<'PY'
import inspect, sys
from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4_moe as m
sys.exit(0 if "_use_triton_w4a16" in inspect.getsource(m) else 1)
PY
then
  echo "ERROR: this sglang build lacks the W4A16 MXFP4 MoE path; on gfx1151 the MXFP4 MoE" >&2
  echo "       either crashes or returns garbage. Apply patches/quark-mxfp4-moe-w4a16-triton.patch" >&2
  echo "       (sgl-project/sglang#41389) to your sglang checkout first." >&2
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
  2>&1 | tee "llm_${LLM_MODEL}_$(date +%Y%m%d_%H%M%S).log"
