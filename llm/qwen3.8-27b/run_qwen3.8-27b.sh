#!/usr/bin/env bash
# Qwen/Qwen3.8-27B (Dense BF16) SGLang server on a single gfx1151 (Radeon 8060S, 96GB VRAM carve-out).
# Works, but slow (every token reads all 55 GB of weights).
#
# Usage:
#   bash run_qwen3.8-27b.sh
#   PORT=30000 CTX=8192 bash run_qwen3.8-27b.sh
#   MODEL=/path/to/checkpoint bash run_qwen3.8-27b.sh
#
# Measurements and notes: see README.md in this folder.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="${MODELS_DIR:-$(cd ../.. && pwd)/models}"
MODEL="${MODEL:-${MODELS_DIR}/Qwen/Qwen3.8-27B}"

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
  2>&1 | tee "qwen3.8-27b_$(date +%Y%m%d_%H%M%S).log"
