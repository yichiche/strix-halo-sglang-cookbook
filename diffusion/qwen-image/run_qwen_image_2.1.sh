#!/usr/bin/env bash
# Qwen-Image-2.1 text-to-image / image editing on a single gfx1151 (Radeon 8060S, 96GB VRAM carve-out)
#
# Why this configuration (see the sglang cookbook: docs/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1.mdx):
#  - Weights total ~33GB (text_encoder 17.5 + transformer 14.2 + vae 1.4); measured
#    peak 40.9GB at 1024x1024. 96GB VRAM holds all of it, so everything stays resident.
#  - No offload: host RAM is only ~30GB, and on ROCm sglang misjudges non_blocking in
#    component-offload's finish_use(), which doubles host pinned copies
#    (see the notes in ../wan/run_wan2.2.sh).
#  - The gfx1151 has no FlashAttention, so use torch_sdpa.
#
# Usage:
#   bash run_qwen_image_2.1.sh                                  # text-to-image
#   IMAGE=/path/in.png PROMPT="..." bash run_qwen_image_2.1.sh  # image editing
#   HEIGHT=768 WIDTH=1344 STEPS=40 bash run_qwen_image_2.1.sh
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="${MODELS_DIR:-$(cd ../.. && pwd)/models}"
MODEL="${MODEL:-${MODELS_DIR}/Qwen/Qwen-Image-2.1}"

export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rocm
export SGLANG_DIFFUSION_ATTENTION_BACKEND=torch_sdpa
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=FAST
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

PROMPT="${PROMPT:-A capybara reading a book by candlelight in a cozy library, warm light, highly detailed}"

EXTRA_ARGS=()
if [[ -n "${IMAGE:-}" ]]; then
  EXTRA_ARGS+=(--image-path "${IMAGE}")
fi

sglang generate \
  --model-path "${MODEL}" \
  --model-id Qwen-Image-2.1 \
  --log-level info \
  --backend sglang \
  --attention-backend torch_sdpa \
  --prompt "${PROMPT}" \
  --height "${HEIGHT:-1024}" \
  --width "${WIDTH:-1024}" \
  --num-inference-steps "${STEPS:-40}" \
  --seed "${SEED:-42}" \
  --save-output \
  --output-path outputs \
  --num-gpus 1 \
  --component-residency all=resident \
  --warmup-mode request \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "qwen_image_2.1_$(date +%Y%m%d_%H%M%S).log"
