#!/usr/bin/env bash
# Wan2.2 text-to-video on a single gfx1151 (Radeon 8060S, 96GB VRAM carve-out)
#
# Models (pick with WAN_MODEL):
#   a14b     Wan2.2-T2V-A14B-Diffusers   recommended (default)
#   ti2v-5b  Wan2.2-TI2V-5B-Diffusers    runs, but ~1h per request at 720p
#
# Usage:
#   bash run_wan2.2.sh
#   WAN_MODEL=ti2v-5b bash run_wan2.2.sh
#   WAN_MODEL=ti2v-5b IMAGE=/path/first_frame.png bash run_wan2.2.sh
#   HEIGHT=... WIDTH=... FRAMES=... STEPS=... PROMPT="..." bash run_wan2.2.sh
#
# Why these settings, measurements and known issues: see README.md in this folder.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="${MODELS_DIR:-$(cd ../.. && pwd)/models}"
WAN_MODEL="${WAN_MODEL:-a14b}"

EXTRA_ARGS=()
case "${WAN_MODEL}" in
  a14b)
    MODEL="${MODEL:-${MODELS_DIR}/Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
    HEIGHT="${HEIGHT:-320}"; WIDTH="${WIDTH:-576}"
    FRAMES="${FRAMES:-33}"; STEPS="${STEPS:-16}"
    EXTRA_ARGS+=(--performance-mode memory)
    ;;
  ti2v-5b)
    echo "WARNING: Wan2.2-TI2V-5B runs but takes ~1h+ per request at 720p on gfx1151;" \
         "121 frames triggers a GPU hang. See README.md in this folder." >&2
    MODEL="${MODEL:-${MODELS_DIR}/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
    HEIGHT="${HEIGHT:-704}"; WIDTH="${WIDTH:-1280}"
    FRAMES="${FRAMES:-49}"; STEPS="${STEPS:-50}"
    if [[ -n "${IMAGE:-}" ]]; then
      EXTRA_ARGS+=(--image-path "${IMAGE}")
    fi
    ;;
  *)
    echo "Unknown WAN_MODEL=${WAN_MODEL} (expected a14b or ti2v-5b)" >&2
    exit 1
    ;;
esac

export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rocm
export SGLANG_DIFFUSION_ATTENTION_BACKEND=torch_sdpa
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=FAST
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

PROMPT="${PROMPT:-A cat and a dog baking a cake together in a kitchen. \
The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. \
The kitchen is cozy, with sunlight streaming through the window.}"

sglang generate \
  --model-path "${MODEL}" \
  --log-level info \
  --backend sglang \
  --attention-backend torch_sdpa \
  --prompt "${PROMPT}" \
  --negative-prompt " " \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num-inference-steps "${STEPS}" \
  --num-frames "${FRAMES}" \
  --seed "${SEED:-42}" \
  --save-output \
  --output-path outputs \
  --num-gpus 1 \
  --component-residency all=resident \
  --vae-tiling true \
  --warmup-mode request \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "wan2.2_${WAN_MODEL}_$(date +%Y%m%d_%H%M%S).log"
