#!/usr/bin/env bash
# Wan2.2 text-to-video on a single gfx1151 (Radeon 8060S, 96GB VRAM carve-out)
#
# Supported models (pick with WAN_MODEL):
#   a14b     Wan2.2-T2V-A14B-Diffusers   ✅ recommended  (default)
#   ti2v-5b  Wan2.2-TI2V-5B-Diffusers    ⚠️ runs, but NOT recommended on this machine
#
# Usage:
#   bash run_wan2.2.sh                                    # A14B, 320x576, 33 frames, 16 steps
#   WAN_MODEL=ti2v-5b bash run_wan2.2.sh                  # 5B, 704x1280 (720p), 49 frames
#   WAN_MODEL=ti2v-5b IMAGE=/path/first_frame.png bash run_wan2.2.sh   # 5B image-to-video
#   HEIGHT=... WIDTH=... FRAMES=... STEPS=... PROMPT="..." bash run_wan2.2.sh
#
# ---------------------------------------------------------------------------
# A14B notes
# ---------------------------------------------------------------------------
# A14B is a two-expert MoE (transformer + transformer_2). Both experts resident take
# 53.24GB, text_encoder 21.16GB, vae 0.47GB -> 74.87GB resident, leaving ~21GB of the
# 96GB pool for activations. VAE decode is the tight spot: at 720p a single conv3d
# allocation needs ~8.9GB and OOMs in DecodingStage after all denoise steps finish.
#
# Pitfalls found along the way (4 iterations before it was stable):
#  - Do not leave residency unspecified: --performance-mode auto puts both 26.6GB
#    experts in host pageable memory, which blows the ~30GB host RAM and gets the
#    process SIGKILLed by the kernel OOM killer (exit code -9).
#  - Do not use --cpu-offload-components text_encoder. Upstream sglang bug on ROCm:
#    finish_use() (component_residency_strategies.py) calls
#    module.to("cpu", non_blocking=X), where X depends on
#    current_platform.device_shares_host_memory(). That check only recognizes
#    NVIDIA GB10/Jetson unified memory; RocmPlatform does not override it, so on this
#    unified-memory APU non_blocking is wrongly True and the host pins a second copy
#    of the component (21GB -> 42GB), exhausting RAM+swap with
#    "HIP error: out of memory". --pin-cpu-memory false does not help (it only
#    affects the load path). So keep everything resident instead.
#  - Note: component-residency takes precedence over cpu-offload-components, and the
#    "all" selector also makes text_encoder resident.
#  - Decode memory is NOT linear in resolution: roughly ~6.8GB fixed + ~2.3e-6 GB/pixel.
#    480x832 still needed 7.71GB with only 7.98GB free (fragmented) -> too tight.
#  - --performance-mode memory frees the ~5.7GB of leftover denoise activations
#    before decode.
#
# Verified: 320x576, 33 frames, 16 steps runs end to end in ~330s (warm),
# peak ~89.4GB. When raising resolution/frames, watch "free vs tried to allocate"
# at the decode step.
#
# ---------------------------------------------------------------------------
# TI2V-5B notes -- runs, but too slow to be useful here
# ---------------------------------------------------------------------------
# Weights: transformer ~19GB (fp32) + UMT5 text_encoder 21.16GB + VAE 2.63GB
# (~38GB resident, host RAM only 6-9GB). Memory is not the problem; speed is.
#
#  - 704x1280, 121 frames (the model default): GPU HANG + full GPU MODE2 reset.
#    Not an OOM. dmesg shows the *desktop* gfx ring (gfx_0.0.0) timing out: a single
#    long compute kernel holds the GPU past amdgpu's ~10s lockup timeout, the ring
#    reset fails, and the whole GPU is reset (VRAM lost, desktop glitches).
#  - 704x1280, 49 frames, 2 steps: completes (peak 68.4GB), but took 7725s for
#    warmup + 1 request, i.e. roughly 1 hour of 100%-busy GPU per request.
#    The stage timers are misleading (no GPU sync): the log attributes ~3700s to
#    TextEncodingStage and 0.05s/step to denoising. Suspected real bottleneck:
#    the Wan2.2 VAE decode (fp32 by default, conv3d via MIOpen on gfx1151).
#  - 704x1280, 49 frames, 50 steps: still in progress after ~47 min at 100% GPU
#    (interrupted, not completed).
#  - Untested ideas: --vae-decode-precision bf16, lower resolution, or raising
#    amdgpu.lockup_timeout (kernel parameter, needs reboot).
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
         "121 frames triggers a GPU hang. See notes at the top of this script." >&2
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
