# strix-halo-sglang-cookbook

Recipes and scripts for running [SGLang](https://github.com/sgl-project/sglang) on an AMD Strix Halo (Ryzen AI Max+ 395) mini PC.
Every script listed below has been run end to end on this machine.

## Hardware

| Item | Spec |
|---|---|
| APU | AMD Ryzen AI Max+ 395 (Strix Halo), 16C / 32T, up to 5.19 GHz |
| GPU | Radeon 8060S integrated (RDNA 3.5, `gfx1151`) |
| Memory | 128GB LPDDR5X unified memory, BIOS split: **96GB VRAM** carve-out and ~30GB host RAM (+8GB swap) |
| Storage | 1TB NVMe (Kingston OM8PGP41024Q-A0) |
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.x |
| Container | Docker 29.8.1, image `rocm/sgl-dev:v0.5.20-rocm724-gfx1151-20260923` (ROCm 7.2.4) |

Things to keep in mind on this platform:

- **Host RAM is small (~30GB).** Offload modes that keep a host copy of a component (component offload, snapshot offload, pinned memory) run out of RAM quickly. Prefer keeping everything resident in the 96GB VRAM pool.
- **No FlashAttention on gfx1151.** Use `--attention-backend torch_sdpa` (AOTriton).
- **No FP8 GEMM.** sglang's FP8 linear uses `torch._scaled_mm`, which PyTorch on ROCm only supports on MI300+ (gfx94x/95x).
- **ROCm component-offload bug.** `RocmPlatform` doesn't override `device_shares_host_memory()`, so `finish_use()` moves components back with `non_blocking=True` and pins a second host copy. Avoid `--cpu-offload-components` for now.
- **Long kernels can hang the whole GPU.** The GPU is shared with the desktop. If one compute kernel runs longer than amdgpu's lockup timeout (~10s), the desktop's gfx ring times out and the driver resets the whole GPU. Keep per-request work (resolution × frames) moderate.

## Verified models

### Diffusion

| Model | Task | Script | Settings | Time (warm) | Peak VRAM | Status |
|---|---|---|---|---|---|---|
| [Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Text → Video | [`diffusion/wan/run_wan2.2.sh`](diffusion/wan/run_wan2.2.sh) | 320×576, 33 frames, 16 steps, all resident, `--performance-mode memory` | ~330 s | ~89.4 GB | ✅ Recommended |
| [Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | Text/Image → Video | [`diffusion/wan/run_wan2.2.sh`](diffusion/wan/run_wan2.2.sh) (`WAN_MODEL=ti2v-5b`) | 704×1280, 49 frames, all resident | ~1 h per request | ~66.8 GB | ⚠️ Runs, not recommended |
| [Qwen-Image-2.1](https://huggingface.co/Qwen) | Text → Image / Edit | [`diffusion/qwen-image/run_qwen_image_2.1.sh`](diffusion/qwen-image/run_qwen_image_2.1.sh) | 1024×1024, 40 steps, all resident | ~146 s | ~39.9 GB | ✅ Recommended |

**About Wan2.2-TI2V-5B.** It fits in memory easily (~38GB resident), but it's too slow to be useful on gfx1151:

- At 720p with 49 frames, one request keeps the GPU 100% busy for about an hour. The 2-step test finished; a 50-step run was still going after 47 minutes.
- At the model's default 121 frames, it triggers a GPU hang and a full GPU reset.
- The suspected bottleneck is the fp32 Wan2.2 VAE decode (conv3d on MIOpen).
- Details and untested ideas are at the top of the script.

### LLM

None verified yet. See [`llm/`](llm/).

## Layout

```
strix-halo-sglang-cookbook
├── diffusion
│   ├── wan          # Wan2.2 text-to-video (A14B, TI2V-5B)
│   └── qwen-image   # Qwen-Image text-to-image / editing
├── llm
├── run_docker.sh    # start the SGLang ROCm container
└── README.md
```

## Quick start

```bash
# 1. Put models under ./models (or point MODELS_DIR elsewhere), e.g.
#    models/Wan-AI/Wan2.2-T2V-A14B-Diffusers
#    models/Qwen/Qwen-Image-2.1

# 2. Start the container (mounts this repo and MODELS_DIR at the same paths)
MODELS_DIR=$PWD/models bash run_docker.sh

# 3. Inside the container
bash diffusion/qwen-image/run_qwen_image_2.1.sh
bash diffusion/wan/run_wan2.2.sh                       # A14B (default)
WAN_MODEL=ti2v-5b bash diffusion/wan/run_wan2.2.sh     # 5B (slow, see above)
```

Outputs are written to `outputs/` next to each script, and a log file is saved next to it.
Every script accepts overrides through environment variables (`PROMPT`, `HEIGHT`, `WIDTH`, `FRAMES`, `STEPS`, `SEED`, `MODEL`, ...). The header of each script lists them.
