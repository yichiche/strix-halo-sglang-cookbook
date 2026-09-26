# Wan2.2 on Strix Halo (gfx1151)

Script: [`run_wan2.2.sh`](run_wan2.2.sh). Pick the model with `WAN_MODEL`.

| `WAN_MODEL` | Model | Default settings | Time (warm) | Peak VRAM | Status |
|---|---|---|---|---|---|
| `a14b` (default) | Wan2.2-T2V-A14B-Diffusers | 320×576, 33 frames, 16 steps | ~330 s | ~89.4 GB | ✅ Recommended |
| `ti2v-5b` | Wan2.2-TI2V-5B-Diffusers | 704×1280, 49 frames, 50 steps | ~1 h per request | ~66.8 GB | ⚠️ Runs, not recommended |

```bash
bash run_wan2.2.sh                                                # A14B
WAN_MODEL=ti2v-5b bash run_wan2.2.sh                              # 5B, 720p
WAN_MODEL=ti2v-5b IMAGE=/path/first_frame.png bash run_wan2.2.sh  # 5B image-to-video
HEIGHT=... WIDTH=... FRAMES=... STEPS=... PROMPT="..." bash run_wan2.2.sh
```

## A14B notes

A14B is a two-expert MoE (`transformer` + `transformer_2`). Resident memory:

| Component | Size |
|---|---|
| 2 experts | 53.24 GB |
| text_encoder (UMT5) | 21.16 GB |
| vae | 0.47 GB |
| **Total** | **74.87 GB** |

That leaves ~21GB of the 96GB pool for activations. VAE decode is the tight spot: at 720p a single conv3d allocation needs ~8.9GB and OOMs in `DecodingStage` after every denoise step has already finished.

Pitfalls (it took 4 iterations to get a stable config):

- **Always specify residency.** With `--performance-mode auto`, both 26.6GB experts go to host pageable memory. That exceeds the ~30GB host RAM, and the kernel OOM killer SIGKILLs the process (exit code -9).
- **Don't use `--cpu-offload-components text_encoder`.** This is an upstream sglang bug on ROCm:
  - `finish_use()` in `component_residency_strategies.py` calls `module.to("cpu", non_blocking=X)`, where `X` comes from `current_platform.device_shares_host_memory()`.
  - That check only recognizes NVIDIA GB10/Jetson unified memory. `RocmPlatform` doesn't override it, so on this unified-memory APU `non_blocking` is wrongly `True`.
  - The host then pins a second copy of the component (21GB → 42GB), exhausting RAM and swap with `HIP error: out of memory`.
  - `--pin-cpu-memory false` doesn't help, because it only affects the load path. Keep everything resident instead.
- **`component-residency` takes precedence over `cpu-offload-components`**, and the `all` selector also makes text_encoder resident.
- **Decode memory is not linear in resolution.** It's roughly ~6.8GB fixed plus ~2.3e-6 GB per pixel. At 480×832, decode still needed 7.71GB with only 7.98GB free (fragmented), which is too tight to run reliably.
- **`--performance-mode memory` frees the ~5.7GB of leftover denoise activations** before decode.

When raising resolution or frames, watch the "free vs tried to allocate" numbers at the decode step.

## TI2V-5B notes: runs, but too slow to be useful here

Weights are the transformer (~19GB, fp32), the UMT5 text_encoder (21.16GB), and the VAE (2.63GB). That's ~38GB resident, and host RAM use is only 6–9GB. **Memory is not the problem; speed is.**

| Settings | Result |
|---|---|
| 704×1280, 121 frames (model default), 2 steps | ❌ GPU hang + full GPU MODE2 reset |
| 704×1280, 49 frames, 2 steps | ✅ Completes, peak 68.4GB, but 7725 s for warmup + 1 request (~1 h of 100%-busy GPU per request) |
| 704×1280, 49 frames, 50 steps | ⚠️ Still running after ~47 min at 100% GPU (interrupted by a reboot, never completed) |

- **The GPU hang is not an OOM.** dmesg shows the *desktop* gfx ring (`gfx_0.0.0`) timing out. A single long compute kernel holds the GPU past amdgpu's ~10s lockup timeout, the ring reset fails, and the driver resets the whole GPU (VRAM is lost and the desktop glitches).
- **The stage timers are misleading** because they don't sync with the GPU. The log attributes ~3700 s to `TextEncodingStage` and 0.05 s/step to denoising.
- **Suspected bottleneck:** the Wan2.2 VAE decode (fp32 by default, conv3d via MIOpen on gfx1151).
- **Untested ideas:** `--vae-decode-precision bf16`, lower resolution, or raising `amdgpu.lockup_timeout` (a kernel parameter; needs a reboot).
