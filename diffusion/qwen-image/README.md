# Qwen-Image on Strix Halo (gfx1151)

Script: [`run_qwen_image_2.1.sh`](run_qwen_image_2.1.sh)

| Model | Default settings | Time (warm) | Peak VRAM | Status |
|---|---|---|---|---|
| Qwen-Image-2.1 | 1024×1024, 40 steps | ~146 s | ~39.9 GB | ✅ Recommended |

```bash
bash run_qwen_image_2.1.sh                                  # text-to-image
IMAGE=/path/in.png PROMPT="..." bash run_qwen_image_2.1.sh  # image editing
HEIGHT=768 WIDTH=1344 STEPS=40 bash run_qwen_image_2.1.sh
```

## Notes

- **Everything stays resident.** Weights total ~33GB (text_encoder 17.5GB + transformer 14.2GB + vae 1.4GB), with a measured peak of 40.9GB at 1024×1024. The 96GB VRAM pool holds all of it. See also the sglang cookbook page `docs/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1.mdx`.
- **No offload.** Host RAM is only ~30GB, and component offload on ROCm pins a duplicate host copy (see the ROCm offload bug in [`../wan/README.md`](../wan/README.md)).
- **The gfx1151 has no FlashAttention**, so the script uses `torch_sdpa`.
