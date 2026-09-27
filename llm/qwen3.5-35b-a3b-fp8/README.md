# Qwen3.5-35B-A3B-FP8 on Strix Halo (gfx1151)

Script: [`run_qwen3.5-35b-a3b-fp8.sh`](run_qwen3.5-35b-a3b-fp8.sh) · Checkpoint: [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)

| Type | Weights | MGSM-en (chat, thinking off) | GSM8K 5-shot | Decode bs=1 | Median TPOT | Batched (16 concurrent) | Status |
|---|---|---|---|---|---|---|---|
| MoE FP8 (block 128x128) | 37.5 GB | 0.98 (100 q) | 0.77 (200 q)* | 2.20 tok/s | 442 ms | 11.5 tok/s | ⚠️ Accurate, but ~4x slower than BF16 |

```bash
bash run_qwen3.5-35b-a3b-fp8.sh
```

\* See the [GSM8K 5-shot note](../README.md#gsm8k-5-shot-and-qwen35-moe): the lower score comes from truncated `<think>` blocks, not from worse reasoning.

## Notes

- **Prefer BF16 or MXFP4 on this GPU.** gfx1151 has no FP8 hardware. The Triton W8A8 block-FP8 GEMM and fused-MoE kernels run without gfx1151-tuned configs, so this checkpoint is ~4x slower than BF16 even though it has fewer bytes to read.
- **The kernels are numerically correct.** Against an fp32 reference on real checkpoint weights, the relative error is ~2.6% for linear layers and ~4.3% for the fused MoE, which is normal for FP8 with dynamic per-token-group activation quantization. With thinking off, MGSM matches BF16.
