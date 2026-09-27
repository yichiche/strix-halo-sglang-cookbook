# Qwen3.5-35B-A3B-MXFP4 on Strix Halo (gfx1151)

Script: [`run_qwen3.5-35b-a3b-mxfp4.sh`](run_qwen3.5-35b-a3b-mxfp4.sh) · Checkpoint: [amd/Qwen3.5-35B-A3B-MXFP4](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4) · Patch: [`patches/quark-mxfp4-moe-w4a16-triton.patch`](patches/quark-mxfp4-moe-w4a16-triton.patch) (upstream PR [sgl-project/sglang#41389](https://github.com/sgl-project/sglang/pull/41389))

| Type | Weights | MGSM-en (chat, thinking off) | GSM8K 5-shot | Decode bs=1 | Median TPOT | Batched (16 concurrent) | Status |
|---|---|---|---|---|---|---|---|
| MoE MXFP4 | 24.6 GB | 0.97 (100 q) | 0.970 (200 q) | 8.37 tok/s | 118 ms | 79.7 tok/s | ✅ With patch (best batched throughput) |

```bash
cd /sgl-workspace/sglang
git apply /path/to/strix-halo-sglang-cookbook/llm/qwen3.5-35b-a3b-mxfp4/patches/quark-mxfp4-moe-w4a16-triton.patch
bash /path/to/strix-halo-sglang-cookbook/llm/qwen3.5-35b-a3b-mxfp4/run_qwen3.5-35b-a3b-mxfp4.sh
```

The script refuses to start if the patch isn't applied.

## Why it needs a patch

Without it, this checkpoint fails on gfx1151 in one of two ways:

- **Default:** it crashes after weight loading with `NameError: name 'e8m0_shuffle' is not defined`.
- **With `SGLANG_USE_AITER=1`:** it starts, but the output is garbage (GSM8K 8%, and chat just repeats the question). The scales are preshuffled for gfx950, and aiter's FP4 fused MoE kernel doesn't produce correct results on gfx1151.

The patch adds a W4A16 path for HIP GPUs without FP4 hardware. The MXFP4 weights stay packed and are upcast inside `triton_kernels`' matmul, with bf16 activations. Only the MoE experts of this checkpoint are MXFP4; attention and the shared expert are BF16.

## Notes

- **Speed:** batched throughput is 1.7x BF16. Single-request decode is slightly slower than BF16, because the W4A16 kernel runs with default `triton_kernels` flags (`num_warps=8`) and isn't tuned for gfx1151 yet.
- **Memory:** weights load in ~16 s and take about a third of BF16's memory.
- **Kernel check:** on real layer weights, the patched MoE output is within ~0.4% of an fp32 reference built from the dequantized weights.
