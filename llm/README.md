# LLM on Strix Halo (gfx1151)

Scripts:

- [`run_llm_server.sh`](run_llm_server.sh) — launch an OpenAI-compatible SGLang server (pick the model with `LLM_MODEL`).
- [`validate_llm.sh`](validate_llm.sh) — launch, run a chat check, accuracy evals and a bs=1 speed test, then shut down.
- [`patches/quark-mxfp4-moe-w4a16-triton.patch`](patches/quark-mxfp4-moe-w4a16-triton.patch) — needed for MXFP4 MoE on gfx1151 (upstream PR [sgl-project/sglang#41389](https://github.com/sgl-project/sglang/pull/41389)).

All numbers below come from one gfx1151 with these server flags: `--attention-backend triton --mem-fraction-static 0.85 --context-length 8192 --max-running-requests 16`. The image is `rocm/sgl-dev:v0.5.20-rocm724-gfx1151-20260923` (sglang 0.5.21.dev, torch 2.9.1, triton 3.7.0).

## Results

| `LLM_MODEL` | Checkpoint | Type | Weights | MGSM-en (chat, thinking off) | GSM8K 5-shot | Decode bs=1 | Median TPOT | Status |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5-35b-a3b` (default) | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | MoE BF16 | 71.9 GB | 0.98 (100 q) | 0.945 (200 q) | 9.38 tok/s | 104 ms | ✅ Recommended |
| `qwen3.5-35b-a3b-mxfp4` | [amd/Qwen3.5-35B-A3B-MXFP4](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4) | MoE MXFP4 | 24.6 GB | 0.97 (100 q) | 0.970 (200 q) | 8.37 tok/s | 118 ms | ✅ With patch (best batched throughput) |
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Dense BF16 | 55.6 GB | — | 0.98 (50 q) | 2.69 tok/s | 367 ms | ✅ Works, slow |
| `qwen3.5-35b-a3b-fp8` | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) | MoE FP8 | 37.5 GB | 0.98 (100 q) | 0.77 (200 q)* | 2.20 tok/s | 442 ms | ⚠️ Accurate, but ~4x slower than BF16 |
| — | [amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16) | Dense INT4 | 19.5 GB | — | — | — | — | ❌ Not supported by sglang |

Batched throughput while serving the 100 MGSM requests at 16 concurrent:

| Checkpoint | Output throughput |
|---|---|
| Qwen3.5-35B-A3B BF16 | 46.8 tok/s |
| Qwen3.5-35B-A3B-MXFP4 (patched) | 79.7 tok/s |
| Qwen3.5-35B-A3B-FP8 | 11.5 tok/s |

\* See [the GSM8K 5-shot note below](#gsm8k-5-shot-and-qwen35-moe): the FP8 checkpoint's reasoning isn't worse, but its answers get truncated more often.

## Which one to use

- **Qwen3.5-35B-A3B, MoE.** MoE is the sweet spot for this machine, because decode speed is bounded by memory bandwidth (~256 GB/s) and only ~3B parameters are active per token.
  - **BF16** has the best single-request latency.
  - **MXFP4 with the patch** has the best batched throughput (1.7x BF16), and its weights take a third of the memory, which leaves more room for KV cache.
- **Dense 27B BF16** works and is accurate, but decode is ~2.7 tok/s: every token reads all 55 GB of weights.
- **Skip FP8 on this GPU.** gfx1151 has no FP8 hardware, and the Triton W8A8 block-FP8 GEMM and MoE kernels have no gfx1151-tuned configs.

## Notes

### MXFP4 MoE needs a patch

Without it, `amd/Qwen3.5-35B-A3B-MXFP4` fails on gfx1151 in one of two ways:

- **Default:** it crashes after weight loading with `NameError: name 'e8m0_shuffle' is not defined`.
- **With `SGLANG_USE_AITER=1`:** it starts, but the output is garbage (GSM8K 8%). The scales are preshuffled for gfx950 and aiter's FP4 fused MoE kernel doesn't produce correct results on gfx1151.

The patch adds a W4A16 path for HIP GPUs without FP4 hardware. The MXFP4 weights stay packed and are upcast inside `triton_kernels`' matmul, with bf16 activations. Only the MoE experts of this checkpoint are MXFP4; attention and the shared expert are BF16. To apply it:

```bash
cd /sgl-workspace/sglang
git apply /path/to/strix-halo-sglang-cookbook/llm/patches/quark-mxfp4-moe-w4a16-triton.patch
```

`run_llm_server.sh` refuses to start the MXFP4 model if the patch isn't applied. The W4A16 kernel still runs with default `triton_kernels` flags and hasn't been tuned for gfx1151 yet.

### FP8 is accurate, just slow

The block-FP8 kernels are numerically fine on gfx1151. Checked against an fp32 reference on real checkpoint weights, the relative error is ~2.6% for the linear layers and ~4.3% for the fused MoE, which is normal for FP8 with dynamic activation quantization. With thinking off, MGSM matches BF16 (0.98).

### GSM8K 5-shot and Qwen3.5 MoE

`sglang.test.few_shot_gsm8k` uses completion mode and caps each answer at 512 tokens. The Qwen3.5 MoE checkpoints sometimes open a long `<think>` block there and get cut off before answering. Breaking down the 200-question run:

| Checkpoint | Truncated `<think>` | Accuracy on answers that finished |
|---|---|---|
| BF16 | 4 / 200 | 187 / 193 (96.9%) |
| FP8 | 33 / 200 | 147 / 154 (95.5%) |
| MXFP4 | 0 / 200 | 189 / 195 (96.9%) |

For these models, prefer the chat eval with thinking off: `MGSM_N=100 GSM_N=0 bash validate_llm.sh`.

### AWQ INT4 (Quark W4A16) isn't supported

`amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` fails at load time with `NotImplementedError: No quark compatible scheme was found` (int4, per-group 128). sglang's Quark integration has no W4A16 int4 scheme yet, and the model card only documents vLLM.

### Other things to keep in mind

- All four working checkpoints are hybrid attention: Gated DeltaNet linear-attention layers plus full attention. On gfx1151 they run on the Triton GDN kernels and `--attention-backend triton`. CUDA graph capture works.
- Host RAM is only ~30 GB, so keep weights on the GPU. Don't use CPU offload (see the top-level README).
