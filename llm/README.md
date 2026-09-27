# LLM on Strix Halo (gfx1151)

Each model has its own folder with a run script and notes. [`validate_llm.sh`](validate_llm.sh) is shared: it launches a server, runs a chat check, the accuracy evals and a bs=1 speed test, then shuts the server down.

All numbers come from one gfx1151 with these server flags: `--attention-backend triton --mem-fraction-static 0.85 --context-length 8192 --max-running-requests 16`. The image is `rocm/sgl-dev:v0.5.20-rocm724-gfx1151-20260923` (sglang 0.5.21.dev, torch 2.9.1, triton 3.7.0).

## Results

| Folder | Checkpoint | Type | Weights | MGSM-en (chat, thinking off) | GSM8K 5-shot | Decode bs=1 | Median TPOT | Status |
|---|---|---|---|---|---|---|---|---|
| [`qwen3.5-35b-a3b`](qwen3.5-35b-a3b/) | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | MoE BF16 | 71.9 GB | 0.98 (100 q) | 0.945 (200 q) | 9.38 tok/s | 104 ms | ✅ Recommended |
| [`qwen3.5-35b-a3b-mxfp4`](qwen3.5-35b-a3b-mxfp4/) | [amd/Qwen3.5-35B-A3B-MXFP4](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4) | MoE MXFP4 | 24.6 GB | 0.97 (100 q) | 0.970 (200 q) | 8.37 tok/s | 118 ms | ✅ With patch (best batched throughput) |
| [`qwen3.8-27b`](qwen3.8-27b/) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Dense BF16 | 55.6 GB | — | 0.98 (50 q) | 2.69 tok/s | 367 ms | ✅ Works, slow |
| [`qwen3.5-35b-a3b-fp8`](qwen3.5-35b-a3b-fp8/) | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) | MoE FP8 | 37.5 GB | 0.98 (100 q) | 0.77 (200 q)* | 2.20 tok/s | 442 ms | ⚠️ Accurate, but ~4x slower than BF16 |

Batched throughput while serving the 100 MGSM requests at 16 concurrent:

| Checkpoint | Output throughput |
|---|---|
| Qwen3.5-35B-A3B BF16 | 46.8 tok/s |
| Qwen3.5-35B-A3B-MXFP4 (patched) | 79.7 tok/s |
| Qwen3.5-35B-A3B-FP8 | 11.5 tok/s |

\* See [the GSM8K 5-shot note below](#gsm8k-5-shot-and-qwen35-moe): the FP8 checkpoint's reasoning isn't worse, but its answers get truncated more often.

```bash
bash llm/qwen3.5-35b-a3b/run_qwen3.5-35b-a3b.sh      # OpenAI-compatible server on :30000
```

## Which one to use

- **Qwen3.5-35B-A3B, MoE.** MoE is the sweet spot for this machine, because decode speed is bounded by memory bandwidth (~256 GB/s) and only ~3B parameters are active per token.
  - **BF16** has the best single-request latency.
  - **MXFP4 with the patch** has the best batched throughput (1.7x BF16), and its weights take a third of the memory.
- **Dense 27B BF16** works and is accurate, but slow.
- **Skip FP8 on this GPU.** It's accurate, but gfx1151 has no FP8 hardware, which makes it ~4x slower than BF16.

## Validating a model

```bash
MODEL=/path/to/checkpoint MGSM_N=100 GSM_N=0 bash llm/validate_llm.sh   # chat eval, thinking off (recommended for Qwen3.5)
MODEL=/path/to/checkpoint GSM_N=200 bash llm/validate_llm.sh            # GSM8K 5-shot + bs=1 speed
```

Results go to `llm/llm_runs/<tag>_<time>/`.

## GSM8K 5-shot and Qwen3.5 MoE

`sglang.test.few_shot_gsm8k` uses completion mode and caps each answer at 512 tokens. The Qwen3.5 MoE checkpoints sometimes open a long `<think>` block there and get cut off before answering. Breaking down the 200-question run:

| Checkpoint | Truncated `<think>` | Accuracy on answers that finished |
|---|---|---|
| BF16 | 4 / 200 | 187 / 193 (96.9%) |
| FP8 | 33 / 200 | 147 / 154 (95.5%) |
| MXFP4 | 0 / 200 | 189 / 195 (96.9%) |

For these models, prefer the chat eval with thinking off (`MGSM_N=100 GSM_N=0`).

## General notes

- **Hybrid attention.** All of these checkpoints mix Gated DeltaNet linear-attention layers with full attention. On gfx1151 they run on the Triton GDN kernels and `--attention-backend triton`. CUDA graph capture works.
- **Host RAM is only ~30 GB.** Keep weights on the GPU and don't use CPU offload (see the top-level README).
