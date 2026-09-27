# Qwen3.5-35B-A3B (BF16) on Strix Halo (gfx1151)

Script: [`run_qwen3.5-35b-a3b.sh`](run_qwen3.5-35b-a3b.sh) · Checkpoint: [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

| Type | Weights | MGSM-en (chat, thinking off) | GSM8K 5-shot | Decode bs=1 | Median TPOT | Batched (16 concurrent) | Status |
|---|---|---|---|---|---|---|---|
| MoE BF16 | 71.9 GB | 0.98 (100 q) | 0.945 (200 q) | 9.38 tok/s | 104 ms | 46.8 tok/s | ✅ Recommended |

```bash
bash run_qwen3.5-35b-a3b.sh          # OpenAI-compatible server on :30000
```

## Notes

- **Best single-request latency of the tested LLMs.** Only ~3B parameters are active per token, and decode on this machine is bounded by memory bandwidth (~256 GB/s).
- **Memory:** weights load in ~64 s. With `--mem-fraction-static 0.85`, ~25 GB of VRAM is left for KV cache (~610k tokens).
- **Hybrid attention:** Gated DeltaNet linear-attention layers run on the Triton GDN kernels, full attention on `--attention-backend triton`. CUDA graph capture works.
