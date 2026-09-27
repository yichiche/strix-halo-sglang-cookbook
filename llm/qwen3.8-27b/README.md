# Qwen3.8-27B (BF16) on Strix Halo (gfx1151)

Script: [`run_qwen3.8-27b.sh`](run_qwen3.8-27b.sh) · Checkpoint: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

| Type | Weights | GSM8K 5-shot | Decode bs=1 | Median TPOT | Status |
|---|---|---|---|---|---|
| Dense BF16 | 55.6 GB | 0.98 (50 q) | 2.69 tok/s | 367 ms | ✅ Works, slow |

```bash
bash run_qwen3.8-27b.sh
```

## Notes

- **Accurate but slow.** Every decoded token reads all ~55 GB of weights, so decode tops out around 4.6 tok/s at this machine's ~256 GB/s memory bandwidth. It measures 2.69 tok/s.
- **Memory:** the server is ready in ~160 s, and ~22 GB of VRAM is left for KV cache.
- **Hybrid attention:** same as Qwen3.5 (Gated DeltaNet + full attention), on Triton kernels. CUDA graph capture works.
