#!/usr/bin/env bash
# Validate an LLM on gfx1151: launch server -> health -> chat -> accuracy -> decode speed -> shutdown.
# Same server flags as run_llm_server.sh. Results go to llm_runs/<tag>_<time>/.
#
#   MODEL=/path/to/checkpoint bash validate_llm.sh                 # chat + GSM8K 5-shot (100 q) + bs=1 speed
#   MODEL=... MGSM_N=100 GSM_N=0 bash validate_llm.sh             # chat eval with thinking off (recommended for Qwen3.5)
#   MODEL=... GSM_N=200 SKIP_BENCH=1 bash validate_llm.sh
#
# Note: GSM8K 5-shot caps each answer at 512 tokens. Qwen3.5 MoE checkpoints sometimes start a long
# <think> block in completion mode and get truncated, which reads as an accuracy drop; see README.md.
set -uo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:?set MODEL}"
TAG="${TAG:-$(basename "$MODEL")}"
PORT="${PORT:-30000}"
OUT="llm_runs/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=FAST
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --attention-backend "${ATTN:-triton}" \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
  --context-length "${CTX:-8192}" \
  --max-running-requests "${MAX_REQ:-16}" \
  --trust-remote-code \
  ${EXTRA:-} \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; sleep 5; pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null' EXIT

echo "[$TAG] server pid $SERVER_PID, log $OUT/server.log"
for i in $(seq 1 "${READY_TIMEOUT:-1800}"); do
  if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then echo "[$TAG] ready after ${i}s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[$TAG] SERVER DIED"; tail -40 "$OUT/server.log"; exit 1; fi
  sleep 1
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "[$TAG] NOT READY (timeout)"; tail -40 "$OUT/server.log"; exit 1; }

echo "[$TAG] chat:"
curl -s -m 600 "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "default",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
  "max_tokens": 256, "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}' | tee "$OUT/chat.json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:300])' || echo "[$TAG] CHAT FAILED"

if [[ -n "${MGSM_N:-}" ]]; then
  echo "[$TAG] mgsm_en chat, thinking off (${MGSM_N} q):"
  python3 -m sglang.test.run_eval --port "$PORT" --eval-name mgsm_en --num-examples "$MGSM_N" --num-threads 16 \
    --max-tokens "${MGSM_MAX_TOKENS:-1024}" --temperature 0 --chat-template-kwargs '{"enable_thinking": false}' \
    2>&1 | tee "$OUT/mgsm.log" | grep -iE "score|accuracy|latency"
fi

[[ "${GSM_N:-100}" == 0 ]] || {
echo "[$TAG] gsm8k (${GSM_N:-100} q):"
python3 -m sglang.test.few_shot_gsm8k --num-questions "${GSM_N:-100}" --parallel 16 --port "$PORT" 2>&1 | tee "$OUT/gsm8k.log" | grep -E "Accuracy|Invalid|Latency|throughput"
}

[[ "${SKIP_BENCH:-0}" == 1 ]] || {
echo "[$TAG] decode speed (bs=1, in 512 / out 256):"
python3 -m sglang.bench_serving --backend sglang --port "$PORT" --dataset-name random \
  --random-input-len 512 --random-output-len 256 --random-range-ratio 1 \
  --num-prompts 4 --max-concurrency 1 2>&1 | tee "$OUT/bench_bs1.log" \
  | grep -E "Output token throughput|Median TPOT|Median TTFT"
}

grep -E "Load weight end|KV Cache is allocated|max_total_num_tokens|avail mem" "$OUT/server.log" | tail -4
echo "[$TAG] DONE -> $OUT"
