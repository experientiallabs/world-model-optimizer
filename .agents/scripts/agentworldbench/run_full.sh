#!/bin/bash
# Full AgentWorldBench run: 7 domains + terminal base ablation (~2,524 rows).
# Infer = wmh WMs on Bedrock Opus 4.8 (one arm at a time, concurrency 3, resumable);
# judge = gpt-5.4-mini via the Azure shim on :8766 (their eval.py, unmodified, temp 0),
# backgrounded per arm so judging overlaps the next arm's infer.
set -uo pipefail

ROOT="$HOME/Desktop/Projects/wmh-wm-benchmarks"
EVAL=/tmp/qwen-agentworld/eval
RES="$ROOT/.wmh/agentworldbench/results54"
DATA="$ROOT/.wmh/agentworldbench/data_full"
PY="$ROOT/.venv/bin/python"
SHIM_URL=http://127.0.0.1:8766

ensure_shim() {
  if ! curl -sf "$SHIM_URL/usage" > /dev/null; then
    echo "[driver] shim down — starting it"
    export "$(grep AZURE_OPENAI_API_KEY "$ROOT/.env" | tail -1)"
    (cd "$ROOT" && nohup uv run python .agents/scripts/agentworldbench/judge_shim.py \
      --backend azure --endpoint https://google-sheets.openai.azure.com \
      --model gpt-5.4-mini --port 8766 >> "$RES/shim.log" 2>&1 &)
    sleep 6
  fi
}

JUDGE_PIDS=()
run_arm() {
  local name=$1 data=$2 mode=$3 modeldir=${4:-}
  echo "[driver] === $name (infer) $(date) ==="
  mkdir -p "$RES/$name"
  (cd "$ROOT" && uv run python .agents/scripts/agentworldbench/awb_infer.py \
    --data "$DATA/${data}_test.jsonl" --mode "$mode" \
    ${modeldir:+--model-dir "$modeldir"} \
    --concurrency 3 --resume \
    --output "$RES/$name/predictions.jsonl") || { echo "[driver] $name INFER FAILED"; return 1; }
  local pred_n jud_n
  pred_n=$(wc -l < "$RES/$name/predictions.jsonl")
  jud_n=$(wc -l < "$RES/$name/judged.jsonl" 2>/dev/null || echo 0)
  if [ "$jud_n" -eq "$pred_n" ]; then
    echo "[driver] === $name judge already complete ($jud_n rows) — skipping ==="
    return 0
  fi
  ensure_shim
  echo "[driver] === $name (judge, backgrounded) $(date) ==="
  (cd "$EVAL" && "$PY" eval.py judge \
    --predictions "$RES/$name/predictions.jsonl" \
    --judge-base-url "$SHIM_URL/v1" --judge-model gpt-5.4-mini --judge-api-key EMPTY \
    --temperature 0 --output-dir "$RES/$name" > "$RES/$name/judge.log" 2>&1) &
  JUDGE_PIDS+=($!)
}

run_arm terminal_wm  terminal wm  packages/environment-capture/terminal-tasks/models/terminal-tasks
run_arm terminal_base terminal base
run_arm mcp_wm       mcp      wm  packages/environment-capture/tau-bench/models/tau-bench
run_arm swe_wm       swe      wm  packages/environment-capture/swe-bench/models/swe-bench
run_arm search_base  search   base
run_arm android_base android  base
run_arm web_base     web      base
run_arm os_base      os       base

echo "[driver] all infer done $(date); waiting for judges..."
wait "${JUDGE_PIDS[@]}"
echo "[driver] all judges done $(date)"
curl -s "$SHIM_URL/usage"; echo
for d in "$RES"/*/judged.jsonl; do
  echo "=== $d ==="
  (cd "$EVAL" && "$PY" eval.py score --predictions "$d" 2>/dev/null | tail -15)
done
echo "[driver] COMPLETE $(date)"
