#!/bin/bash
# wmh-max AgentWorldBench row: 7 domains on the merged-main max configs.
# WM arms = --fidelity max rebuilds (auto_fidelity winners, applied via --max-fidelity);
# base arms = BASE_ENV_PROMPT + reasoning + verify (corpus-free max levers).
# Judge = pinned gpt-5.4-mini via the Azure shim (:8766), their eval.py unmodified, temp 0.
set -uo pipefail

ROOT="$HOME/Desktop/Projects/wmh-wm-benchmarks"
EVAL=/tmp/qwen-agentworld/eval
RES="$ROOT/.wmh/agentworldbench/results54max"
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
  local extra=()
  if [ "$mode" = "wm" ]; then extra=(--model-dir "$modeldir" --max-fidelity); else extra=(--reasoning --verify); fi
  (cd "$ROOT" && uv run python .agents/scripts/agentworldbench/awb_infer.py \
    --data "$DATA/${data}_test.jsonl" --mode "$mode" "${extra[@]}" \
    --concurrency 3 --resume \
    --output "$RES/$name/predictions.jsonl" < /dev/null) || { echo "[driver] $name INFER FAILED"; return 1; }
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

run_arm terminal_wm  terminal wm  .wmh/models/terminal-tasks-max
run_arm mcp_wm       mcp      wm  .wmh/models/tau-bench-max
run_arm swe_wm       swe      wm  .wmh/models/swe-bench-max
run_arm search_base  search   base
run_arm android_base android  base
run_arm web_base     web      base
run_arm os_base      os       base

echo "[driver] all infer done $(date); waiting for judges..."
wait "${JUDGE_PIDS[@]}"
echo "[driver] all judges done $(date)"
curl -s "$SHIM_URL/usage"; echo
echo "[driver] COMPLETE $(date)"
