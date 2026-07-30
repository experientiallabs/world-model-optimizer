# Eval grid (`wmo eval grid`)

A **grid** scores one eval suite across many **(model × condition)** cells on the *same* held-out
split, by the *same* judge - answering the project's core question: does `base → +RAG → +GEPA →
+GEPA+RAG` actually lift world-model reconstruction fidelity, for which serving models, at what
cost? It's the multi-model sibling of `wmo eval` (one trace) and `wmo eval run` (one suite), built
on the same open-loop scorer (`wmo.evals.open_loop.evaluate_files`).

## Conditions

Each model is scored under four conditions:

| condition | label | prompt | retrieval |
|---|---|---|---|
| `base` | `base` | `BASE_ENV_PROMPT` | off |
| `base_rag` | `wmo/rag` | base | DreamGym top-k |
| `gepa` | `wmo/gepa` | per-(suite × model) GEPA-evolved prompt | off |
| `gepa_rag` | `wmo/gepa/rag` | evolved prompt | DreamGym top-k |

A model with no evolved prompt in the `--gepa-prompts` dir is scored on `base`/`base_rag` only.

## Invariants (what makes cells comparable)

- **Pinned judge.** One Bedrock Opus-4.8 `RubricJudge` grades every cell, regardless of target - a
  Qwen target is never judged by Qwen - and it **never switches models**. Its `JUDGE_VERSION`
  (currently `rubric-v2`) is stamped on the result JSON; numbers from different judge versions are
  not comparable, so never mix them in one chart.
- **Same-model capacity fallover.** A Bedrock target fails over on capacity errors across regions
  and then to the *same model on the direct Anthropic API* (Bedrock Opus is heavily throttled; the
  direct API is the identical model, so what's measured is unchanged - see `wmo.evals.failover`).
  The pinned judge fails over *only* to that same-model direct API, never to a different model - a
  judge that swapped models mid-grid would score cells on different scales (cf.
  `docs/reference/failover.md`).
- **Leak-free splits.** Cells score the reserved `test` band of the same 3-way `train/val/test`
  split GEPA used (`--val-frac`, default `(1 - train_split)/2`, matching `wmo build`); GEPA selects
  on `val`, so a `+GEPA` cell is never scored on the traces its prompt was tuned on. A GEPA prompt
  byte-identical to base is treated as "no evolved prompt" (its GEPA cells are skipped) so
  same-prompt noise is never reported as a `+GEPA` delta.
- **Cost is target-side.** A `MeteredProvider` wraps only the target, so each cell's `$` is target
  inference cost (never judge cost); a self-hosted model has no pricing row and reports no cost
  (blank, not a misleading `$0.00`).
- **Bounded target output.** `CappedProvider` clamps the target's `max_tokens` so a reasoning target
  can't make a grid take hours; the judge is uncapped.

## Commands

```bash
# One benchmark, N API models × 4 conditions -> result JSON + fidelity bar chart PNG
wmo eval grid <suite> \
  --models "Opus 4.8:bedrock:us.anthropic.claude-opus-4-8,GPT-5.5:openai:gpt-5.5" \
  --gepa-prompts <dir-of-<label>.txt-prompts> --limit-traces 8 --out grid.png

# A self-hosted model (OpenAI-compatible) runs in its own process (its base URL is process-global),
# so its cells land in a separate JSON; grid-plot merges them into one chart:
wmo eval grid-plot <api>.json <qwen>.json --out grid.png --dataset-label <suite>

# The whole grid (every benchmark × model × condition) as one heatmap:
wmo eval grid-heatmap <result.json>... --out heatmap.png
```

`--models` entries are `Label:provider:model` (a self-hosted vLLM model is just `provider=openai`
with `OPENAI_BASE_URL` set). `--gepa-prompts` points at a directory of `<label>.txt` evolved prompts
(a model with no matching file, or one whose file equals the base prompt, is scored on
`base`/`base_rag` only). Results write to `.wmo/evals/grid/<suite>-<runid>.json` unless `--out` is
given; every figure is re-renderable from that JSON with `grid-plot` / `grid-heatmap`, so charts
are never committed.

## Reproducing a full grid

The trace corpora are the ones under `packages/environment-capture/<suite>/` (e.g.
`tau-bench`, `terminal-tasks`, `swe-bench`) - fetch a corpus with
`wmo download <suite>`, evolve one prompt per (suite × model) with `wmo build`, collect them into a
`--gepa-prompts` directory as `<label>.txt`, then run `wmo eval grid <suite>/default` per benchmark
and `grid-heatmap` across the saved JSONs. All cells in a comparison must share one `JUDGE_VERSION`.
