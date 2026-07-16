# docs — finished products only, kept deliberately small

Production-ready documents: the final deliverable of a PR (AGENTS.md rule 5). Everything else —
working drafts, design notes, raw experiment results, plans — lives in `.agents/docs/`. Every
file here must justify its existence in the table below; a doc that can't gets deleted.

## Layout

- **`research/`** — completed research *writeups* and the figure each one renders. Raw result
  JSONs, vector sources, and experiment logs stay in `.agents/docs/research/`.
- **`reference/`** — how-to references for user-facing systems, verified against current `main`
  at promotion time.

## Why each doc exists

| File | Justification |
|---|---|
| `README.md` | The manifest that makes the justification rule enforceable. |
| `research/trace_scaling_law.md` | The repo's first completed scaling-law result and a load-bearing product claim: measured against a no-RAG (n=0) anchor, trace data lifts fidelity +0.29 for tau-bench but only ~0.02 for terminal-tasks/swe-bench, because retrieval only pays off when the observation is a retrievable function of the recorded (state, action). Cited by launch material and the benchmark work. |
| `research/trace_scaling_law.png` | The figure the writeup renders; also the brand-system visual reference cited by AGENTS.md rule 15. |
| `research/fidelity_transfer_curve.md` | Completed BENCH-B2 research result: training-WM fidelity does not measurably change real-env transfer at smoke scale, but high-fidelity envs destabilize training (the fidelity/stability trade-off, 3× replicated). Grounds the "cheap WMs are enough for training" product claim. |
| `research/fidelity_transfer_curve.png` | The figure the writeup renders. |
| `research/terminal_headtohead.md` | The BENCH-B2 headline result: on-policy WM training did not damage real-environment behavior while offline SFT on demonstrations did (−0.042 within noise vs −0.163, a +0.121 paired gap vs the same base row), with the in-WM channel predicting the ordering. The program's core "training data made with the WM survives reality" claim. |
| `research/terminal_headtohead.png` | The figure the writeup renders. |
| `research/concurrency_scaling_law.md` | The honest cost claim behind the world-model pitch: a world model saves wall-clock **iff real standup cost > reconstruction cost** — so it wins on expensive-build envs (swe ~3–4×, terminal ~5–12×) and *loses* on cheap in-process envs (tau ~0.3×). Documents the fair-comparison methodology and the selection-sensitivity finding (why results use `--select random`, not `simplest`). |
| `research/concurrency_speedup.png` | The cross-benchmark speed-up figure the writeup renders — how many times faster the world model is than the real environment, per benchmark and concurrency level (the "what"). |
| `research/concurrency_cost.png` | The cross-benchmark cost figure the writeup renders — world-model reconstruction vs. real-environment setup cost at W=1, the crossover that explains the speed-up (the "why"). |
| `research/rag_optimization.md` | The follow-up product claim: holding the corpus fixed, `top_k`+an observation cap lift fidelity ~+0.015–0.02 (and fix a token-crowding failure), but semantic embeddings / key engineering / HyDE do not beat lexical retrieval — retrieval is near its oracle ceiling, so the leverage for stateful benchmarks is trace-format state capture and judge weighting, not retrieval. Guides where not to spend effort. |
| `research/rag_optimization.png` | The optimized-vs-unoptimized trace-scaling figure the writeup renders. |
| `research/gepa_scaling_law.md` | The optimization-side complement to the trace scaling law: how fidelity scales with GEPA iterations and training traces, and whether prompt optimization moves the benchmarks retrieval couldn't. Grounds the "prompt/optimization is the leverage" claim in measurements. |
| `research/gepa_scaling_law.png` | The figure the writeup renders. |
| `research/fidelity_tiers.md` | The design + evidence record for `wmh build --fidelity` and `--max-fidelity`: the tier ladder measured on all three benchmarks, the evidence audit that removed unproven ingredients (semantic phi, extra GEPA iterations), and the traps (runaway valset cost, judge pinning). The fidelity-tier UX is a product surface; this is its justification. |
| `research/fidelity_tiers.png` | The figure `research/fidelity_tiers.md` renders (tier ladder, three benchmarks). |
| `research/confidence_calibration.md` | The WS-A6 result the confidence lever ships on: stated confidence is calibrated (AUROC .84–.98, underconfident), abstention buys 5–10 fidelity points, and confidence-gated verify Pareto-dominates always-verify. The product claim behind per-step confidence in serving. |
| `research/confidence_gated_frontier.png` | The one figure the writeup renders: the gated-verify cost frontier (fidelity vs $/cell, never/gated/always) — the headline Pareto claim. |
| `reference/eval_suites.md` | The reproducibility contract every benchmark number in this repo rests on (`examples/<task>/evals/*.toml` + `wmh eval`); commands verified against `main` at promotion. |
| `reference/failover.md` | The `.wmh/fallback.toml` failover contract: which calls ride the chain (world-model) and which never do (the judge), plus the cross-account ladder format; verified live against both AWS accounts. |
| `reference/eval_grid.md` | `wmh eval grid` - the model × condition fidelity grid (base/+RAG/+GEPA/+GEPA+RAG across models, one pinned judge, target-side cost); commands + judge version self-contained; fresh results land in `.wmh/evals/grid/`. |
