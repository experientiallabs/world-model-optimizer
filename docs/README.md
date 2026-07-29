# docs — finished products only, kept deliberately small

Production-ready documents: the final deliverable of a PR (AGENTS.md rule 5). Working drafts,
design notes, raw experiment results, and plans are not docs and never land here. Every file
here must justify its existence in the table below; a doc that can't gets deleted.

## Layout

- **`research/`**: completed research *writeups*, with the figures they render under
  `research/figures/`. Raw result JSONs, vector sources, and experiment logs are not docs.
- **`reference/`** — how-to references for user-facing systems, verified against current `main`
  at promotion time.
- **`cookbook/`**: end-to-end walks through the whole pipeline on one benchmark, each step one
  real CLI command plus the artifact it creates. One file per benchmark.
- **`usage.md`**: the only root page besides this one. The terse map of the CLI surface, one line
  of purpose and one artifact per command.

## Why each doc exists

| File | Justification |
|---|---|
| `README.md` | The manifest that makes the justification rule enforceable. |
| `usage.md` | The one-page map of the CLI surface: every command's purpose and the artifact it leaves behind, grouped by pipeline / optimizers / traces / platform. The reference-style counterpart to the cookbook's narrative order, and the page that keeps every other doc from re-explaining what a command is. |
| `cookbook/tau-bench.md` | The canonical end-to-end example: one pass through the whole pipeline (setup, build, pool, optimize, optional distill and compression, serve) told on tau-bench, each step one real CLI command plus the artifact it creates. Commands verified against `main`; the provenance rules (world-model simulated vs real-episode, cache-adjusted effective cost per completed task, savings as estimates) are stated once here and inherited by every number in the walk. |
| `research/world_model_findings.md` | The single research record: six layered studies (data, retrieval, optimization, test-time compute, self-knowledge, economics; PRs #72, #97, #55, #120, #41, with #83/#98 as instruments) with shared protocol and judge provenance stated once. Every product claim about world-model fidelity and cost traces to a section of this document. |
| `research/figures/trace_scaling_law.png` | The trace-scaling figure (fidelity vs trace count, n=0 anchored) the record's data layer renders; also the brand-system visual reference cited by AGENTS.md rule 15. |
| `research/figures/rag_optimization.png` | The retrieval-optimization figure: optimized vs unoptimized retrieval curves per benchmark. |
| `research/figures/gepa_scaling_law.png` | The GEPA-scaling figure: fidelity vs GEPA budget and trace count, RAG baselines, judge panel. |
| `research/figures/fidelity_tiers.png` | The fidelity-tier figure: the build-tier ladder on three benchmarks. |
| `research/figures/confidence_gated_frontier.png` | The gated-verify cost frontier (fidelity vs $/cell, never/gated/always), the confidence layer's headline Pareto claim. |
| `research/figures/concurrency_speedup.png` | The concurrency speed-up figure: how many times faster the world model is than the real environment, per benchmark and concurrency level (the "what"). |
| `research/figures/concurrency_cost.png` | The concurrency cost figure: world-model reconstruction vs real-environment setup cost at W=1, the crossover that explains the speed-up (the "why"). |
| `research/canonical_tau_result.md` | The canonical tau-bench result: the compound-loop argument (routing + compaction + distillation measured jointly), the master document improvement reports and external claims trace back to. DRAFT: carries a publication-gate banner until Silen lifts it; numbers slot in from the grid, corner analyses, and real-episode validation. |
| `reference/eval_suites.md` | The reproducibility contract every benchmark number in this repo rests on (`examples/<task>/evals/*.toml` + `wmo eval`); commands verified against `main` at promotion. |
| `reference/failover.md` | The `.wmo/fallback.toml` failover contract: which calls ride the chain (world-model) and which never do (the judge), plus the cross-account ladder format; verified live against both AWS accounts. |
| `reference/eval_grid.md` | `wmo eval grid` - the model × condition fidelity grid (base/+RAG/+GEPA/+GEPA+RAG across models, one pinned judge, target-side cost); commands + judge version self-contained; fresh results land in `.wmo/evals/grid/`. |
| `reference/closed_loop.md` | The other half of eval: `wmo eval --mode closed-loop` runs a live agent against the world model and scores task success (gold-judged) instead of per-step fidelity; the contract `wmo/evals/closed_loop.py` and `agreement.py` implement. |
| `reference/ingest.md` | The ingestion contract behind `wmo build --source`: one pluggable `TraceAdapter` seam that turns traces from any observability stack (or plain chat logs) into the harness trace format, plus one section per source adapter (Phoenix, Langfuse, LangSmith, Braintrust, PostHog, Mastra) with its export shape and field mapping. |
| `reference/harness_delta.md` | The `HarnessDelta` interface `wmo optimize` mutates through: the typed, precondition-guarded update representation that defines the optimizer agent's search space; the contract `wmo/harness/` implements. |
| `reference/cost_quality_dial.md` | The one operator control on a routing endpoint: `cost_quality` in [0, 1], what each leg of the mapping changes, the five measured anchors with their limits (interpolation is of the knobs, not the outcome; the savings leg trades guard strictness), and the three ways to set it (Python, `wmo optimize route tune`, `endpoint.toml`) plus the live config and savings routes. |
| `reference/distill.md` | The `wmo optimize distill` how-to: distillation of a Tinker LoRA student from real benchmark rollouts (config-selected: harbor's terminus-2 or tau2-bench episodes) (run config TOML, cost/budget model, run-dir artifacts, `report`, resume, troubleshooting); the contract `wmo/distill/` implements, verified against completed end-to-end runs. |
| `reference/connect-library.md` | The programmatic contract behind `wmo.connect`: `get_connector(name).pull(auth, query)` for host-side consumers (the platform's connector tools), the `ConnectorAuth`/`PullQuery`/`ContextItem` shapes, per-service targeting, and the caller-supplies-tokens rule; verified against `wmo/connect/`. |
