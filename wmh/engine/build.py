"""The build pipeline behind `wmh build`.

ingest -> normalize -> split(train/test) -> embed/index -> GEPA optimize -> write `.wmh/` artifact.
Ingestion is part of the build: there is no separate ingest step.
"""

from __future__ import annotations

import hashlib
import json
import shutil

from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.config.manifest import write_manifest
from wmh.core.types import Trace
from wmh.engine.autoconfig import (
    DEFAULT_VAL_CAP,
    AutoFidelityReport,
    CorpusSignature,
    WinnerSpec,
    search_max_fidelity,
    select_candidates,
    signature_estimate,
)
from wmh.engine.knowledge import KnowledgeBase, seed_knowledge
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.reporting import BuildReporter, NullReporter
from wmh.ingest import VendorPull, drop_degenerate_traces, get_adapter
from wmh.optimize import GEPAOptimizer, OptimizeResult, RubricJudge
from wmh.providers import get_provider
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def _count_steps(traces: list[Trace]) -> int:
    return sum(len(trace.steps) for trace in traces)


def ingest(
    config: HarnessConfig, *, file: str | None = None, vendor: VendorPull | None = None
) -> list[Trace]:
    """Load + normalize traces from a file upload or a vendor SDK pull into `Trace` objects."""
    adapter = get_adapter(config.trace_adapter)
    if file is not None:
        return adapter.from_file(file)
    if vendor is not None:
        return adapter.from_vendor(vendor)
    raise ValueError("ingest needs either a file path or a vendor pull")


def _trace_fraction(trace: Trace) -> float:
    """Stable hash of `trace_id` mapped to [0, 1). Order-independent, reproducible across runs."""
    digest = hashlib.blake2b(trace.trace_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def split_traces(traces: list[Trace], train_split: float) -> tuple[list[Trace], list[Trace]]:
    """Deterministic train/held-out split for GEPA (held-out is never seen during evolution).

    Assignment is by a stable hash of `trace_id`, so the same corpus always splits the same way
    regardless of order and rebuilds are reproducible. `train_split` is the target train fraction.
    """
    train: list[Trace] = []
    test: list[Trace] = []
    for trace in traces:
        (train if _trace_fraction(trace) < train_split else test).append(trace)
    return train, test


def split_traces_3way(
    traces: list[Trace], train_frac: float, val_frac: float
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    """Deterministic train / validation / test split by the same stable `trace_id` hash.

    GEPA needs THREE disjoint sets, not two: `train` seeds reflection minibatches, `val` selects
    among candidate prompts, and `test` is the truly-held-out set we report on — never seen by GEPA.
    Using the val set as the reported held-out number (the old 2-way behaviour) lets GEPA select a
    candidate on the very examples we then grade it on, so any "lift" can be selection overfitting.

    Cut points are on the SAME [0,1) hash line as `split_traces`, so `train` is prefix-compatible:
    `[0, train_frac)` = train, `[train_frac, train_frac+val_frac)` = val, rest = test.
    Requires `train_frac + val_frac < 1` so test is non-empty.
    """
    valid = train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1
    if not valid:
        raise ValueError(
            f"need train_frac>0, val_frac>0, train_frac+val_frac<1; "
            f"got train_frac={train_frac}, val_frac={val_frac}"
        )
    train: list[Trace] = []
    val: list[Trace] = []
    test: list[Trace] = []
    val_cut = train_frac + val_frac
    for trace in traces:
        f = _trace_fraction(trace)
        if f < train_frac:
            train.append(trace)
        elif f < val_cut:
            val.append(trace)
        else:
            test.append(trace)
    return train, val, test


def build(
    config: HarnessConfig,
    *,
    file: str | None = None,
    vendor: VendorPull | None = None,
    root: str = ".wmh",
    serve_provider: Provider | None = None,
    judge_provider: Provider | None = None,
    embedder: Embedder | None = None,
    reporter: BuildReporter | None = None,
    max_fidelity: bool = False,
    fidelity_budget: int = DEFAULT_VAL_CAP,
    full_search: bool = False,
    cheap_search: bool = False,
    estimate_only: bool = False,
    drop_degenerate: bool = False,
    gepa_val_cap: int | None = None,
) -> OptimizeResult:
    """Ingest traces and run the full build, creating + persisting the artifact under `root`.

    `serve_provider` / `embedder` are injectable for testing; in production they are constructed
    from `config` (serve provider via the registry, embedder = offline HashingEmbedder sized to
    `config.embed_dim`). `judge_provider`, when given, runs the judge separately from the serve
    provider: pass a pinned single backend when `serve_provider` is a failover chain, so GEPA's
    fitness metric is scored by one model throughout even while rollouts fail over. `reporter`
    receives progress events (defaults to a no-op). Returns the GEPA OptimizeResult (also
    persisted).

    `config.gepa_budget <= 0` skips prompt optimization entirely (the low fidelity tier: RAG
    over the base prompt). `max_fidelity` runs the auto-configuration search after the build
    (see `wmh.engine.autoconfig`): candidates — pruned by corpus signature unless
    `full_search` — are replay-scored on `fidelity_budget` held-out traces with the prompt the
    artifact will serve, and the result lands in `auto_fidelity.json`. The search never changes
    the serve DEFAULTS (plain RAG unless flags were set explicitly): the winner activates at
    runtime via `--max-fidelity`.
    """
    report = reporter or NullReporter()
    paths = ArtifactPaths(root)
    traces = ingest(config, file=file, vendor=vendor)
    if drop_degenerate:
        # Some captures are polluted with all-empty-observation traces (swe-bench is 66% such
        # junk, D24); building on them trains and measures the model on capture damage. Reuses
        # `wmh.ingest.drop_degenerate_traces` — the same filter the research runners
        # (run_trace_scaling / run_fidelity_tiers, via their own --drop-degenerate) apply.
        traces, dropped = drop_degenerate_traces(traces)
        report.activity(f"dropped {dropped} degenerate (all-empty-observation) traces")
    if not traces:
        raise ValueError("no traces ingested; nothing to build")
    report.ingest_done(len(traces), _count_steps(traces))

    # Three disjoint sets: train seeds GEPA's reflection minibatches, val (capped) selects
    # among candidate prompts, and test is never seen by GEPA — `wmh eval` grades on it without
    # selection overfitting. The remainder after train splits evenly into val/test.
    train, val, test = split_traces_3way(traces, config.train_split, (1.0 - config.train_split) / 2)
    report.split_done(len(train), len(val), len(test))

    provider = serve_provider or get_provider(config.serve_provider_config())
    # The GEPA judge can run on a cheaper model (config.judge_model) of the same provider kind;
    # with no judge_model it shares the serve provider.
    if judge_provider is None:
        serve_cfg = config.serve_provider_config()
        if config.judge_model and config.judge_model != serve_cfg.model:
            judge_provider = get_provider(
                serve_cfg.model_copy(update={"model": config.judge_model})
            )
        else:
            judge_provider = provider
    embed = embedder or HashingEmbedder(dim=config.embed_dim)

    # Optional knowledge base: extract canonical env facts (rules/gates, entities, schemas) from
    # the TRAIN split only — the same leak-free discipline as retrieval — into human-editable
    # markdown under knowledge/. Falls back to the full corpus only when the corpus is too small
    # to split (mirrors the `test or train` GEPA fallback below).
    # Known simplification: GEPA below still evolves the prompt under the BASE output contract,
    # even when this artifact will serve with knowledge/reasoning. That composes — the evolved
    # SYSTEM prompt never encodes the output contract (it is appended per-completion by
    # build_env_prompt) — but the optimizer is not yet selecting under agentic-mode conditions.
    if config.knowledge:
        seed_knowledge(KnowledgeBase(paths.knowledge), train or traces, provider)

    # Serving index over the full corpus: at serve time we retrieve from everything we have seen.
    retriever = EmbeddingRetriever(embed)
    retriever.index(traces)
    report.index_done(_count_steps(traces))

    # GEPA evolves the env prompt under serving conditions: it retrieves demos the same way the
    # world model will, but from a SEPARATE retriever it re-indexes over train-only (so held-out
    # steps never retrieve themselves). The embedder is stateless, so it's safe to share.
    # The progress total is GEPA's REAL translated metric-call budget (reported via on_budget),
    # not the iteration count — sizing the bar with iterations made it hit 100% while thousands
    # of valset calls were still running.
    metric_total = {"calls": config.gepa_budget}

    def _on_budget(total: int) -> None:
        metric_total["calls"] = total
        report.optimize_start(total)

    def _on_rollout(done: int, score: float | None) -> None:
        report.rollout(done, metric_total["calls"], score)

    if config.gepa_budget <= 0:
        # Low fidelity tier: no prompt optimization — the artifact serves the base prompt with
        # RAG. OptimizeResult's zeroed metrics honestly say "nothing was optimized".
        result = OptimizeResult(prompt=BASE_ENV_PROMPT)
    else:
        # Optimize against the SAME rubric we evaluate with (RubricJudge). NOTE: the judge MODEL
        # may differ — config.judge_model defaults to a cheap per-provider model for GEPA cost,
        # while `wmh eval` pins the judge to the requested serve-grade model — so
        # held_out_accuracy is only directly comparable to eval fidelity when --judge-model
        # matches the eval judge.
        optimizer = GEPAOptimizer(
            provider,
            RubricJudge(judge_provider),
            retriever=EmbeddingRetriever(embed),
            on_rollout=_on_rollout,
            on_budget=_on_budget,
            on_activity=report.activity,
        )
        # Candidate selection only needs a stable, SMALL sample: every GEPA iteration re-scores
        # the whole valset, and steps (not traces) are what bound cost — a long-trace corpus once
        # turned a 4-iteration tier into $131. The default ceiling applies always; a fidelity
        # tier may widen it (`gepa_val_cap`, in steps).
        gepa_val = _cap_gepa_valset(val or train, gepa_val_cap or _GEPA_VAL_STEP_CAP)
        result = optimizer.optimize(train, gepa_val, BASE_ENV_PROMPT, config.gepa_budget)
        # A GEPA candidate can be empty - a weak reflection LM (e.g. a self-reflecting open model)
        # sometimes proposes a blank env prompt that still scores acceptably on easy steps and
        # gets selected. An empty env prompt is never a valid artifact, so fall back to base.
        if not result.prompt.strip():
            result = OptimizeResult(
                prompt=BASE_ENV_PROMPT,
                frontier=result.frontier or [BASE_ENV_PROMPT],
                metrics=result.metrics,
            )
    report.optimize_done(
        result.metrics.held_out_accuracy, len(result.frontier), result.metrics.rollouts_used
    )

    if estimate_only:
        # Low tier: ship the signature's strongest ESTIMATED config with no LLM search — the
        # ladder's floor. Persisted as the winner so `--max-fidelity` serves it and so the
        # higher tiers (rebuilt separately) seed the identical incumbent from the same signature.
        signature = CorpusSignature.from_traces(train or traces)
        estimate = signature_estimate(signature, has_pins=False)
        if estimate.knowledge and not paths.knowledge.is_dir():
            seed_knowledge(KnowledgeBase(paths.knowledge), train or traces, provider)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.auto_fidelity.write_text(
            AutoFidelityReport(
                winner_label=estimate.label,
                considered=[estimate.label],
                winner_spec=WinnerSpec.from_candidate(estimate),
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
    elif max_fidelity:
        # Score the candidate configs on the held-out split with the prompt the artifact will
        # serve (leak-free: demos + candidate KB from train only; `test or train` mirrors GEPA's
        # tiny-corpus fallback). The result is a RUNTIME menu, not a serve default: the report is
        # persisted (plus the KB when the winner needs it) and `--max-fidelity` activates the
        # winner when the model is run — plain runs stay pure RAG.
        # The build's search considers only candidates the RUNTIME can activate — no
        # source_pins here: the workspace candidate is research-measurable (the tiers runner
        # passes pins), but serve-side activation doesn't exist yet, and a persisted winner
        # the runtime can't serve would make auto_fidelity.json a lie.
        # Seed the KB into the ARTIFACT before the search and hand its rendered text in, so a
        # knowledge winner's score was measured on the exact KB that serves (and the extraction
        # runs once, not once ephemeral + once to persist). If no knowledge candidate wins, the
        # seeded dir is removed again — a plain artifact stays knowledge-free.
        kb_seeded_for_search = False
        signature = CorpusSignature.from_traces(train or traces)
        # The floor: the same estimate the `low` tier ships. Every searching tier carries it as
        # its incumbent, so medium/high/max improve-or-hold on low, never regress (monotonic).
        incumbent = signature_estimate(signature, has_pins=False)
        menu = select_candidates(signature, full_ladder=full_search, cheap_only=cheap_search)
        if (any(c.knowledge for c in menu) or incumbent.knowledge) and not paths.knowledge.is_dir():
            seed_knowledge(KnowledgeBase(paths.knowledge), train or traces, provider)
            kb_seeded_for_search = True
        kb = KnowledgeBase(paths.knowledge)
        auto = search_max_fidelity(
            result.prompt,
            train or traces,
            test or train,
            provider,
            RubricJudge(judge_provider),
            embed,
            val_cap=fidelity_budget,
            # Pass the already-computed menu so the search doesn't recompute the identical
            # signature+select_candidates (and so the KB-seed decision above and the scored set
            # can't drift from separate calls).
            candidates=menu,
            incumbent=incumbent,
            # Whatever the artifact's KB renders (even empty) is what candidates measure —
            # passing None here would let the search re-extract a DIFFERENT ephemeral KB
            # and crown a winner the shipped artifact cannot reproduce.
            knowledge_text=kb.render() if paths.knowledge.is_dir() else None,
        )
        if kb_seeded_for_search and not auto.winner.knowledge and not config.knowledge:
            # The search created the KB only to measure the kb candidate; a non-knowledge
            # winner means the artifact should stay knowledge-free (a knowledge/ dir is a
            # user-visible surface).
            shutil.rmtree(paths.knowledge, ignore_errors=True)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.auto_fidelity.write_text(auto.model_dump_json(indent=2), encoding="utf-8")

    _persist(paths, config, retriever, result)
    return result


# Ceiling on GEPA's candidate-selection valset, in steps (~one full-val pass per iteration).
# Small on purpose: selection only needs a stable ranking signal, and fidelity saturates fast.
_GEPA_VAL_STEP_CAP = 16


def _cap_gepa_valset(traces: list[Trace], max_steps: int = _GEPA_VAL_STEP_CAP) -> list[Trace]:
    """A prefix of `traces` totalling at most `max_steps` steps (always at least one trace).

    `split_traces` already shuffles deterministically, so the prefix is an unbiased, stable
    sample of the held-out split.
    """
    capped: list[Trace] = []
    steps = 0
    for trace in traces:
        if capped and steps + len(trace.steps) > max_steps:
            break
        capped.append(trace)
        steps += len(trace.steps)
    return capped


def _persist(
    paths: ArtifactPaths,
    config: HarnessConfig,
    retriever: EmbeddingRetriever,
    result: OptimizeResult,
) -> None:
    """Write config, prompts, frontier, metrics, retrieval index, and manifest under `.wmh/`."""
    save_config(config, paths.root)
    paths.base_prompt.parent.mkdir(parents=True, exist_ok=True)
    paths.base_prompt.write_text(BASE_ENV_PROMPT, encoding="utf-8")
    paths.optimized_prompt.write_text(result.prompt, encoding="utf-8")
    paths.frontier.write_text(json.dumps(result.frontier, indent=2), encoding="utf-8")
    paths.metrics.write_text(result.metrics.model_dump_json(indent=2), encoding="utf-8")
    retriever.save(paths.index)
    # Hash every artifact file and record the digests so `wmh verify` and
    # `WorldModel.load` can detect corruption or partial writes before serving.
    write_manifest(
        paths.root,
        [
            paths.config,
            paths.optimized_prompt,
            paths.base_prompt,
            paths.frontier,
            paths.metrics,
            paths.index / "embeddings.npy",
            paths.index / "steps.jsonl",
        ],
    )
