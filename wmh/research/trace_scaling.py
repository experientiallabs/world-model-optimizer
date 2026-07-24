"""Trace scaling law — how world-model fidelity grows with the number of training traces.

The question: feed the harness *more* recorded traces and does its reconstruction fidelity keep
climbing, or saturate? This experiment sweeps the TRAIN trace count (10, 20, 50, … up to the corpus
size) against a **fixed** held-out test set and reports fidelity at each count, so the curve is a
clean "data in → fidelity out" law rather than a moving target (see `scaling_split`).

Two curves, selectable so the cheap one can scout the range before the expensive one runs:

- `base` — the bundled `BASE_ENV_PROMPT`, scored on the fixed test set with a retrieval buffer built
  from the `n` train traces. Isolates how far retrieval *alone* scales (no GEPA, cheap).
- `gepa` — GEPA optimizes the prompt on the `n` train traces, selecting against the fixed valid set,
  then the winning prompt is scored on the fixed test set. The real "learning from more traces"
  curve (one GEPA run per count × seed — expensive).

It is an `Ablation` like every other experiment: conditions = mode × trace-count, swept across seeds
by `run_ablation`, so the across-seed std gives error bars at each point for free. Each `run` reuses
the canonical `optimize_prompt` / `score_prompt` (the same GEPA + replay the harness ships), so the
metric is comparable to `wmh eval` and any judge upgrade lands here automatically. Backends are
factory-injected, so the unit test drives it with fakes and the `scripts/` runner with live Bedrock.

Extensible across benchmarks: it takes an already-ingested corpus and a base prompt, so tau2 today
and terminal-tasks / swe-bench tomorrow are just a different trace file in — no code change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from wmh.core.types import JsonValue, Trace
from wmh.engine.grounding import FetchGrounder, SourceResolver
from wmh.engine.knowledge import seeded_knowledge_text
from wmh.engine.replay import ReplayReport
from wmh.engine.workspace import RepoTreeResolver
from wmh.optimize.judge import RubricDimension
from wmh.research.ablation import Condition, as_int
from wmh.research.pipeline import optimize_prompt, score_prompt
from wmh.research.scaling_split import CorpusSplit, partition_corpus, subsample_train
from wmh.research.seed_stability import BackendFactory
from wmh.retrieval import RetrievalKey

# The prompt/agentic configurations the sweep compares. `base` = the shipped prompt + RAG only
# (cheap); `gepa` = GEPA-optimized on the train sample (expensive); `reason` = base + the
# deliberate-then-answer contract; `reason+kb` = reason + a knowledge base seeded from the train
# sample (train-only, leak-free). Strings (not an enum) so they read straight from a CLI flag and
# serialize into the report's `Condition.params` unchanged.
Mode = str
BASE: Mode = "base"
GEPA: Mode = "gepa"
REASON: Mode = "reason"
REASON_KB: Mode = "reason+kb"
# reason + live prefetch of read-only curl GET URLs (FetchGrounder). NON-HERMETIC: hits the real
# web, and the web has moved since capture — label results accordingly.
REASON_FETCH: Mode = "reason+fetch"
REASON_KB_FETCH: Mode = "reason+kb+fetch"  # both levers together (composability cell)
REASON_VERIFY: Mode = "reason+verify"  # + a second self-check completion per step (2x cost)
# reason + pinned-source grounding of first-touch file reads (SourceResolver). Needs traces
# pinned by instance_id via `source_pins`; unpinned corpora make it a measured no-op.
REASON_SOURCE: Mode = "reason+source"
# reason + a per-step history-digest completion: the ICL history revised into a current-state
# belief profile ("what is running NOW") before predicting. Extra completion per step.
REASON_PROFILE: Mode = "reason+profile"
# reason+source with the staleness gate relaxed to ANNOTATED base versions of edited files —
# the low-risk face of "test-time RAG over the agent's working directory".
REASON_SOURCE2: Mode = "reason+source2"
# source2 + repo-tree grounding of ls/find/grep — the full "test-time RAG over the working
# directory" composite. Needs source_pins.
REASON_WORKSPACE: Mode = "reason+workspace"
# reason + two zero-completion channels: live PyPI/npm registry polls for package actions
# (NON-HERMETIC) + deterministic wc/sort/uniq answers over session-written content (hermetic).
REASON_POLL: Mode = "reason+poll"
MODES: tuple[Mode, ...] = (BASE, GEPA)
# Composable confidence suffixes (WS-A6, D75): any mode above also accepts `+conf` (the
# verbalized-confidence contract field), `+confwhy` (confidence with its one-line justification),
# and `+gateverify@<t>` (confidence-gated verify: the second self-check completion runs only when
# the draft states confidence < t; implies +conf). E.g. `base+conf`, `reason+confwhy`,
# `reason+workspace+conf`, `reason+gateverify@0.7`. Split off by `split_mode` before dispatch.
ALL_MODES: tuple[Mode, ...] = (
    BASE,
    GEPA,
    REASON,
    REASON_KB,
    REASON_FETCH,
    REASON_KB_FETCH,
    REASON_VERIFY,
    REASON_SOURCE,
    REASON_SOURCE2,
    REASON_WORKSPACE,
    REASON_PROFILE,
    REASON_POLL,
)


def split_mode(mode: Mode) -> tuple[Mode, bool, bool, float | None]:
    """Split the composable confidence suffixes off `mode`.

    Returns `(base_mode, confidence, confidence_why, verify_below)`. The base mode must be one of
    `ALL_MODES` — raising here (not deep in dispatch) keeps a typo like `reasn+conf` loud.
    """
    confidence = False
    confidence_why = False
    verify_below: float | None = None
    kept: list[str] = []
    for part in mode.split("+"):
        if part == "conf":
            confidence = True
        elif part == "confwhy":
            confidence = True
            confidence_why = True
        elif part.startswith("gateverify@"):
            confidence = True
            raw = part.removeprefix("gateverify@")
            try:
                verify_below = float(raw)
            except ValueError:
                raise ValueError(
                    f"unknown mode {mode!r}: gateverify threshold {raw!r} is not a number"
                ) from None
            if not 0.0 < verify_below <= 1.0:  # also rejects NaN
                raise ValueError(
                    f"unknown mode {mode!r}: gateverify threshold must be in (0, 1] —"
                    f" {verify_below} would mean {'never' if verify_below <= 0 else 'always'}"
                    " verifying (use the plain mode or +verify instead)"
                )
        else:
            kept.append(part)
    base = "+".join(kept)
    if base not in ALL_MODES:
        raise ValueError(f"unknown mode {mode!r}: base {base!r} is not one of {ALL_MODES}")
    return base, confidence, confidence_why, verify_below


def _as_mode(value: JsonValue) -> Mode:
    assert isinstance(value, str)
    return value


class TraceScalingAblation:
    """Sweep train-trace count × mode against a fixed test set; metric = test fidelity (0..1).

    Conditions are the cartesian product of `modes` and `counts` (e.g. base@10, base@20, gepa@10…).
    `run(condition, seed)` subsamples `n` train traces (nested as `n` grows, seeded), then:
      - `base`: scores `base_prompt` on the fixed test set with a RAG buffer over the `n` traces;
      - `gepa`: optimizes on the `n` traces (selecting on fixed valid) and scores the winner.
    Across-seed std (from `run_ablation`) is the error bar at each point.
    """

    name = "trace-scaling-law"

    def __init__(
        self,
        corpus: list[Trace],
        base_prompt: str,
        *,
        make_backends: BackendFactory,
        counts: Sequence[int],
        modes: Sequence[Mode] = MODES,
        budget: int,
        top_k: int = 5,
        test_frac: float = 0.2,
        valid_frac: float = 0.15,
        sample_turns: str = "all",
        test_cap: int | None = None,
        concurrency: int = 1,
        max_retrieved_observation_chars: int | None = None,
        retrieval_key: RetrievalKey = "state_action",
        score_dimension: RubricDimension | None = None,
        source_pins: str | None = None,
        results_dir: str | None = None,
    ) -> None:
        self._base_prompt = base_prompt
        self._make_backends = make_backends
        self._budget = budget
        self._top_k = top_k
        self._sample_turns = sample_turns
        self._concurrency = concurrency
        self._max_retrieved_observation_chars = max_retrieved_observation_chars
        self._retrieval_key = retrieval_key
        self._score_dimension = score_dimension
        self._source_pins = source_pins
        self._source = SourceResolver.from_file(source_pins) if source_pins else None
        self._tree = RepoTreeResolver(self._source.pins) if self._source is not None else None
        # Per-cell ReplayReports (per-step scores + stated confidences) land here as
        # `<label>_seed<seed>.json` when set. Calibration analysis (WS-A6) joins per-step
        # (confidence, judge score) pairs — the scalar fidelity the ablation returns can't carry
        # that, and rerunning cells to get it would double the serve bill.
        self._results_dir = results_dir
        # Validate every mode string NOW: a typo'd mode must fail before the sweep spends hours
        # of provider budget on the cells that precede it, not when run() finally reaches it.
        for mode in modes:
            split_mode(mode)
        self._modes = list(modes)
        # GEPA optimizes under DEFAULT retrieval (optimize_prompt does not yet thread these knobs),
        # so scoring the evolved prompt under a non-default key/cap would measure it on a retrieval
        # distribution it never optimized for — a confounded gepa-vs-base comparison. Fail fast
        # rather than silently bias the curve; the base arm honours these knobs.
        if GEPA in self._modes and (
            retrieval_key != "state_action" or max_retrieved_observation_chars is not None
        ):
            raise ValueError(
                "gepa mode does not honour retrieval_key / max_retrieved_observation_chars yet; "
                "combining them would score the evolved prompt under retrieval it never optimized "
                "for. Use gepa with default retrieval, or restrict these knobs to base mode."
            )
        self._split: CorpusSplit = partition_corpus(
            corpus, test_frac=test_frac, valid_frac=valid_frac
        )
        # Optionally score against a fixed-size subsample of the (large) test set: replay scores the
        # test serially, so a 200-trace test is ~1000 judge calls per point. A seeded cap keeps the
        # y-axis stable and reproducible across every point at a fraction of the cost. The subsample
        # is fixed (seed 0) so every count/mode/seed is scored on the SAME test traces.
        test = self._split.test
        self._test = subsample_train(test, test_cap, seed=0) if test_cap else test
        # Cap counts at the train pool and drop duplicates created by the cap, preserving order, so
        # a 1000-point sweep on a small corpus collapses cleanly to the few counts it can serve.
        pool = len(self._split.train_pool)
        self._counts = list(dict.fromkeys(min(c, pool) for c in counts if c > 0))

    @property
    def split(self) -> CorpusSplit:
        """The fixed test/valid sets + train pool this sweep runs against (for the header)."""
        return self._split

    @property
    def counts(self) -> list[int]:
        """The effective train counts (capped at the pool, deduped)."""
        return self._counts

    @property
    def scored_test(self) -> list[Trace]:
        """The test traces scored each point (the `test_cap` subsample, or the full test)."""
        return self._test

    def conditions(self) -> list[Condition]:
        return [
            Condition(label=f"{mode}@{n}", params={"mode": mode, "n_train": n})
            for mode in self._modes
            for n in self._counts
        ]

    def run(self, condition: Condition, seed: int) -> float:
        """Score one (mode, n_train) point at `seed` on the fixed test set; fidelity 0..1."""
        mode, confidence, confidence_why, verify_below = split_mode(
            _as_mode(condition.params["mode"])
        )
        n_train = as_int(condition.params["n_train"])
        train = subsample_train(self._split.train_pool, n_train, seed=seed)
        provider, judge, embedder = self._make_backends()

        if mode == GEPA:
            result = optimize_prompt(
                train,
                self._split.valid,
                self._base_prompt,
                provider=provider,
                judge=judge,
                embedder=embedder,
                budget=self._budget,
                seed=seed,
            )
            prompt = result.prompt
        else:
            prompt = self._base_prompt

        # Agentic-mode cells (roadmap f): same base prompt and RAG buffer, plus the deliberation
        # contract; reason+kb seeds a knowledge base from THIS run's train sample only;
        # reason+fetch adds the live curl-GET prefetch (non-hermetic by definition).
        agentic = (
            REASON,
            REASON_KB,
            REASON_FETCH,
            REASON_KB_FETCH,
            REASON_VERIFY,
            REASON_SOURCE,
            REASON_SOURCE2,
            REASON_WORKSPACE,
            REASON_PROFILE,
            REASON_POLL,
        )
        reasoning = mode in agentic
        with_kb = mode in (REASON_KB, REASON_KB_FETCH)
        knowledge = seeded_knowledge_text(train, provider) if with_kb else None
        grounder = FetchGrounder() if mode in (REASON_FETCH, REASON_KB_FETCH) else None
        verify = mode == REASON_VERIFY
        # Resolvers are shared across every (mode, size, seed) cell: their URL/tree memo
        # caches are the point — per-cell reconstruction re-downloads identical pinned files.
        source = None
        tree = None
        if mode in (REASON_SOURCE, REASON_SOURCE2, REASON_WORKSPACE) and self._source is not None:
            source = self._source
        if mode == REASON_WORKSPACE and self._tree is not None:
            tree = self._tree
        profile = mode == REASON_PROFILE
        poll = mode == REASON_POLL

        return score_prompt(
            prompt,
            self._test,
            provider=provider,
            judge=judge,
            embedder=embedder,
            train=train,
            top_k=self._top_k,
            sample_turns=self._sample_turns,
            seed=seed,
            concurrency=self._concurrency,
            max_retrieved_observation_chars=self._max_retrieved_observation_chars,
            retrieval_key=self._retrieval_key,
            score_dimension=self._score_dimension,
            knowledge=knowledge,
            reasoning=reasoning,
            grounder=grounder,
            verify=verify,
            source=source,
            source_annotate_stale=mode in (REASON_SOURCE2, REASON_WORKSPACE),
            tree=tree,
            profile=profile,
            poll=poll,
            confidence=confidence,
            confidence_why=confidence_why,
            verify_below=verify_below,
            on_report=self._report_sink(condition.label, seed),
        )

    def _report_sink(self, label: str, seed: int) -> Callable[[ReplayReport], None] | None:
        """A writer persisting the cell's full ReplayReport under `results_dir`, or None."""
        if self._results_dir is None:
            return None
        out_dir = Path(self._results_dir)

        def write(report: ReplayReport) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{label.replace('/', '_')}_seed{seed}.json"
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        return write
