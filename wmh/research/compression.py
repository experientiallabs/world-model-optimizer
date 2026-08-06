"""Token-compression method slate and append-stability audit primitives (track C1).

The compression track's Round 0 is an offline audit that runs before any live accuracy
spend: same-input determinism, append-stability (compress X, compress X+delta, diff the
common prefix), token-reduction ratio, and compressor latency. This module holds the
pieces that are pure functions of their inputs: the method implementations, the two
selection rules under test (per-input percentile vs absolute threshold), and the audit
metrics. Model-backed scorers (GPT-2 self-information, LLMLingua-2 keep probabilities)
live in the runner script and plug in as callables, so this module and its tests carry
no torch dependency.

A method here is a callable over a turn-structured transcript: `compress(turns) -> str`,
where `turns[0]` is the task text and each later element is one serving turn. Append
stability is then exactly `compress(turns[:k+1]).startswith(compress(turns[:k]))`.

Terminology pin: "word" in this module means a whitespace-delimited unit with its
trailing whitespace preserved (lossless re-join). Token-reduction ratios reported in
findings use a real tokenizer in the runner; method-internal budgets are word budgets.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

TURN_SEP = "\n\n"

Turns = Sequence[str]
Compressor = Callable[[Turns], str]

WORD_RE = re.compile(r"\S+\s*")
UNIT_RE = re.compile(r"[^\n]*\n|[^\n]+$")


def join_turns(turns: Turns) -> str:
    """Canonical raw text of a transcript prefix."""
    return TURN_SEP.join(turns)


def split_words(text: str) -> list[str]:
    """Whitespace-delimited units, trailing whitespace attached, lossless on join."""
    return WORD_RE.findall(text)


def split_units(text: str) -> list[str]:
    """Line-level units (trailing newline attached), lossless on join.

    Lines rather than sentences: our corpora are tool calls, logs, and JSON, where
    sentence splitters produce garbage and lines are the natural salience unit.
    """
    return UNIT_RE.findall(text)


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Selection rules: the variable Round 0 actually tests.
# ---------------------------------------------------------------------------


def select_percentile(scores: Sequence[float], keep_ratio: float) -> list[bool]:
    """Stock selection: keep the top `keep_ratio` fraction by score, per input.

    This is the per-input percentile rule Selective Context and the LLMLingua family
    ship. The threshold depends on the whole input's score distribution, so appending
    text can flip keep/drop decisions arbitrarily far from the edit.
    """
    if not scores:
        return []
    n_keep = max(1, round(len(scores) * keep_ratio))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    kept = set(order[:n_keep])
    return [i in kept for i in range(len(scores))]


def select_absolute(scores: Sequence[float], threshold: float) -> list[bool]:
    """Append-stable selection: keep any unit whose score clears a fixed threshold.

    Each decision is local to the unit, so previously emitted decisions can never
    change when the input grows. This is the ~5-line fix the lit review identified.
    """
    return [s >= threshold for s in scores]


# ---------------------------------------------------------------------------
# Heuristic family.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadTruncateAbsolute:
    """H3a: keep the first `budget_words` words. Absolute budgets are append-only."""

    budget_words: int
    name: str = "head-truncate-absolute"

    def __call__(self, turns: Turns) -> str:
        words = split_words(join_turns(turns))
        return "".join(words[: self.budget_words])


@dataclass(frozen=True)
class HeadTruncateRatio:
    """H3b: keep the first `keep_ratio` fraction of words.

    Round 0 correction to the lit review's truncation table: head-keep with a ratio
    budget IS append-only, because round(ratio * n) is nondecreasing in n, so a longer
    input only ever extends the kept prefix. The "ratio budgets are never append-only"
    rule is real but applies to percentile selection, not head-keep truncation.
    """

    keep_ratio: float
    name: str = "head-truncate-ratio"

    def __call__(self, turns: Turns) -> str:
        words = split_words(join_turns(turns))
        return "".join(words[: max(1, round(len(words) * self.keep_ratio))])


@dataclass(frozen=True)
class TailRecencyWindow:
    """H3c: keep the last `budget_words` words (recency window). Worst case for caches."""

    budget_words: int
    name: str = "tail-recency-window"

    def __call__(self, turns: Turns) -> str:
        words = split_words(join_turns(turns))
        return "".join(words[-self.budget_words :])


@dataclass(frozen=True)
class RandomRemoval:
    """Control: remove `remove_ratio` of words, rng seeded from the input text.

    Seeding from the text makes same-input-twice deterministic while remaining a fair
    matched-ratio control; append stability is expected to fail (the sample is drawn
    over the whole input).
    """

    remove_ratio: float
    name: str = "random-removal"

    def __call__(self, turns: Turns) -> str:
        words = split_words(join_turns(turns))
        rng = random.Random(_stable_seed(join_turns(turns)))
        n_remove = round(len(words) * self.remove_ratio)
        drop = set(rng.sample(range(len(words)), min(n_remove, len(words))))
        return "".join(w for i, w in enumerate(words) if i not in drop)


def _shingles(line: str, k: int = 4) -> set[str]:
    normalized = " ".join(line.split()).lower()
    if len(normalized) <= k:
        return {normalized} if normalized else set()
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


@dataclass(frozen=True)
class DedupKeepFirst:
    """H2: drop lines that exactly repeat, or near-repeat, an earlier kept line.

    Keep-first with a fixed Jaccard threshold is natively append-only: decisions scan
    in order and depend only on earlier kept lines, so appending text can only ever
    affect the new text. Blank lines pass through (they are structure, not content).
    """

    jaccard: float = 0.9
    name: str = "dedup-keep-first"

    def __call__(self, turns: Turns) -> str:
        kept: list[str] = []
        seen_exact: set[str] = set()
        seen_shingles: list[set[str]] = []
        for unit in split_units(join_turns(turns)):
            body = unit.strip()
            if not body:
                kept.append(unit)
                continue
            if body in seen_exact:
                continue
            sh = _shingles(body)
            near = False
            for prior in seen_shingles:
                inter = len(sh & prior)
                union = len(sh | prior)
                if union and inter / union >= self.jaccard:
                    near = True
                    break
            if near:
                continue
            kept.append(unit)
            seen_exact.add(body)
            seen_shingles.append(sh)
        return "".join(kept)


# ---------------------------------------------------------------------------
# Symbolic family.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerTurnTruncateAtAppend:
    """S1: truncate each turn to a word budget once, at append time, never after.

    The strictly append-only variant of observation masking that the lit review found
    unevaluated by anyone: a turn's fate is a pure function of that turn alone, so the
    emitted prefix is byte-stable forever. Turn 0 (the task) is kept whole.
    """

    per_turn_budget_words: int
    marker: str = " [truncated-at-append]"
    name: str = "per-turn-truncate-at-append"

    def __call__(self, turns: Turns) -> str:
        out: list[str] = []
        for i, turn in enumerate(turns):
            words = split_words(turn)
            if i == 0 or len(words) <= self.per_turn_budget_words:
                out.append(turn)
            else:
                out.append("".join(words[: self.per_turn_budget_words]).rstrip() + self.marker)
        return join_turns(out)


@dataclass(frozen=True)
class RollingObservationMask:
    """S1-control: mask turns older than the last `window`, recomputed every call.

    The Complexity Trap's measured variant. Expected to fail the append bar: the mask
    boundary moves with total length, so one turn flips from visible to masked at
    every append.
    """

    window: int
    placeholder: str = "[old output omitted]"
    name: str = "rolling-observation-mask"

    def __call__(self, turns: Turns) -> str:
        out = [
            turn if i == 0 or i > len(turns) - 1 - self.window else self.placeholder
            for i, turn in enumerate(turns)
        ]
        return join_turns(out)


def _minify_json_line(line: str) -> str:
    stripped = line.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return line
    try:
        return json.dumps(json.loads(stripped), separators=(",", ":"), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return line


@dataclass(frozen=True)
class JsonMinify:
    """S2: re-serialize JSON-shaped lines compactly. A pure per-line data function.

    Perfect prefix stability by construction; measures how much of our tool-call
    traffic is recoverable whitespace/formatting rather than content.
    """

    name: str = "json-minify"

    def __call__(self, turns: Turns) -> str:
        out: list[str] = []
        for turn in turns:
            lines = turn.split("\n")
            out.append("\n".join(_minify_json_line(ln) for ln in lines))
        return join_turns(out)


# ---------------------------------------------------------------------------
# Scored filters (heuristic H1 and learned L1 share this shape).
# ---------------------------------------------------------------------------

UnitScoreFn = Callable[[str], float]
"""Score one unit in isolation; locality is what makes the absolute rule append-stable."""

TokenScoreFn = Callable[[str], list[tuple[str, float]]]
"""Score one turn's words in isolation: returns (word-with-whitespace, score) pairs."""


@dataclass(frozen=True)
class ScoredUnitFilter:
    """Keep line-units whose score passes the selection rule (H1 shape).

    `mode="percentile"` reproduces the stock per-input rule (threshold over the whole
    transcript's unit scores); `mode="absolute"` applies a fixed threshold per unit.
    The scorer itself is per-unit and context-free in both modes, so any measured
    churn in percentile mode is attributable to the selection rule alone and is a
    lower bound on the stock implementations (whose chunking adds more churn).
    """

    score_fn: UnitScoreFn
    mode: str
    keep_ratio: float = 0.5
    threshold: float = 0.0
    name: str = "scored-unit-filter"

    def __call__(self, turns: Turns) -> str:
        units = split_units(join_turns(turns))
        scores = [self.score_fn(u) for u in units]
        if self.mode == "percentile":
            keep = select_percentile(scores, self.keep_ratio)
        elif self.mode == "absolute":
            keep = select_absolute(scores, self.threshold)
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return "".join(u for u, k in zip(units, keep, strict=True) if k)


@dataclass(frozen=True)
class ScoredTokenFilter:
    """Keep words whose score passes the selection rule (L1 shape, LLMLingua-2 style).

    The token scorer runs per turn (locality by construction; the stock global
    512-token chunking would only add churn on top). Percentile mode thresholds over
    all turns' word scores jointly, which is the stock behavior under test.
    """

    token_score_fn: TokenScoreFn
    mode: str
    keep_ratio: float = 0.5
    threshold: float = 0.5
    name: str = "scored-token-filter"

    def __call__(self, turns: Turns) -> str:
        scored_turns = [self.token_score_fn(turn) for turn in turns]
        if self.mode == "percentile":
            flat = [s for turn in scored_turns for _, s in turn]
            keep_flags = select_percentile(flat, self.keep_ratio)
            out: list[str] = []
            idx = 0
            for turn in scored_turns:
                kept_words: list[str] = []
                for word, _ in turn:
                    if keep_flags[idx]:
                        kept_words.append(word)
                    idx += 1
                out.append("".join(kept_words))
            return join_turns(out)
        if self.mode == "absolute":
            return join_turns(
                ["".join(w for w, s in turn if s >= self.threshold) for turn in scored_turns]
            )
        raise ValueError(f"unknown mode: {self.mode}")


# ---------------------------------------------------------------------------
# Audit metrics.
# ---------------------------------------------------------------------------


def common_prefix_len(a: str, b: str) -> int:
    """Length of the shared leading substring of two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def is_deterministic(compress: Compressor, turns: Turns, repeats: int = 3) -> bool:
    """Same input `repeats` times, byte-identical output every time."""
    first = compress(turns)
    return all(compress(turns) == first for _ in range(repeats - 1))


def append_churn(compress: Compressor, turns: Turns) -> list[float]:
    """Per-append-point churn: how much already-emitted prefix changed.

    For each k, compares compress(turns[:k]) against compress(turns[:k+1]); churn is
    the fraction of the previously emitted output that is no longer a byte-for-byte
    prefix of the new output. 0.0 everywhere = append-only.
    """
    churns: list[float] = []
    prev = compress(turns[:1])
    for k in range(2, len(turns) + 1):
        cur = compress(turns[:k])
        if prev:
            churns.append(1.0 - common_prefix_len(prev, cur) / len(prev))
        prev = cur
    return churns


@dataclass
class MethodAudit:
    """Aggregated Round 0 verdicts for one method on one corpus."""

    method: str
    corpus: str
    deterministic: bool = True
    churns: list[float] = field(default_factory=list)
    raw_chars: int = 0
    compressed_chars: int = 0
    latencies_s: list[float] = field(default_factory=list)

    @property
    def churn_mean(self) -> float:
        return statistics.fmean(self.churns) if self.churns else math.nan

    @property
    def churn_max(self) -> float:
        return max(self.churns) if self.churns else math.nan

    @property
    def frac_append_stable(self) -> float:
        """Fraction of append events with exactly zero churn."""
        if not self.churns:
            return math.nan
        return sum(1 for c in self.churns if c == 0.0) / len(self.churns)

    @property
    def char_ratio(self) -> float:
        """Compressed/raw character ratio (token ratio is computed in the runner)."""
        return self.compressed_chars / self.raw_chars if self.raw_chars else math.nan

    @property
    def append_only(self) -> bool:
        return bool(self.churns) and self.churn_max == 0.0
