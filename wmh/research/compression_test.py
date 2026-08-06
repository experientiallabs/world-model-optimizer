"""Tests for the compression method slate and append-stability audit primitives."""

from __future__ import annotations

import json

from wmh.research.compression import (
    DedupKeepFirst,
    HeadTruncateAbsolute,
    HeadTruncateRatio,
    JsonMinify,
    MethodAudit,
    PerTurnTruncateAtAppend,
    RandomRemoval,
    RollingObservationMask,
    ScoredTokenFilter,
    ScoredUnitFilter,
    TailRecencyWindow,
    append_churn,
    common_prefix_len,
    is_deterministic,
    join_turns,
    select_absolute,
    select_percentile,
    split_units,
    split_words,
)

TURNS = [
    "Task: exchange the camera for the max zoom variant.",
    '{"tool": "find_user", "arguments": {"name": "Sofia Li", "zip": "78260"}}',
    "The user has three orders. The delivered order contains a digital camera.",
    '{"tool": "get_product", "arguments": {"product_id": "cam-1"}}',
    "Variant 7255224608 has 10x zoom, 30MP, CF card. Same specs except zoom.",
    '{"tool": "exchange", "arguments": {"item": "cam-1", "to": "7255224608"}}',
]


def _long_turns(n: int = 8, words_per_turn: int = 40) -> list[str]:
    return [" ".join(f"t{i}w{j}" for j in range(words_per_turn)) for i in range(n)]


def test_split_words_lossless() -> None:
    text = "a  b\tc\nd   "
    assert "".join(split_words(text)) == text


def test_split_units_lossless() -> None:
    text = "line one\nline two\n\nline four"
    assert "".join(split_units(text)) == text


def test_select_percentile_keeps_top_fraction() -> None:
    keep = select_percentile([0.1, 0.9, 0.5, 0.8], keep_ratio=0.5)
    assert keep == [False, True, False, True]


def test_select_absolute_is_local() -> None:
    scores = [0.1, 0.9, 0.5]
    keep_short = select_absolute(scores, threshold=0.5)
    keep_long = select_absolute(scores + [0.99, 0.99], threshold=0.5)
    assert keep_long[: len(scores)] == keep_short


def test_select_percentile_flips_under_append() -> None:
    """The stock rule's failure mode: appending high scores evicts old survivors."""
    scores = [0.4, 0.6]
    keep_short = select_percentile(scores, keep_ratio=0.5)
    keep_long = select_percentile(scores + [0.9, 0.95], keep_ratio=0.5)
    assert keep_short == [False, True]
    assert keep_long[: len(scores)] != keep_short


def test_head_truncate_absolute_is_append_only() -> None:
    method = HeadTruncateAbsolute(budget_words=60)
    assert max(append_churn(method, _long_turns())) == 0.0


def test_head_truncate_ratio_is_append_only_too() -> None:
    """Contradicts the lit review's table: head-keep ratio budgets extend monotonically."""
    method = HeadTruncateRatio(keep_ratio=0.5)
    assert max(append_churn(method, _long_turns())) == 0.0


def test_tail_recency_window_churns() -> None:
    method = TailRecencyWindow(budget_words=50)
    assert max(append_churn(method, _long_turns())) > 0.0


def test_random_removal_deterministic_but_not_append_only() -> None:
    method = RandomRemoval(remove_ratio=0.5)
    assert is_deterministic(method, TURNS)
    assert max(append_churn(method, _long_turns())) > 0.0


def test_random_removal_ratio_is_matched() -> None:
    method = RandomRemoval(remove_ratio=0.5)
    raw_words = len(split_words(join_turns(TURNS)))
    kept_words = len(split_words(method(TURNS)))
    assert abs(kept_words - raw_words * 0.5) <= 1


def test_dedup_keep_first_drops_exact_and_near_repeats() -> None:
    turns = [
        "task",
        "error: connection refused at port 8080",
        "error: connection refused at port 8080",
        "error: connection refused at port 8081",
        "something completely different happened",
    ]
    out = DedupKeepFirst(jaccard=0.8)(turns)
    assert out.count("connection refused") == 1
    assert "completely different" in out


def test_dedup_keep_first_is_append_only() -> None:
    turns = ["task"] + ["error: connection refused at port 8080"] * 4 + ["done ok"]
    assert max(append_churn(DedupKeepFirst(), turns)) == 0.0


def test_per_turn_truncate_at_append_is_append_only_and_keeps_task() -> None:
    method = PerTurnTruncateAtAppend(per_turn_budget_words=5)
    turns = _long_turns()
    assert max(append_churn(method, turns)) == 0.0
    assert method(turns).startswith(turns[0])
    assert "[truncated-at-append]" in method(turns)


def test_rolling_observation_mask_churns() -> None:
    method = RollingObservationMask(window=2)
    assert max(append_churn(method, _long_turns())) > 0.0


def test_json_minify_shrinks_json_and_keeps_prose() -> None:
    turns = ['{"tool":  "bash",   "arguments": {"command":   "ls"}}', "plain prose line"]
    out = JsonMinify()(turns)
    assert '{"tool":"bash","arguments":{"command":"ls"}}' in out
    assert "plain prose line" in out
    parsed = json.loads(out.split("\n\n")[0])
    assert parsed["tool"] == "bash"


def test_json_minify_is_append_only_and_idempotent() -> None:
    method = JsonMinify()
    assert max(append_churn(method, TURNS)) == 0.0
    once = method(TURNS)
    assert method([once]) == once


def test_scored_unit_filter_absolute_is_append_only() -> None:
    method = ScoredUnitFilter(score_fn=len, mode="absolute", threshold=20)
    assert max(append_churn(method, _long_turns())) == 0.0


def test_scored_unit_filter_percentile_churns() -> None:
    turns = ["short", "a much longer line of text here", "mid line", "x", "yy", "zzz"]
    method = ScoredUnitFilter(score_fn=len, mode="percentile", keep_ratio=0.4)
    assert max(append_churn(method, turns)) > 0.0


def _fake_token_scores(turn: str) -> list[tuple[str, float]]:
    return [(w, float(len(w))) for w in split_words(turn)]


def test_scored_token_filter_absolute_is_append_only() -> None:
    method = ScoredTokenFilter(token_score_fn=_fake_token_scores, mode="absolute", threshold=5.0)
    assert max(append_churn(method, _long_turns())) == 0.0


def test_scored_token_filter_percentile_churns() -> None:
    turns = ["aa bb cc", "dddddd eeeeee ffffff", "g h i", "jjjjjjjj kkkkkkkk"]
    method = ScoredTokenFilter(token_score_fn=_fake_token_scores, mode="percentile", keep_ratio=0.5)
    assert max(append_churn(method, turns)) > 0.0


def test_scored_token_filter_percentile_matches_ratio() -> None:
    method = ScoredTokenFilter(token_score_fn=_fake_token_scores, mode="percentile", keep_ratio=0.5)
    raw = len(split_words(join_turns(TURNS)))
    kept = len(split_words(method(TURNS)))
    assert abs(kept - raw * 0.5) <= 2


def test_common_prefix_len() -> None:
    assert common_prefix_len("abcd", "abxd") == 2
    assert common_prefix_len("abc", "abc") == 3
    assert common_prefix_len("", "abc") == 0


def test_append_churn_zero_for_identity() -> None:
    churns = append_churn(join_turns, TURNS)
    assert len(churns) == len(TURNS) - 1
    assert max(churns) == 0.0


def test_method_audit_aggregates() -> None:
    audit = MethodAudit(method="m", corpus="c", churns=[0.0, 0.0, 0.5])
    assert audit.frac_append_stable == 2 / 3
    assert audit.churn_max == 0.5
    assert not audit.append_only
    stable = MethodAudit(method="m", corpus="c", churns=[0.0, 0.0])
    assert stable.append_only
