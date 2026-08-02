"""Offline tests for the bounded WMO serving verification runner."""

from __future__ import annotations

import json
from pathlib import Path

import coding_model_router_serve_verify as runner
import httpx
import numpy as np
import pytest

from wmo.optimize.policy import EmbedderSpec, KnnBank, RoutingPolicy
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name=runner.OPENAI_ARM,
            kind=ProviderKind.OPENAI,
            model="test-openai",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        ),
        PoolEntry(
            name=runner.ANTHROPIC_ARM,
            kind=ProviderKind.ANTHROPIC,
            model="test-anthropic",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        ),
    ]


def _response(
    request: httpx.Request,
    *,
    model: str = runner.OPENAI_PROBE,
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=request,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        },
    )


def test_post_accepts_openai_compatible_shape() -> None:
    transport = httpx.MockTransport(lambda request: _response(request))
    with httpx.Client(base_url="http://test", transport=transport) as client:
        response = runner._post(
            client,
            runner.OPENAI_PROBE,
            messages=[{"role": "user", "content": "test"}],
        )
    assert response.json()["object"] == "chat.completion"


def test_post_rejects_non_compatible_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"object": "not-a-completion", "model": runner.OPENAI_PROBE},
        )

    with httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ValueError, match="non-OpenAI-compatible"):
            runner._post(
                client,
                runner.OPENAI_PROBE,
                messages=[{"role": "user", "content": "test"}],
            )


def test_validate_rows_requires_and_summarizes_eight_requests() -> None:
    def row(
        endpoint: str,
        model: str,
        *,
        reason: str = "static policy",
        gate: str = "",
        cost: float = 0.01,
        router_cost: float = 0.0,
        cache_credit: float | None = None,
    ) -> dict[str, object]:
        return {
            "endpoint": endpoint,
            "model": model,
            "status": "ok",
            "routing_reason": reason,
            "gate": gate,
            "cost_usd": cost,
            "router_cost_usd": router_cost,
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_tokens": 3,
            "cache_credit_usd": cache_credit,
        }

    rows = [
        row(runner.OPENAI_PROBE, runner.OPENAI_ARM),
        row(
            runner.OPENAI_PROBE,
            runner.OPENAI_ARM,
            reason="sticky: conversation affinity",
        ),
        row(runner.ANTHROPIC_PROBE, runner.ANTHROPIC_ARM),
        row(runner.ENDPOINT, runner.OPENAI_ARM, router_cost=0.001),
        row(
            runner.FALLBACK_PROBE,
            runner.ANTHROPIC_ARM,
            gate="novelty-abstain",
            router_cost=0.001,
        ),
        row(runner.ENDPOINT, runner.ANTHROPIC_ARM, router_cost=0.001),
        row(runner.CACHE_AWARE_PROBE, runner.OPENAI_ARM, router_cost=0.001),
        row(
            runner.CACHE_AWARE_PROBE,
            runner.OPENAI_ARM,
            router_cost=0.001,
            cache_credit=0.004,
        ),
    ]
    result = runner._validate_rows(
        rows,
        {
            "selected_route": runner.OPENAI_ARM,
            "dial_route": runner.ANTHROPIC_ARM,
            "cache_aware_first_route": runner.OPENAI_ARM,
            "cache_aware_second_route": runner.OPENAI_ARM,
        },
        runner.ANTHROPIC_ARM,
        {entry.name for entry in _pool()},
    )
    assert result["affinity_reason"] == "sticky: conversation affinity"
    assert result["fallback_gate"] == "novelty-abstain"
    assert result["cache_aware_credit_usd"] == pytest.approx(0.004)
    assert result["provider_cost_usd"] == pytest.approx(0.08)
    assert result["router_cost_usd"] == pytest.approx(0.005)
    assert result["cached_input_tokens"] == 24
    assert result["input_tokens"] == 80
    assert result["output_tokens"] == 16


def test_validate_rows_rejects_unknown_selected_route() -> None:
    rows: list[dict[str, object]] = [
        {
            "endpoint": runner.OPENAI_PROBE,
            "model": runner.OPENAI_ARM,
            "status": "ok",
            "routing_reason": "static policy",
            "gate": "",
            "cost_usd": 0.0,
            "router_cost_usd": 0.0,
            "input_tokens": 1,
            "output_tokens": 1,
            "cached_tokens": 0,
            "cache_credit_usd": None,
        }
    ] * 8
    with pytest.raises(ValueError, match="unknown routed-model"):
        runner._validate_rows(
            rows,
            {
                "selected_route": "ghost",
                "dial_route": runner.OPENAI_ARM,
                "cache_aware_first_route": runner.OPENAI_ARM,
                "cache_aware_second_route": runner.OPENAI_ARM,
            },
            runner.ANTHROPIC_ARM,
            {entry.name for entry in _pool()},
        )


def test_next_event_id_enforces_three_attempt_limit(tmp_path: Path) -> None:
    ledger = tmp_path / "spend-ledger.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(
                {
                    "event_id": f"serving-verification:{attempt}",
                    "status": "completed",
                    "completion_status": "failed",
                    "model_cost_usd": 0.0,
                }
            )
            for attempt in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    assert runner._next_event_id(tmp_path) == "serving-verification:3"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_id": "serving-verification:3",
                    "status": "completed",
                    "completion_status": "failed",
                    "model_cost_usd": 0.0,
                }
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="exhausted"):
        runner._next_event_id(tmp_path)


def test_next_event_id_refuses_rerun_after_pass(tmp_path: Path) -> None:
    (tmp_path / "spend-ledger.jsonl").write_text(
        json.dumps(
            {
                "event_id": "serving-verification:1",
                "phase": "serving_verification",
                "status": "completed",
                "completion_status": "passed",
                "model_cost_usd": 0.01,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already passed"):
        runner._next_event_id(tmp_path)


def test_copy_policy_stages_portable_knn_bank(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    bank = KnnBank(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        rewards=np.asarray([[1.0, 0.0]], dtype=np.float32),
        costs=np.asarray([[0.02, 0.01]], dtype=np.float32),
        models=[runner.OPENAI_ARM, runner.ANTHROPIC_ARM],
        scenario_ids=["scenario-1"],
    )
    bank.save(source / "bank.npz")
    policy = RoutingPolicy(
        kind="knn",
        default_model=runner.ANTHROPIC_ARM,
        guard_model=runner.ANTHROPIC_ARM,
        pool=_pool(),
        embedder=EmbedderSpec(kind="hashing", dim=2),
        knn_bank_path="bank.npz",
    )
    policy.save(source / "policy.json")

    runner._copy_policy(policy, source, target)

    staged = RoutingPolicy.load(target / "policy.json")
    assert staged.knn_bank().scenario_ids == ["scenario-1"]
    assert staged.bank_path() == target / "bank.npz"
