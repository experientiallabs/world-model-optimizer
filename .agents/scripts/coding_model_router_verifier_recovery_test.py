"""Offline tests for official verifier-only recovery ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from coding_model_router_verifier_recovery import _ingest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import ModelPool, PoolEntry


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_ingest_combines_paid_trace_with_zero_call_official_verifier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    original = root / "full" / "infra-attempt"
    _write_json(
        original / "result.json",
        {
            "task_name": "task",
            "task_checksum": "checksum",
            "verifier_result": None,
            "exception_info": {"exception_type": "VerifierTimeoutError"},
        },
    )
    _write_json(
        original / "agent" / "wmo-run.json",
        {"stop_reason": "submitted", "steps": []},
    )
    replay = root / "full" / "verifier-replay"
    _write_json(
        replay / "result.json",
        {
            "task_name": "task",
            "task_checksum": "checksum",
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
            "agent_info": {"name": "wmo-trace-replay"},
            "agent_result": {
                "n_input_tokens": None,
                "n_cache_tokens": None,
                "n_output_tokens": None,
                "cost_usd": None,
            },
        },
    )
    (replay / "verifier").mkdir()
    (replay / "verifier" / "reward.txt").write_text("1\n", encoding="utf-8")
    pool = ModelPool(
        models=[
            PoolEntry(
                name="model",
                kind=ProviderKind.OPENAI,
                model="test-model",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
        ]
    )
    matrix = OutcomeMatrix(
        pool=pool.models,
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="task",
                model="model",
                benchmark="terminal-bench-2",
                attempt_number=3,
                reward=None,
                completion_status="infrastructure_failure",
                failure_class="infrastructure",
                artifact_dir=str(original),
                cost_usd=0.2,
            )
        ],
    )
    matrix.save(root / "full" / "outcomes.json")
    (root / "spend-ledger.jsonl").write_text(
        json.dumps(
            {
                "event_id": "event",
                "scenario_id": "terminal-bench-2:task",
                "model": "model",
                "attempt_number": 3,
                "status": "completed",
                "model_cost_usd": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    committed_agent = tmp_path / "committed-agent.py"
    executed_agent = tmp_path / "executed-agent.py"
    committed_agent.write_text("same replay implementation\n", encoding="utf-8")
    executed_agent.write_text("same replay implementation\n", encoding="utf-8")

    canonical = _ingest(
        root=root,
        scenario_id="terminal-bench-2:task",
        model="model",
        attempt_number=3,
        replay_trial=replay,
        committed_agent=committed_agent,
        executed_agent=executed_agent,
    )

    recovered = OutcomeMatrix.load(root / "full" / "outcomes.json").outcomes[0]
    assert recovered.reward == 1.0
    assert recovered.success is True
    assert recovered.completion_status == "scored_pass_verifier_replay"
    assert Path(recovered.artifact_dir) == canonical
    assert (canonical / "paid-result.json").is_file()
    assert (canonical / "verifier" / "reward.txt").read_text(encoding="utf-8") == "1\n"
    manifest = json.loads((canonical / "verifier-recovery.json").read_text(encoding="utf-8"))
    assert manifest["zero_model_calls"] is True
    ledger = json.loads((root / "spend-ledger.jsonl").read_text(encoding="utf-8"))
    assert ledger["completion_status"] == "scored_pass_verifier_replay"
    assert ledger["verifier_recovery"]["official_reward"] == 1.0
