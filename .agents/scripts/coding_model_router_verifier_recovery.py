"""Ingest one official verifier-only replay without introducing a new model sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from wmo.core.files import write_text_atomic
from wmo.optimize.outcomes import OutcomeMatrix

logger = logging.getLogger("coding-model-router-verifier-recovery")

# The one bounded replay was launched from this prototype before its typed, lint-clean equivalent
# was committed. Both sources are preserved and hashed in the recovery manifest.
LEGACY_REPLAY_AGENT_SHA256 = "1b4195f3698382389769f572348405e646fc5e8e41b7b0d61f252a5fda9f470e"


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _nested_object(value: object, *keys: str) -> dict[str, object]:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            raise ValueError(f"missing object at {'.'.join(keys)}")
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            raise ValueError(f"missing object at {'.'.join(keys)}")
        current = next_value
    if not isinstance(current, dict):
        raise ValueError(f"missing object at {'.'.join(keys)}")
    return {str(key): item for key, item in current.items()}


def _official_reward(result: dict[str, object]) -> float:
    rewards = _nested_object(result, "verifier_result", "rewards")
    reward = rewards.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise ValueError("verifier replay has no numeric official reward")
    return float(reward)


def _validate_replay(
    *,
    original_artifact: Path,
    replay_trial: Path,
    committed_agent: Path,
    executed_agent: Path,
) -> tuple[float, dict[str, object]]:
    original_result = _read_object(original_artifact / "result.json")
    replay_result = _read_object(replay_trial / "result.json")
    original_exception = _nested_object(original_result, "exception_info")
    if original_exception.get("exception_type") != "VerifierTimeoutError":
        raise ValueError("paid attempt did not fail with VerifierTimeoutError")
    if original_result.get("verifier_result") is not None:
        raise ValueError("paid attempt already carries verifier evidence")
    if replay_result.get("exception_info") is not None:
        raise ValueError("verifier replay carries an exception")
    if original_result.get("task_checksum") != replay_result.get("task_checksum"):
        raise ValueError("verifier replay used a different task checksum")
    if original_result.get("task_name") != replay_result.get("task_name"):
        raise ValueError("verifier replay used a different task")
    agent_info = _nested_object(replay_result, "agent_info")
    if agent_info.get("name") != "wmo-trace-replay":
        raise ValueError("recovery trial did not use the deterministic replay agent")
    agent_result = _nested_object(replay_result, "agent_result")
    for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
        if agent_result.get(field) not in (None, 0, 0.0):
            raise ValueError(f"verifier replay unexpectedly recorded paid model field {field}")
    committed_agent_sha256 = _sha256(committed_agent)
    executed_agent_sha256 = _sha256(executed_agent)
    if executed_agent_sha256 not in {
        committed_agent_sha256,
        LEGACY_REPLAY_AGENT_SHA256,
    }:
        raise ValueError("executed replay agent is not an approved preserved implementation")
    trace_path = original_artifact / "agent" / "wmo-run.json"
    trace = _read_object(trace_path)
    if trace.get("stop_reason") != "submitted":
        raise ValueError("paid trajectory did not submit a candidate for verification")
    reward = _official_reward(replay_result)
    return reward, {
        "original_artifact_dir": str(original_artifact.resolve()),
        "original_artifact_sha256": _artifact_digest(original_artifact),
        "original_result_sha256": _sha256(original_artifact / "result.json"),
        "original_trace_sha256": _sha256(trace_path),
        "replay_trial_dir": str(replay_trial.resolve()),
        "replay_trial_sha256": _artifact_digest(replay_trial),
        "replay_result_sha256": _sha256(replay_trial / "result.json"),
        "replay_reward_sha256": _sha256(replay_trial / "verifier" / "reward.txt"),
        "committed_replay_agent": str(committed_agent.resolve()),
        "replay_agent_sha256": committed_agent_sha256,
        "executed_replay_agent": str(executed_agent.resolve()),
        "executed_replay_agent_sha256": executed_agent_sha256,
        "executed_agent_matches_committed": executed_agent_sha256 == committed_agent_sha256,
        "zero_model_calls": True,
        "official_reward": reward,
    }


def _ingest(
    *,
    root: Path,
    scenario_id: str,
    model: str,
    attempt_number: int,
    replay_trial: Path,
    committed_agent: Path,
    executed_agent: Path,
) -> Path:
    matrix_path = root / "full" / "outcomes.json"
    matrix = OutcomeMatrix.load(matrix_path)
    matches = [
        outcome
        for outcome in matrix.outcomes
        if outcome.scenario_id == scenario_id
        and outcome.model == model
        and outcome.attempt_number == attempt_number
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one paid outcome, found {len(matches)}")
    outcome = matches[0]
    if outcome.reward is not None:
        raise ValueError("paid outcome is already gradeable")
    original_artifact = Path(outcome.artifact_dir)
    reward, provenance = _validate_replay(
        original_artifact=original_artifact,
        replay_trial=replay_trial,
        committed_agent=committed_agent,
        executed_agent=executed_agent,
    )
    benchmark, task_id = scenario_id.split(":", 1)
    canonical = (
        root / "full" / "recovered" / benchmark / model / task_id / f"attempt-{attempt_number}"
    )
    if canonical.exists():
        raise FileExistsError(f"recovered artifact already exists at {canonical}")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original_artifact, canonical)
    shutil.copy2(canonical / "result.json", canonical / "paid-result.json")
    shutil.copytree(replay_trial, canonical / "verifier-replay")
    shutil.copy2(replay_trial / "result.json", canonical / "result.json")
    shutil.copytree(
        replay_trial / "verifier",
        canonical / "verifier",
        dirs_exist_ok=True,
    )
    replay_source = canonical / "replay-source"
    replay_source.mkdir()
    shutil.copy2(committed_agent, replay_source / "committed.py")
    shutil.copy2(executed_agent, replay_source / "executed.py")
    manifest = {
        "protocol": "coding-router-verifier-recovery-v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "scenario_id": scenario_id,
        "model": model,
        "attempt_number": attempt_number,
        "method": "deterministic_tool_trace_replay_then_official_verifier",
        "preserved_committed_replay_agent": str((replay_source / "committed.py").resolve()),
        "preserved_executed_replay_agent": str((replay_source / "executed.py").resolve()),
        **provenance,
    }
    write_text_atomic(
        canonical / "verifier-recovery.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    outcome.reward = reward
    outcome.success = reward == 1.0
    outcome.completion_status = (
        "scored_pass_verifier_replay" if outcome.success else "scored_failure_verifier_replay"
    )
    outcome.failure_class = "" if outcome.success else "task_failure"
    outcome.error = None
    outcome.critique = (
        f"{outcome.critique}; " if outcome.critique else ""
    ) + "official verifier recovered by deterministic zero-model-call tool-trace replay"
    outcome.artifact_dir = str(canonical.resolve())
    matrix.save(matrix_path)

    ledger_path = root / "spend-ledger.jsonl"
    ledger = [
        cast("dict[str, object]", json.loads(line))
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    ledger_matches = [
        row
        for row in ledger
        if row.get("scenario_id") == scenario_id
        and row.get("model") == model
        and row.get("attempt_number") == attempt_number
    ]
    if len(ledger_matches) != 1:
        raise ValueError(f"expected one ledger event, found {len(ledger_matches)}")
    ledger_matches[0].update(
        {
            "completion_status": outcome.completion_status,
            "failure_class": outcome.failure_class,
            "artifact_dir": outcome.artifact_dir,
            "verifier_recovery": manifest,
        }
    )
    write_text_atomic(
        ledger_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger),
    )
    logger.info(
        "ingested official verifier replay for %s x %s attempt %d reward=%s",
        scenario_id,
        model,
        attempt_number,
        reward,
    )
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attempt-number", type=int, required=True)
    parser.add_argument("--replay-trial", type=Path, required=True)
    parser.add_argument("--committed-agent", type=Path, required=True)
    parser.add_argument("--executed-agent", type=Path, required=True)
    args = parser.parse_args()
    _ingest(
        root=args.root.resolve(),
        scenario_id=args.scenario_id,
        model=args.model,
        attempt_number=args.attempt_number,
        replay_trial=args.replay_trial.resolve(),
        committed_agent=args.committed_agent.resolve(),
        executed_agent=args.executed_agent.resolve(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
