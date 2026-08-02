"""Build the paid, label-free OpenAI task-embedding cache under the frozen spend ceiling."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
from coding_model_router_analyze import (
    EXPERIMENT_ID,
    OPENAI_EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL,
    _canonical_matrix,
    _read_object,
    _scenario_tasks,
    _write_json,
)
from openai import OpenAI

from wmo.core.files import write_text_atomic

EMBEDDING_INPUT_PER_MTOK = 0.13
RESERVATION_USD = 1.0
BATCH_SIZE = 64
EVENT_ID = f"embedding:{OPENAI_EMBEDDING_MODEL}:{OPENAI_EMBEDDING_DIM}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ledger_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    return float(value)


def _spent_reserved(rows: list[dict[str, object]]) -> tuple[float, float]:
    spent = 0.0
    reserved = 0.0
    for row in rows:
        if row.get("status") == "reserved":
            reserved += _number(row.get("reserved_usd"))
        elif row.get("status") == "completed" or row.get("status") is None:
            spent += _number(row.get("model_cost_usd"))
    return spent, reserved


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    write_text_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _upsert(rows: list[dict[str, object]], row: dict[str, object]) -> list[dict[str, object]]:
    return [existing for existing in rows if existing.get("event_id") != EVENT_ID] + [row]


def _reserve(root: Path) -> tuple[Path, list[dict[str, object]], float]:
    summary = _read_object(root / "freeze-summary.json")
    ceiling = summary.get("spend_ceiling_usd")
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool) or ceiling <= 0:
        raise ValueError("freeze-summary.json has no authorized positive spend ceiling")
    ledger_path = root / "spend-ledger.jsonl"
    rows = _ledger_rows(ledger_path)
    existing = next((row for row in rows if row.get("event_id") == EVENT_ID), None)
    if existing is not None and existing.get("status") == "completed":
        raise ValueError("embedding ledger event is already complete")
    if existing is None:
        spent, reserved = _spent_reserved(rows)
        if spent + reserved + RESERVATION_USD > float(ceiling):
            raise ValueError(
                f"${RESERVATION_USD:.2f} embedding reservation would exceed the frozen "
                f"${float(ceiling):.2f} ceiling"
            )
        rows = _upsert(
            rows,
            {
                "event_id": EVENT_ID,
                "recorded_at": _utc_now(),
                "phase": "router_embedding",
                "provider": "openai",
                "model": OPENAI_EMBEDDING_MODEL,
                "status": "reserved",
                "reserved_usd": RESERVATION_USD,
            },
        )
        _write_ledger(ledger_path, rows)
    return ledger_path, rows, float(ceiling)


def _save_partial(path: Path, ids: list[str], vectors: np.ndarray, prompt_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        np.savez_compressed(
            temp_path,
            scenario_ids=np.asarray(ids),
            vectors=vectors,
            prompt_tokens=np.asarray([prompt_tokens], dtype=np.int64),
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_partial(path: Path, expected_ids: list[str]) -> tuple[np.ndarray, int, int]:
    if not path.is_file():
        return np.empty((0, OPENAI_EMBEDDING_DIM), dtype=np.float32), 0, 0
    with np.load(path) as data:
        ids = [str(value) for value in data["scenario_ids"].tolist()]
        if ids != expected_ids[: len(ids)]:
            raise ValueError("partial embedding cache belongs to a different scenario order")
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        prompt_tokens = int(data["prompt_tokens"][0])
    if vectors.shape != (len(ids), OPENAI_EMBEDDING_DIM):
        raise ValueError(f"partial embedding cache has invalid shape {vectors.shape}")
    return vectors, len(ids), prompt_tokens


def _build(root: Path) -> None:
    final_path = root / "embeddings" / "text-embedding-3-large-3072.npy"
    if final_path.exists():
        raise ValueError(f"{final_path} already exists; embedding cache is immutable")
    matrix, _ = _canonical_matrix(root)
    scenario_ids, tasks = _scenario_tasks(matrix)
    ledger_path, ledger_rows, ceiling = _reserve(root)
    partial_path = final_path.with_suffix(".partial.npz")
    vectors, completed, prompt_tokens = _load_partial(partial_path, scenario_ids)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    while completed < len(scenario_ids):
        batch_ids = scenario_ids[completed : completed + BATCH_SIZE]
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            dimensions=OPENAI_EMBEDDING_DIM,
            input=[tasks[scenario_id] for scenario_id in batch_ids],
        )
        batch = np.asarray([row.embedding for row in response.data], dtype=np.float32)
        if batch.shape != (len(batch_ids), OPENAI_EMBEDDING_DIM):
            raise ValueError(f"embedding response has invalid shape {batch.shape}")
        vectors = np.concatenate((vectors, batch), axis=0)
        completed += len(batch_ids)
        prompt_tokens += int(response.usage.prompt_tokens)
        _save_partial(
            partial_path,
            scenario_ids[:completed],
            vectors,
            prompt_tokens,
        )

    cost = prompt_tokens / 1_000_000 * EMBEDDING_INPUT_PER_MTOK
    spent, reserved = _spent_reserved(
        [row for row in ledger_rows if row.get("event_id") != EVENT_ID]
    )
    if spent + reserved + cost > ceiling:
        raise ValueError("realized embedding cost exceeds the frozen spend ceiling")
    with tempfile.NamedTemporaryFile(dir=final_path.parent, suffix=".npy", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        np.save(temp_path, vectors)
        temp_path.replace(final_path)
    finally:
        temp_path.unlink(missing_ok=True)
    _write_json(
        final_path.with_suffix(".json"),
        {
            "model": OPENAI_EMBEDDING_MODEL,
            "dimensions": OPENAI_EMBEDDING_DIM,
            "scenarios": len(scenario_ids),
            "prompt_tokens": prompt_tokens,
            "input_per_mtok_usd": EMBEDDING_INPUT_PER_MTOK,
            "cost_usd": cost,
        },
    )
    rows = _upsert(
        _ledger_rows(ledger_path),
        {
            "event_id": EVENT_ID,
            "recorded_at": _utc_now(),
            "phase": "router_embedding",
            "provider": "openai",
            "model": OPENAI_EMBEDDING_MODEL,
            "status": "completed",
            "reserved_usd": RESERVATION_USD,
            "prompt_tokens": prompt_tokens,
            "model_cost_usd": cost,
        },
    )
    _write_ledger(ledger_path, rows)
    partial_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _build(cast("Path", args.root).resolve())


if __name__ == "__main__":
    main()
