"""Prepare a compact paired-outcome source from Open-SWE-Traces.

This script is designed for remote compute. It scans only outcome and identity
columns from the large trajectory parquet files, joins task text from
SWE-rebench-V2, and writes a compact external-only paired-arm dataset. It never
reads target benchmark data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import duckdb  # ty: ignore[unresolved-import]

logger = logging.getLogger("coding-router-open-swe-prepare")

OPEN_SWE_DATASET = "nvidia/Open-SWE-Traces"
REBENCH_DATASET = "nebius/SWE-rebench-V2"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _dataset_parquet(dataset: str) -> list[dict[str, object]]:
    encoded = urllib.parse.quote(dataset, safe="")
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/parquet?dataset={encoded}",
        headers={"User-Agent": "world-model-optimizer-open-swe/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("parquet_files"), list):
        raise ValueError(f"dataset server returned no parquet files for {dataset}")
    files = [
        {str(key): value for key, value in item.items()}
        for item in payload["parquet_files"]
        if isinstance(item, dict)
    ]
    if not files:
        raise ValueError(f"dataset server returned an empty parquet manifest for {dataset}")
    return files


def _urls(rows: Iterable[dict[str, object]]) -> list[str]:
    result = [str(row["url"]) for row in rows]
    if not result or any(not url.startswith("https://") for url in result):
        raise ValueError("parquet manifest contains an invalid URL")
    return result


def _load_outcomes(
    connection: duckdb.DuckDBPyConnection,
    parquet_files: list[dict[str, object]],
) -> list[dict[str, object]]:
    connection.execute(
        """
        CREATE TABLE outcomes (
            instance_id VARCHAR,
            repo VARCHAR,
            language VARCHAR,
            scaffold VARCHAR,
            model_mode VARCHAR,
            reward DOUBLE,
            attempts BIGINT
        )
        """
    )
    partitions = sorted(
        {
            (str(row["config"]), str(row["split"]))
            for row in parquet_files
            if row.get("config") is not None and row.get("split") is not None
        }
    )
    for scaffold, model_mode in partitions:
        urls = _urls(
            row
            for row in parquet_files
            if str(row.get("config")) == scaffold and str(row.get("split")) == model_mode
        )
        logger.info(
            "scanning outcome columns scaffold=%s model_mode=%s files=%d",
            scaffold,
            model_mode,
            len(urls),
        )
        connection.execute(
            """
            INSERT INTO outcomes
            SELECT
                instance_id,
                any_value(repo),
                any_value(language),
                ?,
                ?,
                avg(resolved::DOUBLE),
                count(*)
            FROM read_parquet(?, union_by_name = true)
            WHERE resolved IN (0, 1)
            GROUP BY instance_id
            """,
            [scaffold, model_mode, urls],
        )
    rows = connection.execute(
        """
        SELECT
            scaffold,
            model_mode,
            count(*) AS tasks,
            sum(attempts) AS trajectories,
            avg(reward) AS mean_task_reward
        FROM outcomes
        GROUP BY scaffold, model_mode
        ORDER BY scaffold, model_mode
        """
    ).fetchall()
    return [
        {
            "scaffold": str(row[0]),
            "model_mode": str(row[1]),
            "tasks": int(row[2]),
            "trajectories": int(row[3]),
            "mean_task_reward": float(row[4]),
        }
        for row in rows
    ]


def _select_pair(arm_stats: list[dict[str, object]]) -> dict[str, object]:
    by_scaffold: dict[str, list[dict[str, object]]] = {}
    for row in arm_stats:
        by_scaffold.setdefault(str(row["scaffold"]), []).append(row)
    candidates: list[dict[str, object]] = []
    for scaffold, rows in by_scaffold.items():
        if len(rows) != 2:
            continue
        ordered = sorted(rows, key=lambda row: float(cast(Any, row["mean_task_reward"])))
        weak, strong = ordered
        candidates.append(
            {
                "scaffold": scaffold,
                "weak_model_mode": str(weak["model_mode"]),
                "strong_model_mode": str(strong["model_mode"]),
                "weak_mean_reward": float(cast(Any, weak["mean_task_reward"])),
                "strong_mean_reward": float(cast(Any, strong["mean_task_reward"])),
                "external_arm_gap": float(cast(Any, strong["mean_task_reward"]))
                - float(cast(Any, weak["mean_task_reward"])),
            }
        )
    if not candidates:
        raise ValueError("Open-SWE manifest has no two-arm scaffold")
    return max(
        candidates,
        key=lambda row: (
            float(cast(Any, row["external_arm_gap"])),
            str(row["scaffold"]),
        ),
    )


def _load_task_text(
    connection: duckdb.DuckDBPyConnection,
    parquet_files: list[dict[str, object]],
) -> None:
    urls = _urls(parquet_files)
    logger.info("scanning SWE-rebench task text files=%d", len(urls))
    connection.execute(
        """
        CREATE TABLE tasks AS
        SELECT
            instance_id,
            any_value(repo) AS repo,
            any_value(language) AS language,
            any_value(problem_statement) AS problem_statement,
            any_value(meta.llm_metadata.difficulty) AS difficulty,
            any_value(meta.llm_metadata.intent_completeness) AS intent_completeness,
            any_value(meta.llm_metadata.pr_categories) AS pr_categories
        FROM read_parquet(?, union_by_name = true)
        GROUP BY instance_id
        """,
        [urls],
    )


def _write_compact_source(
    connection: duckdb.DuckDBPyConnection,
    pair: dict[str, object],
    output: Path,
) -> dict[str, object]:
    rows = connection.execute(
        """
        SELECT
            tasks.instance_id,
            tasks.repo,
            tasks.language,
            tasks.problem_statement,
            weak.reward,
            weak.attempts,
            strong.reward,
            strong.attempts
        FROM tasks
        JOIN outcomes AS weak
          ON weak.instance_id = tasks.instance_id
         AND weak.scaffold = ?
         AND weak.model_mode = ?
        JOIN outcomes AS strong
          ON strong.instance_id = tasks.instance_id
         AND strong.scaffold = ?
         AND strong.model_mode = ?
        WHERE length(trim(tasks.problem_statement)) > 0
        ORDER BY tasks.instance_id
        """,
        [
            pair["scaffold"],
            pair["weak_model_mode"],
            pair["scaffold"],
            pair["strong_model_mode"],
        ],
    ).fetchall()
    if not rows:
        raise ValueError("selected Open-SWE pair has no task-text overlap")
    compact = [
        {
            "instance_id": str(row[0]),
            "repo": str(row[1]),
            "language": str(row[2]),
            "text": str(row[3]),
            "cheap_reward": float(row[4]),
            "cheap_attempts": int(row[5]),
            "strong_reward": float(row[6]),
            "strong_attempts": int(row[7]),
        }
        for row in rows
    ]
    _write_json(output, compact)
    return {
        "paired_tasks": len(compact),
        "repositories": len({str(row[1]) for row in rows}),
        "languages": len({str(row[2]) for row in rows}),
        "weak_reward": sum(float(row[4]) for row in rows) / len(rows),
        "strong_reward": sum(float(row[6]) for row in rows) / len(rows),
        "weak_attempts": sum(int(row[5]) for row in rows),
        "strong_attempts": sum(int(row[7]) for row in rows),
    }


def _write_profile_teacher_source(
    connection: duckdb.DuckDBPyConnection,
    pair: dict[str, object],
    output: Path,
) -> dict[str, object]:
    """Write task-profile supervision disjoint from the paired outcome source."""
    rows = connection.execute(
        """
        SELECT
            tasks.instance_id,
            tasks.repo,
            tasks.language,
            tasks.problem_statement,
            tasks.difficulty,
            tasks.intent_completeness,
            tasks.pr_categories
        FROM tasks
        WHERE length(trim(tasks.problem_statement)) > 0
          AND tasks.difficulty IS NOT NULL
          AND tasks.intent_completeness IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM outcomes AS weak
              JOIN outcomes AS strong
                ON strong.instance_id = weak.instance_id
               AND strong.scaffold = weak.scaffold
              WHERE weak.instance_id = tasks.instance_id
                AND weak.scaffold = ?
                AND weak.model_mode = ?
                AND strong.model_mode = ?
          )
        ORDER BY tasks.instance_id
        """,
        [
            pair["scaffold"],
            pair["weak_model_mode"],
            pair["strong_model_mode"],
        ],
    ).fetchall()
    if not rows:
        raise ValueError("SWE-rebench has no disjoint task-profile teacher rows")
    teacher = [
        {
            "instance_id": str(row[0]),
            "repo": str(row[1]),
            "language": str(row[2]),
            "text": str(row[3]),
            "difficulty": str(row[4]),
            "intent_completeness": str(row[5]),
            "pr_categories": [str(value) for value in (row[6] or [])],
        }
        for row in rows
    ]
    _write_json(output, teacher)
    return {
        "tasks": len(teacher),
        "repositories": len({str(row[1]) for row in rows}),
        "languages": len({str(row[2]) for row in rows}),
        "difficulties": sorted({str(row[4]) for row in rows}),
        "intent_completeness": sorted({str(row[5]) for row in rows}),
        "pr_categories": sorted(
            {str(category) for row in rows for category in cast(list[object] | None, row[6]) or []}
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(args.database.resolve()))
    connection.execute("SET threads = 8")
    connection.execute("SET memory_limit = '6GB'")
    open_swe_files = _dataset_parquet(OPEN_SWE_DATASET)
    rebench_files = _dataset_parquet(REBENCH_DATASET)
    arm_stats = _load_outcomes(connection, open_swe_files)
    pair = _select_pair(arm_stats)
    _load_task_text(connection, rebench_files)
    source_stats = _write_compact_source(connection, pair, output)
    teacher_stats = _write_profile_teacher_source(
        connection,
        pair,
        args.profile_teacher_output.resolve(),
    )
    _write_json(
        args.manifest.resolve(),
        {
            "schema": "open-swe-paired-source-v2",
            "source_dataset": OPEN_SWE_DATASET,
            "task_dataset": REBENCH_DATASET,
            "target_data_read": False,
            "arm_selection": "largest mean reward gap within one external scaffold",
            "arm_stats": arm_stats,
            "selected_pair": pair,
            "source_stats": source_stats,
            "profile_teacher_stats": teacher_stats,
            "profile_teacher_outcome_overlap": 0,
            "open_swe_parquet_files": len(open_swe_files),
            "rebench_parquet_files": len(rebench_files),
        },
    )
    logger.info(
        "prepared Open-SWE source tasks=%d scaffold=%s weak=%s strong=%s",
        source_stats["paired_tasks"],
        pair["scaffold"],
        pair["weak_model_mode"],
        pair["strong_model_mode"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-teacher-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
