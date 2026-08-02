"""Materialize frozen broad SWE-smith outcomes and run the format-heldout oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import duckdb
import numpy as np

logger = logging.getLogger("coding-router-swe-smith-broad-oracle")

TRAJECTORY_DATASET = "SWE-bench/SWE-smith-trajectories"
TRAJECTORY_REVISION = "08e109b4a59eaeebf80e4675cd125d42e7ac99a4"
TASK_DATASET = "SWE-bench/SWE-smith"
TASK_REVISION = "ea6d7173829c7ec8fa16c22055699ff2e9188091"
WEAK_MODEL = "claude-3-5-sonnet-20241022"
STRONG_MODEL = "claude-3-7-sonnet-20250219"
FORMATS = ("xml", "ticks")
EXPECTED_FREEZE_SHA256 = "96427a4e3f8db70ff661dece0459da04006869247d7f8cdefd684c82f898c939"
BOOTSTRAPS = 5_000
SEED = 20_260_731


def _trajectory_urls(split: str) -> list[str]:
    """Return immutable native trajectory Parquet URLs."""
    return [
        "https://huggingface.co/datasets/"
        f"{TRAJECTORY_DATASET}/resolve/{TRAJECTORY_REVISION}/data/"
        f"{split}-{index:05d}-of-00008.parquet"
        for index in range(8)
    ]


def _task_urls() -> list[str]:
    """Return immutable native canonical-task Parquet URLs."""
    return [
        "https://huggingface.co/datasets/"
        f"{TASK_DATASET}/resolve/{TASK_REVISION}/data/"
        f"train-{index:05d}-of-00011.parquet"
        for index in range(11)
    ]


def _write_json(path: Path, value: object) -> None:
    """Atomically write deterministic JSON."""
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


def _sha256_file(path: Path) -> str:
    """Hash one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    """Hash normalized task text."""
    return hashlib.sha256(" ".join(value.split()).encode()).hexdigest()


def _load_freeze(path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Load and verify the committed label-free cohort freeze."""
    if _sha256_file(path) != EXPECTED_FREEZE_SHA256:
        raise ValueError("label-free freeze digest differs from the committed protocol")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("label-free freeze is not an object")
    selected = payload.get("selected_cohort")
    if not isinstance(selected, dict):
        raise ValueError("label-free freeze has no selected cohort")
    raw_tasks = selected.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("label-free freeze has no task inventory")
    tasks = [
        {str(key): str(value) for key, value in row.items()}
        for row in raw_tasks
        if isinstance(row, dict)
    ]
    if len(tasks) != 538 or int(selected.get("repositories", 0)) != 72:
        raise ValueError("label-free freeze dimensions differ from the committed protocol")
    return {str(key): value for key, value in selected.items()}, tasks


def _materialize(
    connection: duckdb.DuckDBPyConnection,
    tasks: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Read frozen source labels and create equal-format paired task outcomes."""
    connection.execute("DROP TABLE IF EXISTS selected_ids")
    connection.execute("CREATE TABLE selected_ids (instance_id VARCHAR PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO selected_ids VALUES (?)",
        [(task["instance_id"],) for task in tasks],
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE canonical_tasks AS
        SELECT
            source.instance_id,
            any_value(source.problem_statement) AS problem_statement
        FROM read_parquet(?, union_by_name = true) AS source
        JOIN selected_ids USING (instance_id)
        GROUP BY source.instance_id
        """,
        [_task_urls()],
    )
    expected = {task["instance_id"]: task for task in tasks}
    canonical_rows = connection.execute(
        "SELECT instance_id, problem_statement FROM canonical_tasks ORDER BY instance_id"
    ).fetchall()
    texts: dict[str, str] = {}
    for instance_id_raw, text_raw in canonical_rows:
        instance_id = str(instance_id_raw)
        text = str(text_raw).strip()
        if _sha256_text(text) != expected[instance_id]["prompt_sha256"]:
            raise ValueError(f"canonical task prompt changed: {instance_id}")
        texts[instance_id] = text
    if set(texts) != set(expected):
        raise ValueError("canonical task table is incomplete")
    connection.execute("DROP TABLE IF EXISTS outcomes")
    connection.execute(
        """
        CREATE TABLE outcomes (
            instance_id VARCHAR,
            prompt_format VARCHAR,
            model VARCHAR,
            reward DOUBLE,
            attempts BIGINT
        )
        """
    )
    for prompt_format in FORMATS:
        logger.info("opening frozen source outcomes format=%s", prompt_format)
        connection.execute(
            """
            INSERT INTO outcomes
            SELECT
                source.instance_id,
                ?,
                source.model,
                avg(source.resolved::DOUBLE),
                count(*)
            FROM read_parquet(?, union_by_name = true) AS source
            JOIN selected_ids USING (instance_id)
            WHERE source.model IN (?, ?)
              AND source.resolved IS NOT NULL
            GROUP BY source.instance_id, source.model
            """,
            [prompt_format, _trajectory_urls(prompt_format), WEAK_MODEL, STRONG_MODEL],
        )
    rows = connection.execute(
        """
        SELECT instance_id, prompt_format, model, reward, attempts
        FROM outcomes
        ORDER BY instance_id, prompt_format, model
        """
    ).fetchall()
    by_cell = {
        (str(instance_id), str(prompt_format), str(model)): (float(reward), int(attempts))
        for instance_id, prompt_format, model, reward, attempts in rows
    }
    expected_cells = {
        (task["instance_id"], prompt_format, model)
        for task in tasks
        for prompt_format in FORMATS
        for model in (WEAK_MODEL, STRONG_MODEL)
    }
    if set(by_cell) != expected_cells:
        missing = sorted(expected_cells - set(by_cell))
        extra = sorted(set(by_cell) - expected_cells)
        raise ValueError(f"source matrix is not dense missing={missing[:3]} extra={extra[:3]}")
    compact: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    for task in tasks:
        task_id = task["instance_id"]
        weak_formats = [
            by_cell[(task_id, prompt_format, WEAK_MODEL)][0] for prompt_format in FORMATS
        ]
        strong_formats = [
            by_cell[(task_id, prompt_format, STRONG_MODEL)][0] for prompt_format in FORMATS
        ]
        weak_attempts = sum(
            by_cell[(task_id, prompt_format, WEAK_MODEL)][1] for prompt_format in FORMATS
        )
        strong_attempts = sum(
            by_cell[(task_id, prompt_format, STRONG_MODEL)][1] for prompt_format in FORMATS
        )
        compact.append(
            {
                "instance_id": task_id,
                "repo": task["repo"],
                "text": texts[task_id],
                "cheap_reward": float(np.mean(weak_formats)),
                "cheap_attempts": weak_attempts,
                "strong_reward": float(np.mean(strong_formats)),
                "strong_attempts": strong_attempts,
            }
        )
        for prompt_format in FORMATS:
            for arm, model in (("weak", WEAK_MODEL), ("strong", STRONG_MODEL)):
                reward, attempts = by_cell[(task_id, prompt_format, model)]
                cells.append(
                    {
                        "instance_id": task_id,
                        "repo": task["repo"],
                        "prompt_format": prompt_format,
                        "arm": arm,
                        "model": model,
                        "reward": reward,
                        "attempts": attempts,
                    }
                )
    return compact, cells


def _estimate(
    task_indices: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
) -> tuple[float, float]:
    """Return strong-minus-weak and cross-format oracle headroom for one sample."""
    strong_delta = float(np.mean(strong[task_indices]) - np.mean(weak[task_indices]))
    direction_headrooms = []
    for fit_format, heldout_format in ((0, 1), (1, 0)):
        fit_weak = weak[task_indices, fit_format]
        fit_strong = strong[task_indices, fit_format]
        heldout_weak = weak[task_indices, heldout_format]
        heldout_strong = strong[task_indices, heldout_format]
        choose_strong = fit_strong > fit_weak
        oracle = np.where(choose_strong, heldout_strong, heldout_weak)
        static_is_strong = float(np.mean(fit_strong)) > float(np.mean(fit_weak))
        static = heldout_strong if static_is_strong else heldout_weak
        direction_headrooms.append(float(np.mean(oracle - static)))
    return strong_delta, float(np.mean(direction_headrooms))


def _oracle(compact: list[dict[str, object]], cells: list[dict[str, object]]) -> dict[str, object]:
    """Run the frozen repository bootstrap and oracle gates."""
    task_ids = [str(row["instance_id"]) for row in compact]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    weak = np.empty((len(task_ids), len(FORMATS)), dtype=np.float64)
    strong = np.empty_like(weak)
    for cell in cells:
        index = task_index[str(cell["instance_id"])]
        format_index = FORMATS.index(str(cell["prompt_format"]))
        target = weak if cell["arm"] == "weak" else strong
        target[index, format_index] = float(cell["reward"])
    all_indices = np.arange(len(task_ids), dtype=np.int64)
    strong_delta, headroom = _estimate(all_indices, weak, strong)
    by_repo: dict[str, list[int]] = {}
    for index, row in enumerate(compact):
        by_repo.setdefault(str(row["repo"]), []).append(index)
    groups = sorted(by_repo)
    rng = np.random.default_rng(SEED)
    draws = np.empty((BOOTSTRAPS, 2), dtype=np.float64)
    for draw in range(BOOTSTRAPS):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = np.concatenate(
            [np.asarray(by_repo[str(group)], dtype=np.int64) for group in sampled_groups]
        )
        draws[draw] = _estimate(sampled, weak, strong)
    strong_interval = np.quantile(draws[:, 0], [0.025, 0.5, 0.975]).tolist()
    oracle_interval = np.quantile(draws[:, 1], [0.025, 0.5, 0.975]).tolist()
    gates = {
        "minimum_tasks": len(task_ids) >= 500,
        "minimum_repositories": len(groups) >= 50,
        "strong_arm_positive": strong_delta > 0.0 and strong_interval[0] > 0.01,
        "oracle_mean_headroom": headroom >= 0.03,
        "oracle_lower_bound": oracle_interval[0] > 0.01,
        "dense_two_by_two": len(cells) == len(task_ids) * 4,
    }
    return {
        "tasks": len(task_ids),
        "repositories": len(groups),
        "weak_reward": float(np.mean(weak)),
        "strong_reward": float(np.mean(strong)),
        "strong_minus_weak": strong_delta,
        "strong_minus_weak_95ci": strong_interval,
        "mean_cross_format_oracle_headroom": headroom,
        "cross_format_oracle_headroom_95ci": oracle_interval,
        "bootstraps": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    """Materialize source outcomes and write the frozen oracle verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected, tasks = _load_freeze(args.freeze.resolve())
    connection = duckdb.connect(str(args.database.resolve()))
    connection.execute("SET threads = 8")
    connection.execute("SET memory_limit = '6GB'")
    compact, cells = _materialize(connection, tasks)
    _write_json(output / "paired-source.json", compact)
    _write_json(output / "format-cells.json", cells)
    report = _oracle(compact, cells)
    report["protocol"] = {
        "trajectory_dataset": TRAJECTORY_DATASET,
        "trajectory_revision": TRAJECTORY_REVISION,
        "task_dataset": TASK_DATASET,
        "task_revision": TASK_REVISION,
        "weak_model": WEAK_MODEL,
        "strong_model": STRONG_MODEL,
        "formats": list(FORMATS),
        "cohort_sha256": selected["cohort_sha256"],
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "source_commit": args.source_commit,
        "oracle_script_sha256": _sha256_file(Path(__file__).resolve()),
        "target_outcomes_used": False,
    }
    report["paired_source_sha256"] = _sha256_file(output / "paired-source.json")
    report["format_cells_sha256"] = _sha256_file(output / "format-cells.json")
    _write_json(output / "oracle-report.json", report)
    logger.info(
        "oracle complete strong_delta=%.4f headroom=%.4f passed=%s",
        report["strong_minus_weak"],
        report["mean_cross_format_oracle_headroom"],
        report["passed"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
