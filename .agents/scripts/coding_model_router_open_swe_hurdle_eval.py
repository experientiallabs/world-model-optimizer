"""Evaluate the externally gated Open-SWE router once on DeepSWE v1.1.

The scorer and all 113 target task scores are frozen before the DeepSWE outcome
matrix is read. The action is the preregistered Luna xhigh to max ladder with
20 percent max traffic. No fitted model is persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import cast

import coding_model_router_beyondswe_fit as beyond
import coding_model_router_open_swe_hurdle as hurdle
import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger("coding-router-open-swe-hurdle-eval")

WEAK_ARM = "mini_swe_agent_gpt_5_6_luna_xhigh"
STRONG_ARM = "mini_swe_agent_gpt_5_6_luna_max"
TRAFFIC_FRACTION = 0.20
RANDOM_SAMPLES = 20_000
BOOTSTRAP_SAMPLES = 5_000


def _read_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): value for key, value in raw.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _target_metadata(path: Path) -> tuple[list[str], list[str], list[str]]:
    raw = _read_object(path)
    values = raw.get("rows")
    if not isinstance(values, list):
        raise ValueError(f"{path} has no rows")
    rows = [
        {str(key): value for key, value in row.items()}
        for row in values
        if isinstance(row, dict)
    ]
    if len(rows) != len(values):
        raise ValueError(f"{path} contains a non-object row")
    task_ids = [str(row.get("id", row.get("instance_id"))) for row in rows]
    texts = []
    for row in rows:
        text = str(
            row.get("text", row.get("problem_statement", row.get("prompt", "")))
        )
        if not text:
            title = str(row.get("problem_title") or "")
            description = str(row.get("display_description") or "")
            text = "\n".join(value for value in (title, description) if value)
        texts.append(text)
    groups = [
        hurdle._repo(str(row.get("repository", row.get("repo", "unknown"))))
        for row in rows
    ]
    if len(set(task_ids)) != len(task_ids) or any(not text for text in texts):
        raise ValueError("target metadata has duplicate ids or empty text")
    return task_ids, texts, groups


def _target_feature_view(
    matrix_path: Path,
    metadata_path: Path,
    output: Path,
) -> Path:
    metadata_ids, _, metadata_groups = _target_metadata(metadata_path)
    groups = dict(zip(metadata_ids, metadata_groups, strict=True))
    raw = _read_object(matrix_path)
    values = raw.get("outcomes")
    if not isinstance(values, list):
        raise ValueError(f"{matrix_path} has no outcomes")
    texts: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        task_id = value.get("scenario_id")
        text = value.get("task")
        if not isinstance(task_id, str) or not isinstance(text, str) or not text:
            raise ValueError("target outcome row has no scenario_id or task text")
        existing = texts.setdefault(task_id, text)
        if existing != text:
            raise ValueError(f"target task text differs across arms for {task_id}")
    missing_groups = sorted(texts.keys() - groups.keys())
    if missing_groups:
        raise ValueError(f"target feature rows lack repository metadata: {missing_groups[:5]}")
    path = output / "target-feature-view.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "deepswe-pre-inference-task-feature-view-v1",
                "target_reward_fields_accessed": False,
                "target_cost_fields_accessed": False,
                "rows": [
                    {
                        "id": task_id,
                        "text": texts[task_id],
                        "repository": groups[task_id],
                    }
                    for task_id in metadata_ids
                    if task_id in texts
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _auxiliary_target_scores(
    source_path: Path,
    task_ids: list[str],
    texts: list[str],
    groups: list[str],
    *,
    seed: int,
) -> np.ndarray:
    source = beyond._source(source_path)
    validation = beyond.ValidationData(
        task_ids=task_ids,
        groups=groups,
        texts=texts,
        structural=np.asarray(
            [beyond._structural(text) for text in texts],
            dtype=np.float64,
        ),
        cheap=np.zeros(len(task_ids), dtype=np.float64),
        strong=np.zeros(len(task_ids), dtype=np.float64),
    )
    candidate = beyond.Candidate(
        "structural-extra-trees",
        "structural",
        1,
        0.0,
        "extra-trees",
    )
    return beyond._transfer_scores(source, validation, candidate, seed=seed)


def _target_features(
    task_ids: list[str],
    texts: list[str],
    groups: list[str],
    auxiliary: np.ndarray,
    candidate: hurdle.Candidate,
) -> object:
    size = len(task_ids)
    data = hurdle.Data(
        task_ids=task_ids,
        groups=np.asarray(groups, dtype=object),
        texts=texts,
        structural=np.asarray(
            [hurdle._structural(text) for text in texts],
            dtype=np.float64,
        ),
        auxiliary=auxiliary,
        cheap=np.zeros(size, dtype=np.float64),
        strong=np.zeros(size, dtype=np.float64),
        cheap_attempts=np.ones(size, dtype=np.float64),
        strong_attempts=np.ones(size, dtype=np.float64),
    )
    return hurdle._features(data, candidate)


def _freeze_scores(
    paired_path: Path,
    auxiliary_scores_path: Path,
    beyond_source_path: Path,
    target_tasks_path: Path,
    output: Path,
    *,
    cost_penalty: float,
    seed: int,
) -> tuple[list[str], list[str], np.ndarray, Path]:
    train = hurdle._load_data(paired_path, auxiliary_scores_path)
    candidate = hurdle.Candidate(
        "hybrid-ridge-direct-a10",
        "hybrid",
        "direct-ridge",
        alpha=10.0,
    )
    train_features = hurdle._features(train, candidate)
    task_ids, texts, groups = _target_metadata(target_tasks_path)
    auxiliary = _auxiliary_target_scores(
        beyond_source_path,
        task_ids,
        texts,
        groups,
        seed=seed,
    )
    target_features = _target_features(
        task_ids,
        texts,
        groups,
        auxiliary,
        candidate,
    )
    model = Ridge(alpha=10.0)
    model.fit(
        train_features,
        train.uplift,
        sample_weight=train.precision,
    )
    base_scores = np.asarray(model.predict(target_features), dtype=np.float64)
    scores = base_scores - cost_penalty * auxiliary
    freeze_path = output / "target-score-freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "protocol": "open-swe-hybrid-direct-target-score-freeze-v1",
                "candidate": (
                    candidate.name
                    if cost_penalty == 0.0
                    else f"{candidate.name}-cost-aware-p{cost_penalty:g}"
                ),
                "base_candidate": candidate.name,
                "trace_burden_penalty": cost_penalty,
                "traffic_fraction": TRAFFIC_FRACTION,
                "target_outcomes_used": False,
                "rows": [
                    {
                        "task_id": task_id,
                        "group": groups[index],
                        "score": float(scores[index]),
                        "base_uplift_score": float(base_scores[index]),
                        "auxiliary_beyond_burden": float(auxiliary[index]),
                    }
                    for index, task_id in enumerate(task_ids)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return task_ids, groups, scores, freeze_path


def _matrix(
    path: Path,
) -> tuple[
    dict[tuple[str, str], tuple[float, float]],
    list[dict[str, float | str]],
]:
    raw = _read_object(path)
    values = raw.get("outcomes")
    if not isinstance(values, list):
        raise ValueError(f"{path} has no outcomes")
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    static: dict[str, list[tuple[str, float, float]]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        row = {str(key): item for key, item in value.items()}
        task_id = str(row["scenario_id"])
        arm = str(row["model"])
        key = (task_id, arm)
        if key in cells:
            raise ValueError(f"duplicate target cell {key}")
        reward = _number(row["reward"], name="reward")
        cost = _number(row["cost_usd"], name="cost_usd")
        cells[key] = (reward, cost)
        static.setdefault(arm, []).append((task_id, reward, cost))
    static_rows = [
        {
            "arm": arm,
            "tasks": len(rows),
            "reward": float(np.mean([row[1] for row in rows])),
            "cost_usd": float(np.sum([row[2] for row in rows])),
        }
        for arm, rows in sorted(static.items())
    ]
    return cells, static_rows


def _random_control(
    weak_reward: np.ndarray,
    strong_reward: np.ndarray,
    weak_cost: np.ndarray,
    strong_cost: np.ndarray,
    routed_reward: float,
    routed_cost: float,
    *,
    strong_count: int,
    seed: int,
) -> dict[str, object]:
    probability = strong_count / len(weak_reward)
    expected_reward = float(
        np.mean(weak_reward + probability * (strong_reward - weak_reward))
    )
    expected_cost = float(
        np.sum(weak_cost + probability * (strong_cost - weak_cost))
    )
    rng = np.random.default_rng(seed)
    rewards = np.zeros(RANDOM_SAMPLES, dtype=np.float64)
    costs = np.zeros(RANDOM_SAMPLES, dtype=np.float64)
    for sample in range(RANDOM_SAMPLES):
        selected = rng.choice(len(weak_reward), size=strong_count, replace=False)
        reward = weak_reward.copy()
        cost = weak_cost.copy()
        reward[selected] = strong_reward[selected]
        cost[selected] = strong_cost[selected]
        rewards[sample] = float(np.mean(reward))
        costs[sample] = float(np.sum(cost))
    return {
        "samples": RANDOM_SAMPLES,
        "strong_tasks": strong_count,
        "expected_reward": expected_reward,
        "expected_cost_usd": expected_cost,
        "router_reward_delta_vs_random_mean": routed_reward - expected_reward,
        "router_cost_delta_vs_random_mean_usd": routed_cost - expected_cost,
        "reward_95ci": [
            float(value) for value in np.quantile(rewards, [0.025, 0.975])
        ],
        "cost_95ci_usd": [
            float(value) for value in np.quantile(costs, [0.025, 0.975])
        ],
        "router_quality_percentile": float(np.mean(rewards <= routed_reward)),
        "router_cost_percentile": float(np.mean(costs <= routed_cost)),
        "router_joint_dominance_percentile": float(
            np.mean((rewards <= routed_reward) & (costs >= routed_cost))
        ),
    }


def _bootstrap_vs_mixture(
    groups: list[str],
    routed_reward: np.ndarray,
    routed_cost: np.ndarray,
    expected_reward: np.ndarray,
    expected_cost: np.ndarray,
    *,
    seed: int,
) -> dict[str, list[float]]:
    unique = sorted(set(groups))
    group_array = np.asarray(groups, dtype=object)
    by_group = {
        group: np.flatnonzero(group_array == group)
        for group in unique
    }
    rng = np.random.default_rng(seed)
    reward_delta = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    cost_delta = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in selected_groups])
        reward_delta[sample] = float(
            np.mean(routed_reward[indices] - expected_reward[indices])
        )
        cost_delta[sample] = float(
            np.sum(routed_cost[indices] - expected_cost[indices])
        )
    return {
        "reward_delta_95ci": [
            float(value) for value in np.quantile(reward_delta, [0.025, 0.975])
        ],
        "cost_delta_95ci_usd": [
            float(value) for value in np.quantile(cost_delta, [0.025, 0.975])
        ],
    }


def evaluate(
    paired_path: Path,
    auxiliary_scores_path: Path,
    beyond_source_path: Path,
    external_report_path: Path,
    cost_aware_report_path: Path | None,
    target_tasks_path: Path,
    target_matrix_path: Path,
    output: Path,
    *,
    seed: int,
) -> None:
    external_report = _read_object(external_report_path)
    gate = external_report.get("external_gate")
    selected = external_report.get("selected_final_candidate")
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or not isinstance(selected, dict)
        or selected.get("candidate") != "hybrid-ridge-direct-a10"
    ):
        raise ValueError("external gate or frozen candidate is invalid")
    cost_penalty = 0.0
    if cost_aware_report_path is not None:
        cost_report = _read_object(cost_aware_report_path)
        cost_gate = cost_report.get("external_gate")
        cost_selected = cost_report.get("selected_final")
        if (
            not isinstance(cost_gate, dict)
            or cost_gate.get("passed") is not True
            or not isinstance(cost_selected, dict)
        ):
            raise ValueError("cost-aware external gate is invalid")
        cost_penalty = _number(
            cost_selected.get("penalty"),
            name="cost-aware penalty",
        )
        if cost_penalty <= 0.0:
            raise ValueError("cost-aware penalty must be positive")
    output.mkdir(parents=True, exist_ok=False)
    feature_view_path = _target_feature_view(
        target_matrix_path,
        target_tasks_path,
        output,
    )
    task_ids, groups, scores, freeze_path = _freeze_scores(
        paired_path,
        auxiliary_scores_path,
        beyond_source_path,
        feature_view_path,
        output,
        cost_penalty=cost_penalty,
        seed=seed,
    )
    freeze_sha256 = _sha256(freeze_path)

    cells, static_rows = _matrix(target_matrix_path)
    complete = [
        index
        for index, task_id in enumerate(task_ids)
        if (task_id, WEAK_ARM) in cells and (task_id, STRONG_ARM) in cells
    ]
    if len(complete) < 100:
        raise ValueError(f"only {len(complete)} target tasks have the frozen ladder")
    complete_ids = [task_ids[index] for index in complete]
    complete_groups = [groups[index] for index in complete]
    complete_scores = scores[complete]
    weak_reward = np.asarray(
        [cells[(task_id, WEAK_ARM)][0] for task_id in complete_ids],
        dtype=np.float64,
    )
    weak_cost = np.asarray(
        [cells[(task_id, WEAK_ARM)][1] for task_id in complete_ids],
        dtype=np.float64,
    )
    strong_reward = np.asarray(
        [cells[(task_id, STRONG_ARM)][0] for task_id in complete_ids],
        dtype=np.float64,
    )
    strong_cost = np.asarray(
        [cells[(task_id, STRONG_ARM)][1] for task_id in complete_ids],
        dtype=np.float64,
    )
    strong_count = max(1, int(round(TRAFFIC_FRACTION * len(complete_ids))))
    order = np.argsort(-complete_scores, kind="mergesort")
    decisions = np.zeros(len(complete_ids), dtype=bool)
    decisions[order[:strong_count]] = True
    routed_reward_rows = np.where(decisions, strong_reward, weak_reward)
    routed_cost_rows = np.where(decisions, strong_cost, weak_cost)
    router_reward = float(np.mean(routed_reward_rows))
    router_cost = float(np.sum(routed_cost_rows))
    random_control = _random_control(
        weak_reward,
        strong_reward,
        weak_cost,
        strong_cost,
        router_reward,
        router_cost,
        strong_count=strong_count,
        seed=seed,
    )
    probability = strong_count / len(complete_ids)
    expected_reward_rows = weak_reward + probability * (strong_reward - weak_reward)
    expected_cost_rows = weak_cost + probability * (strong_cost - weak_cost)
    repository_bootstrap = _bootstrap_vs_mixture(
        complete_groups,
        routed_reward_rows,
        routed_cost_rows,
        expected_reward_rows,
        expected_cost_rows,
        seed=seed,
    )
    static_complete = [
        row
        for row in static_rows
        if int(cast(int, row["tasks"])) == len(complete_ids)
    ]
    weak_static = next(row for row in static_complete if row["arm"] == WEAK_ARM)
    strong_static = next(row for row in static_complete if row["arm"] == STRONG_ARM)
    affordable = [
        row for row in static_complete if float(cast(float, row["cost_usd"])) <= router_cost
    ]
    best_affordable = max(
        affordable,
        key=lambda row: float(cast(float, row["reward"])),
    )
    static_dominators = [
        row
        for row in static_complete
        if float(cast(float, row["cost_usd"])) <= router_cost
        and float(cast(float, row["reward"])) >= router_reward
        and (
            float(cast(float, row["cost_usd"])) < router_cost
            or float(cast(float, row["reward"])) > router_reward
        )
    ]
    promotion = {
        "quality_above_matched_random_mean": (
            _number(
                random_control["router_reward_delta_vs_random_mean"],
                name="random reward delta",
            )
            > 0.0
        ),
        "cost_at_or_below_matched_random_mean": (
            _number(
                random_control["router_cost_delta_vs_random_mean_usd"],
                name="random cost delta",
            )
            <= 0.0
        ),
        "quality_percentile_at_least_0_95": (
            _number(
                random_control["router_quality_percentile"],
                name="random quality percentile",
            )
            >= 0.95
        ),
        "quality_retention_vs_max_at_least_0_95": (
            router_reward / float(cast(float, strong_static["reward"])) >= 0.95
        ),
        "cost_savings_vs_max_at_least_0_35": (
            1.0 - router_cost / float(cast(float, strong_static["cost_usd"])) >= 0.35
        ),
        "not_dominated_by_static_arm": not static_dominators,
    }
    promotion["passed"] = all(bool(value) for value in promotion.values())
    oracle_order = np.argsort(-(strong_reward - weak_reward), kind="mergesort")
    oracle_reward_rows = weak_reward.copy()
    oracle_reward_rows[oracle_order[:strong_count]] = strong_reward[
        oracle_order[:strong_count]
    ]
    report = {
        "protocol": (
            "open-swe-router-deepswe-v1.1-confirmation-v2"
            if cost_penalty == 0.0
            else "open-swe-cost-aware-router-deepswe-v1.1-adaptive-v1"
        ),
        "corrects_v1_short_description_feature_mismatch": True,
        "adaptive_after_full_prompt_target_diagnostic": cost_penalty > 0.0,
        "trace_burden_penalty": cost_penalty,
        "external_report_sha256": _sha256(external_report_path),
        "cost_aware_external_report_sha256": (
            _sha256(cost_aware_report_path)
            if cost_aware_report_path is not None
            else None
        ),
        "external_gate_passed": True,
        "target_scores_frozen_before_outcomes": True,
        "target_score_freeze_sha256": freeze_sha256,
        "target_feature_view_sha256": _sha256(feature_view_path),
        "target_feature_view_uses_full_pre_inference_task_prompt": True,
        "target_outcomes_used_for_fit": False,
        "target_outcomes_used_for_threshold": False,
        "target_tasks": len(complete_ids),
        "target_repositories": len(set(complete_groups)),
        "ladder": [WEAK_ARM, STRONG_ARM],
        "traffic": {
            WEAK_ARM: len(complete_ids) - strong_count,
            STRONG_ARM: strong_count,
        },
        "router": {
            "reward": router_reward,
            "cost_usd": router_cost,
            "quality_retention_vs_max": (
                router_reward / float(cast(float, strong_static["reward"]))
            ),
            "cost_savings_vs_max": (
                1.0 - router_cost / float(cast(float, strong_static["cost_usd"]))
            ),
            "quality_gain_vs_xhigh": (
                router_reward - float(cast(float, weak_static["reward"]))
            ),
            "cost_increase_vs_xhigh_usd": (
                router_cost - float(cast(float, weak_static["cost_usd"]))
            ),
        },
        "matched_task_blind_control": random_control,
        "repository_bootstrap_vs_matched_mixture": repository_bootstrap,
        "best_static_at_or_below_router_cost": best_affordable,
        "static_dominators": static_dominators,
        "oracle_at_equal_traffic": {
            "reward": float(np.mean(oracle_reward_rows)),
            "quality_headroom_vs_router": float(
                np.mean(oracle_reward_rows) - router_reward
            ),
        },
        "promotion": promotion,
        "static_arms": static_complete,
        "no_persisted_fitted_model": True,
        "inputs": {
            "paired_sha256": _sha256(paired_path),
            "auxiliary_scores_sha256": _sha256(auxiliary_scores_path),
            "beyond_source_sha256": _sha256(beyond_source_path),
            "target_tasks_sha256": _sha256(target_tasks_path),
            "target_matrix_sha256": _sha256(target_matrix_path),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "target-decisions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "group": complete_groups[index],
                    "score": float(complete_scores[index]),
                    "arm": STRONG_ARM if decisions[index] else WEAK_ARM,
                    "reward": float(routed_reward_rows[index]),
                    "cost_usd": float(routed_cost_rows[index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(complete_ids)
        ),
        encoding="utf-8",
    )
    logger.info(
        "DeepSWE confirmation reward=%.6f cost=%.2f random_reward_delta=%.6f "
        "random_cost_delta=%.2f quality_percentile=%.4f static_dominators=%d pass=%s",
        router_reward,
        router_cost,
        _number(
            random_control["router_reward_delta_vs_random_mean"],
            name="random reward delta",
        ),
        _number(
            random_control["router_cost_delta_vs_random_mean_usd"],
            name="random cost delta",
        ),
        _number(
            random_control["router_quality_percentile"],
            name="random quality percentile",
        ),
        len(static_dominators),
        promotion["passed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--auxiliary-scores", type=Path, required=True)
    parser.add_argument("--beyond-source", type=Path, required=True)
    parser.add_argument("--external-report", type=Path, required=True)
    parser.add_argument("--cost-aware-report", type=Path)
    parser.add_argument("--target-tasks", type=Path, required=True)
    parser.add_argument("--target-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    evaluate(
        args.paired,
        args.auxiliary_scores,
        args.beyond_source,
        args.external_report,
        args.cost_aware_report,
        args.target_tasks,
        args.target_matrix,
        args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
