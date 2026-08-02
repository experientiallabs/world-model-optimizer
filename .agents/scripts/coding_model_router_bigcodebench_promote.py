"""Audit five BigCodeBench heldout reports and decide external promotion.

The module consumes only immutable outer-heldout reports written after the
selection lock. It never reads DeepSWE data. Family-cluster bootstraps and all
promotion thresholds are fixed here before source heldout replay.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from coding_model_router_bigcodebench_evaluate import SeedHeldoutReport, ValueRecord
from coding_model_router_bigcodebench_fit import ARMS, SelectionLock, require_selection_lock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.core.files import write_text_atomic

logger = logging.getLogger(__name__)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_731
ChoiceField = Literal[
    "router_arm", "baseline_arm", "random_arm", "cost_only_arm", "shuffled_label_arm"
]


class Interval(BaseModel):
    """Point estimate and deterministic percentile interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point: float
    lower: float
    upper: float


class SeedGate(BaseModel):
    """Point-estimate quality and cost gates for one outer seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, le=4)
    router_reward: float = Field(ge=0.0, le=1.0)
    baseline_reward: float = Field(ge=0.0, le=1.0)
    quality_retention: float = Field(ge=0.0)
    absolute_quality_delta: float
    router_cost_usd: float = Field(ge=0.0)
    baseline_cost_usd: float = Field(gt=0.0)
    cost_savings: float
    passed: bool


class ControlGate(BaseModel):
    """Paired pooled reward advantage over one negative control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    reward_delta: Interval
    passed: bool


class FamilyGate(BaseModel):
    """Observed quality regression for one sufficiently repeated task family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str = Field(min_length=1)
    unique_tasks: int = Field(gt=0)
    evaluations: int = Field(gt=0)
    router_reward: float = Field(ge=0.0, le=1.0)
    baseline_reward: float = Field(ge=0.0, le=1.0)
    absolute_quality_delta: float
    allowed_loss: float = Field(gt=0.0)
    passed: bool


class ExternalPromotionReport(BaseModel):
    """Complete target-safe promotion verdict from five source heldout reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-external-promotion-v1"] = "bigcodebench-external-promotion-v1"
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    selection_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bootstrap_samples: int = Field(ge=100)
    bootstrap_seed: int
    seed_gates: list[SeedGate] = Field(min_length=5, max_length=5)
    pooled_quality_retention: Interval
    pooled_absolute_quality_delta: Interval
    pooled_cost_savings: Interval
    control_gates: list[ControlGate] = Field(min_length=4, max_length=4)
    family_gates: list[FamilyGate]
    consensus_fit_quality_feasible: bool
    all_seed_gates_passed: bool
    pooled_quality_gate_passed: bool
    task_signal_gate_passed: bool
    family_regression_gate_passed: bool
    passed: bool
    target_outcomes_used: Literal[False] = False

    @model_validator(mode="after")
    def _verdict_matches_gates(self) -> ExternalPromotionReport:
        expected = (
            self.consensus_fit_quality_feasible
            and self.all_seed_gates_passed
            and self.pooled_quality_gate_passed
            and self.task_signal_gate_passed
            and self.family_regression_gate_passed
        )
        if self.passed != expected:
            raise ValueError("external promotion verdict differs from its declared gates")
        if sorted(gate.seed for gate in self.seed_gates) != list(range(5)):
            raise ValueError("external promotion report must contain seeds 0 through 4")
        return self


def _interval(
    groups: list[str],
    first: np.ndarray,
    second: np.ndarray,
    *,
    kind: Literal["delta", "ratio", "savings"],
    samples: int,
    seed: int,
) -> Interval:
    """Return a deterministic task-family cluster bootstrap interval."""
    if first.shape != second.shape or first.shape != (len(groups),):
        raise ValueError("bootstrap values do not match their task-family labels")
    if samples < 100 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("bootstrap inputs are invalid")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("grouped bootstrap needs at least two task families")
    by_group = {
        group: np.asarray([index for index, value in enumerate(groups) if value == group])
        for group in unique
    }

    def statistic(indices: np.ndarray) -> float:
        first_mean = float(np.mean(first[indices]))
        second_mean = float(np.mean(second[indices]))
        if kind == "delta":
            return first_mean - second_mean
        if second_mean <= 0.0:
            if kind == "ratio" and first_mean >= second_mean:
                return 1.0
            raise ValueError("ratio bootstrap denominator is nonpositive")
        ratio = first_mean / second_mean
        return ratio if kind == "ratio" else 1.0 - ratio

    all_indices = np.arange(len(groups), dtype=np.int64)
    point = statistic(all_indices)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([by_group[unique[int(index)]] for index in selected])
        draws[sample] = statistic(indices)
    lower, upper = np.quantile(draws, np.asarray([0.025, 0.975]))
    return Interval(point=point, lower=float(lower), upper=float(upper))


def _chosen_values(
    report: SeedHeldoutReport,
    field: ChoiceField,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired reward and cost rows for one recorded route assignment."""
    choices = [getattr(task, field) for task in report.tasks]
    rewards = np.asarray(
        [task.arm_rewards[choice] for task, choice in zip(report.tasks, choices, strict=True)],
        dtype=np.float64,
    )
    costs = np.asarray(
        [task.arm_costs_usd[choice] for task, choice in zip(report.tasks, choices, strict=True)],
        dtype=np.float64,
    )
    return rewards, costs


def _matched_blind_rewards(report: SeedHeldoutReport) -> np.ndarray:
    """Return each task's expectation under the router's task-blind effort mix."""
    traffic = {arm: report.router.arm_counts[arm] / report.heldout_tasks for arm in ARMS}
    return np.asarray(
        [sum(traffic[arm] * task.arm_rewards[arm] for arm in ARMS) for task in report.tasks],
        dtype=np.float64,
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _verify_report_aggregates(report: SeedHeldoutReport) -> None:
    """Recompute every durable aggregate from task-level paired evidence."""
    assignments: dict[str, tuple[ValueRecord, ChoiceField]] = {
        "router": (report.router, "router_arm"),
        "baseline": (report.baseline, "baseline_arm"),
    }
    controls = {control.name: control for control in report.controls}
    assignments.update(
        {
            "seeded-uniform-random": (controls["seeded-uniform-random"], "random_arm"),
            "fit-cost-only": (controls["fit-cost-only"], "cost_only_arm"),
            "selected-shuffled-labels": (
                controls["selected-shuffled-labels"],
                "shuffled_label_arm",
            ),
        }
    )
    for name, (record, field) in assignments.items():
        rewards, costs = _chosen_values(report, field)
        counts = Counter(getattr(task, field) for task in report.tasks)
        if (
            not _close(record.reward, float(np.mean(rewards)))
            or not _close(record.cost_usd, float(np.mean(costs)))
            or record.arm_counts != {arm: counts[arm] for arm in ARMS}
        ):
            raise ValueError(f"heldout report aggregate differs for {name}")
    blind = controls["selected-matched-task-blind"]
    blind_rewards = _matched_blind_rewards(report)
    traffic = {arm: report.router.arm_counts[arm] / report.heldout_tasks for arm in ARMS}
    blind_cost = float(
        np.mean(
            [sum(traffic[arm] * task.arm_costs_usd[arm] for arm in ARMS) for task in report.tasks]
        )
    )
    if (
        not _close(blind.reward, float(np.mean(blind_rewards)))
        or not _close(blind.cost_usd, blind_cost)
        or blind.arm_counts != report.router.arm_counts
    ):
        raise ValueError("heldout matched task-blind aggregate differs from task evidence")
    for arm in ARMS:
        control = controls[f"static-{arm}"]
        rewards = np.asarray([task.arm_rewards[arm] for task in report.tasks])
        costs = np.asarray([task.arm_costs_usd[arm] for task in report.tasks])
        expected_counts = {value: report.heldout_tasks if value == arm else 0 for value in ARMS}
        if (
            not _close(control.reward, float(np.mean(rewards)))
            or not _close(control.cost_usd, float(np.mean(costs)))
            or control.arm_counts != expected_counts
        ):
            raise ValueError(f"heldout static aggregate differs for {arm}")


def analyze_external_promotion(
    lock: SelectionLock,
    selection_lock_sha256: str,
    reports: list[SeedHeldoutReport],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> ExternalPromotionReport:
    """Validate five reports and apply all frozen external promotion gates."""
    ordered = sorted(reports, key=lambda report: report.seed)
    if [report.seed for report in ordered] != list(range(5)):
        raise ValueError("external promotion needs seeds 0 through 4 exactly once")
    lock_seeds = {selection.seed: selection for selection in lock.seeds}
    for report in ordered:
        selection = lock_seeds[report.seed]
        if report.code_commit != lock.code_commit:
            raise ValueError(f"seed {report.seed} heldout report used a different commit")
        if report.selection_lock_sha256 != selection_lock_sha256:
            raise ValueError(f"seed {report.seed} heldout report used a different lock")
        if (
            report.fit_ids_sha256 != selection.fit_ids_sha256
            or report.heldout_ids_sha256 != selection.heldout_ids_sha256
            or report.candidate_name != selection.selected.name
            or report.candidate_config_sha256 != selection.selected.config_sha256
        ):
            raise ValueError(f"seed {report.seed} heldout report differs from its lock")
        _verify_report_aggregates(report)
    seed_gates = []
    for report in ordered:
        if report.baseline.cost_usd <= 0.0:
            raise ValueError(f"seed {report.seed} baseline cost is nonpositive")
        retention = (
            report.router.reward / report.baseline.reward if report.baseline.reward > 0.0 else 1.0
        )
        cost_savings = 1.0 - report.router.cost_usd / report.baseline.cost_usd
        seed_gates.append(
            SeedGate(
                seed=report.seed,
                router_reward=report.router.reward,
                baseline_reward=report.baseline.reward,
                quality_retention=retention,
                absolute_quality_delta=report.router.reward - report.baseline.reward,
                router_cost_usd=report.router.cost_usd,
                baseline_cost_usd=report.baseline.cost_usd,
                cost_savings=cost_savings,
                passed=(
                    report.router.reward >= 0.95 * report.baseline.reward and cost_savings >= 0.40
                ),
            )
        )
    groups: list[str] = []
    task_ids: list[str] = []
    router_rewards: list[float] = []
    router_costs: list[float] = []
    baseline_rewards: list[float] = []
    baseline_costs: list[float] = []
    blind_rewards: list[float] = []
    random_rewards: list[float] = []
    cost_only_rewards: list[float] = []
    shuffled_rewards: list[float] = []
    for report in ordered:
        routed_reward, routed_cost = _chosen_values(report, "router_arm")
        baseline_reward, baseline_cost = _chosen_values(report, "baseline_arm")
        random_reward, _ = _chosen_values(report, "random_arm")
        cost_only_reward, _ = _chosen_values(report, "cost_only_arm")
        shuffled_reward, _ = _chosen_values(report, "shuffled_label_arm")
        groups.extend(task.group for task in report.tasks)
        task_ids.extend(task.task_id for task in report.tasks)
        router_rewards.extend(routed_reward.tolist())
        router_costs.extend(routed_cost.tolist())
        baseline_rewards.extend(baseline_reward.tolist())
        baseline_costs.extend(baseline_cost.tolist())
        blind_rewards.extend(_matched_blind_rewards(report).tolist())
        random_rewards.extend(random_reward.tolist())
        cost_only_rewards.extend(cost_only_reward.tolist())
        shuffled_rewards.extend(shuffled_reward.tolist())
    router_reward_array = np.asarray(router_rewards)
    baseline_reward_array = np.asarray(baseline_rewards)
    router_cost_array = np.asarray(router_costs)
    baseline_cost_array = np.asarray(baseline_costs)
    retention = _interval(
        groups,
        router_reward_array,
        baseline_reward_array,
        kind="ratio",
        samples=samples,
        seed=seed,
    )
    quality_delta = _interval(
        groups,
        router_reward_array,
        baseline_reward_array,
        kind="delta",
        samples=samples,
        seed=seed + 1,
    )
    savings = _interval(
        groups,
        router_cost_array,
        baseline_cost_array,
        kind="savings",
        samples=samples,
        seed=seed + 2,
    )
    control_arrays = {
        "matched-task-blind": np.asarray(blind_rewards),
        "shuffled-label": np.asarray(shuffled_rewards),
        "seeded-random": np.asarray(random_rewards),
        "fit-cost-only": np.asarray(cost_only_rewards),
    }
    control_gates = []
    for index, (name, values) in enumerate(control_arrays.items()):
        interval = _interval(
            groups,
            router_reward_array,
            values,
            kind="delta",
            samples=samples,
            seed=seed + 10 + index,
        )
        control_gates.append(
            ControlGate(
                name=name,
                reward_delta=interval,
                passed=interval.lower > 0.0,
            )
        )
    family_gates: list[FamilyGate] = []
    for group in sorted(set(groups)):
        indices = np.asarray([index for index, value in enumerate(groups) if value == group])
        unique_tasks = len({task_ids[int(index)] for index in indices})
        if unique_tasks < 3:
            continue
        allowed_loss = 0.10 if unique_tasks >= 5 else 0.25
        router_reward = float(np.mean(router_reward_array[indices]))
        baseline_reward = float(np.mean(baseline_reward_array[indices]))
        delta = router_reward - baseline_reward
        family_gates.append(
            FamilyGate(
                group=group,
                unique_tasks=unique_tasks,
                evaluations=len(indices),
                router_reward=router_reward,
                baseline_reward=baseline_reward,
                absolute_quality_delta=delta,
                allowed_loss=allowed_loss,
                passed=delta >= -allowed_loss,
            )
        )
    all_seed_gates_passed = all(gate.passed for gate in seed_gates)
    pooled_quality_gate_passed = retention.lower >= 0.95
    task_signal_gate_passed = all(gate.passed for gate in control_gates)
    family_regression_gate_passed = all(gate.passed for gate in family_gates)
    consensus_feasible = lock.deployment_consensus.fit_quality_feasible
    passed = (
        consensus_feasible
        and all_seed_gates_passed
        and pooled_quality_gate_passed
        and task_signal_gate_passed
        and family_regression_gate_passed
    )
    return ExternalPromotionReport(
        code_commit=lock.code_commit,
        selection_lock_sha256=selection_lock_sha256,
        bootstrap_samples=samples,
        bootstrap_seed=seed,
        seed_gates=seed_gates,
        pooled_quality_retention=retention,
        pooled_absolute_quality_delta=quality_delta,
        pooled_cost_savings=savings,
        control_gates=control_gates,
        family_gates=family_gates,
        consensus_fit_quality_feasible=consensus_feasible,
        all_seed_gates_passed=all_seed_gates_passed,
        pooled_quality_gate_passed=pooled_quality_gate_passed,
        task_signal_gate_passed=task_signal_gate_passed,
        family_regression_gate_passed=family_regression_gate_passed,
        passed=passed,
    )


def _sha256(path: Path) -> str:
    """Return one immutable evidence file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_external_promotion_report(
    lock: SelectionLock,
    *,
    selection_lock_sha256: str,
    reports: list[SeedHeldoutReport],
    output: Path,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> ExternalPromotionReport:
    """Write one immutable external promotion verdict, or verify its exact replay."""
    report = analyze_external_promotion(
        lock,
        selection_lock_sha256,
        reports,
        samples=samples,
        seed=seed,
    )
    if output.exists():
        existing = ExternalPromotionReport.model_validate_json(output.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError("existing external promotion report differs from exact replay")
        return existing
    write_text_atomic(output, report.model_dump_json(indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    """Parse the external promotion audit command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Recompute five outer reports and publish one target-safe verdict."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    root = args.root.resolve()
    lock_path = args.lock.resolve()
    lock = require_selection_lock(root, lock_path)
    reports = [
        SeedHeldoutReport.model_validate_json(
            (args.reports_dir.resolve() / f"seed-{seed}.json").read_text(encoding="utf-8")
        )
        for seed in range(5)
    ]
    report = write_external_promotion_report(
        lock,
        selection_lock_sha256=_sha256(lock_path),
        reports=reports,
        output=args.output.resolve(),
    )
    logger.info(
        "external promotion audit complete passed=%s output=%s",
        report.passed,
        args.output,
    )


if __name__ == "__main__":
    main()
