"""Search external-only coding router algorithms and evaluate frozen policies on DeepSWE.

This research runner has two explicit phases:

* ``fit`` reads only external coding-task text and execution outcomes. It performs
  repository-grouped cross-validation, freezes candidate models and operating points, and writes
  an append-only trial ledger.
* ``evaluate`` loads those frozen candidates, then opens the published DeepSWE v1.1 matrix exactly
  for evaluation. DeepSWE outcomes never enter a fitted feature transform, estimator, or threshold.

The intended execution location is remote compute. The local workstation may upload inputs, start
the job, and sync artifacts, but must not run the fitting or bootstrap phases.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("coding-model_router_autoresearch")

QUALITY_FLOORS = (0.95, 0.97, 0.99)
FOLDS = 5
BOOTSTRAP_SAMPLES = 10_000
PRIMARY_TARGET_FAMILY = "mini_swe_agent_claude_opus_5"
SWEBENCH_WEAK_ARM = "20251211_mini-v1.17.2_gpt-5.2-2025-12-11"
SWEBENCH_STRONG_ARM = "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high"
TARGET_LADDERS: dict[str, tuple[str, ...]] = {
    "opus-low-high": (
        "mini_swe_agent_claude_opus_5_low",
        "mini_swe_agent_claude_opus_5_high",
    ),
    "luna-xhigh-max": (
        "mini_swe_agent_gpt_5_6_luna_xhigh",
        "mini_swe_agent_gpt_5_6_luna_max",
    ),
    "luna-xhigh-opus-high": (
        "mini_swe_agent_gpt_5_6_luna_xhigh",
        "mini_swe_agent_claude_opus_5_high",
    ),
    "luna-xhigh-max-opus-high": (
        "mini_swe_agent_gpt_5_6_luna_xhigh",
        "mini_swe_agent_gpt_5_6_luna_max",
        "mini_swe_agent_claude_opus_5_high",
    ),
}


class ExternalTaskRow(TypedDict):
    """One cached Nebius task row."""

    instance_id: str
    repo: str
    text: str
    cheap_reward: float
    strong_reward: float
    cheap_attempts: int
    strong_attempts: int


class ProfileTeacherRow(TypedDict):
    """One external task-profile supervision row without model outcomes."""

    instance_id: str
    repo: str
    language: str
    text: str
    difficulty: str
    intent_completeness: str
    pr_categories: list[str]


class JsonObject(Protocol):
    """Protocol for JSON mappings used by the one-off runner."""

    def get(self, key: str, default: object = ...) -> object: ...


class FittedRegressor(Protocol):
    """Estimator surface shared by the searched scikit-learn regressors."""

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> object: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


class FeatureTransformer(Protocol):
    """Feature transform surface shared by sklearn and the native hashing transform."""

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...


class HashingFeatureTransformer:
    """Stateless adapter around WMO's deterministic serve-time hashing embedder."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray([_hashing_vector(text, self.dim) for text in texts], dtype=np.float64)


class StructuralFeatureTransformer:
    """Deterministic issue-shape features available before model inference."""

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray([_structural_vector(text) for text in texts], dtype=np.float64)


class IrtDifficultyRegressor:
    """Map structural task easiness to the expected value of a stronger arm."""

    def __init__(self, alpha: float) -> None:
        self.model = Ridge(alpha=alpha)
        self.weak_offset = 0.0
        self.strong_offset = 0.0

    def fit_pair(
        self,
        features: np.ndarray,
        weak: np.ndarray,
        strong: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        mean_outcome = np.clip((weak + strong) / 2.0, 1e-4, 1.0 - 1e-4)
        easiness = np.log(mean_outcome / (1.0 - mean_outcome))
        self.model.fit(features, easiness, sample_weight=weights)
        predicted = np.asarray(self.model.predict(features), dtype=np.float64)
        self.weak_offset = _weighted_irt_offset(
            predicted,
            float(np.average(weak, weights=weights)),
            weights,
        )
        self.strong_offset = _weighted_irt_offset(
            predicted,
            float(np.average(strong, weights=weights)),
            weights,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        easiness = np.asarray(self.model.predict(features), dtype=np.float64)
        weak = _sigmoid(easiness + self.weak_offset)
        strong = _sigmoid(easiness + self.strong_offset)
        return strong - weak


@dataclasses.dataclass(frozen=True)
class TaskProfileClassifier:
    """Frozen centroid classifier distilled from disjoint external task metadata."""

    labels: tuple[str, ...]
    centroids: np.ndarray
    temperature: float

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = np.asarray(features @ self.centroids.T, dtype=np.float64)
        logits /= self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


@dataclasses.dataclass(frozen=True)
class TaskProfileUpliftRegressor:
    """Estimate marginal strong-arm gain through a latent task-profile prior."""

    classifier: TaskProfileClassifier
    profile_uplift: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.probabilities(features) @ self.profile_uplift


def _hashing_vector(text: str, dim: int) -> list[float]:
    """Mirror WMO HashingEmbedder without importing the package on remote fit workers."""
    if dim <= 0:
        raise ValueError(f"embedding dim must be positive, got {dim}")
    vector = np.zeros(dim, dtype=np.float64)
    normalized = text.lower()
    if len(normalized) < 3:
        normalized = normalized.ljust(3)
    for index in range(len(normalized) - 2):
        gram = normalized[index : index + 3]
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % dim
        vector[bucket] += 1.0 if digest[0] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector.tolist()


def _structural_vector(text: str) -> list[float]:
    normalized = text.replace("\r\n", "\n")
    lowered = normalized.lower()
    length = max(len(normalized), 1)
    lines = normalized.splitlines() or [normalized]
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized)
    unique_words = len({word.lower() for word in words})
    per_thousand = 1_000.0 / length
    counts = [
        normalized.count("\n"),
        normalized.count("```"),
        normalized.count("`"),
        len(re.findall(r"https?://", lowered)),
        len(re.findall(r"""(?:^|[\s`'"])[\w.-]+/[\w./-]+""", normalized)),
        len(re.findall(r"\b(?:error|exception|traceback|panic|failed|failure)\b", lowered)),
        len(re.findall(r"\b(?:test|tests|testing|assert|expected|actual)\b", lowered)),
        len(re.findall(r"\b(?:bug|fix|regression|broken|incorrect)\b", lowered)),
        len(re.findall(r"\b(?:feature|support|implement|add|enhancement)\b", lowered)),
        len(re.findall(r"\b(?:performance|slow|latency|memory|timeout)\b", lowered)),
        len(re.findall(r"\b(?:api|interface|method|function|class)\b", lowered)),
        len(re.findall(r"(?m)^\s*(?:[-*+] |\d+[.)] )", normalized)),
        len(re.findall(r"(?m)^#{1,6}\s", normalized)),
        len(re.findall(r"(?m)^\s*>", normalized)),
        normalized.count("{") + normalized.count("}"),
        normalized.count("[") + normalized.count("]"),
        normalized.count("(") + normalized.count(")"),
        normalized.count("="),
    ]
    uppercase = sum(character.isupper() for character in normalized)
    digits = sum(character.isdigit() for character in normalized)
    punctuation = sum(
        not character.isalnum() and not character.isspace() for character in normalized
    )
    line_lengths = [len(line) for line in lines]
    return [
        math.log1p(length) / 12.0,
        math.log1p(len(lines)) / 8.0,
        math.log1p(len(words)) / 12.0,
        min(sum(line_lengths) / len(line_lengths) / 200.0, 2.0),
        min(max(line_lengths) / 2_000.0, 2.0),
        unique_words / max(len(words), 1),
        uppercase / length,
        digits / length,
        punctuation / length,
        *[math.log1p(count * per_thousand) / 8.0 for count in counts],
    ]


class StaticRow(TypedDict):
    """One DeepSWE static-arm aggregate."""

    arm: str
    reward: float
    cost_usd: float


@dataclasses.dataclass(frozen=True)
class SourceData:
    """One external source normalized to weak and strong execution outcomes."""

    name: str
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    weak: np.ndarray
    strong: np.ndarray
    weak_attempts: np.ndarray
    strong_attempts: np.ndarray


@dataclasses.dataclass(frozen=True)
class ProfileTeacherData:
    """Disjoint external prompt-to-taxonomy supervision with no model outcomes."""

    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    difficulty: list[str]
    intent_completeness: list[str]
    pr_categories: list[list[str]]


@dataclasses.dataclass(frozen=True)
class CombinedData:
    """Deduplicated external task rows and source-balanced sample weights."""

    source_names: list[str]
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    weak: np.ndarray
    strong: np.ndarray
    sample_weight: np.ndarray


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    """One mechanically searchable static-text estimator."""

    name: str
    analyzer: Literal["word", "char", "hashing", "structural"]
    components: int
    estimator: Literal[
        "ridge-uplift",
        "ridge-heads",
        "extra-heads",
        "hist-heads",
        "irt-difficulty",
        "profile-uplift",
    ]
    alpha: float = 1.0
    min_leaf: int = 10
    label_mode: Literal["observed", "shuffled", "task-blind"] = "observed"
    profile_mode: Literal[
        "none",
        "difficulty",
        "intent",
        "pr-category",
        "difficulty-intent",
        "difficulty-pr-category",
    ] = "none"
    profile_temperature: float = 0.1
    profile_prior_strength: float = 50.0
    min_profile_tasks: int = 25


@dataclasses.dataclass(frozen=True)
class TargetData:
    """Published DeepSWE task text, group, arm reward, and arm cost matrices."""

    task_ids: list[str]
    texts: list[str]
    groups: list[str]
    arms: list[str]
    rewards: np.ndarray
    costs: np.ndarray


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"expected numeric value, found {type(value).__name__}")
    return float(value)


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


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _canonical_group(value: str) -> str:
    normalized = value.strip().lower().replace("__", "/")
    return normalized or "unknown"


def _empirical_bayes_rate(successes: float, attempts: int, global_mean: float) -> float:
    """Shrink a repeated success rate toward its source-wide mean."""
    prior_strength = 4.0
    epsilon = float(np.finfo(np.float64).eps)
    alpha = max(epsilon, prior_strength * global_mean)
    beta = max(epsilon, prior_strength * (1.0 - global_mean))
    return (successes + alpha) / (attempts + alpha + beta)


def _load_paired_json(path: Path, *, name: str) -> SourceData:
    """Load one compact paired-arm execution source."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows = [cast(ExternalTaskRow, item) for item in raw if isinstance(item, dict)]
    weak_total = sum(row["cheap_reward"] * row["cheap_attempts"] for row in rows)
    weak_attempts = sum(row["cheap_attempts"] for row in rows)
    strong_total = sum(row["strong_reward"] * row["strong_attempts"] for row in rows)
    strong_attempts = sum(row["strong_attempts"] for row in rows)
    weak_mean = weak_total / weak_attempts
    strong_mean = strong_total / strong_attempts
    weak = np.asarray(
        [
            _empirical_bayes_rate(
                row["cheap_reward"] * row["cheap_attempts"],
                row["cheap_attempts"],
                weak_mean,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    strong = np.asarray(
        [
            _empirical_bayes_rate(
                row["strong_reward"] * row["strong_attempts"],
                row["strong_attempts"],
                strong_mean,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return SourceData(
        name=name,
        task_ids=[row["instance_id"] for row in rows],
        groups=[_canonical_group(row["repo"]) for row in rows],
        texts=[row["text"] for row in rows],
        weak=weak,
        strong=strong,
        weak_attempts=np.asarray([row["cheap_attempts"] for row in rows], dtype=np.float64),
        strong_attempts=np.asarray([row["strong_attempts"] for row in rows], dtype=np.float64),
    )


def _load_nebius(path: Path) -> SourceData:
    return _load_paired_json(path, name="nebius-swe-agent-8b-70b")


def _load_profile_teacher(path: Path) -> ProfileTeacherData:
    """Load disjoint external task taxonomy rows without outcome labels."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows = [cast(ProfileTeacherRow, item) for item in raw if isinstance(item, dict)]
    if not rows:
        raise ValueError(f"{path} has no task-profile teacher rows")
    task_ids = [row["instance_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path} contains duplicate task-profile instance ids")
    return ProfileTeacherData(
        task_ids=task_ids,
        groups=[_canonical_group(row["repo"]) for row in rows],
        texts=[row["text"] for row in rows],
        difficulty=[row["difficulty"].strip().lower() or "unknown" for row in rows],
        intent_completeness=[
            row["intent_completeness"].strip().lower() or "unknown" for row in rows
        ],
        pr_categories=[
            sorted(
                {category.strip().lower() for category in row["pr_categories"] if category.strip()}
            )
            for row in rows
        ],
    )


def _r2e_string_set(path: Path, column: str) -> set[str]:
    values = pq.read_table(path, columns=[column])[column].to_pylist()
    strings = [str(value) for value in values]
    if len(strings) != len(set(strings)):
        raise ValueError(f"{path.name} contains duplicate {column} values")
    return set(strings)


def _read_r2e_task_bundle(task_id: str, payload: bytes) -> tuple[str, str]:
    """Read pre-call text and repository from one compressed R2E task bundle."""
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        instruction_member = archive.getmember("instruction.md")
        metadata_member = archive.getmember("environment/workspace/metadata.json")
        if instruction_member.size > 2_000_000 or metadata_member.size > 2_000_000:
            raise ValueError(f"R2E task {task_id} contains an oversized metadata member")
        instruction_file = archive.extractfile(instruction_member)
        metadata_file = archive.extractfile(metadata_member)
        if instruction_file is None or metadata_file is None:
            raise ValueError(f"R2E task {task_id} is missing task metadata")
        text = instruction_file.read().decode("utf-8")
        metadata = json.loads(metadata_file.read())
    if not isinstance(metadata, dict):
        raise TypeError(f"R2E task {task_id} metadata must be a JSON object")
    repo = metadata.get("repo_name")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError(f"R2E task {task_id} has no repository name")
    return text, repo


def _load_r2e(root: Path) -> SourceData:
    """Load paired, gradeable GPT-5 Codex and Kimi 2.5 R2E-Gym outcomes."""
    attempted = _r2e_string_set(root / "gpt5_codex_attempted.parquet", "task")
    solved = _r2e_string_set(root / "gpt5_codex_solved.parquet", "path")
    if not solved <= attempted:
        raise ValueError("R2E solved tasks must be a subset of attempted tasks")

    kimi_rows = pq.read_table(
        root / "kimi25_outcomes.parquet",
        columns=["task", "result"],
    ).to_pylist()
    kimi: dict[str, float] = {}
    excluded_kimi = 0
    for row in kimi_rows:
        task_id = str(row["task"])
        result = str(row["result"])
        if task_id in kimi:
            raise ValueError(f"duplicate R2E Kimi outcome for {task_id}")
        if result in {"0.0", "1.0"}:
            kimi[task_id] = float(result)
        else:
            excluded_kimi += 1

    tasks = sorted(attempted & kimi.keys())
    if not tasks:
        raise ValueError("R2E has no paired gradeable outcomes")
    task_set = set(tasks)
    task_table = pq.read_table(
        root / "dcagent_tasks.parquet",
        columns=["path", "task_binary"],
    )
    bundles: dict[str, bytes] = {}
    for row in task_table.to_pylist():
        task_id = str(row["path"])
        if task_id not in task_set:
            continue
        payload = row["task_binary"]
        if task_id in bundles:
            raise ValueError(f"duplicate R2E task bundle for {task_id}")
        if not isinstance(payload, bytes):
            raise TypeError(f"R2E task {task_id} bundle must be bytes")
        bundles[task_id] = payload
    missing = sorted(task_set - bundles.keys())
    if missing:
        raise ValueError(f"R2E is missing {len(missing)} task bundles, first={missing[0]}")

    texts: list[str] = []
    groups: list[str] = []
    for task_id in tasks:
        text, repo = _read_r2e_task_bundle(task_id, bundles[task_id])
        texts.append(text)
        groups.append(_canonical_group(repo))
    weak = np.asarray([float(task_id in solved) for task_id in tasks], dtype=np.float64)
    strong = np.asarray([kimi[task_id] for task_id in tasks], dtype=np.float64)
    logger.info(
        "R2E normalized tasks=%d groups=%d excluded_kimi=%d "
        "weak=gpt-5-codex/terminus-2 mean=%.4f strong=kimi-2.5/terminus-2 mean=%.4f",
        len(tasks),
        len(set(groups)),
        excluded_kimi,
        weak.mean(),
        strong.mean(),
    )
    return SourceData(
        name="r2e-gym-terminus",
        task_ids=tasks,
        groups=groups,
        texts=texts,
        weak=weak,
        strong=strong,
        weak_attempts=np.ones(len(tasks), dtype=np.float64),
        strong_attempts=np.ones(len(tasks), dtype=np.float64),
    )


def _coderouter_group(row: dict[str, object]) -> str:
    original = str(row.get("original_task_id", ""))
    if "__" in original:
        return _canonical_group(original.split("__", 1)[0])
    source = str(row.get("source_dataset", row.get("bench", "unknown")))
    return _canonical_group(source)


def _load_coderouter(root: Path) -> SourceData:
    task_path = root / "data" / "coderouterbench" / "ood176_tasks.jsonl"
    result_path = root / "data" / "coderouterbench" / "ood176_results_long.csv"
    tasks: dict[str, dict[str, object]] = {}
    for line in task_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if isinstance(row, dict):
            tasks[str(row["task_id"])] = {str(key): value for key, value in row.items()}
    cells: dict[tuple[str, str], float] = {}
    with result_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["resolved"] != "":
                cells[(row["model"], row["task_id"])] = float(row["resolved"])
    requested = ("Qwen3-Max", "gpt-5.4")
    task_ids = sorted(
        task_id for task_id in tasks if all((arm, task_id) in cells for arm in requested)
    )
    return SourceData(
        name="coderouterbench-ood176",
        task_ids=[str(tasks[task_id].get("original_task_id", task_id)) for task_id in task_ids],
        groups=[_coderouter_group(tasks[task_id]) for task_id in task_ids],
        texts=[str(tasks[task_id]["prompt"]) for task_id in task_ids],
        weak=np.asarray([cells[(requested[0], task_id)] for task_id in task_ids]),
        strong=np.asarray([cells[(requested[1], task_id)] for task_id in task_ids]),
        weak_attempts=np.ones(len(task_ids), dtype=np.float64),
        strong_attempts=np.ones(len(task_ids), dtype=np.float64),
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.where(
        value >= 0.0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def _weighted_irt_offset(
    easiness: np.ndarray,
    target_mean: float,
    weights: np.ndarray,
) -> float:
    """Calibrate one IRT arm ability to a weighted observed solve rate."""
    lower = -30.0
    upper = 30.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        predicted = float(np.average(_sigmoid(easiness + midpoint), weights=weights))
        if predicted < target_mean:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _calibrated_irt_probability(easiness: np.ndarray, target_mean: float) -> np.ndarray:
    """Map task easiness to probabilities whose aggregate matches one observed arm."""
    lower = -30.0
    upper = 30.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if float(_sigmoid(easiness + midpoint).mean()) < target_mean:
            lower = midpoint
        else:
            upper = midpoint
    return _sigmoid(easiness + (lower + upper) / 2.0)


def _load_swebench(matrix_path: Path, task_path: Path) -> SourceData:
    """Build a denoised reasoning-effort source from the published bash-only matrix."""
    raw = _read_json(matrix_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{matrix_path} must contain one JSON object")
    submissions = cast(dict[str, dict[str, object]], raw)
    missing = [arm for arm in (SWEBENCH_WEAK_ARM, SWEBENCH_STRONG_ARM) if arm not in submissions]
    if missing:
        raise ValueError(f"{matrix_path} is missing effort-pair submissions: {missing}")

    valid_arms: list[str] = []
    details_by_arm: dict[str, dict[str, dict[str, object]]] = {}
    for arm, submission in submissions.items():
        details = cast(dict[str, dict[str, object]], submission["details"])
        resolved = sum(bool(cell.get("resolved")) for cell in details.values())
        spend = sum(_as_float(cell.get("cost", 0.0) or 0.0) for cell in details.values())
        if resolved == 0 and spend > 1.0:
            continue
        valid_arms.append(arm)
        details_by_arm[arm] = details
    task_ids = sorted(set.intersection(*(set(details_by_arm[arm]) for arm in valid_arms)))
    metadata_table = pq.read_table(
        task_path,
        columns=["instance_id", "repo", "problem_statement"],
    )
    metadata = {
        str(instance_id): (str(repo), str(problem))
        for instance_id, repo, problem in zip(
            metadata_table["instance_id"].to_pylist(),
            metadata_table["repo"].to_pylist(),
            metadata_table["problem_statement"].to_pylist(),
            strict=True,
        )
    }
    task_ids = [task_id for task_id in task_ids if task_id in metadata]
    outcomes = np.asarray(
        [
            [float(bool(details_by_arm[arm][task_id].get("resolved"))) for task_id in task_ids]
            for arm in valid_arms
        ],
        dtype=np.float64,
    )
    smoothed_solve_rate = (outcomes.sum(axis=0) + 0.5) / (len(valid_arms) + 1.0)
    easiness = np.log(smoothed_solve_rate / (1.0 - smoothed_solve_rate))
    easiness -= float(easiness.mean())
    weak_observed = np.asarray(
        [
            float(bool(details_by_arm[SWEBENCH_WEAK_ARM][task_id].get("resolved")))
            for task_id in task_ids
        ],
        dtype=np.float64,
    )
    strong_observed = np.asarray(
        [
            float(bool(details_by_arm[SWEBENCH_STRONG_ARM][task_id].get("resolved")))
            for task_id in task_ids
        ],
        dtype=np.float64,
    )
    weak = _calibrated_irt_probability(easiness, float(weak_observed.mean()))
    strong = _calibrated_irt_probability(easiness, float(strong_observed.mean()))
    logger.info(
        "SWE-bench effort source normalized with arms=%d tasks=%d weak=%.4f strong=%.4f",
        len(valid_arms),
        len(task_ids),
        float(weak.mean()),
        float(strong.mean()),
    )
    return SourceData(
        name="swebench-verified-gpt52-effort-irt",
        task_ids=task_ids,
        groups=[_canonical_group(metadata[task_id][0]) for task_id in task_ids],
        texts=[metadata[task_id][1] for task_id in task_ids],
        weak=weak,
        strong=strong,
        weak_attempts=np.full(len(task_ids), len(valid_arms), dtype=np.float64),
        strong_attempts=np.full(len(task_ids), len(valid_arms), dtype=np.float64),
    )


def _combine(sources: list[SourceData]) -> CombinedData:
    seen_text: set[str] = set()
    assigned_task_ids: set[str] = set()
    source_names: list[str] = []
    task_ids: list[str] = []
    groups: list[str] = []
    texts: list[str] = []
    weak: list[float] = []
    strong: list[float] = []
    for source in sources:
        for index, text in enumerate(source.texts):
            digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if digest in seen_text:
                continue
            seen_text.add(digest)
            task_id = source.task_ids[index]
            if task_id in assigned_task_ids:
                task_id = f"{source.name}:{task_id}:{digest[:12]}"
            if task_id in assigned_task_ids:
                raise ValueError(f"combined task id collision after namespacing: {task_id}")
            assigned_task_ids.add(task_id)
            source_names.append(source.name)
            task_ids.append(task_id)
            groups.append(source.groups[index])
            texts.append(text)
            weak.append(float(source.weak[index]))
            strong.append(float(source.strong[index]))
    retained = collections.Counter(source_names)
    weights = [1.0 / retained[source] for source in source_names]
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array *= len(weight_array) / weight_array.sum()
    return CombinedData(
        source_names=source_names,
        task_ids=task_ids,
        groups=groups,
        texts=texts,
        weak=np.asarray(weak, dtype=np.float64),
        strong=np.asarray(strong, dtype=np.float64),
        sample_weight=weight_array,
    )


def _candidate_space(
    family: Literal["full", "native-linear", "structural-irt", "task-profile"] = "full",
) -> list[CandidateSpec]:
    if family == "task-profile":
        return [
            CandidateSpec(
                "task-blind-uplift",
                "hashing",
                512,
                "ridge-uplift",
                label_mode="task-blind",
            ),
            CandidateSpec(
                "profile-difficulty-pr-shuffled-t0.1-p50",
                "hashing",
                512,
                "profile-uplift",
                label_mode="shuffled",
                profile_mode="difficulty-pr-category",
            ),
            CandidateSpec(
                "profile-difficulty-t0.1-p50",
                "hashing",
                512,
                "profile-uplift",
                profile_mode="difficulty",
            ),
            CandidateSpec(
                "profile-intent-t0.1-p50",
                "hashing",
                512,
                "profile-uplift",
                profile_mode="intent",
            ),
            CandidateSpec(
                "profile-pr-category-t0.1-p50",
                "hashing",
                512,
                "profile-uplift",
                profile_mode="pr-category",
            ),
            CandidateSpec(
                "profile-difficulty-intent-t0.1-p50",
                "hashing",
                512,
                "profile-uplift",
                profile_mode="difficulty-intent",
            ),
            *[
                CandidateSpec(
                    f"profile-difficulty-pr-t{temperature:g}-p{prior:g}",
                    "hashing",
                    512,
                    "profile-uplift",
                    profile_mode="difficulty-pr-category",
                    profile_temperature=temperature,
                    profile_prior_strength=prior,
                )
                for temperature, prior in (
                    (0.05, 20.0),
                    (0.1, 20.0),
                    (0.1, 100.0),
                    (0.25, 100.0),
                )
            ],
        ]
    if family == "structural-irt":
        return [
            CandidateSpec(
                "task-blind-uplift",
                "structural",
                27,
                "ridge-uplift",
                label_mode="task-blind",
            ),
            CandidateSpec(
                "structural-irt-shuffled-a1",
                "structural",
                27,
                "irt-difficulty",
                alpha=1.0,
                label_mode="shuffled",
            ),
            *[
                CandidateSpec(
                    f"structural-irt-a{alpha:g}",
                    "structural",
                    27,
                    "irt-difficulty",
                    alpha=alpha,
                )
                for alpha in (0.1, 1.0, 10.0, 100.0)
            ],
            CandidateSpec(
                "structural-ridge-heads-a1",
                "structural",
                27,
                "ridge-heads",
                alpha=1.0,
            ),
        ]
    if family == "native-linear":
        return [
            CandidateSpec(
                "task-blind-uplift",
                "word",
                1,
                "ridge-uplift",
                label_mode="task-blind",
            ),
            CandidateSpec(
                "hash2048-ridge-heads-shuffled-a1",
                "hashing",
                2048,
                "ridge-heads",
                alpha=1.0,
                label_mode="shuffled",
            ),
            *[
                CandidateSpec(
                    f"hash{dim}-ridge-heads-a{alpha:g}",
                    "hashing",
                    dim,
                    "ridge-heads",
                    alpha=alpha,
                )
                for dim in (512, 2048, 8192)
                for alpha in (0.1, 1.0, 10.0)
            ],
        ]
    return [
        CandidateSpec(
            "task-blind-uplift",
            "word",
            1,
            "ridge-uplift",
            label_mode="task-blind",
        ),
        CandidateSpec(
            "word128-ridge-uplift-shuffled-a10",
            "word",
            128,
            "ridge-uplift",
            alpha=10.0,
            label_mode="shuffled",
        ),
        CandidateSpec("word64-ridge-uplift-a1", "word", 64, "ridge-uplift", alpha=1.0),
        CandidateSpec("word128-ridge-uplift-a10", "word", 128, "ridge-uplift", alpha=10.0),
        CandidateSpec("char128-ridge-uplift-a10", "char", 128, "ridge-uplift", alpha=10.0),
        CandidateSpec("word128-ridge-heads-a1", "word", 128, "ridge-heads", alpha=1.0),
        CandidateSpec("char128-ridge-heads-a1", "char", 128, "ridge-heads", alpha=1.0),
        CandidateSpec("word128-extra-heads-l5", "word", 128, "extra-heads", min_leaf=5),
        CandidateSpec("word128-extra-heads-l20", "word", 128, "extra-heads", min_leaf=20),
        CandidateSpec("word128-hist-heads-l10", "word", 128, "hist-heads", min_leaf=10),
        CandidateSpec("word128-hist-heads-l30", "word", 128, "hist-heads", min_leaf=30),
    ]


def _is_control(spec: CandidateSpec) -> bool:
    return spec.label_mode != "observed"


def _training_outcomes(
    spec: CandidateSpec,
    weak: np.ndarray,
    strong: np.ndarray,
    source_names: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.label_mode != "shuffled":
        return weak, strong
    rng = np.random.default_rng(seed)
    shuffled_weak = weak.copy()
    shuffled_strong = strong.copy()
    source_array = np.asarray(source_names, dtype=object)
    for source in sorted(set(source_names)):
        indices = np.flatnonzero(source_array == source)
        permutation = rng.permutation(indices)
        shuffled_weak[indices] = weak[permutation]
        shuffled_strong[indices] = strong[permutation]
    return shuffled_weak, shuffled_strong


def _features(spec: CandidateSpec) -> FeatureTransformer:
    if spec.analyzer == "hashing":
        return HashingFeatureTransformer(spec.components)
    if spec.analyzer == "structural":
        if spec.components != len(_structural_vector("")):
            raise ValueError(
                f"structural feature dimension is {len(_structural_vector(''))}, "
                f"not {spec.components}"
            )
        return StructuralFeatureTransformer()
    if spec.analyzer == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=60_000,
            sublinear_tf=True,
        )
    return cast(
        FeatureTransformer,
        Pipeline(
            [
                ("tfidf", vectorizer),
                ("svd", TruncatedSVD(n_components=spec.components, random_state=17)),
                ("scale", StandardScaler()),
            ]
        ),
    )


def _profile_labels(spec: CandidateSpec, teacher: ProfileTeacherData) -> list[str]:
    """Build the frozen task taxonomy for one task-profile candidate."""
    if spec.profile_mode == "none":
        raise ValueError(f"{spec.name} has no task-profile mode")
    raw: list[str] = []
    for index in range(len(teacher.task_ids)):
        difficulty = teacher.difficulty[index]
        intent = teacher.intent_completeness[index]
        categories = teacher.pr_categories[index]
        category = categories[0] if categories else "uncategorized"
        if spec.profile_mode == "difficulty":
            label = difficulty
        elif spec.profile_mode == "intent":
            label = intent
        elif spec.profile_mode == "pr-category":
            label = category
        elif spec.profile_mode == "difficulty-intent":
            label = f"{difficulty}|{intent}"
        else:
            label = f"{difficulty}|{category}"
        raw.append(label)
    counts = collections.Counter(raw)
    return [label if counts[label] >= spec.min_profile_tasks else "other" for label in raw]


def _fit_profile_classifier(
    spec: CandidateSpec,
    teacher: ProfileTeacherData,
    teacher_features: np.ndarray,
) -> TaskProfileClassifier:
    if spec.profile_temperature <= 0.0:
        raise ValueError(f"{spec.name} profile temperature must be positive")
    labels = _profile_labels(spec, teacher)
    vocabulary = tuple(sorted(set(labels)))
    if len(vocabulary) < 2:
        raise ValueError(f"{spec.name} task taxonomy collapsed to fewer than two profiles")
    centroids = np.zeros((len(vocabulary), teacher_features.shape[1]), dtype=np.float64)
    labels_array = np.asarray(labels, dtype=object)
    for index, label in enumerate(vocabulary):
        centroid = np.asarray(
            teacher_features[labels_array == label].mean(axis=0), dtype=np.float64
        )
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            centroid /= norm
        centroids[index] = centroid
    return TaskProfileClassifier(
        labels=vocabulary,
        centroids=centroids,
        temperature=spec.profile_temperature,
    )


def _fit_profile_uplift(
    spec: CandidateSpec,
    classifier: TaskProfileClassifier,
    features: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    weights: np.ndarray,
) -> TaskProfileUpliftRegressor:
    probabilities = classifier.probabilities(features)
    uplift = strong - weak
    global_uplift = float(np.average(uplift, weights=weights))
    weighted_probabilities = probabilities * weights[:, None]
    denominators = weighted_probabilities.sum(axis=0)
    numerators = weighted_probabilities.T @ uplift
    prior = spec.profile_prior_strength
    effects = (numerators + prior * global_uplift) / (denominators + prior)
    return TaskProfileUpliftRegressor(
        classifier=classifier,
        profile_uplift=np.asarray(effects, dtype=np.float64),
    )


def _estimators(spec: CandidateSpec) -> tuple[FittedRegressor, FittedRegressor | None]:
    if spec.estimator == "profile-uplift":
        raise ValueError("task-profile estimators require the external taxonomy fitter")
    if spec.estimator == "irt-difficulty":
        return cast(FittedRegressor, IrtDifficultyRegressor(spec.alpha)), None
    if spec.estimator in ("ridge-uplift", "ridge-heads"):
        first = Ridge(alpha=spec.alpha)
        second = Ridge(alpha=spec.alpha) if spec.estimator == "ridge-heads" else None
        return cast(FittedRegressor, first), cast(FittedRegressor | None, second)
    if spec.estimator == "extra-heads":
        first = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=spec.min_leaf,
            max_features=0.7,
            n_jobs=-1,
            random_state=29,
        )
        second = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=spec.min_leaf,
            max_features=0.7,
            n_jobs=-1,
            random_state=31,
        )
        return cast(FittedRegressor, first), cast(FittedRegressor, second)
    first = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=spec.min_leaf,
        l2_regularization=1.0,
        random_state=37,
    )
    second = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=spec.min_leaf,
        l2_regularization=1.0,
        random_state=41,
    )
    return cast(FittedRegressor, first), cast(FittedRegressor, second)


def _fit_estimators(
    spec: CandidateSpec,
    features: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    weights: np.ndarray,
) -> tuple[FittedRegressor, FittedRegressor | None]:
    first, second = _estimators(spec)
    if spec.estimator == "irt-difficulty":
        model = cast(IrtDifficultyRegressor, first)
        model.fit_pair(features, weak, strong, weights)
        return first, None
    if spec.estimator == "ridge-uplift":
        first.fit(features, strong - weak, sample_weight=weights)
        return first, None
    first.fit(features, weak, sample_weight=weights)
    if second is None:
        raise AssertionError(f"{spec.name} requires two potential-outcome heads")
    second.fit(features, strong, sample_weight=weights)
    return first, second


def _predict_score(
    spec: CandidateSpec,
    estimators: tuple[FittedRegressor, FittedRegressor | None],
    features: np.ndarray,
) -> np.ndarray:
    first, second = estimators
    if spec.estimator in ("ridge-uplift", "irt-difficulty"):
        return np.asarray(first.predict(features), dtype=np.float64)
    if second is None:
        raise AssertionError(f"{spec.name} has no strong-outcome head")
    weak = np.clip(np.asarray(first.predict(features), dtype=np.float64), 0.0, 1.0)
    strong = np.clip(np.asarray(second.predict(features), dtype=np.float64), 0.0, 1.0)
    return strong - weak


def _native_linear_heads(
    spec: CandidateSpec,
    estimators: tuple[FittedRegressor, FittedRegressor | None],
    operating_points: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Serialize a hashing plus Ridge scorer as plain numeric WMO policy inputs."""
    if spec.analyzer != "hashing" or spec.estimator != "ridge-heads":
        raise ValueError("native linear heads require a hashing ridge-heads candidate")
    weak, strong = estimators
    if strong is None or not isinstance(weak, Ridge) or not isinstance(strong, Ridge):
        raise TypeError("native linear heads require two fitted Ridge estimators")
    weak_weights = np.asarray(weak.coef_, dtype=np.float64).reshape(-1)
    strong_weights = np.asarray(strong.coef_, dtype=np.float64).reshape(-1)
    if weak_weights.size != spec.components or strong_weights.size != spec.components:
        raise ValueError("native linear head dimensions do not match the hashing embedder")
    return {
        "schema": "wmo-linear-heads-v1",
        "candidate": dataclasses.asdict(spec),
        "embedder": {"kind": "hashing", "dim": spec.components},
        "weak_weights": weak_weights.tolist(),
        "strong_weights": strong_weights.tolist(),
        "weak_bias": float(np.asarray(weak.intercept_).reshape(-1)[0]),
        "strong_bias": float(np.asarray(strong.intercept_).reshape(-1)[0]),
        "operating_points": operating_points,
        "target_outcomes_used": False,
        "target_embeddings_used": False,
    }


def _operating_point(
    scores: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    source_names: list[str],
    quality_floor: float,
) -> dict[str, float]:
    """Choose the least strong traffic that meets source-balanced external quality."""
    unique_sources = sorted(set(source_names))
    source_array = np.asarray(source_names, dtype=object)
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0.0, 1.0, 401)),
                np.asarray([np.nextafter(scores.max(), math.inf)]),
            ]
        )
    )
    feasible: list[dict[str, float]] = []
    for threshold in thresholds:
        use_strong = scores >= threshold
        routed = np.where(use_strong, strong, weak)
        retentions: list[float] = []
        traffic: list[float] = []
        for source in unique_sources:
            mask = source_array == source
            baseline = float(strong[mask].mean())
            retentions.append(float(routed[mask].mean() / baseline) if baseline else 1.0)
            traffic.append(float(use_strong[mask].mean()))
        mean_retention = float(np.mean(retentions))
        minimum_retention = float(np.min(retentions))
        if mean_retention >= quality_floor and minimum_retention >= quality_floor - 0.05:
            feasible.append(
                {
                    "threshold": float(threshold),
                    "strong_traffic": float(np.mean(traffic)),
                    "mean_retention": mean_retention,
                    "minimum_source_retention": minimum_retention,
                }
            )
    if not feasible:
        return {
            "threshold": float(np.nextafter(scores.min(), -math.inf)),
            "strong_traffic": 1.0,
            "mean_retention": 1.0,
            "minimum_source_retention": 1.0,
        }
    return min(
        feasible,
        key=lambda row: (
            row["strong_traffic"],
            -row["minimum_source_retention"],
            -row["mean_retention"],
        ),
    )


def _source_metrics(
    scores: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    source_names: list[str],
    threshold: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    source_array = np.asarray(source_names, dtype=object)
    for source in sorted(set(source_names)):
        mask = source_array == source
        use_strong = scores[mask] >= threshold
        routed = np.where(use_strong, strong[mask], weak[mask])
        baseline = float(strong[mask].mean())
        result[source] = {
            "tasks": int(mask.sum()),
            "weak_reward": float(weak[mask].mean()),
            "strong_reward": baseline,
            "router_reward": float(routed.mean()),
            "quality_retention": float(routed.mean() / baseline) if baseline else 1.0,
            "strong_traffic": float(use_strong.mean()),
            "uplift_spearman": _spearman(scores[mask], (strong - weak)[mask]),
        }
    return result


def _leave_source_out_metrics(
    spec: CandidateSpec,
    combined: CombinedData,
) -> dict[str, dict[str, float]]:
    """Measure whether one source's text-to-uplift relation transfers to unseen corpora."""
    source_array = np.asarray(combined.source_names, dtype=object)
    result: dict[str, dict[str, float]] = {}
    for source in sorted(set(combined.source_names)):
        heldout = np.flatnonzero(source_array == source)
        train = np.flatnonzero(source_array != source)
        transformer = _features(spec)
        train_features = np.asarray(
            transformer.fit_transform([combined.texts[index] for index in train]),
            dtype=np.float64,
        )
        heldout_features = np.asarray(
            transformer.transform([combined.texts[index] for index in heldout]),
            dtype=np.float64,
        )
        estimators = _fit_estimators(
            spec,
            train_features,
            combined.weak[train],
            combined.strong[train],
            combined.sample_weight[train],
        )
        scores = _predict_score(spec, estimators, heldout_features)
        uplift = (combined.strong - combined.weak)[heldout]
        result[source] = {
            "tasks": int(len(heldout)),
            "uplift_spearman": _spearman(scores, uplift),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        logger.info("candidate=%s leave-source-out=%s complete", spec.name, source)
    return result


def _leave_source_out_profile_metrics(
    spec: CandidateSpec,
    classifier: TaskProfileClassifier,
    features: np.ndarray,
    combined: CombinedData,
) -> dict[str, dict[str, float]]:
    """Measure task-profile uplift transfer when every outcome source is held out."""
    source_array = np.asarray(combined.source_names, dtype=object)
    result: dict[str, dict[str, float]] = {}
    for source in sorted(set(combined.source_names)):
        heldout = np.flatnonzero(source_array == source)
        train = np.flatnonzero(source_array != source)
        model = _fit_profile_uplift(
            spec,
            classifier,
            features[train],
            combined.weak[train],
            combined.strong[train],
            combined.sample_weight[train],
        )
        scores = model.predict(features[heldout])
        uplift = (combined.strong - combined.weak)[heldout]
        result[source] = {
            "tasks": int(len(heldout)),
            "uplift_spearman": _spearman(scores, uplift),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        logger.info("candidate=%s leave-source-out=%s complete", spec.name, source)
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _disjoint_profile_teacher(
    teacher: ProfileTeacherData,
    combined: CombinedData,
    *,
    minimum_rows: int = 100,
) -> ProfileTeacherData:
    """Remove exact outcome-task identity and prompt overlap from taxonomy supervision."""
    outcome_ids = set(combined.task_ids)
    outcome_texts = {
        hashlib.sha256(" ".join(text.split()).encode()).hexdigest() for text in combined.texts
    }
    keep = [
        index
        for index, task_id in enumerate(teacher.task_ids)
        if task_id not in outcome_ids
        and hashlib.sha256(" ".join(teacher.texts[index].split()).encode()).hexdigest()
        not in outcome_texts
    ]
    if len(keep) < minimum_rows:
        raise ValueError(f"fewer than {minimum_rows} disjoint task-profile teacher rows remain")
    return ProfileTeacherData(
        task_ids=[teacher.task_ids[index] for index in keep],
        groups=[teacher.groups[index] for index in keep],
        texts=[teacher.texts[index] for index in keep],
        difficulty=[teacher.difficulty[index] for index in keep],
        intent_completeness=[teacher.intent_completeness[index] for index in keep],
        pr_categories=[teacher.pr_categories[index] for index in keep],
    )


def _fit(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.source_set == "open-swe-only":
        if args.open_swe_tasks is None:
            raise ValueError("--source-set open-swe-only requires --open-swe-tasks")
        sources = [
            _load_paired_json(
                args.open_swe_tasks.resolve(),
                name="open-swe-traces-paired",
            )
        ]
    else:
        sources = [
            _load_nebius(args.nebius_tasks.resolve()),
            _load_r2e(args.r2e_root.resolve()),
        ]
        if (args.swebench_matrix is None) != (args.swebench_tasks is None):
            raise ValueError("--swebench-matrix and --swebench-tasks must be supplied together")
        if args.swebench_matrix is not None and args.swebench_tasks is not None:
            sources.append(
                _load_swebench(
                    args.swebench_matrix.resolve(),
                    args.swebench_tasks.resolve(),
                )
            )
        sources.append(_load_coderouter(args.coderouter_root.resolve()))
        if args.source_set == "legacy-plus-open-swe":
            if args.open_swe_tasks is None:
                raise ValueError("--source-set legacy-plus-open-swe requires --open-swe-tasks")
            sources.append(
                _load_paired_json(
                    args.open_swe_tasks.resolve(),
                    name="open-swe-traces-paired",
                )
            )
    combined = _combine(sources)
    raw_profile_teacher: ProfileTeacherData | None = None
    profile_teacher: ProfileTeacherData | None = None
    if args.candidate_family == "task-profile":
        if args.profile_teacher_tasks is None:
            raise ValueError("--candidate-family task-profile requires --profile-teacher-tasks")
        raw_profile_teacher = _load_profile_teacher(args.profile_teacher_tasks.resolve())
        profile_teacher = _disjoint_profile_teacher(
            raw_profile_teacher,
            combined,
        )
    _write_json(
        output / "external-sources.json",
        {
            "sources": [
                {
                    "name": source.name,
                    "tasks": len(source.task_ids),
                    "groups": len(set(source.groups)),
                    "weak_mean": float(source.weak.mean()),
                    "strong_mean": float(source.strong.mean()),
                }
                for source in sources
            ],
            "deduplicated_tasks": len(combined.task_ids),
            "deduplicated_tasks_by_source": dict(
                sorted(collections.Counter(combined.source_names).items())
            ),
            "source_weight_totals": {
                source: float(
                    combined.sample_weight[
                        np.asarray(combined.source_names, dtype=object) == source
                    ].sum()
                )
                for source in sorted(set(combined.source_names))
            },
            "deduplicated_groups": len(set(combined.groups)),
            "profile_teacher_tasks": (
                len(profile_teacher.task_ids) if profile_teacher is not None else 0
            ),
            "raw_profile_teacher_tasks": (
                len(raw_profile_teacher.task_ids) if raw_profile_teacher is not None else 0
            ),
            "profile_teacher_rows_removed_for_outcome_overlap": (
                len(raw_profile_teacher.task_ids) - len(profile_teacher.task_ids)
                if raw_profile_teacher is not None and profile_teacher is not None
                else 0
            ),
            "profile_teacher_outcome_identity_overlap": (
                len(set(profile_teacher.task_ids) & set(combined.task_ids))
                if profile_teacher is not None
                else 0
            ),
            "profile_teacher_outcome_normalized_text_overlap": (
                len(
                    {
                        hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
                        for text in profile_teacher.texts
                    }
                    & {
                        hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
                        for text in combined.texts
                    }
                )
                if profile_teacher is not None
                else 0
            ),
            "deep_swe_labels_read": False,
        },
    )
    folds = list(
        GroupKFold(n_splits=FOLDS).split(
            np.arange(len(combined.task_ids)),
            groups=np.asarray(combined.groups, dtype=object),
        )
    )
    candidate_specs = _candidate_space(args.candidate_family)
    hashing_features: dict[int, np.ndarray] = {}
    teacher_hashing_features: dict[int, np.ndarray] = {}
    profile_classifiers: dict[str, TaskProfileClassifier] = {}
    leaderboard: list[dict[str, object]] = []
    for spec in candidate_specs:
        oof = np.empty(len(combined.task_ids), dtype=np.float64)
        profile_classifier: TaskProfileClassifier | None = None
        if spec.estimator == "profile-uplift":
            if profile_teacher is None:
                raise AssertionError("task-profile candidate has no taxonomy teacher")
            teacher_features = teacher_hashing_features.get(spec.components)
            if teacher_features is None:
                teacher_features = _features(spec).fit_transform(profile_teacher.texts)
                teacher_hashing_features[spec.components] = teacher_features
            profile_classifier = _fit_profile_classifier(
                spec,
                profile_teacher,
                teacher_features,
            )
            profile_classifiers[spec.name] = profile_classifier
        for fold_index, (train, heldout) in enumerate(folds):
            if spec.label_mode == "task-blind":
                oof[heldout] = float(
                    np.average(
                        (combined.strong - combined.weak)[train],
                        weights=combined.sample_weight[train],
                    )
                )
                logger.info("candidate=%s fold=%d/%d complete", spec.name, fold_index + 1, FOLDS)
                continue
            if spec.analyzer == "hashing":
                full_hashing = hashing_features.get(spec.components)
                if full_hashing is None:
                    full_hashing = _features(spec).fit_transform(combined.texts)
                    hashing_features[spec.components] = full_hashing
                train_features = full_hashing[train]
                heldout_features = full_hashing[heldout]
            else:
                transformer = _features(spec)
                train_features = np.asarray(
                    transformer.fit_transform([combined.texts[index] for index in train]),
                    dtype=np.float64,
                )
                heldout_features = np.asarray(
                    transformer.transform([combined.texts[index] for index in heldout]),
                    dtype=np.float64,
                )
            train_weak, train_strong = _training_outcomes(
                spec,
                combined.weak[train],
                combined.strong[train],
                [combined.source_names[index] for index in train],
                seed=10_000 + fold_index,
            )
            if spec.estimator == "profile-uplift":
                if profile_classifier is None:
                    raise AssertionError(f"{spec.name} has no fitted task-profile classifier")
                profile_model = _fit_profile_uplift(
                    spec,
                    profile_classifier,
                    train_features,
                    train_weak,
                    train_strong,
                    combined.sample_weight[train],
                )
                oof[heldout] = profile_model.predict(heldout_features)
            else:
                estimators = _fit_estimators(
                    spec,
                    train_features,
                    train_weak,
                    train_strong,
                    combined.sample_weight[train],
                )
                oof[heldout] = _predict_score(spec, estimators, heldout_features)
            logger.info("candidate=%s fold=%d/%d complete", spec.name, fold_index + 1, FOLDS)
        operating_points = {
            str(floor): _operating_point(
                oof,
                combined.weak,
                combined.strong,
                combined.source_names,
                floor,
            )
            for floor in QUALITY_FLOORS
        }
        primary = operating_points[str(QUALITY_FLOORS[0])]
        source_metrics = _source_metrics(
            oof,
            combined.weak,
            combined.strong,
            combined.source_names,
            primary["threshold"],
        )
        row: dict[str, object] = {
            "candidate": dataclasses.asdict(spec),
            "external_oof_uplift_spearman": _spearman(
                oof,
                combined.strong - combined.weak,
            ),
            "operating_points": operating_points,
            "primary_source_metrics": source_metrics,
            "is_control": _is_control(spec),
            "deep_swe_labels_read": False,
        }
        if spec.name in {"word128-ridge-uplift-a10", "word128-ridge-heads-a1"}:
            row["leave_source_out_metrics"] = _leave_source_out_metrics(spec, combined)
        if spec.estimator == "profile-uplift" and profile_classifier is not None:
            row["leave_source_out_metrics"] = _leave_source_out_profile_metrics(
                spec,
                profile_classifier,
                hashing_features[spec.components],
                combined,
            )
        leaderboard.append(row)
    if args.candidate_family == "task-profile":
        control_traffic = min(
            cast(dict[str, dict[str, float]], row["operating_points"])["0.95"]["strong_traffic"]
            for row in leaderboard
            if bool(row["is_control"])
        )
        for row in leaderboard:
            primary_metrics = cast(
                dict[str, dict[str, float]],
                row["primary_source_metrics"],
            )
            leave_source_out = cast(
                dict[str, dict[str, float]],
                row.get("leave_source_out_metrics", {}),
            )
            positive_primary = sum(
                metric["uplift_spearman"] > 0.0 for metric in primary_metrics.values()
            )
            positive_leave_source_out = sum(
                metric["uplift_spearman"] > 0.0 for metric in leave_source_out.values()
            )
            strong_traffic = cast(
                dict[str, dict[str, float]],
                row["operating_points"],
            )["0.95"]["strong_traffic"]
            row["predeclared_cross_source_gate"] = {
                "aggregate_oof_uplift_positive": (float(row["external_oof_uplift_spearman"]) > 0.0),
                "primary_positive_sources": positive_primary,
                "primary_required_positive_sources": 4,
                "leave_source_out_positive_sources": positive_leave_source_out,
                "leave_source_out_required_positive_sources": 4,
                "strong_traffic": strong_traffic,
                "best_control_strong_traffic": control_traffic,
                "traffic_better_than_controls": strong_traffic < control_traffic,
                "passed": (
                    not bool(row["is_control"])
                    and float(row["external_oof_uplift_spearman"]) > 0.0
                    and positive_primary >= 4
                    and positive_leave_source_out >= 4
                    and strong_traffic < control_traffic
                ),
            }
    for row in leaderboard:
        _append_jsonl(output / "trials.jsonl", row)
    leaderboard.sort(
        key=lambda row: (
            bool(row["is_control"]),
            (
                not bool(cast(dict[str, object], row["predeclared_cross_source_gate"])["passed"])
                if args.candidate_family == "task-profile"
                else False
            ),
            cast(dict[str, dict[str, float]], row["operating_points"])["0.95"]["strong_traffic"],
            -float(row["external_oof_uplift_spearman"]),
        )
    )
    selected_name = str(cast(dict[str, object], leaderboard[0]["candidate"])["name"])
    for row in leaderboard:
        candidate = cast(dict[str, object], row["candidate"])
        row["selected_for_target_evaluation"] = candidate["name"] == selected_name
    _append_jsonl(
        output / "trials.jsonl",
        {
            "event": "external-selection",
            "candidate": selected_name,
            "target_candidate_count": 1,
            "deep_swe_labels_read": False,
        },
    )
    selected_spec = next(spec for spec in candidate_specs if spec.name == selected_name)
    transformer = _features(selected_spec)
    full_features = (
        hashing_features[selected_spec.components]
        if selected_spec.analyzer == "hashing"
        else np.asarray(transformer.fit_transform(combined.texts), dtype=np.float64)
    )
    estimators: tuple[FittedRegressor, FittedRegressor | None] | None = None
    if selected_spec.estimator == "profile-uplift":
        profile_model = _fit_profile_uplift(
            selected_spec,
            profile_classifiers[selected_spec.name],
            full_features,
            combined.weak,
            combined.strong,
            combined.sample_weight,
        )
        joblib_payload = {
            "spec": dataclasses.asdict(selected_spec),
            "transformer": None,
            "profile_model": profile_model,
        }
    else:
        estimators = _fit_estimators(
            selected_spec,
            full_features,
            combined.weak,
            combined.strong,
            combined.sample_weight,
        )
        joblib_payload = {
            "spec": dataclasses.asdict(selected_spec),
            "transformer": transformer if selected_spec.analyzer != "hashing" else None,
            "weak_estimator": estimators[0],
            "strong_estimator": estimators[1],
        }
    joblib.dump(
        joblib_payload,
        output / f"{selected_spec.name}.joblib",
        compress=3,
    )
    selected_row = next(
        row
        for row in leaderboard
        if cast(dict[str, object], row["candidate"])["name"] == selected_name
    )
    if (
        selected_spec.analyzer == "hashing"
        and selected_spec.estimator == "ridge-heads"
        and estimators is not None
    ):
        _write_json(
            output / "native-linear-heads.json",
            _native_linear_heads(
                selected_spec,
                estimators,
                cast(dict[str, dict[str, float]], selected_row["operating_points"]),
            ),
        )
    _write_json(
        output / "frozen-candidates.json",
        {
            "protocol": {
                "fit_sources": [source.name for source in sources],
                "folds": FOLDS,
                "grouping": "canonical_repository",
                "source_weighting": "equal_total_weight_per_source",
                "quality_floors": list(QUALITY_FLOORS),
                "target_outcomes_used": False,
                "target_embeddings_used": False,
                "swebench_effort_pair": [SWEBENCH_WEAK_ARM, SWEBENCH_STRONG_ARM],
                "negative_controls": ["task-blind-uplift", "within-source-shuffled-labels"],
                "leave_source_out_candidates": [
                    "word128-ridge-uplift-a10",
                    "word128-ridge-heads-a1",
                ],
                "profile_teacher_source": (
                    "disjoint-swe-rebench-v2-task-taxonomy" if profile_teacher is not None else None
                ),
                "profile_teacher_tasks": (
                    len(profile_teacher.task_ids) if profile_teacher is not None else 0
                ),
                "profile_serving": (
                    "local-hashing-centroids-plus-shrunk-uplift-table"
                    if args.candidate_family == "task-profile"
                    else None
                ),
                "predeclared_task_profile_gate": (
                    "positive aggregate OOF uplift, positive source OOF and leave-source-out "
                    "uplift on at least four of five sources, and lower strong traffic than "
                    "both negative controls"
                    if args.candidate_family == "task-profile"
                    else None
                ),
                "target_candidate_count": 1,
                "candidate_family": args.candidate_family,
                "source_set": args.source_set,
                "post_target_family_adaptation": args.candidate_family != "full",
                "selection": (
                    "passed_cross_source_gate_then_minimum_strong_traffic_then_uplift"
                    if args.candidate_family == "task-profile"
                    else "minimum_source_balanced_strong_traffic_then_uplift_spearman"
                ),
            },
            "leaderboard": leaderboard,
        },
    )
    logger.info(
        "external fit complete: tasks=%d candidates=%d leader=%s",
        len(combined.task_ids),
        len(leaderboard),
        cast(dict[str, object], leaderboard[0]["candidate"])["name"],
    )


def _load_target(matrix_path: Path, task_meta_path: Path) -> TargetData:
    matrix = _read_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError(f"{matrix_path} must contain one JSON object")
    matrix_object = {str(key): value for key, value in matrix.items()}
    outcomes = matrix_object.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError(f"{matrix_path} has no outcomes list")
    meta = _read_json(task_meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"{task_meta_path} must contain one JSON object")
    meta_object = {str(key): value for key, value in meta.items()}
    if not isinstance(meta_object.get("rows"), list):
        raise ValueError(f"{task_meta_path} has no task rows")
    task_rows = cast(list[dict[str, object]], meta_object["rows"])
    groups = {str(row["id"]): str(row["repository"]) for row in task_rows}
    texts: dict[str, str] = {}
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    for untyped in outcomes:
        if not isinstance(untyped, dict):
            continue
        row = {str(key): value for key, value in untyped.items()}
        task_id = str(row["scenario_id"])
        arm = str(row["model"])
        texts.setdefault(task_id, str(row["task"]))
        cells[(arm, task_id)] = (_as_float(row["reward"]), _as_float(row["cost_usd"]))
    arms = sorted({arm for arm, _ in cells})
    task_ids = sorted(texts)
    complete = [
        task_id
        for task_id in task_ids
        if task_id in groups and all((arm, task_id) in cells for arm in arms)
    ]
    rewards = np.asarray(
        [[cells[(arm, task_id)][0] for task_id in complete] for arm in arms],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[cells[(arm, task_id)][1] for task_id in complete] for arm in arms],
        dtype=np.float64,
    )
    return TargetData(
        task_ids=complete,
        texts=[texts[task_id] for task_id in complete],
        groups=[_canonical_group(groups[task_id]) for task_id in complete],
        arms=arms,
        rewards=rewards,
        costs=costs,
    )


def _candidate_score(path: Path, texts: list[str]) -> np.ndarray:
    fitted = cast(dict[str, object], joblib.load(path))
    raw_spec = cast(dict[str, object], fitted["spec"])
    analyzer = str(raw_spec["analyzer"])
    estimator = str(raw_spec["estimator"])
    if analyzer not in ("word", "char", "hashing", "structural"):
        raise ValueError(f"invalid frozen analyzer {analyzer!r}")
    if estimator not in (
        "ridge-uplift",
        "ridge-heads",
        "extra-heads",
        "hist-heads",
        "irt-difficulty",
        "profile-uplift",
    ):
        raise ValueError(f"invalid frozen estimator {estimator!r}")
    spec = CandidateSpec(
        name=str(raw_spec["name"]),
        analyzer=cast(Literal["word", "char", "hashing", "structural"], analyzer),
        components=int(_as_float(raw_spec["components"])),
        estimator=cast(
            Literal[
                "ridge-uplift",
                "ridge-heads",
                "extra-heads",
                "hist-heads",
                "irt-difficulty",
                "profile-uplift",
            ],
            estimator,
        ),
        alpha=_as_float(raw_spec.get("alpha", 1.0)),
        min_leaf=int(_as_float(raw_spec.get("min_leaf", 10))),
        label_mode=cast(
            Literal["observed", "shuffled", "task-blind"],
            str(raw_spec.get("label_mode", "observed")),
        ),
        profile_mode=cast(
            Literal[
                "none",
                "difficulty",
                "intent",
                "pr-category",
                "difficulty-intent",
                "difficulty-pr-category",
            ],
            str(raw_spec.get("profile_mode", "none")),
        ),
        profile_temperature=_as_float(raw_spec.get("profile_temperature", 0.1)),
        profile_prior_strength=_as_float(raw_spec.get("profile_prior_strength", 50.0)),
        min_profile_tasks=int(_as_float(raw_spec.get("min_profile_tasks", 25))),
    )
    transformer = (
        _features(spec)
        if analyzer == "hashing"
        else cast(FeatureTransformer, fitted["transformer"])
    )
    features = np.asarray(transformer.transform(texts), dtype=np.float64)
    if estimator == "profile-uplift":
        profile_model = cast(TaskProfileUpliftRegressor, fitted["profile_model"])
        return profile_model.predict(features)
    estimators = (
        cast(FittedRegressor, fitted["weak_estimator"]),
        cast(FittedRegressor | None, fitted["strong_estimator"]),
    )
    return _predict_score(spec, estimators, features)


def _ladder_indices(target: TargetData, ladder: tuple[str, ...]) -> np.ndarray:
    positions = {arm: index for index, arm in enumerate(target.arms)}
    missing = [arm for arm in ladder if arm not in positions]
    if missing:
        raise ValueError(f"DeepSWE matrix is missing frozen ladder arms: {missing}")
    return np.asarray([positions[arm] for arm in ladder], dtype=np.int64)


def _thresholds(
    operating_points: dict[str, dict[str, float]],
    ladder_size: int,
    quality_floor: str | None = None,
) -> list[float]:
    if ladder_size == 2:
        if quality_floor not in {str(value) for value in QUALITY_FLOORS}:
            raise ValueError(f"two-arm ladder requires a frozen quality floor, got {quality_floor}")
        return [operating_points[quality_floor]["threshold"]]
    if ladder_size == 3:
        if quality_floor is not None:
            raise ValueError("three-arm ladder uses the frozen tiered operating point")
        return [
            operating_points["0.99"]["threshold"],
            operating_points["0.95"]["threshold"],
        ]
    raise ValueError(f"unsupported target ladder size {ladder_size}")


def _route_indices(scores: np.ndarray, thresholds: list[float], ladder_size: int) -> np.ndarray:
    if ladder_size == 2:
        return (scores >= thresholds[0]).astype(np.int64)
    lower, upper = sorted(thresholds)
    return np.where(scores >= upper, 2, np.where(scores >= lower, 1, 0)).astype(np.int64)


def _matched_task_blind_control(
    target: TargetData,
    arm_indices: np.ndarray,
    decisions: np.ndarray,
    routed_reward: np.ndarray,
    routed_cost: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object] | None:
    """Compare a two-arm router with random task-blind assignments at identical traffic."""
    if len(arm_indices) != 2:
        return None
    tasks = len(target.task_ids)
    strong_count = int(np.sum(decisions == 1))
    strong_probability = strong_count / tasks
    weak_reward = target.rewards[arm_indices[0]]
    strong_reward = target.rewards[arm_indices[1]]
    weak_cost = target.costs[arm_indices[0]]
    strong_cost = target.costs[arm_indices[1]]
    expected_reward = float(
        np.mean((1.0 - strong_probability) * weak_reward + strong_probability * strong_reward)
    )
    expected_cost = float(
        np.sum((1.0 - strong_probability) * weak_cost + strong_probability * strong_cost)
    )

    rng = np.random.default_rng(seed)
    random_rewards = np.empty(samples, dtype=np.float64)
    random_costs = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        strong_rows = rng.choice(tasks, size=strong_count, replace=False)
        reward = weak_reward.copy()
        cost = weak_cost.copy()
        reward[strong_rows] = strong_reward[strong_rows]
        cost[strong_rows] = strong_cost[strong_rows]
        random_rewards[sample] = float(reward.mean())
        random_costs[sample] = float(cost.sum())
    router_reward = float(routed_reward.mean())
    router_cost = float(routed_cost.sum())
    return {
        "traffic_matched": {"weak": tasks - strong_count, "strong": strong_count},
        "expected_reward": expected_reward,
        "reward_95ci": [float(value) for value in np.quantile(random_rewards, [0.025, 0.975])],
        "expected_cost_usd": expected_cost,
        "cost_95ci_usd": [float(value) for value in np.quantile(random_costs, [0.025, 0.975])],
        "router_reward_delta_vs_random_mean": router_reward - expected_reward,
        "router_cost_delta_vs_random_mean_usd": router_cost - expected_cost,
        "router_quality_percentile": float(np.mean(random_rewards <= router_reward)),
        "router_cost_percentile": float(np.mean(random_costs <= router_cost)),
    }


def _bootstrap(
    router_reward: np.ndarray,
    router_cost: np.ndarray,
    baseline_reward: np.ndarray,
    baseline_cost: np.ndarray,
    groups: list[str],
) -> dict[str, list[float]]:
    rng = np.random.default_rng(73)
    unique = sorted(set(groups))
    group_array = np.asarray(groups, dtype=object)
    by_group = {group: np.flatnonzero(group_array == group) for group in unique}
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    ratios = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by_group[str(group)] for group in selected_groups])
        deltas[sample] = float(router_reward[selected].mean() - baseline_reward[selected].mean())
        router_total = float(router_cost[selected].sum())
        ratios[sample] = (
            float(baseline_cost[selected].sum() / router_total) if router_total else math.inf
        )
    return {
        "quality_delta_95ci": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "cost_ratio_95ci": [float(value) for value in np.quantile(ratios, [0.025, 0.975])],
    }


def _static_rows(target: TargetData) -> list[StaticRow]:
    return [
        {
            "arm": arm,
            "reward": float(target.rewards[index].mean()),
            "cost_usd": float(target.costs[index].sum()),
        }
        for index, arm in enumerate(target.arms)
    ]


def _promotion_decision(
    router_reward: float,
    router_cost: float,
    best_static: StaticRow,
    quality_delta_95ci: list[float],
) -> dict[str, float | bool]:
    """Apply the preregistered quality, savings, and paired-interval gates."""
    best_reward = float(best_static["reward"])
    best_cost = float(best_static["cost_usd"])
    retention = router_reward / best_reward if best_reward else 0.0
    savings = 1.0 - router_cost / best_cost if best_cost else -math.inf
    allowed_quality_delta = -0.05 * best_reward
    point_estimate_passed = retention >= 0.95 and savings >= 0.40
    paired_quality_passed = quality_delta_95ci[0] >= allowed_quality_delta
    return {
        "quality_retention": retention,
        "cost_savings": savings,
        "allowed_quality_delta": allowed_quality_delta,
        "point_estimate_passed": point_estimate_passed,
        "paired_quality_passed": paired_quality_passed,
        "passed": point_estimate_passed and paired_quality_passed,
    }


def _evaluate(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    frozen = _read_json(output / "frozen-candidates.json")
    if not isinstance(frozen, dict):
        raise ValueError("frozen-candidates.json is absent or invalid")
    frozen_object = {str(key): value for key, value in frozen.items()}
    if not isinstance(frozen_object.get("leaderboard"), list):
        raise ValueError("frozen-candidates.json is absent or invalid")
    protocol = cast(dict[str, object], frozen_object.get("protocol", {}))
    post_target_adaptation = protocol.get("post_target_family_adaptation") is True
    target = _load_target(args.deep_matrix.resolve(), args.deep_tasks.resolve())
    static = _static_rows(target)
    best_quality = max(static, key=lambda row: float(row["reward"]))
    selected_rows = [
        row
        for row in cast(list[dict[str, object]], frozen_object["leaderboard"])
        if row.get("selected_for_target_evaluation") is True
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            "frozen external fit must select exactly one candidate for target evaluation"
        )
    rows: list[dict[str, object]] = []
    for untyped in selected_rows:
        candidate = cast(dict[str, object], untyped["candidate"])
        if (
            str(candidate.get("label_mode", "observed")) != "observed"
            or untyped.get("selected_for_target_evaluation") is not True
        ):
            continue
        name = str(candidate["name"])
        scores = _candidate_score(output / f"{name}.joblib", target.texts)
        operating_points = cast(dict[str, dict[str, float]], untyped["operating_points"])
        for ladder_name, ladder in TARGET_LADDERS.items():
            operating_labels: tuple[str | None, ...] = (
                tuple(str(value) for value in QUALITY_FLOORS) if len(ladder) == 2 else (None,)
            )
            for quality_floor in operating_labels:
                arm_indices = _ladder_indices(target, ladder)
                thresholds = _thresholds(operating_points, len(ladder), quality_floor)
                decisions = _route_indices(scores, thresholds, len(ladder))
                columns = np.arange(len(target.task_ids), dtype=np.int64)
                selected_arms = arm_indices[decisions]
                routed_reward = target.rewards[selected_arms, columns]
                routed_cost = target.costs[selected_arms, columns]
                router_reward = float(routed_reward.mean())
                router_cost = float(routed_cost.sum())
                dominating = [
                    row
                    for row in static
                    if row["reward"] >= router_reward and row["cost_usd"] <= router_cost
                ]
                comparable = [row for row in static if row["reward"] >= router_reward]
                matched_static = (
                    min(comparable, key=lambda row: row["cost_usd"]) if comparable else best_quality
                )
                baseline_index = target.arms.index(str(matched_static["arm"]))
                best_static_index = target.arms.index(str(best_quality["arm"]))
                matched_bootstrap = _bootstrap(
                    routed_reward,
                    routed_cost,
                    target.rewards[baseline_index],
                    target.costs[baseline_index],
                    target.groups,
                )
                best_static_bootstrap = _bootstrap(
                    routed_reward,
                    routed_cost,
                    target.rewards[best_static_index],
                    target.costs[best_static_index],
                    target.groups,
                )
                counts = collections.Counter(int(value) for value in decisions)
                blind_seed = int.from_bytes(
                    hashlib.sha256(f"{name}:{ladder_name}:{quality_floor}".encode()).digest()[:8],
                    "big",
                )
                row: dict[str, object] = {
                    "candidate": name,
                    "ladder": ladder_name,
                    "operating_point": (
                        f"external-quality-{quality_floor}"
                        if quality_floor is not None
                        else "external-tiered-0.99-0.95"
                    ),
                    "arms": list(ladder),
                    "thresholds": thresholds,
                    "tasks": len(target.task_ids),
                    "repositories": len(set(target.groups)),
                    "router_reward": router_reward,
                    "router_cost_usd": router_cost,
                    "quality_retention_vs_best_static": (
                        router_reward / best_quality["reward"] if best_quality["reward"] else 0.0
                    ),
                    "best_static_quality_arm": best_quality,
                    "matched_static": matched_static,
                    "cost_ratio_vs_matched_static": (
                        matched_static["cost_usd"] / router_cost if router_cost else math.inf
                    ),
                    "cost_ratio_vs_best_static": (
                        best_quality["cost_usd"] / router_cost if router_cost else math.inf
                    ),
                    "cost_savings_vs_best_static": (
                        1.0 - router_cost / best_quality["cost_usd"]
                        if best_quality["cost_usd"]
                        else -math.inf
                    ),
                    "dominated_by_static": bool(dominating),
                    "dominating_static_arms": dominating,
                    "traffic": {
                        ladder[index]: counts.get(index, 0) for index in range(len(ladder))
                    },
                    "matched_task_blind_control": _matched_task_blind_control(
                        target,
                        arm_indices,
                        decisions,
                        routed_reward,
                        routed_cost,
                        seed=blind_seed,
                    ),
                    "target_labels_used_for_fit": False,
                    "target_labels_used_for_thresholds": False,
                    "target_static_aggregates_used_for_ladder_design": True,
                    "post_target_family_adaptation": post_target_adaptation,
                    "quality_delta_95ci": matched_bootstrap["quality_delta_95ci"],
                    "cost_ratio_95ci": matched_bootstrap["cost_ratio_95ci"],
                    "best_static_quality_delta_95ci": best_static_bootstrap["quality_delta_95ci"],
                    "best_static_cost_ratio_95ci": best_static_bootstrap["cost_ratio_95ci"],
                    "promotion": _promotion_decision(
                        router_reward,
                        router_cost,
                        best_quality,
                        best_static_bootstrap["quality_delta_95ci"],
                    ),
                }
                rows.append(row)
                _append_jsonl(output / "target-trials.jsonl", row)
    rows.sort(
        key=lambda row: (
            not bool(cast(dict[str, object], row["promotion"])["passed"]),
            float(row["router_cost_usd"]),
            -float(row["quality_retention_vs_best_static"]),
            bool(row["dominated_by_static"]),
        )
    )
    _write_json(
        output / "deep-swe-evaluation.json",
        {
            "dataset": "DeepSWE v1.1 published execution-scored outcomes",
            "target_tasks": len(target.task_ids),
            "target_repositories": len(set(target.groups)),
            "static_arms": static,
            "results": rows,
            "target_labels_used_for_fit": False,
            "target_labels_used_for_thresholds": False,
            "research_adaptation_note": (
                (
                    "This policy family was chosen after inspecting an earlier DeepSWE result. "
                    "Its candidate grid and thresholds were then fitted and frozen using only "
                    "external data before this evaluation, so this is an adaptive engineering "
                    "checkpoint, not untouched confirmation."
                )
                if post_target_adaptation
                else (
                    "The external candidate grid and thresholds were frozen before this phase. "
                    "The target ladders use previously known DeepSWE static aggregate results, "
                    "so this is deployment calibration rather than untouched confirmation."
                )
            ),
        },
    )
    logger.info(
        "DeepSWE evaluation complete: rows=%d leader=%s/%s reward=%.4f cost=%.2f",
        len(rows),
        rows[0]["candidate"],
        rows[0]["ladder"],
        _as_float(rows[0]["router_reward"]),
        _as_float(rows[0]["router_cost_usd"]),
    )


def _export_linear(args: argparse.Namespace) -> None:
    """Build a validated WMO linear policy from remotely fitted plain numeric heads."""
    from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
    from wmo.providers.pool import PoolEntry

    heads_path = args.heads.resolve()
    raw_heads = _read_json(heads_path)
    if not isinstance(raw_heads, dict):
        raise ValueError(f"{heads_path} must contain one JSON object")
    heads = {str(key): value for key, value in raw_heads.items()}
    if heads.get("schema") != "wmo-linear-heads-v1":
        raise ValueError("native heads use an unsupported schema")
    if (
        heads.get("target_outcomes_used") is not False
        or heads.get("target_embeddings_used") is not False
    ):
        raise ValueError("native heads do not prove an external-only fit")
    embedder = cast(dict[str, object], heads["embedder"])
    operating_points = cast(dict[str, dict[str, float]], heads["operating_points"])
    quality_floor = str(args.quality_floor)
    if quality_floor not in operating_points:
        raise ValueError(f"native heads have no quality floor {quality_floor}")

    raw_matrix = _read_json(args.matrix.resolve())
    if not isinstance(raw_matrix, dict):
        raise ValueError("the matrix must contain a pool snapshot")
    raw_pool = raw_matrix.get("pool")
    if not isinstance(raw_pool, list):
        raise ValueError("the matrix must contain a pool snapshot")
    matrix_pool = cast(list[object], raw_pool)
    requested = {args.weak_model, args.strong_model}
    pool = [
        PoolEntry.model_validate(row)
        for row in matrix_pool
        if isinstance(row, dict) and row.get("name") in requested
    ]
    if {entry.name for entry in pool} != requested:
        raise ValueError("the matrix pool does not contain both requested linear arms")

    policy = RoutingPolicy(
        kind="linear",
        default_model=args.strong_model,
        pool=pool,
        embedder=EmbedderSpec(
            kind="hashing",
            dim=int(_as_float(embedder["dim"])),
        ),
        linear_weak_model=args.weak_model,
        linear_strong_model=args.strong_model,
        linear_weak_weights=cast(list[float], heads["weak_weights"]),
        linear_strong_weights=cast(list[float], heads["strong_weights"]),
        linear_weak_bias=_as_float(heads["weak_bias"]),
        linear_strong_bias=_as_float(heads["strong_bias"]),
        linear_threshold=_as_float(operating_points[quality_floor]["threshold"]),
        fitted_from=f"external-native-heads-sha256:{_sha256_file(heads_path)}",
    )
    policy.save(args.output.resolve())
    logger.info(
        "exported linear policy weak=%s strong=%s quality_floor=%s dim=%d path=%s",
        args.weak_model,
        args.strong_model,
        quality_floor,
        policy.embedder.dim,
        args.output,
    )


def _verify_linear(args: argparse.Namespace) -> None:
    """Prove the WMO policy selects exactly the same arms as the fitted scorer."""
    from wmo.optimize.policy import RoutingPolicy, select_model

    policy_path = args.policy.resolve()
    policy = RoutingPolicy.load(policy_path)
    if policy.kind != "linear":
        raise ValueError(f"{policy_path} is not a linear policy")
    target = _load_target(args.deep_matrix.resolve(), args.deep_tasks.resolve())
    scores = _candidate_score(args.joblib.resolve(), target.texts)
    reference = (scores >= policy.linear_threshold).astype(np.int64)
    served_names = [select_model(policy, text).model for text in target.texts]
    model_indices = {
        policy.linear_weak_model: 0,
        policy.linear_strong_model: 1,
    }
    served = np.asarray(
        [model_indices.get(name, -1) for name in served_names],
        dtype=np.int64,
    )
    mismatches = np.flatnonzero(reference != served)
    _write_json(
        args.output.resolve(),
        {
            "policy_sha256": _sha256_file(policy_path),
            "heads_provenance": policy.fitted_from,
            "tasks": len(target.task_ids),
            "weak_model": policy.linear_weak_model,
            "strong_model": policy.linear_strong_model,
            "threshold": policy.linear_threshold,
            "reference_traffic": {
                "weak": int(np.sum(reference == 0)),
                "strong": int(np.sum(reference == 1)),
            },
            "served_traffic": {
                "weak": int(np.sum(served == 0)),
                "strong": int(np.sum(served == 1)),
                "unknown": int(np.sum(served == -1)),
            },
            "mismatches": int(len(mismatches)),
            "first_mismatch_task": (
                target.task_ids[int(mismatches[0])] if len(mismatches) else None
            ),
            "target_outcomes_used_for_selection": False,
            "target_text_used_for_parity_replay": True,
        },
    )
    if len(mismatches):
        raise ValueError(f"WMO linear policy disagrees on {len(mismatches)} target tasks")
    logger.info("linear policy parity passed tasks=%d", len(target.task_ids))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--nebius-tasks", type=Path, required=True)
    fit.add_argument("--r2e-root", type=Path, required=True)
    fit.add_argument("--coderouter-root", type=Path, required=True)
    fit.add_argument("--swebench-matrix", type=Path)
    fit.add_argument("--swebench-tasks", type=Path)
    fit.add_argument("--open-swe-tasks", type=Path)
    fit.add_argument("--profile-teacher-tasks", type=Path)
    fit.add_argument(
        "--source-set",
        choices=("legacy", "legacy-plus-open-swe", "open-swe-only"),
        default="legacy",
    )
    fit.add_argument(
        "--candidate-family",
        choices=("full", "native-linear", "structural-irt", "task-profile"),
        default="full",
    )
    fit.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--deep-matrix", type=Path, required=True)
    evaluate.add_argument("--deep-tasks", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    export = subparsers.add_parser("export-linear")
    export.add_argument("--heads", type=Path, required=True)
    export.add_argument("--matrix", type=Path, required=True)
    export.add_argument("--weak-model", required=True)
    export.add_argument("--strong-model", required=True)
    export.add_argument("--quality-floor", choices=("0.95", "0.97", "0.99"), required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-linear")
    verify.add_argument("--policy", type=Path, required=True)
    verify.add_argument("--joblib", type=Path, required=True)
    verify.add_argument("--deep-matrix", type=Path, required=True)
    verify.add_argument("--deep-tasks", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fit":
        _fit(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "export-linear":
        _export_linear(args)
    else:
        _verify_linear(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
