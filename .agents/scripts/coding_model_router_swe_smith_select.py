"""Prepare and fit locked broad SWE-smith router candidates without heldout replay."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import cast

import coding_model_router_autoresearch as autoresearch
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-swe-smith-select")

EXPECTED_FREEZE_SHA256 = "96427a4e3f8db70ff661dece0459da04006869247d7f8cdefd684c82f898c939"
EXPECTED_SOURCE_SHA256 = "3ebc3cbf2ee69e6220b9751193b7c148e6f65d7dca2588df4943b9a66f722553"
QUALITY_FLOORS = (0.95, 0.97, 0.99)
INNER_FOLDS = 5


@dataclasses.dataclass(frozen=True)
class KnnConfig:
    """One frozen guarded local kNN candidate."""

    dim: int
    neighbors: int
    relative_similarity: float
    guard_z: float
    minimum_support: int

    @property
    def name(self) -> str:
        """Return a stable candidate identity."""
        return (
            f"knn-d{self.dim}-k{self.neighbors}-r{self.relative_similarity:g}"
            f"-z{self.guard_z:g}-n{self.minimum_support}"
        )


def _sha256_file(path: Path) -> str:
    """Hash one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> float:
    """Convert one JSON number to float without accepting another type."""
    if not isinstance(value, int | float):
        raise TypeError(f"expected a JSON number, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    """Convert one JSON integer without accepting another type."""
    if not isinstance(value, int):
        raise TypeError(f"expected a JSON integer, got {type(value).__name__}")
    return value


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


def _read_rows(path: Path) -> list[dict[str, object]]:
    """Load one compact paired-source JSON list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a JSON list")
    rows = [
        {str(key): value for key, value in row.items()} for row in payload if isinstance(row, dict)
    ]
    if len(rows) != len(payload):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def _prepare_splits(args: argparse.Namespace) -> None:
    """Create immutable per-seed fit and heldout files from identity-only splits."""
    source = args.source.resolve()
    freeze = args.freeze.resolve()
    if _sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("paired source digest differs from the passed oracle artifact")
    if _sha256_file(freeze) != EXPECTED_FREEZE_SHA256:
        raise ValueError("label-free freeze digest differs from the committed protocol")
    rows = _read_rows(source)
    by_id = {str(row["instance_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("paired source contains duplicate task ids")
    freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    selected = cast(dict[str, object], freeze_payload["selected_cohort"])
    frozen_tasks = cast(list[dict[str, object]], selected["tasks"])
    frozen_ids = {str(row["instance_id"]) for row in frozen_tasks}
    if set(by_id) != frozen_ids:
        raise ValueError("paired source task ids differ from the label-free freeze")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_reports: list[dict[str, object]] = []
    for raw_split in cast(list[dict[str, object]], selected["splits"]):
        seed = _as_int(raw_split["seed"])
        heldout_ids = {str(value) for value in cast(list[object], raw_split["heldout_ids"])}
        fit_rows = [row for task_id, row in by_id.items() if task_id not in heldout_ids]
        heldout_rows = [row for task_id, row in by_id.items() if task_id in heldout_ids]
        fit_rows.sort(key=lambda row: str(row["instance_id"]))
        heldout_rows.sort(key=lambda row: str(row["instance_id"]))
        fit_path = output / f"seed-{seed}-fit.json"
        heldout_path = output / f"seed-{seed}-heldout.json"
        _write_json(fit_path, fit_rows)
        _write_json(heldout_path, heldout_rows)
        seed_reports.append(
            {
                "seed": seed,
                "fit_tasks": len(fit_rows),
                "heldout_tasks": len(heldout_rows),
                "fit_sha256": _sha256_file(fit_path),
                "heldout_sha256": _sha256_file(heldout_path),
                "heldout_ids_sha256": raw_split["heldout_ids_sha256"],
            }
        )
    _write_json(
        output / "split-manifest.json",
        {
            "schema": "swe-smith-broad-source-splits-v1",
            "freeze_sha256": EXPECTED_FREEZE_SHA256,
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "cohort_sha256": selected["cohort_sha256"],
            "seeds": seed_reports,
            "target_outcomes_used": False,
        },
    )
    logger.info("source split preparation complete seeds=%d", len(seed_reports))


def _candidate_specs() -> list[autoresearch.CandidateSpec]:
    """Return the frozen non-kNN candidate union with stable de-duplication."""
    result: list[autoresearch.CandidateSpec] = []
    seen: set[str] = set()
    for family in ("native-linear", "full", "structural-irt"):
        for spec in autoresearch._candidate_space(family):
            if autoresearch._is_control(spec) or spec.name in seen:
                continue
            seen.add(spec.name)
            result.append(spec)
    return result


def _control_specs() -> list[autoresearch.CandidateSpec]:
    """Return each frozen numeric negative control exactly once."""
    result: list[autoresearch.CandidateSpec] = []
    seen: set[str] = set()
    for family in ("native-linear", "full", "structural-irt"):
        for spec in autoresearch._candidate_space(family):
            if not autoresearch._is_control(spec) or spec.name in seen:
                continue
            seen.add(spec.name)
            result.append(spec)
    return result


def _knn_configs() -> list[KnnConfig]:
    """Return the frozen 432-point guarded local kNN grid."""
    return [
        KnnConfig(dim, neighbors, relative, guard_z, support)
        for dim in (512, 2048, 8192)
        for neighbors in (8, 16, 32, 64)
        for relative in (0.90, 0.95, 0.98)
        for guard_z in (0.0, 0.5, 1.0, 1.645)
        for support in (8, 16, 32)
    ]


def _arrays(
    rows: list[dict[str, object]],
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    """Convert compact paired rows to typed fit arrays."""
    task_ids = [str(row["instance_id"]) for row in rows]
    groups = [str(row["repo"]) for row in rows]
    texts = [str(row["text"]) for row in rows]
    weak = np.asarray([_as_float(row["cheap_reward"]) for row in rows], dtype=np.float64)
    strong = np.asarray([_as_float(row["strong_reward"]) for row in rows], dtype=np.float64)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("fit source contains duplicate task ids")
    if len(set(groups)) < INNER_FOLDS:
        raise ValueError("fit source has too few repository groups")
    return task_ids, groups, texts, weak, strong


def _nonknn_oof(
    spec: autoresearch.CandidateSpec,
    texts: list[str],
    groups: list[str],
    weak: np.ndarray,
    strong: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Create repository-grouped out-of-fold scores for one numeric candidate."""
    scores = np.empty(len(texts), dtype=np.float64)
    weights = np.ones(len(texts), dtype=np.float64)
    for fold_index, (train, heldout) in enumerate(folds):
        if spec.label_mode == "task-blind":
            scores[heldout] = float(np.average((strong - weak)[train], weights=weights[train]))
            logger.info("candidate=%s inner_fold=%d/%d", spec.name, fold_index + 1, len(folds))
            continue
        transformer = autoresearch._features(spec)
        train_features = np.asarray(
            transformer.fit_transform([texts[index] for index in train]),
            dtype=np.float64,
        )
        heldout_features = np.asarray(
            transformer.transform([texts[index] for index in heldout]),
            dtype=np.float64,
        )
        train_weak = weak[train]
        train_strong = strong[train]
        if spec.label_mode == "shuffled":
            train_weak, train_strong = _shuffle_within_groups(
                train_weak,
                train_strong,
                [groups[index] for index in train],
                seed=10_000 + fold_index,
            )
        estimators = autoresearch._fit_estimators(
            spec,
            train_features,
            train_weak,
            train_strong,
            weights[train],
        )
        scores[heldout] = autoresearch._predict_score(spec, estimators, heldout_features)
        logger.info("candidate=%s inner_fold=%d/%d", spec.name, fold_index + 1, len(folds))
    return scores


def _shuffle_within_groups(
    weak: np.ndarray,
    strong: np.ndarray,
    groups: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly permute paired outcomes within each fit repository."""
    if len(weak) != len(strong) or len(weak) != len(groups):
        raise ValueError("shuffle inputs have different lengths")
    rng = np.random.default_rng(seed)
    group_array = np.asarray(groups, dtype=object)
    shuffled_weak = weak.copy()
    shuffled_strong = strong.copy()
    for group in sorted(set(groups)):
        indices = np.flatnonzero(group_array == group)
        permutation = rng.permutation(indices)
        shuffled_weak[indices] = weak[permutation]
        shuffled_strong[indices] = strong[permutation]
    return shuffled_weak, shuffled_strong


def _knn_fold_scores(
    train_features: np.ndarray,
    heldout_features: np.ndarray,
    train_uplift: np.ndarray,
    configs: list[KnnConfig],
) -> dict[str, np.ndarray]:
    """Score every kNN grid point from one shared neighbor ordering."""
    similarities = heldout_features @ train_features.T
    top_count = min(64, train_features.shape[0])
    top_indices = np.argpartition(similarities, -top_count, axis=1)[:, -top_count:]
    top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
    order = np.argsort(top_similarities, axis=1)[:, ::-1]
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_similarities = np.take_along_axis(top_similarities, order, axis=1)
    result: dict[str, np.ndarray] = {}
    for config in configs:
        values = np.full(heldout_features.shape[0], -1_000_000.0, dtype=np.float64)
        for row_index in range(heldout_features.shape[0]):
            best = float(top_similarities[row_index, 0])
            if best <= 0.0:
                continue
            candidate_sims = top_similarities[row_index, : config.neighbors]
            eligible = candidate_sims >= best * config.relative_similarity
            indices = top_indices[row_index, : config.neighbors][eligible]
            if len(indices) < config.minimum_support:
                continue
            uplift = train_uplift[indices]
            standard_error = float(np.std(uplift, ddof=1) / math.sqrt(len(uplift)))
            values[row_index] = float(np.mean(uplift)) - config.guard_z * standard_error
        result[config.name] = values
    return result


def _knn_oof(
    configs: list[KnnConfig],
    texts: list[str],
    weak: np.ndarray,
    strong: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Create repository-grouped out-of-fold scores for the complete kNN grid."""
    by_dim: dict[int, np.ndarray] = {}
    result = {config.name: np.empty(len(texts), dtype=np.float64) for config in configs}
    for fold_index, (train, heldout) in enumerate(folds):
        for dim in (512, 2048, 8192):
            features = by_dim.get(dim)
            if features is None:
                features = autoresearch.HashingFeatureTransformer(dim).fit_transform(texts)
                by_dim[dim] = features
            dim_configs = [config for config in configs if config.dim == dim]
            fold_scores = _knn_fold_scores(
                features[train],
                features[heldout],
                strong[train] - weak[train],
                dim_configs,
            )
            for config in dim_configs:
                result[config.name][heldout] = fold_scores[config.name]
        logger.info("kNN inner_fold=%d/%d", fold_index + 1, len(folds))
    return result


def _candidate_row(
    name: str,
    family: str,
    config: dict[str, object],
    scores: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    *,
    is_control: bool = False,
) -> dict[str, object]:
    """Evaluate frozen fit-only thresholds for one candidate."""
    operating_points = {
        str(floor): autoresearch._operating_point(
            scores,
            weak,
            strong,
            ["swe-smith-broad"] * len(scores),
            floor,
        )
        for floor in QUALITY_FLOORS
    }
    primary = operating_points["0.95"]
    use_strong = scores >= primary["threshold"]
    routed = np.where(use_strong, strong, weak)
    strong_reward = float(np.mean(strong))
    return {
        "name": name,
        "family": family,
        "is_control": is_control,
        "config": config,
        "operating_points": operating_points,
        "primary": {
            "threshold": primary["threshold"],
            "strong_traffic": float(np.mean(use_strong)),
            "router_reward": float(np.mean(routed)),
            "strong_reward": strong_reward,
            "retention": float(np.mean(routed) / strong_reward) if strong_reward else 1.0,
            "uplift_spearman": autoresearch._spearman(scores, strong - weak),
        },
    }


def _fit_artifact(
    winner: dict[str, object],
    texts: list[str],
    weak: np.ndarray,
    strong: np.ndarray,
    path: Path,
) -> None:
    """Fit and persist the selected fit-only candidate on the full outer-fit partition."""
    config = cast(dict[str, object], winner["config"])
    threshold = _as_float(cast(dict[str, object], winner["primary"])["threshold"])
    if winner["family"] == "knn":
        knn = KnnConfig(
            dim=_as_int(config["dim"]),
            neighbors=_as_int(config["neighbors"]),
            relative_similarity=_as_float(config["relative_similarity"]),
            guard_z=_as_float(config["guard_z"]),
            minimum_support=_as_int(config["minimum_support"]),
        )
        features = autoresearch.HashingFeatureTransformer(knn.dim).fit_transform(texts)
        payload = {
            "schema": "swe-smith-broad-knn-artifact-v1",
            "family": "knn",
            "config": dataclasses.asdict(knn),
            "threshold": threshold,
            "fit_features": features,
            "fit_uplift": strong - weak,
            "target_outcomes_used": False,
        }
    else:
        specs = {spec.name: spec for spec in _candidate_specs()}
        spec = specs[str(winner["name"])]
        transformer = autoresearch._features(spec)
        features = np.asarray(transformer.fit_transform(texts), dtype=np.float64)
        estimators = autoresearch._fit_estimators(
            spec,
            features,
            weak,
            strong,
            np.ones(len(texts), dtype=np.float64),
        )
        payload = {
            "schema": "swe-smith-broad-numeric-artifact-v1",
            "family": "numeric",
            "spec": dataclasses.asdict(spec),
            "transformer": transformer,
            "estimators": estimators,
            "threshold": threshold,
            "target_outcomes_used": False,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path, compress=3)


def _verify_fit_source(source: Path, manifest_path: Path, seed: int) -> str:
    """Bind one selection job to the exact prepared fit partition."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("split manifest is not a JSON object")
    if manifest.get("schema") != "swe-smith-broad-source-splits-v1":
        raise ValueError("unexpected split manifest schema")
    if manifest.get("freeze_sha256") != EXPECTED_FREEZE_SHA256:
        raise ValueError("split manifest freeze digest differs from the protocol")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("split manifest source digest differs from the oracle artifact")
    if manifest.get("target_outcomes_used") is not False:
        raise ValueError("split manifest does not prove target outcomes stayed sealed")
    raw_seeds = manifest.get("seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError("split manifest has no seed inventory")
    seed_rows = [row for row in raw_seeds if isinstance(row, dict) and row.get("seed") == seed]
    if len(seed_rows) != 1:
        raise ValueError(f"split manifest has {len(seed_rows)} entries for seed {seed}")
    expected = seed_rows[0].get("fit_sha256")
    actual = _sha256_file(source)
    if expected != actual:
        raise ValueError(f"seed {seed} fit source digest differs from the split manifest")
    return actual


def _select(args: argparse.Namespace) -> None:
    """Run one outer seed's fit-only nested selection and persist its winner."""
    source = args.fit_source.resolve()
    manifest_path = args.split_manifest.resolve()
    fit_source_sha256 = _verify_fit_source(source, manifest_path, args.seed)
    rows = _read_rows(source)
    task_ids, groups, texts, weak, strong = _arrays(rows)
    folds = list(
        GroupKFold(n_splits=INNER_FOLDS).split(
            np.arange(len(rows)),
            groups=np.asarray(groups, dtype=object),
        )
    )
    leaderboard: list[dict[str, object]] = []
    for spec in _candidate_specs():
        scores = _nonknn_oof(spec, texts, groups, weak, strong, folds)
        leaderboard.append(
            _candidate_row(
                spec.name,
                "numeric",
                dataclasses.asdict(spec),
                scores,
                weak,
                strong,
            )
        )
    for spec in _control_specs():
        scores = _nonknn_oof(spec, texts, groups, weak, strong, folds)
        leaderboard.append(
            _candidate_row(
                spec.name,
                "numeric-control",
                dataclasses.asdict(spec),
                scores,
                weak,
                strong,
                is_control=True,
            )
        )
    knn_configs = _knn_configs()
    knn_scores = _knn_oof(knn_configs, texts, weak, strong, folds)
    for config in knn_configs:
        leaderboard.append(
            _candidate_row(
                config.name,
                "knn",
                dataclasses.asdict(config),
                knn_scores[config.name],
                weak,
                strong,
            )
        )
    candidate_order = {row["name"]: index for index, row in enumerate(leaderboard)}
    leaderboard.sort(
        key=lambda row: (
            _as_float(cast(dict[str, object], row["primary"])["strong_traffic"]),
            -_as_float(cast(dict[str, object], row["primary"])["router_reward"]),
            candidate_order[str(row["name"])],
        )
    )
    winner = next(row for row in leaderboard if not bool(row["is_control"]))
    _fit_artifact(winner, texts, weak, strong, args.artifact.resolve())
    report = {
        "schema": "swe-smith-broad-fit-selection-v1",
        "seed": args.seed,
        "source_commit": args.source_commit,
        "selector_sha256": _sha256_file(Path(__file__).resolve()),
        "split_manifest_sha256": _sha256_file(manifest_path),
        "fit_source_sha256": fit_source_sha256,
        "fit_tasks": len(task_ids),
        "fit_repositories": len(set(groups)),
        "weak_reward": float(np.mean(weak)),
        "strong_reward": float(np.mean(strong)),
        "candidate_count": len(leaderboard),
        "control_count": sum(bool(row["is_control"]) for row in leaderboard),
        "winner": winner,
        "artifact_sha256": _sha256_file(args.artifact.resolve()),
        "leaderboard": leaderboard,
        "heldout_outcomes_used": False,
        "target_outcomes_used": False,
    }
    _write_json(args.output.resolve(), report)
    logger.info(
        "fit-only selection complete seed=%d winner=%s traffic=%.4f",
        args.seed,
        winner["name"],
        _as_float(cast(dict[str, object], winner["primary"])["strong_traffic"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-splits")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--freeze", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--fit-source", type=Path, required=True)
    select.add_argument("--split-manifest", type=Path, required=True)
    select.add_argument("--seed", type=int, choices=range(5), required=True)
    select.add_argument("--source-commit", required=True)
    select.add_argument("--artifact", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Dispatch source split preparation or one fit-only seed selection."""
    args = _parser().parse_args()
    if args.command == "prepare-splits":
        _prepare_splits(args)
    else:
        _select(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
