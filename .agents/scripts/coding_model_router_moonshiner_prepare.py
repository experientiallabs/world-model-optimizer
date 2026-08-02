"""Freeze a fast external coding corpus from Moonshiner's verified seed library.

The script is intended to run on remote experiment compute. It never reads a target
outcome. An optional DeepSWE metadata file is used only to reject exact task-id or
normalized-prompt overlap before a task can enter the external matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("coding-model-router-moonshiner-prepare")

DEFAULT_LANGUAGES = frozenset(
    {
        "asm",
        "bash",
        "c",
        "cpp",
        "js",
        "py",
        "python",
        "ts",
        "typescript",
        "zsh",
    }
)


class _WorkspaceManager(Protocol):
    def remove_workspace(self, workspace: Path) -> None: ...


@dataclass(frozen=True)
class SeedRow:
    task_id: str
    prompt: str
    language: str
    category: str
    seed_dir: Path
    verify_cmd: str
    verify_timeout_s: int
    fixture_files: int
    fixture_bytes: int


@dataclass(frozen=True)
class ValidationRow:
    task_id: str
    baseline_passed: bool | None
    reference_setup_passed: bool
    reference_passed: bool | None
    protected_intact: bool
    elapsed_s: float
    baseline_detail: str
    reference_detail: str
    error: str

    @property
    def valid(self) -> bool:
        return (
            self.baseline_passed is False
            and self.reference_setup_passed
            and self.reference_passed is True
            and self.protected_intact
            and not self.error
        )


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _fixture_shape(seed_dir: Path) -> tuple[int, int]:
    files = [path for path in (seed_dir / "files").rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _eligible_seed(
    seed_dir: Path,
    *,
    languages: frozenset[str],
    max_files: int,
    max_bytes: int,
    max_verify_timeout_s: int,
) -> SeedRow | None:
    task_path = seed_dir / "task.json"
    reference_path = seed_dir / "reference_fix.patch"
    files_dir = seed_dir / "files"
    if not (task_path.is_file() and reference_path.is_file() and files_dir.is_dir()):
        return None
    raw = _read_object(task_path)
    task_id = str(raw.get("id") or seed_dir.name)
    prompt = str(raw.get("prompt") or "")
    language = str(raw.get("lang") or raw.get("language") or "").casefold()
    verify_cmd = str(raw.get("verify_cmd") or "")
    category = str(raw.get("category") or "")
    verify_timeout = min(int(raw.get("verify_timeout") or 180), 360)
    if not task_id or not prompt or not verify_cmd or language not in languages:
        return None
    if str(raw.get("kind") or "") == "tool_behavior":
        return None
    if raw.get("interaction") or raw.get("follow_up_turns"):
        return None
    if raw.get("network"):
        return None
    if verify_timeout > max_verify_timeout_s:
        return None
    fixture_files, fixture_bytes = _fixture_shape(seed_dir)
    if fixture_files > max_files or fixture_bytes > max_bytes:
        return None
    return SeedRow(
        task_id=task_id,
        prompt=prompt,
        language=language,
        category=category,
        seed_dir=seed_dir,
        verify_cmd=verify_cmd,
        verify_timeout_s=verify_timeout,
        fixture_files=fixture_files,
        fixture_bytes=fixture_bytes,
    )


def _round_robin_candidates(rows: list[SeedRow], *, count: int, seed: int) -> list[SeedRow]:
    buckets: dict[tuple[str, str], list[SeedRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.language, row.category)].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: _stable_key(seed, row.task_id))
    keys = sorted(
        buckets,
        key=lambda key: (
            _stable_key(seed, f"{key[0]}:{key[1]}"),
            key,
        ),
    )
    selected: list[SeedRow] = []
    while keys and len(selected) < count:
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < count:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _target_metadata(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    raw = _read_object(path)
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows")
    ids: set[str] = set()
    prompts: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = row.get("id", row.get("instance_id"))
        text = row.get("text", row.get("problem_statement", row.get("prompt")))
        if isinstance(task_id, str):
            ids.add(task_id)
        if isinstance(text, str) and text.strip():
            prompts.add(_normalize_text(text))
    return ids, prompts


def _source_task_ids(path: Path) -> set[str]:
    """Read task identities from a pinned Hugging Face dataset manifest."""
    raw = _read_object(path)
    shards = raw.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"{path} has no shards")
    task_ids: set[str] = set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError(f"{path} contains an invalid shard")
        tasks = shard.get("tasks")
        if not isinstance(tasks, list) or not all(
            isinstance(task_id, str) and task_id for task_id in tasks
        ):
            raise ValueError(f"{path} contains an invalid task inventory")
        task_ids.update(tasks)
    if not task_ids:
        raise ValueError(f"{path} contains no task identities")
    return task_ids


def _configure_moonshiner(moonshiner_root: Path, state_root: Path) -> None:
    os.environ["MOONSHINER_BUNDLE_ROOT"] = str(moonshiner_root)
    os.environ["MOONSHINER_HOME"] = str(state_root)
    source = str(moonshiner_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _remove_workspace(common: _WorkspaceManager, workspace: Path) -> None:
    """Remove a Moonshiner workspace on Python versions without rmtree onexc."""
    try:
        common.remove_workspace(workspace)
        return
    except TypeError as error:
        if "onexc" not in str(error):
            raise

    def force_writable(
        function: Callable[[str], object], path: str, _error: object
    ) -> None:
        Path(path).chmod(Path(path).stat().st_mode | stat.S_IWUSR)
        function(path)

    shutil.rmtree(workspace, onerror=force_writable)


def _validate_one(row: SeedRow) -> ValidationRow:
    common = importlib.import_module("common")

    started = time.monotonic()
    workspace: Path | None = None
    baseline_detail = ""
    reference_detail = ""
    try:
        seed = _read_object(row.seed_dir / "task.json")
        seed["_dir"] = row.seed_dir
        workspace = common.materialize(seed, name=f"router-preflight-{row.task_id}")
        protected_before = common.protected_hashes(seed, workspace)
        baseline_passed, baseline_detail = common.run_verify(
            seed,
            workspace,
            timeout=row.verify_timeout_s,
        )
        patch = subprocess.run(
            ["git", "apply", str(row.seed_dir / "reference_fix.patch")],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if patch.returncode:
            return ValidationRow(
                row.task_id,
                baseline_passed,
                False,
                None,
                False,
                time.monotonic() - started,
                baseline_detail[-2_000:],
                "",
                (patch.stdout + patch.stderr)[-2_000:],
            )
        setup_passed, setup_detail = common.run_setup(seed, workspace)
        reference_passed, reference_detail = common.run_verify(
            seed,
            workspace,
            timeout=row.verify_timeout_s,
        )
        protected_intact = protected_before == common.protected_hashes(seed, workspace)
        return ValidationRow(
            row.task_id,
            baseline_passed,
            setup_passed,
            reference_passed,
            protected_intact,
            time.monotonic() - started,
            baseline_detail[-2_000:],
            f"{setup_detail[-1_000:]}\n{reference_detail[-2_000:]}",
            "",
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        return ValidationRow(
            row.task_id,
            None,
            False,
            None,
            False,
            time.monotonic() - started,
            baseline_detail[-2_000:],
            reference_detail[-2_000:],
            repr(error),
        )
    finally:
        if workspace is not None:
            _remove_workspace(common, workspace)


def _validate(rows: list[SeedRow], *, workers: int) -> list[ValidationRow]:
    results: dict[str, ValidationRow] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_validate_one, row): row.task_id for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result.task_id] = result
            logger.info(
                "validated %d/%d task=%s valid=%s elapsed_s=%.2f",
                completed,
                len(rows),
                result.task_id,
                result.valid,
                result.elapsed_s,
            )
    return [results[row.task_id] for row in rows]


def prepare(
    moonshiner_root: Path,
    output: Path,
    *,
    source_manifest: Path,
    target_tasks: Path | None,
    candidate_count: int,
    task_count: int,
    workers: int,
    seed: int,
    max_files: int,
    max_bytes: int,
    max_verify_timeout_s: int,
) -> None:
    if task_count <= 0 or candidate_count < task_count:
        raise ValueError("candidate_count must be at least task_count > 0")
    if workers <= 0:
        raise ValueError("workers must be positive")
    commit = subprocess.check_output(
        ["git", "-C", str(moonshiner_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    source_ids = _source_task_ids(source_manifest)
    target_ids, target_prompts = _target_metadata(target_tasks)
    eligible = [
        row
        for directory in sorted((moonshiner_root / "tasks/seeds").iterdir())
        if directory.is_dir()
        if (
            row := _eligible_seed(
                directory,
                languages=DEFAULT_LANGUAGES,
                max_files=max_files,
                max_bytes=max_bytes,
                max_verify_timeout_s=max_verify_timeout_s,
            )
        )
        is not None
        and row.task_id in source_ids
    ]
    identity_overlap = sorted(row.task_id for row in eligible if row.task_id in target_ids)
    text_overlap = sorted(
        row.task_id for row in eligible if _normalize_text(row.prompt) in target_prompts
    )
    if identity_overlap or text_overlap:
        raise ValueError(
            "external Moonshiner candidates overlap target metadata: "
            f"ids={identity_overlap[:10]} prompts={text_overlap[:10]}"
        )
    candidates = _round_robin_candidates(eligible, count=candidate_count, seed=seed)
    state_root = output / "moonshiner-state"
    state_root.mkdir(parents=True, exist_ok=True)
    _configure_moonshiner(moonshiner_root, state_root)
    validation = _validate(candidates, workers=workers)
    validation_by_id = {row.task_id: row for row in validation}
    selected = [
        row for row in candidates if validation_by_id[row.task_id].valid
    ][:task_count]
    if len(selected) < task_count:
        raise ValueError(
            f"only {len(selected)} of {len(candidates)} candidates passed corpus validation"
        )
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "task_id": row.task_id,
            "prompt": row.prompt,
            "language": row.language,
            "category": row.category,
            "verify_cmd": row.verify_cmd,
            "verify_timeout_s": row.verify_timeout_s,
            "fixture_files": row.fixture_files,
            "fixture_bytes": row.fixture_bytes,
            "seed_relpath": row.seed_dir.relative_to(moonshiner_root).as_posix(),
            "seed_fingerprint": _sha256_bytes(
                (row.seed_dir / "task.json").read_bytes()
                + (row.seed_dir / "reference_fix.patch").read_bytes()
            ),
        }
        for row in selected
    ]
    task_payload = {
        "protocol": "moonshiner-effort-corpus-v1",
        "source_repo": "https://github.com/greghavens/moonshiner",
        "source_commit": commit,
        "source_dataset": "greghavens/kimi-k3-coding-and-debugging-traces",
        "source_manifest_sha256": _sha256_path(source_manifest),
        "source_manifest_tasks": len(source_ids),
        "seed": seed,
        "tasks": tasks,
    }
    task_path = output / "tasks.json"
    task_path.write_text(json.dumps(task_payload, indent=2, sort_keys=True) + "\n")
    validation_path = output / "validation.jsonl"
    validation_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in validation)
    )
    manifest = {
        "protocol": "moonshiner-effort-corpus-manifest-v1",
        "source_repo": task_payload["source_repo"],
        "source_commit": commit,
        "source_dataset": task_payload["source_dataset"],
        "source_manifest_sha256": task_payload["source_manifest_sha256"],
        "source_manifest_tasks": task_payload["source_manifest_tasks"],
        "eligible_tasks": len(eligible),
        "candidate_tasks": len(candidates),
        "validated_tasks": sum(row.valid for row in validation),
        "selected_tasks": len(selected),
        "language_mix": dict(Counter(row.language for row in selected)),
        "category_mix": dict(Counter(row.category for row in selected)),
        "target_metadata_path_used": target_tasks is not None,
        "target_outcomes_used": False,
        "target_identity_overlap": 0,
        "target_normalized_prompt_overlap": 0,
        "tasks_sha256": _sha256_path(task_path),
        "validation_sha256": _sha256_path(validation_path),
        "moonshiner_state_retained": False,
    }
    shutil.rmtree(state_root)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "corpus frozen eligible=%d candidates=%d valid=%d selected=%d",
        len(eligible),
        len(candidates),
        sum(row.valid for row in validation),
        len(selected),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moonshiner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-tasks", type=Path)
    parser.add_argument("--candidate-count", type=int, default=160)
    parser.add_argument("--task-count", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    parser.add_argument("--max-verify-timeout-s", type=int, default=120)
    args = parser.parse_args()
    prepare(
        args.moonshiner_root,
        args.output,
        source_manifest=args.source_manifest,
        target_tasks=args.target_tasks,
        candidate_count=args.candidate_count,
        task_count=args.task_count,
        workers=args.workers,
        seed=args.seed,
        max_files=args.max_files,
        max_bytes=args.max_bytes,
        max_verify_timeout_s=args.max_verify_timeout_s,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
