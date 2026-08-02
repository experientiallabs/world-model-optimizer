"""Exact task selection for the harbor scorer.

Harbor's own `DatasetConfig.task_names` filter uses fnmatch semantics, so a task id containing a
glob character would silently over-match, and an optimizer's train/heldout split firewall relies
on exact selection. This module resolves a dataset once, post-filters by exact id, downloads any
remote (git/package) tasks ONCE, and returns pinned `TaskConfig`s that candidate jobs run
directly (`tasks=[...]`, `overwrite=False`): per-candidate jobs must never re-clone or clobber
the shared task cache that concurrent jobs are reading from.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from filelock import AsyncFileLock
from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig
from harbor.tasks.client import BatchDownloadResult, TaskClient, TaskIdType

_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$")
_TASK_DOWNLOAD_LOCK_TIMEOUT_S = 300.0


class HarborTaskDownloader(Protocol):
    """The task-download slice of harbor's TaskClient (fakes replace it in tests)."""

    async def download_tasks(
        self,
        task_ids: list[TaskIdType],
        overwrite: bool = False,
        output_dir: Path | None = None,
    ) -> BatchDownloadResult: ...


async def resolve_harbor_tasks(
    dataset: DatasetConfig | Path,
    task_ids: Sequence[str],
    *,
    task_client: HarborTaskDownloader | None = None,
) -> list[TaskConfig]:
    """Resolve `task_ids` from a harbor dataset (or local task dir) by exact identity.

    Args:
        dataset: A harbor `DatasetConfig`, or a local directory of task dirs (shorthand for
            `DatasetConfig(path=...)`).
        task_ids: Exact task names to select, in the order the caller wants them evaluated.
        task_client: Download seam; defaults to harbor's `TaskClient`.

    Returns:
        One pinned `TaskConfig` per requested id, in request order. Git and package tasks are
        downloaded here exactly once (git with `overwrite=True`, since harbor's cache does not
        verify that an existing checkout still matches the requested commit) and returned with
        the resolved commit pinned and `overwrite=False`, so every candidate job reuses the
        resolved local bytes instead of re-cloning or clobbering the shared cache.

    Raises:
        ValueError: On empty/duplicate ids, a dataset that resolves duplicate task names, ids
            the dataset does not contain, or a git download whose commit cannot be pinned.
    """
    requested = list(task_ids)
    if not requested or any(not task_id for task_id in requested):
        raise ValueError("task_ids must be nonempty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("task_ids must be unique")

    if isinstance(dataset, Path):
        dataset = DatasetConfig(path=dataset)
    # Resolve the dataset WITHOUT harbor's filters, then select by exact id ourselves:
    # task_names is an fnmatch pattern list, not an exact-selection API.
    resolved = DatasetConfig.model_validate(dataset.model_dump(mode="python"))
    resolved.task_names = None
    resolved.exclude_task_names = None
    resolved.n_tasks = None
    configs = await resolved.get_task_configs()

    by_id: dict[str, TaskConfig] = {}
    for config in configs:
        task_id = config.get_task_id().get_name()
        if task_id in by_id:
            raise ValueError(f"harbor dataset resolved duplicate task {task_id!r}")
        by_id[task_id] = config
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(
            f"harbor task selection was not exact: missing={missing}; "
            "check the ids against the dataset's task names"
        )

    selected = [by_id[task_id].model_copy(deep=True) for task_id in requested]
    return await _pin_remote_tasks(
        selected,
        dataset=resolved,
        task_client=task_client or TaskClient(),
    )


async def _pin_remote_tasks(
    selected: list[TaskConfig],
    *,
    dataset: DatasetConfig,
    task_client: HarborTaskDownloader,
) -> list[TaskConfig]:
    """Download remote tasks once and pin their provenance with `overwrite=False`."""
    remote_indexes = [
        index
        for index, config in enumerate(selected)
        if config.is_git_task() or config.is_package_task()
    ]
    if not remote_indexes:
        return selected
    download_root = (
        Path(dataset.download_dir)
        if dataset.download_dir is not None
        else Path(tempfile.gettempdir()) / "wmo-harbor-task-downloads"
    )
    download_root.mkdir(parents=True, exist_ok=True)
    # Harbor refreshes one shared git checkout in place. Separate scorer construction calls may
    # happen on different event loops and even in different processes, so an asyncio-only lock
    # cannot protect `git checkout` from a concurrent cache refresh. The async file lock keeps the
    # event loop responsive while serializing only this short download/pin boundary.
    lock = AsyncFileLock(
        download_root / ".wmo-task-download.lock",
        timeout=_TASK_DOWNLOAD_LOCK_TIMEOUT_S,
        mode=0o600,
    )
    async with lock:
        downloads = await task_client.download_tasks(
            [selected[index].get_task_id() for index in remote_indexes],
            # Refresh git checkouts once, here: harbor's cache does not verify that an existing
            # checkout still came from the requested commit.
            overwrite=dataset.overwrite
            or any(selected[index].is_git_task() for index in remote_indexes),
            output_dir=dataset.download_dir,
        )
    if len(downloads.results) != len(remote_indexes):
        raise ValueError("harbor returned an incomplete task download result")
    for index, download in zip(remote_indexes, downloads.results, strict=True):
        config = selected[index]
        updates: dict[str, object] = {"overwrite": False}
        if config.is_git_task():
            commit = download.resolved_git_commit_id
            if commit is None or _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
                raise ValueError(
                    "harbor did not resolve a git commit for task "
                    f"{config.get_task_id().get_name()!r}"
                )
            requested_commit = config.git_commit_id
            if requested_commit is not None and requested_commit.lower() != commit.lower():
                raise ValueError(
                    "harbor resolved a different git commit for task "
                    f"{config.get_task_id().get_name()!r}"
                )
            updates["git_commit_id"] = commit.lower()
        selected[index] = config.model_copy(update=updates)
    return selected
