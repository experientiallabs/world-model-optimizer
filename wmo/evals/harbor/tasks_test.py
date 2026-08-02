"""Tests for exact harbor task selection."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig
from harbor.tasks.client import BatchDownloadResult, TaskDownloadResult, TaskIdType

from wmo.evals.harbor.tasks import resolve_harbor_tasks


def _make_task_dir(dataset: Path, task_id: str) -> None:
    task_dir = dataset / task_id
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.19\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    (task_dir / "instruction.md").write_text(f"Complete {task_id}.\n", encoding="utf-8")
    (task_dir / "task.toml").write_text('version = "1.0"\n\n[environment]\n', encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    for task_id in ("task-a", "task-b", "task-abc"):
        _make_task_dir(root, task_id)
    return root


def test_resolves_exact_ids_in_request_order(dataset: Path) -> None:
    selected = asyncio.run(resolve_harbor_tasks(dataset, ["task-b", "task-a"]))
    assert [task.get_task_id().get_name() for task in selected] == ["task-b", "task-a"]
    assert all(isinstance(task, TaskConfig) for task in selected)


def test_glob_shaped_ids_never_over_match(dataset: Path) -> None:
    """harbor's task_names filter is fnmatch; exact post-filtering must not expand globs."""
    with pytest.raises(ValueError, match=r"missing=\['task-\*'\]"):
        asyncio.run(resolve_harbor_tasks(dataset, ["task-*"]))
    # A glob that WOULD match several tasks under fnmatch selects nothing here, while the
    # literal ids it would have matched remain individually selectable.
    selected = asyncio.run(resolve_harbor_tasks(dataset, ["task-a", "task-abc"]))
    assert [task.get_task_id().get_name() for task in selected] == ["task-a", "task-abc"]


def test_rejects_missing_duplicate_and_empty_ids(dataset: Path) -> None:
    with pytest.raises(ValueError, match="missing=\\['task-z'\\]"):
        asyncio.run(resolve_harbor_tasks(dataset, ["task-a", "task-z"]))
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(resolve_harbor_tasks(dataset, ["task-a", "task-a"]))
    with pytest.raises(ValueError, match="nonempty"):
        asyncio.run(resolve_harbor_tasks(dataset, []))


def test_dataset_filters_are_ignored(dataset: Path) -> None:
    # Preconfigured fnmatch filters on the dataset must not shadow exact selection.
    config = DatasetConfig(path=dataset, task_names=["task-b"])
    selected = asyncio.run(resolve_harbor_tasks(config, ["task-a"]))
    assert [task.get_task_id().get_name() for task in selected] == ["task-a"]


class _Downloader:
    def __init__(self, commit: str) -> None:
        self.commit = commit
        self.calls: list[tuple[list[TaskIdType], bool, Path | None]] = []

    async def download_tasks(
        self,
        task_ids: list[TaskIdType],
        overwrite: bool = False,
        output_dir: Path | None = None,
    ) -> BatchDownloadResult:
        self.calls.append((list(task_ids), overwrite, output_dir))
        return BatchDownloadResult(
            results=[
                TaskDownloadResult(
                    path=Path("/cache") / task_id.get_name(),
                    download_time_sec=0.1,
                    cached=False,
                    resolved_git_commit_id=self.commit,
                )
                for task_id in task_ids
            ],
            total_time_sec=0.1,
        )


def test_git_tasks_download_once_at_resolve_and_pin_without_overwrite(dataset: Path) -> None:
    """The fresh clone happens HERE, once; candidate jobs get overwrite=False configs so they
    reuse the resolved bytes instead of re-cloning (and clobbering concurrent jobs' cache)."""
    git_task = TaskConfig(
        path=Path("tasks/task-g"),
        git_url="https://example.com/tasks.git",
    )

    async def fake_get_task_configs(
        self: DatasetConfig,
        disable_verification: bool = False,
    ) -> list[TaskConfig]:
        del self, disable_verification
        return [git_task]

    downloader = _Downloader(commit="A" * 40)
    original = DatasetConfig.get_task_configs
    DatasetConfig.get_task_configs = fake_get_task_configs
    try:
        [pinned] = asyncio.run(
            resolve_harbor_tasks(DatasetConfig(path=dataset), ["task-g"], task_client=downloader)
        )
    finally:
        DatasetConfig.get_task_configs = original
    # One download, with the git-cache refresh, at resolve time.
    assert len(downloader.calls) == 1
    assert downloader.calls[0][1] is True
    # The pinned config never re-clones and carries the resolved commit.
    assert pinned.overwrite is False
    assert pinned.git_commit_id == "a" * 40
    assert git_task.overwrite is False  # the caller's config is never mutated
    assert git_task.git_commit_id is None


def test_concurrent_git_resolves_serialize_shared_cache_refresh(dataset: Path) -> None:
    git_task = TaskConfig(
        path=Path("tasks/task-g"),
        git_url="https://example.com/tasks.git",
    )

    async def fake_get_task_configs(
        self: DatasetConfig,
        disable_verification: bool = False,
    ) -> list[TaskConfig]:
        del self, disable_verification
        return [git_task]

    class _ConcurrentDownloader(_Downloader):
        def __init__(self) -> None:
            super().__init__(commit="A" * 40)
            self.active = 0
            self.max_active = 0

        async def download_tasks(
            self,
            task_ids: list[TaskIdType],
            overwrite: bool = False,
            output_dir: Path | None = None,
        ) -> BatchDownloadResult:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.02)
                return await super().download_tasks(task_ids, overwrite, output_dir)
            finally:
                self.active -= 1

    async def run_pair(
        config: DatasetConfig,
        downloader: _ConcurrentDownloader,
    ) -> tuple[list[TaskConfig], list[TaskConfig]]:
        first, second = await asyncio.gather(
            resolve_harbor_tasks(config, ["task-g"], task_client=downloader),
            resolve_harbor_tasks(config, ["task-g"], task_client=downloader),
        )
        return first, second

    downloader = _ConcurrentDownloader()
    config = DatasetConfig(path=dataset, download_dir=dataset / "cache")
    original = DatasetConfig.get_task_configs
    DatasetConfig.get_task_configs = fake_get_task_configs
    try:
        first, second = asyncio.run(run_pair(config, downloader))
    finally:
        DatasetConfig.get_task_configs = original

    assert first[0].git_commit_id == "a" * 40
    assert second[0].git_commit_id == "a" * 40
    assert downloader.max_active == 1


def test_local_tasks_never_touch_the_downloader(dataset: Path) -> None:
    class _ExplodingDownloader:
        async def download_tasks(
            self,
            task_ids: list[TaskIdType],
            overwrite: bool = False,
            output_dir: Path | None = None,
        ) -> BatchDownloadResult:
            raise AssertionError("local tasks must not be downloaded")

    selected = asyncio.run(
        resolve_harbor_tasks(dataset, ["task-a"], task_client=_ExplodingDownloader())
    )
    assert [task.get_task_id().get_name() for task in selected] == ["task-a"]
