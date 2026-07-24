"""Serve-side trace access: read a model's local traces, or fetch them from the Hugging Face Hub.

The raw trace corpus (`traces.otel.jsonl`) is large and need not be committed. When it is present
locally it is used directly; otherwise, if the model's card declares a `traces_hf` source, the
backend streams it from the Hub's public resolve URL (no auth, no client-side Hub API) into the
model directory, reporting byte progress the website can poll. A local copy always supersedes the
Hub. Recorded traces are grouped by task into replayable scenarios for the Explore-traces tab.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from wmh.config.card import TracesSource
from wmh.core.types import Action
from wmh.ingest import get_adapter

TRACES_FILENAME = "traces.otel.jsonl"


def resolve_url(source: TracesSource) -> str:
    """The public Hub resolve URL for a traces source (works unauthenticated for public repos)."""
    prefix = "datasets/" if source.kind == "dataset" else ""
    return f"https://huggingface.co/{prefix}{source.repo}/resolve/{source.revision}/{source.path}"


def local_traces_path(model_dir: Path) -> Path | None:
    """The traces file for a model, if present: a downloaded copy, else the example sibling."""
    downloaded = model_dir / TRACES_FILENAME
    if downloaded.is_file():
        return downloaded
    # examples/<task>/traces.otel.jsonl sits two levels above examples/<task>/models/<name>/.
    sibling = model_dir.parent.parent / TRACES_FILENAME
    if sibling.is_file():
        return sibling
    return None


def _action_label(action: Action) -> str:
    """Format an action in the wmh-play grammar (matches the web index generator)."""
    if action.kind.value == "tool_call":
        return f"{action.name} {action.arguments}" if action.arguments else (action.name or "")
    return f"say {action.content or ''}"


class ScenarioStep(BaseModel):
    action: Action
    action_label: str
    observation: str
    is_error: bool


class TraceScenario(BaseModel):
    id: str
    label: str
    task: str | None
    steps: list[ScenarioStep]


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def scenarios_from_traces(
    path: Path, *, max_scenarios: int = 6, max_steps: int = 10
) -> list[TraceScenario]:
    """Normalize a traces.otel.jsonl into a bounded list of replayable scenarios (one per trace)."""
    traces = get_adapter("otel-genai").from_file(str(path))
    out: list[TraceScenario] = []
    for i, trace in enumerate(traces):
        steps: list[ScenarioStep] = []
        for step in trace.steps:
            if step.observation.content is None:
                continue
            steps.append(
                ScenarioStep(
                    action=step.action,
                    action_label=_clip(_action_label(step.action), 100),
                    observation=step.observation.content,
                    is_error=step.observation.is_error,
                )
            )
            if len(steps) >= max_steps:
                break
        if not steps:
            continue
        task = trace.steps[0].task if trace.steps else None
        out.append(
            TraceScenario(id=f"t{i}", label=_scenario_label(task, i), task=task, steps=steps)
        )
        if len(out) >= max_scenarios:
            break
    return out


def _scenario_label(task: str | None, index: int) -> str:
    if not task:
        return f"Scenario {index + 1}"
    try:
        parsed = json.loads(task)
        text = parsed.get("reason_for_call") or parsed.get("task_instructions") or task
    except (json.JSONDecodeError, AttributeError):
        text = task
    return _clip(text, 68)


class DownloadStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class DownloadProgress(BaseModel):
    status: DownloadStatus
    downloaded: int = 0
    total: int | None = None  # bytes, when the server reports Content-Length
    error: str | None = None


class _DownloadState:
    def __init__(self) -> None:
        self.progress = DownloadProgress(status=DownloadStatus.RUNNING)
        self.lock = threading.Lock()


class TracesDownloader:
    """Runs Hub trace downloads on background threads, one in flight per model name."""

    def __init__(
        self, *, fetch: Callable[[str, Path, Callable[[int, int | None], None]], None] | None = None
    ) -> None:
        self._fetch = fetch or _stream_to_file
        self._states: dict[str, _DownloadState] = {}
        self._lock = threading.Lock()

    def start(self, name: str, url: str, dest: Path) -> None:
        with self._lock:
            existing = self._states.get(name)
            if existing and existing.progress.status is DownloadStatus.RUNNING:
                return  # already downloading; the client just polls
            self._states[name] = _DownloadState()
        thread = threading.Thread(
            target=self._run, args=(name, url, dest), name=f"wmh-traces-{name}", daemon=True
        )
        thread.start()

    def _run(self, name: str, url: str, dest: Path) -> None:
        state = self._states[name]

        def on_progress(downloaded: int, total: int | None) -> None:
            with state.lock:
                state.progress.downloaded = downloaded
                state.progress.total = total

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            self._fetch(url, tmp, on_progress)
            tmp.replace(dest)  # atomic: a partial download never looks complete
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            tmp.unlink(missing_ok=True)
            with state.lock:
                state.progress.status = DownloadStatus.FAILED
                state.progress.error = str(exc)
            return
        with state.lock:
            state.progress.status = DownloadStatus.DONE

    def progress(self, name: str) -> DownloadProgress | None:
        state = self._states.get(name)
        if state is None:
            return None
        with state.lock:
            return state.progress.model_copy()


def _stream_to_file(url: str, dest: Path, on_progress: Callable[[int, int | None], None]) -> None:
    """Stream a public Hub file to disk in chunks, following the CDN redirect."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        raw = resp.headers.get("content-length")
        total = int(raw) if raw and raw.isdigit() else None
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                on_progress(downloaded, total)


class TracesResponse(BaseModel):
    """What the Explore-traces tab needs: local scenarios if present, else a Hub download offer."""

    source: str  # "local" | "hub" | "none"
    downloadable: bool
    scenarios: list[TraceScenario] = Field(default_factory=list)
    download: DownloadProgress | None = None
