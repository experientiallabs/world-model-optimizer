"""Project-local settings stored under the selected harness root."""

from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field, ValidationError

from wmh.config.config import ARTIFACT_DIR

SETTINGS_FILENAME = "settings.toml"


class TelemetrySettings(BaseModel):
    """Usage telemetry preferences for this harness project."""

    enabled: bool = True
    anonymous_id: str | None = None


class ModelRole(BaseModel):
    """One named model role: which provider/model handles this class of work."""

    provider: str  # a ProviderKind value ("bedrock", "azure", "openai", ...)
    model: str
    region: str | None = None  # AWS Bedrock region
    endpoint: str | None = None  # Azure OpenAI / custom base URL
    deployment: str | None = None  # Azure OpenAI deployment name
    api_version: str | None = None  # Azure OpenAI API version (azure roles get a default)
    reasoning_effort: str | None = None  # structured reasoning effort, when supported


class ModelsSettings(BaseModel):
    """Role-based model defaults for this project (`.wmh/settings.toml`, `[models.<role>]`).

    Five roles keep the surface small: `worker` does quality-critical generation (scenario
    synthesis, cluster naming, agent rollouts); `judge` grades (checklist judging, inline
    validity gates) and should be a different model family from `worker` so the grader carries
    no self-preference bias toward the generator's outputs; `summary` does high-volume cheap
    extraction (trace facets/digests); `meta` is the harness-search delta proposer, which
    needs a long-context, long-output model (one proposal holds every harness surface in its
    prompt and replies with a complete replacement surface); `agent` is the agent-under-test
    whose harness `wmh optimize` searches. Set it to optimize a harness for a model
    distinct from the world model's serve provider (e.g. a small self-hosted agent against a
    frontier-served world model). Unset `judge`/`summary` fall back to `worker`; each command
    documents which explicit model flags and opt-in roles it uses.
    """

    worker: ModelRole | None = None
    judge: ModelRole | None = None
    summary: ModelRole | None = None
    meta: ModelRole | None = None
    agent: ModelRole | None = None

    def resolve(self, role: str) -> ModelRole | None:
        """The configured role, with unset `judge`/`summary` falling back to `worker`.

        `meta` and `agent` deliberately do NOT fall back to `worker`: each is picked for a
        need the scenario worker does not serve (the proposer's long-context/long-output
        surface; the agent-under-test's own identity), so when unset they return None and the
        caller keeps its own default (`wmh optimize` and closed-loop eval use the world
        model's provider for the opt-in roles when unset).
        """
        if role not in ("worker", "judge", "summary", "meta", "agent"):
            raise ValueError(
                f"unknown model role {role!r}; expected worker, judge, summary, meta, or agent"
            )
        configured: ModelRole | None = getattr(self, role)
        if role in ("meta", "agent"):
            return configured
        return configured or self.worker


class ProjectSettings(BaseModel):
    """Settings that are local to one harness project root."""

    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)


def settings_path(root: str | Path = ARTIFACT_DIR) -> Path:
    """The settings.toml path under a project artifact root."""
    return Path(root) / SETTINGS_FILENAME


def load_settings(root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    """Load a project's settings, returning defaults when no file exists.

    Raises:
        ValueError: the file is not valid TOML, or does not match the settings schema.
    """
    path = settings_path(root)
    if not path.exists():
        return ProjectSettings()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML ({exc})") from exc
    try:
        return ProjectSettings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path} does not match the current settings schema ({exc})") from exc


def save_settings(settings: ProjectSettings, root: str | Path = ARTIFACT_DIR) -> None:
    """Persist settings to settings.toml, writing to a temp file and renaming for atomicity."""
    path = settings_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json", exclude_none=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as fh:
        tomli_w.dump(data, fh)
    tmp.replace(path)


def set_telemetry_enabled(enabled: bool, root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    """Toggle telemetry opt-in and persist it, returning the updated settings."""
    settings = load_settings(root)
    settings.telemetry.enabled = enabled
    save_settings(settings, root)
    return settings


def ensure_telemetry_anonymous_id(root: str | Path = ARTIFACT_DIR) -> str:
    """Return the stable anonymous telemetry id, generating and persisting one on first use."""
    settings = load_settings(root)
    if settings.telemetry.anonymous_id is None:
        settings.telemetry.anonymous_id = uuid.uuid4().hex
        save_settings(settings, root)
    return settings.telemetry.anonymous_id
