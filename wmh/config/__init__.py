"""Project config + the `.wmh/` artifact layout."""

from wmh.config.config import (
    ARTIFACT_DIR,
    FIDELITY_TIERS,
    PROVIDER_ENV_VARS,
    ArtifactPaths,
    FidelityTier,
    HarnessConfig,
    TierSpec,
    load_config,
    save_config,
)
from wmh.config.dotenv import load_env_file, upsert_env_var
from wmh.config.manifest import (
    MANIFEST_FILE,
    FileResult,
    ManifestMismatch,
    VerifyResult,
    verify_manifest,
    write_manifest,
)
from wmh.config.settings import (
    ProjectSettings,
    TelemetrySettings,
    ensure_telemetry_anonymous_id,
    load_settings,
    save_settings,
    set_telemetry_enabled,
    settings_path,
)
from wmh.config.store import (
    DEFAULT_MODEL_NAME,
    ModelInfo,
    WorldModelStore,
    normalize_name,
    validate_name,
)

__all__ = [
    "ARTIFACT_DIR",
    "DEFAULT_MODEL_NAME",
    "FIDELITY_TIERS",
    "MANIFEST_FILE",
    "FileResult",
    "FidelityTier",
    "ManifestMismatch",
    "TierSpec",
    "PROVIDER_ENV_VARS",
    "ArtifactPaths",
    "HarnessConfig",
    "ModelInfo",
    "ProjectSettings",
    "TelemetrySettings",
    "VerifyResult",
    "WorldModelStore",
    "ensure_telemetry_anonymous_id",
    "load_config",
    "load_env_file",
    "load_settings",
    "normalize_name",
    "save_config",
    "save_settings",
    "set_telemetry_enabled",
    "settings_path",
    "upsert_env_var",
    "validate_name",
    "verify_manifest",
    "write_manifest",
]
