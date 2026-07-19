"""Artifact manifest: SHA-256 checksums for every file a build writes.

`write_manifest` is called at the end of `_persist` (wmh.engine.build) and records the
SHA-256 digest of every artifact file into `manifest.json`. `verify_manifest` is called by
`wmh.engine.loader` on load and by `wmh verify` in the CLI — it re-hashes each listed file
and compares against the stored digest, reporting exactly which files pass, which are missing,
and which have a checksum mismatch.

The manifest file itself is excluded from its own checksums (it cannot hash itself). Files are
recorded by name relative to the artifact root so the manifest is portable across machines.
Files that do not exist at write time are silently skipped — a partial build may not have
written every optional file yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MANIFEST_FILE = "manifest.json"
_SCHEMA_VERSION = 1
_CHUNK = 65536  # 64 KiB read chunks; keeps memory flat for large index files


def _sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of `path`."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _wmh_version() -> str:
    try:
        return version("world-model-harness")
    except PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class FileResult:
    """Verification outcome for one file listed in the manifest."""

    name: str  # path relative to the artifact root
    ok: bool  # True = digest matched; False = missing or mismatch
    detail: str  # human-readable reason when ok=False, empty string when ok=True


@dataclass(frozen=True)
class VerifyResult:
    """Aggregate result of verifying one artifact directory."""

    ok: bool
    files: list[FileResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of files checked."""
        return len(self.files)

    @property
    def passed(self) -> int:
        """Number of files whose digest matched."""
        return sum(1 for f in self.files if f.ok)

    @property
    def failed(self) -> int:
        """Number of files that were missing or had a checksum mismatch."""
        return self.total - self.passed


class ManifestMismatch(Exception):
    """Raised when the manifest file is absent or structurally unreadable.

    This is a structural problem (no manifest = cannot verify), distinct from per-file
    failures (missing file, wrong digest) which are reported as `FileResult(ok=False)`
    entries inside a `VerifyResult` so callers can surface exactly what is wrong.
    """


def write_manifest(artifact_dir: str | Path, files: list[Path]) -> None:
    """Hash `files` and write `manifest.json` into `artifact_dir`.

    Files that do not exist are silently skipped. The manifest file itself is never
    included in its own checksums. Writes atomically via a temp-file rename so an
    interrupted write never leaves a truncated manifest behind.

    Args:
        artifact_dir: The model artifact directory (e.g. `.wmh/models/my-model`).
        files: Paths to hash. Relative or absolute; non-existent paths are skipped.
    """
    root = Path(artifact_dir)
    manifest_path = root / MANIFEST_FILE
    checksums: dict[str, str] = {}
    for path in files:
        if not path.exists():
            continue
        if path.resolve() == manifest_path.resolve():
            continue  # never hash the manifest itself
        # Record by name when the file is directly under root, else by relative path.
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue  # File is outside the artifact root; skip.
        checksums[str(rel)] = _sha256(path)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "wmh_version": _wmh_version(),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": checksums,
    }
    tmp = manifest_path.with_name(f"{MANIFEST_FILE}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


def verify_manifest(artifact_dir: str | Path) -> VerifyResult:
    """Re-hash every file listed in `manifest.json` and compare against stored digests.

    Raises `ManifestMismatch` when the manifest file is absent or unreadable — that is a
    structural problem distinct from per-file failures. Per-file failures (missing file,
    wrong digest) are returned as `FileResult(ok=False)` entries so callers can display
    exactly which files are affected.

    Args:
        artifact_dir: The model artifact directory to verify.

    Returns:
        A `VerifyResult` whose `ok` field is True only when every listed file passes.

    Raises:
        ManifestMismatch: When `manifest.json` is absent or cannot be parsed.
    """
    root = Path(artifact_dir)
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.exists():
        raise ManifestMismatch(
            f"no manifest found at {manifest_path}; rebuild with `wmh build` to generate one"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestMismatch(
            f"manifest at {manifest_path} is unreadable ({exc}); "
            "rebuild with `wmh build` to regenerate it"
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
        raise ManifestMismatch(
            f"manifest at {manifest_path} has an unexpected format; "
            "rebuild with `wmh build` to regenerate it"
        )
    results: list[FileResult] = []
    for name, expected in raw["files"].items():
        path = root / name
        if not path.exists():
            results.append(FileResult(name=name, ok=False, detail="file missing"))
            continue
        actual = _sha256(path)
        if actual != expected:
            results.append(FileResult(name=name, ok=False, detail="checksum mismatch"))
        else:
            results.append(FileResult(name=name, ok=True, detail=""))
    return VerifyResult(ok=all(f.ok for f in results), files=results)
