"""Tests for artifact manifest: write, verify, and tamper detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.config.manifest import ManifestMismatch, VerifyResult, verify_manifest, write_manifest


def _write(path: Path, content: str = "hello") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_write_and_verify_clean(tmp_path: Path) -> None:
    f = _write(tmp_path / "config.toml")
    write_manifest(tmp_path, [f])
    result = verify_manifest(tmp_path)
    assert result.ok
    assert result.total == 1
    assert result.passed == 1
    assert result.failed == 0
    assert result.files[0].ok
    assert result.files[0].name == "config.toml"


def test_verify_detects_tampered_file(tmp_path: Path) -> None:
    f = _write(tmp_path / "config.toml")
    write_manifest(tmp_path, [f])
    f.write_text("tampered", encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert not result.files[0].ok
    assert "mismatch" in result.files[0].detail


def test_verify_detects_missing_file(tmp_path: Path) -> None:
    f = tmp_path / "embeddings.npy"
    f.write_bytes(b"\x00\x01\x02")
    write_manifest(tmp_path, [f])
    f.unlink()
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert not result.files[0].ok
    assert "missing" in result.files[0].detail


def test_verify_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestMismatch, match="no manifest"):
        verify_manifest(tmp_path)


def test_write_manifest_is_idempotent(tmp_path: Path) -> None:
    f = _write(tmp_path / "prompt.txt", "base prompt")
    write_manifest(tmp_path, [f])
    write_manifest(tmp_path, [f])
    result = verify_manifest(tmp_path)
    assert result.ok


def test_verify_multiple_files(tmp_path: Path) -> None:
    files = [
        _write(tmp_path / "config.toml", "cfg"),
        _write(tmp_path / "metrics.json", "{}"),
    ]
    write_manifest(tmp_path, files)
    result = verify_manifest(tmp_path)
    assert result.ok
    assert result.total == 2
    assert result.passed == 2


def test_verify_partial_failure_reports_all(tmp_path: Path) -> None:
    good = _write(tmp_path / "config.toml", "cfg")
    bad = _write(tmp_path / "metrics.json", "original")
    write_manifest(tmp_path, [good, bad])
    bad.write_text("corrupted", encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert result.passed == 1
    assert result.failed == 1
    names = {f.name for f in result.files if not f.ok}
    assert names == {"metrics.json"}


def test_manifest_excludes_itself(tmp_path: Path) -> None:
    f = _write(tmp_path / "config.toml")
    write_manifest(tmp_path, [f])
    # Re-read the manifest and confirm manifest.json is not listed as a checked file.
    import json

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "manifest.json" not in raw["files"]


def test_verify_result_is_verifyresult_type(tmp_path: Path) -> None:
    f = _write(tmp_path / "config.toml")
    write_manifest(tmp_path, [f])
    result = verify_manifest(tmp_path)
    assert isinstance(result, VerifyResult)


def test_write_skips_nonexistent_files(tmp_path: Path) -> None:
    existing = _write(tmp_path / "config.toml")
    missing = tmp_path / "does_not_exist.npy"
    write_manifest(tmp_path, [existing, missing])
    result = verify_manifest(tmp_path)
    assert result.ok
    assert result.total == 1  # only the existing file was recorded
