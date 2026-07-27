from __future__ import annotations

from pathlib import Path

import pytest

from wmo.research.routing_corpus import (
    DEFAULT_ROUTING_DATA,
    ENV_ROUTING_DATA,
    routing_data,
    routing_data_root,
)


def test_root_defaults_to_the_checkout_artifact_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_ROUTING_DATA, raising=False)
    assert routing_data_root() == DEFAULT_ROUTING_DATA
    assert DEFAULT_ROUTING_DATA.parts[-2:] == (".wmo", "routing-data")


def test_env_override_wins_and_expands_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_ROUTING_DATA, str(tmp_path))
    assert routing_data_root() == tmp_path

    monkeypatch.setenv(ENV_ROUTING_DATA, "~/somewhere")
    assert routing_data_root() == Path.home() / "somewhere"


def test_routing_data_returns_an_existing_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_ROUTING_DATA, str(tmp_path))
    assert routing_data() == tmp_path


def test_routing_data_exits_naming_the_path_and_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing corpus must stop the run: reading zero rows silently would look like a result."""
    missing = tmp_path / "absent"
    monkeypatch.setenv(ENV_ROUTING_DATA, str(missing))
    with pytest.raises(SystemExit) as excinfo:
        routing_data()
    message = str(excinfo.value)
    assert str(missing) in message
    assert ENV_ROUTING_DATA in message


def test_root_does_not_require_the_corpus_to_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`skipif` guards resolve the path on machines with no corpus, so this must not raise."""
    missing = tmp_path / "absent"
    monkeypatch.setenv(ENV_ROUTING_DATA, str(missing))
    assert routing_data_root() == missing
