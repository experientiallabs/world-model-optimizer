"""Soak test: bounded-rung spill turns deadline deaths into fast failovers.

Two live native engines serve the same two-rung certified pool under the same
mixed two-identity burst. The lead rung is a loopback provider that serializes
its work behind an internal capacity of one (the "box" whose internal queue
kills deep-queued requests at the request deadline in production). The
baseline engine's catalog authors NO dispatch policy: every burst request
dispatches onto the box, queues behind its capacity, and the deep ones die at
the gateway deadline, reproducing the incident shape. The spill engine's
catalog authors ``concurrency_bound=1`` on that rung: exactly one request
occupies the box and every other one ladders immediately to the healthy
fallback, so the same burst finishes with zero deadline deaths and each
spilled attempt discloses ``queue_bound`` against the bypassed preferred rung.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.models.catalog import (
    GatewayRungDispatchPolicy,
    load_model_catalog,
    write_model_catalog,
)
from exp.runtime.gateway.catalog_authority import snapshot_current_catalog
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.tests.native_waterfall_test import (
    _DRIVER_SOURCE,
    _content_chunk,
    _terminal_frames,
)

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"
# The box serves one request in 2.5s behind a capacity of one; the gateway
# deadline is 6s, so an unbounded burst of six queues four of them to death
# while a bounded rung spills them to the fallback in milliseconds.
_SERVICE_SECONDS = 2.5
_REQUEST_TIMEOUT_SECONDS = 6.0
_BURST_REQUESTS = 6


class _CapacityOneBox(BaseHTTPRequestHandler):
    """The house rung: one slot of real capacity, everyone else queues inside."""

    capacity = threading.Semaphore(1)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Serve one request after the box's single slot frees."""
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        with self.capacity:
            time.sleep(_SERVICE_SECONDS)
            try:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(_content_chunk("from-house"))
                self.wfile.write(_terminal_frames())
            except OSError:
                return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


class _FastSpillTarget(BaseHTTPRequestHandler):
    """The spill rung: answers immediately, unlimited capacity."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Stream one instant success identifying the spill rung."""
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        try:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_content_chunk("from-spill"))
            self.wfile.write(_terminal_frames())
        except OSError:
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


class _DetachedServer(ThreadingHTTPServer):
    """Loopback provider whose stuck handlers never block test teardown."""

    daemon_threads = True


@dataclass(frozen=True)
class _SoakEngine:
    """One live native serving subprocess and its access facts."""

    port: int
    raw_keys: tuple[str, str]
    database_path: Path

    @property
    def base(self) -> str:
        """Return the public gateway origin."""
        return f"http://{_HOST}:{self.port}"


def _author_lead_rung_bound(root: Path) -> tuple[Path, str]:
    """Author ``concurrency_bound=1`` on the pool's lead rung and re-snapshot.

    Mirrors the hosted platform's authoring path: the dispatch policy is
    catalog data on the deployment's gateway metadata, so opting in is a
    catalog write plus a new alias revision, never a code change.

    Returns:
        The new content-addressed snapshot path and its identity digest.
    """
    catalog_path = root / ".exp" / "models.toml"
    if not catalog_path.exists():
        catalog_path = root / "models.toml"
    catalog = load_model_catalog(catalog_path)
    lead = catalog.models["alpha"]
    assert lead.gateway is not None
    bounded = lead.model_copy(
        update={
            "gateway": lead.gateway.model_copy(
                update={"dispatch": GatewayRungDispatchPolicy(concurrency_bound=1)}
            )
        }
    )
    write_model_catalog(
        catalog_path,
        catalog.model_copy(update={"models": {**catalog.models, "alpha": bounded}}),
    )
    _catalog, normalized, snapshot = snapshot_current_catalog(root)
    return snapshot, normalized.identity_sha256()


def _serve_engine(
    root: Path,
    *,
    bounded: bool,
) -> Iterator[tuple[_SoakEngine, subprocess.Popen[str], list[_DetachedServer]]]:
    """Seed one pool root, optionally author the bound, and boot the engine."""
    house = _DetachedServer((_HOST, 0), _CapacityOneBox)
    spill = _DetachedServer((_HOST, 0), _FastSpillTarget)
    for server in (house, spill):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    manager, first_key = _configured_pool_gateway(
        root,
        base_urls=(
            f"http://{_HOST}:{house.server_address[1]}/v1",
            f"http://{_HOST}:{spill.server_address[1]}/v1",
        ),
    )
    if bounded:
        snapshot, digest = _author_lead_rung_bound(root)
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="revision-pool-bounded",
            pool_id="coding",
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=digest,
        )
    manager.create_identity(identity_id="tenant-b", display_name="Tenant B")
    manager.add_grant(identity_id="tenant-b", alias_id="coding")
    second_key = manager.issue_key(identity_id="tenant-b", key_id="key-two").raw_key
    driver = root / "native_spill_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps({"root": str(root), "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS})
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = "provider-secret-canary"
    with stderr_log.open("wb") as stderr_sink:
        process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
            [sys.executable, str(driver), config],
            stdout=subprocess.PIPE,
            stderr=stderr_sink,
            env=environment,
            text=True,
        )
        try:
            announced_ports: list[int] = []

            def _collect_announcements() -> None:
                """Record every port announcement the driver prints on stdout."""
                assert process.stdout is not None
                for line in process.stdout:
                    announced_ports.append(int(json.loads(line)["port"]))

            threading.Thread(target=_collect_announcements, daemon=True).start()
            live_deadline = time.monotonic() + 30
            port = 0
            while True:
                if announced_ports:
                    port = announced_ports[-1]
                    try:
                        live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=1.0)
                        if live.status_code == 200:
                            models = httpx.get(
                                f"http://{_HOST}:{port}/v1/models",
                                headers={"authorization": f"Bearer {first_key}"},
                                timeout=2.0,
                            )
                            if models.status_code == 200 and [
                                item["id"] for item in models.json()["data"]
                            ] == ["coding"]:
                                break
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        pass
                assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
                assert time.monotonic() < live_deadline, "native engine never became live"
                time.sleep(0.05)
            yield (
                _SoakEngine(
                    port=port,
                    raw_keys=(first_key, second_key),
                    database_path=manager.database_path,
                ),
                process,
                [house, spill],
            )
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)
            for server in (house, spill):
                server.shutdown()
                server.server_close()


@pytest.fixture(name="spill_engine")
def _spill_engine(tmp_path: Path) -> Iterator[_SoakEngine]:
    """Serve the pool with ``concurrency_bound=1`` authored on the lead rung."""
    generator = _serve_engine(tmp_path / "bounded-root", bounded=True)
    engine, _process, _servers = next(generator)
    yield engine
    for _tail in generator:
        pass


@pytest.fixture(name="baseline_engine")
def _baseline_engine(tmp_path: Path) -> Iterator[_SoakEngine]:
    """Serve the identical pool with no dispatch policy authored (today)."""
    generator = _serve_engine(tmp_path / "baseline-root", bounded=False)
    engine, _process, _servers = next(generator)
    yield engine
    for _tail in generator:
        pass


def _burst(engine: _SoakEngine) -> list[httpx.Response]:
    """Fire the mixed two-identity burst concurrently and collect responses."""

    def _one(index: int) -> httpx.Response:
        """Send one chat completion under the alternating identity's key."""
        return httpx.post(
            f"{engine.base}/v1/chat/completions",
            headers={"authorization": f"Bearer {engine.raw_keys[index % 2]}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": f"burst-{index}"}],
            },
            timeout=30.0,
        )

    with ThreadPoolExecutor(max_workers=_BURST_REQUESTS) as pool:
        return list(pool.map(_one, range(_BURST_REQUESTS)))


def test_bounded_rung_spills_the_burst_with_zero_deadline_deaths(
    spill_engine: _SoakEngine,
) -> None:
    """Every burst request completes; overflow serves off the spill rung.

    The box's one slot serves at most a couple of requests inside the
    deadline; everything else must have ladders disclosed as ``queue_bound``
    against the preferred house rung, and no attempt may die on the gateway
    deadline.
    """
    responses = _burst(spill_engine)
    assert [response.status_code for response in responses] == [200] * _BURST_REQUESTS
    contents = [response.json()["choices"][0]["message"]["content"] for response in responses]
    assert contents.count("from-spill") >= _BURST_REQUESTS - 2
    with sqlite3.connect(spill_engine.database_path) as connection:
        sheds = connection.execute(
            "SELECT dispatch_reason, preferred_deployment_id FROM gateway_attempts"
            " WHERE dispatch_reason IS NOT NULL"
        ).fetchall()
        (timeouts,) = connection.execute(
            "SELECT count(*) FROM gateway_attempts WHERE failure_class = 'timeout'"
        ).fetchone()
    assert timeouts == 0
    assert len(sheds) >= _BURST_REQUESTS - 2
    assert {reason for reason, _preferred in sheds} == {"queue_bound"}
    assert {preferred for _reason, preferred in sheds} == {"alpha"}


def test_unbounded_rung_queues_the_burst_into_deadline_deaths(
    baseline_engine: _SoakEngine,
) -> None:
    """Without the bound, the same burst dies in the box's internal queue.

    This is the incident baseline the bound exists to fix: every request
    dispatches onto the box, the box serializes them behind its single slot,
    and the deep ones exceed the gateway deadline while a healthy fallback
    rung sits idle.
    """
    responses = _burst(baseline_engine)
    statuses = [response.status_code for response in responses]
    assert any(status != 200 for status in statuses), statuses
    with sqlite3.connect(baseline_engine.database_path) as connection:
        (disclosures,) = connection.execute(
            "SELECT count(*) FROM gateway_attempts WHERE dispatch_reason IS NOT NULL"
        ).fetchone()
    assert disclosures == 0
