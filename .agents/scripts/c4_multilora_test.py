"""Regression tests for the C4 multi-adapter prototype (CPU, real weights, tiny payloads).

Run explicitly (not discovered by the root pytest config, .agents is workspace):
    <venv>/bin/python -m pytest .agents/scripts/c4_multilora_test.py -q

Requires the LLMLingua-2 base in the HF cache and C1's LoRA checkpoints in the
compression data root; skips cleanly when either is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_ROOT = Path("~/Desktop/Projects/wmh-compression-data").expanduser()
ADAPTER_ROOT = DATA_ROOT / "cache/c4-adapters"

if not ADAPTER_ROOT.is_dir():  # pragma: no cover - environment guard
    pytest.skip("C1 LoRA checkpoints not present", allow_module_level=True)

from c4_multilora import (  # noqa: E402 - after the env guard
    BASE_ADAPTER,
    SERVER,
    MultiAdapterCompressor,
)

SEGMENTS = [
    "The quarterly revenue report shows total revenue increased by 12 percent year over "
    "year, driven primarily by strong performance in the enterprise segment.",
    '{"tool": "search_flights", "args": {"origin": "SFO", "destination": "JFK"}}',
    "Traceback (most recent call last): File main.py line 42 in run KeyError: 'user_id'",
]


@pytest.fixture(scope="module")
def compressor() -> MultiAdapterCompressor:
    return MultiAdapterCompressor(
        SERVER.DEFAULT_MODEL_ID,
        "cpu",
        {
            "financebench": ADAPTER_ROOT / "adapted-financebench-lora",
            "tau-bench": ADAPTER_ROOT / "adapted-tau-bench-lora",
        },
    )


def test_fingerprints_distinct_and_shaped(compressor: MultiAdapterCompressor) -> None:
    prints = compressor.adapter_fingerprints
    assert set(prints) == {"financebench", "tau-bench"}
    assert prints["financebench"] != prints["tau-bench"]
    assert all(len(value) == 16 for value in prints.values())
    assert compressor.base_fingerprint != prints["financebench"]


def test_adapters_change_output_and_are_deterministic(compressor: MultiAdapterCompressor) -> None:
    fin = compressor.compress_as("financebench", SEGMENTS, 0.5).segments
    tau = compressor.compress_as("tau-bench", SEGMENTS, 0.5).segments
    again = compressor.compress_as("financebench", SEGMENTS, 0.5).segments
    assert fin == again, "same adapter twice must be byte-identical"
    assert fin != tau, "different adapters should score differently on mixed text"


def test_base_passthrough_matches_plain_server(compressor: MultiAdapterCompressor) -> None:
    plain = SERVER.LLMLingua2FixedThreshold(SERVER.DEFAULT_MODEL_ID, "cpu")
    assert plain.model_fingerprint == compressor.base_fingerprint
    for threshold in (0.2, 0.5, 0.8):
        assert (
            compressor.compress_as(BASE_ADAPTER, SEGMENTS, threshold).segments
            == plain.compress(SEGMENTS, threshold).segments
        )


def test_grouped_and_mixed_match_switch(compressor: MultiAdapterCompressor) -> None:
    workload = [
        ("financebench", [SEGMENTS[0]], 0.5),
        ("tau-bench", [SEGMENTS[1]], 0.5),
        (BASE_ADAPTER, [SEGMENTS[2]], 0.5),
        ("tau-bench", [SEGMENTS[0]], 0.5),
    ]
    expected = [
        list(compressor.compress_as(adapter, segments, threshold).segments)
        for adapter, segments, threshold in workload
    ]
    assert compressor.compress_grouped(workload) == expected
    assert compressor.compress_mixed(workload) == expected


def test_threshold_zero_lossless_per_adapter(compressor: MultiAdapterCompressor) -> None:
    for adapter in (*compressor.adapter_names, BASE_ADAPTER):
        assert compressor.compress_as(adapter, SEGMENTS, 0.0).segments == SEGMENTS


def test_unknown_adapter_refused(compressor: MultiAdapterCompressor) -> None:
    with pytest.raises(KeyError, match="unknown adapter"):
        compressor.compress_as("someone-elses-tenant", SEGMENTS, 0.5)
    with pytest.raises(KeyError, match="unknown adapter"):
        compressor.compress_mixed([("someone-elses-tenant", SEGMENTS, 0.5)])
