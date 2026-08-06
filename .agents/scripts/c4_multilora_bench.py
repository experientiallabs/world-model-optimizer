"""C4 throughput/latency/memory curves for multi-tenant compressor serving.

Answers the product question "what does adapter #40 cost us" with measured curves
on a verified-idle GPU:

- single-model baseline re-anchored in this harness (the comparison anchor for
  kill bar KB5; C1's round 1a number is a different harness and is not reused),
- s/10k tokens and request-latency p50/p95 at concurrency {1,4,16} x resident
  adapters {1,4,16,40} under uniform round-robin tenant traffic, for SWITCH /
  GROUPED / MIXED serving modes,
- adapter-switch overhead (sticky vs alternating tenants, plus bare set_adapter),
- GPU memory per resident adapter (rung b) and per resident full checkpoint
  (rung c), with the ceiling each implies against the 40-tenant bar.

Payload: live user-segment episodes captured by C1 (one request = one episode's
segments, the shape production requests have). Results: cache/c4-bench-<device>.json
in the compression data root plus a RunRecord row in runs/c4.jsonl.

Usage (GPU box, after verifying the GPU is idle):
    python c4_multilora_bench.py --device cuda:1 --full-max 12
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import queue
import statistics
import threading
import time
from pathlib import Path

import torch
from c4_multilora import BASE_ADAPTER, SERVER, MultiAdapterCompressor

log = logging.getLogger("c4_bench")

DATA_ROOT = Path(
    os.environ.get("C4_DATA_ROOT", "~/Desktop/Projects/wmh-compression-data")
).expanduser()
ADAPTER_ROOT = DATA_ROOT / "cache/c4-adapters"
CORPORA = ("financebench", "swe-bench", "tau-bench", "terminal-tasks")


def load_requests(per_corpus: int, max_segment_chars: int) -> list[list[str]]:
    """One request per captured episode: its non-empty segments, char-capped."""
    requests: list[list[str]] = []
    for corpus in CORPORA:
        path = DATA_ROOT / f"cache/live-segments-{corpus}.jsonl"
        taken = 0
        for line in path.open():
            if taken >= per_corpus:
                break
            record = json.loads(line)
            segments = [s[:max_segment_chars] for s in record.get("segments", []) if s.strip()]
            if segments:
                requests.append(segments)
                taken += 1
    return requests


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50_ms": round(1000 * ordered[len(ordered) // 2], 2),
        "p95_ms": round(1000 * ordered[int(len(ordered) * 0.95)], 2),
        "mean_ms": round(1000 * statistics.mean(ordered), 2),
    }


def run_switch(
    compressor: MultiAdapterCompressor,
    workload: list[tuple[str, list[str]]],
    concurrency: int,
    threshold: float,
) -> dict:
    """Concurrent per-request serving with per-request adapter switching."""
    tasks: queue.SimpleQueue = queue.SimpleQueue()
    for item in workload:
        tasks.put(item)
    latencies: list[float] = []
    tokens = [0]
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                adapter, segments = tasks.get_nowait()
            except queue.Empty:
                return
            started = time.perf_counter()
            outcome = compressor.compress_as(adapter, segments, threshold)
            elapsed = time.perf_counter() - started
            with lock:
                latencies.append(elapsed)
                tokens[0] += outcome.tokens_in

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    return {
        "mode": "switch",
        "wall_s": round(wall, 3),
        "tokens_in": tokens[0],
        "s_per_10k_wall": round(wall / (tokens[0] / 10_000), 4),
        "request_latency": _percentiles(latencies),
        "n_requests": len(workload),
    }


def run_windowed(
    compressor: MultiAdapterCompressor,
    workload: list[tuple[str, list[str]]],
    window: int,
    threshold: float,
    mode: str,
    total_tokens: int,
) -> dict:
    """Queue-window batching: GROUPED or MIXED cross-request forwards.

    Single dispatcher thread because the GPU lock serializes anyway; the window is
    what stands in for "requests that arrived together". Latency is per window
    (every request in a window completes when the window does). `total_tokens` is
    the workload's precomputed token count, so nothing but serving sits inside the
    timed region.
    """
    call = compressor.compress_grouped if mode == "grouped" else compressor.compress_mixed
    latencies: list[float] = []
    started_all = time.perf_counter()
    for offset in range(0, len(workload), window):
        chunk = [
            (adapter, segments, threshold)
            for adapter, segments in workload[offset : offset + window]
        ]
        started = time.perf_counter()
        call(chunk)
        elapsed = time.perf_counter() - started
        latencies.extend([elapsed] * len(chunk))
    wall = time.perf_counter() - started_all
    return {
        "mode": mode,
        "window": window,
        "wall_s": round(wall, 3),
        "tokens_in": total_tokens,
        "s_per_10k_wall": round(wall / (total_tokens / 10_000), 4),
        "request_latency": _percentiles(latencies),
        "n_requests": len(workload),
    }


def switch_overhead(compressor: MultiAdapterCompressor, threshold: float) -> dict:
    """Sticky vs alternating tenants on identical payloads, plus bare set_adapter."""
    if len(compressor.adapter_names) < 2:
        return {"skipped": "needs 2+ resident adapters to alternate"}
    names = compressor.adapter_names[:2]
    payload = [SERVER.SELF_TEST_SEGMENTS[0]]
    for adapter in (*names, names[0]):
        compressor.compress_as(adapter, payload, threshold)

    def timed(sequence: list[str]) -> float:
        started = time.perf_counter()
        for adapter in sequence:
            compressor.compress_as(adapter, payload, threshold)
        return (time.perf_counter() - started) / len(sequence)

    sticky = timed([names[0]] * 40)
    alternating = timed([names[0], names[1]] * 20)
    bare: list[float] = []
    for index in range(100):
        target = names[index % 2]
        started = time.perf_counter()
        compressor.model.set_adapter(target)
        bare.append(time.perf_counter() - started)
    compressor._active = names[1]
    compressor.model.set_adapter(names[1])
    return {
        "sticky_ms_per_request": round(sticky * 1000, 3),
        "alternating_ms_per_request": round(alternating * 1000, 3),
        "overhead_ms_per_switch": round((alternating - sticky) * 1000, 3),
        "bare_set_adapter_ms_p50": round(1000 * sorted(bare)[50], 3),
    }


def gpu_mem_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--residency", default="1,4,16,40")
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--requests-per-corpus", type=int, default=15)
    parser.add_argument("--max-segment-chars", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--full-max", type=int, default=12, help="rung (c) resident full models")
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        raise SystemExit("curves are a GPU deliverable; run on a verified-idle CUDA device")
    torch.cuda.set_device(args.device)

    residency = [int(part) for part in args.residency.split(",")]
    concurrency = [int(part) for part in args.concurrency.split(",")]
    requests = load_requests(args.requests_per_corpus, args.max_segment_chars)
    if not requests:
        raise SystemExit(
            f"no live-segment episodes found under {DATA_ROOT}/cache/; every timing below "
            "would divide by zero. Point C4_DATA_ROOT at the compression data root."
        )
    log.info("%d requests (episodes) loaded", len(requests))
    results: dict = {
        "owner": "c4",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(),
        "server_sha256": SERVER.LOADED_SERVER_SHA256,
        "threshold": args.threshold,
        "n_requests": len(requests),
    }

    # Baseline anchor: the plain single-tenant server class in this same harness.
    torch.cuda.empty_cache()
    mem_before = gpu_mem_mb()
    plain = SERVER.LLMLingua2FixedThreshold(SERVER.DEFAULT_MODEL_ID, args.device)
    results["base_model_mem_mb"] = round(gpu_mem_mb() - mem_before, 1)
    plain.compress(SERVER.SELF_TEST_SEGMENTS, args.threshold)
    anchor_workload = [(None, segments) for segments in requests]
    latencies: list[float] = []
    tokens = 0
    started = time.perf_counter()
    for _, segments in anchor_workload:
        call_start = time.perf_counter()
        outcome = plain.compress(segments, args.threshold)
        latencies.append(time.perf_counter() - call_start)
        tokens += outcome.tokens_in
    wall = time.perf_counter() - started
    results["baseline_single_model"] = {
        "wall_s": round(wall, 3),
        "tokens_in": tokens,
        "s_per_10k_wall": round(wall / (tokens / 10_000), 4),
        "request_latency": _percentiles(latencies),
    }
    log.info("baseline: %s", results["baseline_single_model"])
    del plain
    torch.cuda.empty_cache()

    # Rung (b): adapters resident on one base.
    real_dirs = {c: ADAPTER_ROOT / f"adapted-{c}-lora" for c in CORPORA}
    curves: list[dict] = []
    mem_curve: list[dict] = []
    total_tokens = tokens
    for n_resident in residency:
        adapter_dirs = {
            f"tenant-{index:02d}": real_dirs[CORPORA[index % len(CORPORA)]]
            for index in range(n_resident)
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = gpu_mem_mb()
        compressor = MultiAdapterCompressor(SERVER.DEFAULT_MODEL_ID, args.device, adapter_dirs)
        mem_after_load = gpu_mem_mb()
        names = compressor.adapter_names
        workload = [
            (names[index % len(names)], segments) for index, segments in enumerate(requests)
        ]
        for level in concurrency:
            row = run_switch(compressor, workload, level, args.threshold)
            row.update({"resident_adapters": n_resident, "concurrency": level})
            curves.append(row)
            log.info("%s", row)
        for mode in ("grouped", "mixed"):
            for window in concurrency:
                if window == 1 and mode == "grouped":
                    continue
                try:
                    row = run_windowed(
                        compressor, workload, window, args.threshold, mode, total_tokens
                    )
                except Exception as error:  # noqa: BLE001 - a mode may be unsupported
                    row = {
                        "mode": mode,
                        "window": window,
                        "error": f"{type(error).__name__}: {error}",
                    }
                row.update({"resident_adapters": n_resident, "concurrency": window})
                curves.append(row)
                log.info("%s", row)
        peak = torch.cuda.max_memory_allocated() / 1e6
        mem_curve.append(
            {
                "resident_adapters": n_resident,
                "mem_after_load_mb": round(mem_after_load - mem_before, 1),
                "adapter_mem_mb_each": round(
                    (mem_after_load - mem_before - results["base_model_mem_mb"]) / n_resident, 2
                ),
                "peak_serving_mb": round(peak / 1, 1),
            }
        )
        if n_resident == max(residency):
            results["switch_overhead"] = switch_overhead(compressor, args.threshold)
            log.info("switch overhead: %s", results["switch_overhead"])
        del compressor
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    results["curves"] = curves
    results["memory_rung_b"] = mem_curve

    # Rung (c): N resident FULL checkpoints (each its own scorer instance).
    if not args.skip_full:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        instances = []
        deltas = []
        base_mem = gpu_mem_mb()
        try:
            for _ in range(args.full_max):
                before = gpu_mem_mb()
                instances.append(
                    SERVER.LLMLingua2FixedThreshold(SERVER.DEFAULT_MODEL_ID, args.device)
                )
                deltas.append(round(gpu_mem_mb() - before, 1))
        except torch.cuda.OutOfMemoryError:
            log.info("OOM at %d resident full models", len(instances))
        total_gb = torch.cuda.get_device_properties(args.device).total_memory / 1e9
        if not deltas:
            # Even the first full model OOMed (tenant holding the GPU): report the fact
            # instead of crashing past the already-collected curve results.
            results["memory_rung_c"] = {
                "resident_loaded": 0,
                "gpu_total_gb": round(total_gb, 1),
                "error": "first full-model load OOMed; GPU not idle enough for rung (c)",
            }
        else:
            per_model = statistics.mean(deltas[1:]) if len(deltas) > 1 else deltas[0]
            results["memory_rung_c"] = {
                "resident_loaded": len(instances),
                "mem_mb_per_model": round(per_model, 1),
                "deltas_mb": deltas,
                "gpu_total_gb": round(total_gb, 1),
                "implied_ceiling_models": int((total_gb * 1000 * 0.9 - base_mem) / per_model),
            }
        log.info("rung c memory: %s", results["memory_rung_c"])
        del instances
        torch.cuda.empty_cache()

    device_slug = args.device.replace(":", "")
    out_path = DATA_ROOT / f"cache/c4-bench-{device_slug}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("results -> %s", out_path)
    run_row = {
        "run_id": f"c4-bench-{device_slug}",
        "owner": "c4",
        "ts": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "matrix": "c4-multilora-bench",
        "variant": args.device,
        "params": {
            "residency": residency,
            "concurrency": concurrency,
            "requests_per_corpus": args.requests_per_corpus,
            "threshold": args.threshold,
            "base_adapter_name": BASE_ADAPTER,
        },
        "result": {
            "baseline_s_per_10k": results["baseline_single_model"]["s_per_10k_wall"],
            "n_curve_rows": len(curves),
        },
        "notes": f"full results in {out_path.name}",
    }
    with (DATA_ROOT / "runs/c4.jsonl").open("a") as handle:
        handle.write(json.dumps(run_row) + "\n")


if __name__ == "__main__":
    main()
