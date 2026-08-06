"""C4 prototype: many per-customer LoRA adapters on one LLMLingua-2 compressor base.

The production compressor endpoint serves ONE checkpoint per process. The product
direction is per-customer adapted scorers, so this module prototypes the middle rung
of the C4 feasibility ladder: hold the 177M base resident once, load N PEFT LoRA
adapters (a few MB each) beside it, and select an adapter per request.

The canonical scorer class is loaded from deploy/compressor-endpoint/server.py by
path (sha256 recorded), never reimplemented: chunk packing, unscored-words-KEEP,
fp32, and the serialization lock are all inherited. Three candidate serving modes:

- SWITCH: whole request on one adapter, selected with set_adapter under a selection
  lock. This runs the canonical compress() end to end (the PeftModel forwards
  through), so it inherits the production scoring math by construction and serves
  as the byte-reference for the other two modes.
- GROUPED: a queue window of requests is batched cross-request, but every forward
  sub-batch holds exactly one adapter's chunks.
- MIXED: one forward carries rows from several adapters at once via PEFT's
  per-row `adapter_names` (peft>=0.20 routes modules_to_save classifier heads
  per row too).

run_invariance_suite() is the measurement instrument for kill bars KB1-KB4 in the
track's findings/c4.md: determinism per adapter, residency isolation, base
passthrough, grouped/mixed byte-equality against SWITCH, merged-vs-unmerged
equivalence, and threshold-0 losslessness. Adapters are identified by
(base_fingerprint, adapter_fingerprint); the fingerprint recipe matches the
server's own (sha256 over sorted state-dict items).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from peft import PeftModel
from peft.utils import get_peft_model_state_dict

log = logging.getLogger("c4_multilora")

HERE = Path(__file__).resolve().parent
SERVER_CANDIDATES = (
    HERE.parents[1] / "deploy/compressor-endpoint/server.py",
    HERE / "server.py",
)

# PEFT's reserved name for "no adapter" rows in mixed batches, recognized by BOTH
# adapter code paths: the LoRA layers skip the delta for "__base__" rows
# (tuners/lora/model.py filters it from unique_adapters) and the saved classifier
# head routes them to original_module (utils/other.py,
# AuxiliaryTrainingWrapper._mixed_batch_forward's `active_adapter == "__base__"`
# branch). Reused here as the public name for stock compression so one vocabulary
# covers all serving modes; the invariance suite byte-checks base passthrough in
# every mode against a plain single-tenant server instance.
BASE_ADAPTER = "__base__"


def load_server():  # noqa: ANN201 - a module object, typed at runtime
    """Import the canonical server module by path, recording its sha256."""
    path = next((p for p in SERVER_CANDIDATES if p.is_file()), None)
    if path is None:
        raise SystemExit(f"canonical server.py not found at any of: {SERVER_CANDIDATES}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("compressor_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compressor_server"] = mod
    spec.loader.exec_module(mod)
    mod.LOADED_SERVER_PATH = str(path)
    mod.LOADED_SERVER_SHA256 = digest
    return mod


SERVER = load_server()


def _fingerprint_items(items: Sequence[tuple[str, torch.Tensor]]) -> str:
    """The server's fingerprint recipe applied to an explicit (name, tensor) list."""
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


class MultiAdapterCompressor(SERVER.LLMLingua2FixedThreshold):
    """The canonical fixed-threshold scorer with N resident LoRA adapters.

    `adapter_dirs` maps adapter name -> PEFT checkpoint directory. Names are the
    serving identity (a request selects by name, never by path); each loaded
    adapter gets a weight fingerprint so (base_fingerprint, adapter_fingerprint)
    can be the compressor_version grain.
    """

    def __init__(self, model_id: str, device: str, adapter_dirs: Mapping[str, Path | str]) -> None:
        super().__init__(model_id, device)
        if not adapter_dirs:
            raise ValueError("adapter_dirs is empty; use the plain server class for no adapters")
        self.base_fingerprint = self.model_fingerprint
        self._adapter_dirs = {name: str(path) for name, path in adapter_dirs.items()}
        names = list(adapter_dirs)
        peft_model = PeftModel.from_pretrained(
            self.model, str(adapter_dirs[names[0]]), adapter_name=names[0], is_trainable=False
        )
        for name in names[1:]:
            peft_model.load_adapter(str(adapter_dirs[name]), adapter_name=name)
        peft_model = peft_model.to(device=device)
        peft_model.eval()
        wrong = {str(p.dtype) for p in peft_model.parameters()} - {"torch.float32"}
        if wrong:
            raise RuntimeError(f"refusing to serve adapters in {wrong}; fp32 only")
        self.model = peft_model
        self.adapter_fingerprints = {
            name: _fingerprint_items(
                list(get_peft_model_state_dict(peft_model, adapter_name=name).items())
            )
            for name in names
        }
        # set_adapter mutates model state, so selection and the GPU pass must be
        # atomic together; the base class lock only covers the pass itself.
        self._select_lock = threading.RLock()
        self._active = names[0]
        peft_model.set_adapter(self._active)

    @property
    def adapter_names(self) -> list[str]:
        return list(self.adapter_fingerprints)

    def compress_as(self, adapter: str, segments: Sequence[str], threshold: float):  # noqa: ANN201
        """SWITCH mode: the canonical compress() with `adapter` active.

        `BASE_ADAPTER` serves the stock scorer (adapters disabled entirely, LoRA and
        the saved classifier head both), which is what a customer without an adapted
        scorer gets; it must be byte-identical to the single-tenant server.
        """
        with self._select_lock:
            if adapter == BASE_ADAPTER:
                with self.model.disable_adapter():
                    return self.compress(segments, threshold)
            if adapter not in self.adapter_fingerprints:
                raise KeyError(f"unknown adapter {adapter!r}; loaded: {self.adapter_names}")
            if self._active != adapter:
                self.model.set_adapter(adapter)
                self._active = adapter
            return self.compress(segments, threshold)

    def compress_grouped(
        self, requests: Sequence[tuple[str, Sequence[str], float]]
    ) -> list[list[str]]:
        """GROUPED mode: batch a window of requests, one adapter per forward group.

        Requests sharing (adapter, threshold) have their segments concatenated into
        one canonical compress() call (chunks never span segments, so concatenation
        changes batching shape only), then results are split back per request.
        """
        grouped: dict[tuple[str, float], list[int]] = defaultdict(list)
        for index, (adapter, _, threshold) in enumerate(requests):
            grouped[(adapter, threshold)].append(index)
        results: list[list[str] | None] = [None] * len(requests)
        for (adapter, threshold), indices in grouped.items():
            flat: list[str] = []
            counts: list[int] = []
            for index in indices:
                segments = list(requests[index][1])
                flat.extend(segments)
                counts.append(len(segments))
            outcome = self.compress_as(adapter, flat, threshold)
            cursor = 0
            for index, count in zip(indices, counts, strict=True):
                results[index] = outcome.segments[cursor : cursor + count]
                cursor += count
        return [r for r in results if r is not None]

    def compress_mixed(
        self, requests: Sequence[tuple[str, Sequence[str], float]]
    ) -> list[list[str]]:
        """MIXED mode: one forward stream carrying rows from several adapters.

        Reproduces the canonical scoring loop (same packing, same unscored-KEEP
        rule, same FORWARD_BATCH) but tags every chunk with its request's adapter
        and passes per-row `adapter_names` to the PEFT model. This is the one mode
        with its own scoring loop, which is why the invariance suite byte-checks it
        against SWITCH before any throughput number is allowed to matter.
        """
        flat_segments: list[str] = []
        flat_adapters: list[str] = []
        thresholds: list[float] = []
        counts: list[int] = []
        for adapter, segments, threshold in requests:
            if adapter != BASE_ADAPTER and adapter not in self.adapter_fingerprints:
                raise KeyError(f"unknown adapter {adapter!r}; loaded: {self.adapter_names}")
            segments = list(segments)
            flat_segments.extend(segments)
            flat_adapters.extend([adapter] * len(segments))
            thresholds.extend([threshold] * len(segments))
            counts.append(len(segments))

        with self._select_lock, self._lock:
            words_per_segment = [SERVER.WORD_RE.findall(s) for s in flat_segments]
            leading = [s[: len(s) - len(s.lstrip())] for s in flat_segments]
            flat_lengths = self._subword_lengths(
                [word for words in words_per_segment for word in words]
            )
            lengths: list[list[int]] = []
            cursor = 0
            for words in words_per_segment:
                lengths.append(flat_lengths[cursor : cursor + len(words)])
                cursor += len(words)

            chunks: list[tuple[int, int, int]] = []
            for index, words in enumerate(words_per_segment):
                chunks.extend(
                    (index, start, end) for start, end in SERVER._pack_chunks(words, lengths[index])
                )
            probabilities = [[1.0] * len(words) for words in words_per_segment]
            for offset in range(0, len(chunks), SERVER.FORWARD_BATCH):
                batch = chunks[offset : offset + SERVER.FORWARD_BATCH]
                encoded = self.tokenizer(
                    [
                        [w.strip() for w in words_per_segment[index][start:end]]
                        for index, start, end in batch
                    ],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                row_adapters = [flat_adapters[index] for index, _, _ in batch]
                with torch.no_grad():
                    logits = self.model(**encoded, adapter_names=row_adapters).logits
                keep = torch.softmax(logits.float(), dim=-1)[:, :, self.keep_index].cpu()
                for row, (index, start, end) in enumerate(batch):
                    sums = [0.0] * (end - start)
                    counts_row = [0] * (end - start)
                    for position, word_index in enumerate(encoded.word_ids(row)):
                        if word_index is None:
                            continue
                        sums[word_index] += float(keep[row, position])
                        counts_row[word_index] += 1
                    for word in range(end - start):
                        if counts_row[word]:
                            probabilities[index][start + word] = sums[word] / counts_row[word]

            out: list[str] = []
            for index, words in enumerate(words_per_segment):
                out.append(
                    leading[index]
                    + "".join(
                        word
                        for word, p in zip(words, probabilities[index], strict=True)
                        if p >= thresholds[index]
                    )
                )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

        results: list[list[str]] = []
        cursor = 0
        for count in counts:
            results.append(out[cursor : cursor + count])
            cursor += count
        return results


def merged_copy(model_id: str, device: str, adapter_dir: Path | str, out_dir: Path):  # noqa: ANN201
    """Materialize base+adapter as a single merged checkpoint and load it canonically.

    This is the representation the canonical eval measured (merge_lora.py); live
    multi-tenant serving keeps adapters unmerged, so KB2's sub-check compares the
    two byte-for-byte through compress().
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    base = AutoModelForTokenClassification.from_pretrained(model_id, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    return SERVER.LLMLingua2FixedThreshold(str(out_dir), device)


def run_invariance_suite(
    compressor: MultiAdapterCompressor,
    segments: Sequence[str],
    thresholds: Sequence[float] = (0.2, 0.5, 0.8),
    merged_thresholds: Sequence[float] | None = None,
    merged_workdir: Path | None = None,
    single_adapter_reference: bool = True,
) -> dict:
    """Run the KB1-KB3 invariance matrix and return JSON-able verdicts.

    SWITCH outputs are the reference everywhere. Verdicts are all-or-nothing byte
    comparisons; any mismatch is reported with the adapter, threshold, and segment
    index that produced it (first few only, enough to reproduce).
    """
    adapters = [*compressor.adapter_names, BASE_ADAPTER]
    verdicts: dict = {
        "server_sha256": SERVER.LOADED_SERVER_SHA256,
        "device": compressor.device,
        "base_fingerprint": compressor.base_fingerprint,
        "adapter_fingerprints": compressor.adapter_fingerprints,
        "n_segments": len(segments),
        "thresholds": list(thresholds),
    }
    mismatches: list[dict] = []

    def note(kind: str, **detail) -> None:  # noqa: ANN003
        if len(mismatches) < 20:
            mismatches.append({"kind": kind, **detail})

    reference: dict[tuple[str, float], list[str]] = {}
    for adapter in adapters:
        for threshold in thresholds:
            reference[(adapter, threshold)] = list(
                compressor.compress_as(adapter, segments, threshold).segments
            )

    # KB1: determinism, three passes per (adapter, threshold).
    determinism = True
    for adapter in adapters:
        for threshold in thresholds:
            for _ in range(2):
                again = compressor.compress_as(adapter, segments, threshold).segments
                if list(again) != reference[(adapter, threshold)]:
                    determinism = False
                    note("determinism", adapter=adapter, threshold=threshold)
    verdicts["kb1_determinism"] = determinism

    # KB1b: adapter switch cross-talk, A -> B -> A returns A's bytes.
    crosstalk_ok = True
    first, second = adapters[0], adapters[1]
    for threshold in thresholds:
        compressor.compress_as(second, segments, threshold)
        back = compressor.compress_as(first, segments, threshold).segments
        if list(back) != reference[(first, threshold)]:
            crosstalk_ok = False
            note("switch_crosstalk", adapter=first, threshold=threshold)
    verdicts["kb1_switch_crosstalk_free"] = crosstalk_ok

    # KB2a: base passthrough, stock rows unaffected by resident adapters.
    plain = SERVER.LLMLingua2FixedThreshold(compressor.model_id, compressor.device)
    base_ok = plain.model_fingerprint == compressor.base_fingerprint
    if not base_ok:
        note("base_fingerprint", plain=plain.model_fingerprint, multi=compressor.base_fingerprint)
    for threshold in thresholds:
        expected = plain.compress(segments, threshold).segments
        if list(expected) != reference[(BASE_ADAPTER, threshold)]:
            base_ok = False
            note("base_passthrough", threshold=threshold)
    verdicts["kb2_base_passthrough"] = base_ok

    # KB2b: residency isolation, adapter A alone vs A with the full roster loaded.
    residency_ok = True
    if single_adapter_reference:
        solo_name = compressor.adapter_names[0]
        solo = MultiAdapterCompressor(
            compressor.model_id,
            compressor.device,
            {solo_name: _adapter_dir_of(compressor, solo_name)},
        )
        for threshold in thresholds:
            expected = solo.compress_as(solo_name, segments, threshold).segments
            if list(expected) != reference[(solo_name, threshold)]:
                residency_ok = False
                note("residency", adapter=solo_name, threshold=threshold)
        del solo
    verdicts["kb2_residency_isolation"] = residency_ok

    # KB2c: grouped and mixed against SWITCH on an interleaved workload.
    workload: list[tuple[str, list[str], float]] = []
    for index, segment in enumerate(segments):
        workload.append((adapters[index % len(adapters)], [segment], 0.5))
    switch_expected = [
        list(compressor.compress_as(adapter, segs, threshold).segments)
        for adapter, segs, threshold in workload
    ]
    grouped = compressor.compress_grouped(workload)
    verdicts["kb2_grouped_vs_switch"] = grouped == switch_expected
    if grouped != switch_expected:
        for index, (got, want) in enumerate(zip(grouped, switch_expected, strict=True)):
            if got != want:
                note("grouped", request=index, adapter=workload[index][0])
    try:
        mixed = compressor.compress_mixed(workload)
        verdicts["kb2_mixed_vs_switch"] = mixed == switch_expected
        if mixed != switch_expected:
            for index, (got, want) in enumerate(zip(mixed, switch_expected, strict=True)):
                if got != want:
                    note("mixed", request=index, adapter=workload[index][0])
    except Exception as error:  # noqa: BLE001 - the verdict IS whether this raises
        verdicts["kb2_mixed_vs_switch"] = f"UNSUPPORTED: {type(error).__name__}: {error}"

    # KB2d: merged-vs-unmerged representation equivalence, fine threshold sweep.
    if merged_workdir is not None:
        sweep = list(merged_thresholds or [round(0.05 * k, 2) for k in range(1, 20)])
        merged_results: dict[str, dict] = {}
        for name in compressor.adapter_names:
            merged = merged_copy(
                compressor.model_id,
                compressor.device,
                _adapter_dir_of(compressor, name),
                merged_workdir / f"merged-{name}",
            )
            flips = [
                threshold
                for threshold in sweep
                if list(merged.compress(segments, threshold).segments)
                != list(compressor.compress_as(name, segments, threshold).segments)
            ]
            merged_results[name] = {
                "thresholds_swept": len(sweep),
                "byte_equal_everywhere": not flips,
                "flip_thresholds": flips,
                "merged_fingerprint": merged.model_fingerprint,
            }
            del merged
        verdicts["kb2_merged_vs_unmerged"] = merged_results

    # KB3: threshold-0 losslessness per adapter.
    lossless = True
    for adapter in adapters:
        if list(compressor.compress_as(adapter, segments, 0.0).segments) != list(segments):
            lossless = False
            note("threshold0", adapter=adapter)
    verdicts["kb3_threshold0_lossless"] = lossless

    # KB4: the per-adapter startup self-test, timed.
    self_test_seconds: dict[str, float] = {}
    for adapter in adapters:
        started = time.perf_counter()
        base_output = compressor.compress_as(adapter, SERVER.SELF_TEST_SEGMENTS, 0.5).segments
        for _ in range(2):
            repeat = compressor.compress_as(adapter, SERVER.SELF_TEST_SEGMENTS, 0.5).segments
            if repeat != base_output:
                note("selftest_determinism", adapter=adapter)
        isolated = compressor.compress_as(adapter, SERVER.SELF_TEST_SEGMENTS[:1], 0.5).segments
        if isolated[0] != base_output[0]:
            note("selftest_batch_invariance", adapter=adapter)
        if (
            compressor.compress_as(adapter, SERVER.SELF_TEST_SEGMENTS, 0.0).segments
            != SERVER.SELF_TEST_SEGMENTS
        ):
            note("selftest_threshold0", adapter=adapter)
        self_test_seconds[adapter] = round(time.perf_counter() - started, 4)
    verdicts["kb4_self_test_seconds_per_adapter"] = self_test_seconds

    verdicts["mismatches"] = mismatches
    return verdicts


def _adapter_dir_of(compressor: MultiAdapterCompressor, name: str) -> str:
    """Recover the checkpoint directory an adapter was loaded from."""
    return compressor._adapter_dirs[name]
