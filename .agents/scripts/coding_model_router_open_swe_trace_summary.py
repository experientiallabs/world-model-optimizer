"""Stream Open-SWE trajectories into compact, text-free burden summaries.

The full trajectory payload is intended to run on remote compute. Each pinned
Parquet shard is verified, reduced independently, and removed before the next
shard. The resulting rows contain counts and hashes but no message text, tool
arguments, tool output, or patch content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("coding-router-open-swe-trace-summary")

DATASET = "nvidia/Open-SWE-Traces"
REVISION = "9c0e4579a4ee0effa3e5f7a552494a045f29377d"
EXPECTED_SHARDS = 84
EXPECTED_BYTES = 18_338_420_390
SHARD_DIRECTORIES = (
    "data/minimax_m25_openhands_trajectories",
    "data/minimax_m25_sweagent_trajectories",
    "data/qwen35_openhands_trajectories",
    "data/qwen35_sweagent_trajectories",
)

_SHELL = re.compile(r"(?:bash|shell|terminal|execute|run_command)", re.IGNORECASE)
_SEARCH = re.compile(r"(?:grep|ripgrep|\brg\b|search|find|glob)", re.IGNORECASE)
_READ = re.compile(r"(?:read|view|open_file|cat\s|sed\s+-n|head\s|tail\s)", re.IGNORECASE)
_EDIT = re.compile(r"(?:edit|write|patch|str_replace|create_file)", re.IGNORECASE)
_TEST = re.compile(
    r"(?:pytest|unittest|tox|nox|cargo\s+test|go\s+test|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|mvn\s+test|gradle\w*\s+test)",
    re.IGNORECASE,
)


class SourceShard(TypedDict):
    """Immutable identity for one native Open-SWE Parquet shard."""

    path: str
    bytes: int
    sha256: str


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_payload(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "world-model-optimizer-open-swe-trace-summary/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _source_manifest() -> list[SourceShard]:
    rows: list[SourceShard] = []
    for directory in SHARD_DIRECTORIES:
        encoded = urllib.parse.quote(directory, safe="/")
        payload = _json_payload(
            f"https://huggingface.co/api/datasets/{DATASET}/tree/"
            f"{REVISION}/{encoded}?recursive=true&expand=true&limit=100"
        )
        if not isinstance(payload, list):
            raise ValueError(f"invalid Hugging Face tree response for {directory}")
        for item_value in payload:
            if not isinstance(item_value, dict):
                continue
            item = cast(dict[str, object], item_value)
            if not str(item.get("path", "")).endswith(".parquet"):
                continue
            lfs = item.get("lfs")
            if not isinstance(lfs, dict):
                raise ValueError(f"source shard has no LFS identity: {item.get('path')}")
            lfs_row = cast(dict[str, object], lfs)
            rows.append(
                {
                    "path": str(item.get("path", "")),
                    "bytes": _integer(item.get("size"), field="source shard size"),
                    "sha256": str(lfs_row.get("oid", "")),
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    if len(rows) != EXPECTED_SHARDS:
        raise ValueError(f"expected {EXPECTED_SHARDS} shards, found {len(rows)}")
    if sum(row["bytes"] for row in rows) != EXPECTED_BYTES:
        raise ValueError("Open-SWE source byte total changed")
    if len({str(row["path"]) for row in rows}) != EXPECTED_SHARDS:
        raise ValueError("Open-SWE source manifest contains duplicate paths")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) for row in rows):
        raise ValueError("Open-SWE source manifest has an invalid LFS digest")
    return rows


def _partition(path: str) -> tuple[str, str]:
    parent = Path(path).parent.name
    match = re.fullmatch(r"(minimax_m25|qwen35)_(openhands|sweagent)_trajectories", parent)
    if match is None:
        raise ValueError(f"unrecognized Open-SWE partition: {parent}")
    model_mode, scaffold = match.groups()
    return scaffold, model_mode


def _as_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _model_patch_counts(metadata: object) -> tuple[int, int]:
    if metadata is None:
        return 0, 0
    root = _as_mapping(metadata, field="metadata")
    patch = root.get("model_patch")
    if patch is None:
        return 0, 0
    model_patch = _as_mapping(patch, field="metadata.model_patch")
    files = model_patch.get("num_modified_files")
    lines = model_patch.get("num_modified_lines")
    return (
        0 if files is None else _integer(files, field="num_modified_files"),
        0 if lines is None else _integer(lines, field="num_modified_lines"),
    )


def _tool_summary(messages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    signatures: list[str] = []
    categories = {"shell": 0, "search": 0, "read": 0, "edit": 0, "test": 0}
    names: set[str] = set()
    for message in messages:
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("trajectory tool_calls must be a list")
        for call_value in calls:
            call = _as_mapping(call_value, field="tool call")
            function = _as_mapping(call.get("function"), field="tool call function")
            name = _as_text(function.get("name"))
            arguments = _as_text(function.get("arguments"))
            names.add(name)
            signature = hashlib.sha256(f"{name}\0{arguments}".encode()).hexdigest()
            signatures.append(signature)
            haystack = f"{name}\n{arguments}"
            categories["shell"] += int(bool(_SHELL.search(haystack)))
            categories["search"] += int(bool(_SEARCH.search(haystack)))
            categories["read"] += int(bool(_READ.search(haystack)))
            categories["edit"] += int(bool(_EDIT.search(haystack)))
            categories["test"] += int(bool(_TEST.search(haystack)))
    longest_run = 0
    current_run = 0
    previous = ""
    for signature in signatures:
        current_run = current_run + 1 if signature == previous else 1
        longest_run = max(longest_run, current_run)
        previous = signature
    return {
        "tool_calls": len(signatures),
        "distinct_tools": len(names),
        "repeated_calls": len(signatures) - len(set(signatures)),
        "max_repeated_call_run": longest_run,
        **{f"{name}_calls": count for name, count in categories.items()},
    }


def _summarize_row(
    row: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    source_row: int,
) -> dict[str, object]:
    instance_id = _as_text(row.get("instance_id"))
    trajectory_id = _as_text(row.get("trajectory_id"))
    repo = _as_text(row.get("repo"))
    language = _as_text(row.get("language"))
    if not instance_id or not trajectory_id or not repo:
        raise ValueError("trajectory row is missing identity")
    resolved = _integer(row.get("resolved"), field="resolved")
    if resolved not in (0, 1):
        raise ValueError("resolved must be binary")
    trajectory_value = row.get("trajectory")
    if not isinstance(trajectory_value, list) or not trajectory_value:
        raise ValueError("trajectory must be a nonempty list")
    messages = [_as_mapping(message, field="trajectory message") for message in trajectory_value]
    scaffold, model_mode = _partition(source_path)
    model_patch_files, model_patch_lines = _model_patch_counts(row.get("metadata"))
    raw_digest = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    summary: dict[str, object] = {
        "instance_id": instance_id,
        "repo": repo,
        "language": language,
        "scaffold": scaffold,
        "model_mode": model_mode,
        "trajectory_id": trajectory_id,
        "resolved": resolved,
        "messages": len(messages),
        "assistant_turns": sum(
            _as_text(message.get("role")) == "assistant" for message in messages
        ),
        "reasoning_characters": sum(
            len(_as_text(message.get("reasoning_content"))) for message in messages
        ),
        "content_characters": sum(len(_as_text(message.get("content"))) for message in messages),
        "model_patch_files": model_patch_files,
        "model_patch_lines": model_patch_lines,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_row": source_row,
        "source_row_sha256": raw_digest,
    }
    summary.update(_tool_summary(messages))
    return summary


def _download(url: str, destination: Path) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "world-model-optimizer-open-swe-trace-summary/1"},
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=300) as response:
        with destination.open("wb") as handle:
            while chunk := response.read(4 * 1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest()


def _summarize_shard(
    source: Path,
    source_info: SourceShard,
    output: Path,
) -> dict[str, int]:
    parquet = pq.ParquetFile(source)
    summaries: list[dict[str, object]] = []
    rejected = 0
    source_row = 0
    columns = [
        "instance_id",
        "repo",
        "language",
        "trajectory_id",
        "trajectory",
        "resolved",
        "metadata",
    ]
    for batch in parquet.iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            try:
                summaries.append(
                    _summarize_row(
                        row,
                        source_path=str(source_info["path"]),
                        source_sha256=str(source_info["sha256"]),
                        source_row=source_row,
                    )
                )
            except (KeyError, TypeError, ValueError):
                rejected += 1
            source_row += 1
    if not summaries:
        raise ValueError(f"source shard produced no valid summaries: {source_info['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.parquet")
    pq.write_table(pa.Table.from_pylist(summaries), temporary, compression="zstd")
    os.replace(temporary, output)
    return {"rows": len(summaries), "rejected": rejected, "source_rows": source_row}


def run(*, work_dir: Path, report: Path, max_shards: int | None) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    outputs = work_dir / "summaries"
    downloads = work_dir / "downloads"
    outputs.mkdir(exist_ok=True)
    downloads.mkdir(exist_ok=True)
    state_path = work_dir / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"schema": "open-swe-trace-summary-state-v1", "completed": {}}
    )
    completed = state.get("completed")
    if not isinstance(completed, dict):
        raise ValueError("summary state has invalid completed map")
    manifest = _source_manifest()
    selected = manifest if max_shards is None else manifest[:max_shards]
    for index, source_info in enumerate(selected):
        source_path = str(source_info["path"])
        output = outputs / f"{index:03d}.parquet"
        prior = completed.get(source_path)
        if (
            isinstance(prior, dict)
            and output.exists()
            and _sha256(output) == prior.get("output_sha256")
        ):
            logger.info("resume skip shard=%s rows=%s", source_path, prior.get("rows"))
            continue
        encoded_path = urllib.parse.quote(source_path, safe="/")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{encoded_path}"
        download = downloads / f"{index:03d}.parquet"
        logger.info("download shard=%d/%d path=%s", index + 1, len(selected), source_path)
        observed_sha = _download(url, download)
        if observed_sha != source_info["sha256"]:
            download.unlink(missing_ok=True)
            raise ValueError(f"source hash mismatch for {source_path}")
        try:
            stats = _summarize_shard(download, source_info, output)
        finally:
            download.unlink(missing_ok=True)
        completed[source_path] = {
            **stats,
            "source_sha256": source_info["sha256"],
            "output_sha256": _sha256(output),
            "output_bytes": output.stat().st_size,
        }
        _write_json(state_path, state)
        logger.info(
            "summarized shard=%s rows=%d rejected=%d",
            source_path,
            stats["rows"],
            stats["rejected"],
        )
    selected_completed = [completed[str(row["path"])] for row in selected]
    result = {
        "schema": "open-swe-trace-summary-report-v1",
        "dataset": DATASET,
        "revision": REVISION,
        "source_shards": len(manifest),
        "source_bytes": sum(row["bytes"] for row in manifest),
        "selected_shards": len(selected),
        "completed_shards": len(selected_completed),
        "rows": sum(int(row["rows"]) for row in selected_completed),
        "rejected_rows": sum(int(row["rejected"]) for row in selected_completed),
        "summary_bytes": sum(int(row["output_bytes"]) for row in selected_completed),
        "target_outcomes_used": False,
        "raw_trace_retained": False,
        "source_manifest": manifest,
    }
    _write_json(report, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_shards is not None and args.max_shards < 1:
        raise ValueError("max shards must be positive")
    result = run(
        work_dir=args.work_dir.resolve(),
        report=args.report.resolve(),
        max_shards=args.max_shards,
    )
    logger.info(
        "trace summary complete shards=%d rows=%d rejected=%d",
        result["completed_shards"],
        result["rows"],
        result["rejected_rows"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
