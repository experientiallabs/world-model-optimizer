"""Patch the pinned SWE-rebench taskset for local rows and graded F2P reward."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "coding-router-swerebench-graded-taskset-v1"
SOURCE_SHA256 = "a2790c3f296a28f40eb8732d68c091cc7b9899e08916aedec6b2b53a644f7b3e"

IMPORT_OLD = "import json\nimport re\nfrom typing import Literal\n"
IMPORT_NEW = "import json\nimport re\nfrom pathlib import Path\nfrom typing import Literal\n"

IMAGE_OLD = '''def _coding_router_docker_image(image: str) -> str:
    """Invert the filtered dataset's documented Prime registry rewrite."""
    prefix = "prime/primeintellect/"
    if not image.startswith(prefix):
        raise ValueError(f"unexpected SWE-rebench image alias: {image!r}")
    suffix = image.removeprefix(prefix)
    if not suffix or suffix != suffix.strip() or suffix.startswith("/"):
        raise ValueError(f"invalid SWE-rebench image alias: {image!r}")
    return f"docker.io/swerebenchv2/{suffix}"
'''
IMAGE_NEW = '''def _coding_router_docker_image(image: str) -> str:
    """Resolve either the source registry image or the Prime dataset alias."""
    direct_prefix = "docker.io/swerebenchv2/"
    prime_prefix = "prime/primeintellect/"
    if image.startswith(direct_prefix):
        suffix = image.removeprefix(direct_prefix)
    elif image.startswith(prime_prefix):
        suffix = image.removeprefix(prime_prefix)
    else:
        raise ValueError(f"unexpected SWE-rebench image alias: {image!r}")
    if not suffix or suffix != suffix.strip() or suffix.startswith("/"):
        raise ValueError(f"invalid SWE-rebench image alias: {image!r}")
    return f"docker.io/swerebenchv2/{suffix}"
'''

RESOLVED_OLD = (
    "def is_resolved(status_map: dict[str, str], fail_to_pass: list[str], "
    "pass_to_pass: list[str]) -> bool:\n"
    "    normalized = {normalize_test_name(k): v for k, v in status_map.items()}\n"
    "    expected = [normalize_test_name(t) for t in fail_to_pass + pass_to_pass]\n"
    "    if not expected:\n"
    "        return False\n"
    "    return all(normalized.get(t) == log_parsers.TestStatus.PASSED.value "
    "for t in expected)\n"
)
RESOLVED_NEW = '''def is_resolved(
    status_map: dict[str, str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> bool:
    normalized = {normalize_test_name(k): v for k, v in status_map.items()}
    expected = [normalize_test_name(t) for t in fail_to_pass + pass_to_pass]
    if not expected:
        return False
    return all(normalized.get(t) == log_parsers.TestStatus.PASSED.value for t in expected)


def graded_f2p_reward(status_map: dict[str, str], fail_to_pass: list[str]) -> float:
    """Return the fraction of curated fail-to-pass tests that now pass."""
    normalized = {normalize_test_name(k): v for k, v in status_map.items()}
    expected = [normalize_test_name(t) for t in fail_to_pass]
    if not expected:
        return 0.0
    passed = sum(
        normalized.get(test) == log_parsers.TestStatus.PASSED.value
        for test in expected
    )
    return passed / len(expected)
'''

REWARD_OLD = (
    "        return 1.0 if is_resolved(status_map, self.data.fail_to_pass, "
    "self.data.pass_to_pass) else 0.0\n"
)
REWARD_NEW = """        return graded_f2p_reward(status_map, self.data.fail_to_pass)
"""

CONFIG_OLD = '''    filter_fn: str | None = None
    """Optional Python expression string applied with `datasets.Dataset.filter` to raw HF rows."""
'''
CONFIG_NEW = '''    filter_fn: str | None = None
    """Optional Python expression string applied with `datasets.Dataset.filter` to raw HF rows."""
    task_json: str | None = None
    """Optional local single-task JSON prepared by the coding-router runner."""
'''

LOAD_OLD = '''    def load(self) -> list[SWERebenchV2Task]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.split)
        if self.config.filter_fn is not None:
            rows = rows.filter(_resolve_filter_fn(self.config.filter_fn))
        return [
'''
LOAD_NEW = '''    def load(self) -> list[SWERebenchV2Task]:
        if self.config.task_json is not None:
            row = json.loads(Path(self.config.task_json).read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError("local task JSON must be an object")
            return [
                SWERebenchV2Task(
                    SWERebenchV2Data(
                        idx=0,
                        name=row["task_id"],
                        prompt=row["prompt"],
                        image=_coding_router_docker_image(row["image_name"]),
                        workdir=repo_workdir(row["repository"]),
                        resources=vf.TaskResources(cpu=4, memory=4, disk=10),
                        install_config=row["install_config"],
                        base_commit=row.get("base_commit") or "",
                        test_patch=row.get("test_patch") or "",
                        gold_patch=row.get("gold_patch") or "",
                        fail_to_pass=list(row.get("fail_to_pass") or []),
                        pass_to_pass=list(row.get("pass_to_pass") or []),
                    ),
                    self.config.task,
                )
            ]

        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.split)
        if self.config.filter_fn is not None:
            rows = rows.filter(_resolve_filter_fn(self.config.filter_fn))
        return [
'''

REPLACEMENTS = (
    (IMPORT_OLD, IMPORT_NEW),
    (IMAGE_OLD, IMAGE_NEW),
    (RESOLVED_OLD, RESOLVED_NEW),
    (REWARD_OLD, REWARD_NEW),
    (CONFIG_OLD, CONFIG_NEW),
    (LOAD_OLD, LOAD_NEW),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_text(source: str) -> str:
    patched = source
    for old, new in REPLACEMENTS:
        if patched.count(old) != 1:
            raise ValueError("pinned taskset patch anchor changed")
        patched = patched.replace(old, new, 1)
    if patched == source:
        raise ValueError("taskset patch made no change")
    return patched


def run(args: argparse.Namespace) -> None:
    """Apply the exact patch and write a machine-readable audit."""
    if _sha256(args.source) != SOURCE_SHA256:
        raise ValueError("pinned taskset source changed")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("taskset patch output already exists")
    patched = _patch_text(args.source.read_text(encoding="utf-8"))
    compile(patched, str(args.output), "exec")
    args.output.write_text(patched, encoding="utf-8")
    report = {
        "protocol": PROTOCOL,
        "source_sha256": SOURCE_SHA256,
        "patched_sha256": _sha256(args.output),
        "changes": {
            "accept_direct_source_image": True,
            "load_one_local_task_json": True,
            "reward_is_f2p_passed_over_total": True,
            "pass_to_pass_used_in_graded_reward": False,
            "test_execution_or_parser_changed": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
