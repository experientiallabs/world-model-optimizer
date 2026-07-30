"""Fetch and list trace-corpus data bundles from the Hugging Face Hub (stdlib-only).

Every publishable benchmark's bundle — the trace corpus plus its task data / gold / evidence
dirs — lives in a dataset repo under the org. This module is the READ core behind
`python -m environment_capture.hub fetch` and the contract the website's serving
trace-download endpoint adopts (PR #52). Plain-HTTP against the Hub's public REST API, so it
needs no extra dependency and no token for public repos (pass ``token`` for private ones).
Uploading lives in `environment_capture.hub_push` (the ``fetch`` extra).

`wmo download` runs a VENDORED copy of the read half, `wmo/hub.py`: the flagship wheel must
install with no dependency on anything under `packages/` (AGENTS.md § Monorepo). Registry,
repo-id convention and data root are pinned across the two by
`wmo/hub_test.py::test_the_two_copies_agree_on_the_registry_and_the_data_root`; the rest of the
body is not, so mirror edits here into that copy by hand.

Bundles are local-first: capture writes into the benchmark dir, nothing here deletes local
files, and fetching never overwrites an existing file unless forced. Downloads stream to a
``.part`` sibling and are atomically renamed, so a failed fetch never looks like a corpus.

Usage (from the repo root):
    uv run wmo download                                              # interactive picker
    uv run python -m environment_capture.hub fetch dabstep           # skip if already present
    uv run python -m environment_capture.hub fetch all --force       # overwrite local copies
    uv run python -m environment_capture.hub_push bird-sql           # publish/update (write side)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

from environment_capture.trajectory import JsonValue

_ORG = "experiential-labs"
_CORPUS_FILE = "traces.otel.jsonl"
# Honors enterprise/mirror endpoints the way huggingface_hub does.
_HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
_CHUNK_BYTES = 1 << 20

# Dataset-repo name prefixes, most-preferred first. LEGACY FALLBACK: the project renamed
# `wmh` -> `wmo`, but the org's dataset repos on the Hub are still published under the old
# `wmh-<benchmark>-traces` name, so every read must try both. Once the Hub repos are renamed,
# drop "wmh" from this tuple and the fallback disappears with it.
_REPO_PREFIXES = ("wmo", "wmh")
_REPO_SUFFIX = "-traces"
# What the Hub answers for a repo id that does not resolve: 404 anonymously, 401 when a token
# is attached (it will not admit a repo is missing to a caller that might lack access), 403
# for a gated repo. Any of these means "try the next candidate name", not "the Hub is broken".
_MISSING_REPO_CODES = frozenset({401, 403, 404})

# on_progress(bytes_done, bytes_total): called after every streamed chunk, across ALL files in
# the fetch (front-ends render one bar for the whole bundle).
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class CorpusSpec:
    """One publishable corpus: where it lives locally and how its dataset card reads."""

    benchmark: str  # dir name under packages/environment-capture/
    license_id: str  # Hub license identifier (must match the upstream terms)
    upstream: str  # attribution line for the dataset card
    description: str  # one-sentence environment summary
    extra_terms: str = ""  # disclosures that ride below the boilerplate
    # Data payload dirs published alongside the trace corpus (task indexes, gold sidecars,
    # evidence docs, ...). Same license as the corpus — they ARE the upstream-derived data.
    data_dirs: tuple[str, ...] = ()
    # Whether a dataset repo for this bundle actually exists on the Hub yet. False means
    # "registered here so the write side knows how to publish it, but nothing to download":
    # reads must never advertise it, or `wmo download <name>` sends the user at a repo that
    # can only answer 401. Flipping this to True is part of the push that publishes it.
    published: bool = True


# The publishable corpora. appworld is deliberately ABSENT: its protected data may only be
# redistributed in encrypted form (plain-text posting is disallowed), so that corpus stays
# local-only — see packages/environment-capture/appworld/README.md § License.
CORPORA: dict[str, CorpusSpec] = {
    spec.benchmark: spec
    for spec in (
        CorpusSpec(
            benchmark="financebench",
            data_dirs=("data", "gold", "corpus"),
            license_id="cc-by-nc-4.0",
            upstream="PatronusAI/financebench (CC BY-NC 4.0)",
            description=(
                "Financial-document QA over real SEC-filing evidence excerpts: the agent greps "
                "a workspace of evidence docs plus distractors and submits a final answer."
            ),
        ),
        CorpusSpec(
            benchmark="bird-sql",
            data_dirs=("data", "gold", "schemas"),
            license_id="cc-by-sa-4.0",
            upstream="bird-bench mini-dev (CC BY-SA 4.0)",
            description=(
                "Text-to-SQL over real SQLite databases: the agent explores a copy of the "
                "task's database and schema, then submits a SQL query."
            ),
        ),
        CorpusSpec(
            benchmark="continual-learning",
            data_dirs=("data", "gold"),
            license_id="cc-by-4.0",
            upstream="Continual Learning Bench (CC BY 4.0)",
            description=(
                "Database-exploration QA over a large, deliberately obfuscated SQLite product "
                "database: the agent maps cryptic tables to answer catalog questions."
            ),
        ),
        CorpusSpec(
            benchmark="dabstep",
            data_dirs=("data", "gold", "datafiles"),
            license_id="cc-by-4.0",
            upstream="adyen/DABstep (CC BY 4.0)",
            description=(
                "Data-analysis QA over a shared payments dataset and a business-rules manual, "
                "answered with pandas in a Python shell."
            ),
        ),
        CorpusSpec(
            benchmark="crmarena",
            data_dirs=("data", "gold"),
            license_id="cc-by-nc-4.0",
            upstream="Salesforce CRMArena (CC BY-NC 4.0)",
            description=(
                "Professional CRM analytics over a realistic Salesforce org snapshot: case "
                "routing, handle-time analytics, and entity disambiguation via SQL."
            ),
        ),
        CorpusSpec(
            benchmark="gaia2",
            data_dirs=("data",),
            license_id="cc-by-4.0",
            upstream="meta-agents-research-environments/gaia2 (CC-BY-4.0, attribution to Meta)",
            description=(
                "A stateful multi-app simulated world (Contacts, Email, Calendar, Shopping, "
                "...): the agent drives app tools with Python against live scenario state."
            ),
            extra_terms=(
                "Rewards in this corpus come from a deterministic STRUCTURAL grader "
                "(oracle-action matching), not GAIA2's official judge — scores are not "
                "comparable to the official leaderboard. GAIA2's authors ask that models not "
                "be trained on evaluation data; the test split carries that request."
            ),
        ),
        CorpusSpec(
            benchmark="tau-bench",
            license_id="mit",
            upstream="sierra-research/tau2-bench (MIT)",
            description=(
                "Customer-service tool-agent episodes from the real tau2-bench harness "
                "(airline/retail/telecom domains)."
            ),
        ),
        CorpusSpec(
            benchmark="terminal-tasks",
            license_id="apache-2.0",
            upstream="terminal-bench (Apache 2.0)",
            description=(
                "Computer-use agent runs in real terminal containers: bash commands and their "
                "true outputs from live task environments."
            ),
        ),
        CorpusSpec(
            benchmark="swe-bench",
            license_id="mit",
            upstream="princeton-nlp/SWE-bench Verified + mini-swe-agent (MIT)",
            description=(
                "Software-engineering agent runs from real SWE-bench Verified instances: shell "
                "exploration and repo edits inside per-instance Docker images."
            ),
        ),
    )
}


@dataclass(frozen=True)
class PublishedCorpus:
    """One live dataset repo under the org, mapped back to its benchmark."""

    benchmark: str
    repo_id: str
    last_modified: str  # ISO date, "" when the Hub omits it


class CorpusRepoUnavailable(urllib.error.HTTPError):
    """Every candidate dataset repo for one benchmark was refused by the Hub.

    Subclasses ``HTTPError`` so front-ends that already catch it (``wmo download`` skips the
    benchmark and keeps fetching the rest) behave exactly as before. A bare code cannot say
    WHICH name was wrong once there is more than one candidate, so the message and
    ``attempted`` name every repo id that was tried.
    """

    def __init__(
        self,
        benchmark: str,
        revision: str,
        failures: Sequence[tuple[str, urllib.error.HTTPError]],
    ) -> None:
        """Build the combined error from each (repo id, Hub refusal) attempt, in order."""
        last = failures[-1][1]
        tried = ", ".join(f"{repo_id} answered {error.code}" for repo_id, error in failures)
        super().__init__(
            last.url,
            last.code,
            f"no dataset repo for {benchmark!r} at revision {revision!r}: {tried}",
            last.headers,
            None,
        )
        self.attempted: tuple[str, ...] = tuple(repo_id for repo_id, _ in failures)


def repo_id_for(benchmark: str) -> str:
    """The canonical dataset repo backing one benchmark's corpus (the id we publish to)."""
    return candidate_repo_ids(benchmark)[0]


def candidate_repo_ids(benchmark: str) -> tuple[str, ...]:
    """Every dataset repo id that may hold this benchmark's bundle, most-preferred first.

    Reads walk this tuple and take the first id the Hub resolves; see ``_REPO_PREFIXES`` for
    why there is more than one.
    """
    return tuple(f"{_ORG}/{prefix}-{benchmark}{_REPO_SUFFIX}" for prefix in _REPO_PREFIXES)


def _benchmark_from_repo_name(name: str) -> str | None:
    """The benchmark a bare repo name encodes, or ``None`` when it is not a corpus repo.

    Args:
        name: Repo name without the org prefix, e.g. ``wmo-gaia2-traces``.

    Returns:
        The benchmark name, or ``None`` when ``name`` matches no known prefix convention.
    """
    if not name.endswith(_REPO_SUFFIX):
        return None
    for prefix in _REPO_PREFIXES:
        if name.startswith(f"{prefix}-"):
            return name[len(prefix) + 1 : -len(_REPO_SUFFIX)]
    return None


def corpus_path(benchmark: str) -> Path:
    """The canonical local path of the benchmark's trace corpus (whether or not it exists yet).

    This is the "is it local, and where" resolver every front-end shares: check
    ``corpus_path(b).exists()`` before deciding to download or to serve from disk.
    """
    return _data_root() / benchmark / _CORPUS_FILE


def published_corpora(*, token: str | None = None) -> list[PublishedCorpus]:
    """The org's live corpus datasets (Hub REST API), newest first, mapped to benchmark names.

    Only repos that follow a ``<prefix>-<benchmark>-traces`` convention (see
    ``_REPO_PREFIXES``) AND appear in the local manifest are returned: those are the ones
    ``fetch_corpus`` knows where to place. A benchmark published under two prefixes at once
    (mid-rename) is listed once, under the most-preferred repo.
    """
    token = token if token is not None else _default_token()
    listing = _http_json_pages(f"{_HUB}/api/datasets?author={_ORG}&limit=100", token=token)
    by_benchmark: dict[str, PublishedCorpus] = {}
    for entry in listing:
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("id", ""))
        benchmark = _benchmark_from_repo_name(repo_id.removeprefix(f"{_ORG}/"))
        if benchmark is None or benchmark not in CORPORA:
            continue
        candidates = candidate_repo_ids(benchmark)
        kept = by_benchmark.get(benchmark)
        if kept is not None and candidates.index(kept.repo_id) <= candidates.index(repo_id):
            continue  # already listed under a more-preferred repo name
        modified = str(entry.get("lastModified") or "")
        by_benchmark[benchmark] = PublishedCorpus(
            benchmark=benchmark, repo_id=repo_id, last_modified=modified[:10]
        )
    return sorted(by_benchmark.values(), key=lambda c: c.last_modified, reverse=True)


def downloadable_benchmarks() -> list[str]:
    """Registered benchmarks whose bundle really is on the Hub, sorted.

    The offline answer to "what can I pass to ``wmo download``": every list of names a read path
    offers comes from here rather than from ``CORPORA``, so it can never advertise a benchmark
    whose only possible reply is a 401.
    """
    return sorted(name for name, spec in CORPORA.items() if spec.published)


def fetch_corpus(
    benchmark: str,
    *,
    dest: Path | None = None,
    force: bool = False,
    token: str | None = None,
    revision: str = "main",
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download the benchmark's corpus AND published data dirs into place; returns the corpus path.

    Local-first and resumable at file granularity: every published file that is missing locally
    is fetched; existing files are kept unless ``force=True`` — fetching must never silently
    clobber a corpus that local capture waves have grown past the published one, and an
    interrupted fetch picks up the files it hasn't finished. With an explicit ``dest`` only the
    corpus file is written (no data dirs). ``on_progress(bytes_done, bytes_total)`` fires per
    streamed chunk across the whole bundle.

    Raises ``ValueError`` for unknown/unpublished corpora and ``urllib.error.URLError`` (incl.
    ``HTTPError``, and ``CorpusRepoUnavailable`` when no candidate repo id resolves) when the
    Hub is unreachable: front-ends translate those for their users.
    """
    spec = CORPORA.get(benchmark)
    if spec is None:
        available = ", ".join(downloadable_benchmarks())
        raise ValueError(f"{benchmark!r} has no published corpus (available: {available})")
    if not spec.published:
        available = ", ".join(downloadable_benchmarks())
        raise ValueError(
            f"{benchmark!r} is a registered benchmark but its data bundle has not been "
            f"published to the Hub yet, so there is nothing to download; pick one of: {available}"
        )
    token = token if token is not None else _default_token()
    root = _data_root()
    target = dest or corpus_path(benchmark)

    # One recursive tree call covers the whole repo; sizes make the total known up front so a
    # single progress bar can span the bundle.
    repo_id, sizes = _resolve_repo(benchmark, revision, token=token)
    work: list[tuple[str, Path, int]] = []
    if not target.exists() or force:
        if _CORPUS_FILE not in sizes:
            raise ValueError(
                f"{repo_id} has no {_CORPUS_FILE} at revision {revision!r} — the dataset repo "
                "exists but the corpus was never pushed; push it or pick another benchmark"
            )
        work.append((_CORPUS_FILE, target, sizes[_CORPUS_FILE]))
    if dest is None:
        for remote_path, size in sorted(sizes.items()):
            top = remote_path.split("/", 1)[0]
            if top not in spec.data_dirs:
                continue
            local = root / benchmark / remote_path
            # File-level skip, not dir-level: an interrupted fetch that materialized only part
            # of a dir resumes with the missing files instead of treating the dir as done.
            if local.exists() and not force:
                continue
            work.append((remote_path, local, size))

    total = sum(size for _, _, size in work)
    done = 0

    def chunk_done(n: int) -> None:
        nonlocal done
        done += n
        if on_progress is not None:
            on_progress(done, total)

    for remote_path, local, size in work:
        url = f"{_HUB}/datasets/{repo_id}/resolve/{revision}/{urllib.parse.quote(remote_path)}"
        file_base = done
        written = -1
        last: Exception | None = None
        for delay_s in (0, 1, 3):  # transient failures retry THIS file, not the whole bundle
            if delay_s:
                time.sleep(delay_s)
                done = file_base  # roll the bar back to the file boundary before re-streaming
            try:
                written = _stream_to(
                    url,
                    local,
                    token=token,
                    chunk_done=chunk_done,
                    # `size` is 0 when the tree does not advertise one; there is nothing to
                    # check against then, and 0 would read as "expect an empty file".
                    expect_bytes=size or None,
                )
                break
            except urllib.error.HTTPError as error:
                if error.code < 500:
                    raise
                last = error
            except urllib.error.URLError as error:
                last = error
            except OSError as error:
                # A short transfer (or a local write failure). Not retried: `_stream_to` left
                # the destination untouched, so re-running the whole fetch is the honest remedy
                # and it costs nothing. Named with the file, since a bundle is many of them.
                raise OSError(f"{remote_path}: {error}") from error
        if written < 0:
            assert last is not None
            raise last
    return target


def _resolve_repo(
    benchmark: str, revision: str, *, token: str | None
) -> tuple[str, dict[str, int]]:
    """The repo id that actually serves this benchmark, plus its file tree (path -> size).

    Catch-and-retry over ``candidate_repo_ids``, not probe-then-fetch: the tree call the fetch
    already needs IS the existence check, so the preferred id costs exactly one request and
    only a miss pays for a second lookup. The resolved id is returned so every file in the
    bundle streams from the same repo the tree came from.

    Raises:
        CorpusRepoUnavailable: every candidate id was refused (404/401/403), naming them all.
        urllib.error.HTTPError: any other Hub failure, from the first candidate that hit it.
    """
    failures: list[tuple[str, urllib.error.HTTPError]] = []
    for repo_id in candidate_repo_ids(benchmark):
        try:
            return repo_id, dict(_repo_tree(repo_id, revision, token=token))
        except urllib.error.HTTPError as error:
            if error.code not in _MISSING_REPO_CODES:
                raise
            failures.append((repo_id, error))
    raise CorpusRepoUnavailable(benchmark, revision, failures)


def _data_root() -> Path:
    """Where benchmark data dirs live (``<root>/<benchmark>/traces.otel.jsonl``).

    Resolution order: the ``ENVCAP_DATA_ROOT`` env var; a repo checkout (the dir holding
    this package — its sibling benchmark dirs); else, for an installed wheel,
    ``environment-capture-data/`` under the current directory — a pip user's bundles land in
    their project, never inside site-packages.
    """
    override = os.environ.get("ENVCAP_DATA_ROOT")
    if override:
        return Path(override)
    sibling = Path(__file__).resolve().parents[1]
    if (sibling / "pyproject.toml").exists():  # repo checkout: the member dir itself
        return sibling
    return Path.cwd() / "environment-capture-data"


def _default_token() -> str | None:
    """The token huggingface_hub would use: HF_TOKEN env, else the stored `hf auth login`."""
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    stored = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    try:
        return stored.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


class _AuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when a redirect leaves the original host.

    Hub `resolve/` URLs 302 large files to a CDN/S3 presigned host; forwarding the bearer
    token there leaks it to a third-party origin and can 400 on presigned URLs (dual auth).
    This mirrors huggingface_hub's behavior.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            new is not None
            and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc
        ):
            new.headers.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_AuthStrippingRedirectHandler)


def _request(url: str, token: str | None) -> urllib.request.Request:
    headers = {"User-Agent": "environment-capture/hub"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)  # noqa: S310 - https-only constants


def _http_json_pages(url: str, *, token: str | None) -> list[JsonValue]:
    """GET a paginated JSON listing, following RFC5988 Link rel="next" headers to the end."""
    items: list[JsonValue] = []
    next_url: str | None = url
    while next_url:
        body, next_url = _http_json_page(next_url, token=token)
        items.extend(body if isinstance(body, list) else [body])
    return items


def _http_json_page(url: str, *, token: str | None) -> tuple[JsonValue, str | None]:
    """One GET with transient-failure retry; returns (json, next-page url from the Link header)."""
    last: Exception | None = None
    for delay_s in (0, 1, 3):
        if delay_s:
            time.sleep(delay_s)
        try:
            with _OPENER.open(_request(url, token), timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
                link = response.headers.get("Link", "")
                next_url = None
                for part in link.split(","):
                    if 'rel="next"' in part and "<" in part:
                        next_url = part.split("<", 1)[1].split(">", 1)[0]
                return body, next_url
        except urllib.error.HTTPError as error:
            if error.code < 500:
                raise  # 4xx is a real answer, not a flake
            last = error
        except urllib.error.URLError as error:
            last = error
    assert last is not None
    raise last


def _repo_tree(repo_id: str, revision: str, *, token: str | None) -> list[tuple[str, int]]:
    """(path, size) for every FILE in the dataset repo (recursive, follows pagination)."""
    url = f"{_HUB}/api/datasets/{repo_id}/tree/{revision}?recursive=true"
    listing = _http_json_pages(url, token=token)
    files: list[tuple[str, int]] = []
    for entry in listing:
        if isinstance(entry, dict) and entry.get("type") == "file":
            size = entry.get("size")
            files.append((str(entry.get("path", "")), size if isinstance(size, int) else 0))
    return files


def _stream_to(
    url: str,
    dest: Path,
    *,
    token: str | None,
    chunk_done: Callable[[int], None],
    expect_bytes: int | None = None,
) -> int:
    """Stream ``url`` to ``dest`` atomically; returns the byte count written.

    Writes a ``.part`` sibling and renames over, so a partially-downloaded corpus is never
    mistaken for a complete one by a concurrent reader; a failed stream removes its ``.part``.

    ``expect_bytes`` is the size the Hub's tree advertises for this file, and it is verified
    BEFORE the rename. Checking after would mean a short transfer had already replaced ``dest``,
    which is only harmless when ``dest`` did not exist: under ``force`` it holds a corpus that
    is currently valid, and clobbering it with a truncated one — or deleting it afterwards —
    turns a failed refresh into data loss. Verifying first makes a short transfer a no-op:
    ``dest`` keeps whatever it had, the ``.part`` is dropped, and the fetch is genuinely
    re-runnable.

    Raises:
        OSError: fewer (or more) bytes arrived than ``expect_bytes``. ``dest`` is untouched.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    written = 0
    try:
        with _OPENER.open(_request(url, token), timeout=300) as response:
            with part.open("wb") as sink:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    sink.write(chunk)
                    written += len(chunk)
                    chunk_done(len(chunk))
        if expect_bytes is not None and written != expect_bytes:
            raise OSError(
                f"downloaded {written} bytes but the Hub tree lists {expect_bytes} — truncated "
                f"transfer; {dest} was left as it was, re-run the fetch"
            )
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # Pushing lives in hub_push (needs the huggingface_hub extra); this stub keeps the old
    # command discoverable without importing the write side here (imports stay module-scope).
    push = sub.add_parser("push", help="Moved: use `python -m environment_capture.hub_push`.")
    push.add_argument("benchmark", nargs="?")
    push.add_argument("--private", action="store_true")

    fetch = sub.add_parser("fetch", help="Download full data bundles into the benchmark dirs.")
    fetch.add_argument(
        "benchmark", help=f"Benchmark name, or 'all' ({', '.join(downloadable_benchmarks())})"
    )
    fetch.add_argument(
        "--force", action="store_true", help="Overwrite existing local corpus/data files."
    )

    args = parser.parse_args()
    if args.command == "push":
        raise SystemExit(
            "pushing moved to the write module: "
            "`uv run python -m environment_capture.hub_push <benchmark>|all [--private]` "
            "(needs the fetch extra + a write token)"
        )
    names = downloadable_benchmarks() if args.benchmark == "all" else [args.benchmark]
    for name in names:
        existing = corpus_path(name).exists()
        path = fetch_corpus(name, force=args.force)
        state = "kept local" if existing and not args.force else "fetched"
        print(f"{state} {name} -> {path}")


if __name__ == "__main__":
    main()
