"""Tests for the vendored Hub read core: listing, fetching, atomicity, drift (no network).

This is the copy `pip install world-model-optimizer` actually runs, so it carries its own
coverage rather than leaning on the member's. Two of these tests exist only because it IS a
copy: `test_the_vendored_copy_has_not_drifted_from_its_origin` diffs the shared source regions
against the member's file, and the data-root case pins the one line that could NOT be copied
verbatim. Neither imports the member — nothing under `wmo/` does.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from http.client import HTTPMessage
from pathlib import Path

import pytest

from wmo import hub
from wmo.hub import (
    CORPORA,
    CorpusRepoUnavailable,
    CorpusSpec,
    candidate_repo_ids,
    downloadable_benchmarks,
    fetch_corpus,
    published_corpora,
)


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hub, "_data_root", lambda: tmp_path)
    return tmp_path


@dataclass
class _HubCalls:
    """The repo ids the fake Hub was asked for, in order (one entry per request)."""

    trees: list[str] = field(default_factory=list)
    resolves: list[str] = field(default_factory=list)


def _fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    *,
    live_repos: Container[str] | None = None,
    missing_code: int = 404,
    claimed_sizes: dict[str, int] | None = None,
) -> _HubCalls:
    """Stand in for the Hub REST API: a tree listing plus resolve-URL streaming.

    Args:
        monkeypatch: The patcher used to swap the module's HTTP seams.
        files: Repo path -> content, served by every live repo.
        live_repos: Repo ids that resolve; every other id answers ``missing_code``. ``None``
            (the default) means every id resolves.
        missing_code: Status the Hub returns for a repo id outside ``live_repos``.
        claimed_sizes: Repo path -> the size the TREE advertises, when it should disagree with
            the bytes served (how a truncated transfer is detected).

    Returns:
        A live record of the repo ids requested, so a test can assert the request COUNT and
        not just the downloaded bytes.
    """
    calls = _HubCalls()
    sizes = claimed_sizes or {}

    def http_json_page(url: str, *, token: str | None) -> tuple[object, None]:
        assert "/api/datasets/" in url and "/tree/main?recursive=true" in url
        repo_id = url.split("/api/datasets/", 1)[1].split("/tree/", 1)[0]
        calls.trees.append(repo_id)
        if live_repos is not None and repo_id not in live_repos:
            raise urllib.error.HTTPError(url, missing_code, "not found", HTTPMessage(), None)
        listing = [
            {"type": "file", "path": path, "size": sizes.get(path, len(content))}
            for path, content in files.items()
        ]
        return listing, None

    def stream_to(
        url: str,
        dest: Path,
        *,
        token: str | None,
        chunk_done: Callable[[int], None],
        expect_bytes: int | None = None,
    ) -> int:
        calls.resolves.append(url.split("/datasets/", 1)[1].split("/resolve/", 1)[0])
        remote_path = urllib.parse.unquote(url.split("/resolve/main/", 1)[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = files[remote_path]
        chunk_done(len(content))
        # The real streamer verifies the advertised size BEFORE renaming its `.part` over
        # `dest`, so a short transfer leaves `dest` exactly as it was. A fake that wrote first
        # would hide precisely the data-loss case this seam exists to prevent.
        if expect_bytes is not None and len(content) != expect_bytes:
            raise OSError(
                f"downloaded {len(content)} bytes but the Hub tree lists {expect_bytes} — "
                f"truncated transfer; {dest} was left as it was, re-run the fetch"
            )
        dest.write_bytes(content)
        return len(content)

    monkeypatch.setattr(hub, "_http_json_page", http_json_page)
    monkeypatch.setattr(hub, "_stream_to", stream_to)
    return calls


def test_fetch_downloads_corpus_and_data_dirs_with_one_progress_bar(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(
        monkeypatch,
        {
            "traces.otel.jsonl": b"spans\n",
            "data/train.jsonl": b"tasks\n",
            "gold/t1.json": b"{}",
        },
    )
    progress: list[tuple[int, int]] = []

    path = fetch_corpus(
        "continual-learning", on_progress=lambda done, total: progress.append((done, total))
    )

    assert path == data_root / "continual-learning" / "traces.otel.jsonl"
    assert path.read_bytes() == b"spans\n"
    assert (data_root / "continual-learning" / "data" / "train.jsonl").read_bytes() == b"tasks\n"
    assert (data_root / "continual-learning" / "gold" / "t1.json").read_bytes() == b"{}"
    # one monotone bar over the WHOLE bundle: total constant, done reaches it
    total = 6 + 6 + 2
    assert progress == [(6, total), (12, total), (14, total)]


def test_fetch_keeps_existing_local_files_unless_forced(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local-first: a corpus grown by local capture waves must never be silently clobbered."""
    _fake_hub(monkeypatch, {"traces.otel.jsonl": b"published\n", "data/train.jsonl": b"tasks\n"})
    bench = data_root / "gaia2"
    (bench / "data").mkdir(parents=True)
    (bench / "traces.otel.jsonl").write_text("local-waves\n")
    (bench / "data" / "train.jsonl").write_text("local-edit\n")

    fetch_corpus("gaia2")
    assert (bench / "traces.otel.jsonl").read_text() == "local-waves\n"  # kept
    assert (bench / "data" / "train.jsonl").read_text() == "local-edit\n"  # kept

    fetch_corpus("gaia2", force=True)
    assert (bench / "traces.otel.jsonl").read_text() == "published\n"
    assert (bench / "data" / "train.jsonl").read_text() == "tasks\n"


def test_fetch_resumes_missing_files_inside_an_existing_dir(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted fetch that materialized only part of a data dir picks up the missing
    files on re-run — dir presence alone must not mean 'complete'."""
    _fake_hub(
        monkeypatch,
        {
            "traces.otel.jsonl": b"spans\n",
            "data/train.jsonl": b"tasks\n",
            "data/test.jsonl": b"held-out\n",
        },
    )
    bench = data_root / "gaia2"
    (bench / "data").mkdir(parents=True)
    (bench / "traces.otel.jsonl").write_text("local\n")
    (bench / "data" / "train.jsonl").write_text("already-here\n")

    fetch_corpus("gaia2")
    assert (bench / "data" / "train.jsonl").read_text() == "already-here\n"  # kept
    assert (bench / "data" / "test.jsonl").read_bytes() == b"held-out\n"  # resumed


def test_fetch_with_dest_writes_only_the_corpus_file(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_hub(monkeypatch, {"traces.otel.jsonl": b"spans\n", "data/train.jsonl": b"tasks\n"})
    dest = tmp_path / "elsewhere" / "corpus.jsonl"
    assert fetch_corpus("gaia2", dest=dest) == dest
    assert dest.read_bytes() == b"spans\n"
    assert not (data_root / "gaia2" / "data").exists()


def test_fetch_falls_back_to_the_legacy_repo_name(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The org's datasets are still published under the pre-rename ``wmh-`` name (the code
    renamed wmh -> wmo, the Hub repos did not), so the canonical id 404s and the whole bundle
    must come from the legacy repo instead of failing the download.

    This is the fix vendoring exists to ship: it landed in the member's 0.1.1, which was never
    published, so until `wmo` owned this code no pip user's `wmo download` could resolve a
    single one of the org's datasets.
    """
    canonical, legacy = candidate_repo_ids("gaia2")
    calls = _fake_hub(
        monkeypatch,
        {"traces.otel.jsonl": b"spans\n", "data/train.jsonl": b"tasks\n"},
        live_repos={legacy},
    )

    path = fetch_corpus("gaia2")

    assert path.read_bytes() == b"spans\n"
    assert (data_root / "gaia2" / "data" / "train.jsonl").read_bytes() == b"tasks\n"
    assert calls.trees == [canonical, legacy]
    # every file streams from the repo that actually resolved, not the preferred name
    assert set(calls.resolves) == {legacy}


def test_fetch_asks_once_when_the_canonical_repo_resolves(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch-and-retry, not probe-then-fetch: once the Hub repos are renamed the fallback
    costs nothing, because a resolving canonical id is never followed by a legacy lookup."""
    canonical, _legacy = candidate_repo_ids("gaia2")
    calls = _fake_hub(monkeypatch, {"traces.otel.jsonl": b"spans\n"}, live_repos={canonical})

    fetch_corpus("gaia2")

    assert calls.trees == [canonical]


@pytest.mark.parametrize("code", [404, 401])
def test_fetch_names_every_repo_id_it_tried_when_none_resolve(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """A miss on BOTH names is a real error (404 anonymously, 401 with a token attached), and
    it has to say which ids were looked for or the user cannot tell what to publish."""
    calls = _fake_hub(
        monkeypatch, {"traces.otel.jsonl": b"spans\n"}, live_repos=set(), missing_code=code
    )

    with pytest.raises(CorpusRepoUnavailable) as caught:
        fetch_corpus("gaia2")

    assert isinstance(caught.value, urllib.error.HTTPError)  # front-ends still catch it
    assert caught.value.code == code
    assert caught.value.attempted == candidate_repo_ids("gaia2")
    assert all(repo_id in str(caught.value) for repo_id in candidate_repo_ids("gaia2"))
    assert calls.trees == list(candidate_repo_ids("gaia2"))


def test_fetch_does_not_try_another_name_on_a_non_missing_hub_error(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate limiting is the Hub misbehaving, not the wrong repo name: surface it as-is rather
    than burning a second request and reporting it as an unpublished corpus."""
    calls = _fake_hub(
        monkeypatch, {"traces.otel.jsonl": b"spans\n"}, live_repos=set(), missing_code=429
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch_corpus("gaia2")

    assert not isinstance(caught.value, CorpusRepoUnavailable)
    assert caught.value.code == 429
    assert calls.trees == [candidate_repo_ids("gaia2")[0]]


def test_fetch_names_a_repo_missing_its_corpus(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch, {"data/train.jsonl": b"tasks\n"})
    with pytest.raises(ValueError, match="never pushed"):
        fetch_corpus("gaia2")


def test_fetch_unknown_benchmark_names_the_available_ones(data_root: Path) -> None:
    with pytest.raises(ValueError, match="no published corpus"):
        fetch_corpus("nope")


def test_fetch_rejects_a_transfer_the_tree_says_is_short(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short read is a corrupt corpus, not a small one.

    It must RAISE, with a type `wmo download` can tell apart from an outage (it catches
    `OSError` to keep one bad bundle from stranding a batch), and it must leave nothing behind:
    a fetch skips any path that exists, so a truncated file at the destination would make the
    error's own "re-run the fetch" remedy answer `kept local` over a corpus missing most of its
    spans, at exit 0.
    """
    _fake_hub(
        monkeypatch,
        {"traces.otel.jsonl": b"spans\n"},
        claimed_sizes={"traces.otel.jsonl": 4096},
    )
    with pytest.raises(OSError, match="truncated transfer") as caught:
        fetch_corpus("gaia2")
    assert not isinstance(caught.value, urllib.error.URLError)  # not mistaken for an outage
    assert "traces.otel.jsonl" in str(caught.value)  # a bundle is many files; name the one
    corpus = data_root / "gaia2" / "traces.otel.jsonl"
    assert not corpus.exists()  # nothing to make the next fetch think it is done
    assert not corpus.with_name(corpus.name + ".part").exists()


def test_a_truncated_forced_refresh_keeps_the_corpus_it_was_replacing(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--force` over a good local corpus must not be able to destroy it.

    The size is checked BEFORE the `.part` is renamed over the destination, so a short transfer
    is a no-op. Checking after would already have clobbered a valid corpus with a truncated one
    — and then deleting that to keep the re-run honest turns a failed refresh into data loss.
    """
    _fake_hub(
        monkeypatch,
        {"traces.otel.jsonl": b"spans\n"},
        claimed_sizes={"traces.otel.jsonl": 4096},
    )
    corpus = data_root / "gaia2" / "traces.otel.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("the corpus that was already here\n")

    with pytest.raises(OSError, match="truncated transfer"):
        fetch_corpus("gaia2", force=True)

    assert corpus.read_text() == "the corpus that was already here\n"
    assert not corpus.with_name(corpus.name + ".part").exists()


def test_unknown_benchmark_never_offers_an_unpublished_name(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offering a name the Hub can only answer 401 for sends the user down a dead end, so the
    # "available:" list is the published subset, not the whole registry.
    #
    # The unpublished entry is registered here rather than borrowed from the shipped registry.
    # It used to be a real corpus, so retiring that corpus turned this test into a no-op via its
    # own "meaningless" guard - the guard fired, which is exactly what it was for.
    monkeypatch.setitem(
        CORPORA,
        "never-pushed",
        CorpusSpec(
            benchmark="never-pushed",
            published=False,
            license_id="mit",
            upstream="synthetic fixture",
            description="A registered bundle whose Hub push never landed.",
        ),
    )
    unpublished = sorted(name for name, spec in CORPORA.items() if not spec.published)
    assert unpublished, "this test is meaningless once every registered corpus is published"
    with pytest.raises(ValueError) as caught:
        fetch_corpus("nope")
    for name in unpublished:
        assert name not in str(caught.value)
        assert name not in downloadable_benchmarks()


def test_fetch_of_an_unpublished_benchmark_fails_offline_and_says_why(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No Hub round trip at all: a registered-but-unpushed bundle is knowable locally, so the
    # user gets the reason instead of a 401 they cannot act on.
    #
    # The unpublished entry is synthetic. It used to be a real registered benchmark, which made
    # the test hostage to that bundle's fate: retiring it would have deleted the coverage.
    # Registering one here keeps the behaviour pinned whether or not any shipped corpus is
    # currently unpublished.
    unpublished = CorpusSpec(
        benchmark="never-pushed",
        published=False,
        license_id="mit",
        upstream="synthetic fixture",
        description="A registered bundle whose Hub push never landed.",
    )
    monkeypatch.setitem(CORPORA, unpublished.benchmark, unpublished)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not touch the Hub for an unpublished benchmark")

    monkeypatch.setattr(hub, "_resolve_repo", explode)
    with pytest.raises(ValueError, match="has not been published to the Hub yet"):
        fetch_corpus(unpublished.benchmark)


def test_stream_to_is_atomic(tmp_path: Path) -> None:
    """The real streamer writes a .part sibling and renames over — a partial download must
    never be mistaken for a complete corpus by a concurrent reader."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (3 * 1024))
    dest = tmp_path / "out" / "corpus.jsonl"
    seen: list[int] = []

    hub._stream_to(source.as_uri(), dest, token=None, chunk_done=seen.append)

    assert dest.read_bytes() == b"x" * (3 * 1024)
    assert not dest.with_name(dest.name + ".part").exists()
    assert sum(seen) == 3 * 1024


def test_published_corpora_maps_repos_to_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [
        {"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-07-07T06:00:00.000Z"},
        {
            "id": "experiential-labs/wmo-bird-sql-traces",
            "lastModified": "2026-07-05T00:00:00.000Z",
        },
        {"id": "experiential-labs/unrelated-dataset", "lastModified": "2026-07-06T00:00:00.000Z"},
        {"id": "experiential-labs/wmo-not-a-benchmark-traces", "lastModified": ""},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    published = published_corpora()
    assert [(c.benchmark, c.last_modified) for c in published] == [
        ("gaia2", "2026-07-07"),
        ("bird-sql", "2026-07-05"),
    ]
    assert published[0].repo_id == candidate_repo_ids("gaia2")[0]


def test_published_corpora_accepts_the_legacy_repo_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wmo download` with no arguments lists what the org publishes; everything it publishes
    today still carries the pre-rename ``wmh-`` prefix, and dropping those empties the picker."""
    listing = [
        {"id": "experiential-labs/wmh-gaia2-traces", "lastModified": "2026-07-07T06:00:00.000Z"},
        {
            "id": "experiential-labs/wmh-bird-sql-traces",
            "lastModified": "2026-07-05T00:00:00.000Z",
        },
        {"id": "experiential-labs/unrelated-dataset", "lastModified": "2026-07-06T00:00:00.000Z"},
        {"id": "experiential-labs/wmh-not-a-benchmark-traces", "lastModified": ""},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    published = published_corpora()
    assert [(c.benchmark, c.repo_id) for c in published] == [
        ("gaia2", "experiential-labs/wmh-gaia2-traces"),
        ("bird-sql", "experiential-labs/wmh-bird-sql-traces"),
    ]


def test_published_corpora_lists_a_double_published_benchmark_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-rename both repo names can exist at once; the picker shows one row per benchmark,
    under the canonical id (which is also the one a fetch resolves first)."""
    listing = [
        {"id": "experiential-labs/wmh-gaia2-traces", "lastModified": "2026-07-01T00:00:00.000Z"},
        {"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-07-07T00:00:00.000Z"},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    assert [(c.benchmark, c.repo_id) for c in published_corpora()] == [
        ("gaia2", candidate_repo_ids("gaia2")[0])
    ]


def test_published_corpora_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """An org with more datasets than one page must not hide corpora beyond page 1."""
    pages = {
        "page1": (
            [{"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-07-07T00:00:00Z"}],
            "page2",
        ),
        "page2": (
            [
                {
                    "id": "experiential-labs/wmo-bird-sql-traces",
                    "lastModified": "2026-07-06T00:00:00Z",
                }
            ],
            None,
        ),
    }

    def page(url: str, *, token: str | None) -> tuple[object, str | None]:
        key = "page2" if url == "page2" else "page1"
        return pages[key]

    monkeypatch.setattr(hub, "_http_json_page", page)
    assert [c.benchmark for c in published_corpora()] == ["gaia2", "bird-sql"]


def test_data_root_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Env override wins; a checkout resolves the CAPTURE MEMBER's dir; an installed wheel lands
    bundles under the CWD, never inside site-packages.

    The checkout branch is the line vendoring could not copy: the member finds that directory as
    its own parent, `wmo/hub.py` is one level shallower and has to name it. Getting it wrong
    would silently point the flagship at the repo root, where no benchmark dir exists.
    """
    monkeypatch.setenv("ENVCAP_DATA_ROOT", str(tmp_path / "override"))
    assert hub._data_root() == tmp_path / "override"

    monkeypatch.delenv("ENVCAP_DATA_ROOT")
    checkout = hub._data_root()
    assert checkout == Path(hub.__file__).resolve().parents[1] / "packages" / "environment-capture"
    assert (checkout / "pyproject.toml").exists()

    site = tmp_path / "venv" / "site-packages" / "wmo"
    site.mkdir(parents=True)
    monkeypatch.setattr(hub, "__file__", str(site / "hub.py"))
    monkeypatch.chdir(tmp_path)
    assert hub._data_root() == tmp_path / "environment-capture-data"


#: Source regions `wmo/hub.py` copied verbatim from the member and must keep copying verbatim,
#: as (start marker, end marker) slices. Everything the two copies genuinely share lives here:
#: the spec dataclass and the registry, the repo-name convention, and the prefix constants
#: behind it. The rest of each file is allowed to differ — that is what "narrowed to what `wmo`
#: consumes" means, and `_data_root` is deliberately not the same line in both.
_SHARED_SOURCE_REGIONS = (
    ("class CorpusSpec:", "class PublishedCorpus:"),
    ("def candidate_repo_ids", "def _benchmark_from_repo_name"),
    ("_REPO_PREFIXES", "_MISSING_REPO_CODES"),
)


def test_the_vendored_copy_has_not_drifted_from_its_origin() -> None:
    """A vendored copy that drifts from its origin is worse than the import it replaced.

    A benchmark registered on one side only makes `wmo download` and the member's capture
    disagree about what exists; a repo-id rule that differs makes them look for the same bundle
    under different dataset names. Both are silent until someone cannot find their data.

    Compared as SOURCE TEXT, read off disk, rather than by importing the member: `wmo/` imports
    nothing from `packages/`, tests included, and this test is not entitled to an exemption from
    the rule the rest of the PR exists to establish. Text is also the stricter comparison — it
    catches a comment or a docstring going stale, which a value check waves through.
    """
    copy_path = Path(hub.__file__).resolve()
    member = (
        copy_path.parents[1] / "packages" / "environment-capture" / "environment_capture/hub.py"
    )
    if not member.is_file():  # installed sdist, or the member is gone: nothing to compare
        pytest.skip("the environment-capture member is not present in this tree")
    origin = member.read_text(encoding="utf-8")
    copy = copy_path.read_text(encoding="utf-8")
    for start, end in _SHARED_SOURCE_REGIONS:
        assert _region(copy, start, end) == _region(origin, start, end), (
            f"wmo/hub.py drifted from the member's hub.py in the region {start!r}..{end!r}; "
            "the two are hand-mirrored copies, so make the same edit in both (see the VENDORED "
            "note at the top of wmo/hub.py)"
        )


def _region(source: str, start: str, end: str) -> str:
    """The text of `source` from the line introducing `start` up to the one introducing `end`."""
    assert start in source, f"marker {start!r} is gone; update _SHARED_SOURCE_REGIONS"
    assert end in source, f"marker {end!r} is gone; update _SHARED_SOURCE_REGIONS"
    return source[source.index(start) : source.index(end)]
