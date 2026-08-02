"""Pure repository-tree features for the frozen graded router follow-up.

The module accepts only commit-addressed Git tree metadata and the initial issue text. It has no
network, filesystem, outcome, or serialization surface. All rules below are frozen before the
first repository tree is acquired.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROTOCOL = "coding-router-graded-repository-tree-v1"
BM25_K1 = 1.2
BM25_B = 0.75

EXCLUDED_DIRECTORY_COMPONENTS = frozenset(
    {
        ".git",
        ".tox",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "target",
        "third-party",
        "third_party",
        "vendor",
        "vendors",
        "venv",
    }
)
SOURCE_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".clj",
        ".cljs",
        ".cmake",
        ".cpp",
        ".cs",
        ".dart",
        ".erl",
        ".ex",
        ".exs",
        ".fs",
        ".fsx",
        ".go",
        ".h",
        ".hh",
        ".hpp",
        ".hs",
        ".java",
        ".jl",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".lua",
        ".m",
        ".mm",
        ".php",
        ".pl",
        ".pm",
        ".py",
        ".r",
        ".rb",
        ".rs",
        ".scala",
        ".sh",
        ".sol",
        ".swift",
        ".ts",
        ".tsx",
        ".vue",
        ".zig",
    }
)
TEST_COMPONENTS = frozenset(
    {"__tests__", "benchmark", "benchmarks", "spec", "specs", "test", "tests"}
)
DOCUMENTATION_COMPONENTS = frozenset(
    {"doc", "docs", "documentation", "guide", "guides", "manual"}
)
DOCUMENTATION_EXTENSIONS = frozenset(
    {".adoc", ".md", ".mdx", ".rst", ".tex", ".txt"}
)
CONFIGURATION_EXTENSIONS = frozenset(
    {".cfg", ".conf", ".ini", ".json", ".toml", ".xml", ".yaml", ".yml"}
)
CONFIGURATION_BASENAMES = frozenset(
    {
        ".babelrc",
        ".editorconfig",
        ".eslintrc",
        ".flake8",
        ".prettierrc",
        "biome.json",
        "mypy.ini",
        "pyrightconfig.json",
        "ruff.toml",
        "tsconfig.json",
    }
)
EXAMPLE_COMPONENTS = frozenset({"demo", "demos", "example", "examples", "sample", "samples"})
GENERATED_EXTENSIONS = frozenset(
    {".class", ".dll", ".dylib", ".jar", ".map", ".min.css", ".min.js", ".o", ".pyc", ".so"}
)
PACKAGE_MANIFESTS = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "cargo.toml",
        "composer.json",
        "gemfile",
        "go.mod",
        "package.json",
        "pom.xml",
        "project.toml",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
    }
)
LOCK_FILES = frozenset(
    {
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "go.sum",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }
)
SOURCE_ROOTS = frozenset({"app", "cmd", "include", "lib", "pkg", "src"})
TEST_ROOTS = frozenset(TEST_COMPONENTS)
LANGUAGE_EXTENSIONS: dict[str, frozenset[str]] = {
    "c": frozenset({".c", ".h"}),
    "c#": frozenset({".cs"}),
    "c++": frozenset({".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}),
    "go": frozenset({".go"}),
    "java": frozenset({".java"}),
    "javascript": frozenset({".js", ".jsx", ".mjs", ".cjs"}),
    "julia": frozenset({".jl"}),
    "php": frozenset({".php"}),
    "python": frozenset({".py", ".pyi"}),
    "ruby": frozenset({".rb"}),
    "rust": frozenset({".rs"}),
    "swift": frozenset({".swift"}),
    "typescript": frozenset({".ts", ".tsx"}),
}
BUILD_MARKER_ORDER = (
    "python-packaging",
    "node",
    "typescript",
    "go",
    "rust",
    "java-maven",
    "java-gradle",
    "bazel",
    "cmake",
    "make",
    "ruby",
    "php",
    "dotnet",
    "julia",
    "swift",
    "mixed-language",
    "monorepo",
)
CI_COMPONENTS = frozenset({".circleci", ".github", ".gitlab", "ci"})

_TOKEN_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_ALPHANUMERIC = re.compile(r"[a-z0-9]+")
_BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")
_SLASH_SPAN = re.compile(r"(?<![\w.-])([\w.@+-]+(?:/[\w.@+-]+)+)")
_FILENAME_SPAN = re.compile(r"(?<![\w/.-])([\w@+-]+(?:\.[\w@+-]+)+)")


@dataclass(frozen=True)
class RawTreeEntry:
    """One untrusted row returned by a recursive Git tree query."""

    path: str
    object_type: str
    mode: str
    size: int | None


@dataclass(frozen=True)
class TreeFile:
    """One validated, normalized file row allowed into feature construction."""

    path: str
    mode: str
    size: int

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.path.split("/"))

    @property
    def basename(self) -> str:
        return self.components[-1]

    @property
    def extension(self) -> str:
        suffixes = PurePosixPath(self.basename).suffixes
        if len(suffixes) >= 2 and "".join(suffixes[-2:]) in GENERATED_EXTENSIONS:
            return "".join(suffixes[-2:])
        return suffixes[-1] if suffixes else ""

    @property
    def executable(self) -> bool:
        return self.mode == "100755"

    @property
    def symlink(self) -> bool:
        return self.mode == "120000"


@dataclass(frozen=True)
class RepositoryFeatureBlocks:
    """All frozen raw feature blocks for one task."""

    structure: np.ndarray
    localization: np.ndarray
    prompt_shape: np.ndarray


def _normalize_path(path: str) -> str:
    normalized = unicodedata.normalize("NFKC", path).replace("\\", "/").casefold()
    if not normalized or normalized.startswith("/"):
        raise ValueError("tree path must be nonempty and relative")
    components = normalized.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("tree path contains an empty or traversal component")
    return "/".join(components)


def validate_tree(entries: Sequence[RawTreeEntry], *, truncated: bool) -> tuple[TreeFile, ...]:
    """Validate one raw recursive Git tree and retain only eligible file metadata."""
    if truncated:
        raise ValueError("truncated Git trees are forbidden")
    files: list[TreeFile] = []
    seen: set[str] = set()
    for entry in entries:
        path = _normalize_path(entry.path)
        if entry.object_type == "commit" or entry.mode == "160000":
            raise ValueError("submodule objects are forbidden")
        if entry.object_type == "tree":
            continue
        if entry.object_type != "blob" or entry.mode not in {"100644", "100755", "120000"}:
            raise ValueError("unsupported Git tree object")
        if entry.size is None or entry.size < 0:
            raise ValueError("blob size must be a nonnegative integer")
        if path in seen:
            raise ValueError("normalized Git tree paths must be unique")
        seen.add(path)
        components = path.split("/")
        if any(component in EXCLUDED_DIRECTORY_COMPONENTS for component in components[:-1]):
            continue
        files.append(TreeFile(path=path, mode=entry.mode, size=entry.size))
    if not files:
        raise ValueError("Git tree has no eligible files")
    return tuple(sorted(files, key=lambda file: file.path))


def _tokens(text: str) -> tuple[str, ...]:
    expanded = _TOKEN_BOUNDARY.sub(" ", unicodedata.normalize("NFKC", text))
    terms = _ALPHANUMERIC.findall(expanded.replace("_", " ").replace("-", " ").casefold())
    return tuple(term for term in terms if len(term) >= 2 and term not in ENGLISH_STOP_WORDS)


def _prompt_shape_row(text: str) -> list[float]:
    """Return the existing frozen 15-value SWE-smith prompt-shape block."""
    lower = text.casefold()
    lines = text.splitlines()
    path_tokens = sum(token.count("/") for token in text.split())
    stack_markers = sum(
        lower.count(marker)
        for marker in ("traceback", "stack trace", " at ", "exception:")
    )
    return [
        math.log1p(len(text)),
        math.log1p(len(text.split())),
        math.log1p(len(lines)),
        math.log1p(text.count("```")),
        math.log1p(stack_markers),
        math.log1p(path_tokens),
        math.log1p(text.count("`") + text.count('"') + text.count("'")),
        math.log1p(sum(lower.count(word) for word in ("fix", "bug", "repair"))),
        math.log1p(lower.count("test")),
        math.log1p(
            sum(lower.count(word) for word in ("dependency", "package", "build"))
        ),
        float("python" in lower or ".py" in lower),
        float("javascript" in lower or ".js" in lower or "node" in lower),
        float("typescript" in lower or ".ts" in lower),
        float("rust" in lower or ".rs" in lower or "cargo" in lower),
        float("golang" in lower or ".go" in lower),
    ]


def _category(file: TreeFile) -> str:
    components = set(file.components[:-1])
    basename = file.basename
    stem = PurePosixPath(basename).stem
    extension = file.extension
    if extension in GENERATED_EXTENSIONS:
        return "generated"
    if (
        components & TEST_COMPONENTS
        or stem.startswith("test_")
        or stem.endswith(("_test", ".test", ".spec"))
    ):
        return "test"
    if components & DOCUMENTATION_COMPONENTS or extension in DOCUMENTATION_EXTENSIONS:
        return "documentation"
    if basename in CONFIGURATION_BASENAMES or extension in CONFIGURATION_EXTENSIONS:
        return "configuration"
    if components & EXAMPLE_COMPONENTS:
        return "examples"
    if extension in SOURCE_EXTENSIONS:
        return "source"
    return "unclassified"


def _entropy(values: Iterable[float]) -> float:
    positive = np.asarray([value for value in values if value > 0.0], dtype=np.float64)
    if positive.size <= 1:
        return 0.0
    probabilities = positive / float(np.sum(positive))
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(positive.size))


def _summary(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [
        float(np.mean(array)),
        float(np.std(array)),
        float(np.median(array)),
        float(np.quantile(array, 0.9)),
        float(np.max(array)),
    ]


def _build_markers(files: Sequence[TreeFile]) -> tuple[dict[str, float], int]:
    basenames = {file.basename for file in files}
    paths = {file.path for file in files}
    extensions = {file.extension for file in files if file.extension in SOURCE_EXTENSIONS}
    markers = {
        "python-packaging": bool(basenames & {"pyproject.toml", "setup.cfg", "setup.py"}),
        "node": "package.json" in basenames,
        "typescript": "tsconfig.json" in basenames,
        "go": "go.mod" in basenames,
        "rust": "cargo.toml" in basenames,
        "java-maven": "pom.xml" in basenames,
        "java-gradle": bool(
            basenames
            & {
                "build.gradle",
                "build.gradle.kts",
                "settings.gradle",
                "settings.gradle.kts",
            }
        ),
        "bazel": bool(
            basenames & {"build.bazel", "module.bazel", "workspace", "workspace.bazel"}
        ),
        "cmake": "cmakelists.txt" in basenames,
        "make": bool(basenames & {"gnumakefile", "makefile"}),
        "ruby": bool(basenames & {"gemfile"})
        or any(name.endswith(".gemspec") for name in basenames),
        "php": "composer.json" in basenames,
        "dotnet": any(file.extension in {".csproj", ".fsproj", ".sln"} for file in files),
        "julia": "project.toml" in basenames and ".jl" in extensions,
        "swift": "package.swift" in basenames,
        "mixed-language": len(extensions) >= 3,
        "monorepo": len(
            {
                PurePosixPath(path).parent.parts[0]
                for path in paths
                if "/" in path and PurePosixPath(path).name in PACKAGE_MANIFESTS
            }
        )
        >= 2,
    }
    build_count = sum(bool(markers[name]) for name in BUILD_MARKER_ORDER[:15])
    return {name: float(markers[name]) for name in BUILD_MARKER_ORDER}, build_count


def structure_features(files: Sequence[TreeFile], language: str) -> np.ndarray:
    """Return the frozen repository-structure feature block."""
    if not files:
        raise ValueError("repository features require at least one file")
    count = len(files)
    depths = [float(len(file.components)) for file in files]
    log_sizes = [math.log1p(file.size) for file in files]
    categories = Counter(_category(file) for file in files)
    extensions = Counter(file.extension for file in files if file.extension)
    language_extensions = LANGUAGE_EXTENSIONS.get(language.casefold(), frozenset())
    top_level = {file.components[0] for file in files if len(file.components) > 1}
    basenames = {file.basename for file in files}
    markers, build_count = _build_markers(files)
    category_order = (
        "source",
        "test",
        "documentation",
        "configuration",
        "examples",
        "generated",
        "unclassified",
    )
    source = categories["source"]
    tests = categories["test"]
    docs = categories["documentation"]
    ratios = [
        float(source / tests) if tests else 0.0,
        float(source / docs) if docs else 0.0,
        float(categories["configuration"] / count),
        float(tests == 0),
        float(docs == 0),
    ]
    extension_entropy = _entropy(extensions.values())
    dominant_extension = max(extensions.values(), default=0) / count
    language_fraction = sum(file.extension in language_extensions for file in files) / count
    ci_count = sum(bool(set(file.components[:-1]) & CI_COMPONENTS) for file in files)
    values = [
        math.log1p(count),
        math.log1p(sum(file.size for file in files)),
        sum(file.executable for file in files) / count,
        sum(file.symlink for file in files) / count,
        *_summary(depths),
        *_summary(log_sizes),
        *(categories[name] / count for name in category_order),
        *ratios,
        float(len(extensions)),
        extension_entropy,
        dominant_extension,
        language_fraction,
        *(markers[name] for name in BUILD_MARKER_ORDER),
        float(len(top_level)),
        float(len(top_level & SOURCE_ROOTS)),
        float(len(top_level & TEST_ROOTS)),
        float(len(basenames & PACKAGE_MANIFESTS)),
        float(len(basenames & LOCK_FILES)),
        float(ci_count),
        float(build_count),
        *(sum(depth == bucket for depth in depths) / count for bucket in (1.0, 2.0, 3.0, 4.0)),
        sum(depth >= 5.0 for depth in depths) / count,
        float(not language_extensions),
        float(not extensions),
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError("repository structure features are invalid")
    return result


def _bm25(issue_tokens: Sequence[str], documents: Sequence[Sequence[str]]) -> np.ndarray:
    if not documents:
        return np.empty(0, dtype=np.float64)
    document_frequency = Counter(term for document in documents for term in set(document))
    average_length = float(np.mean([len(document) for document in documents])) or 1.0
    query = Counter(issue_tokens)
    scores: list[float] = []
    total_documents = len(documents)
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term, query_frequency in query.items():
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (total_documents - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * len(document) / average_length
            )
            score += (
                query_frequency
                * inverse_frequency
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def _score_block(scores: np.ndarray, *, full: bool) -> list[float]:
    positive = scores[scores > 0.0]
    ordered = np.sort(scores)[::-1]
    maximum = float(ordered[0]) if ordered.size else 0.0
    second = float(ordered[1]) if ordered.size > 1 else 0.0
    concentration = maximum / float(np.sum(positive)) if positive.size else 0.0
    common = [maximum, maximum - second, concentration, float(positive.size), _entropy(positive)]
    if not full:
        return common + [float(scores.size == 0)]
    return [
        maximum,
        float(np.mean(scores)) if scores.size else 0.0,
        float(np.std(scores)) if scores.size else 0.0,
        float(np.quantile(scores, 0.9)) if scores.size else 0.0,
        float(np.quantile(scores, 0.99)) if scores.size else 0.0,
        maximum - second,
        concentration,
        float(positive.size),
        float(positive.size / scores.size) if scores.size else 0.0,
        _entropy(positive),
        float(scores.size == 0),
    ]


def _span_features(issue: str, files: Sequence[TreeFile]) -> list[float]:
    span_groups = (
        _BACKTICK_SPAN.findall(issue),
        _SLASH_SPAN.findall(issue),
        _FILENAME_SPAN.findall(issue),
    )
    spans = [
        unicodedata.normalize("NFKC", span).casefold().strip("`'\".,:;()[]{}")
        for group in span_groups
        for span in group
    ]
    paths = {file.path for file in files}
    basenames = {file.basename for file in files}
    stems = {PurePosixPath(file.basename).stem for file in files}
    extensions = {file.extension.lstrip(".") for file in files if file.extension}
    directories = {component for file in files for component in file.components[:-1]}
    return [
        *(float(len(group)) for group in span_groups),
        float(len(spans)),
        float(sum(span in paths for span in spans)),
        float(
            sum(
                any(path.endswith(f"/{span}") or path == span for path in paths)
                for span in spans
            )
        ),
        float(sum(span in basenames for span in spans)),
        float(sum(span in stems for span in spans)),
        float(sum(span.lstrip(".") in extensions for span in spans)),
        float(sum(span in directories for span in spans)),
    ]


def localization_features(files: Sequence[TreeFile], issue: str) -> np.ndarray:
    """Return frozen issue-to-path BM25 and exact-match features."""
    issue_tokens = _tokens(issue)
    documents = [_tokens(file.path.replace("/", " ")) for file in files]
    scores = _bm25(issue_tokens, documents)
    categories = [_category(file) for file in files]
    source_indices = np.asarray(
        [index for index, category in enumerate(categories) if category == "source"],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, category in enumerate(categories) if category == "test"],
        dtype=np.int64,
    )
    path_token_union = {token for document in documents for token in document}
    issue_token_set = set(issue_tokens)
    ranked = np.argsort(-scores, kind="stable")
    top_directory_counts = []
    for limit in (1, 3, 5, 10):
        top_directory_counts.append(
            float(len({files[index].components[0] for index in ranked[:limit]}))
        )
    values = [
        *_score_block(scores, full=True),
        *_score_block(scores[source_indices], full=False),
        *_score_block(scores[test_indices], full=False),
        *_span_features(issue, files),
        float(len(issue_token_set & path_token_union) / len(issue_token_set))
        if issue_token_set
        else 0.0,
        float(not issue_token_set),
        *top_directory_counts,
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError("repository localization features are invalid")
    return result


def feature_blocks(
    files: Sequence[TreeFile], *, issue: str, language: str
) -> RepositoryFeatureBlocks:
    """Construct all three unstandardized frozen feature blocks."""
    prompt = np.asarray(_prompt_shape_row(issue), dtype=np.float64)
    result = RepositoryFeatureBlocks(
        structure=structure_features(files, language),
        localization=localization_features(files, issue),
        prompt_shape=prompt,
    )
    if result.prompt_shape.shape != (15,) or not np.isfinite(result.prompt_shape).all():
        raise RuntimeError("prompt-shape block is invalid")
    return result


def feature_views(blocks: RepositoryFeatureBlocks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three frozen nested feature views in grid order."""
    structure = blocks.structure.copy()
    localization = np.concatenate([structure, blocks.localization])
    prompt = np.concatenate([localization, blocks.prompt_shape])
    return structure, localization, prompt
