"""Project config + the `.wmh/` artifact layout.

`.wmh/` holds everything `wmh build` produces and `wmh serve` / `WorldModel.load` consume.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field, JsonValue, ValidationError

from wmh.core.types import JsonObject
from wmh.providers.base import EmbedderKind, ProviderConfig, ProviderKind
from wmh.providers.models import resolve_provider_model

ARTIFACT_DIR = ".wmh"


class FidelityTier(StrEnum):
    """Build-effort tiers: how much is spent making the artifact faithful.

    The build command exposes ONLY this knob (raw iteration counts live in the Python API):
    - low: RAG only — index the traces, ship the base prompt. Fast and near-free.
    - medium: RAG + a light GEPA pass over the prompt.
    - high: RAG + full GEPA + a cheap auto-config search (candidates pruned by corpus signature).
    - max: RAG + deep GEPA + the full auto-config ladder, scored on more held-out traces.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


@dataclass(frozen=True)
class TierSpec:
    """What one fidelity tier spends: GEPA rollouts, the auto-config search, retrieval phi."""

    # GEPA optimization ITERATIONS (candidates proposed+evaluated). Each iteration costs about
    # `minibatch + gepa_val_cap` metric calls (predict+judge pairs), so cost is bounded per tier
    # regardless of corpus size — an uncapped valset once turned "budget 50" into ~7000 calls,
    # and a TRACE-denominated cap let one long-trace corpus (swe: ~7 huge steps/trace) burn $131
    # on a single "4-iteration" tier. STEPS are the unit that actually bounds cost.
    gepa_budget: int
    gepa_val_cap: int  # STEPS GEPA selects candidates on (caps per-iteration cost)
    config_search: bool
    search_budget: int  # held-out traces scored per candidate
    full_ladder: bool  # False = prune candidates by corpus signature
    # True = search only the CHEAP frontier (base/reason/grounding class — levers that cost
    # roughly nothing extra to serve). Tiers ration the expensive knobs (GEPA iterations,
    # kb/verify scoring), but a nearly-free grounding win should be discoverable cheaply.
    cheap_frontier_only: bool
    # True = ship the corpus-signature's strongest ESTIMATED config with no LLM search (the
    # `low` tier). It's the ladder's FLOOR: every searching tier seeds its incumbent from this
    # same estimate and only replaces it on a clear win, so no searching tier ships worse than
    # low. (This guarantees "tier >= low", NOT "high >= medium" — medium and high search
    # different menus on different samples, so adjacent tiers are not strictly ordered; the
    # floor is at low, not at the previous tier.)
    estimate_only: bool
    # Recommend provider-backed semantic phi. Kept as a field for explicit opt-in experiments,
    # but NO tier sets it: semantic retrieval was measured WORSE than lexical hashing on every
    # benchmark (PR #72 matrix: ada-002 terminal 0.790 vs hashing 0.818, swe 0.635 vs 0.640 —
    # command outputs are predicted by literal token overlap, which char-trigrams capture and
    # semantics blur), and the tier ladder's only semantic cells coincide with tau's decline
    # (medium 0.891 hashing -> high 0.886 / max 0.882 semantic).
    semantic_embeddings: bool


FIDELITY_TIERS: dict[FidelityTier, TierSpec] = {
    # Low no longer means "plain base RAG": it ships the signature's strongest ESTIMATED
    # config (reason / +fetch / +workspace / +kb by corpus shape — see `signature_estimate`)
    # with zero LLM search. Free, and a strong floor the higher tiers can only build on.
    FidelityTier.LOW: TierSpec(
        gepa_budget=0,
        gepa_val_cap=0,
        config_search=False,
        search_budget=0,
        full_ladder=False,
        cheap_frontier_only=False,
        estimate_only=True,
        semantic_embeddings=False,
    ),
    # Medium still searches the CHEAP frontier: grounding levers serve at ~base cost and score
    # in a handful of traces, so even a budget tier can discover a workspace/fetch win. What
    # medium rations is the expensive knobs — GEPA iterations and kb/verify scoring.
    FidelityTier.MEDIUM: TierSpec(
        gepa_budget=4,
        gepa_val_cap=24,
        config_search=True,
        search_budget=4,
        full_ladder=False,
        cheap_frontier_only=True,
        estimate_only=False,
        semantic_embeddings=False,
    ),
    # High's GEPA stays at medium's 4 iterations: the 8-iteration increment measured ~noise
    # on every benchmark once the config-search winners' known lifts are subtracted (ladder:
    # tau -0.005, terminal +0.007, swe +0.006), and the GEPA scaling work (PR #97) found the
    # base template's iteration lift ≈0 pre-fix. What high buys over medium: the full
    # signature-pruned candidate menu (kb/verify) instead of the cheap frontier.
    FidelityTier.HIGH: TierSpec(
        gepa_budget=4,
        gepa_val_cap=24,
        config_search=True,
        search_budget=8,
        full_ladder=False,
        cheap_frontier_only=False,
        estimate_only=False,
        semantic_embeddings=False,
    ),
    FidelityTier.MAX: TierSpec(
        gepa_budget=16,
        gepa_val_cap=32,
        config_search=True,
        search_budget=16,
        full_ladder=True,
        cheap_frontier_only=False,
        estimate_only=False,
        semantic_embeddings=False,
    ),
}

# Env var names each provider backend reads its credentials from (documented for the user).
PROVIDER_ENV_VARS: dict[ProviderKind, list[str]] = {
    ProviderKind.ANTHROPIC: ["ANTHROPIC_API_KEY"],
    ProviderKind.BEDROCK: ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    ProviderKind.AZURE_OPENAI: ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
    ProviderKind.OPENAI: ["OPENAI_API_KEY"],
    ProviderKind.OPENAI_RESPONSES: ["OPENAI_API_KEY"],
}


class HarnessConfig(BaseModel):
    """Persisted to `.wmh/config.toml` and reloaded by `wmh serve` / `WorldModel.load`."""

    providers: list[ProviderConfig] = Field(default_factory=list)
    serve_provider: ProviderKind = ProviderKind.ANTHROPIC  # serves the live world model
    # Which embedder supplies phi for retrieval. Defaults to the offline HashingEmbedder (no creds);
    # set to a provider-backed kind (bedrock/openai/azure) for semantic phi.
    embed_provider: EmbedderKind = EmbedderKind.HASHING
    embed_dim: int = 512  # phi dimensionality; index + query embedder must agree on this
    top_k: int = 5  # demos retrieved per step (DreamGym k)
    # train/held-out ratio for GEPA; a proper fraction so both splits can be non-empty
    train_split: float = Field(default=0.8, gt=0.0, lt=1.0)
    gepa_budget: int = 10  # GEPA iterations; ~valset_cap calls each (see _cap_gepa_valset)
    # Model id the GEPA judge runs on (same provider kind as serve). None = the serve model.
    judge_model: str | None = None
    trace_adapter: str = "otel-genai"
    # Agentic-mode flags (all default OFF: artifacts built before these fields serve unchanged).
    # `knowledge`: seed a knowledge base from train traces at build and render it into the env
    # prompt at serve. `reasoning`: deliberate-then-answer output contract. `grounder`: web-search
    # backend for grounding unknown entities ("none" keeps everything hermetic; see
    # `wmh.engine.grounding` for backends).
    knowledge: bool = False
    reasoning: bool = False
    grounder: str = "none"
    # Second self-check completion per step (draft re-examined against the evidence). ~2x serve
    # cost; earns it only where content prediction is hardest (empirically: swe-style suites).
    verify: bool = False
    # Verbalized confidence (WS-A6, D75): the contract asks for a 0.0-1.0 self-assessment of the
    # emitted output (carried in Observation.metadata, never shown to the judge).
    # `confidence_why` adds its one-line justification. Analysis/abstention lever, off by default.
    confidence: bool = False
    confidence_why: bool = False

    def provider_config(self, kind: ProviderKind) -> ProviderConfig:
        """Return the configured ProviderConfig for `kind` (model + backend knobs)."""
        for pc in self.providers:
            if pc.kind == kind:
                return pc
        raise ValueError(
            f"no provider config for {kind.value}; configure it before building/serving "
            f"(have: {[pc.kind.value for pc in self.providers]})"
        )

    def serve_provider_config(self) -> ProviderConfig:
        """The ProviderConfig that serves the live world model."""
        return self.provider_config(self.serve_provider)

    def embed_provider_config(self) -> ProviderConfig:
        """The ProviderConfig backing phi retrieval, with `embed_dim` stamped on.

        Stamping `embed_dim` makes the backend request vectors of exactly the persisted dimension,
        so the index and query embedders agree. Raises for `EmbedderKind.HASHING` (the offline
        embedder has no provider) — guard with `embed_provider is EmbedderKind.HASHING` first.
        """
        config = self.provider_config(self.embed_provider.provider_kind())
        return config.model_copy(update={"embed_dim": self.embed_dim})

    @classmethod
    def for_build(
        cls,
        *,
        serve_provider: ProviderKind,
        serve_model: str,
        region: str | None,
        embed_provider: EmbedderKind,
        embed_model: str | None,
        embed_dim: int,
        gepa_budget: int,
        train_split: float = 0.8,
        judge_model: str | None = None,
        trace_adapter: str = "otel-genai",
    ) -> HarnessConfig:
        """Assemble a build config from the choices `wmh build` collects.

        Owns the one piece of provider wiring: a provider-backed embedder either **reuses** the
        serve provider's config (same backend — just add `embed_model`) or gets **its own**
        `ProviderConfig`. Keeping this here (not in the CLI) makes it unit-testable and gives every
        entry point one place to construct a build config. Callers must already have validated that
        a non-hashing embedder has an `embed_model`.
        """
        serve_spec = resolve_provider_model(serve_provider, serve_model)
        serve = ProviderConfig(
            kind=serve_provider,
            model_type=serve_spec.model_type,
            model=serve_spec.model_id,
            region=region,
        )
        providers = [serve]
        if embed_provider is not EmbedderKind.HASHING:
            embed_kind = embed_provider.provider_kind()
            if embed_kind == serve_provider:
                providers[0] = serve.model_copy(update={"embed_model": embed_model})
            else:
                providers.append(
                    ProviderConfig(
                        kind=embed_kind,
                        model=embed_model or "",
                        embed_model=embed_model,
                        region=region,
                    )
                )
        return cls(
            providers=providers,
            serve_provider=serve_provider,
            embed_provider=embed_provider,
            embed_dim=embed_dim,
            gepa_budget=gepa_budget,
            train_split=train_split,
            judge_model=(
                resolve_provider_model(serve_provider, judge_model).model_id
                if judge_model is not None
                else None
            ),
            trace_adapter=trace_adapter,
        )


class ArtifactPaths:
    """Resolves the files under `.wmh/`."""

    def __init__(self, root: str | Path = ARTIFACT_DIR) -> None:
        self.root = Path(root)

    @property
    def config(self) -> Path:
        return self.root / "config.toml"

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def runs(self) -> Path:
        """Directory of persisted run records (build + serve), one JSON per run."""
        return self.root / "runs"

    @property
    def base_prompt(self) -> Path:
        return self.root / "prompts" / "base.txt"

    @property
    def optimized_prompt(self) -> Path:
        return self.root / "prompts" / "optimized.txt"

    @property
    def frontier(self) -> Path:
        return self.root / "prompts" / "frontier.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def knowledge(self) -> Path:
        """Cross-session knowledge base directory (optional; absent on pre-knowledge artifacts)."""
        return self.root / "knowledge"

    @property
    def auto_fidelity(self) -> Path:
        """The auto-config search report (present on high/max-tier builds)."""
        return self.root / "auto_fidelity.json"

    @property
    def manifest(self) -> Path:
        """SHA-256 checksums of every artifact file, written by `wmh build`."""
        return self.root / "manifest.json"


def _strip_none(value: JsonValue) -> JsonValue:
    """Drop `None`-valued keys recursively so TOML (which has no null) can represent the config.

    On load, pydantic refills the missing optional fields with their `None` defaults, so dropping
    them here round-trips losslessly.
    """
    if isinstance(value, dict):
        return _strip_none_object(value)
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def _strip_none_object(obj: dict[str, JsonValue]) -> JsonObject:
    return {k: _strip_none(v) for k, v in obj.items() if v is not None}


def load_config(root: str | Path = ARTIFACT_DIR) -> HarnessConfig:
    """Read `.wmh/config.toml`. Raises a friendly error if the project hasn't been built yet."""
    paths = ArtifactPaths(root)
    if not paths.root.exists():
        raise FileNotFoundError(
            f"no {ARTIFACT_DIR}/ directory at {paths.root}; run `wmh build` first to create it"
        )
    if not paths.config.exists():
        raise FileNotFoundError(
            f"{paths.config} is missing; run `wmh build` to (re)generate the project config"
        )
    try:
        with paths.config.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"{paths.config} is not valid TOML ({exc}); re-run `wmh build` to regenerate it"
        ) from exc
    try:
        return HarnessConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"{paths.config} does not match the current config schema ({exc}); "
            "re-run `wmh build` to regenerate it"
        ) from exc


def save_config(config: HarnessConfig, root: str | Path = ARTIFACT_DIR) -> None:
    """Write `config` to `.wmh/config.toml`, creating `.wmh/` if missing.

    Writes to a temp file in the same directory and renames into place so an interrupted or
    failed write never leaves a truncated `config.toml` behind.
    """
    paths = ArtifactPaths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    data = _strip_none_object(config.model_dump(mode="json"))
    tmp = paths.config.with_name(f"{paths.config.name}.tmp")
    with tmp.open("wb") as fh:
        tomli_w.dump(data, fh)
    tmp.replace(paths.config)
