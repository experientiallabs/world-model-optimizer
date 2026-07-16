"""Serve the prebuilt tau-bench WM in the BENCH-B training or eval configuration.

One script for both serving configs so fixes (failover chain, ports, warm-up behavior)
cannot diverge between the training env and the eval env:

- ``train``: env + reward judge on Bedrock Haiku 4.5 (dated profile id — cost control;
  the artifact's built-in Opus provider is overridden at load). ``WMH_ENV_TEMPERATURE``
  optionally pins the env's sampling temperature (judge unaffected). KNOWN LIMIT: the
  pin only reaches backends whose request path forwards sampling params (OpenAI, Nova,
  Converse third-party). The default Haiku env rides ``BedrockProvider.complete()``'s
  Anthropic invoke_model path, which omits sampling params by endpoint contract
  (wmh/providers/bedrock.py) — so for Bedrock-Anthropic env models the pin is INERT and
  the env samples at the API default. This matches the D62/D65 measurements: identical
  4-step action replays scored {0.95, 0.15, 0.3, 0.15, 0.65, 0.95} (stdev 0.34), and
  even nominal temp-0 only reached ~0.24 — temperature pins token sampling, not the
  imagined world. The operative env-luck fix is ``seed_state`` (scenario v2 pins), which
  the probe measured near-deterministic. Do not attribute variance reduction to this
  env var on Bedrock-Anthropic envs; see the PR #73 board correction (2026-07-15).
- ``eval`` (D71/D76 sonnet-era; supersedes the D30 GPT-5.5 env, terminated D68): env on
  Bedrock sonnet-5 (+ the artifact's RAG), reward judge on Opus 4.8, rubrics passed by
  the caller where the benchmark pins them. Rows from this env are labeled sonnet-era
  and never compared to gpt5-era absolutes.

Both providers are wrapped in same-model FallbackProvider chains (D18): throttles fail
over instantly; a hung read fails over at the bedrock client's 600s bound instead of
killing the episode.

Promotion note (AGENTS rule 7): serving a built WM with an overridden provider is
dataset-agnostic and now has three consumers — it belongs in `wmh serve` as a provider
override flag; tracked in DECISIONS.

Run from the wmh repo root:
    uv run python .agents/scripts/serve_tau_wm.py train [port]   # default port 8000
    uv run python .agents/scripts/serve_tau_wm.py eval [port]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from wmh.config.dotenv import load_env_file
from wmh.engine.world_model import WorldModel
from wmh.providers import get_provider
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
# WMH_MODEL_DIR/WMH_WM_NAME swap the served artifact for the D67 cross-benchmark
# smokes (terminal/swe/gui reuse this script unchanged apart from these two).
_DEFAULT_MODEL_DIR = (
    REPO_ROOT / "packages" / "environment-capture" / "tau-bench" / "models" / "tau-bench"
)
MODEL_DIR = Path(os.environ.get("WMH_MODEL_DIR", _DEFAULT_MODEL_DIR))
WM_NAME = os.environ.get("WMH_WM_NAME", "tau-bench")
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (required)
# D71/D76 sonnet-era eval env (GPT-5.5 terminated, D68); label rows sonnet-era.
EVAL_ENV_MODEL = "us.anthropic.claude-sonnet-5"
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"  # the artifact's own serve model id


class FallbackProvider:
    """Sequential same-call failover that FORWARDS temperature (unlike WaterfallProvider,
    which drops sampling params by design). The pass-through keeps the pin alive as far
    as the provider seam; whether it reaches the wire is then per-backend —
    Bedrock-Anthropic models drop sampling params at the request layer (see the module
    docstring), OpenAI/Nova/Converse paths honor them.
    """

    def __init__(self, chain: list[Provider]) -> None:
        if not chain:
            raise ValueError("FallbackProvider needs at least one provider")
        self._chain = chain
        self.config = chain[0].config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        last: Exception | None = None
        for provider in self._chain:
            try:
                return provider.complete(
                    system, messages, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - any backend failure moves down the chain
                last = exc
        assert last is not None
        raise last

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._chain[0].embed(texts)

    def verify(self) -> VerifyResult:
        return self._chain[0].verify()


def _fallback_chain(config: ProviderConfig, *, cross_provider: bool = False) -> Provider:
    """Same-model failover: a second same-region instance, then us-west-2 for Bedrock.

    The cross-region link rides through regional brownouts (observed live:
    us-east-1 ServiceUnavailableException storms on the Opus judge stall whole evals —
    both same-region links 503 together).
    """
    chain = [get_provider(config), get_provider(config)]
    if config.kind is ProviderKind.BEDROCK and config.region != "us-west-2":
        chain.append(get_provider(config.model_copy(update={"region": "us-west-2"})))
    # Cross-PROVIDER last resort for Opus 4.8: the Anthropic direct API (own quota pool,
    # rides through Bedrock-wide storms). Key distributed to the box .envs (Silen ack'd
    # direct-key use after the D68 OpenAI termination).
    # Cross-provider is opt-in and reserved for the JUDGE chain: letting the ENV chain
    # hop providers would silently serve part of a fidelity-curve point from a different
    # backend than the one the point claims to measure.
    if cross_provider and "opus-4-8" in config.model and os.environ.get("ANTHROPIC_API_KEY"):
        chain.append(
            get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
        )
    return FallbackProvider(chain)


class PinnedTemperatureProvider:
    """Forces one sampling temperature on every ``complete`` call it forwards.

    Wraps ONLY the WM's serve provider: callers' temperature arguments (the WM never
    passes one, so it otherwise gets the 0.7 provider default) are replaced, while the
    reward judge keeps its own unwrapped provider and its explicit temperature=0.0.
    INERT for Bedrock-Anthropic env models — that request path omits sampling params
    (see the module docstring); the value is only honored by OpenAI/Nova/Converse
    backends.
    """

    def __init__(self, inner: Provider, temperature: float) -> None:
        self._inner = inner
        self._temperature = temperature
        self.config = inner.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        del temperature
        return self._inner.complete(
            system, messages, temperature=self._temperature, max_tokens=max_tokens
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)

    def verify(self) -> VerifyResult:
        return self._inner.verify()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        raise SystemExit(f"usage: {sys.argv[0]} train|eval [port]")
    mode = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    load_env_file(
        REPO_ROOT / ".env"
    )  # OPENAI/ANTHROPIC keys for backend swaps + the direct-API opus link

    top_k: int | None = None
    if mode == "train":
        # Fidelity->transfer curve (D67): WMH_ENV_MODEL/WMH_ENV_PROVIDER swap the env
        # backend per curve point; the reward judge stays PINNED on the haiku chain so
        # points differ by environment fidelity only. WMH_TOP_K=0 = the no-RAG point.
        haiku = ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU_MODEL, region="us-east-1")
        reward_provider = _fallback_chain(haiku, cross_provider=True)
        env_model = os.environ.get("WMH_ENV_MODEL")
        env_kind = ProviderKind(os.environ.get("WMH_ENV_PROVIDER", "bedrock"))
        if env_model is None:
            serve_provider = reward_provider
        else:
            if env_kind is ProviderKind.OPENAI:
                load_env_file(REPO_ROOT / ".env")
            serve_provider = _fallback_chain(
                ProviderConfig(
                    kind=env_kind,
                    model=env_model,
                    region="us-east-1" if env_kind is ProviderKind.BEDROCK else None,
                )
            )
        raw_top_k = os.environ.get("WMH_TOP_K")
        if raw_top_k is not None:
            top_k = int(raw_top_k)
        env_temp = os.environ.get("WMH_ENV_TEMPERATURE")
        if env_temp is not None:
            serve_provider = PinnedTemperatureProvider(serve_provider, float(env_temp))
    else:
        load_env_file(REPO_ROOT / ".env")
        serve_provider = _fallback_chain(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=EVAL_ENV_MODEL, region="us-east-1")
        )
        reward_provider = _fallback_chain(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=JUDGE_MODEL, region="us-east-1"),
            cross_provider=True,
        )

    wm = WorldModel.load(
        str(MODEL_DIR), serve_provider, reward_provider=reward_provider, top_k=top_k
    )
    app = create_app(world_models={WM_NAME: wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
