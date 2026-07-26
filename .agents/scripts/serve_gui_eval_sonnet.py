"""Serve the gui-tasks WM in sonnet-era EVAL configuration (D71/D76).

Env backend = sonnet-5 (cross-region waterfall, temp-0 steps per the D66/D78 substrate
floor); reward judge = Opus 4.8 on a cross-geo waterfall (EU-first under US brownouts).
Superseded as the official kimi env by the D86 haiku re-pin; kept for the sonnet-era rows.

Run from the wmh repo root:  uv run python .agents/scripts/serve_gui_eval_sonnet.py [port]
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

from wmh.engine.world_model import WorldModel
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.fallback import FallbackProvider
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "examples" / "gui-tasks" / "models" / "gui-tasks"
ENV_MODEL = "us.anthropic.claude-sonnet-5"  # D71/D76 sonnet-era eval env
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"
REGIONS = ("us-east-1", "us-west-2", "us-east-2")


class _Temp0Provider:
    """Delegate that pins env-step completions to temperature 0 (D66/D78 substrate floor)."""

    def __init__(self, inner) -> None:  # noqa: ANN001 - provider protocol
        self._inner = inner
        self.config = inner.config

    def complete(self, system, messages, *, temperature=0.7, max_tokens=2048):  # noqa: ANN001
        return self._inner.complete(system, messages, temperature=0.0, max_tokens=max_tokens)

    def embed(self, texts):  # noqa: ANN001
        return self._inner.embed(texts)

    def verify(self):  # noqa: ANN001
        return self._inner.verify()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    chain = [
        get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=ENV_MODEL, region=r))
        for r in REGIONS
    ]
    provider = FallbackProvider(chain)
    judge_chain = [
        get_provider(
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model=(JUDGE_MODEL.replace("us.", "eu.", 1) if r.startswith("eu") else JUDGE_MODEL),
                region=r,
            )
        )
        for r in ("eu-west-1", "eu-central-1", "us-east-1", "us-west-2")
    ]
    wm = WorldModel.load(
        str(MODEL_DIR), _Temp0Provider(provider), reward_provider=FallbackProvider(judge_chain)
    )
    app = create_app(world_models={"gui-tasks": wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
