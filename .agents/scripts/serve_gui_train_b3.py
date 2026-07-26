"""Serve the gui-tasks WM for B3 training runs: Haiku 4.5 with a cross-region waterfall.

Same provider swap as .agents/scripts/serve_tau_haiku.py, but wrapped in a
FallbackProvider chain of the SAME dated haiku profile id across regions
(us-east-1 -> us-west-2 -> us-east-2). Bulk GRPO rollouts (n=8 concurrent episodes,
each stepping the WM) hit per-region Bedrock throttling — observed as multi-minute
step stalls during B2's PPO smoke. Failing over across regions keeps the ENV MODEL
IDENTICAL (a cross-model fallback would silently change the environment mid-training).

Run from the wmh repo root:  uv run python .agents/scripts/serve_gui_train_b3.py [port]
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
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (required)
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
        get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU_MODEL, region=r))
        for r in REGIONS
    ]
    provider = FallbackProvider(chain)
    wm = WorldModel.load(str(MODEL_DIR), _Temp0Provider(provider), reward_provider=provider)
    app = create_app(world_models={"gui-tasks": wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
