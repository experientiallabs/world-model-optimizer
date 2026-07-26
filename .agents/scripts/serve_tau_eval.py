"""Serve the prebuilt tau-bench WM in EVAL configuration (BENCH-B shared protocol).

Env backend = PINNED GPT-5.5 (OpenAI; different family from the Haiku training WM —
strongest circularity blunting available). Reward judge = Opus 4.8 on Bedrock us-east-1
(D12 pins it for fidelity rows; third family vs both WM backends avoids same-family bias).

Requires OPENAI_API_KEY (gitignored .env at the repo root) and AWS creds for Bedrock.

Run from the wmh repo root:  uv run python .agents/scripts/serve_tau_eval.py [port]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from wmh.engine.world_model import WorldModel
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "packages" / "environment-capture" / "tau-bench" / "models" / "tau-bench"
EVAL_ENV_MODEL = "gpt-5.5"
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"  # the artifact's own serve model id (config.toml)


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing: put it in the gitignored .env at the repo root")
    from wmh.providers.fallback import FallbackProvider

    serve_provider = get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=EVAL_ENV_MODEL))
    # CROSS-REGION + CROSS-GEO same-model failover: the whole US Opus inference
    # profile capacity-crashed for hours during a live eval (every us-* region
    # ServiceUnavailable) while the EU profile answered instantly. Same dated
    # model version in every link — D12/D30 pin the MODEL, not the geography.
    # EU-FIRST while the US profile is in a brownout: FallbackProvider has no
    # cross-call memory, so a hung head link is re-tried on EVERY call — with
    # us-east-1 first, each judge call burned the full US timeout budget before
    # reaching the healthy EU links (observed: 50-min /score stalls while
    # eu-west-1 answered in 1.6s). Restore US-first when the profile recovers,
    # or better: give FallbackProvider last-good stickiness (wmh follow-up).
    judge_provider = FallbackProvider(
        [
            get_provider(
                ProviderConfig(kind=ProviderKind.BEDROCK, model=JUDGE_MODEL.replace("us.", "eu.", 1), region=r)
            )
            for r in ("eu-west-1", "eu-central-1")
        ]
        + [
            get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=JUDGE_MODEL, region=r))
            for r in ("us-east-1", "us-west-2")
        ]
    )
    wm = WorldModel.load(str(MODEL_DIR), serve_provider, reward_provider=judge_provider)
    app = create_app(world_models={"tau-bench": wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
