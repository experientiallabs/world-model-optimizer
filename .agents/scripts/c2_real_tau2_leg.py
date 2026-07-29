"""C2 real tau2 leg: scoped compression validated on the REAL benchmark (round 3, GO'd).

Pre-registered pass bar (findings/c2.md 2026-07-28 evening): the wm_simulated
corpus-grain result (+4.4 +- 4.3 quality at -29.4% effective cost) must REPRODUCE IN
SIGN on real tau2 episodes, scored by tau2's own deterministic evaluator (judge-free).

Arms, both scoped configs because aggressiveness is identity-grained (D-DIAL ruling):
off (uncompressed), scoped@0.48 (the C1-amended handoff threshold), scoped@0.5 (the
wm-evidence match). 25 tasks (13 retail + 12 airline, sorted prefix of the 720-episode
study's 40 known-good ids) x {gpt-5.4-mini via azure, sonnet-5 via bedrock} x 2
episodes = 300 episodes, ~$57 at the measured $0.19/ep, hard budget guard at $75
(cap $80 all-in). Protocol pins per the training-chat entry: max_turns 100, timeout
1800, user simulator azure/gpt-5.4-mini with {} args; any deviation is a new cohort
label.

Wiring: every arm's agent traffic goes through ONE loopback `EpisodeProxy`
(wmo.distill.tau2_proxy) with a per-episode alias, uniformly (the off arm too, so the
provider stack is identical across arms). The scoped arms wrap the pool provider in
`ToolRoleScoped`, which compresses TOOL-ROLE message content only, through the live
llmlingua2 endpoint; system, user (dialogue in both directions), and assistant
messages pass through byte-exact. Deterministic per message content, so the growing
transcript stays append-stable; live keep is ENDOGENOUS (C1) and reported per episode
as (achieved keep, accuracy) points, never a nominal claim.

Run from the c2-r2 worktree with the compressor + azure + aws env exported:

    uv run python .agents/scripts/c2_real_tau2_leg.py [--smoke]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

from llm_waterfall.types import ChatRequest, ChatResponse

from wmo.optimize.compression import CompressionConfig, estimate_tokens, get_compressor
from wmo.providers.base import ProviderKind, TokenUsage, ToolCallingProvider
from wmo.providers.pool import PoolEntry, pool_provider
from wmo.tracking.pricing import cost_usd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("c2_real_tau2")

TAU2_BIN = Path.home() / "Desktop/Projects/tau2-bench/.venv/bin/tau2"
TAU2_DATA = Path.home() / "Desktop/Projects/tau2-bench/data"  # tasks under data/tau2/, sims under data/simulations/
OUT = Path.home() / "Desktop/Projects/wmh-compression-data/fits/c2-r3/real-tau2"
ROWS = OUT / "rows.jsonl"

MAX_TURNS = 100
TIMEOUT_S = 1800
USER_LLM = "azure/gpt-5.4-mini"
KILL_MARGIN_S = 120
BUDGET_GUARD_USD = 75.0
EPISODES = 2
CONCURRENCY = 4

ENTRIES = [
    PoolEntry(
        name="gpt-5.4-mini",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.4-mini",
        deployment="gpt-5.4-mini",
        endpoint="https://google-sheets.openai.azure.com",
        api_version="2024-10-21",
        api_key_env="AZURE_GOOGLE_SHEETS_API_KEY",
        input_per_mtok=0.75,
        output_per_mtok=4.5,
    ),
    # Same weights as the wm arms' sonnet-5, served via Bedrock (the anthropic-direct
    # provider has no complete_chat); the provider difference is ledgered.
    PoolEntry(
        name="sonnet-5",
        kind=ProviderKind.BEDROCK,
        model="us.anthropic.claude-sonnet-5",
        region="us-west-2",
        input_per_mtok=3.0,
        output_per_mtok=15.0,
    ),
]

ARMS = {
    "off": None,
    "scoped048": 0.48,
    "scoped05": 0.5,
}


def task_ids() -> list[str]:
    matrix = json.loads(
        (Path.home() / "Desktop/Projects/wmh-routing-data/matrices/tau-bench-real_matrix.json")
        .read_text()
    )
    ids = sorted({o["scenario_id"] for o in matrix["outcomes"]})
    retail = [i for i in ids if i.startswith("retail:")][:13]
    airline = [i for i in ids if i.startswith("airline:")][:12]
    return retail + airline


class ToolRoleScoped:
    """Tool-role-only compression at tau2's real message boundary.

    Compresses the content of role == "tool" messages through the live endpoint
    compressor; every other message passes through byte-exact. Accounts achieved keep
    (proxy-token estimates), compressor cost/latency, and the provider's own usage
    cost per episode.
    """

    def __init__(self, inner: ToolCallingProvider, model_id: str, aggressiveness: float | None):
        self._inner = inner
        self._model_id = model_id
        self._compressor = get_compressor("llmlingua2-endpoint") if aggressiveness else None
        self._config = (
            CompressionConfig(
                compressor_id="llmlingua2-endpoint",
                compressor_version=self._compressor.version,
                aggressiveness=aggressiveness,
            )
            if aggressiveness
            else None
        )
        self.tokens_raw = 0
        self.tokens_kept = 0
        self.compressor_usd = 0.0
        self.compressor_s = 0.0
        self.provider_usd = 0.0
        self.calls = 0

    @property
    def config(self):  # noqa: ANN201 - Provider protocol surface
        return self._inner.config

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self._model_id.startswith("us.anthropic."):
            # Bedrock Converse rejects `temperature` for sonnet-5 ("deprecated for this
            # model"); stripped uniformly for this model across ALL arms, so the arms stay
            # comparable within the model.
            request = request.model_copy(update={"temperature": None})
        if self._compressor is not None:
            tool_indices = [
                i
                for i, m in enumerate(request.messages)
                if m.role == "tool" and isinstance(m.content, str) and m.content.strip()
            ]
            if tool_indices:
                started = time.monotonic()
                result = self._compressor.compress(
                    [request.messages[i].content for i in tool_indices], self._config
                )
                self.compressor_s += time.monotonic() - started
                self.compressor_usd += result.cost_usd
                messages = list(request.messages)
                for index, compressed in zip(tool_indices, result.segments, strict=True):
                    self.tokens_raw += estimate_tokens(messages[index].content)
                    self.tokens_kept += estimate_tokens(compressed)
                    messages[index] = messages[index].model_copy(update={"content": compressed})
                request = request.model_copy(update={"messages": messages})
        response = self._inner.complete_chat(request)
        if response.usage is not None:
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None) or {}
            cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
            self.provider_usd += cost_usd(
                self._model_id,
                TokenUsage(
                    input_tokens=usage.prompt_tokens or 0,
                    output_tokens=usage.completion_tokens or 0,
                    cached_input_tokens=cached or 0,
                ),
            )
        return response


async def run_episode(
    proxy, semaphore, arm: str, entry: PoolEntry, task_id: str, episode: int, state: dict
) -> None:  # noqa: ANN001
    alias = f"{arm}-{entry.name}-{task_id.replace(':', '_')}-e{episode}"
    episode_dir = OUT / arm / entry.name / f"{task_id.replace(':', '_')}-e{episode}"
    results_copy = episode_dir / "results.json"
    if results_copy.exists():
        payload = json.loads(results_copy.read_text())
        sims = payload.get("simulations") or []
        prior = (sims[0].get("reward_info") or {}).get("reward") if sims else None
        if prior is not None:
            log.info("resume: %s already has verifier evidence", alias)
            return
        # A results file without a reward is an errored simulation, not evidence: re-run.
        shutil.rmtree(episode_dir)
    async with semaphore:
        if state["spend"] >= BUDGET_GUARD_USD:
            state["skipped"] += 1
            return
        episode_dir.mkdir(parents=True, exist_ok=True)
        provider = ToolRoleScoped(
            pool_provider(entry), entry.model, ARMS[arm]
        )
        proxy.register(alias, provider)
        domain, tau2_task = task_id.split(":", 1)
        save_name = f"c2r3-{alias}"
        sim_dir = TAU2_DATA / "simulations" / save_name
        shutil.rmtree(sim_dir, ignore_errors=True)
        agent_args = {
            "api_base": proxy.base_url,
            "api_key": "wmo-local-proxy",
            "temperature": 0.7,
            "max_tokens": 1024,
            "num_retries": 0,
            "timeout": TIMEOUT_S,
        }
        command = [
            str(TAU2_BIN), "run", "--domain", domain,
            "--task-ids", tau2_task, "--num-trials", "1",
            "--max-steps", str(MAX_TURNS), "--max-errors", "10",
            "--timeout", str(TIMEOUT_S),
            "--agent-llm", f"openai/{alias}",
            "--agent-llm-args", json.dumps(agent_args),
            "--user-llm", USER_LLM, "--user-llm-args", json.dumps({}),
            "--save-to", save_name, "--max-retries", "0", "--auto-resume",
        ]
        env = os.environ | {
            "TAU2_DATA_DIR": str(TAU2_DATA),
            "AZURE_API_KEY": os.environ.get("AZURE_GOOGLE_SHEETS_API_KEY", ""),
            "AZURE_API_BASE": "https://google-sheets.openai.azure.com",
            "AZURE_API_VERSION": "2024-10-21",
        }
        started = time.monotonic()
        log_path = episode_dir / "runner.log"
        with log_path.open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=log_file, stderr=asyncio.subprocess.STDOUT, env=env
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=TIMEOUT_S + KILL_MARGIN_S)
            except (TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise
        wall = time.monotonic() - started
        reward = None
        results_path = sim_dir / "results.json"
        if results_path.exists():
            shutil.copyfile(results_path, results_copy)
            payload = json.loads(results_copy.read_text())
            sims = payload.get("simulations") or []
            if sims and isinstance(sims[0], dict):
                info = sims[0].get("reward_info") or {}
                reward = info.get("reward")
        row = {
            "arm": arm, "model": entry.name, "task": task_id, "episode": episode,
            "reward": reward, "infra_failed": reward is None,
            "provider_usd": round(provider.provider_usd, 5),
            "compressor_usd": round(provider.compressor_usd, 5),
            "compressor_s": round(provider.compressor_s, 2),
            "achieved_keep": round(provider.tokens_kept / provider.tokens_raw, 4)
            if provider.tokens_raw else None,
            "agent_calls": provider.calls, "wall_s": round(wall, 1),
            "cohort": "real-tau2 maxturns100 timeout1800 usersim-gpt-5.4-mini "
                      "sonnet-5-via-bedrock",
        }
        state["spend"] += provider.provider_usd + provider.compressor_usd
        with ROWS.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        log.info(
            "%s: reward=%s $%.3f keep=%s wall=%.0fs (cumulative $%.2f)",
            alias, reward, provider.provider_usd, row["achieved_keep"], wall, state["spend"],
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="2 episodes only")
    args = parser.parse_args()

    from wmo.distill.tau2_proxy import EpisodeProxy

    OUT.mkdir(parents=True, exist_ok=True)
    spend0 = 0.0
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            row = json.loads(line)
            spend0 += row.get("provider_usd", 0.0) + row.get("compressor_usd", 0.0)
    state = {"spend": spend0, "skipped": 0}
    proxy = EpisodeProxy()
    proxy.start()
    tasks = task_ids()
    log.info("tasks (%d): %s", len(tasks), tasks)
    jobs = []
    if args.smoke:
        jobs = [("off", ENTRIES[0], tasks[0], 0), ("scoped048", ENTRIES[0], tasks[0], 0)]
    else:
        for arm in ARMS:
            for entry in ENTRIES:
                for task_id in tasks:
                    for episode in range(EPISODES):
                        jobs.append((arm, entry, task_id, episode))
    semaphore = asyncio.Semaphore(CONCURRENCY)
    try:
        await asyncio.gather(
            *(run_episode(proxy, semaphore, *job, state) for job in jobs)
        )
    finally:
        proxy.stop()
    log.info(
        "leg done: cumulative $%.2f, %d skipped by budget guard", state["spend"], state["skipped"]
    )


if __name__ == "__main__":
    asyncio.run(main())
