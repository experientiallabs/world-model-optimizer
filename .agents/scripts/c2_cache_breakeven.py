"""C2 Q4: the compression-vs-cache break-even, replayed on real transcripts.

Replays C1's audit transcripts (multi-turn conversations reconstructed from the
routing matrices' stored replies) through a per-conversation provider prefix-cache
model and computes effective INPUT cost per conversation under three compression
policies x three provider archetypes, using C1's measured per-(method, corpus)
token ratio r and append churn c:

Policies
- no-compress: prompt at turn t = full history; prefix N_{t-1} reads at the cache
  tier, appended delta writes.
- full-compress: whole prompt recompressed each turn. An append-stable method
  (c = 0) keeps the compressed prefix cacheable, so both tiers scale by r. A churny
  method invalidates fraction c of the compressed prefix per turn: that mass rebills
  at the write tier instead of the read tier.
- turn-local-commit: compress each turn once when appended, never retroactively
  (C1's per-turn-truncate-at-append shape, applicable to any method). Append-only by
  construction, so c = 0 regardless of the method's own churn; ratio r applies to
  appended content only, and the task turn (turn 1) is also compressed at r.

Provider archetypes (tiers from C3's pricing table on compress/c3, which carries the
vendors' published multipliers): anthropic (read 0.1x, write 1.25x), openai (read
0.1x, new tokens 1.0x, no write premium), no-cache (every prompt token 1.0x, the
open-model pool rows without a provider prompt cache).

Assumptions, stated: input cost only (output tokens are identical across policies);
turns arrive within the cache TTL; the conversation is served by one model (affinity,
#248); token counts via the GPT-2 tokenizer (same counter as C1's ratios). Cross-
conversation prefix sharing (system prompts reused across conversations) is NOT in
the transcripts (replies only); that leg stays analytic: the crossover for a
cache-hostile compressor vs caching alone is rho* = (c_w - r) / (c_w - c_r).

Run inside C1's scratch venv:

    ~/Desktop/Projects/wmh-compression-data/cache/venv-c1/bin/python \
        .agents/scripts/c2_cache_breakeven.py
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("c2_cache_breakeven")

DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
CORPUS_PATH = DATA_ROOT / "cache/audit-transcripts.jsonl"
ROUND0_PATH = DATA_ROOT / "cache/round0-results.json"
OUT_PATH = DATA_ROOT / "cache/cachesim-results.json"

ARCHETYPES = {
    "anthropic": {"read": 0.1, "write": 1.25},
    "openai": {"read": 0.1, "write": 1.0},
    "no-cache": {"read": 1.0, "write": 1.0},
}


def token_counter():  # noqa: ANN202 - .agents pragma
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return lambda text: len(tok(text)["input_ids"])


def conversation_cost(
    turn_tokens: list[int], *, r: float, churn: float, policy: str, tiers: dict[str, float]
) -> float:
    """Input-cost units (token x rate) for one conversation under one policy.

    `turn_tokens[i]` is the token count of turn i (task, then each reply-turn).
    """
    read, write = tiers["read"], tiers["write"]
    cost = 0.0
    prefix = 0.0  # tokens currently sitting in the provider cache
    for i, tokens in enumerate(turn_tokens):
        if policy == "no-compress":
            new = float(tokens)
            stable_prefix = prefix
        elif policy == "full-compress":
            new = r * tokens
            stable_prefix = prefix * (1.0 - churn)
        elif policy == "turn-local-commit":
            new = r * tokens
            stable_prefix = prefix
        else:
            raise ValueError(policy)
        rebilled = prefix - stable_prefix  # churned cache mass rebills at the write tier
        cost += read * stable_prefix + write * (new + rebilled)
        prefix = stable_prefix + rebilled + new
        del i
    return cost


def main() -> None:
    count = token_counter()
    transcripts: dict[str, list[list[int]]] = {}
    with CORPUS_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            transcripts.setdefault(row["corpus"], []).append(
                [count(turn) for turn in row["turns"]]
            )

    round0 = json.loads(ROUND0_PATH.read_text())
    results: list[dict] = []
    prefix_share: dict[str, float] = {}
    for corpus, convs in transcripts.items():
        total = sum(sum(t[: i + 1]) for t in convs for i in range(len(t)))
        reused = sum(sum(t[:i]) for t in convs for i in range(1, len(t)))
        prefix_share[corpus] = reused / total
        log.info(
            "%s: %d conversations, within-conversation prefix share %.3f",
            corpus, len(convs), prefix_share[corpus],
        )

    for row in round0:
        corpus, method = row["corpus"], row["method"]
        if corpus not in transcripts:
            continue
        r, churn = row["token_ratio"], row["churn_mean"]
        for arch, tiers in ARCHETYPES.items():
            base = [
                conversation_cost(t, r=1.0, churn=0.0, policy="no-compress", tiers=tiers)
                for t in transcripts[corpus]
            ]
            for policy in ("full-compress", "turn-local-commit"):
                cost = [
                    conversation_cost(t, r=r, churn=churn, policy=policy, tiers=tiers)
                    for t in transcripts[corpus]
                ]
                multipliers = [c / b for c, b in zip(cost, base)]
                results.append(
                    {
                        "corpus": corpus,
                        "method": method,
                        "token_ratio": r,
                        "churn_mean": churn,
                        "archetype": arch,
                        "policy": policy,
                        "cost_multiplier_mean": statistics.mean(multipliers),
                        "cost_multiplier_max": max(multipliers),
                        "n_conversations": len(multipliers),
                    }
                )

    OUT_PATH.write_text(
        json.dumps({"prefix_share": prefix_share, "rows": results}, indent=1)
    )
    log.info("wrote %d rows -> %s", len(results), OUT_PATH)

    # The headline table: mean input-cost multiplier vs no-compress (values < 1 save money).
    methods = sorted({r["method"] for r in results})
    corpora = sorted(transcripts)
    for arch in ARCHETYPES:
        print(f"\n=== {arch}: mean input-cost multiplier vs no-compress ===")
        print(f"{'method':<32} {'policy':<18} " + " ".join(f"{c:>15}" for c in corpora))
        for method in methods:
            for policy in ("full-compress", "turn-local-commit"):
                cells = []
                for corpus in corpora:
                    match = [
                        r for r in results
                        if (r["method"], r["policy"], r["archetype"], r["corpus"])
                        == (method, policy, arch, corpus)
                    ]
                    cells.append(f"{match[0]['cost_multiplier_mean']:>15.3f}" if match else " " * 15)
                print(f"{method:<32} {policy:<18} " + " ".join(cells))

    # The analytic churn bound on cached providers: full compression beats no-compress
    # only when r * (read + (write - read) * c) < read, i.e. c < read/(write-read) * (1/r - 1).
    print("\nchurn bound (anthropic tiers): full-compress loses to NO compression when "
          "churn > 0.087 * (1/r - 1) / 0.5 ... printed per method below")
    for row in round0:
        if row["corpus"] not in transcripts:
            continue
        r, c = row["token_ratio"], row["churn_mean"]
        if r >= 1.0:
            continue
        bound = 0.1 / 1.15 * (1.0 / r - 1.0)
        if c > 0:
            print(f"  {row['method']:<32} {row['corpus']:<16} r={r:.2f} churn={c:.2f} "
                  f"bound={bound:.3f} -> {'NET LOSS' if c > bound else 'ok'}")


if __name__ == "__main__":
    main()
