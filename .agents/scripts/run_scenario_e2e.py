"""E2E verification of scenario-set construction on tau-bench traces with small Bedrock models.

Runs the full pipeline and the verification battery, writing raw JSON results + a summary to
.agents/docs/research/. Models: Amazon Nova Lite (facets/naming/synthesis/judge/world model),
Nova Micro + Nova Lite (agent configs), Titan v2 (facet embeddings).

Usage (from the repo root):
    uv run python .agents/scripts/run_scenario_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from wmh.core.types import Trace  # noqa: E402
from wmh.engine.world_model import WorldModel  # noqa: E402
from wmh.env.llm_agent import LLMAgent  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.providers import get_provider  # noqa: E402
from wmh.providers.base import (  # noqa: E402
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
)
from wmh.providers.retry import RetryingProvider  # noqa: E402
from wmh.research.scenario_fidelity import (  # noqa: E402
    fidelity_report,
    random_subsets,
    score_matrix,
)
from wmh.research.scenario_recovery import ground_truth_labels, recovery_report  # noqa: E402
from wmh.scenarios import (  # noqa: E402
    ChecklistJudge,
    FacetExtractor,
    ScenarioBuildConfig,
    build_scenario_set,
    cluster_facets,
    hybrid_select,
    trace_digest,
    verify_scenarios,
)

REGION = "us-west-2"
NOVA_LITE = "us.amazon.nova-lite-v1:0"
NOVA_MICRO = "us.amazon.nova-micro-v1:0"
PER_DOMAIN = 20  # corpus subsample per tau2 domain
POOL_BUDGET = 30  # the "full distribution" stand-in for Test 2
OURS_K = 8  # our method's scenario budget
PASSES = 3  # every metric is a mean of 3 passes (house rule)
MAX_STEPS = 5
WORKERS = 6
OUT_DIR = REPO / ".agents" / "docs" / "research"
TRACES = REPO / "packages" / "environment-capture" / "tau-bench" / "traces.otel.jsonl"
WM_DIR = REPO / "packages" / "environment-capture" / "tau-bench" / "models" / "tau-bench"


def _retrying(inner: Provider) -> Provider:
    """Retry capacity errors with llm-waterfall's classifier (string-matching "Throttl"/"Timeout"
    against messages — the previous hand-rolled version here — misses httpx transients and
    misclassifies e.g. ValidationException("timeout too large") as retriable)."""
    return RetryingProvider(inner, delays=(2.0, 4.0, 8.0, 16.0))


def bedrock(model: str, region: str = REGION) -> Provider:
    return _retrying(
        get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region))
    )


def titan_embedder() -> Provider:
    return get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=NOVA_LITE, region=REGION, embed_dim=512)
    )


def subsample(traces: list[Trace]) -> list[Trace]:
    by_domain: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        domain = trace.metadata.get("domain")
        by_domain[str(domain)].append(trace)
    picked: list[Trace] = []
    for domain in sorted(by_domain):
        picked.extend(by_domain[domain][:PER_DOMAIN])
    return picked


def embed_batch(embedder: Provider, texts: list[str]) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        vectors = list(pool.map(lambda t: embedder.embed([t])[0], texts))
    return np.asarray(vectors)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "config": {
            "region": REGION,
            "per_domain": PER_DOMAIN,
            "pool_budget": POOL_BUDGET,
            "ours_k": OURS_K,
            "passes": PASSES,
            "max_steps": MAX_STEPS,
            "models": {
                "pipeline": NOVA_LITE,
                "agents": [NOVA_MICRO, NOVA_LITE],
                "embed": "amazon.titan-embed-text-v2:0 (512d)",
            },
        }
    }
    t0 = time.time()

    lite = bedrock(NOVA_LITE)
    lite.complete("", [Message(role="user", content="ping")], max_tokens=1)  # warm client

    print("== ingest + subsample ==", flush=True)
    corpus = subsample(get_adapter("otel-genai").from_file(str(TRACES)))
    print(f"corpus: {len(corpus)} traces", flush=True)

    print("== facet extraction (Nova Lite) ==", flush=True)
    extractor = FacetExtractor(lite)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        facets = list(pool.map(extractor.extract, corpus))
    outcomes = defaultdict(int)
    for facet in facets:
        outcomes[facet.outcome.value] += 1
    print(f"facets: {dict(outcomes)}", flush=True)

    print("== embeddings (Titan v2) ==", flush=True)
    embedder = titan_embedder()
    facet_embeddings = embed_batch(embedder, [f.embed_text() for f in facets])
    digest_embeddings = embed_batch(embedder, [trace_digest(t)[:8000] for t in corpus])

    print("== Test 1: domain recovery ==", flush=True)
    # This corpus records exactly ONE rollout per (domain, task_id), so task-identity recovery is
    # degenerate (k = n makes any clustering trivially perfect). The available non-trivial ground
    # truth is the tau2 domain (airline / retail / telecom): cluster at k=3 and measure recovery.
    truth = [label.split("/")[0] for label in ground_truth_labels(corpus)]
    k_true = len(set(truth))
    facet_labels, _ = cluster_facets(facets, facet_embeddings, k=k_true, seed=0)
    digest_labels, _ = cluster_facets(facets, digest_embeddings, k=k_true, seed=0)
    rec_facet = recovery_report(facet_labels.tolist(), truth)
    rec_digest = recovery_report(digest_labels.tolist(), truth)
    results["test1_domain_recovery"] = {
        "facet_embedding": rec_facet.model_dump(),
        "raw_digest_embedding_baseline": rec_digest.model_dump(),
    }
    print(
        f"facet: purity={rec_facet.purity:.3f} ARI={rec_facet.adjusted_rand_index:.3f} | "
        f"digest: purity={rec_digest.purity:.3f} ARI={rec_digest.adjusted_rand_index:.3f}",
        flush=True,
    )

    print("== build pool (budget 30) + ours-K8 ==", flush=True)
    pool_set = build_scenario_set(
        corpus, facets, lite, embedder, ScenarioBuildConfig(budget=POOL_BUDGET, seed=0)
    )
    pool_set.save(OUT_DIR / "scenario_pool_tau_bench.json")
    pool_ids = [s.scenario_id for s in pool_set.scenarios]
    id_by_trace = {s.provenance[0]: s.scenario_id for s in pool_set.scenarios}
    print(f"pool: {len(pool_ids)} scenarios, coverage {pool_set.corpus_coverage:.0%}", flush=True)

    # Our method, restricted to the pool: recluster the pool members' facets, select K.
    pool_trace_ids = {s.provenance[0] for s in pool_set.scenarios}
    pool_rows = [i for i, f in enumerate(facets) if f.trace_id in pool_trace_ids]
    pool_facets = [facets[i] for i in pool_rows]
    pool_embeddings = facet_embeddings[np.asarray(pool_rows)]
    pool_labels, _ = cluster_facets(pool_facets, pool_embeddings, seed=0)
    ours = hybrid_select(pool_facets, pool_embeddings, pool_labels, OURS_K)
    ours_weights = {id_by_trace[s.trace_id]: s.weight for s in ours}
    results["ours_selection"] = [s.model_dump() for s in ours]

    print("== Test 4: closed-loop verification of ours-K8 ==", flush=True)
    world_model = WorldModel.load(str(WM_DIR), lite, telemetry_root=str(REPO / ".wmh"))
    ours_scenarios = [s for s in pool_set.scenarios if s.scenario_id in ours_weights]
    ours_set = pool_set.model_copy(update={"scenarios": ours_scenarios})
    verification = verify_scenarios(
        ours_set, corpus, world_model, LLMAgent(lite), ChecklistJudge(lite), max_steps=MAX_STEPS
    )
    results["test4_verification"] = {
        "back_agreement_rate": verification.back_agreement_rate,
        "solvable_rate": verification.solvable_rate,
        "verdicts": [v.model_dump() for v in verification.verdicts],
    }
    print(
        f"back-agreement {verification.back_agreement_rate:.0%}, "
        f"solvable {verification.solvable_rate:.0%}",
        flush=True,
    )

    print("== Test 2: score matrix (4 agents x pool x 3 passes) ==", flush=True)
    micro = bedrock(NOVA_MICRO)
    agents = {
        "nova-micro-t0.0": LLMAgent(micro, temperature=0.0),
        "nova-micro-t0.9": LLMAgent(micro, temperature=0.9),
        "nova-lite-t0.0": LLMAgent(lite, temperature=0.0),
        "nova-lite-t0.9": LLMAgent(lite, temperature=0.9),
    }
    matrix = score_matrix(
        world_model,
        agents,
        pool_set.scenarios,
        ChecklistJudge(lite),
        passes=PASSES,
        max_steps=MAX_STEPS,
        workers=WORKERS,
    )
    subsets = {"ours-k8": ours_weights, **random_subsets(pool_ids, OURS_K, seeds=(0, 1, 2))}
    fidelity = fidelity_report(matrix, pool_ids, subsets)
    results["test2_fidelity"] = fidelity.model_dump()
    results["score_matrix"] = matrix.model_dump()
    for method in fidelity.methods:
        print(
            f"{method.method:20} mae={method.mae:.3f} spearman={method.spearman:+.2f} "
            f"kendall={method.kendall:+.2f}",
            flush=True,
        )

    results["wall_clock_seconds"] = round(time.time() - t0, 1)
    out = OUT_DIR / "scenario_e2e_results_tau_bench.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out} ({results['wall_clock_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
