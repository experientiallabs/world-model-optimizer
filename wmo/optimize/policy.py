"""The routing policy artifact: what the routing optimizer emits and the endpoint serves.

An endpoint = {world model, policy, evidence, URL}; this module is the policy leg.

WHICH KIND THE PRODUCT USES, because the four below are not peers:

- `knn` is THE learned router. `wmo optimize model` fits nothing else (it pins
  `kind="knn"`), so every optimized endpoint serves a knn policy.
- `static` is not an algorithm, it is the pre-optimization state: an endpoint must serve from
  the moment it exists, before any fit has run, and the improvement report needs an honest
  "before" to measure against. Two producers choose its model differently: the hosted platform
  seeds one over the strongest serving model available to the org, while `wmo optimize route
  pin` installs one for whichever `--model` an operator names, which may deliberately be a
  weaker one.
- `rank` is a RESEARCH DIRECTION, retained deliberately: a faithful replication kept for
  comparison, reachable only through the manual `wmo optimize route fit --kind rank`. The
  staged pipeline never fits one, so nothing serves one today - but the runtime DOES dispatch a
  manually installed rank artifact through `rank_decision`, so read this as unfitted by the
  product rather than unservable. It is also the only kind with clusters, which is why a
  request log's cluster columns are empty for everything the product serves.
- `linear` is an offline-fitted, pre-inference router for workloads where model or reasoning
  effort is the dominant axis. It predicts the weak and strong arm rewards from one deterministic
  request embedding and serves the strong arm only when their predicted uplift clears a frozen
  threshold. The artifact stores plain numeric weights, never a pickle or executable estimator.

The four kinds:

- `static`: every request goes to `default_model`. Valid without any optimizer run, so an
  endpoint serves from day one and the improvement report has an honest "before" state.
- `knn`: the validated champion (see `wmo.optimize.knn` for the fit and the measured result).
  Nonparametric: retrieve the request's neighbors among the fit scenarios, read each pool
  model's sim-weighted mean reward over exactly those neighbors, and route to the argmax ONLY
  when a paired statistical guard says the evidence supports leaving the baseline. Otherwise the
  baseline (`guard_model`) serves. The neighbor evidence is heavy (an L2-normalized fit
  embedding matrix plus the per-scenario reward and cost cells), so it lives in a `.npz`
  sidecar next to `policy.json` (`knn_bank_path`) and loads lazily on first use.
- `rank`: the Avengers cluster-rank router (arXiv 2505.19797), replicated faithfully from the
  reference implementation (ZhangYiqun018/Avengers, core/routing/rank_router.py): embed the
  request, softmax the distances to the `top_k_clusters` nearest k-means centres
  (`-beta * (1 - centre . query)` logits), score every pool model by the probability-weighted
  reciprocal of its per-cluster accuracy rank (`1 / (rank + 0.1)`, summed over the mixed
  clusters the model appears in), and route to the argmax. `default_rank` is the floor for a
  model absent from ALL of the mixed clusters (`1 / default_rank`); a model ranked in one and
  absent from another simply collects nothing from the latter, it is not charged a default rank
  there. This matches the reference.
- `linear`: a two-head potential-outcome router. Both heads score the same normalized request
  vector; their clipped strong-minus-weak difference is compared with `linear_threshold`.
  This adds no inference call when paired with the deterministic hashing embedder.

The FIT that produces rank policies lives in `wmo.optimize.routing`; this module pins the
artifact schema and the serve-time selection so serving, reports, and the platform stay stable
across fitter iterations.

Serve-time stickiness: `select_model` keeps a conversation's incumbent model whenever the policy
is sticky (the default). Provider prompt caches are per-model, so switching mid-conversation
forfeits warm cache reads and pays cold writes; until the fitter learns a real switching rule
(expected gain vs switch cost), pure affinity is the honest default.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from wmo.core.files import write_bytes_atomic
from wmo.optimize.compression import (
    CompressionConfig,
    Compressor,
    compression_signature,
    same_compression,
    servable_compressor,
)
from wmo.providers.base import (
    Embedder,
    ProviderConfig,
    ProviderKind,
    UsageReportingEmbedder,
)
from wmo.providers.pool import PoolEntry
from wmo.providers.registry import get_provider
from wmo.retrieval.embedders import BatchedEmbedder, HashingEmbedder
from wmo.tracking.pricing import cost_usd

POLICY_VERSION = 2
POLICY_FILENAME = "policy.json"

# Avengers reference defaults (config/experts_template.yaml: top_k=2, beta=6.0,
# default_rank=999). Kept on the policy so serving needs no side-channel config.
DEFAULT_TOP_K_CLUSTERS = 2
DEFAULT_BETA = 6.0
DEFAULT_RANK = 999

# kNN champion defaults: the exact configuration validated on routerbench-ours9
# (+1.0 accuracy point over the best single model at -27% cost, 5/5 split seeds).
KNN_BANK_FILENAME = "policy_knn_bank.npz"
# The sidecar suffix a fit DERIVES from its policy path, so two policies fitted into one
# directory own two banks instead of racing for one shared name. APPENDED rather than
# substituted for the policy's extension (`support.json` -> `support.json.bank.npz`): appending
# is injective on filenames, while replacing the extension would map `support.json` and
# `support.yaml` onto one bank and reintroduce the collision. `KNN_BANK_FILENAME` above stays
# the resolution fallback for artifacts that record no path of their own; see
# `RoutingPolicy.knn_bank_path`.
KNN_BANK_SUFFIX = ".bank.npz"
DEFAULT_RAG_NUM = 50
DEFAULT_RAG_THRES = 0.95
DEFAULT_KNN_Z = 0.5
DEFAULT_KNN_MIN_PAIRS = 8
# Below this many paired neighbors the guard floors its standard error at the maximal binomial
# SE, sqrt(0.25/n). A SMALL-SAMPLE correction only: on a thin bank a lucky zero-variance
# neighborhood otherwise makes any positive mean difference look significant, which is how a
# small-bank router talks itself into pricier-and-worse picks. Large-n empirical SEs are
# reliable, and flooring them there would just tax real wins.
SE_FLOOR_MAX_PAIRS = 30

# Banks are read once per policy instance; the lock is module level so the cache costs the
# policy no per-instance state (a lock stored on the model would make two otherwise identical
# policies compare unequal).
_BANK_LOAD_LOCK = threading.Lock()


def write_artifact_atomically(path: Path, payload: bytes) -> None:
    """Write `payload` to `path` through a staging file, replacing it in one step.

    Serving reads an artifact directory while the optimizer writes it, and a half-written
    policy.json is a mount failure rather than a slightly stale endpoint. It is also what lets
    a command that writes several artifacts promise that a failure leaves the old ones intact.
    `KnnBank.save` stages the same way for the sidecar it streams through numpy.

    The per-call staging name, the cleanup on failure, and the durability are
    `wmo.core.files.write_bytes_atomic`'s; this wrapper is the artifact layer's name for it, kept
    because the reason above is the artifact directory's, not a property of writing files.
    """
    write_bytes_atomic(path, payload)


def knn_bank_path_for(policy_path: Path) -> Path:
    """The evidence sidecar that belongs to one policy file.

    `models/support.json` -> `models/support.json.bank.npz`. Distinct policy filenames always
    give distinct bank filenames (see `KNN_BANK_SUFFIX`), and one owner for the derivation means
    the fitter, the CLI's console line, and any tooling that cleans an artifact directory cannot
    disagree about which `.npz` belongs to which policy.
    """
    return policy_path.with_name(f"{policy_path.name}{KNN_BANK_SUFFIX}")


class EmbedderSpec(BaseModel):
    """How to reproduce the policy's embedding function at serve time.

    `hashing` is deterministic, offline, and credential-free, so a policy file is fully
    self-contained. `openai` uses a direct OpenAI embedding model, and `azure` uses an Azure
    embedding deployment. Both use the model-pool credential convention where `api_key_env`
    names the key variable. The fitter records the exact backend and serving reconstructs it.
    """

    kind: Literal["hashing", "openai", "azure"] = "hashing"
    # gt=0 because a zero-width embedding is not a smaller embedding, it is no embedding: it
    # would reach the provider as `dimensions=0` and build a bank of empty rows.
    dim: int = Field(default=512, gt=0)
    deployment: str | None = None  # azure embedding deployment name
    endpoint: str | None = None
    api_key_env: str | None = None
    batch: int = 256  # provider embeds are chunked to this many texts per request

    @model_validator(mode="after")
    def _validate_backend(self) -> EmbedderSpec:
        if self.kind == "azure" and not (self.deployment and self.endpoint):
            raise ValueError("an azure embedder spec needs deployment and endpoint")
        if self.kind == "openai" and not self.deployment:
            raise ValueError("an openai embedder spec needs an embedding model")
        return self

    def build(self) -> Embedder:
        if self.kind == "hashing":
            return HashingEmbedder(dim=self.dim)
        api_key = None
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(
                    f"embedder spec: environment variable {self.api_key_env} is unset or "
                    "empty; export that account's API key"
                )
        provider = get_provider(
            ProviderConfig(
                kind=(ProviderKind.OPENAI if self.kind == "openai" else ProviderKind.AZURE_OPENAI),
                model=self.deployment or "",
                embed_model=self.deployment,
                embed_dim=self.dim,
                endpoint=self.endpoint if self.kind == "azure" else None,
                api_version="2024-10-21" if self.kind == "azure" else None,
            ),
            api_key=api_key,
        )
        return BatchedEmbedder(provider, batch=self.batch)


def embedder_provenance(spec: EmbedderSpec) -> str:
    """The embedding-function half of `fitted_from`, taken from the spec serving rebuilds.

    Derived from `EmbedderSpec` rather than from the caller's flags, so it cannot describe an
    embedder different from the one recorded in the artifact. The azure RESOURCE is part of the
    identity and not just the deployment name: two accounts routinely hold a deployment of the
    same name and dimension, their embeddings are not interchangeable, and a refit that only
    repointed the endpoint therefore has to read as a different fit. The credential variable is
    left out on purpose: renaming it does not move a single vector.
    """
    tag = f"{spec.kind}-{spec.dim}"
    if spec.kind == "hashing":
        return tag
    if spec.kind == "openai":
        return f"{tag}/{spec.deployment}"
    return f"{tag}/{spec.deployment}@{spec.endpoint}"


@dataclass(frozen=True)
class KnnBank:
    """The fit-split evidence a `knn` policy routes against: the `.npz` sidecar's contents.

    Arrays are aligned: row `i` is `scenario_ids[i]`, column `j` is `models[j]`. A reward or
    cost cell is NaN when that model has no scored episode for that scenario, which is how a
    ragged outcome matrix stays a dense array; the router weighs only the scored cells.

    Embeddings are float32 and L2-normalized at fit time, so serve-time neighbor search is one
    matrix product. float32 halves a bank that is already ~10MB at 3072 dimensions, and costs
    nothing measurable: cosine similarities agree with float64 to ~1e-7, far below the reward
    differences the guard tests.
    """

    embeddings: np.ndarray  # (scenarios, dim) float32, L2-normalized
    rewards: np.ndarray  # (scenarios, models) float32, NaN where unscored
    costs: np.ndarray  # (scenarios, models) float32 USD, NaN where unscored
    models: list[str]
    scenario_ids: list[str]

    def __post_init__(self) -> None:
        scenarios, models = len(self.scenario_ids), len(self.models)
        if not scenarios or not models:
            raise ValueError("a knn bank needs at least one scenario and one model")
        for name, array in (("rewards", self.rewards), ("costs", self.costs)):
            if array.shape != (scenarios, models):
                raise ValueError(
                    f"{name} has shape {array.shape}, expected "
                    f"({scenarios}, {models}) from the scenario and model lists"
                )
        if self.embeddings.shape[0] != scenarios or self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings has shape {self.embeddings.shape}, expected "
                f"({scenarios}, dim) from the scenario list"
            )

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def mean_costs(self) -> np.ndarray:
        """Per-model mean cost over the bank's scored cells (NaN for a model with none).

        The unit the guard's pricier-than-baseline test compares. Cells, not episodes: a model
        that happened to be rerun more often on some scenario does not get extra weight.
        """
        scored = ~np.isnan(self.costs)
        counts = scored.sum(axis=0)
        totals = np.where(scored, np.nan_to_num(self.costs), 0.0).sum(axis=0)
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    def save(self, path: Path) -> None:
        """Write the sidecar atomically (a half-written 10MB bank must not be loadable).

        Staged under a name unique to this call, for the reason spelled out in
        `write_artifact_atomically`: two fits racing on one `--out` derive the same bank path,
        and a shared staging file would let one publish the OTHER's evidence under its own name.
        A bank is streamed through numpy rather than buffered into bytes, so it stages itself
        instead of going through that helper.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with staging.open("wb") as handle:
                np.savez(
                    handle,
                    embeddings=self.embeddings.astype(np.float32),
                    rewards=self.rewards.astype(np.float32),
                    costs=self.costs.astype(np.float32),
                    models=np.asarray(self.models),
                    scenario_ids=np.asarray(self.scenario_ids),
                )
            staging.replace(path)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> KnnBank:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                embeddings=np.asarray(data["embeddings"], dtype=np.float32),
                rewards=np.asarray(data["rewards"], dtype=np.float32),
                costs=np.asarray(data["costs"], dtype=np.float32),
                models=[str(name) for name in data["models"]],
                scenario_ids=[str(sid) for sid in data["scenario_ids"]],
            )


class ClusterRanking(BaseModel):
    """One fitted cluster: its centroid and the pool models ranked by in-cluster accuracy."""

    cluster_id: int
    label: str = ""  # human-readable, surfaces as cluster_label in the request log
    centroid: list[float]
    ranking: list[str] = Field(min_length=1)  # pool entry names, best first
    scores: dict[str, float] = Field(default_factory=dict)  # per-model mean reward (evidence)
    costs: dict[str, float] = Field(default_factory=dict)  # per-model mean cost (evidence)
    # Per-model scored EPISODES behind those means: the support the fit-time guard weighs, kept
    # so `rerank_policy` can re-apply the identical check without the outcome matrix.
    support: dict[str, int] = Field(default_factory=dict)
    total: int = 0  # fit scenarios that landed in this cluster
    # D-COMPRESS: this cluster's compression choice, fitted jointly with the model ranking.
    # None (the default) = no cluster-level choice; consumers fall back to the policy-level
    # `compression`. Additive: pre-compression artifacts load unchanged.
    compression: CompressionConfig | None = None


# How a non-baseline candidate fared, as `RoutingEvidence.gate` records it. "passed" and
# "reverted" are the two outcomes of the paired statistical guard; "novelty-abstain" is the
# coverage floor firing before the guard ever runs. "contained" belongs to the containment gate
# (a global win-vs-baseline check on top of the neighborhood's verdict), which is NOT shipped:
# the value is pinned here so that landing it later does not change a persisted log's vocabulary,
# and nothing emits it today.
GateOutcome = Literal["passed", "reverted", "contained", "novelty-abstain"]

# Why this request went where it did, in the one distinction an offline analysis needs: "greedy"
# means the router served the candidate its own scoring preferred (which includes the baseline
# winning on merit), "fallback-forced" means something overrode that preference back to the
# baseline. Counting fallback-forced requests is how a coverage problem shows up as a number
# rather than as a string that has to be grepped out of `reason`.
Propensity = Literal["greedy", "fallback-forced"]


class RoutingEvidence(BaseModel):
    """The numbers behind one routing decision, as structured fields rather than prose.

    `reason` is written for a human reading a request log; this is the same decision in a shape
    something can aggregate. The fields are exactly what `knn_decision` already computes on the
    way to its answer, so recording them costs nothing and invents nothing.

    The paired statistics are present only when the guard actually ran, which means only when a
    non-baseline candidate led the neighborhood: a request the baseline won outright, or that
    abstained before any candidate was scored, has no paired comparison to report and leaves them
    None rather than reporting a zero that would average in as evidence.
    """

    mean_diff: float | None = None  # mean paired reward difference, candidate minus baseline
    se: float | None = None  # its standard error, after any small-sample floor
    n_pairs: int | None = None  # neighbors scored on BOTH sides, the guard's sample size
    gate: GateOutcome | None = None  # None when no gate was reached
    propensity: Propensity
    # The incumbent's cache credit that entered this decision's cost arithmetic, in USD
    # (cache-aware policies only; None whenever no credit was applied). What lets the request
    # log explain a cache-driven pick: "the incumbent was effectively cheaper by this much".
    cache_credit_usd: float | None = None


class RoutingDecision(BaseModel):
    """Where one request goes and why (the request log's model/cluster/routing_reason)."""

    model: str
    cluster_id: int | None = None
    cluster_label: str = ""
    reason: str
    # Populated by `knn_decision`; None for the kinds that compute no paired evidence (static and
    # rank) and for a sticky decision, which never consults the policy at all.
    evidence: RoutingEvidence | None = None

    # The L2-normalized vector this request was routed on, attached by `select_model` so serving
    # can persist it (`wmo.serving.query_embeddings`) without re-embedding the text. Private
    # because it is per-request state, not part of the decision's shape: it must not appear in a
    # log row, a report, or a comparison between two decisions.
    _query_embedding: np.ndarray | None = PrivateAttr(default=None)
    _router_cost_usd: float = PrivateAttr(default=0.0)

    def attach_query_embedding(self, vector: np.ndarray) -> None:
        """Record the vector this decision was made from (see `_query_embedding`)."""
        self._query_embedding = vector

    def query_embedding(self) -> np.ndarray | None:
        """The vector this decision was made from, when one was embedded for it."""
        return self._query_embedding

    def attach_router_cost(self, value: float) -> None:
        """Record the semantic embedder's billable cost for this one decision."""
        self._router_cost_usd = value

    def router_cost_usd(self) -> float:
        """The routing decision's own provider cost, zero for offline embedders."""
        return self._router_cost_usd


class RoutingPolicy(BaseModel):
    """The persisted policy artifact (see module docstring)."""

    version: int = POLICY_VERSION
    kind: Literal["static", "rank", "knn", "linear"]
    default_model: str  # the static answer; also the fallback for degenerate rank inputs
    pool: list[PoolEntry]  # snapshot of the roster this policy was defined over
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    clusters: list[ClusterRanking] = Field(default_factory=list)
    top_k_clusters: int = Field(default=DEFAULT_TOP_K_CLUSTERS, ge=1)
    beta: float = Field(default=DEFAULT_BETA, gt=0.0)
    default_rank: int = Field(default=DEFAULT_RANK, ge=1)
    sticky: bool = True  # keep a conversation's incumbent model (see module docstring)
    # Cache-aware decisions (Silen directive, 2026-07-28): when on, a knn policy with a
    # conversation incumbent does NOT take the unconditional sticky return; the decision runs
    # with the incumbent's cache credit inside its COST arithmetic (the pick_lam tilt and the
    # guard's pricier test), so stickiness becomes a priced advantage the router can weigh
    # against quality evidence instead of a blunt rule. The quality evidence test itself is
    # unchanged: cache state never buys quality confidence. Off (the default) is bit-identical
    # to today for every policy file, eval, and serving path.
    #
    # Safety outranks cache economics: the novelty abstain and the no-scored-candidates
    # returns still serve the pinned fallback WITHOUT the switch gate. Consequence vs plain
    # sticky mode (which never reaches the novelty check): a mid-conversation turn that looks
    # unlike the fit data leaves a cache-warm incumbent for the fallback and pays full
    # prefill. Ruled deliberate (routing master, 2026-07-28): on novel input the pinned-
    # fallback contract wins. The switch gate governs evidence-backed decisions only; an
    # incumbent absent from the fitted bank cannot be tested and sticks.
    cache_aware: bool = False
    # ProxRouter-inspired support tilt (2510.09852, ADAPTED to clusters: their exponential
    # tilt reweights nonparametric scores by a prior; ours multiplies cluster probabilities
    # by support^gamma so thin outlier clusters lose routing weight). 0 = off (reference).
    support_tilt_gamma: float = Field(default=0.0, ge=0.0)
    # The unit a cost knob trades against one reward point: "one average call". Derived per kind,
    # each matching its own knob's reference implementation. rank: the fit set's mean cost per
    # scored EPISODE. knn: the mean of the per-model mean cell costs, which is what
    # `pick_lam`'s reference divides by (coverage-robust, since a model measured on few
    # scenarios cannot drag the unit toward its own price). 0 when the fit carried no costs.
    cost_scale: float = 0.0
    # The fit-time only-replace-if-better guard, recorded so any later transform of this policy
    # (today `rerank_policy`) re-applies the SAME floor instead of quietly dropping it. None
    # means the fit ran unguarded.
    guard_model: str | None = None
    min_support: int | None = None  # scored episodes a challenger needs to lead its cluster
    guard_margin: float | None = None  # reward the challenger must beat the guard by
    fitted_from: str | None = None  # provenance: the outcome matrix the fitter used
    # Scenario-level provenance for evaluation integrity. Reports exclude these ids when they
    # appear in the supplied matrix; an entirely separate matrix has no overlap and is all held
    # out. Empty is retained for static and legacy artifacts that learned no recorded split.
    fit_scenario_ids: list[str] = Field(default_factory=list)
    # D-COMPRESS: the endpoint-level compression choice, applied by the serving compress stage
    # BEFORE routing (the router embeds what the model will see), so it cannot vary by cluster
    # at serve time; per-cluster overrides live on ClusterRanking for the joint fit and eval
    # grids. None (the default) = compression off, today's behavior exactly.
    compression: CompressionConfig | None = None
    # D-COMPRESS representation consistency: the compression config the ROUTING EVIDENCE was
    # fitted under (bank rows, cluster centroids, and the novelty floor quantile all live in the
    # embedding geometry of whatever text the fit embedded). None (the default) = fitted on raw
    # text, which is every artifact written before this field existed, so they load unchanged.
    # A policy that routes may only serve the config it was fitted under: see
    # `_check_compression`.
    fit_compression: CompressionConfig | None = None

    # kNN policies only (see module docstring and `wmo.optimize.knn`). The fitter records the
    # bank it actually wrote (`knn_bank_path_for(<policy path>)`), so serving resolves the
    # sidecar EXPLICITLY instead of by convention and two policies can share a directory. The
    # value is a bare FILENAME resolved next to policy.json, so a model directory stays
    # portable; an absolute path is honored as given (research code and tests point at banks
    # elsewhere). The default is the legacy conventional name, which is what an artifact
    # written before the fitter recorded a derived name resolves to.
    knn_bank_path: str = KNN_BANK_FILENAME
    rag_num: int = Field(default=DEFAULT_RAG_NUM, ge=1)  # neighbor budget
    # A neighbor is a fit scenario with similarity above `rag_thres` times the `rag_num`-th best
    # similarity: a RELATIVE rule, so a query in a dense region keeps more evidence than one out
    # on its own, instead of every query being handed exactly k neighbors of any quality.
    rag_thres: float = Field(default=DEFAULT_RAG_THRES, gt=0.0, le=1.0)
    # The confidence knob: standard errors of paired evidence a non-baseline pick must clear
    # (doubled when the pick is pricier than the baseline). Higher = stricter = fewer requests
    # leave the baseline. 0 routes on any positive mean difference.
    knn_z: float = Field(default=DEFAULT_KNN_Z, ge=0.0)
    # How the guard treats the two sides of the price comparison (R1's `stat`/`stat_asym`).
    # "symmetric" (the champion): a pricier pick needs 2z standard errors, a cheaper one needs z,
    # so BOTH must be positively supported. "asymmetric": a pricier pick needs z, and a cheaper
    # one only has to clear -z, i.e. it is accepted unless the evidence says it is significantly
    # worse. The asymmetric bar is what makes `pick_lam` able to act at all: under the symmetric
    # bar a cost-tilted cheap pick is usually reverted straight back to the pricier baseline,
    # which spends MORE, not less (measured; see `wmo.optimize.knn.apply_cost_quality`).
    guard_mode: Literal["symmetric", "asymmetric"] = "symmetric"
    knn_min_pairs: int = Field(default=DEFAULT_KNN_MIN_PAIRS, ge=0)  # neighbors scored on both
    se_floor: bool = True  # small-sample variance floor (see SE_FLOOR_MAX_PAIRS)
    # Novelty floor: queries whose best bank similarity is below this abstain to the baseline
    # (None = off). Set at fit time from the floor_q quantile of bank self-NN similarities;
    # the serving-side coverage/robustness knob for task drift.
    floor_sim: float | None = None
    # The QUANTILE `floor_sim` came from. Kept alongside it because the threshold alone cannot be
    # read back: 0.4 similarity is a strict floor on one bank and a loose one on another, so
    # nothing downstream can tell which coverage setting a policy is on (which dial position it
    # matches, what to report on the config endpoint) from `floor_sim`. None means unknown: a
    # policy written before this field existed, or one whose threshold was set by hand.
    floor_q: float | None = Field(default=None, ge=0.0, le=1.0)
    # The cost knob (R1 `pick_lam`): the raw pick maximizes
    # `profile[m] - pick_lam * mean_cost[m] / cost_scale` instead of the profile alone, so
    # pick_lam is "reward points paid per average-call-cost unit". The guard runs AFTER, on the
    # untilted paired evidence, so cost pressure can only ever demote a pick, never promote one
    # the evidence rejects. 0 = off (the validated champion). Slide it with
    # `wmo.optimize.knn.apply_cost_quality`, which maps one operator-facing dial onto this.
    pick_lam: float = Field(default=0.0, ge=0.0)
    # The operator dial that produced the knobs above, when one did: 0 = max quality,
    # 1 = max savings (`wmo.optimize.knn.apply_cost_quality`). None means "as fitted", never
    # slid. Provenance, not an input: serving reads the knobs, and the mapping is absolute, so
    # re-applying any dial setting to any policy of this kind lands on the same knobs.
    cost_quality: float | None = Field(default=None, ge=0.0, le=1.0)

    # Linear potential-outcome policies only. The two reward heads share `embedder`; serving
    # normalizes the request vector, clips both predictions to [0, 1], and routes to the strong
    # arm when predicted strong minus predicted weak reaches the frozen threshold. Plain JSON
    # arrays keep the policy auditable and avoid loading an executable estimator artifact.
    linear_weak_model: str | None = None
    linear_strong_model: str | None = None
    linear_weak_weights: list[float] = Field(default_factory=list)
    linear_strong_weights: list[float] = Field(default_factory=list)
    linear_weak_bias: float = 0.0
    linear_strong_bias: float = 0.0
    linear_threshold: float = 0.0

    # Set by `save`/`load` so a relative `knn_bank_path` resolves against the policy file, and
    # the lazily loaded bank. Private: not part of the artifact.
    _source_dir: Path | None = PrivateAttr(default=None)
    _bank: KnnBank | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate(self) -> RoutingPolicy:
        names = {entry.name for entry in self.pool}
        if self.default_model not in names:
            raise ValueError(
                f"default_model '{self.default_model}' is not in the policy pool "
                f"(available: {sorted(names)})"
            )
        if self.guard_model is not None and self.guard_model not in names:
            raise ValueError(
                f"guard_model '{self.guard_model}' is not in the policy pool "
                f"(available: {sorted(names)})"
            )
        self._check_compression()
        if self.kind != "rank" and self.clusters:
            raise ValueError(f"a {self.kind} policy carries no clusters; use kind='rank'")
        if self.kind == "knn":
            # The whole algorithm is "leave the baseline only on evidence", so a knn policy
            # without a baseline is not a weaker policy, it is an undefined one. default_model
            # and guard_model are the same thing here, and being able to set them apart would
            # only invite two disagreeing fallbacks.
            if self.guard_model is None:
                raise ValueError(
                    "a knn policy needs guard_model set to its baseline model (the fallback "
                    "every unsupported request reverts to); fit with --fallback"
                )
            if self.guard_model != self.default_model:
                raise ValueError(
                    f"knn guard_model '{self.guard_model}' must equal default_model "
                    f"'{self.default_model}': the baseline and the fallback are one model"
                )
            if self.pick_lam > 0.0 and self.cost_scale <= 0.0:
                # Caught here rather than per request: a policy whose cost knob has no cost
                # unit to divide by is an unservable artifact, and finding that out inside a
                # request would turn a config mistake into a 502 per call.
                raise ValueError(
                    f"pick_lam={self.pick_lam:g} needs a positive cost_scale, but this policy "
                    "has none; refit on a matrix that carries per-episode costs "
                    "(`wmo optimize route fit --kind knn`) before turning the cost knob"
                )
        if self.kind == "rank":
            if not self.clusters:
                raise ValueError("a rank policy needs at least one fitted cluster")
            for cluster in self.clusters:
                unknown = [name for name in cluster.ranking if name not in names]
                if unknown:
                    raise ValueError(
                        f"cluster {cluster.cluster_id} ranks {unknown}, "
                        f"not in the policy pool (available: {sorted(names)})"
                    )
                if len(cluster.centroid) != self.embedder.dim:
                    raise ValueError(
                        f"cluster {cluster.cluster_id} centroid has dim "
                        f"{len(cluster.centroid)}, embedder dim is {self.embedder.dim}"
                    )
        if self.kind == "linear":
            if self.linear_weak_model is None or self.linear_strong_model is None:
                raise ValueError("a linear policy needs weak and strong model names")
            if self.linear_weak_model == self.linear_strong_model:
                raise ValueError("a linear policy needs distinct weak and strong models")
            unknown = [
                name
                for name in (self.linear_weak_model, self.linear_strong_model)
                if name not in names
            ]
            if unknown:
                raise ValueError(
                    f"linear policy models {unknown} are not in the policy pool "
                    f"(available: {sorted(names)})"
                )
            for label, weights in (
                ("weak", self.linear_weak_weights),
                ("strong", self.linear_strong_weights),
            ):
                if len(weights) != self.embedder.dim:
                    raise ValueError(
                        f"linear {label} head has {len(weights)} weights, "
                        f"embedder dim is {self.embedder.dim}"
                    )
                if not all(math.isfinite(value) for value in weights):
                    raise ValueError(f"linear {label} head weights must all be finite")
            scalars = (
                self.linear_weak_bias,
                self.linear_strong_bias,
                self.linear_threshold,
            )
            if not all(math.isfinite(value) for value in scalars):
                raise ValueError("linear policy biases and threshold must all be finite")
        return self

    def _check_compression(self) -> None:
        """The two D-COMPRESS mount gates, applied wherever a policy is built or loaded.

        1. SERVABILITY: every compressor this artifact names must exist and must attest append
           stability (`servable_compressor`). Unknown ids and churny compressors fail here, at
           mount, rather than on the first request mid-conversation.
        2. REPRESENTATION CONSISTENCY: a policy that ROUTES on embeddings may only serve the
           compression config its evidence was fitted under. A bank, its cluster centroids, and
           its novelty floor are geometry in the space of the text the fit embedded; serving a
           different representation against them was measured (C2 Q2) to trip the novelty floor
           10-13x more often, collapse route-away, and raise cost 11-41% while accuracy sat flat
           to negative. That is a silently BROKEN policy, not a degraded one, so it does not
           mount. Static policies embed nothing and are exempt: there is no geometry to mismatch.

        Per-cluster overrides are checked for servability only. They are fit and eval inputs
        (`ClusterRanking.compression`); the serve-time stage reads the policy-level config,
        because cluster assignment is not known until after routing.
        """
        servable_compressor(self.compression)
        for cluster in self.clusters:
            servable_compressor(cluster.compression)
        if self.kind == "static" or same_compression(self.compression, self.fit_compression):
            return
        raise ValueError(
            f"this {self.kind} policy's routing evidence was fitted on "
            f"{compression_signature(self.fit_compression)}, but the endpoint would serve "
            f"{compression_signature(self.compression)}. A routing bank and its novelty floor "
            "are only valid for the representation they were fitted on. Refit under the serving "
            "config (`wmo optimize route fit --compressor <id> --aggressiveness <a>`), or serve "
            "the config the artifact was fitted under."
        )

    def serving_compressor(self) -> Compressor | None:
        """The compressor the serving stage applies, or None when this endpoint compresses nothing.

        Runs both mount gates again rather than trusting validation, for two reasons: the
        cost/quality dial installs a NEW policy object on a live runtime
        (`wmo.serving.chat.EndpointRuntime._install_policy`), so the compressor has to follow the
        object requests actually read; and a policy assembled in memory through `model_copy`
        never went through a validator at all, so this is the only place a hand-built mismatch
        is caught before it serves traffic.
        """
        self._check_compression()
        return servable_compressor(self.compression)

    def save(self, path: Path) -> None:
        """Write the policy artifact atomically (a torn policy.json must not be loadable)."""
        write_artifact_atomically(path, self.model_dump_json(indent=2).encode("utf-8"))
        self._source_dir = path.parent

    @classmethod
    def load(cls, path: Path) -> RoutingPolicy:
        policy = cls.model_validate_json(path.read_text(encoding="utf-8"))
        policy._source_dir = path.parent
        return policy

    def bank_path(self) -> Path:
        """Where this policy's kNN sidecar lives (see `knn_bank_path`)."""
        if self.kind != "knn":
            raise ValueError(f"a {self.kind} policy has no knn bank")
        candidate = Path(self.knn_bank_path)
        if candidate.is_absolute():
            return candidate
        return (self._source_dir or Path()) / candidate

    def attach_bank(self, bank: KnnBank) -> None:
        """Use `bank` as this policy's evidence instead of reading the sidecar.

        The fitter primes a freshly fitted policy with the bank it just built, so fit-then-
        evaluate does not round trip through disk; tests use it to route against a hand-built
        bank. Validated exactly like a loaded one.
        """
        self._validate_bank(bank, source="the attached bank")
        self._bank = bank

    def knn_bank(self) -> KnnBank:
        """Load (once, then cached) the neighbor evidence this kNN policy routes against."""
        if self.kind != "knn":
            raise ValueError(f"a {self.kind} policy has no knn bank")
        with _BANK_LOAD_LOCK:
            if self._bank is None:
                path = self.bank_path()
                if not path.is_file():
                    raise FileNotFoundError(
                        f"knn policy bank not found at {path}: a knn policy.json is served "
                        f"together with its '{self.knn_bank_path}' sidecar. Copy the sidecar "
                        f"next to the policy file, or refit with "
                        f"`wmo optimize route fit --kind knn`."
                    )
                try:
                    bank = KnnBank.load(path)
                    self._validate_bank(bank, source=str(path))
                except (ValueError, KeyError) as error:
                    raise ValueError(f"invalid knn bank at {path}: {error}") from error
                self._bank = bank
            return self._bank

    def _validate_bank(self, bank: KnnBank, *, source: str) -> None:
        """Check a bank against this policy: same embedder dimension, known models, a baseline."""
        if bank.dim != self.embedder.dim:
            raise ValueError(
                f"{source} has {bank.dim}-dimensional embeddings but the policy's embedder "
                f"spec is {self.embedder.dim}-dimensional; the bank was fitted with a "
                "different embedder than this policy would embed requests with"
            )
        names = {entry.name for entry in self.pool}
        unknown = [model for model in bank.models if model not in names]
        if unknown:
            raise ValueError(
                f"{source} carries models {unknown} that are not in the policy pool "
                f"(available: {sorted(names)})"
            )
        if self.guard_model not in bank.models:
            raise ValueError(
                f"{source} does not cover baseline '{self.guard_model}' (it covers "
                f"{sorted(bank.models)}); the bank was fitted over a different pool"
            )
        column = bank.rewards[:, bank.models.index(self.guard_model)]
        if bool(np.isnan(column).all()):
            raise ValueError(
                f"{source} has no scored reward for baseline '{self.guard_model}' on any fit "
                "scenario, so the guard could never compare a pick against it and every "
                "request would revert unmeasured; fit on a matrix that measured the baseline, "
                "or pin a baseline it did measure"
            )


CHARS_PER_TOKEN = 4  # the documented, conservative prefix estimate (design: cacheaware.md)


def cache_credit_usd(policy: RoutingPolicy, incumbent: str, prefix_chars: int) -> float:
    """The incumbent's expected cache saving on this request, in USD.

    credit = prefix_tokens x (input rate - cached read rate) at the INCUMBENT's pool prices,
    with prefix_tokens estimated as prefix_chars / 4 (conservative; errs low). Every other
    model's credit is zero by definition: a switch pays full prefill. A pool entry with no
    cached_input_per_mtok earns no credit (its cache reads bill at the full input rate,
    matching PoolEntry.cost_usd), and an unknown incumbent earns none either.

    The estimate assumes a warm provider cache, which holds inside the provider prefix-cache
    window (~minutes); the pre-registered replay gate measures under the same assumption.
    """
    if prefix_chars <= 0:
        return 0.0
    entry = next((e for e in policy.pool if e.name == incumbent), None)
    if entry is None or entry.cached_input_per_mtok is None:
        return 0.0
    saving_per_mtok = entry.price().input_per_mtok - entry.cached_input_per_mtok
    if saving_per_mtok <= 0.0:
        return 0.0
    return (prefix_chars / CHARS_PER_TOKEN) * saving_per_mtok / 1_000_000


def select_model(
    policy: RoutingPolicy,
    text: str,
    *,
    incumbent: str | None = None,
    embedder: Embedder | None = None,
    conversation_chars: int = 0,
) -> RoutingDecision:
    """Pick the pool model for one request.

    `text` is whatever the caller deems the routable content (serving passes the latest user
    message). `incumbent` is the model already serving this conversation, if any: a sticky
    policy keeps it (per-model prompt caches make switching expensive), unless it has been
    retired from the pool, in which case the request re-routes as if fresh.

    `embedder` lets a caller that routes many requests reuse ONE embedder built from
    `policy.embedder` (an azure spec otherwise constructs a fresh HTTP client per call); it must
    be the function this policy's spec describes, or the fitted centroids (rank) and neighbor
    bank (knn) are meaningless. Default None builds it from the spec per call.

    `conversation_chars` is the length of the transcript the affinity fingerprint matched (0
    when there is none); a cache-aware knn policy turns it into the incumbent's cache credit
    (`cache_credit_usd`). It is ignored everywhere else, so passing it is always safe.
    """
    names = {entry.name for entry in policy.pool}
    cache_aware_knn = policy.cache_aware and policy.kind == "knn" and incumbent in names
    if incumbent is not None and incumbent in names and policy.sticky and not cache_aware_knn:
        return RoutingDecision(model=incumbent, reason="sticky: conversation affinity")
    if policy.kind == "static":
        return RoutingDecision(model=policy.default_model, reason="static policy")

    resolved_embedder = embedder or policy.embedder.build()
    router_cost = 0.0
    if isinstance(resolved_embedder, UsageReportingEmbedder):
        embedded = resolved_embedder.embed_with_usage([text])
        query = np.asarray(embedded.vectors[0])
        router_cost = cost_usd(embedded.model, embedded.usage)
    else:
        query = np.asarray(resolved_embedder.embed([text])[0])
    if policy.kind == "knn":
        credit = cache_credit_usd(policy, incumbent, conversation_chars) if cache_aware_knn else 0.0
        decision = knn_decision(
            policy,
            query,
            incumbent=incumbent if cache_aware_knn else None,
            cache_credit=credit,
        )
    elif policy.kind == "linear":
        decision = linear_decision(policy, query)
    else:
        decision = rank_decision(policy, query)
    # Normalized separately rather than by normalizing `query` first: all decision functions do
    # their own normalization in their own precision, and pre-normalizing here would perturb the
    # champion's numerical path for the sake of a logging side effect.
    norm = float(np.linalg.norm(query))
    decision.attach_query_embedding(query / norm if norm > 0.0 else query)
    decision.attach_router_cost(router_cost)
    return decision


def linear_decision(policy: RoutingPolicy, query: np.ndarray) -> RoutingDecision:
    """Route between two arms using frozen linear potential-outcome heads."""
    if policy.kind != "linear":
        raise ValueError(f"linear_decision needs a linear policy, got kind='{policy.kind}'")
    if policy.linear_weak_model is None or policy.linear_strong_model is None:
        raise ValueError("linear policy has no weak or strong model")

    vector = np.asarray(query, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return RoutingDecision(
            model=policy.default_model,
            reason=f"linear router: empty embedding, serving fallback {policy.default_model}",
        )
    vector = vector / norm
    weak = float(
        np.clip(
            np.dot(np.asarray(policy.linear_weak_weights, dtype=np.float64), vector)
            + policy.linear_weak_bias,
            0.0,
            1.0,
        )
    )
    strong = float(
        np.clip(
            np.dot(np.asarray(policy.linear_strong_weights, dtype=np.float64), vector)
            + policy.linear_strong_bias,
            0.0,
            1.0,
        )
    )
    uplift = strong - weak
    use_strong = uplift >= policy.linear_threshold
    model = policy.linear_strong_model if use_strong else policy.linear_weak_model
    return RoutingDecision(
        model=model,
        reason=(
            f"linear router: predicted uplift {uplift:.4f} "
            f"{'>=' if use_strong else '<'} threshold {policy.linear_threshold:.4f}"
        ),
    )


def rank_decision(policy: RoutingPolicy, query: np.ndarray) -> RoutingDecision:
    """The Avengers rank-routing core, on an already-embedded query.

    Shared by `select_model` (one live request) and batch evaluation (the benchmark embeds
    all scenarios once) so the served path and the measured path can never diverge. Faithful
    to the reference implementation: the query is L2-normalized HERE, so no caller can skip the
    step and silently change the softmax temperature (`beta * distance` is scale-sensitive even
    though the nearest-cluster order is not); centres are used exactly as fitted, NOT
    re-normalized (k-means centres of unit vectors are near-unit; the reference dots them raw,
    so we do too).
    """
    names = {entry.name for entry in policy.pool}
    centres = np.asarray([cluster.centroid for cluster in policy.clusters])
    vector = np.asarray(query, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        # A zero query keeps its zeros, exactly like sklearn's Normalizer.
        vector = vector / norm
    dists = 1.0 - centres @ vector
    top = np.argsort(dists)[: policy.top_k_clusters]
    logits = -policy.beta * dists[top]
    probs = np.exp(logits - logits.max())
    if policy.support_tilt_gamma > 0.0:
        support = np.asarray(
            [max(policy.clusters[int(index)].total, 1) for index in top], dtype=np.float64
        )
        probs = probs * support**policy.support_tilt_gamma
    probs /= probs.sum()

    scores: dict[str, float] = {}
    for cluster_index, prob in zip(top, probs, strict=True):
        ranking = policy.clusters[int(cluster_index)].ranking
        for name in names:
            if name in ranking:
                rank_score = 1.0 / (ranking.index(name) + 0.1)
                scores[name] = scores.get(name, 0.0) + float(prob) * rank_score
    for name in names:
        scores.setdefault(name, 1.0 / policy.default_rank)

    # Argmax; ties break by pool order (deterministic; the reference relies on dict order).
    pool_order = {entry.name: index for index, entry in enumerate(policy.pool)}
    winner = max(scores.items(), key=lambda kv: (kv[1], -pool_order[kv[0]]))[0]
    nearest = policy.clusters[int(top[0])]
    label = f" ({nearest.label})" if nearest.label else ""
    return RoutingDecision(
        model=winner,
        cluster_id=nearest.cluster_id,
        cluster_label=nearest.label,
        reason=f"rank router: nearest cluster {nearest.cluster_id}{label}",
    )


def knn_decision(
    policy: RoutingPolicy,
    query: np.ndarray,
    *,
    incumbent: str | None = None,
    cache_credit: float = 0.0,
) -> RoutingDecision:
    """The kNN reward-profile core with its paired guard, on an already-embedded query.

    Shared by `select_model` (one live request) and batch evaluation, so the served path and the
    measured path cannot diverge. Three stages, each of which the returned `reason` accounts for:

    1. Neighbors: the fit scenarios whose similarity beats `rag_thres` times the `rag_num`-th
       best similarity. The query is L2-normalized HERE (bank rows already are), so no caller
       can skip the step and turn the similarity threshold into a magnitude test.
    2. Profile: each model's similarity-weighted mean reward over the neighbors it was scored
       on. The raw pick maximizes that profile minus the cost knob's price term,
       `pick_lam * mean_cost / cost_scale` (zero at the default pick_lam=0; ties break by pool
       order, deterministically).
    3. Guard: a non-baseline pick must beat the baseline on PAIRED per-neighbor reward
       differences, mean > `knn_z` standard errors (doubled when the pick is also pricier, the
       asymmetry that kills confidently-wrong pricier-and-worse picks; see `guard_mode` for the
       economic variant that only asks a cheaper pick not to be significantly worse), on at
       least `knn_min_pairs` neighbors scored on both sides. Otherwise the baseline serves.

    The guard is what makes the policy safe to deploy: absent evidence, the answer is the
    baseline, so the worst case is the baseline's behavior rather than a confident stranger's.
    It runs on the UNTILTED evidence and after the cost knob, exactly as in the research code:
    the knob reorders candidates, and the guard then re-vetoes whatever it cannot support, so
    turning the knob up can never talk the router into a pick the evidence rejects.

    Cache-aware calls (`incumbent` supplied by `select_model` when the policy has
    `cache_aware` on) do two things. The `cache_credit` is subtracted from THE INCUMBENT's
    mean cost in the two places cost enters the decision: the pick_lam tilt and the pricier
    tests. And any decision that would ABANDON the incumbent must additionally clear the
    switch gate (see `switch_gate` below): the same paired-evidence bar, anchored on the
    incumbent, at effective prices. With NO incumbent this function is bit-identical to the
    cache-blind path (the offline-eval invariant); quality evidence (profile, paired diffs,
    z thresholds) never sees the credit.
    """
    if policy.kind != "knn":
        raise ValueError(f"knn_decision needs a knn policy, got kind='{policy.kind}'")
    bank = policy.knn_bank()
    baseline = policy.default_model

    vector = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    sims = bank.embeddings @ vector
    if policy.floor_sim is not None and float(np.max(sims)) < policy.floor_sim:
        return RoutingDecision(
            model=baseline,
            reason=(
                f"knn novelty abstain: best similarity {float(np.max(sims)):.3f} below the "
                f"fit-bank floor {policy.floor_sim:.3f}, serving {baseline}"
            ),
            evidence=RoutingEvidence(gate="novelty-abstain", propensity="fallback-forced"),
        )

    budget = min(policy.rag_num, sims.shape[0])
    kth = float(np.sort(sims)[-budget])
    rows = np.flatnonzero(sims > policy.rag_thres * kth)
    if rows.size == 0:
        # Every similarity at or below the cut (possible when the kth is negative, so the
        # threshold moves the wrong way): fall back to the single nearest fit scenario.
        rows = np.asarray([int(np.argmax(sims))])

    weights = np.clip(sims[rows], 0.0, None).astype(np.float64)[:, None]
    rewards = bank.rewards[rows].astype(np.float64)
    scored = ~np.isnan(rewards)
    weight_totals = (scored * weights).sum(axis=0)
    reward_totals = (np.where(scored, np.nan_to_num(rewards), 0.0) * weights).sum(axis=0)
    profile = np.full(weight_totals.shape, np.nan)
    np.divide(reward_totals, weight_totals, out=profile, where=weight_totals > 0.0)

    pool_order = {entry.name: index for index, entry in enumerate(policy.pool)}
    candidates = [
        (index, name) for index, name in enumerate(bank.models) if not np.isnan(profile[index])
    ]
    if not candidates:
        return RoutingDecision(
            model=baseline,
            reason=f"knn: {rows.size} neighbors carry no scored reward, serving {baseline}",
            evidence=RoutingEvidence(propensity="fallback-forced"),
        )
    mean_cost = bank.mean_costs()
    # Cache-aware: the incumbent's expected cost on THIS request is its sticker price minus
    # the cache credit (floored at zero); every other model pays full prefill. Effective
    # costs feed the tilt and the guard's pricier test below; with credit 0 they ARE the
    # mean costs and the whole path is bit-identical to the cache-blind decision.
    effective_cost = mean_cost
    credit_applied = 0.0
    if cache_credit > 0.0 and incumbent is not None and incumbent in bank.models:
        inc_index = bank.models.index(incumbent)
        if not np.isnan(mean_cost[inc_index]):
            effective_cost = mean_cost.copy()
            effective_cost[inc_index] = max(mean_cost[inc_index] - cache_credit, 0.0)
            credit_applied = cache_credit

    def switch_gate(serve_index: int, serve: str) -> RoutingDecision | None:
        """Cache-aware switch gate (design amendment 1, findings/cacheaware.md).

        Serving `serve` would abandon the cache-warm incumbent, so the challenger must clear
        the SAME paired-evidence bar against the incumbent that any pick clears against the
        safety baseline, with the pricier flag on EFFECTIVE costs (incumbent credited,
        challenger at full prefill). None = switch justified (or gate not applicable);
        otherwise the decision that keeps the incumbent. Quality thresholds are identical to
        the baseline guard: cache state raises the switch's price bar, never its confidence.
        """
        if incumbent is None or serve == incumbent:
            return None
        if incumbent not in bank.models:
            # The bank predates this pool model, so the incumbent cannot be tested at all;
            # the amendment's conservative rule applies: an untestable incumbent sticks.
            return RoutingDecision(
                model=incumbent,
                reason=(
                    f"cache-aware switch gate: kept incumbent {incumbent}, not in the fitted "
                    f"bank so {serve} cannot be justified against it"
                ),
                evidence=RoutingEvidence(
                    n_pairs=0,
                    gate="reverted",
                    propensity="fallback-forced",
                    cache_credit_usd=credit_applied or None,
                ),
            )
        inc_index = bank.models.index(incumbent)
        inc_paired = scored[:, serve_index] & scored[:, inc_index]
        inc_diffs = rewards[inc_paired, serve_index] - rewards[inc_paired, inc_index]
        inc_pairs = int(inc_diffs.size)
        inc_mean = float(inc_diffs.mean()) if inc_pairs else 0.0
        inc_error = float(inc_diffs.std(ddof=1)) / inc_pairs**0.5 if inc_pairs > 1 else 0.0
        if policy.se_floor and 0 < inc_pairs < SE_FLOOR_MAX_PAIRS:
            inc_error = max(inc_error, (0.25 / inc_pairs) ** 0.5)
        switch_pricier = bool(effective_cost[serve_index] > effective_cost[inc_index])
        if policy.guard_mode == "asymmetric":
            z_switch = policy.knn_z if switch_pricier else -policy.knn_z
        else:
            z_switch = 2 * policy.knn_z if switch_pricier else policy.knn_z
        if inc_pairs >= policy.knn_min_pairs and inc_mean > z_switch * inc_error:
            return None
        detail = (
            f"{inc_pairs} paired neighbors vs incumbent, delta={inc_mean:+.3f}, needs "
            f"> {z_switch:g}xSE={z_switch * inc_error:.3f} at effective prices"
        )
        if inc_pairs < policy.knn_min_pairs:
            detail = f"{inc_pairs} paired neighbors vs incumbent < {policy.knn_min_pairs}"
        return RoutingDecision(
            model=incumbent,
            reason=(
                f"cache-aware switch gate: kept incumbent {incumbent}, {serve} not "
                f"justified against it ({detail})"
            ),
            evidence=RoutingEvidence(
                mean_diff=inc_mean,
                se=inc_error,
                n_pairs=inc_pairs,
                gate="reverted",
                propensity="fallback-forced",
                cache_credit_usd=credit_applied or None,
            ),
        )

    # The cost knob prices each candidate in average-call units before the argmax; at
    # pick_lam=0 the key is the bare profile, bit-identical to the validated champion. A model
    # the bank never priced pays exactly one unit, the reference's default.
    tilt = np.zeros(profile.shape)
    if policy.pick_lam > 0.0:
        priced = np.where(np.isnan(effective_cost), policy.cost_scale, effective_cost)
        tilt = policy.pick_lam * priced / policy.cost_scale
    pick_index, pick = max(
        candidates, key=lambda item: (profile[item[0]] - tilt[item[0]], -pool_order[item[1]])
    )
    if pick == baseline:
        # Two different decisions land here, and the log has to tell them apart: the baseline
        # genuinely led on reward, or the cost knob demoted a leader that priced badly. The
        # second is the dominant case on the savings leg of the dial, so naming it (and the model
        # it outbid) is what makes that leg debuggable from the request log at all.
        leader_index, leader = max(
            candidates, key=lambda item: (profile[item[0]], -pool_order[item[1]])
        )
        if leader != baseline:
            gated = switch_gate(pick_index, baseline)
            if gated is not None:
                return gated
            return RoutingDecision(
                model=baseline,
                reason=(
                    f"knn cost knob (lam={policy.pick_lam:g}): {baseline} serves; {leader} led "
                    f"{rows.size} neighbors on evidence (profile {profile[leader_index]:.3f} vs "
                    f"{profile[pick_index]:.3f}) but not on price"
                ),
                # Greedy: the baseline IS the argmax once the cost knob has priced the
                # candidates, so nothing overrode the router's own preference.
                evidence=RoutingEvidence(
                    propensity="greedy", cache_credit_usd=credit_applied or None
                ),
            )
        gated = switch_gate(pick_index, baseline)
        if gated is not None:
            return gated
        return RoutingDecision(
            model=baseline,
            reason=f"knn: baseline {baseline} leads {rows.size} neighbors "
            f"(profile {profile[pick_index]:.3f})",
            evidence=RoutingEvidence(propensity="greedy", cache_credit_usd=credit_applied or None),
        )

    base_index = bank.models.index(baseline)
    paired = scored[:, pick_index] & scored[:, base_index]
    diffs = rewards[paired, pick_index] - rewards[paired, base_index]
    pairs = int(diffs.size)
    mean_diff = float(diffs.mean()) if pairs else 0.0
    error = float(diffs.std(ddof=1)) / pairs**0.5 if pairs > 1 else 0.0
    if policy.se_floor and 0 < pairs < SE_FLOOR_MAX_PAIRS:
        error = max(error, (0.25 / pairs) ** 0.5)
    pricier = bool(effective_cost[pick_index] > effective_cost[base_index])
    if policy.guard_mode == "asymmetric":
        z_effective = policy.knn_z if pricier else -policy.knn_z
    else:
        z_effective = 2 * policy.knn_z if pricier else policy.knn_z
    needed = z_effective * error

    if pairs < policy.knn_min_pairs or not mean_diff > needed:
        detail = (
            f"{pairs} paired neighbors, delta={mean_diff:+.3f}, needs "
            f"> {z_effective:g}xSE={needed:.3f}"
        )
        if pairs < policy.knn_min_pairs:
            detail = f"{pairs} paired neighbors < {policy.knn_min_pairs} required"
        base_pool_index = bank.models.index(baseline)
        gated = switch_gate(base_pool_index, baseline)
        if gated is not None:
            return gated
        return RoutingDecision(
            model=baseline,
            reason=f"knn guard: reverted to {baseline}, evidence insufficient ({detail})",
            evidence=RoutingEvidence(
                mean_diff=mean_diff,
                se=error,
                n_pairs=pairs,
                gate="reverted",
                propensity="fallback-forced",
                cache_credit_usd=credit_applied or None,
            ),
        )
    gated = switch_gate(pick_index, pick)
    if gated is not None:
        return gated

    knob = f", cost knob lam={policy.pick_lam:g}" if policy.pick_lam > 0.0 else ""
    # Only the symmetric bar doubles z for a pricier pick; the asymmetric one holds it at z.
    price_note = ""
    if pricier:
        price_note = " (pricier)" if policy.guard_mode == "asymmetric" else " (pricier, doubled z)"
    return RoutingDecision(
        model=pick,
        reason=f"knn: {rows.size} neighbors, delta={mean_diff:+.3f} > {z_effective:g}xSE"
        f"={needed:.3f}{price_note}{knob}",
        evidence=RoutingEvidence(
            mean_diff=mean_diff,
            se=error,
            n_pairs=pairs,
            gate="passed",
            propensity="greedy",
            cache_credit_usd=credit_applied or None,
        ),
    )


# --- `--embedder` resolution -------------------------------------------------------------
# Lives beside `EmbedderSpec` (what it produces) rather than in the CLI, because two commands
# now fit a policy: `wmo optimize route fit` and `wmo optimize model`. It raises ValueError so
# the library stays free of the CLI framework; `route_app` translates that to
# `typer.BadParameter` at the boundary.
# What `--embedder auto` looks for. The project's standard Azure OpenAI convention, the same pair
# `wmo.config` requires of an AZURE_OPENAI provider and that `.env.example` documents; auto does
# not invent a new variable, it notices the one an operator has already set up.
AZURE_EMBEDDER_ENV = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDER_DEPLOYMENT = "text-embedding-3-large"
AZURE_EMBEDDER_DIM = 3072  # the deployment's native width, and what the champion was measured at
HASHING_EMBEDDER_DIM = 512

# Quoted verbatim in the downgrade notice: the measured cost of routing on hashed features instead
# of semantic ones. Both numbers are accuracy points over the best single model on
# routerbench-ours9, so the gap is what auto is telling the operator they are leaving on the table.
HASHING_DOWNGRADE_NOTICE = (
    "hashing-512 measured +0.60pt (4/5 seeds) vs the semantic champion's +1.04pt (5/5) on the "
    "full-power benchmark"
)

# Native output widths of the embedding models WMO knows about, matched as a SUBSTRING of the
# deployment name because an Azure deployment is operator-named and usually carries the model
# family in it. `dim` is not bookkeeping: it becomes the `dimensions` parameter of the embeddings
# request, so asking a model for more than it has is an API error, and asking for less silently
# truncates the vectors the policy is then fitted and served on.
_NATIVE_EMBEDDING_DIMS: tuple[tuple[str, int], ...] = (
    ("text-embedding-3-large", 3072),
    ("text-embedding-3-small", 1536),
    ("text-embedding-ada-002", 1536),
)


def _native_dim(deployment: str) -> tuple[int, bool]:
    """(native width, whether the deployment name identified the model) for an azure deployment.

    An unrecognized name assumes `text-embedding-3-large`, which is the deployment `auto`
    provisions and the one the champion was measured on. The assumption is stated in the
    resolution line rather than buried, and `--dim` overrides it, so the cost of guessing wrong
    is one visible flag rather than a policy quietly fitted on truncated vectors.
    """
    for family, width in _NATIVE_EMBEDDING_DIMS:
        if family in deployment:
            return width, True
    return AZURE_EMBEDDER_DIM, False


def resolve_embedder(
    choice: str,
    *,
    dim: int | None,
    deployment: str | None,
    endpoint: str | None,
    api_key_env: str | None,
) -> tuple[EmbedderSpec, str]:
    """Turn `--embedder` into the spec to fit with, plus the one line explaining the choice.

    `auto` is the default because the two backends are not equivalent and the difference is not
    visible in the artifact: a policy fitted on hashed features routes on lexical overlap, and one
    fitted on `text-embedding-3-large` routes on meaning. An operator who has already configured
    an Azure resource should get the better one without having to know that, and one who has not
    should be told what they are getting instead. So the resolution is ALWAYS printed, and the
    downgrade quotes the measured gap rather than a vague warning.

    Resolving to azure makes the fit BILL an embedding API, which is spend nobody typed a flag
    for, so the resolution line says so. It also makes the fit depend on that resource actually
    hosting an embedding deployment, which the env variables do not promise; `probe_embedder`
    is what turns that from a mid-fit traceback into a usage error.

    `--dim` defaults to the RESOLVED backend's native width on every path: 512 for hashing, and
    the embedding model's own width for azure (see `_native_dim`). It used to default to 512
    everywhere, which meant `--embedder azure --deployment text-embedding-3-large` silently
    requested 512-dimensional vectors from a 3072-dimensional model and fitted the policy on the
    truncation. An explicit `--dim` is still honored verbatim, including a deliberate reduction.

    Returns:
        The `EmbedderSpec` to fit with, and the resolution line to print.

    Raises:
        ValueError: An unknown `--embedder`, or an explicit azure spec missing a flag.
            `route fit` re-raises these as `typer.BadParameter`.
    """
    if choice not in ("auto", "hashing", "openai", "azure"):
        raise ValueError(f"unknown embedder '{choice}'; use auto, hashing, openai or azure")

    if choice == "auto":
        present = all(os.environ.get(name) for name in AZURE_EMBEDDER_ENV)
        if not present:
            missing = [name for name in AZURE_EMBEDDER_ENV if not os.environ.get(name)]
            spec = EmbedderSpec(dim=HASHING_EMBEDDER_DIM if dim is None else dim)
            return spec, (
                f"embedder: hashing-{spec.dim} (auto; {', '.join(missing)} unset). "
                f"[yellow]{HASHING_DOWNGRADE_NOTICE}[/yellow]. Set "
                f"{' and '.join(AZURE_EMBEDDER_ENV)} to fit on semantic embeddings instead."
            )
        resolved_endpoint = endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
        resolved_deployment = deployment or AZURE_EMBEDDER_DEPLOYMENT
        return _azure_spec(
            deployment=resolved_deployment,
            endpoint=resolved_endpoint,
            api_key_env=api_key_env or AZURE_EMBEDDER_ENV[0],
            dim=dim,
            how=f"auto; {' and '.join(AZURE_EMBEDDER_ENV)} present",
        )

    if choice == "hashing":
        spec = EmbedderSpec(dim=HASHING_EMBEDDER_DIM if dim is None else dim)
        return spec, f"embedder: hashing-{spec.dim} (explicit). {HASHING_DOWNGRADE_NOTICE}"

    if choice == "openai":
        if not deployment:
            raise ValueError(
                "--embedder openai needs --deployment naming the embedding model "
                "(for example text-embedding-3-large)"
            )
        if endpoint:
            raise ValueError(
                "--embedder openai uses the direct OpenAI endpoint; drop --endpoint or use "
                "--embedder azure"
            )
        native, recognized = _native_dim(deployment)
        spec = EmbedderSpec(
            kind="openai",
            dim=native if dim is None else dim,
            deployment=deployment,
            api_key_env=api_key_env or "OPENAI_API_KEY",
        )
        width = (
            f"{spec.dim}d as asked"
            if dim is not None
            else f"{spec.dim}d native"
            if recognized
            else f"{spec.dim}d assumed"
        )
        return spec, (
            f"embedder: openai {deployment} ({width}) (explicit). "
            "This calls the OpenAI embedding API and is billed to that account."
        )

    if not (deployment and endpoint):
        # EmbedderSpec would reject this too, but only after the matrix has been read; say which
        # flag is missing, at the boundary, in the vocabulary the operator typed.
        raise ValueError(
            "--embedder azure needs --deployment and --endpoint (or use --embedder auto, which "
            f"reads {' and '.join(AZURE_EMBEDDER_ENV)})"
        )
    return _azure_spec(
        deployment=deployment,
        endpoint=endpoint,
        api_key_env=api_key_env,
        dim=dim,
        how="explicit",
    )


def _azure_spec(
    *, deployment: str, endpoint: str, api_key_env: str | None, dim: int | None, how: str
) -> tuple[EmbedderSpec, str]:
    """One azure spec plus its resolution line, shared by the auto and explicit paths.

    Shared so the two paths cannot drift on the thing that matters here, which is how the
    embedding width is chosen when the operator did not name one.
    """
    native, recognized = _native_dim(deployment)
    spec = EmbedderSpec(
        kind="azure",
        dim=native if dim is None else dim,
        deployment=deployment,
        endpoint=endpoint,
        api_key_env=api_key_env,
    )
    hint = ""
    if dim is not None:
        width = f"{spec.dim}d as asked"
    elif recognized:
        width = f"{spec.dim}d native"
    else:
        width = f"{spec.dim}d assumed"
        hint = (
            f". Unrecognized deployment name, so the width is a guess: pass --dim if "
            f"{deployment} is not {spec.dim}-dimensional"
        )
    # Stated because it is spend the operator did not type a flag for: `auto` reaching the azure
    # branch means this fit BILLS an embedding API, and the probe below bills the first call.
    return spec, (
        f"embedder: azure {deployment} ({width}) at {endpoint} ({how}){hint}. "
        "This calls the embedding API and is billed to that resource."
    )


def probe_embedder(spec: EmbedderSpec) -> None:
    """Embed one short text before the fit does, so a broken backend fails cleanly and early.

    `--embedder auto` turns the mere PRESENCE of `AZURE_OPENAI_*` into a network dependency, and
    those variables routinely point at a resource that serves chat but hosts no embedding
    deployment. Without this, the first thing an operator sees is a traceback wall from inside
    the fit, after the matrix has been read and (for a large corpus) after real spend. One
    throwaway embedding turns that into a usage error at the boundary, which is rule 9's bar:
    say what went wrong and what to do about it.

    Hashing specs are checked too. It costs nothing, and it keeps the failure shape identical on
    both branches rather than leaving one path with a different error surface.

    Raises:
        ValueError: The embedder could not produce a vector, naming what was tried and
            the escape hatch.
    """
    try:
        vectors = spec.build().embed(["routing embedder probe"])
    except Exception as exc:  # noqa: BLE001 - any backend failure here is a usage error
        raise ValueError(_probe_failure(spec, str(exc) or type(exc).__name__)) from exc
    if not vectors or not vectors[0]:
        raise ValueError(_probe_failure(spec, "it returned an empty vector"))


def _probe_failure(spec: EmbedderSpec, detail: str) -> str:
    """What to tell an operator whose embedder does not work, in the vocabulary they typed."""
    if spec.kind == "hashing":
        return f"the hashing embedder failed to embed a probe text: {detail}"
    if spec.kind == "openai":
        return (
            f"OpenAI embedding model '{spec.deployment}' could not embed a probe text: "
            f"{detail}. Check {spec.api_key_env or 'OPENAI_API_KEY'}, use a current embedding "
            "model, or fit with --embedder hashing."
        )
    return (
        f"embedding deployment '{spec.deployment}' at {spec.endpoint} could not embed a probe "
        f"text: {detail}. That resource may serve chat models without hosting an embedding "
        f"deployment, or '{spec.deployment}' may be named differently there. Deploy "
        f"{AZURE_EMBEDDER_DEPLOYMENT} on it, point --deployment/--endpoint at one that has it, "
        "or fit on offline features with --embedder hashing (which needs no credentials and no "
        "network, at the accuracy cost this command printed above)."
    )
