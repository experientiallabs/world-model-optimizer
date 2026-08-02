"""Tests for the routing policy artifact and the Avengers-style rank selection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError

from wmo.optimize.compression import (
    CompressionConfig,
    CompressionResult,
    Compressor,
    estimate_tokens,
    register_compressor,
)
from wmo.optimize.policy import (
    AZURE_EMBEDDER_DEPLOYMENT,
    AZURE_EMBEDDER_DIM,
    HASHING_DOWNGRADE_NOTICE,
    HASHING_EMBEDDER_DIM,
    KNN_BANK_FILENAME,
    ClusterRanking,
    EmbedderSpec,
    KnnBank,
    RoutingPolicy,
    cache_credit_usd,
    embedder_provenance,
    knn_decision,
    linear_decision,
    probe_embedder,
    rank_decision,
    resolve_embedder,
    select_model,
    write_artifact_atomically,
)
from wmo.providers.base import ProviderKind
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder
from wmo.tracking.pricing import ModelPrice


class _ChurnyCompressor:
    """A compressor that admits it rewrites its own emitted prefix (C1's percentile family).

    Registered by the requirement-B tests to prove the mount gate refuses it. It never runs: the
    policy is rejected before a request can reach `compress`.
    """

    id = "churny-for-tests"
    version = "1"
    append_stable = False

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        raw = sum(estimate_tokens(segment) for segment in segments)
        return CompressionResult(
            segments=list(segments), tokens_in_raw=raw, tokens_in_compressed=raw, latency_s=0.0
        )


_CHURNY = _ChurnyCompressor()


def _register_churny() -> str:
    """Put the churny stand-in in the registry and hand back its id.

    One shared instance: registration is idempotent for the same object but refuses to rebind an
    id to a different one, so a fresh instance per test would be the rebinding it forbids.
    """
    register_compressor(_CHURNY)
    return _CHURNY.id


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
    ]


def _static() -> RoutingPolicy:
    return RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())


def _rank_policy(top_k_clusters: int = 1) -> RoutingPolicy:
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed(["SELECT count(*) FROM superheroes", "write a friendly email"])
    return RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=top_k_clusters,
        clusters=[
            ClusterRanking(
                cluster_id=0,
                label="sql",
                centroid=sql,
                ranking=["fable-5", "haiku-4-5"],
                scores={"fable-5": 0.9, "haiku-4-5": 0.5},
                total=10,
            ),
            ClusterRanking(
                cluster_id=1,
                label="prose",
                centroid=prose,
                ranking=["haiku-4-5", "fable-5"],
                scores={"haiku-4-5": 0.8, "fable-5": 0.7},
                total=10,
            ),
        ],
    )


def _linear_policy() -> RoutingPolicy:
    embedder = EmbedderSpec(dim=8)
    query = np.asarray(embedder.build().embed(["route this coding task"])[0])
    return RoutingPolicy(
        kind="linear",
        default_model="fable-5",
        pool=_pool(),
        embedder=embedder,
        linear_weak_model="haiku-4-5",
        linear_strong_model="fable-5",
        linear_weak_weights=[0.0] * 8,
        linear_strong_weights=query.tolist(),
        linear_weak_bias=0.4,
        linear_strong_bias=0.4,
        linear_threshold=0.5,
    )


def test_static_policy_routes_to_default() -> None:
    decision = select_model(_static(), "anything at all")
    assert decision.model == "haiku-4-5"
    assert decision.cluster_id is None
    assert "static" in decision.reason


def test_rank_policy_routes_by_nearest_cluster_ranking() -> None:
    policy = _rank_policy()
    sql_decision = select_model(policy, "SELECT name FROM superheroes WHERE power = 'flight'")
    assert sql_decision.model == "fable-5"
    assert sql_decision.cluster_label == "sql"
    assert "rank router" in sql_decision.reason
    prose_decision = select_model(policy, "write a friendly email to the team")
    assert prose_decision.model == "haiku-4-5"
    assert prose_decision.cluster_label == "prose"


def test_rank_policy_soft_mixing_follows_the_closer_cluster() -> None:
    # With both clusters mixed in (top_k=2) and symmetric opposite rankings, the winner is
    # decided by which cluster is closer: score(m) = sum_c p_c / (rank_c(m) + 0.1).
    policy = _rank_policy(top_k_clusters=2)
    assert select_model(policy, "SELECT count(*) FROM superheroes").model == "fable-5"
    assert select_model(policy, "write a friendly email").model == "haiku-4-5"


def test_linear_policy_routes_on_predicted_uplift() -> None:
    policy = _linear_policy()
    strong = select_model(policy, "route this coding task")
    assert strong.model == "fable-5"
    assert "predicted uplift" in strong.reason

    weak = policy.model_copy(update={"linear_threshold": 0.7})
    assert select_model(weak, "route this coding task").model == "haiku-4-5"


def test_linear_decision_normalizes_query_and_falls_back_on_empty() -> None:
    policy = _linear_policy()
    query = np.asarray(policy.embedder.build().embed(["route this coding task"])[0])
    assert linear_decision(policy, query) == linear_decision(policy, query * 13.0)
    empty = linear_decision(policy, np.zeros(policy.embedder.dim))
    assert empty.model == policy.default_model
    assert "empty embedding" in empty.reason


def test_linear_policy_validates_models_weights_and_finite_values() -> None:
    valid = _linear_policy().model_dump()
    for update, match in (
        ({"linear_strong_model": None}, "weak and strong"),
        ({"linear_strong_model": "haiku-4-5"}, "distinct"),
        ({"linear_strong_model": "missing"}, "not in the policy pool"),
        ({"linear_weak_weights": [0.0]}, "embedder dim"),
        ({"linear_strong_weights": [float("nan")] * 8}, "finite"),
        ({"linear_threshold": float("inf")}, "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            RoutingPolicy.model_validate({**valid, **update})


def test_models_missing_from_rankings_score_default_rank() -> None:
    # Ranking mentions only fable-5; haiku scores 1/default_rank and cannot win.
    embedder = HashingEmbedder(dim=32)
    (centroid,) = embedder.embed(["anything"])
    policy = RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=32),
        top_k_clusters=1,
        clusters=[ClusterRanking(cluster_id=0, centroid=centroid, ranking=["fable-5"])],
    )
    assert select_model(policy, "anything").model == "fable-5"


def test_incumbent_sticks_by_default() -> None:
    decision = select_model(_rank_policy(), "SELECT 1", incumbent="haiku-4-5")
    assert decision.model == "haiku-4-5"  # affinity wins over the cluster preference
    assert "sticky" in decision.reason


def test_retired_incumbent_reroutes() -> None:
    decision = select_model(_rank_policy(), "SELECT 1", incumbent="gone-model")
    assert decision.model == "fable-5"


def test_default_model_must_be_in_pool() -> None:
    with pytest.raises(ValueError, match="default_model"):
        RoutingPolicy(kind="static", default_model="nope", pool=_pool())


def test_ranking_models_must_be_in_pool() -> None:
    with pytest.raises(ValueError, match="missing"):
        RoutingPolicy(
            kind="rank",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=4),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1, 0, 0, 0], ranking=["missing"])],
        )


def test_rank_kind_requires_clusters_and_matching_dims() -> None:
    with pytest.raises(ValueError, match="cluster"):
        RoutingPolicy(kind="rank", default_model="haiku-4-5", pool=_pool())
    with pytest.raises(ValueError, match="dim"):
        RoutingPolicy(
            kind="rank",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=8),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1.0, 0.0], ranking=["fable-5"])],
        )


def test_static_kind_rejects_clusters() -> None:
    with pytest.raises(ValueError, match="static"):
        RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=2),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1.0, 0.0], ranking=["fable-5"])],
        )


def test_policy_round_trips_through_json(tmp_path: Path) -> None:
    policy = _rank_policy()
    path = tmp_path / "policy.json"
    policy.save(path)
    assert RoutingPolicy.load(path) == policy


def test_linear_policy_round_trips_through_json(tmp_path: Path) -> None:
    policy = _linear_policy()
    path = tmp_path / "linear-policy.json"
    policy.save(path)
    loaded = RoutingPolicy.load(path)
    assert loaded == policy
    assert select_model(loaded, "route this coding task").model == "fable-5"


def test_openrouter_candidate_keeps_the_price_it_was_fitted_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OpenRouter candidate is priced once, at fit, and the policy carries that price.

    The pool snapshot on the artifact is the record: serving reloads it, so a live catalog
    fetch (or a vendor price change) can never silently re-price a policy already deployed.
    """
    catalog = tmp_path / "openrouter-prices.json"
    catalog.write_text(
        PriceCatalog(
            fetched_at=time.time(),
            source="test fixture",
            prices={"z-ai/glm-4.6": ModelPrice(input_per_mtok=0.4, output_per_mtok=1.75)},
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setenv(CATALOG_PATH_ENV, str(catalog))
    pool = [
        PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
        PoolEntry(name="or-glm", kind=ProviderKind.OPENROUTER, model="z-ai/glm-4.6"),
    ]
    policy = RoutingPolicy(kind="static", default_model="or-glm", pool=pool)
    path = tmp_path / "policy.json"
    policy.save(path)

    # The catalog doubles its price after the fit; the deployed artifact must not follow.
    catalog.write_text(
        PriceCatalog(
            fetched_at=time.time(),
            source="test fixture",
            prices={"z-ai/glm-4.6": ModelPrice(input_per_mtok=0.8, output_per_mtok=3.5)},
        ).model_dump_json(),
        encoding="utf-8",
    )
    served = RoutingPolicy.load(path)

    assert select_model(served, "anything").model == "or-glm"
    assert served.pool[1].price().input_per_mtok == 0.4
    assert served.pool[1].price().output_per_mtok == 1.75


def test_a_failed_policy_save_leaves_the_previous_artifact_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serving reads the artifact dir while the optimizer writes it: no torn policy.json.

    The failure is injected at the payload's fsync, which `wmo.core.files.write_bytes_atomic`
    reaches with the bytes on the staging file and the rename not yet done. That is the moment an
    in-place write would already have destroyed the served artifact.
    """
    path = tmp_path / "policy.json"
    _rank_policy().save(path)
    served = RoutingPolicy.load(path)

    def _die(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", _die)
    with pytest.raises(OSError, match="disk full"):
        _static().save(path)
    assert RoutingPolicy.load(path) == served  # the staged write never replaced it
    assert [entry.name for entry in tmp_path.iterdir()] == ["policy.json"]  # no staging litter


def test_interleaved_artifact_writes_do_not_share_a_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers mid-flight on one path must not stage through the same file.

    Regression: the staging name was a fixed `<name>.partial`, so a second writer overwrote the
    first's staged bytes before the first renamed. One save then published the OTHER's payload
    under its own success message, and the loser failed on a file already renamed away.

    The second write is driven to completion inside the first one's `replace`, which is the
    interleaving that breaks a shared staging path, without depending on thread timing.
    """
    path = tmp_path / "policy.json"
    staged: list[Path] = []
    real_replace = Path.replace

    def _replace(self: Path, target: Path) -> Path:
        staged.append(self)
        if len(staged) == 1:  # the first writer is mid-flight: run the second start to finish
            write_artifact_atomically(path, b'{"second": true}')
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _replace)
    write_artifact_atomically(path, b'{"first": true}')

    assert len({entry.name for entry in staged}) == 2  # two writers, two staging files
    # The first writer renamed last, so its own bytes are what landed -- not the second's.
    assert path.read_bytes() == b'{"first": true}'
    assert [entry.name for entry in tmp_path.iterdir()] == ["policy.json"]


def test_interleaved_bank_writes_do_not_share_a_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fits racing on one `--out` derive one bank path; they must not stage through one file.

    Regression: `KnnBank.save` staged through a fixed `<bank>.partial`, so one fit could publish
    the OTHER fit's evidence under its own bank name while the loser raised `FileNotFoundError`.
    A policy serving a competing fit's rewards is the failure this whole change is about, and
    the derived bank name does not prevent it when both fits target the same policy path.
    """
    path = tmp_path / "bank.npz"
    mine = _knn_bank([[1.0, 0.0]] * 12)  # fable-5 wins every neighbor
    theirs = _knn_bank([[0.0, 1.0]] * 12)  # ...and haiku-4-5 wins every neighbor
    staged: list[Path] = []
    real_replace = Path.replace

    def _replace(self: Path, target: Path) -> Path:
        staged.append(self)
        if len(staged) == 1:  # this fit is mid-flight: run the competing one start to finish
            theirs.save(path)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _replace)
    mine.save(path)

    assert len({entry.name for entry in staged}) == 2  # two fits, two staging files
    # The bank that renamed last is the one on disk, with ITS rewards, not the competitor's.
    assert np.array_equal(KnnBank.load(path).rewards, mine.rewards)
    assert [entry.name for entry in tmp_path.iterdir()] == ["bank.npz"]


def test_pre_compression_policy_json_loads_with_compression_off(tmp_path: Path) -> None:
    # The #259 additive-fields guarantee, pinned for D-COMPRESS: a policy.json written before
    # the compression fields existed must load with compression defaulted to off everywhere.
    policy = _rank_policy()
    path = tmp_path / "policy.json"
    raw = policy.model_dump_json(indent=2)
    assert '"compression"' in raw  # sanity: the field serializes
    dropped = ("compression", "fit_compression")
    stripped = {
        key: value for key, value in policy.model_dump(mode="json").items() if key not in dropped
    }
    stripped["clusters"] = [
        {k: v for k, v in cluster.items() if k not in dropped} for cluster in stripped["clusters"]
    ]
    path.write_text(json.dumps(stripped), encoding="utf-8")
    loaded = RoutingPolicy.load(path)
    assert loaded.compression is None
    assert loaded.fit_compression is None  # an old artifact reads as fitted on raw text
    assert all(cluster.compression is None for cluster in loaded.clusters)


def test_policy_with_compression_round_trips(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    # Both halves: what the endpoint serves, and what its clusters were fitted under. A rank
    # policy carrying only the first is the mismatch the next test rejects.
    policy = _rank_policy().model_copy(update={"compression": config, "fit_compression": config})
    path = tmp_path / "policy.json"
    policy.save(path)
    loaded = RoutingPolicy.load(path)
    assert loaded.compression is not None
    assert loaded.compression.compressor_id == "truncate"
    assert loaded.compression.aggressiveness == 0.5
    assert loaded.fit_compression == loaded.compression


def test_policy_rejects_unknown_compressor_at_load() -> None:
    # Fail at mount, not on the first request mid-conversation.
    with pytest.raises(ValidationError, match="unknown compressor 'nope'"):
        RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            compression=CompressionConfig(compressor_id="nope"),
        )


# --- D-COMPRESS requirement A: representation consistency (C2 Q2) ---


def test_routing_policy_refuses_to_serve_compression_it_was_not_fitted_under(
    tmp_path: Path,
) -> None:
    # C2 Q2: compressed queries against a raw-fit bank sit farther from every row, so the
    # novelty floor trips 10-13x more often and the router abstains to the expensive fallback
    # (+11-41% cost, accuracy flat to negative). A broken policy, so it must not mount.
    policy = _rank_policy().model_copy(
        update={"compression": CompressionConfig(compressor_id="truncate")}
    )
    path = tmp_path / "policy.json"
    policy.save(path)
    with pytest.raises(ValidationError, match="fitted on raw text"):
        RoutingPolicy.load(path)
    # And the mount path refuses the same object even though model_copy skipped the validator.
    with pytest.raises(ValueError, match="fitted on raw text"):
        policy.serving_compressor()


def test_a_compressed_fit_refuses_to_serve_raw_queries(tmp_path: Path) -> None:
    # The inverse hole, and just as broken: a bank whose rows are compressed text cannot be
    # queried with raw text either.
    policy = _rank_policy().model_copy(
        update={"fit_compression": CompressionConfig(compressor_id="truncate")}
    )
    path = tmp_path / "policy.json"
    policy.save(path)
    with pytest.raises(ValidationError, match="would serve raw text"):
        RoutingPolicy.load(path)


def test_a_mismatched_aggressiveness_is_a_mismatched_representation(tmp_path: Path) -> None:
    # Same compressor, different level = different bytes = different geometry.
    policy = _rank_policy().model_copy(
        update={
            "compression": CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
            "fit_compression": CompressionConfig(compressor_id="truncate", aggressiveness=0.25),
        }
    )
    path = tmp_path / "policy.json"
    policy.save(path)
    with pytest.raises(ValidationError, match="aggressiveness 0.25"):
        RoutingPolicy.load(path)


def test_a_static_policy_may_compress_without_a_fit_stamp() -> None:
    # A static policy embeds nothing, so it has no geometry to be inconsistent with. This is the
    # path the demo endpoint and the seam walk use.
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
    )
    assert policy.fit_compression is None
    assert policy.serving_compressor() is not None


# --- D-COMPRESS requirement B: append-stability attestation (C2 Q4) ---


def test_a_churny_compressor_is_refused_at_mount() -> None:
    # C2 Q4 measured churny full recompression at up to 2.65x the input cost of compressing
    # NOTHING on cached providers. v1 has no turn-local-commit path, so it does not mount.
    churny = _register_churny()
    with pytest.raises(ValidationError, match="not attested append-stable"):
        RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            compression=CompressionConfig(compressor_id=churny),
        )


def test_a_churny_per_cluster_override_is_refused_too(tmp_path: Path) -> None:
    churny = _register_churny()
    policy = _rank_policy()
    clusters = [
        policy.clusters[0].model_copy(
            update={"compression": CompressionConfig(compressor_id=churny)}
        ),
        *policy.clusters[1:],
    ]
    path = tmp_path / "policy.json"
    policy.model_copy(update={"clusters": clusters}).save(path)
    with pytest.raises(ValidationError, match="not attested append-stable"):
        RoutingPolicy.load(path)


def test_an_unattested_compressor_cannot_be_registered() -> None:
    class _Forgetful:
        id = "forgetful"
        version = "1"

        def compress(
            self, segments: list[str], config: CompressionConfig
        ) -> CompressionResult:  # pragma: no cover - never reached
            raise NotImplementedError

    with pytest.raises(ValueError, match="does not declare `append_stable`"):
        register_compressor(cast("Compressor", _Forgetful()))


def test_azure_embedder_spec_requires_backend_fields() -> None:
    with pytest.raises(ValueError, match="deployment"):
        EmbedderSpec(kind="azure", dim=3072)


def test_azure_embedder_spec_builds_a_batched_provider_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZ_EMBED_KEY", "sk-test")
    spec = EmbedderSpec(
        kind="azure",
        dim=3072,
        deployment="text-embedding-3-large",
        endpoint="https://example.openai.azure.com",
        api_key_env="AZ_EMBED_KEY",
        batch=128,
    )
    embedder = spec.build()  # constructs lazily; no network until embed()
    from wmo.retrieval.embedders import BatchedEmbedder

    assert isinstance(embedder, BatchedEmbedder)


def test_azure_embedder_spec_missing_key_env_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZ_EMBED_KEY", raising=False)
    spec = EmbedderSpec(
        kind="azure",
        dim=8,
        deployment="d",
        endpoint="https://example.openai.azure.com",
        api_key_env="AZ_EMBED_KEY",
    )
    with pytest.raises(ValueError, match="AZ_EMBED_KEY"):
        spec.build()


def test_openai_embedder_spec_builds_a_batched_provider_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_EMBED_KEY", "sk-test")
    spec = EmbedderSpec(
        kind="openai",
        dim=3072,
        deployment="text-embedding-3-large",
        api_key_env="OPENAI_EMBED_KEY",
        batch=128,
    )
    from wmo.retrieval.embedders import BatchedEmbedder

    assert isinstance(spec.build(), BatchedEmbedder)


def test_support_tilt_shifts_weight_to_supported_clusters() -> None:
    # Two near-equidistant clusters with opposite rankings; the tiny cluster (total=1) wins
    # untilted (slightly closer), the big one (total=400) wins under tilt.
    embedder = HashingEmbedder(dim=64)
    near, far = embedder.embed(["SELECT count(*) FROM t", "SELECT sum(x) FROM t"])

    def build(gamma: float) -> RoutingPolicy:
        return RoutingPolicy(
            kind="rank",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=64),
            top_k_clusters=2,
            beta=1.0,
            support_tilt_gamma=gamma,
            clusters=[
                ClusterRanking(
                    cluster_id=0, centroid=near, ranking=["fable-5", "haiku-4-5"], total=1
                ),
                ClusterRanking(
                    cluster_id=1, centroid=far, ranking=["haiku-4-5", "fable-5"], total=400
                ),
            ],
        )

    query = "SELECT count(*) FROM t"
    assert select_model(build(0.0), query).model == "fable-5"
    assert select_model(build(1.0), query).model == "haiku-4-5"


def test_select_model_uses_a_caller_supplied_embedder() -> None:
    # The seam a many-request caller needs: build the policy's embedder once and hand it in.
    # Proven by handing in one that maps SQL text onto the prose centroid.
    class _ProseEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            (prose,) = HashingEmbedder(dim=64).embed(["write a friendly email"])
            return [prose for _ in texts]

    policy = _rank_policy()
    assert select_model(policy, "SELECT 1").model == "fable-5"
    assert select_model(policy, "SELECT 1", embedder=_ProseEmbedder()).model == "haiku-4-5"


def test_support_tilt_gamma_rejects_negative_values() -> None:
    # A negative tilt would reweight AWAY from supported clusters, silently inverting the lever.
    with pytest.raises(ValidationError):
        RoutingPolicy(
            kind="static", default_model="haiku-4-5", pool=_pool(), support_tilt_gamma=-1.0
        )


def test_rank_decision_normalizes_the_query_itself() -> None:
    # `beta * distance` is a softmax temperature, so an unnormalized query silently changes how
    # sharply the top clusters mix (with the support tilt on, that flips the winner). The
    # normalization is enforced inside rank_decision, so no caller can skip it.
    embedder = HashingEmbedder(dim=64)
    near, far = embedder.embed(["SELECT count(*) FROM t", "SELECT sum(x) FROM t"])
    policy = RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=2,
        beta=1.0,
        support_tilt_gamma=1.0,
        clusters=[
            ClusterRanking(cluster_id=0, centroid=near, ranking=["fable-5", "haiku-4-5"], total=1),
            ClusterRanking(cluster_id=1, centroid=far, ranking=["haiku-4-5", "fable-5"], total=400),
        ],
    )
    unit = np.asarray(embedder.embed(["SELECT count(*) FROM t"])[0])
    expected = rank_decision(policy, unit)
    assert expected.model == "haiku-4-5"
    for scale in (0.02, 50.0):  # 50.0 selected fable-5 before the fix
        assert rank_decision(policy, unit * scale) == expected


# --- kNN policies: the guarded nearest-neighbor champion (see wmo.optimize.knn) ------------

_CHEAP, _PRICEY = 0.001, 0.010


def _knn_bank(
    rewards: list[list[float]],
    *,
    sims: list[float] | None = None,
    costs: list[float] | None = None,
) -> KnnBank:
    """A 2-dimensional bank: one row per neighbor, columns [fable-5, haiku-4-5].

    Rows sit on the unit circle at the requested cosine similarity to the query (1, 0), which
    makes neighbor selection and the similarity weights exact rather than embedder-dependent.
    Per-model costs default to fable-5 pricey / haiku-4-5 cheap.
    """
    similarities = sims if sims is not None else [1.0] * len(rewards)
    reward_rows = np.asarray(rewards, dtype=np.float32)
    model_costs = costs if costs is not None else [_PRICEY, _CHEAP]
    return KnnBank(
        embeddings=np.asarray(
            [[sim, (1.0 - sim**2) ** 0.5] for sim in similarities], dtype=np.float32
        ),
        rewards=reward_rows,
        costs=np.where(np.isnan(reward_rows), np.nan, np.asarray(model_costs, dtype=np.float32)),
        models=["fable-5", "haiku-4-5"],
        scenario_ids=[f"s{index}" for index in range(len(rewards))],
    )


def _knn_policy(
    bank: KnnBank,
    *,
    embedder: EmbedderSpec | None = None,
    rag_num: int = 50,
    knn_z: float = 0.5,
    knn_min_pairs: int = 8,
    se_floor: bool = True,
    pick_lam: float = 0.0,
    guard_mode: Literal["symmetric", "asymmetric"] = "symmetric",
) -> RoutingPolicy:
    """A knn policy over `bank` with fable-5 as the pinned baseline (the production contract)."""
    policy = RoutingPolicy(
        kind="knn",
        default_model="fable-5",
        guard_model="fable-5",
        pool=_pool(),
        embedder=embedder or EmbedderSpec(dim=2),
        rag_num=rag_num,
        knn_z=knn_z,
        knn_min_pairs=knn_min_pairs,
        se_floor=se_floor,
        pick_lam=pick_lam,
        guard_mode=guard_mode,
        # The bank's two models cost _PRICEY and _CHEAP, so their mean is the unit the cost
        # knob divides by (what `wmo.optimize.knn.bank_cost_scale` computes at fit time).
        cost_scale=(_PRICEY + _CHEAP) / 2,
    )
    policy.attach_bank(bank)
    return policy


_QUERY = np.asarray([1.0, 0.0])


class _UnitEmbedder:
    """Embeds every text to the bank's query direction, so routing is about the bank alone."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_knn_routes_away_from_the_baseline_on_strong_evidence() -> None:
    # 12 neighbors where haiku always wins and fable always loses: the evidence is as clean as
    # it gets, and haiku is the cheaper model, so the guard applies the single-z bar.
    policy = _knn_policy(_knn_bank([[0.0, 1.0]] * 12))
    decision = knn_decision(policy, _QUERY)
    assert decision.model == "haiku-4-5"
    assert "12 neighbors" in decision.reason
    assert "delta=+1.000" in decision.reason


def test_knn_reverts_when_too_few_neighbors_were_scored_on_both_sides() -> None:
    # Same unanimous evidence, but only 5 paired neighbors: below min_pairs the guard refuses,
    # because five agreeing neighbors are how a router talks itself into a confident mistake.
    policy = _knn_policy(_knn_bank([[0.0, 1.0]] * 5))
    decision = knn_decision(policy, _QUERY)
    assert decision.model == "fable-5"
    assert "5 paired neighbors < 8 required" in decision.reason


def test_knn_z_is_the_confidence_knob_on_identical_evidence() -> None:
    # Noisy evidence: haiku wins 7 of 12 neighbors, loses 5, mean delta +0.083 with a standard
    # error of 0.149. At z=0.5 the mean clears the bar; at z=1.0 the same evidence does not.
    rewards = [[0.5, 1.0]] * 7 + [[0.5, 0.0]] * 5
    assert knn_decision(_knn_policy(_knn_bank(rewards), knn_z=0.5), _QUERY).model == "haiku-4-5"
    stricter = knn_decision(_knn_policy(_knn_bank(rewards), knn_z=1.0), _QUERY)
    assert stricter.model == "fable-5"
    assert "evidence insufficient" in stricter.reason


def test_knn_doubles_the_bar_for_a_pricier_pick() -> None:
    # The economic asymmetry: identical reward evidence, and the only difference is whether the
    # pick costs more than the baseline. Paying more requires twice the confidence.
    rewards = [[0.5, 1.0]] * 7 + [[0.5, 0.0]] * 5
    cheap_pick = knn_decision(_knn_policy(_knn_bank(rewards, costs=[_PRICEY, _CHEAP])), _QUERY)
    pricey_pick = knn_decision(_knn_policy(_knn_bank(rewards, costs=[_CHEAP, _PRICEY])), _QUERY)
    assert cheap_pick.model == "haiku-4-5"
    assert pricey_pick.model == "fable-5"
    assert "pricier" not in cheap_pick.reason
    assert "1xSE" in pricey_pick.reason  # z doubled from 0.5


def test_knn_se_floor_stops_a_zero_variance_neighborhood_from_looking_significant() -> None:
    # Nine neighbors that all agree haiku is 0.05 better: the empirical standard error is ZERO,
    # so an unfloored guard treats a rounding-sized edge as certain. The floor (sqrt(0.25/9))
    # asks for 0.083 instead, and the edge does not clear it.
    bank = _knn_bank([[0.5, 0.55]] * 9)
    assert knn_decision(_knn_policy(bank, se_floor=False), _QUERY).model == "haiku-4-5"
    assert knn_decision(_knn_policy(bank, se_floor=True), _QUERY).model == "fable-5"


def test_knn_keeps_the_baseline_when_it_leads_the_neighborhood() -> None:
    decision = knn_decision(_knn_policy(_knn_bank([[1.0, 0.0]] * 12)), _QUERY)
    assert decision.model == "fable-5"
    assert "leads 12 neighbors" in decision.reason


def test_cost_knob_prices_candidates_before_the_argmax() -> None:
    # Twelve neighbors where the pricey baseline is 0.05 better on every one: on reward alone
    # fable-5 leads and serves. The knob charges each candidate pick_lam * cost / cost_scale,
    # and haiku is 10x cheaper, so a big enough lam makes it the raw pick instead - and the
    # economic bar (-0.5 x SE = -0.072) tolerates a 0.05 deficit, so the pick survives.
    bank = _knn_bank([[0.55, 0.5]] * 12)
    assert knn_decision(_knn_policy(bank, guard_mode="asymmetric"), _QUERY).model == "fable-5"
    tilted = knn_decision(_knn_policy(bank, pick_lam=0.2, guard_mode="asymmetric"), _QUERY)
    assert tilted.model == "haiku-4-5"
    assert "cost knob lam=0.2" in tilted.reason
    assert "-0.5xSE" in tilted.reason


def test_symmetric_guard_sends_the_cost_knobs_pick_back_to_the_baseline() -> None:
    # The same tilt under the shipped strict bar: haiku is the raw pick and is then reverted,
    # because there a cheaper pick must still be positively supported. This is the measured
    # reason the dial's price leg moves the guard too - the knob alone spends MORE, not less,
    # since every tilted pick lands back on the pricier baseline.
    guarded = knn_decision(_knn_policy(_knn_bank([[0.55, 0.5]] * 12), pick_lam=0.2), _QUERY)
    assert guarded.model == "fable-5"
    assert "evidence insufficient" in guarded.reason
    assert "delta=-0.050" in guarded.reason


def test_cost_knob_cannot_promote_a_pick_the_evidence_rejects() -> None:
    # Evidence that says haiku is clearly worse (delta -0.1 against a floored SE of 0.144, so
    # below even the -0.072 economic bar). No amount of cost pressure buys that pick: the guard
    # runs after the tilt, on the untilted paired evidence.
    bank = _knn_bank([[0.6, 0.5]] * 12)
    for mode in ("symmetric", "asymmetric"):
        decision = knn_decision(_knn_policy(bank, pick_lam=0.5, guard_mode=mode), _QUERY)
        assert decision.model == "fable-5"
        assert "evidence insufficient" in decision.reason


def test_asymmetric_guard_still_holds_a_pricier_pick_to_a_positive_bar() -> None:
    # Cheap baseline, pricey challenger: the economic bar is lenient on the CHEAPER side only,
    # so a pricier pick with thin evidence is reverted exactly as before.
    thin = _knn_bank([[0.5, 0.55]] * 12, costs=[_CHEAP, _PRICEY])
    decision = knn_decision(_knn_policy(thin, guard_mode="asymmetric"), _QUERY)
    assert decision.model == "fable-5"
    assert "evidence insufficient" in decision.reason


def test_cost_knob_needs_a_cost_unit_to_divide_by() -> None:
    with pytest.raises(ValidationError, match="cost_scale"):
        RoutingPolicy(
            kind="knn",
            default_model="fable-5",
            guard_model="fable-5",
            pool=_pool(),
            pick_lam=0.05,
        )


def test_knn_serves_the_baseline_when_neighbors_carry_no_scored_reward() -> None:
    bank = _knn_bank([[0.4, float("nan")]] * 3 + [[float("nan"), float("nan")]] * 9)
    # The query's neighbors are the 9 unscored rows only (the scored ones sit far away).
    bank = KnnBank(
        embeddings=np.asarray(
            [[0.2, 0.98] for _ in range(3)] + [[1.0, 0.0] for _ in range(9)], dtype=np.float32
        ),
        rewards=bank.rewards,
        costs=bank.costs,
        models=bank.models,
        scenario_ids=bank.scenario_ids,
    )
    decision = knn_decision(_knn_policy(bank, rag_num=9), _QUERY)
    assert decision.model == "fable-5"
    assert "no scored reward" in decision.reason


def test_knn_neighbor_rule_is_relative_not_a_fixed_k() -> None:
    # rag_num is a budget, not a count: everything within rag_thres of the budget-th best
    # similarity joins, so a query in a dense region routes on MORE evidence than k. Here 12
    # rows are equally near, and a budget of 3 keeps all 12.
    decision = knn_decision(_knn_policy(_knn_bank([[0.0, 1.0]] * 12), rag_num=3), _QUERY)
    assert "12 neighbors" in decision.reason
    # Far rows stay out while the budget is covered by near ones, and join once it is not: 10
    # near rows plus 6 distant ones give 10 neighbors at a budget of 10, all 16 at a budget of 16.
    mixed = _knn_bank([[0.0, 1.0]] * 10 + [[1.0, 0.0]] * 6, sims=[1.0] * 10 + [0.5] * 6)
    assert "10 neighbors" in knn_decision(_knn_policy(mixed, rag_num=10), _QUERY).reason
    assert "16 neighbors" in knn_decision(_knn_policy(mixed, rag_num=16), _QUERY).reason


def test_knn_bank_loads_lazily_from_the_sidecar_and_stays_cached(tmp_path: Path) -> None:
    bank = _knn_bank([[0.0, 1.0]] * 12)
    bank.save(tmp_path / KNN_BANK_FILENAME)
    saved = _knn_policy(bank)
    saved.save(tmp_path / "policy.json")

    loaded = RoutingPolicy.load(tmp_path / "policy.json")
    assert loaded.bank_path() == tmp_path / KNN_BANK_FILENAME
    assert select_model(loaded, "anything", embedder=_UnitEmbedder()).model == "haiku-4-5"
    # Cached after the first decision: the sidecar is read once per policy instance, not per
    # request (a 3072-dimensional bank is megabytes).
    (tmp_path / KNN_BANK_FILENAME).unlink()
    assert select_model(loaded, "anything", embedder=_UnitEmbedder()).model == "haiku-4-5"


def test_knn_bank_path_falls_back_to_the_legacy_sidecar_name(tmp_path: Path) -> None:
    """A policy artifact that records no bank path still resolves the conventional sidecar.

    Backward compatibility: policies fitted before the bank name was derived from the policy
    file live beside `policy_knn_bank.npz`, and hand-built artifacts may omit the field.
    """
    bank = _knn_bank([[0.0, 1.0]] * 12)
    bank.save(tmp_path / KNN_BANK_FILENAME)
    document = _knn_policy(bank).model_dump(mode="json")
    document.pop("knn_bank_path")
    (tmp_path / "policy.json").write_text(json.dumps(document), encoding="utf-8")

    loaded = RoutingPolicy.load(tmp_path / "policy.json")
    assert loaded.bank_path() == tmp_path / KNN_BANK_FILENAME
    assert select_model(loaded, "anything", embedder=_UnitEmbedder()).model == "haiku-4-5"


def test_knn_policy_without_its_sidecar_says_where_the_file_should_be(tmp_path: Path) -> None:
    _knn_policy(_knn_bank([[0.0, 1.0]] * 12)).save(tmp_path / "policy.json")
    loaded = RoutingPolicy.load(tmp_path / "policy.json")
    with pytest.raises(FileNotFoundError, match=KNN_BANK_FILENAME):
        loaded.knn_bank()


def test_knn_kind_requires_a_baseline_that_is_also_the_default() -> None:
    with pytest.raises(ValueError, match="guard_model"):
        RoutingPolicy(kind="knn", default_model="fable-5", pool=_pool())
    with pytest.raises(ValueError, match="one model"):
        RoutingPolicy(kind="knn", default_model="fable-5", guard_model="haiku-4-5", pool=_pool())


def test_knn_kind_rejects_clusters() -> None:
    with pytest.raises(ValueError, match="knn policy carries no clusters"):
        RoutingPolicy(
            kind="knn",
            default_model="fable-5",
            guard_model="fable-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=2),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1.0, 0.0], ranking=["fable-5"])],
        )


def test_knn_bank_must_match_the_policys_embedder_and_pool() -> None:
    bank = _knn_bank([[0.0, 1.0]] * 12)
    with pytest.raises(ValueError, match="dimensional"):
        _knn_policy(bank, embedder=EmbedderSpec(dim=64))
    stranger = KnnBank(
        embeddings=bank.embeddings,
        rewards=bank.rewards,
        costs=bank.costs,
        models=["fable-5", "retired-model"],
        scenario_ids=bank.scenario_ids,
    )
    with pytest.raises(ValueError, match="not in the policy pool"):
        _knn_policy(stranger)


def test_knn_bank_must_carry_evidence_for_the_baseline() -> None:
    # A baseline the fit never measured cannot be compared against, so every request would
    # revert to it unmeasured: that is a static policy wearing a router's clothes.
    with pytest.raises(ValueError, match="no scored reward for baseline"):
        _knn_policy(_knn_bank([[float("nan"), 1.0]] * 12))


def test_knn_bank_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="rewards has shape"):
        KnnBank(
            embeddings=np.zeros((3, 2), dtype=np.float32),
            rewards=np.zeros((3, 5), dtype=np.float32),
            costs=np.zeros((3, 2), dtype=np.float32),
            models=["fable-5", "haiku-4-5"],
            scenario_ids=["a", "b", "c"],
        )


def test_knn_bank_round_trips_through_the_sidecar(tmp_path: Path) -> None:
    bank = _knn_bank([[0.5, 1.0], [float("nan"), 0.0]] * 6)
    bank.save(tmp_path / KNN_BANK_FILENAME)
    reloaded = KnnBank.load(tmp_path / KNN_BANK_FILENAME)
    assert reloaded.models == bank.models
    assert reloaded.scenario_ids == bank.scenario_ids
    np.testing.assert_array_equal(reloaded.rewards, bank.rewards)  # NaN cells included
    np.testing.assert_allclose(reloaded.embeddings, bank.embeddings)


def test_the_asymmetric_bar_does_not_claim_to_have_doubled_z() -> None:
    # Cheap baseline, pricey challenger with evidence that clears the bar. The symmetric bar
    # doubles z for the pricier pick and says so; the asymmetric one holds it at z, so claiming
    # a doubling would misreport which bar the request was actually held to.
    strong = _knn_bank([[0.0, 1.0]] * 12, costs=[_CHEAP, _PRICEY])
    symmetric = knn_decision(_knn_policy(strong), _QUERY)
    asymmetric = knn_decision(_knn_policy(strong, guard_mode="asymmetric"), _QUERY)
    assert symmetric.model == asymmetric.model == "haiku-4-5"
    assert "(pricier, doubled z)" in symmetric.reason
    assert "1xSE" in symmetric.reason  # z doubled from 0.5
    assert "(pricier)" in asymmetric.reason
    assert "doubled" not in asymmetric.reason
    assert "0.5xSE" in asymmetric.reason


def test_a_cost_demoted_leader_is_named_in_the_reason() -> None:
    # The savings leg's dominant decision: haiku led the neighborhood on reward, the cost knob
    # priced it out, and the baseline serves. The log must not call that "baseline leads".
    bank = _knn_bank([[0.5, 0.55]] * 12, costs=[_CHEAP, _PRICEY])
    decision = knn_decision(_knn_policy(bank, pick_lam=0.5, guard_mode="asymmetric"), _QUERY)
    assert decision.model == "fable-5"
    assert "cost knob (lam=0.5)" in decision.reason
    assert "haiku-4-5 led" in decision.reason
    assert "not on price" in decision.reason
    assert "leads 12 neighbors" not in decision.reason


def test_a_genuine_baseline_win_still_says_the_baseline_led() -> None:
    # The other way into the same branch: no cost pressure, the baseline simply scored best.
    decision = knn_decision(_knn_policy(_knn_bank([[1.0, 0.0]] * 12)), _QUERY)
    assert decision.model == "fable-5"
    assert "leads 12 neighbors" in decision.reason
    assert "cost knob" not in decision.reason


def test_embedder_provenance_separates_two_azure_resources() -> None:
    """A deployment name is not an embedder identity; the resource behind it is.

    Two Azure accounts routinely hold a deployment of the same name and dimension, and their
    embeddings are not interchangeable. If both fits render the same `fitted_from`, `tune`
    accepts the pre-refit snapshot and dials the superseded fit over the new one.
    """

    def azure(endpoint: str, api_key_env: str | None = None) -> EmbedderSpec:
        return EmbedderSpec(
            kind="azure",
            dim=3072,
            deployment="text-embedding-3-large",  # the SAME deployment name on both resources
            endpoint=endpoint,
            api_key_env=api_key_env,
        )

    east = embedder_provenance(azure("https://east.openai.azure.com"))
    assert east != embedder_provenance(azure("https://west.openai.azure.com"))
    # ...and the axes that do not move a vector stay out of it: renaming the credential variable
    # must not read as a refit.
    assert embedder_provenance(azure("https://east.openai.azure.com", "OTHER_KEY")) == east
    assert embedder_provenance(EmbedderSpec(dim=512)) == "hashing-512"


def test_openai_embedder_provenance_and_resolution_are_provider_specific() -> None:
    spec, line = resolve_embedder(
        "openai",
        dim=None,
        deployment="text-embedding-3-large",
        endpoint=None,
        api_key_env=None,
    )
    assert spec == EmbedderSpec(
        kind="openai",
        dim=3072,
        deployment="text-embedding-3-large",
        api_key_env="OPENAI_API_KEY",
    )
    assert embedder_provenance(spec) == "openai-3072/text-embedding-3-large"
    assert "billed to that account" in line


def test_openai_embedder_rejects_an_azure_endpoint() -> None:
    with pytest.raises(ValueError, match="drop --endpoint"):
        resolve_embedder(
            "openai",
            dim=None,
            deployment="text-embedding-3-large",
            endpoint="https://example.openai.azure.com",
            api_key_env=None,
        )


def test_evidence_records_the_guards_numbers_when_a_pick_is_routed() -> None:
    # The same decision the reason string describes in prose, in a shape something can aggregate.
    decision = knn_decision(_knn_policy(_knn_bank([[0.0, 1.0]] * 12)), _QUERY)
    assert decision.evidence is not None
    assert decision.evidence.gate == "passed"
    assert decision.evidence.propensity == "greedy"
    assert decision.evidence.n_pairs == 12
    assert decision.evidence.mean_diff == pytest.approx(1.0)
    assert decision.evidence.se is not None


def test_evidence_records_a_guard_revert_as_fallback_forced() -> None:
    decision = knn_decision(_knn_policy(_knn_bank([[0.0, 1.0]] * 5)), _QUERY)
    assert decision.evidence is not None
    assert decision.evidence.gate == "reverted"
    assert decision.evidence.propensity == "fallback-forced"
    # The numbers behind the refusal are kept, which is what makes a revert diagnosable in bulk.
    assert decision.evidence.n_pairs == 5


def test_evidence_calls_a_baseline_that_won_on_merit_greedy_with_no_gate() -> None:
    # Nothing overrode the router here: the baseline WAS its preference, so counting this as a
    # forced fallback would make a healthy endpoint look like a coverage problem.
    decision = knn_decision(_knn_policy(_knn_bank([[1.0, 0.0]] * 12)), _QUERY)
    assert decision.evidence is not None
    assert decision.evidence.gate is None
    assert decision.evidence.propensity == "greedy"
    assert decision.evidence.n_pairs is None


def test_evidence_records_a_novelty_abstain() -> None:
    policy = _knn_policy(_knn_bank([[0.0, 1.0]] * 12))
    abstaining = policy.model_copy(update={"floor_sim": 2.0})  # no similarity can clear it
    abstaining.attach_bank(policy.knn_bank())
    decision = knn_decision(abstaining, _QUERY)
    assert decision.model == "fable-5"
    assert decision.evidence is not None
    assert decision.evidence.gate == "novelty-abstain"
    assert decision.evidence.propensity == "fallback-forced"


def test_select_model_attaches_the_normalized_query_vector() -> None:
    policy = _knn_policy(_knn_bank([[0.0, 1.0]] * 12))
    decision = select_model(policy, "anything", embedder=_UnitEmbedder())
    vector = decision.query_embedding()
    assert vector is not None
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)
    # Private state, not part of the decision's shape: it must never reach a log row or a report.
    assert "query_embedding" not in decision.model_dump()


def test_a_sticky_decision_carries_no_evidence_and_no_vector() -> None:
    # It never consulted the policy at all, so reporting evidence would be inventing it.
    policy = _knn_policy(_knn_bank([[0.0, 1.0]] * 12))
    decision = select_model(policy, "anything", incumbent="haiku-4-5", embedder=_UnitEmbedder())
    assert decision.evidence is None
    assert decision.query_embedding() is None


def test_a_static_policy_carries_no_evidence() -> None:
    assert select_model(_static(), "anything").evidence is None


# --- `--embedder` resolution -------------------------------------------------------------
# Moved here with the code it covers when `resolve_embedder`/`probe_embedder` left the CLI
# (they are now used by two fit commands). They assert ValueError; that the CLI turns it into a
# usage error is covered in `wmo/cli/route_app_test.py`.
def test_embedder_auto_resolves_to_azure_when_the_standard_env_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://sheets.openai.azure.com")
    spec, line = resolve_embedder(
        "auto", dim=None, deployment=None, endpoint=None, api_key_env=None
    )
    assert spec.kind == "azure"
    assert spec.deployment == AZURE_EMBEDDER_DEPLOYMENT
    assert spec.dim == AZURE_EMBEDDER_DIM  # the champion's width, not the hashing default
    assert spec.endpoint == "https://sheets.openai.azure.com"
    assert spec.api_key_env == "AZURE_OPENAI_API_KEY"
    assert "azure text-embedding-3-large (3072d native)" in line


def test_embedder_auto_falls_back_to_hashing_and_quotes_the_measured_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    spec, line = resolve_embedder(
        "auto", dim=None, deployment=None, endpoint=None, api_key_env=None
    )
    assert spec.kind == "hashing"
    assert spec.dim == HASHING_EMBEDDER_DIM
    # The downgrade is never silent, and it quotes the numbers rather than warning vaguely.
    assert HASHING_DOWNGRADE_NOTICE in line
    assert "AZURE_OPENAI_API_KEY" in line and "AZURE_OPENAI_ENDPOINT" in line


def test_embedder_auto_needs_both_variables_not_just_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A key with no endpoint cannot embed anything; falling back is right, and the line says
    # which half is missing so the operator can finish the setup.
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    spec, line = resolve_embedder(
        "auto", dim=None, deployment=None, endpoint=None, api_key_env=None
    )
    assert spec.kind == "hashing"
    assert "AZURE_OPENAI_ENDPOINT unset" in line


def test_explicit_hashing_is_unchanged_even_with_the_azure_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://sheets.openai.azure.com")
    spec, line = resolve_embedder(
        "hashing", dim=None, deployment=None, endpoint=None, api_key_env=None
    )
    assert (spec.kind, spec.dim) == ("hashing", HASHING_EMBEDDER_DIM)
    assert "explicit" in line


def test_explicit_azure_keeps_its_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    spec, line = resolve_embedder(
        "azure",
        dim=None,
        deployment="text-embedding-3-large",
        endpoint="https://x",
        api_key_env="MY_KEY",
    )
    assert (spec.kind, spec.endpoint, spec.api_key_env) == ("azure", "https://x", "MY_KEY")
    assert "explicit" in line


def test_explicit_azure_gets_the_models_native_width_not_the_hashing_default() -> None:
    # The footgun this closes: `--dim` used to default to 512 everywhere, so an explicit azure
    # fit asked a 3072-dimensional model for 512-dimensional vectors and fitted on the
    # truncation. `dim` is the request's `dimensions` parameter, not bookkeeping.
    spec, line = resolve_embedder(
        "azure",
        dim=None,
        deployment="text-embedding-3-large",
        endpoint="https://x",
        api_key_env=None,
    )
    assert spec.dim == AZURE_EMBEDDER_DIM
    assert "3072d native" in line


def test_a_smaller_model_resolves_to_its_own_native_width() -> None:
    spec, _ = resolve_embedder(
        "azure",
        dim=None,
        deployment="text-embedding-3-small",
        endpoint="https://x",
        api_key_env=None,
    )
    assert spec.dim == 1536


def test_an_unrecognized_deployment_name_assumes_3_large_and_says_so() -> None:
    # Azure deployment names are operator-chosen, so this is common; the guess is stated in the
    # line and one flag overrides it.
    spec, line = resolve_embedder(
        "azure", dim=None, deployment="prod-embeddings", endpoint="https://x", api_key_env=None
    )
    assert spec.dim == AZURE_EMBEDDER_DIM
    assert "assumed" in line and "--dim" in line


def test_an_explicit_dim_is_honored_verbatim_on_the_azure_path() -> None:
    # A deliberate reduction is still available; it is just no longer the silent default.
    spec, line = resolve_embedder(
        "azure",
        dim=256,
        deployment="text-embedding-3-large",
        endpoint="https://x",
        api_key_env=None,
    )
    assert spec.dim == 256
    assert "256d as asked" in line


def test_an_explicit_dim_wins_over_the_auto_resolved_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://sheets.openai.azure.com")
    spec, _ = resolve_embedder("auto", dim=256, deployment=None, endpoint=None, api_key_env=None)
    assert spec.dim == 256


def test_explicit_azure_without_a_deployment_says_which_flag_is_missing() -> None:
    with pytest.raises(ValueError) as caught:
        resolve_embedder("azure", dim=None, deployment=None, endpoint="https://x", api_key_env=None)
    assert "--deployment" in str(caught.value)


def test_an_unknown_embedder_is_a_usage_error() -> None:
    with pytest.raises(ValueError) as caught:
        resolve_embedder("word2vec", dim=None, deployment=None, endpoint=None, api_key_env=None)
    assert "auto, hashing, openai or azure" in str(caught.value)


def test_the_azure_resolution_line_says_it_bills_an_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolving to azure is spend nobody typed a flag for, so it must be stated.
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://sheets.openai.azure.com")
    _, line = resolve_embedder("auto", dim=None, deployment=None, endpoint=None, api_key_env=None)
    assert "calls the embedding API and is billed" in line


def test_probe_passes_for_the_offline_hashing_embedder() -> None:
    probe_embedder(EmbedderSpec(dim=32))


def test_probe_turns_a_dead_embedding_deployment_into_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded Azure layout points at resources WITHOUT an embedding deployment.

    Without the probe that surfaced as a traceback wall from inside the fit, after the matrix
    had been read. It must name the deployment, the endpoint, and the way out.
    """

    def _explode(_self: EmbedderSpec) -> object:
        raise RuntimeError("Resource not found (404)")

    monkeypatch.setattr(EmbedderSpec, "build", _explode)
    spec = EmbedderSpec(
        kind="azure", dim=3072, deployment="text-embedding-3-large", endpoint="https://x"
    )
    with pytest.raises(ValueError) as caught:
        probe_embedder(spec)
    message = str(caught.value)
    assert "text-embedding-3-large" in message
    assert "https://x" in message
    assert "404" in message
    assert "--embedder hashing" in message
    assert "Traceback" not in message


def test_probe_rejects_an_embedder_that_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Empty:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[] for _ in texts]

    monkeypatch.setattr(EmbedderSpec, "build", lambda _self: _Empty())
    with pytest.raises(ValueError, match="empty vector"):
        probe_embedder(
            EmbedderSpec(kind="azure", dim=3072, deployment="embed", endpoint="https://x")
        )


def test_a_policy_stamped_with_an_unrunnable_compressor_version_does_not_mount() -> None:
    # Requirement A at the version grain: same id, different implementation, different bytes.
    with pytest.raises(ValidationError, match="fitted against version 99"):
        RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            compression=CompressionConfig(compressor_id="truncate", compressor_version="99"),
        )


def test_a_per_cluster_override_is_version_checked_too(tmp_path: Path) -> None:
    policy = _rank_policy()
    clusters = [
        policy.clusters[0].model_copy(
            update={
                "compression": CompressionConfig(compressor_id="truncate", compressor_version="99")
            }
        ),
        *policy.clusters[1:],
    ]
    path = tmp_path / "policy.json"
    policy.model_copy(update={"clusters": clusters}).save(path)
    with pytest.raises(ValidationError, match="fitted against version 99"):
        RoutingPolicy.load(path)


# --- cache-aware decisions (Silen directive 2026-07-28; design in findings/cacheaware.md) ---


def _cached_pool() -> list[PoolEntry]:
    """The test pool with explicit prices and cache-read rates (fable pricey, haiku cheap)."""
    return [
        PoolEntry(
            name="fable-5",
            kind=ProviderKind.ANTHROPIC,
            model="claude-fable-5",
            input_per_mtok=10.0,
            output_per_mtok=40.0,
            cached_input_per_mtok=1.0,
        ),
        PoolEntry(
            name="haiku-4-5",
            kind=ProviderKind.ANTHROPIC,
            model="claude-haiku-4-5",
            input_per_mtok=1.0,
            output_per_mtok=4.0,
            cached_input_per_mtok=0.1,
        ),
    ]


def test_cache_credit_formula_and_its_zero_cases() -> None:
    policy = _knn_policy(_knn_bank([[0.0, 1.0]] * 12))
    policy = policy.model_copy(update={"pool": _cached_pool()})
    # 4000 chars ~= 1000 tokens; fable saves (10 - 1) per Mtok => 1000 * 9 / 1e6.
    assert cache_credit_usd(policy, "fable-5", 4000) == pytest.approx(0.009)
    assert cache_credit_usd(policy, "fable-5", 0) == 0.0
    assert cache_credit_usd(policy, "a-stranger", 4000) == 0.0
    # No cached rate on the entry: cache reads bill at the full rate, so no credit.
    no_cache = policy.model_copy(update={"pool": _pool()})
    assert cache_credit_usd(no_cache, "fable-5", 4000) == 0.0


def test_cache_blind_paths_are_bit_identical_with_zero_credit() -> None:
    """The hard invariant: no incumbent, or credit 0, must not change the decision at all."""
    for rewards, pick_lam in ([[0.0, 1.0]] * 12, 0.0), ([[1.0, 1.0]] * 12, 0.5):
        policy = _knn_policy(_knn_bank(rewards), pick_lam=pick_lam)
        plain = knn_decision(policy, _QUERY)
        with_args = knn_decision(policy, _QUERY, incumbent="fable-5", cache_credit=0.0)
        no_incumbent = knn_decision(policy, _QUERY, incumbent=None, cache_credit=0.5)
        assert plain.model_dump() == with_args.model_dump()
        assert plain.model_dump() == no_incumbent.model_dump()


def test_cache_credit_flips_the_guards_pricier_test() -> None:
    """A cache-warm pricey incumbent is effectively cheaper, so the challenger doubles z.

    7 wins / 5 losses: delta 0.167 clears z=0.5 (needs > 0.149) but not the doubled bar
    (needs > 0.297). Cache-blind, haiku is the cheaper pick and routes; with the incumbent
    fable credited below haiku's price, haiku becomes the effectively-pricier pick and the
    guard reverts it. Quality evidence identical in both runs: only the billing view moved.
    """
    rewards = [[0.0, 1.0]] * 7 + [[1.0, 0.0]] * 5
    policy = _knn_policy(_knn_bank(rewards), knn_z=0.5, knn_min_pairs=8)
    blind = knn_decision(policy, _QUERY)
    assert blind.model == "haiku-4-5"
    assert blind.evidence is not None
    credited = knn_decision(
        policy,
        _QUERY,
        incumbent="fable-5",
        cache_credit=_PRICEY,  # floors fable to $0
    )
    assert credited.model == "fable-5"
    assert credited.evidence is not None
    assert credited.evidence.cache_credit_usd == pytest.approx(_PRICEY)
    # The evidence numbers the guard tested are the SAME: cache state bought no confidence.
    assert credited.evidence.mean_diff == pytest.approx(blind.evidence.mean_diff)
    assert credited.evidence.se == pytest.approx(blind.evidence.se)


def test_cache_credit_reprices_the_cost_knob_tilt() -> None:
    """Equal quality, pick_lam on: blind tilt prefers cheap haiku (then the guard reverts a
    zero-delta pick); credited, the incumbent fable is effectively free and IS the argmax."""
    policy = _knn_policy(_knn_bank([[1.0, 1.0]] * 12), pick_lam=0.5)
    blind = knn_decision(policy, _QUERY)
    assert blind.model == "fable-5"
    assert blind.evidence is not None and blind.evidence.gate == "reverted"
    credited = knn_decision(policy, _QUERY, incumbent="fable-5", cache_credit=_PRICEY)
    assert credited.model == "fable-5"
    assert credited.evidence is not None
    assert credited.evidence.gate is None  # baseline led the priced argmax outright
    assert credited.evidence.cache_credit_usd == pytest.approx(_PRICEY)


def test_select_model_cache_aware_prices_the_incumbent_instead_of_sticking() -> None:
    rewards = [[0.0, 1.0]] * 12  # haiku dominates on evidence
    bank = _knn_bank(rewards)
    sticky = _knn_policy(bank)
    aware = _knn_policy(bank).model_copy(update={"cache_aware": True, "pool": _cached_pool()})
    aware.attach_bank(bank)
    kwargs = {"incumbent": "fable-5", "embedder": _UnitEmbedder(), "conversation_chars": 4000}
    stuck = select_model(sticky, "anything", **kwargs)
    assert stuck.reason == "sticky: conversation affinity"
    routed = select_model(aware, "anything", **kwargs)
    # The evidence overwhelms the credit: cache-aware routing may still switch.
    assert routed.model == "haiku-4-5"
    assert routed.evidence is not None and routed.evidence.gate == "passed"
    # And with no incumbent the cache-aware policy behaves exactly like today.
    fresh_aware = select_model(aware, "anything", embedder=_UnitEmbedder())
    fresh_sticky = select_model(sticky, "anything", embedder=_UnitEmbedder())
    assert fresh_aware.model_dump() == fresh_sticky.model_dump()


def test_cache_aware_switch_gate_blocks_unjustified_switches() -> None:
    """7/5 mixed evidence clears the baseline guard for haiku but NOT the switch gate when
    the incumbent is haiku itself... exercised the other way: incumbent haiku, evidence says
    baseline fable leads -> switching back to fable must be justified AGAINST haiku."""
    # Fable wins 7, haiku wins 5: fable leads the profile, blind decision serves fable.
    rewards = [[1.0, 0.0]] * 7 + [[0.0, 1.0]] * 5
    policy = _knn_policy(_knn_bank(rewards), knn_z=0.5, knn_min_pairs=8)
    blind = knn_decision(policy, _QUERY)
    assert blind.model == "fable-5"
    # Cache-aware with incumbent haiku: fable's edge over haiku (delta +0.167) does not clear
    # the doubled bar (fable is pricier, effectively too: no credit can rescue a switch TO a
    # pricier model), so the conversation stays on haiku.
    kept = knn_decision(policy, _QUERY, incumbent="haiku-4-5", cache_credit=0.001)
    assert kept.model == "haiku-4-5"
    assert "switch gate" in kept.reason
    assert kept.evidence is not None and kept.evidence.gate == "reverted"


def test_cache_aware_switch_gate_allows_justified_switches() -> None:
    """Unanimous evidence against the incumbent clears the switch gate."""
    rewards = [[1.0, 0.0]] * 12  # fable always wins, haiku always loses
    policy = _knn_policy(_knn_bank(rewards), knn_z=0.5, knn_min_pairs=8)
    switched = knn_decision(policy, _QUERY, incumbent="haiku-4-5", cache_credit=0.001)
    assert switched.model == "fable-5"


def test_cache_aware_incumbent_absent_from_the_bank_sticks() -> None:
    """An incumbent the bank was never fitted on cannot be tested, so the conversation
    sticks with it (the amendment's conservative rule) instead of switching unjudged."""
    rewards = [[1.0, 0.0]] * 12  # unanimous for fable, but the incumbent is not a bank column
    policy = _knn_policy(_knn_bank(rewards), knn_z=0.5, knn_min_pairs=8)
    kept = knn_decision(policy, _QUERY, incumbent="opus-4-8", cache_credit=0.001)
    assert kept.model == "opus-4-8"
    assert "not in the fitted bank" in kept.reason
    assert kept.evidence is not None
    assert kept.evidence.gate == "reverted"
    assert kept.evidence.n_pairs == 0
