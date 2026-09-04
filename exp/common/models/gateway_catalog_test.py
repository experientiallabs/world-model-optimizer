"""Tests for conservative gateway deployment normalization."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models.catalog import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayLongContextTier,
    GatewayPoolRecord,
    GatewayRungDispatchPolicy,
    GatewayTokenPrices,
    ModelCatalog,
    ModelRecord,
    SFTModelProvenance,
)
from exp.common.models.gateway_catalog import (
    SANE_MAX_SNAPSHOT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    CatalogSnapshotDigestError,
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
    is_foreign_snapshot,
    load_forward_compatible,
    normalize_gateway_catalog,
    read_pinned_normalized_snapshot,
)
from exp.common.models.model import ModelCapabilities, ModelSnapshot

_DIGEST = "a" * 64


def _minimal_normalized() -> NormalizedGatewayCatalog:
    """One valid single-deployment normalized catalog for the reader tests."""
    deployment = ExactModelDeployment(
        deployment_id="dep-1",
        source_alias="dep-1",
        exact_model_id="exact-1",
        connection="conn",
        provider="openai",
        provider_model="m-1",
        connection_sha256=_DIGEST,
        capabilities_sha256=_DIGEST,
    )
    pool = ExactModelPool(pool_id="dep-1", exact_model_id="exact-1", deployment_ids=("dep-1",))
    return NormalizedGatewayCatalog(deployments=(deployment,), pools=(pool,))


def test_load_forward_compatible_drops_unknown_fields_and_reports_them() -> None:
    """A snapshot authored by a newer build (unknown top-level and nested pool
    fields) parses with those fields dropped and reported, so an old pod reading
    it mid-roll never hard-fails on ``extra="forbid"``."""
    catalog = _minimal_normalized()
    raw = json.loads(catalog.model_dump_json())
    raw["future_top_level"] = {"anything": 1}
    raw["pools"][0]["future_pool_field"] = "maximize_something_new"
    parsed, dropped = load_forward_compatible(NormalizedGatewayCatalog, json.dumps(raw))
    assert parsed == catalog
    assert ("future_top_level",) in dropped
    assert ("pools", 0, "future_pool_field") in dropped


def test_load_forward_compatible_still_enforces_required_fields_and_invariants() -> None:
    """Only unknown extras are tolerated: a genuinely malformed snapshot (a
    required field removed) still raises instead of being silently served."""
    raw = json.loads(_minimal_normalized().model_dump_json())
    del raw["pools"][0]["exact_model_id"]
    with pytest.raises(ValidationError):
        load_forward_compatible(NormalizedGatewayCatalog, json.dumps(raw))


def test_load_forward_compatible_is_strict_when_there_are_no_extras() -> None:
    """A same-version snapshot parses exactly, reporting no dropped fields."""
    catalog = _minimal_normalized()
    parsed, dropped = load_forward_compatible(NormalizedGatewayCatalog, catalog.model_dump_json())
    assert parsed == catalog
    assert dropped == ()


def test_is_foreign_snapshot_tracks_the_schema_version() -> None:
    """Only a schema-version skew marks a snapshot foreign."""
    assert not is_foreign_snapshot(_minimal_normalized())
    bumped = _minimal_normalized().model_copy(
        update={"schema_version": SNAPSHOT_SCHEMA_VERSION + 1}
    )
    assert is_foreign_snapshot(bumped)


def test_wild_schema_version_is_not_foreign_and_fails_closed() -> None:
    """Corruption guard: a schema_version beyond the sane range is NOT trusted as
    a real cross-build skew, so it takes the strict digest path and fails closed
    (never served unverified) instead of suppressing the pinned-digest check."""
    wild = _minimal_normalized().model_copy(
        update={"schema_version": SANE_MAX_SNAPSHOT_SCHEMA_VERSION + 1}
    )
    assert not is_foreign_snapshot(wild)
    with pytest.raises(CatalogSnapshotDigestError):
        read_pinned_normalized_snapshot(wild.model_dump_json().encode(), "b" * 64)


def test_read_pinned_snapshot_same_version_requires_the_exact_digest() -> None:
    """A same-version snapshot must reproduce its pinned digest; a mismatch is
    content tampering and fails closed with a distinct error."""
    catalog = _minimal_normalized()
    assert (
        read_pinned_normalized_snapshot(
            catalog.model_dump_json().encode(), catalog.identity_sha256()
        )
        == catalog
    )
    with pytest.raises(CatalogSnapshotDigestError):
        read_pinned_normalized_snapshot(catalog.model_dump_json().encode(), "b" * 64)


def test_read_pinned_snapshot_serves_a_cross_version_snapshot_without_the_digest_check() -> None:
    """Roll-safety guard: a snapshot from a NEWER build (higher schema_version,
    an unknown pool field, a digest this build cannot recompute) is SERVED under
    its pinned digest rather than raising, so a rolling deploy never hard-fails.
    """
    raw = json.loads(_minimal_normalized().model_dump_json())
    raw["schema_version"] = SNAPSHOT_SCHEMA_VERSION + 1
    raw["pools"][0]["future_pool_field"] = "maximize_something_new"
    served = read_pinned_normalized_snapshot(json.dumps(raw).encode(), "b" * 64)
    assert served.schema_version == SNAPSHOT_SCHEMA_VERSION + 1
    assert served.pools[0].pool_id == "dep-1"


def _identity_fixture_catalog() -> ModelCatalog:
    """One frozen authored catalog exercising every identity-bearing surface.

    Mixes default and non-default values on purpose: nested gateway capability
    flags, integer prices with a long-context tier, an authored capability
    declaration, both billing sources, a certified non-default-failover pool,
    and a bare all-default record. Never edit this fixture to make a test
    pass; it is the pinned input of the identity change-detector.
    """
    certification = GatewayEquivalenceCertification(
        certification_id="certification-pinned",
        provenance="operator comparison run 2026-09-01",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    return ModelCatalog(
        connections={
            "openai": ConnectionConfig(provider="openai"),
            "local": ConnectionConfig(
                provider="openai-compatible", base_url="https://local.example.test/v1"
            ),
        },
        models={
            "rich": ModelRecord(
                connection="local",
                model="provider-rich",
                revision="2026-09-01",
                billing_source=BillingSource.HOST_MANAGED,
                capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="low"),
                gateway=GatewayDeploymentMetadata(
                    exact_model_id="exact-certified",
                    capabilities=GatewayDeploymentCapabilities(
                        supports_streaming=True, supports_image_input=True
                    ),
                    prices=GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=150,
                        output_micro_usd_per_million_tokens=600,
                        long_context=GatewayLongContextTier(
                            input_threshold_tokens=200_000,
                            input_micro_usd_per_million_tokens=300,
                        ),
                    ),
                ),
            ),
            "twin": ModelRecord(
                connection="openai",
                model="provider-twin",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
            ),
            "bare": ModelRecord(
                connection="openai",
                model="provider-bare",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
        },
        gateway_pools={
            "certified-pool": GatewayPoolRecord(
                exact_model_id="exact-certified",
                deployment_aliases=("rich", "twin"),
                equivalence=certification,
                failover_mode="maximize_cache",
            )
        },
    )


def test_identity_digest_is_pinned_until_a_deliberate_schema_version_bump() -> None:
    """The identity serialization of a frozen fixture cannot drift silently.

    Three consecutive releases (0.7.22, 0.7.23, 0.7.25) each added defaulted
    capability flags that serialized into every published digest without a
    ``SNAPSHOT_SCHEMA_VERSION`` bump, so every catalog read in a mixed-version
    fleet failed its own digest check. If this test fails you changed the
    identity serialization; there are exactly two lawful outs:

    - You ADDED a field: give it a default. Identity excludes default-holding
      fields, so a properly defaulted addition never trips this test at all.
    - You deliberately changed identity output (default change, rename,
      removal, normalization change): bump ``SNAPSHOT_SCHEMA_VERSION`` and
      repin BOTH values below in the same change.
    """
    normalized = normalize_gateway_catalog(_identity_fixture_catalog())
    assert (SNAPSHOT_SCHEMA_VERSION, normalized.identity_sha256()) == (
        3,
        "f9e74a42f5b47ec0a0739836ef15bbfa36df4eaa7c34efee10c421d46962529f",
    )


def test_added_defaulted_fields_and_explicit_defaults_do_not_perturb_identity() -> None:
    """Identity is stable across additive schema growth and default spelling.

    A default-constructed nested declaration must contribute zero identity
    bytes (that is what makes a new defaulted field identity-invisible), and a
    field spelled explicitly at its default must hash identically to leaving
    it unset.
    """
    for model in (
        GatewayDeploymentCapabilities(),
        GatewayTokenPrices(),
        GatewayDeploymentMetadata(),
        GatewayRungDispatchPolicy(),
        ModelCapabilities(),
        NormalizedGatewayCatalog(),
    ):
        assert model.model_dump(mode="json", by_alias=True, exclude_defaults=True) == {}

    explicit = _identity_fixture_catalog().model_copy(deep=True)
    spelled = explicit.models["bare"].model_copy(
        update={"gateway": GatewayDeploymentMetadata(capabilities=GatewayDeploymentCapabilities())}
    )
    respelled = explicit.model_copy(update={"models": {**explicit.models, "bare": spelled}})
    assert (
        normalize_gateway_catalog(respelled).identity_sha256()
        == normalize_gateway_catalog(_identity_fixture_catalog()).identity_sha256()
    )


def test_normalized_schema_change_requires_a_schema_version_bump() -> None:
    """Anti-regression change-detector for the roll-safety contract.

    A rolling deploy detects a cross-version snapshot only by ``schema_version``,
    so every change to the normalized-catalog TOP-LEVEL shape MUST bump
    ``SNAPSHOT_SCHEMA_VERSION`` in the same change (nested additive growth is
    covered by the pinned identity digest test above, which defaulted fields
    pass automatically). If this fails, bump ``SNAPSHOT_SCHEMA_VERSION`` and
    update the pinned fingerprint below together, so the reader keeps serving
    old and new pods through a roll instead of hard-failing.
    """
    fingerprint = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "normalized": sorted(NormalizedGatewayCatalog.model_fields),
        "deployment": sorted(ExactModelDeployment.model_fields),
        "pool": sorted(ExactModelPool.model_fields),
    }
    assert fingerprint == {
        "schema_version": 3,
        "normalized": ["deployments", "pools", "schema_version"],
        "deployment": [
            "billing_source",
            "capabilities",
            "capabilities_sha256",
            "connection",
            "connection_sha256",
            "deployment_id",
            "exact_model_id",
            "gateway",
            "provider",
            "provider_model",
            "revision",
            "source_alias",
        ],
        "pool": [
            "deployment_ids",
            "equivalence",
            "exact_model_id",
            "failover_mode",
            "pool_id",
        ],
    }


def test_singleton_identity_includes_connection_model_revision_and_full_capabilities() -> None:
    """Endpoint and capability variants cannot collide during legacy singleton migration."""
    catalog = ModelCatalog(
        connections={
            "first": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://first.example.test/v1",
            ),
            "second": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://second.example.test/v1",
            ),
        },
        models={
            "first-low": ModelRecord(
                connection="first",
                model="same-name",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                revision="2026-08-01",
                capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="low"),
            ),
            "first-high": ModelRecord(
                connection="first",
                model="same-name",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                revision="2026-08-01",
                capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="high"),
            ),
            "second-low": ModelRecord(
                connection="second",
                model="same-name",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                revision="2026-08-01",
                capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="low"),
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)
    by_alias = {item.source_alias: item for item in normalized.deployments}

    assert len({item.exact_model_id for item in normalized.deployments}) == 3
    assert by_alias["first-low"].connection_sha256 != by_alias["second-low"].connection_sha256
    assert by_alias["first-low"].capabilities_sha256 != by_alias["first-high"].capabilities_sha256
    assert tuple(pool.pool_id for pool in normalized.pools) == (
        "first-high",
        "first-low",
        "second-low",
    )


def test_identical_alias_records_remain_separate_singleton_pools() -> None:
    """Duplicate metadata may share exact identity without being silently grouped for failover."""
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "coding-a": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
            "coding-b": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert normalized.deployments[0].exact_model_id == normalized.deployments[1].exact_model_id
    assert normalized.pools[0].pool_id != normalized.pools[1].pool_id
    assert all(len(pool.deployment_ids) == 1 for pool in normalized.pools)


def test_billing_source_changes_catalog_identity_without_changing_exact_model() -> None:
    """Billing ownership is frozen in deployment identity but not model equivalence."""
    connection = ConnectionConfig(provider="openai")
    customer = ModelCatalog(
        connections={"openai": connection},
        models={
            "coding": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            )
        },
    )
    host = customer.model_copy(
        update={
            "models": {
                "coding": customer.models["coding"].model_copy(
                    update={"billing_source": BillingSource.HOST_MANAGED}
                )
            }
        }
    )

    customer_normalized = normalize_gateway_catalog(customer)
    host_normalized = normalize_gateway_catalog(host)

    assert customer_normalized.deployments[0].exact_model_id == (
        host_normalized.deployments[0].exact_model_id
    )
    assert customer_normalized.deployments[0].billing_source == BillingSource.CUSTOMER_MANAGED
    assert host_normalized.deployments[0].billing_source == BillingSource.HOST_MANAGED
    assert customer_normalized.identity_sha256() != host_normalized.identity_sha256()


def test_legacy_normalized_deployment_defaults_to_customer_managed_billing() -> None:
    """A normalized payload written before billing attribution decodes conservatively."""
    deployment = ExactModelDeployment.model_validate(
        {
            "deployment_id": "coding",
            "source_alias": "coding",
            "exact_model_id": "exact-coding",
            "connection": "openai",
            "provider": "openai",
            "provider_model": "gpt-coding",
            "connection_sha256": "a" * 64,
            "capabilities_sha256": "b" * 64,
        }
    )

    assert deployment.billing_source == BillingSource.CUSTOMER_MANAGED


def test_normalized_deployment_rejects_inconsistent_reasoning_metadata() -> None:
    """Persisted snapshots cannot combine required reasoning with unsupported models."""
    with pytest.raises(ValidationError, match="supports_reasoning=true"):
        ExactModelDeployment(
            deployment_id="coding",
            source_alias="coding",
            exact_model_id="exact-coding",
            connection="openai",
            provider="openai",
            provider_model="gpt-coding",
            connection_sha256="a" * 64,
            capabilities_sha256="b" * 64,
            capabilities=ModelCapabilities(),
            gateway=GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(
                    supported_reasoning_efforts=("medium",),
                    reasoning_default_effort="medium",
                    reasoning_effort_required=True,
                )
            ),
        )


def test_tinker_and_sft_records_are_not_gateway_deployments() -> None:
    """Training handles retain provenance without being treated as operational model routes."""
    sampling_handle = "tinker://sampling/run-1"
    provenance = SFTModelProvenance(
        source_dataset=ArtifactInput(artifact_id="dataset", sha256=_DIGEST),
        optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
        training_spec_sha256="c" * 64,
        run_id="run-1",
        model_id="model-1",
        model_sha256="d" * 64,
        result_id="result-1",
        result_sha256="e" * 64,
        base_model=ModelSnapshot(
            provider="tinker",
            model_id="base",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities_sha256="f" * 64,
            connection_sha256="0" * 64,
        ),
        connection_config_sha256="1" * 64,
        sampling_handle_sha256=sha256_json({"sampling_handle": sampling_handle}),
    )
    catalog = ModelCatalog(
        connections={
            "openai": ConnectionConfig(provider="openai"),
            "tinker": ConnectionConfig(provider="tinker"),
        },
        models={
            "regular": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
            "training": ModelRecord(
                connection="openai",
                model=sampling_handle,
                billing_source=BillingSource.CUSTOMER_MANAGED,
                sft_provenance=provenance,
            ),
            "tinker-handle": ModelRecord(
                connection="tinker",
                model="base",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert tuple(item.source_alias for item in normalized.deployments) == ("regular",)
    assert normalized.identity_sha256() == normalize_gateway_catalog(catalog).identity_sha256()


def test_operator_certified_pool_preserves_explicit_deployment_order_and_provenance() -> None:
    """Only one authored certification groups deployments and fixes waterfall priority."""
    certification = GatewayEquivalenceCertification(
        certification_id="certification-one",
        provenance="operator comparison run 2026-08-18",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    catalog = ModelCatalog(
        connections={
            "anthropic": ConnectionConfig(provider="anthropic"),
            "openai": ConnectionConfig(provider="openai"),
        },
        models={
            "route-a": ModelRecord(
                connection="openai",
                model="provider-a",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
            ),
            "route-b": ModelRecord(
                connection="anthropic",
                model="provider-b",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
            ),
        },
        gateway_pools={
            "certified-pool": GatewayPoolRecord(
                exact_model_id="exact-certified",
                deployment_aliases=("route-b", "route-a"),
                equivalence=certification,
            )
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert normalized.pools == (
        ExactModelPool(
            pool_id="certified-pool",
            exact_model_id="exact-certified",
            deployment_ids=("route-b", "route-a"),
            equivalence=certification,
        ),
    )
    assert normalized.identity_sha256() == normalize_gateway_catalog(catalog).identity_sha256()


def test_authored_pool_failover_mode_carries_into_the_normalized_pool() -> None:
    """A maximize_cache authored pool normalizes to a maximize_cache exact-model pool."""
    certification = GatewayEquivalenceCertification(
        certification_id="certification-cache",
        provenance="operator comparison run 2026-09-01",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "route-a": ModelRecord(
                connection="openai",
                model="m-a",
                billing_source=BillingSource.HOST_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-cache"),
            ),
            "route-b": ModelRecord(
                connection="openai",
                model="m-b",
                billing_source=BillingSource.HOST_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-cache"),
            ),
        },
        gateway_pools={
            "cache-pool": GatewayPoolRecord(
                exact_model_id="exact-cache",
                deployment_aliases=("route-a", "route-b"),
                equivalence=certification,
                failover_mode="maximize_cache",
            )
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert normalized.pools[0].failover_mode == "maximize_cache"
    # An unset authored pool stays on the historical default.
    assert (
        GatewayPoolRecord(
            exact_model_id="x",
            deployment_aliases=("a", "b"),
            equivalence=certification,
        ).failover_mode
        == "maximize_availability"
    )


def test_affinity_pool_and_dispatch_policy_round_trip_and_move_identity() -> None:
    """The affinity opt-in normalizes intact, and AUTHORING it moves the digest.

    Adding the fields to the schema is identity-invisible (the pinned-digest
    test above); actually authoring a value is a real catalog change and must
    produce a new content address.
    """
    certification = GatewayEquivalenceCertification(
        certification_id="certification-affinity",
        provenance="operator comparison run 2026-09-04",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 9, 4, tzinfo=UTC),
    )

    def catalog(*, opted_in: bool) -> ModelCatalog:
        """Build the same two-rung catalog with and without the opt-in."""
        dispatch = (
            GatewayRungDispatchPolicy(concurrency_bound=8, fair_share=True, affinity_weight=10.0)
            if opted_in
            else None
        )
        return ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={
                "route-a": ModelRecord(
                    connection="openai",
                    model="m-a",
                    billing_source=BillingSource.HOST_MANAGED,
                    gateway=GatewayDeploymentMetadata(
                        exact_model_id="exact-affinity", dispatch=dispatch
                    ),
                ),
                "route-b": ModelRecord(
                    connection="openai",
                    model="m-b",
                    billing_source=BillingSource.HOST_MANAGED,
                    gateway=GatewayDeploymentMetadata(exact_model_id="exact-affinity"),
                ),
            },
            gateway_pools={
                "affinity-pool": GatewayPoolRecord(
                    exact_model_id="exact-affinity",
                    deployment_aliases=("route-a", "route-b"),
                    equivalence=certification,
                    failover_mode=(
                        "maximize_cache_affinity" if opted_in else "maximize_availability"
                    ),
                )
            },
        )

    normalized = normalize_gateway_catalog(catalog(opted_in=True))
    assert normalized.pools[0].failover_mode == "maximize_cache_affinity"
    lead = next(
        deployment for deployment in normalized.deployments if deployment.source_alias == "route-a"
    )
    assert lead.gateway.dispatch == GatewayRungDispatchPolicy(
        concurrency_bound=8, fair_share=True, affinity_weight=10.0
    )
    baseline = normalize_gateway_catalog(catalog(opted_in=False))
    assert normalized.identity_sha256() != baseline.identity_sha256()


def test_rung_dispatch_policy_rejects_incoherent_authoring() -> None:
    """Fairness without a bound, and degenerate bounds or weights, fail closed."""
    with pytest.raises(ValueError, match="concurrency_bound"):
        GatewayRungDispatchPolicy(fair_share=True)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(concurrency_bound=0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(affinity_weight=0.0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(affinity_weight=float("inf"))


def test_equivalence_catalog_rejects_implicit_false_or_ambiguous_grouping() -> None:
    """Missing exact declarations, training handles, and repeated membership fail closed."""
    certification = GatewayEquivalenceCertification(
        certification_id="certification-one",
        provenance="operator comparison run",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    with pytest.raises(ValidationError, match="declare exact model identity"):
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={
                "route-a": ModelRecord(
                    connection="openai",
                    model="provider-a",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                ),
                "route-b": ModelRecord(
                    connection="openai",
                    model="provider-b",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                ),
            },
            gateway_pools={
                "pool": GatewayPoolRecord(
                    exact_model_id="exact-certified",
                    deployment_aliases=("route-a", "route-b"),
                    equivalence=certification,
                )
            },
        )
    with pytest.raises(ValidationError, match="more than one pool"):
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={
                alias: ModelRecord(
                    connection="openai",
                    model=alias,
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
                )
                for alias in ("route-a", "route-b", "route-c")
            },
            gateway_pools={
                "pool-a": GatewayPoolRecord(
                    exact_model_id="exact-certified",
                    deployment_aliases=("route-a", "route-b"),
                    equivalence=certification,
                ),
                "pool-b": GatewayPoolRecord(
                    exact_model_id="exact-certified",
                    deployment_aliases=("route-b", "route-c"),
                    equivalence=certification,
                ),
            },
        )


def test_normalized_multi_deployment_pool_requires_operator_certification() -> None:
    """The runtime snapshot cannot construct implicit multi-route equivalence."""
    with pytest.raises(ValidationError, match="operator equivalence certification"):
        ExactModelPool(
            pool_id="unsafe",
            exact_model_id="exact-one",
            deployment_ids=("route-a", "route-b"),
        )
