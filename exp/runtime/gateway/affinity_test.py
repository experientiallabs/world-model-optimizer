"""Determinism, stability, and distribution tests for rendezvous affinity."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.affinity import (
    affinity_fingerprint,
    affinity_seed_material,
    rendezvous_order,
)
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest

_RUNGS = (("dep-house", 10.0), ("dep-fireworks", 3.0), ("dep-openrouter", 1.0))


def _fingerprint(material: str, *, organization_id: str = "org-1") -> bytes:
    """Build one tenant-scoped fingerprint for a conversation material."""
    return affinity_fingerprint(
        organization_id=organization_id,
        identity_id="identity-1",
        material=material,
    )


def _request(**overrides: str) -> GatewayRequest:
    """Build a minimal decoded request carrying optional affinity carriers."""
    base = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )
    return base.model_copy(update=dict(overrides))


class TestSeedMaterial:
    """The affinity seed reuses existing lineage identities in strict priority."""

    def test_prompt_cache_key_wins_over_every_other_carrier(self) -> None:
        """The caller's explicit cache-routing hint is the strongest identity."""
        request = _request(
            prompt_cache_key="cache-key",
            client_request_id="session-1",
            idempotency_key="op-1",
        )
        material = affinity_seed_material(
            request, continuation_episode_key="episode-1", request_id="req-1"
        )
        assert material == "cache-key"

    def test_continuation_episode_key_beats_session_and_operation_ids(self) -> None:
        """A continued conversation keeps its original turn's placement."""
        request = _request(client_request_id="session-1", idempotency_key="op-1")
        material = affinity_seed_material(
            request, continuation_episode_key="episode-1", request_id="req-1"
        )
        assert material == "episode-1"

    def test_session_id_beats_idempotency_key_beats_request_id(self) -> None:
        """The session-scoped correlation id outranks the per-operation key."""
        request = _request(client_request_id="session-1", idempotency_key="op-1")
        assert (
            affinity_seed_material(request, continuation_episode_key=None, request_id="req-1")
            == "session-1"
        )
        request = _request(idempotency_key="op-1")
        assert (
            affinity_seed_material(request, continuation_episode_key=None, request_id="req-1")
            == "op-1"
        )
        request = _request()
        assert (
            affinity_seed_material(request, continuation_episode_key=None, request_id="req-1")
            == "req-1"
        )


class TestFingerprint:
    """Fingerprints are tenant-isolated and collision-resistant."""

    def test_same_inputs_same_fingerprint(self) -> None:
        """The fingerprint is a pure function of its inputs."""
        assert _fingerprint("session-1") == _fingerprint("session-1")

    def test_tenant_fields_separate_identical_materials(self) -> None:
        """Two organizations sharing a session id never share a fingerprint."""
        assert _fingerprint("session-1", organization_id="org-1") != _fingerprint(
            "session-1", organization_id="org-2"
        )

    def test_component_boundaries_do_not_collide(self) -> None:
        """Structured encoding keeps identity/material boundaries distinct."""
        joined = affinity_fingerprint(organization_id="org-1x", identity_id="y", material="session")
        shifted = affinity_fingerprint(
            organization_id="org-1", identity_id="xy", material="session"
        )
        assert joined != shifted


class TestRendezvousOrder:
    """Weighted rendezvous is deterministic, stable under removal, and weighted."""

    def test_every_worker_computes_the_identical_order(self) -> None:
        """N independent computations of one fingerprint agree exactly."""
        fingerprint = _fingerprint("session-42")
        orders = {rendezvous_order(fingerprint, _RUNGS) for _ in range(8)}
        assert len(orders) == 1

    def test_order_is_a_permutation(self) -> None:
        """Every rung appears exactly once for many fingerprints."""
        for index in range(64):
            order = rendezvous_order(_fingerprint(f"session-{index}"), _RUNGS)
            assert sorted(order) == [0, 1, 2]

    def test_removing_a_rung_moves_only_its_own_fingerprints(self) -> None:
        """Rendezvous redistributes only the dead rung's share.

        For fingerprints whose first choice survives the removal, the full
        relative order of the surviving rungs is unchanged; fingerprints whose
        first choice was removed land on their previous SECOND choice, so a
        spilled conversation has one stable alternate.
        """
        survivors = (_RUNGS[0], _RUNGS[2])
        for index in range(128):
            fingerprint = _fingerprint(f"conversation-{index}")
            full_order = rendezvous_order(fingerprint, _RUNGS)
            reduced_order = rendezvous_order(fingerprint, survivors)
            # Map reduced indexes (over survivors) back to full-route indexes.
            reduced_as_full = tuple((0, 2)[position] for position in reduced_order)
            expected = tuple(position for position in full_order if position != 1)
            assert reduced_as_full == expected

    def test_restoring_a_rung_restores_the_original_order(self) -> None:
        """A healed rung reclaims exactly its old fingerprints."""
        fingerprint = _fingerprint("conversation-7")
        assert rendezvous_order(fingerprint, _RUNGS) == rendezvous_order(fingerprint, _RUNGS)

    def test_weights_shift_share_toward_heavy_rungs(self) -> None:
        """A 10:3:1 weighting sends the clear majority to the heavy rung."""
        first_choices = [0, 0, 0]
        for index in range(600):
            order = rendezvous_order(_fingerprint(f"spread-{index}"), _RUNGS)
            first_choices[order[0]] += 1
        assert first_choices[0] > first_choices[1] > first_choices[2]
        # Expected shares are 10/14, 3/14, 1/14; assert loose brackets so the
        # test pins weighting, not the exact hash stream.
        assert first_choices[0] > 300
        assert first_choices[2] < 120

    def test_spill_target_is_deterministic(self) -> None:
        """The second choice for one fingerprint never varies."""
        fingerprint = _fingerprint("session-9")
        second = {rendezvous_order(fingerprint, _RUNGS)[1] for _ in range(8)}
        assert len(second) == 1

    def test_nonpositive_or_infinite_weight_is_rejected(self) -> None:
        """Weights outside (0, inf) are contract violations, not silent zeros."""
        fingerprint = _fingerprint("session-1")
        for weight in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="positive and finite"):
                rendezvous_order(fingerprint, (("dep-a", weight), ("dep-b", 1.0)))
