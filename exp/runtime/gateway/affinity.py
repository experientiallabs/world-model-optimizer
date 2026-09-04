"""Deterministic cache-affinity placement over a pool's certified rungs.

Under a pool's ``maximize_cache_affinity`` policy the initial dispatch order is
a per-request weighted rendezvous (highest-random-weight) permutation: every
worker computes the same order from the same request fingerprint and catalog
weights, so fleet-wide conversation-to-rung affinity needs no per-worker memory
and no shared-state reads. Removing a rung (death, shed, flex-start cycling)
moves only that rung's fingerprints to their deterministic next choice; every
other placement is untouched, so a spilled conversation lands on the SAME
alternate every time and builds warm cache there instead of scattering.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayRequest


def affinity_seed_material(
    request: GatewayRequest,
    *,
    continuation_episode_key: str | None,
    request_id: str,
) -> str:
    """Choose the request's stable conversation identity for affinity hashing.

    Reuses the identities the lineage machinery already carries rather than
    recomputing a prefix digest: the caller's explicit ``prompt_cache_key``
    (OpenAI's own same-prefix cache-routing hint) wins, then a Responses
    continuation's ORIGINAL episode key (so a continued conversation keeps its
    placement), then the session-scoped ``X-Client-Request-Id`` that agent CLIs
    send on every request of a session, then the per-operation idempotency
    key. A request carrying none of these degrades to its request id, which
    rendezvous turns into a weighted spread instead of a fixed rung.

    Args:
        request: Canonical decoded request.
        continuation_episode_key: The original turn's episode key when this
            request continues a stored response, else ``None``.
        request_id: The accepted request's id, the last-resort material.

    Returns:
        The strongest available stable conversation identity.
    """
    return (
        request.prompt_cache_key
        or continuation_episode_key
        or request.client_request_id
        or request.idempotency_key
        or request_id
    )


def affinity_fingerprint(
    *,
    organization_id: str,
    identity_id: str,
    material: str,
) -> bytes:
    """Derive the tenant-isolated affinity fingerprint for one request.

    The fingerprint deliberately EXCLUDES the alias revision: a catalog write
    (the daily price sync, one admin edit) must not scatter every warm
    conversation across rungs. Tenant fields keep one organization's chosen
    material from colliding with another's.

    Args:
        organization_id: Authorized organization.
        identity_id: Authorized identity within the organization.
        material: Stable conversation identity from
            :func:`affinity_seed_material`.

    Returns:
        A 32-byte digest suitable as a rendezvous key.
    """
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "organization_id": organization_id,
                "identity_id": identity_id,
                "material": material,
            }
        )
    ).digest()


def rendezvous_order(
    fingerprint: bytes,
    weighted_rungs: Sequence[tuple[str, float]],
) -> tuple[int, ...]:
    """Order rung indexes by descending weighted rendezvous score.

    Standard highest-random-weight hashing with the weighted score
    ``-weight / ln(u)`` where ``u`` is a uniform draw in (0, 1) keyed on the
    fingerprint and the rung's deployment id alone (never the connection
    digest, so a credential rotation does not move placement). Pure integer
    and float arithmetic over already-loaded catalog data: identical on every
    worker, no state.

    Args:
        fingerprint: 32-byte affinity fingerprint.
        weighted_rungs: One ``(deployment_id, weight)`` per rung, in route
            order; weights must be positive and finite.

    Returns:
        A permutation of ``range(len(weighted_rungs))``, best rung first,
        with the route index breaking (practically impossible) score ties.

    Raises:
        ValueError: A weight is not positive and finite.
    """
    scores: list[float] = []
    for deployment_id, weight in weighted_rungs:
        if not (weight > 0) or math.isinf(weight):
            raise ValueError("rendezvous weights must be positive and finite")
        digest = hashlib.blake2b(
            deployment_id.encode("utf-8"),
            digest_size=8,
            key=fingerprint,
        ).digest()
        # Center the draw inside (0, 1) so ln(u) is finite and nonzero.
        uniform = (int.from_bytes(digest, "big") + 0.5) / 2.0**64
        scores.append(-weight / math.log(uniform))
    return tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))
