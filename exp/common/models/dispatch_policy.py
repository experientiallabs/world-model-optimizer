"""Authored per-rung dispatch policy: admission bounds and affinity weight.

One nested, fully defaulted model hung off ``GatewayDeploymentMetadata``. It
is additive-defaulted on purpose: an unauthored rung contributes zero identity
bytes under the catalog's exclude-defaults digest, so adding this surface
moves no snapshot digests and needs no schema-version bump; authoring a value
is a real catalog change and produces a new content address.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel

FailoverMode = Literal["maximize_availability", "maximize_cache", "maximize_cache_affinity"]
"""How a pool's waterfall orders its rungs and reacts to a failed attempt.

``maximize_availability`` (the default, historical behavior) fails over to the
next rung on any failover-eligible error. ``maximize_cache`` does NOT fail over on
a throttle (429) -- it returns the throttle so the caller retries the warm rung
after backoff, preserving its prompt cache rather than restarting cold on another
provider -- while STILL failing over on operational deadness
(auth/not-found/5xx/transport) and on a stalled lane (a first-byte or
header-phase timeout that never answered), for which there is no warm cache to
preserve. A genuinely retryable timeout (provider 408) redials the warm rung in
both modes. Client errors reject without failover in both modes.

``maximize_cache_affinity`` keeps availability-style failover (a throttle DOES
fail over: the deterministic alternate builds warm cache instead of the caller
waiting out a backoff) but replaces the certified initial rung order with a
per-request weighted rendezvous hash of the request's stable affinity
fingerprint, so every worker independently sends one conversation to the same
rung and, when that rung sheds or dies, to the same alternate. Rung weights
come from each deployment's authored ``GatewayRungDispatchPolicy``.

Widening this literal is deployment-ordered, like every catalog vocabulary
change: a new value may be AUTHORED only after every serving worker runs a
build that parses it, because an older worker rejects the unknown value at
hydration and fails that alias closed (the per-alias fail-safe excludes the
alias; the worker still serves everything else). The hosted platform is the
only author and its release contract pins this order: engine release, then
fleet-wide pin bump, then the catalog opt-in.
"""


class GatewayRungDispatchPolicy(ContractModel):
    """Authored per-rung dispatch controls: admission bounds and affinity weight.

    Every field defaults to inert so an unauthored rung behaves exactly as
    today: unbounded admission, no fairness accounting, rendezvous weight 1.
    The bound and fairness apply on any pool; the affinity weight is read only
    under a pool's ``maximize_cache_affinity`` policy.
    """

    concurrency_bound: int | None = Field(default=None, ge=1)
    """In-flight dispatch cap for this rung, per gateway worker process.

    Beyond the bound a request spills immediately to the waterfall's next rung
    instead of queueing at the deployment (seconds of spill latency, never a
    deadline death). ``None`` leaves admission unbounded (historical behavior).
    The count is per worker process: the platform authors the per-worker value
    (fleet capacity divided by serving replicas), because enforcement is
    in-memory arithmetic with no shared state on the request path.
    """
    fair_share: bool = False
    """Whether contended admission on this rung is weighted max-min fair.

    When the rung is at or near its ``concurrency_bound``, each organization's
    admissions are limited to its weighted share of the bound (weights ride
    ``AuthorizationSnapshot.fair_share_weight``), with freed capacity reserved
    for recently active under-share organizations. Work-conserving: a lone
    organization borrows the whole bound. Requires ``concurrency_bound``.
    """
    affinity_weight: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """Rendezvous weight under ``maximize_cache_affinity`` (``None`` means 1.0).

    Heavier rungs attract proportionally more affinity fingerprints; authoring
    a heavy house rung and a moderate cheap-cached-input rung makes the house
    box the warm home and the cheap rung the stable spill target.
    """

    @model_validator(mode="after")
    def _require_bound_for_fair_share(self) -> GatewayRungDispatchPolicy:
        """Reject fairness without a capacity: shares need a bound to divide."""
        if self.fair_share and self.concurrency_bound is None:
            raise ValueError("fair_share requires a concurrency_bound to share")
        return self
