"""In-process bounded and weighted fair-share admission per certified rung.

One registry per worker tracks in-flight dispatches on rungs that author a
``GatewayRungDispatchPolicy``: a rung at its per-worker ``concurrency_bound``
sheds new dispatches down the waterfall (spill in seconds, never a queue that
dies at the request deadline), and a ``fair_share`` rung additionally bounds
each organization's admissions by its weighted max-min share while the rung is
contended. Every decision is lock-guarded in-memory arithmetic over counters
this registry already holds: no database read, no shared state, no waiting.

Fairness is deliberately conservative because there is no queue and no
preemption. Capacity below the bound is always borrowable (work-conserving: a
lone organization uses the whole rung), and an organization above its weighted
share is shed only when admitting it would eat capacity currently reserved for
another RECENTLY ACTIVE under-share organization. A shed marks its organization
active, so a flooded-out organization accrues a reservation and converges to
its share as slots free, without ever pausing a running request.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

# Physical lane identity: the deployment and its credentialed connection.
# Deliberately NOT revision-scoped (unlike health keys): a catalog write does
# not change the box's capacity, so counters must survive alias revisions.
RungLoadKey = tuple[str, str]

RungShedReason = Literal["fair_share_shed", "queue_bound"]

# How long a request (admitted or shed) keeps its organization "active" for
# share accounting. Long enough that a flooded-out caller's retry cadence keeps
# its reservation alive; short enough that a departed caller's share is
# borrowable again almost immediately.
ACTIVITY_WINDOW_SECONDS = 10.0


@dataclass(frozen=True)
class RungShed:
    """One refused reservation and the disclosure reason for the bypass."""

    reason: RungShedReason


@dataclass
class _OrganizationLoad:
    """Per-organization in-flight count and recency on one rung."""

    inflight: int = 0
    last_seen: float = 0.0
    weight: int = 1


@dataclass
class _RungLoad:
    """Aggregate and per-organization in-flight state for one rung."""

    total: int = 0
    organizations: dict[str, _OrganizationLoad] = field(default_factory=dict)


class RungLoadRegistry:
    """Reservation ledger for bounded and fair-share rung admission.

    A successful reservation returns an opaque ticket the caller either binds
    to the durable attempt id (released later by ``release_attempt``) or
    releases directly when the dispatch never happened. Both release paths are
    idempotent so settle, abandon, and the sweep can all safely fire.
    """

    def __init__(
        self,
        *,
        activity_window_seconds: float = ACTIVITY_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize empty counters with an injectable clock for tests.

        Args:
            activity_window_seconds: Recency horizon for share reservations.
            clock: Monotonic clock.

        Raises:
            ValueError: The activity window is not positive.
        """
        if activity_window_seconds <= 0:
            raise ValueError("activity window must be positive")
        self._window = activity_window_seconds
        self._clock = clock
        self._rungs: dict[RungLoadKey, _RungLoad] = {}
        self._tickets: dict[str, tuple[RungLoadKey, str]] = {}
        self._attempts: dict[str, str] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        key: RungLoadKey,
        *,
        organization_id: str,
        weight: int,
        bound: int,
        fair_share: bool,
        force: bool = False,
    ) -> str | RungShed:
        """Reserve one in-flight slot on a bounded rung, or shed with a reason.

        Args:
            key: Physical rung identity.
            organization_id: Authorized organization for share accounting.
            weight: The organization's fair-share weight (>= 1).
            bound: The rung's authored per-worker in-flight cap.
            fair_share: Whether contended admission is weighted max-min fair.
            force: Admit past the bound (the caller proved no other rung can
                serve; the bound must never manufacture a failure).

        Returns:
            An opaque ticket on admission, else the shed disclosure.
        """
        now = self._clock()
        with self._lock:
            rung = self._rungs.setdefault(key, _RungLoad())
            organization = rung.organizations.setdefault(organization_id, _OrganizationLoad())
            # A shed still marks demand: the flooded-out organization's share
            # is reserved as slots free, which is what converges to fairness.
            organization.last_seen = now
            organization.weight = weight
            self._prune(key, rung, now)
            if not force:
                shed = self._shed_reason(rung, organization, bound=bound, fair_share=fair_share)
                if shed is not None:
                    return shed
            organization.inflight += 1
            rung.total += 1
            ticket = f"rung-{uuid.uuid4().hex}"
            self._tickets[ticket] = (key, organization_id)
            return ticket

    def _shed_reason(
        self,
        rung: _RungLoad,
        organization: _OrganizationLoad,
        *,
        bound: int,
        fair_share: bool,
    ) -> RungShed | None:
        """Decide one reservation under the registry lock; ``None`` admits.

        The bound is hard: at or beyond it every arrival spills, which is the
        queue-death fix. Below it, fairness sheds an over-share organization
        only when the remaining slots are reserved for other recently active
        under-share organizations; otherwise unused capacity is borrowable.
        """
        if rung.total >= bound:
            return RungShed("queue_bound")
        if not fair_share:
            return None
        floor = organization.last_seen - self._window
        active = [
            candidate
            for candidate in rung.organizations.values()
            if candidate.inflight > 0 or candidate.last_seen >= floor
        ]
        total_weight = sum(candidate.weight for candidate in active)
        share = bound * organization.weight / total_weight
        if organization.inflight + 1 <= share:
            return None
        reserved = sum(
            max(0.0, bound * candidate.weight / total_weight - candidate.inflight)
            for candidate in active
            if candidate is not organization
        )
        if rung.total + 1 + reserved > bound:
            return RungShed("fair_share_shed")
        return None

    def bind(self, ticket: str, attempt_id: str) -> None:
        """Attach one reservation to its durable attempt for settle release.

        Args:
            ticket: Reservation returned by :meth:`reserve`.
            attempt_id: The durably reserved attempt now holding the slot.
        """
        with self._lock:
            if ticket in self._tickets:
                self._attempts[attempt_id] = ticket

    def release_ticket(self, ticket: str) -> None:
        """Release one reservation that never dispatched; idempotent.

        Args:
            ticket: Reservation returned by :meth:`reserve`.
        """
        with self._lock:
            self._release_locked(ticket)

    def release_attempt(self, attempt_id: str) -> None:
        """Release the reservation held by one settled attempt; idempotent.

        Args:
            attempt_id: The settled or abandoned attempt.
        """
        with self._lock:
            ticket = self._attempts.pop(attempt_id, None)
            if ticket is not None:
                self._release_locked(ticket)

    def inflight(self, key: RungLoadKey, organization_id: str | None = None) -> int:
        """Return one rung's total or per-organization in-flight count.

        Args:
            key: Physical rung identity.
            organization_id: Scope the count to one organization when given.
        """
        with self._lock:
            rung = self._rungs.get(key)
            if rung is None:
                return 0
            if organization_id is None:
                return rung.total
            organization = rung.organizations.get(organization_id)
            return 0 if organization is None else organization.inflight

    def _release_locked(self, ticket: str) -> None:
        """Return one reserved slot to its rung under the registry lock."""
        entry = self._tickets.pop(ticket, None)
        if entry is None:
            return
        key, organization_id = entry
        rung = self._rungs.get(key)
        if rung is None:
            return
        organization = rung.organizations.get(organization_id)
        if organization is not None and organization.inflight > 0:
            organization.inflight -= 1
        if rung.total > 0:
            rung.total -= 1

    def _prune(self, key: RungLoadKey, rung: _RungLoad, now: float) -> None:
        """Drop idle organizations past the activity window, bounding memory."""
        stale = [
            organization_id
            for organization_id, load in rung.organizations.items()
            if load.inflight == 0 and load.last_seen < now - self._window
        ]
        for organization_id in stale:
            del rung.organizations[organization_id]
        if rung.total == 0 and not rung.organizations:
            self._rungs.pop(key, None)
