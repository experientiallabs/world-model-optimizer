"""Fairness, bound, work-conservation, and release tests for rung admission."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed

_KEY = ("dep-house", "connection-sha")


def _registry(now: list[float]) -> RungLoadRegistry:
    """Build a registry on a mutable fake clock."""
    return RungLoadRegistry(activity_window_seconds=10.0, clock=lambda: now[0])


def _reserve(
    registry: RungLoadRegistry,
    organization_id: str,
    *,
    weight: int = 1,
    bound: int = 4,
    fair_share: bool = False,
    force: bool = False,
) -> str | RungShed:
    """Reserve one slot on the shared test rung."""
    return registry.reserve(
        _KEY,
        organization_id=organization_id,
        weight=weight,
        bound=bound,
        fair_share=fair_share,
        force=force,
    )


class TestConcurrencyBound:
    """The per-worker bound sheds instead of queueing, and frees on release."""

    def test_admits_below_the_bound_and_sheds_at_it(self) -> None:
        """Slots below the bound admit; the arrival at the bound spills."""
        registry = _registry([0.0])
        tickets = [_reserve(registry, "org-a") for _ in range(4)]
        assert all(isinstance(ticket, str) for ticket in tickets)
        shed = _reserve(registry, "org-a")
        assert shed == RungShed("queue_bound")
        assert registry.inflight(_KEY) == 4

    def test_release_frees_the_slot(self) -> None:
        """A settled dispatch returns its slot to new arrivals."""
        registry = _registry([0.0])
        tickets = [_reserve(registry, "org-a") for _ in range(4)]
        first = tickets[0]
        assert isinstance(first, str)
        registry.bind(first, "attempt-1")
        registry.release_attempt("attempt-1")
        assert isinstance(_reserve(registry, "org-b"), str)

    def test_force_admits_past_the_bound(self) -> None:
        """A caller with no other serviceable rung dispatches over the bound."""
        registry = _registry([0.0])
        for _ in range(4):
            assert isinstance(_reserve(registry, "org-a"), str)
        assert isinstance(_reserve(registry, "org-a", force=True), str)
        assert registry.inflight(_KEY) == 5

    def test_releases_are_idempotent(self) -> None:
        """Settle, abandon, and the sweep can all release without corruption."""
        registry = _registry([0.0])
        ticket = _reserve(registry, "org-a")
        assert isinstance(ticket, str)
        registry.bind(ticket, "attempt-1")
        registry.release_attempt("attempt-1")
        registry.release_attempt("attempt-1")
        registry.release_ticket(ticket)
        assert registry.inflight(_KEY) == 0

    def test_unbound_ticket_release_covers_failed_reservations(self) -> None:
        """A ledger write that raised releases its slot without an attempt id."""
        registry = _registry([0.0])
        ticket = _reserve(registry, "org-a")
        assert isinstance(ticket, str)
        registry.release_ticket(ticket)
        assert registry.inflight(_KEY) == 0


class TestFairShare:
    """Weighted max-min admission: shares under contention, borrow when idle."""

    def test_lone_organization_borrows_the_whole_bound(self) -> None:
        """Work-conserving: no other active organization means no reservation."""
        registry = _registry([0.0])
        for _ in range(8):
            assert isinstance(
                _reserve(registry, "org-a", bound=8, fair_share=True),
                str,
            )
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("queue_bound")

    def test_active_underuser_reserves_its_share_from_a_borrower(self) -> None:
        """Freed capacity goes to the recently active under-share organization."""
        now = [0.0]
        registry = _registry(now)
        # org-a floods the rung to its bound.
        tickets = [_reserve(registry, "org-a", bound=8, fair_share=True) for _ in range(8)]
        # org-b arrives at the full rung: shed, but now recorded as demanding.
        assert isinstance(
            _reserve(registry, "org-b", bound=8, fair_share=True),
            RungShed,
        )
        # One org-a slot frees; org-a is over its 4-slot share and the free
        # capacity is reserved for org-b, so org-a is shed by fairness...
        first = tickets[0]
        assert isinstance(first, str)
        registry.release_ticket(first)
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("fair_share_shed")
        # ...while org-b, under its share, admits.
        assert isinstance(_reserve(registry, "org-b", bound=8, fair_share=True), str)

    def test_weights_scale_the_guaranteed_share(self) -> None:
        """A 3:1 weighting guarantees the heavy organization 6 of 8 slots."""
        now = [0.0]
        registry = _registry(now)
        light = [
            _reserve(registry, "org-light", weight=1, bound=8, fair_share=True) for _ in range(8)
        ]
        assert all(isinstance(ticket, str) for ticket in light)
        # The heavy organization arrives; as light slots free one by one, the
        # light organization is denied above its 2-slot share while the heavy
        # one climbs toward 6.
        assert isinstance(
            _reserve(registry, "org-heavy", weight=3, bound=8, fair_share=True),
            RungShed,
        )
        admitted_heavy = 0
        for ticket in light:
            assert isinstance(ticket, str)
            registry.release_ticket(ticket)
            light_retry = _reserve(registry, "org-light", weight=1, bound=8, fair_share=True)
            if isinstance(light_retry, str):
                registry.release_ticket(light_retry)
            heavy = _reserve(registry, "org-heavy", weight=3, bound=8, fair_share=True)
            if isinstance(heavy, str):
                admitted_heavy += 1
        assert admitted_heavy == 6
        assert registry.inflight(_KEY, "org-heavy") == 6

    def test_idle_organization_past_the_window_frees_its_reservation(self) -> None:
        """A departed organization's share becomes borrowable again."""
        now = [0.0]
        registry = _registry(now)
        tickets = [_reserve(registry, "org-a", bound=8, fair_share=True) for _ in range(8)]
        assert isinstance(_reserve(registry, "org-b", bound=8, fair_share=True), RungShed)
        first = tickets[0]
        assert isinstance(first, str)
        registry.release_ticket(first)
        # While org-b is recently active its share is reserved from org-a.
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("fair_share_shed")
        # org-b never returns; past the activity window the freed slot is
        # borrowable by org-a again (work-conserving, no standing reservation).
        now[0] = 11.0
        assert isinstance(_reserve(registry, "org-a", bound=8, fair_share=True), str)

    def test_no_preemption_running_reservations_always_survive(self) -> None:
        """Fairness never revokes a held ticket; only new admissions are shed."""
        now = [0.0]
        registry = _registry(now)
        tickets = [_reserve(registry, "org-a", bound=8, fair_share=True) for _ in range(8)]
        assert registry.inflight(_KEY, "org-a") == 8
        # A higher-weight arrival is shed and changes nothing about the eight.
        assert isinstance(
            _reserve(registry, "org-b", weight=100, bound=8, fair_share=True),
            RungShed,
        )
        assert registry.inflight(_KEY, "org-a") == 8
        assert all(isinstance(ticket, str) for ticket in tickets)


class TestRegistryContracts:
    """Construction and counter contracts."""

    def test_rejects_a_nonpositive_activity_window(self) -> None:
        """The recency horizon must be positive."""
        with pytest.raises(ValueError, match="activity window"):
            RungLoadRegistry(activity_window_seconds=0.0)

    def test_inflight_reads_are_scoped(self) -> None:
        """Totals and per-organization counts stay consistent."""
        registry = _registry([0.0])
        assert registry.inflight(_KEY) == 0
        assert isinstance(_reserve(registry, "org-a"), str)
        assert isinstance(_reserve(registry, "org-b"), str)
        assert registry.inflight(_KEY) == 2
        assert registry.inflight(_KEY, "org-a") == 1
        assert registry.inflight(("other", "connection"), "org-a") == 0
