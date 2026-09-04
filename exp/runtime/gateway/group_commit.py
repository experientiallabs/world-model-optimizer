"""Group-commit write-behind ledger batching durable writes on one writer thread.

Concurrent request-path ledger writes are funneled through a single dedicated
writer thread that drains queued operations into one SQLite transaction per
batch. Every caller awaits its own operation's durable commit before
proceeding, so acceptance, budget reservation, and terminal settlement keep
their exact fail-closed semantics while the per-request fsync cost is
amortized across all operations sharing a batch. There is no flush window:
no caller observes success before its write is durable on disk.

Two callers share the one queue and writer thread: the asyncio engine awaits
:class:`GroupCommitAttemptLedger`, while threads without an event loop (the
native data plane's worker threads) block on :class:`SyncGroupCommitLedger`.
Operations from both interleave in the same batches, so the two gateway
engines amortize their fsyncs together.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar, cast

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayFailure,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.sqlite.migrations import connect_database

_logger = logging.getLogger(__name__)

_DEFAULT_MAX_BATCH_SIZE = 128

_T = TypeVar("_T")


def _resolve_result(future: concurrent.futures.Future[object], value: object) -> None:
    """Resolve one pending future, tolerating an already-cancelled caller.

    Args:
        future: Caller future for one committed operation.
        value: Durably committed operation result.
    """
    if future.set_running_or_notify_cancel():
        future.set_result(value)


def _resolve_exception(future: concurrent.futures.Future[object], error: BaseException) -> None:
    """Fail one pending future, tolerating an already-cancelled caller.

    Args:
        future: Caller future for one failed or rolled-back operation.
        error: Original operation or commit failure.
    """
    if future.set_running_or_notify_cancel():
        future.set_exception(error)


async def abandoned_write_outcome[R](write: asyncio.Task[R]) -> R | None:
    """Wait out a cancellation-abandoned durable write and return its result.

    A shielded ledger write keeps running on the writer thread after the
    awaiting task is cancelled, so callers that must know whether the write
    committed (for example to settle a durably reserved attempt) wait here
    for its real outcome, absorbing repeated cancellation of the waiter.

    Args:
        write: The in-flight write task, which is never itself cancelled here.

    Returns:
        The committed write result, or None when the write failed or the
        task was cancelled before its operation was enqueued.
    """
    while not write.done():
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - the write failure is read below.
            break
    if write.cancelled() or write.exception() is not None:
        return None
    return write.result()


@dataclass(frozen=True)
class _PendingWrite:
    """One queued ledger operation and the future resolved after durable commit."""

    apply: Callable[[sqlite3.Connection], object]
    future: concurrent.futures.Future[object]


class GroupCommitAttemptLedger:
    """Async attempt ledger committing queued writes in shared durable batches.

    Each public method enqueues one operation and resolves only after the
    writer thread has committed the batch containing it, so callers keep
    write-through durability while concurrent requests share one fsync.
    A failed operation rolls back to its own savepoint without disturbing
    the surrounding batch, and its original exception is re-raised to the
    awaiting caller.
    """

    def __init__(
        self,
        core: SQLiteAttemptLedger,
        *,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        """Start the single writer thread over an existing synchronous ledger.

        Args:
            core: Synchronous SQLite ledger owning schema and write logic.
            max_batch_size: Maximum queued operations drained into one commit.
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")
        self.core = core
        self._max_batch_size = max_batch_size
        self._queue: queue.SimpleQueue[_PendingWrite | None] = queue.SimpleQueue()
        self._closed = False
        # Serializes enqueue against close so no operation can land behind the
        # stop sentinel and strand its caller after the writer thread exits.
        self._submit_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="gateway-ledger-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def closed(self) -> bool:
        """Return whether this writer can no longer make any write durable.

        The flag latches on explicit :meth:`close` and when the writer thread
        exits on an unexpected error, so composition health surfaces can fail
        readiness closed instead of queueing writes that only fail per call.
        """
        return self._closed

    async def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Durably persist accepted authority before route selection or dispatch.

        Args:
            authorization: Frozen authority and request identity.
        """
        await self._submit(
            lambda connection: self.core.apply_accept_request(
                connection, authorization=authorization
            )
        )

    async def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> AttemptId:
        """Durably reserve budget and record one dispatch before provider work.

        Args:
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_micro_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
            dispatch_reason: Optional policy-dispatch disclosure code.
            preferred_deployment: The route's bypassed preferred rung, when divergent.

        Returns:
            Stable new attempt ID.
        """
        return await self._submit(
            lambda connection: self.core.apply_start_attempt(
                connection,
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_micro_usd=maximum_cost_micro_usd,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
                dispatch_reason=dispatch_reason,
                preferred_deployment=preferred_deployment,
            )
        )

    async def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Durably settle one attempt with normalized content-free fields.

        Args:
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
            first_token_at: Wall-clock time the attempt streamed its first token, or ``None``.
        """
        await self._submit(
            lambda connection: self.core.apply_finish_attempt(
                connection,
                attempt_id=attempt_id,
                terminal_event=terminal_event,
                failure=failure,
                finalize_request=finalize_request,
                first_token_at=first_token_at,
            )
        )

    async def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Durably terminalize accepted work that never reached dispatch.

        Args:
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
        """
        await self._submit(
            lambda connection: self.core.apply_finish_request(
                connection, authorization=authorization, failure=failure
            )
        )

    async def flush(self) -> None:
        """Resolve after every previously enqueued operation is durably committed."""
        await self._submit(lambda connection: None)

    def close(self) -> None:
        """Stop the writer thread after draining every queued operation.

        Operations enqueued before close still commit durably; the enqueue
        lock guarantees nothing can be queued behind the stop sentinel, and
        every later submission fails fast with a closed-writer error.
        """
        with self._submit_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._thread.join(timeout=30)

    def _run(self) -> None:
        """Drain queued operations into one durable SQLite transaction per batch."""
        try:
            connection = connect_database(
                self.core.database_path,
                busy_timeout_ms=self.core.busy_timeout_ms,
            )
        except Exception:  # noqa: BLE001 - every blocked caller receives the closed error.
            _logger.exception("gateway ledger writer failed to open its connection")
            self._fail_pending()
            return
        try:
            stopping = False
            while not stopping:
                item = self._queue.get()
                if item is None:
                    break
                batch = [item]
                while len(batch) < self._max_batch_size:
                    try:
                        extra = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        stopping = True
                        break
                    batch.append(extra)
                try:
                    self._commit_batch(connection, batch)
                except Exception as exc:  # noqa: BLE001 - blocked callers receive the failure.
                    _logger.exception("gateway ledger batch failed outside its own transaction")
                    for pending in batch:
                        if not pending.future.done():
                            _resolve_exception(pending.future, exc)
                    if connection.in_transaction:
                        try:
                            connection.execute("ROLLBACK")
                        except sqlite3.Error:
                            _logger.exception(
                                "gateway ledger batch rollback failed after batch failure"
                            )
        finally:
            connection.close()
            self._fail_pending()

    def _fail_pending(self) -> None:
        """Latch closed and fail every still-queued operation with a clear error.

        Runs on the writer thread as it exits, whether by graceful close or by
        an unexpected failure, so no blocked caller (async or sync) can be
        stranded waiting on an operation the writer will never commit.
        """
        with self._submit_lock:
            self._closed = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    return
                if item is None:
                    continue
                _resolve_exception(item.future, _closed_error())

    @staticmethod
    def _commit_batch(
        connection: sqlite3.Connection,
        batch: list[_PendingWrite],
    ) -> None:
        """Apply each queued operation under its own savepoint and commit once.

        A failing operation rolls back to its savepoint so its effects never
        commit, while sibling operations in the same batch stay intact. Only
        after the shared COMMIT succeeds are successful futures resolved, so a
        caller never observes success for a write that is not durable.

        Args:
            connection: Writer-owned configured SQLite connection.
            batch: Queued operations resolved after this shared commit.
        """
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            for pending in batch:
                _resolve_exception(pending.future, exc)
            return
        outcomes: list[tuple[_PendingWrite, object, BaseException | None]] = []
        for index, pending in enumerate(batch):
            savepoint = f"gateway_op_{index}"
            connection.execute(f"SAVEPOINT {savepoint}")
            try:
                value = pending.apply(connection)
            except Exception as exc:  # noqa: BLE001 - the caller re-raises its own failure.
                connection.execute(f"ROLLBACK TO {savepoint}")
                connection.execute(f"RELEASE {savepoint}")
                outcomes.append((pending, None, exc))
            else:
                connection.execute(f"RELEASE {savepoint}")
                outcomes.append((pending, value, None))
        try:
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                _logger.exception("gateway ledger batch rollback failed after commit failure")
            for pending in batch:
                _resolve_exception(pending.future, exc)
            return
        for pending, value, error in outcomes:
            if error is not None:
                _resolve_exception(pending.future, error)
            else:
                _resolve_result(pending.future, value)

    def submit_blocking(self, apply: Callable[[sqlite3.Connection], _T]) -> _T:
        """Enqueue one operation and block the calling thread on its durable commit.

        For threads with no running event loop, such as the native data
        plane's worker threads. The operation shares the queue and writer
        thread with async submissions, so both engines' writes interleave in
        the same batches. The writer is a separate dedicated thread, so
        blocking here can never deadlock a caller.

        Args:
            apply: Operation run on the writer connection inside the batch transaction.

        Returns:
            The operation's return value after its batch has committed.

        Raises:
            RuntimeError: The writer is closed, before or while waiting.
            Exception: The operation itself failed and was rolled back.
        """
        return self._enqueue(apply).result()

    async def _submit(self, apply: Callable[[sqlite3.Connection], _T]) -> _T:
        """Enqueue one operation and await its durable batch commit.

        The queued write is shielded from caller cancellation: a cancelled
        request task stops waiting, but the operation still commits durably
        and the writer resolves its future without error.

        Args:
            apply: Operation run on the writer connection inside the batch transaction.

        Returns:
            The operation's return value after its batch has committed.
        """
        return await asyncio.shield(asyncio.wrap_future(self._enqueue(apply)))

    def _enqueue(self, apply: Callable[[sqlite3.Connection], _T]) -> concurrent.futures.Future[_T]:
        """Queue one operation for the writer thread and return its commit future.

        Args:
            apply: Operation run on the writer connection inside the batch transaction.

        Returns:
            Future resolved only after the operation's batch has committed.

        Raises:
            RuntimeError: The writer is closed and can accept no operation.
        """
        future: concurrent.futures.Future[_T] = concurrent.futures.Future()
        with self._submit_lock:
            if self._closed:
                raise _closed_error()
            self._queue.put(
                _PendingWrite(
                    apply=apply,
                    future=cast("concurrent.futures.Future[object]", future),
                )
            )
        return future


def _closed_error() -> RuntimeError:
    """Build the error every closed-writer submission or stranded wait receives."""
    return RuntimeError(
        "gateway ledger writer is closed; the gateway is shutting down, retry after restart"
    )


class SyncGroupCommitLedger:
    """Blocking attempt-ledger facade over one shared group-commit writer.

    Mirrors the async ledger surface with plain synchronous methods for
    callers that run on threads without an event loop (the native data
    plane's worker threads and its settlement sweep). Every method enqueues
    onto the same queue and writer thread as the async path, so operations
    from both engines share batches, and returns only after its own
    operation's batch is durable on disk. A failed or rolled-back operation
    raises its original exception; a closed writer raises a clear
    RuntimeError instead of blocking forever.
    """

    def __init__(self, writer: GroupCommitAttemptLedger) -> None:
        """Bind the shared group-commit writer.

        Args:
            writer: The engine-shared batching writer to submit through.
        """
        self._writer = writer

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Durably persist accepted authority before route selection or dispatch.

        Args:
            authorization: Frozen authority and request identity.
        """
        self._writer.submit_blocking(
            lambda connection: self._writer.core.apply_accept_request(
                connection, authorization=authorization
            )
        )

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> AttemptId:
        """Durably reserve budget and record one dispatch before provider work.

        Args:
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_micro_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
            dispatch_reason: Optional policy-dispatch disclosure code.
            preferred_deployment: The route's bypassed preferred rung, when divergent.

        Returns:
            Stable new attempt ID.
        """
        return self._writer.submit_blocking(
            lambda connection: self._writer.core.apply_start_attempt(
                connection,
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_micro_usd=maximum_cost_micro_usd,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
                dispatch_reason=dispatch_reason,
                preferred_deployment=preferred_deployment,
            )
        )

    def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Durably settle one attempt with normalized content-free fields.

        Args:
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
            first_token_at: Wall-clock time the attempt streamed its first token, or ``None``.
        """
        self._writer.submit_blocking(
            lambda connection: self._writer.core.apply_finish_attempt(
                connection,
                attempt_id=attempt_id,
                terminal_event=terminal_event,
                failure=failure,
                finalize_request=finalize_request,
                first_token_at=first_token_at,
            )
        )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Durably terminalize accepted work that never reached dispatch.

        Args:
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
        """
        self._writer.submit_blocking(
            lambda connection: self._writer.core.apply_finish_request(
                connection, authorization=authorization, failure=failure
            )
        )

    def flush(self) -> None:
        """Return after every previously enqueued operation is durably committed."""
        self._writer.submit_blocking(lambda connection: None)
