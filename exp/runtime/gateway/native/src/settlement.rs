//! Exactly-once settlement of admitted requests and their physical attempts.
//!
//! Every durable accounting write flows through this module: the bounded
//! retry delivery to the control plane, the `AttemptGuard` that owns one
//! request's terminal outcome (including the drop backstop for cancelled
//! handlers and unwound stream tasks), and the shutdown-drain hold that keeps
//! graceful stop waiting for detached stream settlements.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass};
use crate::events::Usage;
use crate::metrics::METRICS;

/// Format one wall-clock instant as an RFC 3339 / ISO 8601 UTC string with
/// millisecond precision, e.g. `2026-08-30T12:34:56.789+00:00`.
///
/// The control plane parses this with `datetime.fromisoformat`, so the output
/// stays inside that grammar (explicit `+00:00` offset, millisecond fraction).
/// The crate carries no datetime dependency, so the civil date is derived from
/// the Unix epoch with Howard Hinnant's `civil_from_days` algorithm; a time
/// before the epoch (never expected for a served token) clamps to the epoch.
fn system_time_to_rfc3339(at: SystemTime) -> String {
    let since_epoch = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since_epoch.as_secs() as i64;
    let millis = since_epoch.subsec_millis();
    let days = secs.div_euclid(86_400);
    let seconds_of_day = secs.rem_euclid(86_400);
    let (hour, minute, second) = (
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60,
        seconds_of_day % 60,
    );
    // civil_from_days: shift the epoch to a 0000-03-01 era so leap handling is
    // branch-free, then recover the Gregorian year/month/day.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097; // [0, 146_096]
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153; // [0, 11], months shifted so March = 0
    let day = day_of_year - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}+00:00")
}

/// Build the settle callback argument shared by explicit settlement and the
/// drop backstop.
#[allow(clippy::too_many_arguments)]
fn settle_argument(
    request_id: &str,
    attempt_id: &str,
    outcome: &str,
    usage: Option<&Usage>,
    tool_names: &[String],
    failure: Option<&Failure>,
    finalize: bool,
    opened: bool,
    first_token_at: Option<SystemTime>,
) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "attempt_id": attempt_id,
        "outcome": outcome,
        "usage": usage.map(|usage| json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        })),
        "tool_names": tool_names,
        "failure": failure.map(|failure| json!({
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
            // The provider's own sanitized rejection sentence (client-error
            // class only); accounting persists it on the failed-attempt row so
            // an operator sees WHY the provider refused the call.
            "provider_detail": failure.provider_detail,
            // The bounded refusal category (refusal class only), so the
            // control plane can count refusals by reason without parsing the
            // free-form provider_detail.
            "refusal_reason": failure.refusal_reason.map(|reason| reason.as_str()),
            // The customer's own BYOK credential/account failure: the ledger
            // files it as the caller's invalid request (see
            // native_accounting.ledger_failure).
            "customer_owned": failure.customer_owned,
        })),
        "finalize": finalize,
        "opened": opened,
        "first_token_at": first_token_at.map(system_time_to_rfc3339),
    }))
}

/// Deliver one control-plane write with bounded backoff; the control plane
/// keeps the in-flight entry on a failed terminal write, so retries can
/// still land. A persistent failure stays latched as accounting-unhealthy
/// control-plane side and is reconciled at the next startup.
async fn deliver(bridge: &Bridge, method: &'static str, argument: String) -> bool {
    for backoff_ms in [0u64, 100, 500, 2_000] {
        if backoff_ms > 0 {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            METRICS.record_settlement_retry();
        }
        if bridge.call(method, argument.clone()).await.is_ok() {
            return true;
        }
    }
    // The control plane keeps the in-flight entry; its sweep keeps retrying
    // and latches readiness if the loss is durable. Leave an operator signal
    // as one structured, content-free stderr line beside the counter.
    METRICS.record_settlement_give_up();
    let parsed: Value = serde_json::from_str(&argument).unwrap_or(Value::Null);
    let line = json!({
        "event": "settlement_give_up",
        "method": method,
        "request_id": parsed.get("request_id").cloned().unwrap_or(Value::Null),
        "attempt_id": parsed.get("attempt_id").cloned().unwrap_or(Value::Null),
        "outcome": parsed.get("outcome").cloned().unwrap_or(Value::Null),
    });
    eprintln!("exp-gateway-native: {line}");
    false
}

/// Exactly-once settlement owner for one admitted request and its physical
/// attempts.
///
/// Every admitted request settles through this guard. Each reserved attempt
/// is bound with `rebind`; a non-finalizing settlement closes that attempt
/// and leaves the request open for the next dispatch. If the owning future
/// is dropped before the terminal settlement lands (client disconnect
/// cancels the handler, a panic unwinds the stream task), `Drop` spawns the
/// closing write so the ledger rows and their budget reservations are always
/// closed: the decided settlement verbatim when delivery was cut short, a
/// cancellation of the active attempt otherwise, or an `abandon` of the
/// accepted request when no attempt is active.
pub struct AttemptGuard {
    pub bridge: Arc<Bridge>,
    request_id: String,
    attempt_id: Option<String>,
    pending: Arc<AtomicUsize>,
    armed: bool,
    outcome_recorded: bool,
    /// Whether the active attempt's provider dispatch opened successfully;
    /// carried into settlement for deployment-health recording.
    opened: bool,
    /// The exact settlement whose delivery is in flight. The drop backstop
    /// re-delivers this decided settlement instead of a cancellation, so a
    /// task cancelled mid-write can neither downgrade the ledger outcome nor
    /// diverge from the recorded metric.
    decided_settlement: Option<String>,
    pub started: Instant,
    /// Wall-clock time the winning attempt streamed its first output token,
    /// reported in the finalizing settlement so the control plane can derive
    /// time-to-first-token. `None` until an attempt observes a first token.
    first_token_at: Option<SystemTime>,
}

/// Holds one unit of the shutdown drain counter for a detached stream task,
/// so graceful shutdown waits (bounded by the graceful timeout) for the
/// task's terminal settlement instead of dropping the runtime under it.
pub struct SettlementTask(Arc<AtomicUsize>);

impl SettlementTask {
    fn hold(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for SettlementTask {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl AttemptGuard {
    /// Count the owning detached task against graceful-shutdown draining.
    pub fn hold_task(&self) -> SettlementTask {
        SettlementTask::hold(self.pending.clone())
    }

    pub fn new(
        bridge: Arc<Bridge>,
        pending: Arc<AtomicUsize>,
        request_id: String,
        started: Instant,
    ) -> Self {
        METRICS.record_served();
        METRICS.enter_request();
        Self {
            bridge,
            request_id,
            attempt_id: None,
            pending,
            armed: true,
            outcome_recorded: false,
            opened: false,
            decided_settlement: None,
            started,
            first_token_at: None,
        }
    }

    /// Bind one freshly reserved attempt as the active settlement target.
    pub fn rebind(&mut self, attempt_id: String) {
        self.attempt_id = Some(attempt_id);
        self.opened = false;
        self.decided_settlement = None;
        // Each physical attempt observes its own first token; a prior failed
        // attempt's timing never carries into its successor.
        self.first_token_at = None;
    }

    /// Record the wall-clock time the active attempt streamed its first output
    /// token, read from its relay just before settlement. Only the first
    /// observation for the attempt is kept.
    pub fn record_first_token(&mut self, at: Option<SystemTime>) {
        if self.first_token_at.is_none() {
            self.first_token_at = at;
        }
    }

    /// Record that the active attempt's provider dispatch opened.
    pub fn mark_opened(&mut self) {
        self.opened = true;
    }

    /// Record this request's terminal outcome and duration exactly once, at
    /// the moment the outcome is decided. Recording happens before delivery
    /// is awaited, so a task cancelled mid-write cannot re-report a decided
    /// outcome as a cancellation.
    fn record_terminal(&mut self, outcome: &str, cancelled: bool) {
        if self.outcome_recorded {
            return;
        }
        self.outcome_recorded = true;
        METRICS.record_outcome(outcome, cancelled);
        METRICS.request_duration_ms.record(self.started.elapsed());
        METRICS.exit_request();
    }

    /// Durably settle the active attempt. A finalizing settlement also
    /// terminalizes the request and disarms the drop backstop; a
    /// non-finalizing one closes only the attempt so the waterfall can
    /// dispatch its successor. Returns whether the write reached the ledger.
    pub async fn settle(
        &mut self,
        outcome: &str,
        usage: Option<&Usage>,
        tool_names: &[String],
        failure: Option<&Failure>,
        finalize: bool,
    ) -> bool {
        let Some(attempt_id) = self.attempt_id.clone() else {
            // No active attempt: nothing durable to close here. The abandon
            // path owns request-only terminalization.
            return true;
        };
        let argument = settle_argument(
            &self.request_id,
            &attempt_id,
            outcome,
            usage,
            tool_names,
            failure,
            finalize,
            self.opened,
            self.first_token_at,
        );
        if finalize {
            let cancelled = failure.map(|failure| failure.failure_class == FailureClass::Cancelled)
                == Some(true);
            self.record_terminal(outcome, cancelled);
        }
        self.decided_settlement = Some(argument.clone());
        let delivered = deliver(&self.bridge, "settle", argument).await;
        if finalize {
            // The control plane retained a failed terminal write verbatim,
            // so its sweep (not the drop backstop) owns the retry.
            self.decided_settlement = None;
            self.armed = false;
        } else if delivered {
            self.decided_settlement = None;
            self.attempt_id = None;
            self.opened = false;
        }
        // A failed non-finalizing delivery keeps the decided settlement
        // armed: the drop backstop re-delivers the ORIGINAL outcome verbatim
        // instead of downgrading the pending provider failure to a
        // cancellation, and the caller treats the failure as fatal so no
        // successor is ever dispatched over an unsettled attempt.
        delivered
    }

    /// Settle the active attempt as cancelled and finalize the request.
    pub async fn settle_cancelled(&mut self, usage: Option<&Usage>, tool_names: &[String]) -> bool {
        if self.attempt_id.is_some() {
            self.settle(
                "failed",
                usage,
                tool_names,
                Some(&Failure::new(
                    FailureClass::Cancelled,
                    "gateway request was cancelled",
                )),
                true,
            )
            .await
        } else {
            self.abandon(&Failure::new(
                FailureClass::Cancelled,
                "gateway request was cancelled",
            ))
            .await
        }
    }

    /// Terminalize an accepted request that has no active attempt.
    pub async fn abandon(&mut self, failure: &Failure) -> bool {
        self.record_terminal("failed", failure.failure_class == FailureClass::Cancelled);
        self.armed = false;
        deliver(
            &self.bridge,
            "abandon",
            compact_json(&json!({
                "request_id": self.request_id,
                "failure": {
                    "failure_class": failure.failure_class.as_str(),
                    "safe_message": failure.safe_message,
                },
            })),
        )
        .await
    }

    /// Disarm the guard after the control plane itself finalized the request
    /// (an exhausted ladder or a terminal budget rejection).
    pub fn disarm_finalized(&mut self, outcome: &str) {
        self.record_terminal(outcome, false);
        self.armed = false;
        self.attempt_id = None;
        self.decided_settlement = None;
    }
}

impl Drop for AttemptGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // A settlement already decided (its delivery was cut short by the
        // cancellation) is re-delivered verbatim, matching the control
        // plane's own never-downgrade sweep semantics; an attempt with no
        // decided outcome settles as cancelled, and an accepted request with
        // no active attempt is abandoned.
        let (method, argument): (&'static str, String) = match self.decided_settlement.take() {
            Some(argument) => ("settle", argument),
            None => match self.attempt_id.take() {
                Some(attempt_id) => {
                    self.record_terminal("failed", true);
                    (
                        "settle",
                        settle_argument(
                            &self.request_id,
                            &attempt_id,
                            "failed",
                            None,
                            &[],
                            Some(&Failure::new(
                                FailureClass::Cancelled,
                                "gateway request was cancelled",
                            )),
                            true,
                            self.opened,
                            self.first_token_at,
                        ),
                    )
                }
                None => {
                    self.record_terminal("failed", true);
                    (
                        "abandon",
                        compact_json(&json!({
                            "request_id": self.request_id,
                            "failure": {
                                "failure_class": "cancelled",
                                "safe_message": "gateway request was cancelled",
                            },
                        })),
                    )
                }
            },
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // Runtime teardown; startup reconciliation closes the row.
            return;
        };
        let bridge = self.bridge.clone();
        let pending = self.pending.clone();
        pending.fetch_add(1, Ordering::SeqCst);
        handle.spawn(async move {
            deliver(&bridge, method, argument).await;
            pending.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_formats_epoch_seconds_and_millis_in_utc() {
        assert_eq!(
            system_time_to_rfc3339(UNIX_EPOCH),
            "1970-01-01T00:00:00.000+00:00"
        );
        // A fixed instant with sub-second precision (1_700_000_000.500s).
        let at = UNIX_EPOCH + Duration::from_millis(1_700_000_000_500);
        assert_eq!(system_time_to_rfc3339(at), "2023-11-14T22:13:20.500+00:00");
        // A leap day exercises the civil-from-days month/day recovery.
        let leap = UNIX_EPOCH + Duration::from_secs(1_582_934_400);
        assert_eq!(
            system_time_to_rfc3339(leap),
            "2020-02-29T00:00:00.000+00:00"
        );
    }

    #[test]
    fn settle_argument_carries_first_token_at_only_when_observed() {
        let observed = UNIX_EPOCH + Duration::from_millis(1_700_000_000_500);
        let with_token = settle_argument(
            "req",
            "att",
            "completed",
            None,
            &[],
            None,
            true,
            true,
            Some(observed),
        );
        let parsed: Value = serde_json::from_str(&with_token).expect("valid json");
        assert_eq!(
            parsed["first_token_at"],
            Value::String("2023-11-14T22:13:20.500+00:00".to_string())
        );
        // A non-streaming attempt observes no first token: the field is null,
        // matching the control plane's backward-compatible parse.
        let without = settle_argument("req", "att", "completed", None, &[], None, true, true, None);
        let parsed: Value = serde_json::from_str(&without).expect("valid json");
        assert_eq!(parsed["first_token_at"], Value::Null);
    }

    #[test]
    fn settle_argument_carries_the_sanitized_provider_detail_on_a_failed_attempt() {
        let failure = Failure::new(FailureClass::InvalidRequest, "provider rejected")
            .with_provider_detail(Some(
                "max_tokens must be greater than thinking budget.".into(),
            ));
        let argument = settle_argument(
            "req",
            "att",
            "failed",
            None,
            &[],
            Some(&failure),
            true,
            true,
            None,
        );
        let parsed: Value = serde_json::from_str(&argument).expect("valid json");
        assert_eq!(parsed["failure"]["failure_class"], "invalid_request");
        assert_eq!(
            parsed["failure"]["provider_detail"],
            "max_tokens must be greater than thinking budget."
        );
        assert_eq!(parsed["failure"]["customer_owned"], false);
        let owned = crate::stream_errors::customer_credential_failure(
            crate::upstream::transport_failure(Some(401)),
            "openai",
        );
        let owned_argument = settle_argument(
            "req",
            "att",
            "failed",
            None,
            &[],
            Some(&owned),
            true,
            true,
            None,
        );
        let parsed: Value = serde_json::from_str(&owned_argument).expect("valid json");
        assert_eq!(
            parsed["failure"]["failure_class"],
            "provider_authentication"
        );
        assert_eq!(parsed["failure"]["customer_owned"], true);
        // A failure with no provider explanation carries an explicit null, which
        // the control plane parses back to None.
        let bare = Failure::new(FailureClass::ProviderInternal, "provider failed");
        let bare_argument = settle_argument(
            "req",
            "att",
            "failed",
            None,
            &[],
            Some(&bare),
            true,
            true,
            None,
        );
        let parsed: Value = serde_json::from_str(&bare_argument).expect("valid json");
        assert_eq!(parsed["failure"]["provider_detail"], Value::Null);
    }
}
