//! Native execution of the certified deployment waterfall.
//!
//! The control plane's `admit` returns the full ordered route (one wire
//! configuration per certified deployment) plus the frozen retry policy
//! facts; this module loops physical dispatches under the request deadline,
//! mirroring the python executor's semantics: each dispatch is durably
//! reserved through the `start_attempt` bridge callback immediately before
//! network work, same-deployment redials happen only for retryable failure
//! classes and only before commitment, failover advances to the next
//! certified deployment for failover-eligible failures before commitment,
//! and the first outward semantic event permanently freezes the serving
//! deployment. When the alias revision enables refusal failover, refusal
//! deltas are withheld in a bounded in-memory buffer so a refusal-only
//! terminal can advance to the next deployment without exposing the refused
//! route; mixed output or buffer overflow commits and flushes. Candidate
//! selection policy (health circuits, budgets, attempt counting) stays in
//! python: the loop only states its position and the classified failure, and
//! the control plane answers with a reservation, a later depth, or
//! exhaustion.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::dialects::Dialect;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::relay::{
    collection_public_error, ended_without_terminal, remaining, track_event, UpstreamRelay,
};
use crate::settlement::AttemptGuard;
use crate::upstream::open_stream;

/// Byte bound for withheld refusal deltas, matching the python executor's
/// `_MAX_WITHHELD_REFUSAL_BYTES`.
pub const MAXIMUM_WITHHELD_REFUSAL_BYTES: usize = 65_536;

/// Event-count bound for withheld refusal deltas, matching the python
/// executor's `_MAX_WITHHELD_REFUSAL_EVENTS`.
pub const MAXIMUM_WITHHELD_REFUSAL_EVENTS: usize = 256;

/// One deployment's wire configuration inside the admitted ordered route.
/// Payloads are built python-side per deployment, since model identities and
/// dialects may differ across one certified pool.
#[derive(Debug, Clone, Deserialize)]
pub struct DeploymentWire {
    pub provider: String,
    pub deployment_id: String,
    pub dialect: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    /// Exact provider model identifier the payload carries; a caller-known
    /// word the stream-error detail screen must not redact.
    #[serde(default)]
    pub model_id: String,
    /// Whether this rung dispatches on the customer's own (BYOK) credential.
    /// A rejected credential or exhausted account on such a rung is the
    /// customer's configuration, surfaced as their 400, never operator
    /// deadness that fails over.
    #[serde(default)]
    pub billing_customer_managed: bool,
    pub timeout_seconds: f64,
    /// Structured payload the data plane serializes itself; null for
    /// body-signing dialects, whose route entry carries `upstream_body`.
    #[serde(default)]
    pub upstream_payload: Value,
    /// Exact pre-serialized body for body-signing dialects (Bedrock SigV4).
    /// When present it is sent verbatim: the signature covers these exact
    /// bytes, so re-serializing a structured payload here could invalidate
    /// it.
    #[serde(default)]
    pub upstream_body: Option<String>,
    #[serde(default)]
    pub fireworks_reasoning_route_sha256: Option<String>,
    /// Tencent Hunyuan preserved-thinking route identity; like the Fireworks
    /// field it turns on provider reasoning-content capture and stamps each
    /// delta with this exact route, but seals under the Hunyuan carrier scheme.
    #[serde(default)]
    pub hunyuan_reasoning_route_sha256: Option<String>,
    /// When true, this rung returns the model's plaintext `reasoning_content`
    /// to the caller for display (Tencent/DeepSeek think mode). The sealed
    /// round-trip carrier is emitted independently; elsewhere reasoning stays
    /// stripped. Defaults false so every other provider is unchanged.
    #[serde(default)]
    pub reasoning_output_exposed: bool,
    /// Caller stop sequences the data plane enforces on this rung's stream
    /// because the provider wire has no stop field (OpenAI Responses). The
    /// relay cuts visible text at the first match and terminates with
    /// `Event::StoppedAtSequence`. Empty when the payload carries `stop`.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// The caller sent `parallel_tool_calls: false` and this rung's wire has
    /// no such control: the relay serializes the turn to one tool call.
    #[serde(default)]
    pub serialize_tool_calls: bool,
    pub idempotency_key: String,
    /// Deployment override for the flat first-byte allowance; the serving
    /// configuration's default applies when absent.
    #[serde(default)]
    pub time_to_first_byte_base_seconds: Option<f64>,
    /// Deployment override for the input-scaled first-byte allowance in
    /// seconds per million approximate input tokens; the serving
    /// configuration's default applies when absent.
    #[serde(default)]
    pub time_to_first_byte_seconds_per_million_input_tokens: Option<f64>,
}

/// The frozen retry-policy facts returned by admission.
#[derive(Debug, Clone, Copy)]
pub struct RoutePolicy {
    pub maximum_total_attempts: u32,
    pub maximum_same_deployment_attempts: u32,
    pub refusal_failover: bool,
}

/// Everything one waterfall run needs besides its request guard.
pub struct WaterfallContext<'a> {
    pub bridge: &'a Arc<Bridge>,
    pub http: &'a reqwest::Client,
    pub request_id: &'a str,
    /// The presented virtual key, forwarded so hosted budget-error policy
    /// can shape a rejected reservation for the caller.
    pub raw_key: &'a str,
    pub route: &'a [DeploymentWire],
    pub policy: RoutePolicy,
    pub deadline: Instant,
    /// Fail-fast flat bound on the wait for each physical attempt's first
    /// provider byte. Applied per attempt (each redial and each failover
    /// advance gets a fresh window); it bounds the connect/header/first-byte
    /// phase only and never caps generation once the provider has started
    /// answering. Deployments may override it per wire entry.
    pub time_to_first_byte: Duration,
    /// Default input-scaled first-byte allowance in seconds per million
    /// approximate input tokens, so a very large prompt whose prefill
    /// legitimately takes longer than the flat bound is not misread as a
    /// dead lane. Deployments may override it per wire entry.
    pub time_to_first_byte_slope_seconds_per_million_input_tokens: f64,
    /// Approximate input tokens for this request: the raw body's bytes
    /// divided by four. An allowance heuristic only, never a billing
    /// quantity.
    pub approximate_input_tokens: f64,
    /// The bridge `remember` argument retaining an output-less turn: a
    /// successful terminal reached before any semantic output still answers
    /// the caller with a response id, and a response id the caller received
    /// must stay continuable (api.openai.com persists `incomplete` responses
    /// too). Only the Responses route carries one; it runs ahead of the
    /// attempt's settlement so the control plane can still resolve the
    /// request's continuation context.
    pub output_less_retention: Option<String>,
}

/// The effective first-byte allowance for one attempt: the deployment's (or
/// serving default's) flat base plus its input-scaled allowance.
pub(crate) fn first_byte_allowance(
    wire: &DeploymentWire,
    default_base: Duration,
    default_slope_seconds_per_million: f64,
    approximate_input_tokens: f64,
) -> Duration {
    let base = wire
        .time_to_first_byte_base_seconds
        .unwrap_or(default_base.as_secs_f64());
    let slope = wire
        .time_to_first_byte_seconds_per_million_input_tokens
        .unwrap_or(default_slope_seconds_per_million);
    let scaled = slope * (approximate_input_tokens.max(0.0) / 1_000_000.0);
    Duration::from_secs_f64((base + scaled).max(0.001))
}

/// The winning outcome of one waterfall run.
pub enum Won {
    /// A deployment committed: its outward prefix is decided and the live
    /// relay continues the same physical attempt.
    Committed(Box<CommittedAttempt>),
    /// The attempt reached a terminal before commitment and is already
    /// durably settled; `events` are the decided outward events.
    Settled(SettledAttempt),
    /// The ladder is exhausted (or accounting failed); the request is
    /// finalized and this public error answers the caller.
    Failed(PublicError),
}

/// One committed physical attempt with its live upstream relay.
pub struct CommittedAttempt {
    pub depth: usize,
    pub prefix: Vec<Event>,
    pub relay: UpstreamRelay,
    pub usage: Option<Usage>,
    pub tool_names: Vec<String>,
    /// Whether refusal deltas already reached (or will reach) the caller;
    /// a later typed refusal terminal then completes instead of failing.
    pub visible_refusal: bool,
}

/// One attempt whose terminal was reached and settled before commitment.
pub struct SettledAttempt {
    pub depth: usize,
    pub events: Vec<Event>,
}

fn is_semantic(event: &Event) -> bool {
    matches!(
        event,
        Event::TextDelta(_)
            | Event::RefusalDelta(_)
            | Event::ProviderTextDelta { .. }
            | Event::ProviderRefusalDelta { .. }
            | Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. }
            | Event::ReasoningContentDelta { .. }
            | Event::ToolCallStarted { .. }
            | Event::ToolArgumentsDelta { .. }
            | Event::ToolCallCompleted { .. }
            | Event::TextBlockStarted { .. }
            | Event::CitationDelta { .. }
            | Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. }
            | Event::ProviderTextAnnotation { .. }
    )
}

/// The control plane's answer to one `start_attempt` callback.
#[derive(Debug, Deserialize)]
pub(crate) struct StartResponse {
    #[serde(default)]
    pub(crate) attempt_id: Option<String>,
    #[serde(default)]
    pub(crate) route_depth: Option<usize>,
    #[serde(default)]
    pub(crate) exhausted: bool,
    #[serde(default)]
    pub(crate) failure: Option<Failure>,
}

/// Whether the classified failure leaves any successor dispatch possible
/// under the rust-side facts (caps, flags, remaining route, deadline). The
/// control plane re-checks with health and budget state and may still answer
/// with exhaustion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn successor_possible(
    policy: RoutePolicy,
    route_length: usize,
    deadline: Instant,
    total_attempts: u32,
    same_deployment_attempts: u32,
    depth: usize,
    failure: &Failure,
    refusal_eligible: bool,
) -> bool {
    if total_attempts >= policy.maximum_total_attempts || remaining(deadline).is_zero() {
        return false;
    }
    let same = failure.retryable_same_deployment
        && same_deployment_attempts < policy.maximum_same_deployment_attempts;
    let failover = (failure.failover_eligible || refusal_eligible) && depth + 1 < route_length;
    same || failover
}

/// One pre-commit attempt outcome, private to the waterfall loop.
enum AttemptEnd {
    Committed(Box<CommittedAttempt>),
    Settled(SettledAttempt),
    /// The attempt failed before commitment; try the ladder.
    Ladder {
        failure: Failure,
        refusal_eligible: bool,
        /// Withheld refusal deltas plus the failing terminal, flushed
        /// outward only when the ladder is exhausted with a non-refusal
        /// failure (the python executor's `withheld_non_refusal_failure`).
        exhaustion_flush: Vec<Event>,
        usage: Option<Usage>,
        tool_names: Vec<String>,
        opened: bool,
    },
    /// Accounting failed mid-attempt; the request is answered internal.
    Accounting,
    /// The attempt settled, but retaining its output-less continuation
    /// failed; the public retention error answers the caller.
    Retention(PublicError),
}

/// Run one certified waterfall to its committed or terminal attempt.
///
/// Every started attempt settles exactly once through `guard`; on return the
/// request is either finalized (`Settled`/`Failed`) or owned by the single
/// committed attempt the caller must settle.
pub async fn acquire_attempt(ctx: &WaterfallContext<'_>, guard: &mut AttemptGuard) -> Won {
    let mut total_attempts: u32 = 0;
    let mut counts: Vec<u32> = vec![0; ctx.route.len()];
    let mut current_depth: Option<usize> = None;
    let mut last_failure: Option<Failure> = None;
    loop {
        let argument = compact_json(&json!({
            "request_id": ctx.request_id,
            "raw_key": ctx.raw_key,
            "attempt_ordinal": total_attempts,
            "current_depth": current_depth,
            "failure": last_failure.as_ref().map(|failure| json!({
                "failure_class": failure.failure_class.as_str(),
                "safe_message": failure.safe_message,
                "retryable_same_deployment": failure.retryable_same_deployment,
                "failover_eligible": failure.failover_eligible,
                // The control plane echoes the exhausting failure back, and its
                // answer wins over this one, so client-error attribution has to
                // survive the round trip to reach the caller.
                "rejected_parameter": failure.rejected_parameter,
                "provider_detail": failure.provider_detail,
                // Ownership survives the round trip too: the echoed exhaustion
                // must still render as the customer's 400.
                "customer_owned": failure.customer_owned,
                // The refusal category survives the round trip so an exhausted
                // refusal ladder still names its reason to the caller.
                "refusal_reason": failure.refusal_reason.map(|reason| reason.as_str()),
            })),
        }));
        let started_text = match ctx.bridge.call("start_attempt", argument).await {
            Ok(text) => text,
            Err(error) => {
                // The control plane finalized the request (budget quota, a
                // pre-dispatch reservation failure, or an expired deadline)
                // before raising; the public error is authoritative.
                guard.disarm_finalized("failed");
                return Won::Failed(error);
            }
        };
        let started: StartResponse = match serde_json::from_str(&started_text) {
            Ok(started) => started,
            Err(_) => {
                guard
                    .abandon(&Failure::new(
                        FailureClass::Internal,
                        "gateway attempt wire contract failed",
                    ))
                    .await;
                return Won::Failed(PublicError::internal());
            }
        };
        if started.exhausted {
            // The control plane already finalized the request with this
            // failure; answer the caller with its public form.
            guard.disarm_finalized("failed");
            let failure = started.failure.or(last_failure).unwrap_or_else(|| {
                Failure::new(
                    FailureClass::ProviderInternal,
                    "all exact-model deployments are unavailable",
                )
            });
            return Won::Failed(collection_public_error(&failure.boundary()));
        }
        let (Some(attempt_id), Some(depth)) = (started.attempt_id, started.route_depth) else {
            guard
                .abandon(&Failure::new(
                    FailureClass::Internal,
                    "gateway attempt wire contract failed",
                ))
                .await;
            return Won::Failed(PublicError::internal());
        };
        let Some(wire) = ctx.route.get(depth) else {
            guard.rebind(attempt_id);
            let failure = Failure::new(
                FailureClass::Internal,
                "gateway attempt wire contract failed",
            );
            guard
                .settle("failed", None, &[], Some(&failure), true)
                .await;
            return Won::Failed(PublicError::internal());
        };
        if current_depth == Some(depth) {
            METRICS.record_open_retry();
        }
        guard.rebind(attempt_id);
        total_attempts += 1;
        counts[depth] += 1;
        let end = run_attempt(ctx, guard, wire, depth).await;
        match end {
            AttemptEnd::Committed(committed) => return Won::Committed(committed),
            AttemptEnd::Settled(settled) => return Won::Settled(settled),
            AttemptEnd::Accounting => return Won::Failed(PublicError::internal()),
            AttemptEnd::Retention(error) => return Won::Failed(error),
            AttemptEnd::Ladder {
                failure,
                refusal_eligible,
                exhaustion_flush,
                usage,
                tool_names,
                opened,
            } => {
                if opened {
                    guard.mark_opened();
                }
                let boundary = failure.clone().boundary();
                let possible = successor_possible(
                    ctx.policy,
                    ctx.route.len(),
                    ctx.deadline,
                    total_attempts,
                    counts[depth],
                    depth,
                    &failure,
                    refusal_eligible,
                );
                if !guard
                    .settle(
                        "failed",
                        usage.as_ref(),
                        &tool_names,
                        Some(&boundary),
                        !possible,
                    )
                    .await
                {
                    return Won::Failed(PublicError::internal());
                }
                if possible {
                    current_depth = Some(depth);
                    last_failure = Some(failure);
                    continue;
                }
                if !exhaustion_flush.is_empty() {
                    // Exhausted with withheld refusals and a non-refusal
                    // failure: flush the bounded refusal output and the
                    // failing terminal outward, exactly once.
                    return Won::Settled(SettledAttempt {
                        depth,
                        events: exhaustion_flush,
                    });
                }
                return Won::Failed(collection_public_error(&boundary));
            }
        }
    }
}

/// Resolve the dispatch headers for one physical open attempt. Body-signing
/// dialects (Bedrock SigV4) are signed here, immediately before the provider
/// POST, so neither queue time nor a spent earlier attempt can age the
/// signature toward AWS's short clock window. Other dialects use the route
/// entry's headers unchanged.
async fn dispatch_headers(
    bridge: &Bridge,
    request_id: &str,
    wire: &DeploymentWire,
) -> Result<HashMap<String, String>, PublicError> {
    let mut headers = wire.headers.clone();
    let Some(body) = wire.upstream_body.as_deref() else {
        return Ok(headers);
    };
    let argument = compact_json(&json!({
        "request_id": request_id,
        "url": wire.url,
        "body": body,
    }));
    let text = bridge.call("sign_dispatch", argument).await?;
    let signed: HashMap<String, String> = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|value| {
            serde_json::from_value(value.get("headers").cloned().unwrap_or(Value::Null)).ok()
        })
        .ok_or_else(PublicError::internal)?;
    headers.extend(signed);
    Ok(headers)
}

/// Open and read one physical attempt up to commitment or its terminal.
/// On a customer-managed rung, a rejected credential or exhausted provider
/// account at stream OPEN is the customer's to fix (see
/// `stream_errors::customer_credential_failure`); failures declared on the
/// open stream take the same path inside the relay. House rungs are unchanged.
fn customer_owned(failure: Failure, wire: &DeploymentWire) -> Failure {
    if wire.billing_customer_managed {
        crate::stream_errors::customer_credential_failure(failure, &wire.provider)
    } else {
        failure
    }
}

async fn run_attempt(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    wire: &DeploymentWire,
    depth: usize,
) -> AttemptEnd {
    let Some(dialect) = Dialect::from_str(&wire.dialect) else {
        // Admission validated every dialect; reaching here is wire drift.
        return AttemptEnd::Ladder {
            failure: Failure::new(
                FailureClass::Internal,
                "gateway engine does not support the resolved provider dialect",
            ),
            refusal_eligible: false,
            exhaustion_flush: Vec::new(),
            usage: None,
            tool_names: Vec::new(),
            opened: false,
        };
    };
    // Body-signing dialects sign immediately before every physical attempt
    // so neither queue time nor a spent prior attempt can age the signature
    // (a same-deployment redial or a failover advance both call `run_attempt`
    // again, so each always gets a fresh signature); signing failures are
    // neither same-deployment-retryable nor failover-eligible, matching the
    // python executor's hard stop on an authentication failure.
    let headers = match dispatch_headers(ctx.bridge, ctx.request_id, wire).await {
        Ok(headers) => headers,
        Err(_) => {
            return AttemptEnd::Ladder {
                failure: Failure::new(
                    FailureClass::ProviderAuthentication,
                    "provider dispatch signing failed",
                ),
                refusal_eligible: false,
                exhaustion_flush: Vec::new(),
                usage: None,
                tool_names: Vec::new(),
                opened: false,
            };
        }
    };
    // The connection's raw timeout bounds each chunk read, exactly like the
    // python streaming path. The open phase is additionally bounded by the
    // fail-fast time-to-first-byte window (fresh per attempt) so a dead lane
    // that never answers is abandoned in seconds, not after the full
    // per-deployment timeout.
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let first_byte_deadline = Instant::now()
        + first_byte_allowance(
            wire,
            ctx.time_to_first_byte,
            ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
            ctx.approximate_input_tokens,
        );
    let open_bound = remaining(ctx.deadline)
        .min(phase_timeout)
        .min(remaining(first_byte_deadline));
    let response = match open_stream(
        ctx.http,
        &wire.url,
        &headers,
        &wire.idempotency_key,
        &wire.upstream_payload,
        wire.upstream_body.as_deref(),
        open_bound,
        dialect,
    )
    .await
    {
        Ok(response) => response,
        Err(failure) => {
            return AttemptEnd::Ladder {
                failure: customer_owned(failure, wire),
                refusal_eligible: false,
                exhaustion_flush: Vec::new(),
                usage: None,
                tool_names: Vec::new(),
                opened: false,
            }
        }
    };
    guard.mark_opened();
    let mut relay = match wire
        .fireworks_reasoning_route_sha256
        .clone()
        .or_else(|| wire.hunyuan_reasoning_route_sha256.clone())
    {
        Some(route_sha256) => UpstreamRelay::new_with_reasoning_content_route(
            response,
            dialect,
            first_byte_deadline,
            Some(route_sha256),
        ),
        None => UpstreamRelay::new(response, dialect, first_byte_deadline),
    };
    relay.set_stop_sequences(wire.stop_sequences.iter().cloned());
    relay.set_serialize_tool_calls(wire.serialize_tool_calls);
    if !wire.model_id.is_empty() {
        relay.set_request_words([wire.model_id.clone()]);
    }
    if wire.billing_customer_managed {
        // Applied to every failure the relay yields, before or after commit,
        // so a committed stream's late credential error is the customer's too.
        relay.set_customer_managed_provider(Some(wire.provider.clone()));
    }
    let mut usage: Option<Usage> = None;
    let mut tool_names: Vec<String> = Vec::new();
    let mut withheld: Vec<Event> = Vec::new();
    let mut withheld_bytes = 0usize;
    loop {
        let event = match relay
            .next_event(ctx.deadline, phase_timeout, guard.started)
            .await
        {
            Ok(Some(event)) => event,
            Ok(None) => {
                return AttemptEnd::Ladder {
                    failure: ended_without_terminal(),
                    refusal_eligible: false,
                    exhaustion_flush: Vec::new(),
                    usage,
                    tool_names,
                    opened: true,
                }
            }
            Err(failure) => {
                return AttemptEnd::Ladder {
                    failure,
                    refusal_eligible: false,
                    exhaustion_flush: Vec::new(),
                    usage,
                    tool_names,
                    opened: true,
                }
            }
        };
        track_event(&event, &mut usage, &mut tool_names);
        let refusal_text = match &event {
            Event::RefusalDelta(text) | Event::ProviderRefusalDelta { delta: text, .. } => {
                Some(text)
            }
            _ => None,
        };
        if let Some(text) = refusal_text {
            if ctx.policy.refusal_failover {
                let event_bytes = text.len();
                if withheld_bytes + event_bytes > MAXIMUM_WITHHELD_REFUSAL_BYTES
                    || withheld.len() + 1 > MAXIMUM_WITHHELD_REFUSAL_EVENTS
                {
                    // Buffer overflow commits and flushes.
                    let mut prefix = std::mem::take(&mut withheld);
                    prefix.push(event);
                    return AttemptEnd::Committed(Box::new(CommittedAttempt {
                        depth,
                        prefix,
                        relay,
                        usage,
                        tool_names,
                        visible_refusal: true,
                    }));
                }
                withheld_bytes += event_bytes;
                withheld.push(event);
                continue;
            }
        }
        if is_semantic(&event) {
            // First outward semantic output freezes this deployment; any
            // withheld refusals flush ahead of it.
            let visible_refusal = !withheld.is_empty()
                || matches!(
                    event,
                    Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. }
                );
            let mut prefix = std::mem::take(&mut withheld);
            prefix.push(event);
            return AttemptEnd::Committed(Box::new(CommittedAttempt {
                depth,
                prefix,
                relay,
                usage,
                tool_names,
                visible_refusal,
            }));
        }
        if !event.is_terminal() {
            // Pre-commit non-semantic events are dropped from the outward
            // stream (usage stays tracked), matching the python executor.
            continue;
        }
        match &event {
            Event::Failed(failure) => {
                let typed_refusal = failure.failure_class == FailureClass::Refusal;
                let exhaustion_flush = if !withheld.is_empty() && !typed_refusal {
                    let mut flush = std::mem::take(&mut withheld);
                    flush.push(event.clone());
                    flush
                } else {
                    withheld.clear();
                    Vec::new()
                };
                return AttemptEnd::Ladder {
                    failure: failure.clone(),
                    refusal_eligible: typed_refusal && ctx.policy.refusal_failover,
                    exhaustion_flush,
                    usage,
                    tool_names,
                    opened: true,
                };
            }
            _ => {
                if !withheld.is_empty() {
                    // A refusal-only stream that terminated successfully is
                    // a provider refusal: withhold the output and advance,
                    // matching the python executor's converted terminal.
                    withheld.clear();
                    return AttemptEnd::Ladder {
                        failure: Failure::new(
                            FailureClass::Refusal,
                            "provider refused the request",
                        ),
                        refusal_eligible: ctx.policy.refusal_failover,
                        exhaustion_flush: Vec::new(),
                        usage,
                        tool_names,
                        opened: true,
                    };
                }
                // A successful terminal with no semantic output: retain the
                // output-less continuation while the attempt is still in
                // flight, settle, then answer with the tracked usage ahead of
                // the terminal so the encoders keep the client-visible token
                // accounting.
                let retention_failure = match &ctx.output_less_retention {
                    Some(argument) => ctx.bridge.call("remember", argument.clone()).await.err(),
                    None => None,
                };
                let outcome = if matches!(event, Event::Incomplete) {
                    "incomplete"
                } else {
                    "completed"
                };
                if !guard
                    .settle(outcome, usage.as_ref(), &tool_names, None, true)
                    .await
                {
                    return AttemptEnd::Accounting;
                }
                if let Some(error) = retention_failure {
                    // The provider outcome settled above, exactly like a
                    // committed attempt's retention failure; only the HTTP
                    // result reports it.
                    return AttemptEnd::Retention(error);
                }
                let mut events = Vec::with_capacity(2);
                if let Some(tracked) = usage {
                    events.push(Event::Usage(tracked));
                }
                events.push(event);
                return AttemptEnd::Settled(SettledAttempt { depth, events });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(base: Option<f64>, slope: Option<f64>) -> DeploymentWire {
        DeploymentWire {
            provider: "openai".to_string(),
            deployment_id: "d".to_string(),
            dialect: "openai_compatible".to_string(),
            url: "https://provider.test".to_string(),
            headers: HashMap::new(),
            timeout_seconds: 60.0,
            upstream_payload: Value::Null,
            upstream_body: None,
            fireworks_reasoning_route_sha256: None,
            hunyuan_reasoning_route_sha256: None,
            reasoning_output_exposed: false,
            stop_sequences: Vec::new(),
            serialize_tool_calls: false,
            model_id: String::new(),
            billing_customer_managed: false,
            idempotency_key: "op".to_string(),
            time_to_first_byte_base_seconds: base,
            time_to_first_byte_seconds_per_million_input_tokens: slope,
        }
    }

    #[test]
    fn first_byte_allowance_scales_with_input_and_honors_overrides() {
        let default_base = Duration::from_secs(15);
        // No overrides, tiny request: effectively the flat default.
        let flat = first_byte_allowance(&wire(None, None), default_base, 240.0, 100.0);
        assert!((flat.as_secs_f64() - 15.024).abs() < 1e-6);
        // No overrides, one million approximate tokens: base plus the
        // full default slope.
        let scaled = first_byte_allowance(&wire(None, None), default_base, 240.0, 1_000_000.0);
        assert!((scaled.as_secs_f64() - 255.0).abs() < 1e-6);
        // Deployment overrides replace both the base and the slope.
        let overridden = first_byte_allowance(
            &wire(Some(30.0), Some(60.0)),
            default_base,
            240.0,
            500_000.0,
        );
        assert!((overridden.as_secs_f64() - 60.0).abs() < 1e-6);
        // A zero slope pins the flat bound regardless of input size.
        let pinned = first_byte_allowance(&wire(None, Some(0.0)), default_base, 240.0, 9e9);
        assert!((pinned.as_secs_f64() - 15.0).abs() < 1e-6);
    }

    fn policy(refusal_failover: bool) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: 8,
            maximum_same_deployment_attempts: 2,
            refusal_failover,
        }
    }

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    #[test]
    fn successor_requires_capacity_and_an_eligible_class() {
        let retryable = Failure::new(FailureClass::ProviderInternal, "boom").with_retry(true, true);
        // Same-deployment retry within the per-deployment cap.
        assert!(successor_possible(
            policy(false),
            1,
            far_deadline(),
            1,
            1,
            0,
            &retryable,
            false,
        ));
        // The per-deployment cap forbids a redial but failover still runs.
        assert!(successor_possible(
            policy(false),
            2,
            far_deadline(),
            2,
            2,
            0,
            &retryable,
            false,
        ));
        // A single-deployment route with the redial cap reached is exhausted.
        assert!(!successor_possible(
            policy(false),
            1,
            far_deadline(),
            2,
            2,
            0,
            &retryable,
            false,
        ));
        // The hard total cap ends the ladder regardless of class.
        assert!(!successor_possible(
            policy(false),
            4,
            far_deadline(),
            8,
            1,
            0,
            &retryable,
            false,
        ));
        // An expired deadline ends the ladder.
        assert!(!successor_possible(
            policy(false),
            4,
            Instant::now(),
            1,
            1,
            0,
            &retryable,
            false,
        ));
    }

    #[test]
    fn ineligible_classes_never_advance_without_refusal_opt_in() {
        let invalid = Failure::new(FailureClass::InvalidRequest, "bad request");
        assert!(!successor_possible(
            policy(false),
            4,
            far_deadline(),
            1,
            1,
            0,
            &invalid,
            false,
        ));
        let refusal = Failure::new(FailureClass::Refusal, "provider refused the request");
        assert!(!successor_possible(
            policy(false),
            4,
            far_deadline(),
            1,
            1,
            0,
            &refusal,
            false,
        ));
        // The refusal advances only when the alias revision opted in.
        assert!(successor_possible(
            policy(true),
            4,
            far_deadline(),
            1,
            1,
            0,
            &refusal,
            true,
        ));
        // Refusal failover cannot pass the last deployment.
        assert!(!successor_possible(
            policy(true),
            1,
            far_deadline(),
            1,
            1,
            0,
            &refusal,
            true,
        ));
    }

    #[test]
    fn failover_only_classes_skip_the_redial_and_advance() {
        let throttled = Failure::new(FailureClass::Throttled, "throttled").with_retry(false, true);
        assert!(successor_possible(
            policy(false),
            2,
            far_deadline(),
            1,
            1,
            0,
            &throttled,
            false,
        ));
        assert!(!successor_possible(
            policy(false),
            1,
            far_deadline(),
            1,
            1,
            0,
            &throttled,
            false,
        ));
    }
}
