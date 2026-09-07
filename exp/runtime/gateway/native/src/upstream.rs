//! Upstream provider HTTP transport over one shared pooled client.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::dialects::Dialect;
use crate::errors::{Failure, FailureClass};
use crate::param_attribution::{
    generic_error_code, rejected_by_lane_limitation, rejected_by_routing_gate, rejected_code,
    rejected_detail, rejected_model_not_found, rejected_parameter,
};

/// Build the shared pooled upstream client, mirroring the pooling constants in
/// `providers.async_transport` (64 keep-alive) and its no-redirect policy so a
/// provider 3xx can never re-send credentials to an attacker-chosen location.
///
/// `connect_timeout` bounds only the TCP+TLS connect phase; a dead lane whose
/// host never accepts the connection fails over after this window instead of
/// hanging on the per-deployment request timeout.
pub fn build_client(connect_timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(connect_timeout)
        .redirect(reqwest::redirect::Policy::none())
        .use_rustls_tls()
        .build()
        .map_err(|error| format!("upstream client construction failed: {error}"))
}

/// Classify one sanitized HTTP or connection failure by status only,
/// mirroring `providers.errors._transport_failure`: classes, wording, the
/// same-deployment retry policy, and failover eligibility across the
/// certified deployment ladder.
pub fn transport_failure(status: Option<u16>) -> Failure {
    let (class, message, retryable, failover) = match status {
        Some(401) | Some(403) => (
            FailureClass::ProviderAuthentication,
            "provider authentication failed; ask the gateway operator to verify \
             the provider connection credential",
            false,
            true,
        ),
        Some(404) => (
            FailureClass::ProviderNotFound,
            "provider deployment was not found; ask the gateway operator to verify \
             the deployment model ID in the catalog",
            false,
            true,
        ),
        Some(429) => (
            FailureClass::Throttled,
            "provider throttled the request; retry after the delay in the Retry-After header",
            false,
            true,
        ),
        // 402 is the provider ACCOUNT's billing state (trial quota exhausted,
        // postpaid billing disabled), never the caller's request fields: it is
        // operator-actionable deadness, so it fails over in every failover mode
        // instead of surfacing a corrective 400 to the caller.
        Some(402) => (
            FailureClass::ProviderQuota,
            "provider account quota or billing is exhausted; ask the gateway operator \
             to fund or enable the provider account",
            false,
            true,
        ),
        Some(408) => (
            FailureClass::Timeout,
            "provider request timed out; retry the request",
            true,
            true,
        ),
        Some(code) if code >= 500 => (
            FailureClass::ProviderInternal,
            "provider service failed; retry after a short delay",
            true,
            true,
        ),
        Some(409) | Some(425) => (
            FailureClass::ProviderInternal,
            "provider reported a transient conflict; retry the request",
            true,
            true,
        ),
        Some(code) if (400..500).contains(&code) => (
            FailureClass::InvalidRequest,
            "provider rejected the request; verify the request fields against \
             the model alias capabilities",
            false,
            false,
        ),
        // Redirects are disabled, so a 3xx (or any other status) is an
        // unexpected provider response, never followed.
        Some(_) => (
            FailureClass::ProviderInternal,
            "provider returned an unexpected status; retry the request",
            false,
            true,
        ),
        None => (
            FailureClass::Transport,
            "provider transport failed; retry the request",
            true,
            true,
        ),
    };
    Failure::new(class, message).with_retry(retryable, failover)
}

/// Classify a lead that connected but never completed the request/response-header
/// phase within `phase_timeout`. A deployment that accepted the connection but
/// stalled awaiting response headers is the same dead-lane signal as a stalled
/// first byte, so it mirrors `relay::first_byte_timeout_failure`: failover-eligible
/// (advance to the next certified rung) but deliberately *not* same-deployment
/// retryable. Redialing the same stalled deployment would only burn another full
/// header-timeout window before failing over; skipping straight to the next rung
/// keeps a stalled lead's cost near one fail-fast window. It stays a
/// `FailureClass::Timeout`, so it feeds the health circuit like other timeouts.
fn open_timeout_failure() -> Failure {
    Failure::new(
        FailureClass::Timeout,
        "provider did not send response headers in time",
    )
    .with_retry(false, true)
}

/// Open one streaming POST and return the response on HTTP success. The
/// timeout bounds only the request/response-header phase; body-read pacing is
/// bounded per chunk by the caller, mirroring the python transport split.
///
/// `raw_body` carries the exact pre-serialized body for body-signing dialects
/// (Bedrock SigV4): its signature covers those exact bytes, so it is sent
/// verbatim with the signed headers instead of re-serializing `payload`.
#[allow(clippy::too_many_arguments)]
pub async fn open_stream(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    idempotency_key: &str,
    payload: &Value,
    raw_body: Option<&str>,
    phase_timeout: Duration,
    dialect: Dialect,
) -> Result<reqwest::Response, Failure> {
    let mut request = client.post(url);
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("idempotency-key") {
            continue;
        }
        request = request.header(name, value);
    }
    request = request.header("Idempotency-Key", idempotency_key);
    let send = match raw_body {
        Some(body) => request.body(body.to_string()).send(),
        None => request.json(payload).send(),
    };
    let phase_started = Instant::now();
    let response = match tokio::time::timeout(phase_timeout, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            if error.is_timeout() {
                return Err(open_timeout_failure());
            }
            return Err(transport_failure(None));
        }
        Err(_) => return Err(open_timeout_failure()),
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let failure = transport_failure(Some(status));
        // Only the generic client-error class may carry attribution: the body
        // is read bounded, and the relayable facts are a validated parameter
        // path plus the provider's own bounded explanation of what the caller
        // got wrong; every other class stays content-free. A 403 is read too,
        // only to tell an aggregator routing gate from a credential verdict.
        if failure.failure_class != FailureClass::InvalidRequest && status != 403 {
            return Err(failure);
        }
        // The attribution read never outlives the rung's own header-phase
        // budget: a provider that answers its status and then stalls the body
        // costs at most what was left of that window, never a further two
        // seconds past the caller's deadline.
        let body_budget =
            ERROR_BODY_READ_TIMEOUT.min(phase_timeout.saturating_sub(phase_started.elapsed()));
        let body = match tokio::time::timeout(body_budget, bounded_error_body(response)).await {
            Ok(Some(body)) => Some(body),
            _ => None,
        };
        if status == 403 {
            if body
                .as_deref()
                .is_some_and(|body| rejected_by_routing_gate(dialect, body))
            {
                return Err(Failure::new(
                    FailureClass::ProviderNotFound,
                    "provider does not route this model for the gateway's account; ask \
                     the gateway operator to change or disable the lane",
                )
                .with_retry(false, true));
            }
            return Err(failure);
        }
        // A client-error status whose body names a missing model is the
        // catalog's fault, not the caller's: it takes the 404 policy so the
        // ladder advances instead of surfacing one dead rung as a 400.
        if body
            .as_deref()
            .is_some_and(|body| rejected_model_not_found(dialect, body))
        {
            return Err(transport_failure(Some(404)));
        }
        let parameter = body
            .as_deref()
            .and_then(|body| rejected_parameter(dialect, body));
        // The payload's own model id is a caller-known word: a provider
        // sentence naming it unquoted (Anthropic's client-version gate does)
        // must not be redacted as infrastructure.
        let request_words: Vec<&str> = payload
            .get("model")
            .and_then(Value::as_str)
            .into_iter()
            .collect();
        let code = body
            .as_deref()
            .and_then(|body| rejected_code(dialect, body));
        let detail = body
            .as_deref()
            .and_then(|body| rejected_detail(dialect, body, &request_words))
            // A sentence the identifier screen dropped still leaves the
            // provider's own code token: "invalid_value" beats "verify the
            // request fields" for the caller and the ledger alike. A generic
            // family type or bare status adds nothing and is not relayed.
            .or_else(|| code.clone().filter(|token| !generic_error_code(token)));
        // A content-filter CODE under a 4xx is the model's verdict on the
        // content (Azure and Gemini answer 400 for it), not a request-shape
        // error: file and answer it as a refusal naming its bounded category,
        // detail kept ledger-only. Only the authoritative code decides here; a
        // sentence saying "blocked by" could be about a firewall or a limit.
        if crate::stream_errors::is_refusal_code(code.as_deref()) {
            let reason = crate::stream_errors::refusal_reason(code.as_deref(), None);
            return Err(Failure::refusal(reason).with_provider_detail(detail));
        }
        // A sentence naming a limitation of THIS lane's serving stack (a chat
        // template that rejects a mid-conversation system turn the OpenAI
        // contract allows) keeps the caller's class and detail but fails over:
        // another rung serves the same request, and only a route with no other
        // rung surfaces the 400.
        let lane_limitation = body
            .as_deref()
            .is_some_and(|body| rejected_by_lane_limitation(dialect, body));
        return Err(failure
            .with_retry(false, lane_limitation)
            .with_rejected_parameter(parameter)
            .with_provider_detail(detail));
    }
    Ok(response)
}

/// Longest provider error body read for parameter attribution.
const ERROR_BODY_READ_LIMIT: usize = 16 * 1024;

/// Bound on the whole attribution body read; a stalling error stream is
/// abandoned and the failure stays content-free.
const ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(2);

/// Read at most `ERROR_BODY_READ_LIMIT` bytes of one error response body.
async fn bounded_error_body(mut response: reqwest::Response) -> Option<String> {
    let mut collected: Vec<u8> = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        if collected.len() + chunk.len() > ERROR_BODY_READ_LIMIT {
            return None;
        }
        collected.extend_from_slice(&chunk);
    }
    String::from_utf8(collected).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_failure_flags_mirror_the_python_taxonomy() {
        // (status, retryable_same_deployment, failover_eligible)
        let table = [
            (Some(401), false, true),
            (Some(403), false, true),
            (Some(404), false, true),
            (Some(429), false, true),
            (Some(402), false, true),
            (Some(408), true, true),
            (Some(500), true, true),
            (Some(503), true, true),
            (Some(409), true, true),
            (Some(425), true, true),
            (Some(400), false, false),
            (Some(422), false, false),
            (Some(301), false, true),
            (None, true, true),
        ];
        for (status, retryable, failover) in table {
            let failure = transport_failure(status);
            assert_eq!(
                failure.retryable_same_deployment, retryable,
                "retryable for {status:?}"
            );
            assert_eq!(
                failure.failover_eligible, failover,
                "failover for {status:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_402_with_the_literal_tokenhub_body_classes_provider_quota() {
        // TokenHub (the Tencent relay) answers HTTP 402 with provider code
        // 401008 on EVERY request shape once the account's free trial is
        // exhausted and postpaid billing is off. Classing that invalid_request
        // blamed 652 callers' request fields for the provider's billing state
        // (2026-09 incident): the class must be the provider-side quota family,
        // the rung must fail over, and the body must never be relayed.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer).await;
            let body = "{\"error\":{\"code\":401008,\"message\":\"free trial quota exhausted \
                        and postpaid billing is not enabled - enable in Console > Online \
                        Inference Service\",\"type\":\"payment_required\"}}";
            let response = format!(
                "HTTP/1.1 402 Payment Required\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let client = build_client(Duration::from_secs(2)).expect("client");
        let failure = open_stream(
            &client,
            &format!("http://{addr}/v1/chat/completions"),
            &HashMap::new(),
            "idem-402",
            &serde_json::json!({"model": "m", "messages": []}),
            None,
            Duration::from_secs(5),
            Dialect::OpenAiCompatible,
        )
        .await
        .expect_err("a 402 must classify as a failure");
        assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
        assert!(failure.failover_eligible, "an unfunded rung must fail over");
        assert!(!failure.retryable_same_deployment);
        assert!(
            failure.provider_detail.is_none() && failure.rejected_parameter.is_none(),
            "a billing failure must stay content-free"
        );
        assert!(
            !failure.safe_message.contains("request fields"),
            "the caller must never be told to fix their fields for a provider billing state"
        );
    }

    async fn open_against_body(status_line: &str, body: &'static str, model: &str) -> Failure {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let status_line = status_line.to_string();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer).await;
            let response = format!(
                "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let client = build_client(Duration::from_secs(2)).expect("client");
        open_stream(
            &client,
            &format!("http://{addr}/v1/chat/completions"),
            &HashMap::new(),
            "idem-4xx",
            &serde_json::json!({"model": model, "messages": []}),
            None,
            Duration::from_secs(5),
            Dialect::OpenAiCompatible,
        )
        .await
        .expect_err("a 4xx must classify as a failure")
    }

    #[tokio::test]
    async fn a_dropped_provider_sentence_still_relays_the_provider_code() {
        // The sentence names an account handle, so the identifier screen drops
        // it; the caller still learns WHICH rejection it was.
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"code\":\"invalid_value\",\"type\":\"invalid_request_error\",\
             \"message\":\"Invalid value for organization org_a1b2c3d4e5f6: not allowed\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert_eq!(failure.provider_detail.as_deref(), Some("invalid_value"));
        assert_eq!(
            failure.public_error().message,
            "provider rejected the request: invalid_value"
        );
    }

    #[tokio::test]
    async fn a_blocked_by_sentence_without_a_refusal_code_stays_a_request_error() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"code\":\"invalid_value\",\"message\":\"Request blocked by the \
             organization policy for this parameter.\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    }

    #[tokio::test]
    async fn a_content_filter_4xx_is_a_refusal_not_a_request_shape_error() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"code\":\"content_filter\",\"message\":\"The response was \
             filtered due to the prompt triggering the content management policy.\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert_eq!(failure.public_error().status_code, 400);
        assert_eq!(failure.public_error().code, "refusal");
        // The content_filter code names the content-policy category.
        assert_eq!(
            failure.refusal_reason,
            Some(crate::errors::RefusalReason::ContentPolicy)
        );
        // The sanitized sentence (or the code token when it must drop) rides
        // to the ledger; a refusal never relays it to the caller.
        assert!(failure.provider_detail.is_some());
        assert_eq!(
            failure.public_error().message,
            "provider refused the request: content policy"
        );
    }

    #[tokio::test]
    async fn an_aggregator_routing_gate_403_is_not_a_credential_failure() {
        let failure = open_against_body(
            "403 Forbidden",
            "{\"error\":{\"message\":\"thinkingmachines/inkling:free is only available \
             on agentic harnesses.\",\"code\":403,\"metadata\":{\"routing_funnel\":\
             [{\"step\":\"Initial Endpoints\",\"endpoint_count\":1}],\
             \"failed_routing_step\":\"Gate Free Endpoints by Agentic Harness\"}}}",
            "thinkingmachines/inkling:free",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
        assert!(failure.failover_eligible);
        assert!(!failure.retryable_same_deployment);
        assert!(failure.provider_detail.is_none());
    }

    #[tokio::test]
    async fn a_plain_403_stays_a_credential_failure() {
        let failure = open_against_body(
            "403 Forbidden",
            "{\"error\":{\"message\":\"Forbidden\",\"code\":403}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
        assert!(failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_lane_limitation_400_keeps_the_class_but_fails_over() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"message\":\"System message must be at the beginning.\",\
             \"type\":\"invalid_request_error\"}}",
            "qwen3.8-27b",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(
            failure.failover_eligible,
            "another rung can carry the request"
        );
        assert!(!failure.retryable_same_deployment);
        assert!(failure
            .provider_detail
            .as_deref()
            .is_some_and(|detail| detail.contains("System message must be at the beginning")));
    }

    #[tokio::test]
    async fn a_lane_limitation_phrase_echoed_outside_the_message_does_not_fail_over() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"message\":\"Invalid value for temperature.\",\
             \"type\":\"invalid_request_error\",\"param\":\"temperature\",\
             \"echo\":\"System message must be at the beginning\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(!failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_403_naming_a_step_without_a_walked_funnel_stays_a_credential_failure() {
        let failure = open_against_body(
            "403 Forbidden",
            "{\"error\":{\"message\":\"Forbidden\",\"code\":403,\
             \"metadata\":{\"failed_routing_step\":\"Authenticate\"}}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
    }

    #[tokio::test]
    async fn an_ordinary_400_does_not_fail_over() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"message\":\"Invalid value for temperature.\",\
             \"type\":\"invalid_request_error\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(!failure.failover_eligible);
    }

    #[test]
    fn header_phase_timeout_fails_over_without_a_same_deployment_redial() {
        // A lead that connects but never completes the response-header phase must
        // skip straight to the next rung (failover-eligible) instead of redialing
        // the same stalled deployment for another full header-timeout window.
        let failure = open_timeout_failure();
        assert_eq!(failure.failure_class, FailureClass::Timeout);
        assert!(
            !failure.retryable_same_deployment,
            "a header-phase stall must not redial the same deployment"
        );
        assert!(
            failure.failover_eligible,
            "a header-phase stall must fail over to the next certified rung"
        );
    }
}
