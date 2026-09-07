//! OpenAI-shaped public errors mirroring `exp.runtime.openai_protocol.errors`.

use serde::{Deserialize, Serialize};
use serde_json::json;

/// One sanitized public protocol error carrying its HTTP representation.
///
/// Field names match the JSON payload attached to `NativeBridgeError` on the
/// Python side so a bridge failure deserializes directly into this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicError {
    pub status_code: u16,
    pub code: String,
    pub message: String,
    #[serde(default = "default_error_type")]
    pub error_type: String,
    #[serde(default)]
    pub param: Option<String>,
    #[serde(default)]
    pub retry_after_seconds: Option<u32>,
    /// The bounded category of a provider refusal (`code: refusal` only):
    /// every refusal answer carries one, `unspecified` when the provider
    /// named no reason. Absent on every other error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<RefusalReason>,
}

fn default_error_type() -> String {
    "invalid_request_error".to_string()
}

impl PublicError {
    pub fn new(status_code: u16, code: &str, message: &str, error_type: &str) -> Self {
        Self {
            status_code,
            code: code.to_string(),
            message: message.to_string(),
            error_type: error_type.to_string(),
            param: None,
            retry_after_seconds: None,
            refusal_reason: None,
        }
    }

    /// The OpenAI error envelope body, matching `OpenAIProtocolError.json_body()`.
    ///
    /// A refusal adds the one additive `refusal_reason` field; every other
    /// error keeps the exact four-field envelope.
    pub fn json_body(&self) -> serde_json::Value {
        let mut body = json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        });
        if let Some(reason) = self.refusal_reason {
            body["error"]["refusal_reason"] = json!(reason.as_str());
        }
        body
    }

    pub fn invalid_key() -> Self {
        Self::new(
            401,
            "invalid_key",
            "A valid gateway Bearer key is required. Send the virtual key as \
             'Authorization: Bearer <key>'.",
            "authentication_error",
        )
    }

    pub fn invalid_json() -> Self {
        Self::new(
            400,
            "invalid_json",
            "Request body must contain valid JSON. Re-encode the payload and resend.",
            "invalid_request_error",
        )
    }

    pub fn internal() -> Self {
        Self::new(
            500,
            "internal_error",
            "The gateway request failed. Retry the request; if this persists, \
             ask the gateway operator to inspect the server logs.",
            "api_error",
        )
    }

    pub fn draining() -> Self {
        let mut error = Self::new(
            503,
            "gateway_draining",
            "The gateway is draining and is not accepting new requests. \
             Retry after the delay in the Retry-After header.",
            "api_error",
        );
        error.retry_after_seconds = Some(10);
        error
    }

    pub fn request_too_large() -> Self {
        Self::new(
            413,
            "request_too_large",
            "Request body exceeds the gateway limit. Reduce the request size and resend.",
            "invalid_request_error",
        )
    }

    pub fn provider_output_too_large() -> Self {
        Self::new(
            502,
            "provider_output_too_large",
            "Provider output exceeded the gateway response limit. \
             Request less output, for example with a lower max_tokens value.",
            "api_error",
        )
    }
}

/// Stable failure classes shared with `GatewayFailureClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    InvalidRequest,
    UnsupportedCapability,
    Authentication,
    Authorization,
    QuotaExceeded,
    Throttled,
    Transport,
    Timeout,
    ProviderAuthentication,
    ProviderNotFound,
    /// The provider ACCOUNT cannot pay for the request (quota exhausted,
    /// billing not enabled): operator-actionable deadness, distinct from
    /// `QuotaExceeded`, which is the CALLER's gateway credit.
    ProviderQuota,
    Refusal,
    MalformedResponse,
    ProviderInternal,
    Cancelled,
    Guardrail,
    Internal,
    Unavailable,
}

impl FailureClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            FailureClass::InvalidRequest => "invalid_request",
            FailureClass::UnsupportedCapability => "unsupported_capability",
            FailureClass::Authentication => "authentication",
            FailureClass::Authorization => "authorization",
            FailureClass::QuotaExceeded => "quota_exceeded",
            FailureClass::Throttled => "throttled",
            FailureClass::Transport => "transport",
            FailureClass::Timeout => "timeout",
            FailureClass::ProviderAuthentication => "provider_authentication",
            FailureClass::ProviderNotFound => "provider_not_found",
            FailureClass::ProviderQuota => "provider_quota",
            FailureClass::Refusal => "refusal",
            FailureClass::MalformedResponse => "malformed_response",
            FailureClass::ProviderInternal => "provider_internal",
            FailureClass::Cancelled => "cancelled",
            FailureClass::Guardrail => "guardrail",
            FailureClass::Internal => "internal",
            FailureClass::Unavailable => "unavailable",
        }
    }
}

/// The bounded category of a provider refusal, derived from the provider's
/// own code and sentence (`stream_errors::refusal_reason`) and shared with
/// the python `GatewayRefusalReason`.
///
/// The caller sees WHICH policy declined the content as a fixed phrase,
/// never the provider's prose: the vocabulary is closed so a client can
/// branch on it, and the raw token keeps riding `provider_detail` into the
/// ledger only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalReason {
    /// A cyber-safety verdict (OpenAI `cyber_policy`).
    CyberPolicy,
    /// A biological, chemical, radiological, or nuclear weapons verdict.
    Cbrn,
    /// A general content-policy or safety-system verdict.
    ContentPolicy,
    /// The output would recite copyrighted material (Gemini `RECITATION`).
    Recitation,
    /// The provider's data-inspection layer blocked the content (Gemini
    /// `SPII`, Qwen `data_inspection_failed`).
    DataInspection,
    /// The provider refused without naming a reason.
    Unspecified,
}

impl RefusalReason {
    /// The wire name shared with the python enum and the public error field.
    pub fn as_str(&self) -> &'static str {
        match self {
            RefusalReason::CyberPolicy => "cyber_policy",
            RefusalReason::Cbrn => "cbrn",
            RefusalReason::ContentPolicy => "content_policy",
            RefusalReason::Recitation => "recitation",
            RefusalReason::DataInspection => "data_inspection",
            RefusalReason::Unspecified => "unspecified",
        }
    }

    /// The fixed caller-facing phrase, or none for an unnamed refusal.
    pub fn phrase(&self) -> Option<&'static str> {
        match self {
            RefusalReason::CyberPolicy => Some("cybersecurity policy"),
            RefusalReason::Cbrn => Some("biological/chemical safety policy"),
            RefusalReason::ContentPolicy => Some("content policy"),
            RefusalReason::Recitation => Some("recitation of copyrighted material"),
            RefusalReason::DataInspection => Some("data inspection"),
            RefusalReason::Unspecified => None,
        }
    }
}

/// The generic refusal sentence every refusal message starts from.
const REFUSAL_MESSAGE: &str = "provider refused the request";

/// One sanitized provider failure, the Rust mirror of `GatewayFailure`,
/// including the executor's per-failure retry classification: whether the
/// same deployment may be redialed and whether a later certified deployment
/// may serve the request instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub failure_class: FailureClass,
    pub safe_message: String,
    #[serde(default)]
    pub retryable_same_deployment: bool,
    #[serde(default)]
    pub failover_eligible: bool,
    /// Validated provider-named parameter path; one of the two facts a
    /// sanitized client-error may relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_parameter: Option<String>,
    /// The provider's own bounded single-line explanation of a client error,
    /// relayed only for that class so the caller sees what was actually
    /// refused; every other class stays caller-content-free. On a coerced
    /// malformed-response boundary it instead carries the gateway's own
    /// static parse-reject reason, which never reaches the caller but does
    /// reach settlement so the ledger can name the exact wire shape that
    /// failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_detail: Option<String>,
    /// Known wait before a retry can dispatch (a throttle window's
    /// remainder). When present on a throttled failure, the public mapping
    /// advertises this value as `Retry-After` (floored at the fixed default)
    /// so the header never contradicts the message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u32>,
    /// The failure is the CUSTOMER's own provider configuration (a rejected
    /// credential or exhausted account on their BYOK rung). The class keeps
    /// its ladder semantics (fail over to any other rung), but a terminal
    /// answer is their 400 naming the fix, and settlement files it client-side.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub customer_owned: bool,
    /// The bounded category of a refusal, set by every refusal builder so
    /// the public error and the settlement ledger can name which policy
    /// declined the content without parsing `provider_detail`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<RefusalReason>,
}

impl Failure {
    pub fn new(failure_class: FailureClass, safe_message: &str) -> Self {
        Self {
            failure_class,
            safe_message: safe_message.to_string(),
            retryable_same_deployment: false,
            failover_eligible: false,
            rejected_parameter: None,
            provider_detail: None,
            retry_after_seconds: None,
            customer_owned: false,
            refusal_reason: None,
        }
    }

    /// The provider refusal for one bounded reason. The message is the fixed
    /// generic sentence followed by the reason's fixed phrase (nothing the
    /// provider wrote), and the reason rides the failure into the public
    /// error and settlement.
    pub fn refusal(reason: RefusalReason) -> Self {
        let safe_message = match reason.phrase() {
            Some(phrase) => format!("{REFUSAL_MESSAGE}: {phrase}"),
            None => REFUSAL_MESSAGE.to_string(),
        };
        Self {
            refusal_reason: Some(reason),
            ..Self::new(FailureClass::Refusal, &safe_message)
        }
    }

    /// Attach one already-validated provider parameter path.
    pub fn with_rejected_parameter(mut self, parameter: Option<String>) -> Self {
        self.rejected_parameter = parameter;
        self
    }

    /// Attach one already-sanitized provider explanation.
    pub fn with_provider_detail(mut self, detail: Option<String>) -> Self {
        self.provider_detail = detail;
        self
    }

    /// Attach the python taxonomy's retry classification to this failure.
    pub fn with_retry(mut self, retryable_same_deployment: bool, failover_eligible: bool) -> Self {
        self.retryable_same_deployment = retryable_same_deployment;
        self.failover_eligible = failover_eligible;
        self
    }

    /// Coerce this failure to its boundary form: the python engine replaces
    /// malformed-response detail with one generic safe message before both
    /// accounting and the public error, so the native plane does the same.
    pub fn boundary(self) -> Self {
        match self.failure_class {
            FailureClass::MalformedResponse
                if self.safe_message
                    != "provider returned a malformed response; retry the request" =>
            {
                // The generic boundary message is about to erase the specific
                // parse-reject reason, so emit it once as a structured,
                // content-free operator line (the crate's stderr idiom) before
                // it is lost. The reason is a static parser label, optionally
                // carrying one provider discriminator (an item or event type)
                // already reduced to a bounded identifier-shaped token, never
                // provider payload, so it is safe to log.
                let line = json!({
                    "event": "malformed_response_boundary",
                    "reason": self.safe_message,
                });
                eprintln!("exp-gateway-native: {line}");
                // The reason also rides provider_detail into settlement so
                // the ledger names the exact wire shape that failed; the
                // public error never relays it (that path is scoped to the
                // invalid-request class).
                Failure::new(
                    FailureClass::MalformedResponse,
                    "provider returned a malformed response; retry the request",
                )
                .with_retry(self.retryable_same_deployment, self.failover_eligible)
                .with_provider_detail(Some(self.safe_message))
            }
            _ => self,
        }
    }

    /// Map one failure to its public error, mirroring `public_failure_error`.
    ///
    /// Quota exhaustion omits the Python engine's month-boundary suffix because
    /// the reset boundary is computed control-plane side; the PoC returns the
    /// plain safe message with a one-hour retry hint instead.
    pub fn public_error(&self) -> PublicError {
        // The customer's own BYOK credential or account: their 400, with the
        // message that names their provider and the fix.
        if self.customer_owned {
            let code = match self.failure_class {
                FailureClass::ProviderQuota => "provider_account_quota",
                _ => "provider_credential_rejected",
            };
            return PublicError::new(400, code, &self.safe_message, "invalid_request_error");
        }
        let (status, code, error_type) = match self.failure_class {
            FailureClass::InvalidRequest => (400, "invalid_request", "invalid_request_error"),
            FailureClass::UnsupportedCapability => {
                (400, "unsupported_capability", "invalid_request_error")
            }
            FailureClass::Authentication => (401, "invalid_key", "authentication_error"),
            FailureClass::Authorization => (403, "model_not_granted", "permission_error"),
            FailureClass::QuotaExceeded => (429, "insufficient_quota", "insufficient_quota"),
            FailureClass::Throttled => (429, "unavailable_route", "api_error"),
            FailureClass::Timeout => (504, "deadline_exceeded", "api_error"),
            FailureClass::Cancelled => (499, "request_cancelled", "api_error"),
            FailureClass::Guardrail => (400, "content_filter", "invalid_request_error"),
            // A provider refusal with no visible refusal text is the model's
            // answer to the request content, not a routing failure: OpenAI
            // rejects such prompts as a 400 `invalid_request_error` ("rejected
            // as a result of our safety system"), so the closest error-shaped
            // convention is that status with its own code. The provider billed
            // the processed input, so a 502 would misdescribe a charged call.
            FailureClass::Refusal => (400, "refusal", "invalid_request_error"),
            FailureClass::Unavailable => (503, "gateway_unavailable", "api_error"),
            _ => (502, "all_routes_failed", "api_error"),
        };
        let mut error = PublicError::new(status, code, &self.safe_message, error_type);
        if self.failure_class == FailureClass::Refusal {
            // Every refusal answer names its category; a refusal built
            // without one (a visible-refusal stream, a bare stop reason) is
            // explicitly `unspecified` so clients can branch on the field.
            error.refusal_reason = Some(self.refusal_reason.unwrap_or(RefusalReason::Unspecified));
        }
        if self.failure_class == FailureClass::InvalidRequest {
            error.param = self.rejected_parameter.clone();
            if let Some(detail) = self.provider_detail.as_deref() {
                // The provider's own sentence replaces the generic "verify the
                // request fields" advice: it says which field and why, which is
                // the whole point of relaying it.
                let head = self
                    .safe_message
                    .split(';')
                    .next()
                    .unwrap_or_default()
                    .trim();
                error.message = format!("{head}: {detail}");
            }
        }
        error.retry_after_seconds = match self.failure_class {
            // A failure carrying its known throttle window advertises that
            // wait (floored at the fixed default) so the Retry-After header a
            // client honors never contradicts the message it reads.
            FailureClass::Throttled => Some(self.retry_after_seconds.map_or(5, |wait| wait.max(5))),
            FailureClass::QuotaExceeded => Some(3600),
            FailureClass::Unavailable => Some(2),
            _ => None,
        };
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_is_a_request_error_with_its_own_code_never_a_routing_failure() {
        // The provider processed (and billed) the prompt and answered with a
        // refusal; `all_routes_failed` would file the model's verdict as an
        // infrastructure fault. Mirrors `public_failure_error`.
        let refused = Failure::new(FailureClass::Refusal, "provider refused the request");
        let error = refused.public_error();
        assert_eq!(error.status_code, 400);
        assert_eq!(error.code, "refusal");
        assert_eq!(error.error_type, "invalid_request_error");
        assert_eq!(error.message, "provider refused the request");
        assert_eq!(error.retry_after_seconds, None);
        // A refusal built without a reason still names its category.
        assert_eq!(error.refusal_reason, Some(RefusalReason::Unspecified));
        assert_eq!(error.json_body()["error"]["refusal_reason"], "unspecified");
        // Untyped provider failures keep the routing-failure shape.
        let internal = Failure::new(FailureClass::ProviderInternal, "provider failed");
        assert_eq!(internal.public_error().status_code, 502);
        assert_eq!(internal.public_error().code, "all_routes_failed");
    }

    #[test]
    fn every_refusal_reason_renders_its_fixed_phrase_and_wire_name() {
        // (reason, wire name, public message): the message is the generic
        // sentence plus the reason's fixed phrase, never provider prose.
        let table = [
            (
                RefusalReason::CyberPolicy,
                "cyber_policy",
                "provider refused the request: cybersecurity policy",
            ),
            (
                RefusalReason::Cbrn,
                "cbrn",
                "provider refused the request: biological/chemical safety policy",
            ),
            (
                RefusalReason::ContentPolicy,
                "content_policy",
                "provider refused the request: content policy",
            ),
            (
                RefusalReason::Recitation,
                "recitation",
                "provider refused the request: recitation of copyrighted material",
            ),
            (
                RefusalReason::DataInspection,
                "data_inspection",
                "provider refused the request: data inspection",
            ),
            (
                RefusalReason::Unspecified,
                "unspecified",
                "provider refused the request",
            ),
        ];
        for (reason, wire, message) in table {
            let failure = Failure::refusal(reason)
                .with_provider_detail(Some("cyber_policy: raw provider prose".into()));
            assert_eq!(failure.failure_class, FailureClass::Refusal);
            assert_eq!(failure.refusal_reason, Some(reason));
            assert_eq!(reason.as_str(), wire);
            let error = failure.public_error();
            // Status, code, and type are unchanged so existing clients behave
            // the same; the reason is the one additive field.
            assert_eq!(error.status_code, 400);
            assert_eq!(error.code, "refusal");
            assert_eq!(error.error_type, "invalid_request_error");
            assert_eq!(error.message, message);
            assert_eq!(error.refusal_reason, Some(reason));
            let body = error.json_body();
            assert_eq!(body["error"]["refusal_reason"], wire);
            assert_eq!(body["error"]["code"], "refusal");
            // The provider's own detail never reaches the caller.
            assert!(!body.to_string().contains("raw provider prose"));
            // The enum round-trips through its snake_case wire name.
            let back: RefusalReason =
                serde_json::from_value(serde_json::Value::String(wire.to_string()))
                    .expect("round trip");
            assert_eq!(back, reason);
        }
        // Every other class stays free of the field, on the struct and in
        // the envelope, so non-refusal payloads are byte-identical.
        let throttled = Failure::new(FailureClass::Throttled, "slow down").public_error();
        assert_eq!(throttled.refusal_reason, None);
        assert!(throttled.json_body()["error"]
            .get("refusal_reason")
            .is_none());
        assert!(serde_json::to_value(&throttled)
            .expect("serializable")
            .get("refusal_reason")
            .is_none());
        // A boundary payload without the field still deserializes.
        let legacy: PublicError = serde_json::from_value(serde_json::json!({
            "status_code": 400, "code": "refusal", "message": "m"
        }))
        .expect("field is optional");
        assert_eq!(legacy.refusal_reason, None);
    }

    #[test]
    fn rejected_parameter_reaches_the_public_error_only_for_invalid_requests() {
        let attributed = Failure::new(FailureClass::InvalidRequest, "provider rejected")
            .with_rejected_parameter(Some("input[1].status".to_string()));
        assert_eq!(
            attributed.public_error().param.as_deref(),
            Some("input[1].status")
        );
        // Any other class stays param-free even if a parameter leaked in.
        let internal = Failure::new(FailureClass::ProviderInternal, "provider failed")
            .with_rejected_parameter(Some("input[1].status".to_string()));
        assert_eq!(internal.public_error().param, None);
        // Serde omits the field when absent, so boundary payloads are unchanged.
        let bare = serde_json::to_value(Failure::new(FailureClass::InvalidRequest, "x"))
            .expect("serializable");
        assert!(bare.get("rejected_parameter").is_none());
        let carried = serde_json::to_value(attributed).expect("serializable");
        assert_eq!(
            carried["rejected_parameter"].as_str(),
            Some("input[1].status")
        );
    }

    #[test]
    fn provider_detail_replaces_the_generic_advice_only_for_invalid_requests() {
        let explained = Failure::new(
            FailureClass::InvalidRequest,
            "provider rejected the request; verify the request fields against \
             the model alias capabilities",
        )
        .with_provider_detail(Some("`top_p` is deprecated for this model.".to_string()));
        assert_eq!(
            explained.public_error().message,
            "provider rejected the request: `top_p` is deprecated for this model."
        );
        // Any other class keeps its own message even if a detail leaked in.
        let internal = Failure::new(FailureClass::ProviderInternal, "provider failed")
            .with_provider_detail(Some("account 4711 is over its map".to_string()));
        assert_eq!(internal.public_error().message, "provider failed");
        let bare = serde_json::to_value(Failure::new(FailureClass::InvalidRequest, "x"))
            .expect("serializable");
        assert!(bare.get("provider_detail").is_none());
        let carried = serde_json::to_value(explained).expect("serializable");
        assert_eq!(
            carried["provider_detail"].as_str(),
            Some("`top_p` is deprecated for this model.")
        );
    }

    #[test]
    fn boundary_replaces_malformed_detail_with_the_generic_message() {
        let coerced = Failure::new(FailureClass::MalformedResponse, "specific detail").boundary();
        assert_eq!(
            coerced.safe_message,
            "provider returned a malformed response; retry the request"
        );
        // The specific reason survives as the settlement-facing detail so the
        // ledger names the wire shape that failed, while the public error
        // stays generic (detail relay is scoped to invalid requests).
        assert_eq!(coerced.provider_detail.as_deref(), Some("specific detail"));
        assert_eq!(
            coerced.public_error().message,
            "provider returned a malformed response; retry the request"
        );
        let transport = Failure::new(FailureClass::Transport, "kept").boundary();
        assert_eq!(transport.safe_message, "kept");
        assert_eq!(transport.provider_detail, None);
    }

    #[test]
    fn failure_classes_round_trip_through_their_wire_names() {
        for class in [
            FailureClass::InvalidRequest,
            FailureClass::QuotaExceeded,
            FailureClass::MalformedResponse,
            FailureClass::Cancelled,
            FailureClass::Unavailable,
        ] {
            let wire = serde_json::to_value(class).expect("serializable");
            let back: FailureClass = serde_json::from_value(wire).expect("round trip");
            assert_eq!(back.as_str(), class.as_str());
        }
    }

    #[test]
    fn a_throttled_failure_advertises_its_known_window_floored_at_the_default() {
        // The throttled-exhaustion message names the remaining window, so the
        // Retry-After header must advertise the same wait: a fixed 5s header
        // beside a "retry in 30s" body sends honoring clients straight back
        // into the window.
        let mut throttled = Failure::new(FailureClass::Throttled, "retry in 30s");
        throttled.retry_after_seconds = Some(30);
        assert_eq!(throttled.public_error().retry_after_seconds, Some(30));
        // A shorter-than-default window keeps the default floor.
        throttled.retry_after_seconds = Some(2);
        assert_eq!(throttled.public_error().retry_after_seconds, Some(5));
        // Without a known window the fixed default backoff applies.
        let bare = Failure::new(FailureClass::Throttled, "provider throttled the request");
        assert_eq!(bare.public_error().retry_after_seconds, Some(5));
    }

    #[test]
    fn unavailable_maps_to_a_retryable_503() {
        // Parity with the python control plane's UNAVAILABLE mapping: a
        // transient roll condition is a retryable 503, not a closed 500/502.
        let failure = Failure::new(FailureClass::Unavailable, "the gateway is updating");
        let error = failure.public_error();
        assert_eq!(error.status_code, 503);
        assert_eq!(error.code, "gateway_unavailable");
        assert_eq!(error.error_type, "api_error");
        assert_eq!(error.retry_after_seconds, Some(2));
        // The wire name matches the python GatewayFailureClass member so a
        // failure serialized on either side deserializes on the other.
        assert_eq!(FailureClass::Unavailable.as_str(), "unavailable");
    }
}
