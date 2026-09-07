//! Provider-rejected parameter attribution for sanitized 400s.
//!
//! When a provider rejects a dispatched request with a client-error status,
//! two facts may reach the caller, and only for that class: the parameter
//! path the provider named, validated against the strict path grammar below,
//! and the provider's own one-sentence explanation, read from the documented
//! message field and sanitized by [`rejected_detail`]. Nothing else from the
//! body crosses the boundary, and no other failure class relays any of it.
//!
//! Extraction classification per dialect (a new [`Dialect`] variant fails to
//! compile until it is classified here, and the exhaustiveness test pins the
//! documented source):
//!
//! | dialect                 | source                                        |
//! |-------------------------|-----------------------------------------------|
//! | `OpenAiResponses`       | `error.param`, else fixed unknown-argument msg |
//! | `OpenAiCompatible`      | `error.param`, else fixed unknown-argument msg |
//! | `AnthropicMessages`     | leading `path: ` or `` `path` `` message token |
//! | `GeminiGenerateContent` | `fieldViolations[].field`, else `* path: ` msg |
//! | `BedrockConverseStream` | none — no machine-readable parameter contract  |
//!
//! The explanation relayed alongside it comes from `error.message` for every
//! dialect except Bedrock, which reports a bare top-level `message`.

use serde_json::Value;

use crate::dialects::Dialect;

/// Longest parameter path relayed; anything longer is treated as prose.
const MAXIMUM_PATH_LENGTH: usize = 128;

/// Longest provider explanation relayed; longer text is a body dump.
const MAXIMUM_DETAIL_LENGTH: usize = 240;

/// Fixed OpenAI-family prefix naming one unsupported argument.
const UNKNOWN_ARGUMENT_PREFIXES: [&str; 2] = [
    "Unrecognized request argument supplied: ",
    "Unknown parameter: ",
];

/// Extract the provider-named parameter path from one client-error body.
///
/// Returns `Some(path)` only when the dialect's documented source yields a
/// string that passes [`valid_parameter_path`]; every other body — missing
/// fields, prose, oversized or non-path content, non-JSON — yields `None`
/// and the caller keeps the content-free sanitized message.
pub fn rejected_parameter(dialect: Dialect, body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let candidate = match dialect {
        Dialect::OpenAiResponses | Dialect::OpenAiCompatible => {
            let error = value.get("error")?;
            match error.get("param").and_then(Value::as_str) {
                Some(param) => Some(param.to_string()),
                None => unknown_argument_name(error.get("message")?.as_str()?),
            }
        }
        Dialect::AnthropicMessages => {
            let message = value.get("error")?.get("message")?.as_str()?;
            match message.split_once(": ") {
                Some((head, _rest)) if valid_parameter_path(head) => Some(head.to_string()),
                _ => quoted_leading_name(message),
            }
        }
        Dialect::GeminiGenerateContent => gemini_field_violation(&value),
        Dialect::BedrockConverseStream => None,
    }?;
    valid_parameter_path(&candidate).then_some(candidate)
}

/// OpenAI-family `error.code` naming a deployment model the provider cannot serve.
const MODEL_NOT_FOUND_CODE: &str = "model_not_found";

/// Sentences a provider answers with a 400 for a request shape the OpenAI
/// contract allows but THIS lane's serving stack cannot carry (a chat
/// template that only accepts a leading system turn). They are lane
/// limitations, not caller errors: the request fails over to the next rung and
/// only a route with no other rung surfaces the sentence.
const LANE_LIMITATION_PHRASES: &[&str] = &[
    "system message must be at the beginning",
    "system message should be at the beginning",
    "only the first message can be a system message",
];

/// The provider's own error sentence from one client-error body, read only
/// from the dialect's documented message field (never from echoed request
/// data or unrelated metadata).
fn error_message_field(dialect: Dialect, value: &Value) -> Option<&str> {
    match dialect {
        Dialect::OpenAiResponses
        | Dialect::OpenAiCompatible
        | Dialect::AnthropicMessages
        | Dialect::GeminiGenerateContent => value.get("error")?.get("message")?.as_str(),
        // Bedrock reports a modeling error as a bare top-level `message`.
        Dialect::BedrockConverseStream => value.get("message")?.as_str(),
    }
}

/// Whether a 4xx body's error SENTENCE describes a limitation of the lane
/// rather than of the caller's request (see [`LANE_LIMITATION_PHRASES`]). Only
/// the dialect's message field is read, so request text echoed elsewhere in
/// the body cannot change routing.
pub fn rejected_by_lane_limitation(dialect: Dialect, body: &str) -> bool {
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return false,
    };
    error_message_field(dialect, &value).is_some_and(|message| {
        let lowered = message.to_ascii_lowercase();
        LANE_LIMITATION_PHRASES
            .iter()
            .any(|phrase| lowered.contains(phrase))
    })
}

/// Whether a 403 body is an aggregator ROUTING verdict rather than a
/// credential one. OpenRouter runs its routing funnel only AFTER the key has
/// authenticated, and reports the funnel it walked (`metadata.routing_funnel`)
/// plus the step that refused (`metadata.failed_routing_step`; "Gate Free
/// Endpoints by Agentic Harness" on its app-allow-listed free endpoints,
/// 2026-09-06). Both fields together mean the key is fine and this lane will
/// not serve this model for the gateway's account, so it takes the not-found
/// policy and the ladder advances. Only the OpenAI-compatible wire OpenRouter
/// speaks is read; other dialects keep the credential verdict.
pub fn rejected_by_routing_gate(dialect: Dialect, body: &str) -> bool {
    if dialect != Dialect::OpenAiCompatible {
        return false;
    }
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let Some(metadata) = value.get("error").and_then(|error| error.get("metadata")) else {
        return false;
    };
    let walked_funnel = metadata
        .get("routing_funnel")
        .and_then(Value::as_array)
        .is_some_and(|steps| !steps.is_empty());
    let failed_step = metadata
        .get("failed_routing_step")
        .and_then(Value::as_str)
        .is_some_and(|step| !step.trim().is_empty());
    walked_funnel && failed_step
}

/// Whether one client-error body reports that the dispatched model does not exist.
///
/// The OpenAI Responses surface answers an unknown model with HTTP 400 and
/// `error.code = "model_not_found"` rather than the 404 that Chat Completions,
/// Anthropic, and Gemini return. The model ID comes from the catalog, never
/// from the caller, so that body is an operator misconfiguration of one rung
/// and the certified ladder must advance past it exactly as it does for a 404.
/// Only the documented code field is read; the message is never inspected.
pub fn rejected_model_not_found(dialect: Dialect, body: &str) -> bool {
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return false,
    };
    match dialect {
        Dialect::OpenAiResponses | Dialect::OpenAiCompatible => {
            value
                .get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str)
                == Some(MODEL_NOT_FOUND_CODE)
        }
        Dialect::AnthropicMessages
        | Dialect::GeminiGenerateContent
        | Dialect::BedrockConverseStream => false,
    }
}

/// Extract the provider's own explanation from one client-error body.
///
/// A client-error body explains what the caller got wrong, and the caller is
/// the only party who can act on it, so the sentence itself is worth more to
/// them than the gateway's generic wording. Only the dialect's documented
/// message field is read, and only after [`sanitized_detail`] proves it is
/// one bounded single-line sentence that names no provider infrastructure; a
/// body dump, a stack trace, or a sentence carrying a deployment, account,
/// endpoint, or request handle yields `None` and the caller keeps the
/// generic message.
///
/// This relays provider wording verbatim, so it is restricted at the call
/// site to the client-error class. Provider messages for authentication,
/// not-found, and server-side failures are operator-facing and can name
/// deployments or accounts, so they stay content-free.
///
/// `request_words` are label-shaped values the dispatched request itself
/// carried (today: the payload's own `model`). The caller already knows
/// them, so a provider sentence naming one verbatim is prose about the
/// request, never infrastructure to redact: Anthropic's client-version gate
/// names the rejected model unquoted ("claude-fable-5-1 requires Claude Code
/// 2.1.251 or later"), and dropping that sentence left callers with a
/// generic 400 for a client-side fix (2026-09-04 ledger).
pub fn rejected_detail(dialect: Dialect, body: &str, request_words: &[&str]) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let message = error_message_field(dialect, &value)?;
    if dialect == Dialect::OpenAiCompatible {
        if let Some(relayed) = value
            .get("error")
            .and_then(|error| upstream_relayed_message(error, message))
        {
            return sanitized_detail(&relayed, request_words);
        }
    }
    sanitized_detail(message, request_words)
}

/// Aggregator sentences that say nothing about what was refused.
const GENERIC_AGGREGATOR_MESSAGES: &[&str] = &["provider returned error", "provider error"];

/// The upstream provider's own sentence behind an aggregator's generic one.
///
/// OpenRouter answers a rejected relay with "Provider returned error" and
/// puts the upstream body in `error.metadata.raw` (a JSON document or plain
/// text) plus the upstream's name in `error.metadata.provider_name`. The
/// generic sentence leaves the caller nothing to act on (673 such 400s across
/// 137 orgs in the 24h to 2026-09-07 00:20 UTC), so when the aggregator's
/// message is generic the upstream sentence is read instead, prefixed with
/// the provider's name. Only the message field of a JSON `raw` is used; a
/// plain-text `raw` is taken whole. The result still passes the same
/// identifier screen and length bound as any relayed sentence.
pub fn upstream_relayed_message(error: &Value, message: &str) -> Option<String> {
    let lowered = message.trim().trim_end_matches('.').to_ascii_lowercase();
    if !GENERIC_AGGREGATOR_MESSAGES.contains(&lowered.as_str()) {
        return None;
    }
    let metadata = error.get("metadata")?;
    let raw = metadata.get("raw")?;
    let upstream = match raw {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(document) => json_error_sentence(&document)?,
            Err(_) => text.trim().to_string(),
        },
        Value::Object(_) => json_error_sentence(raw)?,
        _ => return None,
    };
    if upstream.is_empty() {
        return None;
    }
    let provider = metadata
        .get("provider_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| {
            !name.is_empty()
                && name.chars().all(|c| {
                    c.is_ascii_alphanumeric() || c == ' ' || c == '-' || c == '_' || c == '.'
                })
        });
    Some(match provider {
        Some(name) => format!("{name}: {upstream}"),
        None => upstream,
    })
}

/// The human sentence of one upstream error document, whichever documented
/// spelling it uses (`error.message`, `message`, `detail`, or a bare string
/// `error`).
fn json_error_sentence(document: &Value) -> Option<String> {
    let error = document.get("error");
    let candidates = [
        error.and_then(|error| error.get("message")),
        document.get("message"),
        document.get("detail"),
        error.filter(|value| value.is_string()),
    ];
    candidates
        .into_iter()
        .flatten()
        .find_map(|value| value.as_str())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

/// The provider's own error code or type from one client-error body, as a
/// bounded identifier token (`invalid_value`, `content_filter`,
/// `invalid_request_error`, `INVALID_ARGUMENT`, or a numeric status).
///
/// Read only from the dialect's documented code field; it is a vocabulary
/// token, never prose, so it is safe to relay when [`rejected_detail`] has to
/// drop the sentence (a provider explanation naming a request or account
/// handle otherwise left the caller with nothing but "verify the request
/// fields"). It also classifies the body: a content-filter code is the
/// model's verdict on the content, not a request-shape error.
pub fn rejected_code(dialect: Dialect, body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let error = match dialect {
        Dialect::BedrockConverseStream => return None,
        _ => value.get("error")?,
    };
    let candidate = match dialect {
        Dialect::GeminiGenerateContent => error.get("status").or_else(|| error.get("code")),
        _ => error
            .get("code")
            .filter(|code| !code.is_null())
            .or_else(|| error.get("type")),
    }?;
    let token = match candidate {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        _ => return None,
    };
    let identifier = !token.is_empty()
        && token.len() <= 64
        && token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    identifier.then_some(token)
}

/// Whether one provider code token says nothing beyond "the request was
/// rejected": the family-wide type every 4xx carries, or a bare numeric
/// status. Such a token is still classified but never relayed as detail, so a
/// body whose sentence had to drop keeps the generic message rather than
/// gaining a meaningless suffix.
pub fn generic_error_code(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "invalid_request_error" | "invalid_request" | "bad_request" | "error" | "invalid_argument"
    ) || lower.chars().all(|c| c.is_ascii_digit())
}

/// One provider sentence reduced to bounded, single-line, printable text.
///
/// Control characters end the candidate rather than being escaped: their
/// presence means the field carries a payload, not a sentence. Interior runs
/// of spaces and tabs collapse so the relayed text stays one readable line,
/// and [`carries_provider_identifier`] then rejects any sentence naming
/// provider-side infrastructure, except words the request itself carried.
fn sanitized_detail(message: &str, request_words: &[&str]) -> Option<String> {
    let trimmed = message.trim();
    if trimmed.is_empty() || trimmed.chars().count() > MAXIMUM_DETAIL_LENGTH {
        return None;
    }
    if trimmed
        .chars()
        .any(|c| (c.is_control() && c != '\t') || (c.is_whitespace() && c != ' ' && c != '\t'))
    {
        return None;
    }
    let collapsed = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty()
        || collapsed
            .split(' ')
            .any(|word| carries_provider_identifier(word, request_words))
    {
        return None;
    }
    Some(collapsed)
}

/// Whether one word of a provider sentence names provider-side infrastructure.
///
/// An explanation the caller can act on is prose about their own request, so
/// it never needs an endpoint, a mailbox, a resource name, or an opaque
/// handle. Any unquoted word shaped like one disqualifies the whole sentence:
/// a partially redacted explanation reads as fact while hiding what was cut,
/// and the caller keeps the generic message instead. A quoted word is the
/// value the caller sent back to them, so only the unambiguous network and
/// resource shapes disqualify it; a word matching one of `request_words`
/// (label-shaped values the dispatched payload itself carried, such as its
/// `model`) is the same caller-known exception without the quotes, though
/// the unambiguous network and resource shapes still disqualify it.
pub(crate) fn carries_provider_identifier(word: &str, request_words: &[&str]) -> bool {
    let bare = word.trim_matches(|c: char| !c.is_alphanumeric());
    if word.contains("://") || word.contains('@') || bare.to_ascii_lowercase().starts_with("arn:") {
        return true;
    }
    if bare.len() == 36 && bare.chars().filter(|c| *c == '-').count() == 4 {
        return true;
    }
    if request_words
        .iter()
        .any(|known| bare.eq_ignore_ascii_case(known))
    {
        return false;
    }
    let dotted: Vec<&str> = bare.split('.').collect();
    if dotted.len() == 4
        && dotted
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
    {
        return true;
    }
    // A bare word mixing letters and digits is a label rather than English:
    // an account, a deployment, a region, a revision, or a key. Length is not
    // part of the test, so `prod-7` and `acct-123` are as disqualifying as a
    // full opaque handle. Prose keeps its numbers (`8192`) and parameter
    // names keep their shape (`top_p`, `input[1].status`), because neither
    // mixes the two inside one unpunctuated word.
    //
    // Quoting is the exception. A provider quotes the value the caller sent
    // (`Unsupported model 'gpt-4o-mini'.`) and states its own infrastructure
    // unquoted, so a quoted label is something the caller already knows and
    // needs to see named.
    if word.starts_with(['\'', '"', '`']) {
        return false;
    }
    let label = bare
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    label
        && bare.chars().any(|c| c.is_ascii_digit())
        && bare.chars().any(|c| c.is_ascii_alphabetic())
}

/// The argument named after one fixed OpenAI-family unknown-argument prefix.
///
/// Azure's OpenAI surface reports an unsupported sampling field this way and
/// carries no `param`, so the name lives in an otherwise fixed sentence. Only
/// the trailing name is read, and only when the sentence matches exactly.
fn unknown_argument_name(message: &str) -> Option<String> {
    UNKNOWN_ARGUMENT_PREFIXES.iter().find_map(|prefix| {
        let name = message.strip_prefix(prefix)?.trim_end_matches('.');
        Some(name.trim_matches('\'').to_string())
    })
}

/// The backtick-quoted name one message opens with.
///
/// Anthropic reports a per-model sampling refusal as `` `top_p` is deprecated
/// for this model. ``, naming the field only inside the prose it otherwise
/// owns. Only the quoted leading token is read; the prose is discarded.
fn quoted_leading_name(message: &str) -> Option<String> {
    let rest = message.strip_prefix('`')?;
    let (name, _prose) = rest.split_once('`')?;
    Some(name.to_string())
}

/// `fieldViolations[].field` when a `google.rpc.BadRequest` detail exists,
/// else the leading `* <path>: ` token of the message (the shape the live
/// API returned for a generation-config violation, 2026-08-29).
fn gemini_field_violation(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    if let Some(details) = error.get("details").and_then(Value::as_array) {
        for detail in details {
            let type_url = detail.get("@type").and_then(Value::as_str).unwrap_or("");
            if !type_url.ends_with("google.rpc.BadRequest") {
                continue;
            }
            if let Some(field) = detail
                .get("fieldViolations")
                .and_then(Value::as_array)
                .and_then(|violations| {
                    violations
                        .iter()
                        .find_map(|violation| violation.get("field").and_then(Value::as_str))
                })
            {
                return Some(field.to_string());
            }
        }
    }
    let message = error.get("message")?.as_str()?;
    let head = message.strip_prefix("* ")?;
    let (path, _rest) = head.split_once(": ")?;
    Some(path.to_string())
}

/// Whether one candidate is a parameter path and cannot be prose.
///
/// Grammar: ASCII segments of `[A-Za-z0-9_-]` joined by `.`, with optional
/// numeric `[N]` indexes; no whitespace, no empty segments, bounded length.
/// This is deliberately narrower than what providers could emit: a rejected
/// candidate costs only attribution, while an accepted one crosses the
/// sanitization boundary.
pub fn valid_parameter_path(candidate: &str) -> bool {
    if candidate.is_empty() || candidate.len() > MAXIMUM_PATH_LENGTH {
        return false;
    }
    if !candidate.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return false;
    }
    let mut chars = candidate.chars().peekable();
    let mut segment_open = false;
    while let Some(c) = chars.next() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => segment_open = true,
            '.' => {
                if !segment_open || chars.peek().is_none() {
                    return false;
                }
                segment_open = false;
            }
            '[' => {
                if !segment_open {
                    return false;
                }
                let mut digits = 0usize;
                loop {
                    match chars.next() {
                        Some(d) if d.is_ascii_digit() => digits += 1,
                        Some(']') if digits > 0 => break,
                        _ => return false,
                    }
                }
            }
            _ => return false,
        }
    }
    segment_open
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_param_field_is_relayed_when_it_is_a_path() {
        // Exact body captured live from api.openai.com (2026-08-29).
        let body = r#"{"error": {"message": "Unknown parameter: 'input[1].status'.",
            "type": "invalid_request_error", "param": "input[1].status",
            "code": "unknown_parameter"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiResponses, body).as_deref(),
            Some("input[1].status")
        );
        assert_eq!(
            rejected_parameter(Dialect::OpenAiCompatible, body).as_deref(),
            Some("input[1].status")
        );
    }

    #[test]
    fn anthropic_leading_message_token_is_relayed_when_it_is_a_path() {
        // Anthropic names the field as a leading `path: ` message token.
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "context_management: Extra inputs are not permitted"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, body).as_deref(),
            Some("context_management")
        );
        let nested = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "messages.1.content.0.text: Field required"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, nested).as_deref(),
            Some("messages.1.content.0.text")
        );
    }

    #[test]
    fn anthropic_backtick_quoted_leading_name_is_relayed() {
        // Exact body captured live from api.anthropic.com (2026-08-29): the
        // per-model sampling refusal names the field only inside prose.
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "`top_p` is deprecated for this model."}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, body).as_deref(),
            Some("top_p")
        );
        // A quoted phrase is prose, not a path, so nothing is relayed.
        let quoted_prose = r#"{"type": "error", "error": {
            "message": "`contact support` for account 4821 to continue."}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, quoted_prose),
            None
        );
    }

    #[test]
    fn openai_family_unknown_argument_sentence_is_relayed_without_a_param() {
        // Exact body captured live from Azure's OpenAI surface (2026-08-29).
        let body = r#"{"error": {"code": "unrecognized_request_argument",
            "message": "Unrecognized request argument supplied: top_k",
            "details": "Unrecognized request argument supplied: top_k"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiCompatible, body).as_deref(),
            Some("top_k")
        );
        let quoted = r#"{"error": {"message": "Unknown parameter: 'response_format'."}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiResponses, quoted).as_deref(),
            Some("response_format")
        );
        // Any other message keeps the content-free failure.
        let other = r#"{"error": {"message": "This model is not available to your account."}}"#;
        assert_eq!(rejected_parameter(Dialect::OpenAiCompatible, other), None);
    }

    #[test]
    fn an_aggregators_generic_sentence_yields_to_the_upstream_message() {
        let body = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"{\"error\":{\"message\":\"Input exceeds the maximum context window.\",\"type\":\"invalid_request_error\"}}",
            "provider_name":"Relace"}}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, body, &[]).as_deref(),
            Some("Relace: Input exceeds the maximum context window.")
        );
        // A plain-text raw is relayed whole; a specific aggregator sentence is kept.
        let plain = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"model is overloaded, try again","provider_name":"Fugu"}}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, plain, &[]).as_deref(),
            Some("Fugu: model is overloaded, try again")
        );
        let specific = r#"{"error":{"message":"temperature must be <= 1","code":400,
            "metadata":{"raw":"{\"error\":{\"message\":\"ignored\"}}"}}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, specific, &[]).as_deref(),
            Some("temperature must be <= 1")
        );
        // Without metadata the generic sentence stays what it was.
        let bare = r#"{"error":{"message":"Provider returned error","code":400}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, bare, &[]).as_deref(),
            Some("Provider returned error")
        );
    }

    #[test]
    fn rejected_code_reads_the_documented_code_field_as_a_bounded_token() {
        let openai = r#"{"error": {"code": "invalid_value", "type": "invalid_request_error", "message": "x"}}"#;
        assert_eq!(
            rejected_code(Dialect::OpenAiResponses, openai).as_deref(),
            Some("invalid_value")
        );
        let typed = r#"{"error": {"code": null, "type": "invalid_request_error", "message": "x"}}"#;
        assert_eq!(
            rejected_code(Dialect::OpenAiCompatible, typed).as_deref(),
            Some("invalid_request_error")
        );
        let numeric = r#"{"error": {"code": 400, "message": "x"}}"#;
        assert_eq!(
            rejected_code(Dialect::OpenAiCompatible, numeric).as_deref(),
            Some("400")
        );
        let anthropic =
            r#"{"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}"#;
        assert_eq!(
            rejected_code(Dialect::AnthropicMessages, anthropic).as_deref(),
            Some("invalid_request_error")
        );
        let gemini = r#"{"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "x"}}"#;
        assert_eq!(
            rejected_code(Dialect::GeminiGenerateContent, gemini).as_deref(),
            Some("INVALID_ARGUMENT")
        );
        // Prose or hostile shapes are never a token; Bedrock has no code field.
        let prose = r#"{"error": {"code": "not a code!{}", "message": "x"}}"#;
        assert_eq!(rejected_code(Dialect::OpenAiCompatible, prose), None);
        assert_eq!(
            rejected_code(Dialect::BedrockConverseStream, r#"{"message": "x"}"#),
            None
        );
    }

    #[test]
    fn gemini_message_path_token_is_relayed_without_violation_details() {
        // Exact live shape from generativelanguage.googleapis.com (2026-08-29).
        let body = r#"{"error": {"code": 400, "status": "INVALID_ARGUMENT",
            "message": "* GenerateContentRequest.generation_config.temperature: temperature must be in the range [0.0, 2.0].\n"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::GeminiGenerateContent, body).as_deref(),
            Some("GenerateContentRequest.generation_config.temperature")
        );
        // Prose-leading messages stay content-free.
        let prose = r#"{"error": {"code": 400, "message": "API key not valid: renew it"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::GeminiGenerateContent, prose),
            None
        );
    }

    #[test]
    fn gemini_bad_request_field_violation_is_relayed() {
        let body = r#"{"error": {"code": 400, "status": "INVALID_ARGUMENT",
            "message": "Invalid JSON payload received.",
            "details": [{"@type": "type.googleapis.com/google.rpc.BadRequest",
                "fieldViolations": [{"field": "generation_config.temperature",
                    "description": "out of range"}]}]}}"#;
        assert_eq!(
            rejected_parameter(Dialect::GeminiGenerateContent, body).as_deref(),
            Some("generation_config.temperature")
        );
    }

    #[test]
    fn bedrock_never_attributes() {
        let body = r#"{"message": "Malformed input request: extraneous key [topK]"}"#;
        assert_eq!(
            rejected_parameter(Dialect::BedrockConverseStream, body),
            None
        );
    }

    #[test]
    fn provider_prose_never_crosses_the_boundary() {
        // A prose-bearing param field, prose-only Anthropic messages, and
        // adversarial path-shaped content all stay content-free.
        let prose_param = r#"{"error": {"param": "please contact support at example.com"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiResponses, prose_param),
            None
        );
        let no_path_message =
            r#"{"type": "error", "error": {"message": "Extra inputs are not permitted"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, no_path_message),
            None
        );
        let prose_head = r#"{"error": {"message": "Your credit balance is too low: top up"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, prose_head),
            None
        );
        let not_json = "upstream said no";
        assert_eq!(rejected_parameter(Dialect::OpenAiResponses, not_json), None);
    }

    #[test]
    fn the_path_grammar_is_strict() {
        for accepted in [
            "temperature",
            "input[1].status",
            "messages.1.content.0.text",
            "tools[0].function.name",
            "generation_config.top_k",
            "anthropic-beta",
        ] {
            assert!(valid_parameter_path(accepted), "{accepted}");
        }
        for rejected in [
            "",
            "has space",
            "trailing.",
            ".leading",
            "double..dot",
            "input[].status",
            "input[1.status",
            "input[a]",
            "9starts_with_digit",
            "unicode_ĸey",
            "a]b",
            "semi;colon",
            "path\nnewline",
        ] {
            assert!(!valid_parameter_path(rejected), "{rejected}");
        }
        assert!(!valid_parameter_path(&"x".repeat(129)));
    }

    /// The documented extraction source for one dialect.
    ///
    /// The match is exhaustive on purpose: adding a dialect without deciding
    /// its attribution contract fails this drift gate at compile time, and
    /// `rejected_parameter`'s own exhaustive match enforces the same in the
    /// production path.
    fn extraction_classification(dialect: Dialect) -> &'static str {
        match dialect {
            Dialect::OpenAiResponses | Dialect::OpenAiCompatible => {
                "error.param field, else fixed unknown-argument message"
            }
            Dialect::AnthropicMessages => "leading path or backtick-quoted token of error.message",
            Dialect::GeminiGenerateContent => {
                "google.rpc.BadRequest fieldViolations, else leading message path token"
            }
            Dialect::BedrockConverseStream => "none: no machine-readable parameter contract",
        }
    }

    #[test]
    fn provider_explanation_is_relayed_for_every_dialect_message_field() {
        // Exact shapes captured live from each provider (2026-08-29).
        let openai = r#"{"error": {"message": "Unknown parameter: 'top_k'.",
            "type": "invalid_request_error"}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, openai, &[]).as_deref(),
            Some("Unknown parameter: 'top_k'.")
        );
        let anthropic = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "`top_p` is deprecated for this model."}}"#;
        assert_eq!(
            rejected_detail(Dialect::AnthropicMessages, anthropic, &[]).as_deref(),
            Some("`top_p` is deprecated for this model.")
        );
        let bedrock = r#"{"message": "The provided model does not support tool use."}"#;
        assert_eq!(
            rejected_detail(Dialect::BedrockConverseStream, bedrock, &[]).as_deref(),
            Some("The provided model does not support tool use.")
        );
    }

    #[test]
    fn provider_explanation_is_dropped_when_it_is_not_one_bounded_sentence() {
        let multiline = r#"{"error": {"message": "failed\n  at deployment-7\n"}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, multiline, &[]),
            None
        );
        let oversized = format!(
            r#"{{"error": {{"message": "{}"}}}}"#,
            "x".repeat(MAXIMUM_DETAIL_LENGTH + 1)
        );
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, &oversized, &[]),
            None
        );
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, "{}", &[]), None);
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, "<html>", &[]),
            None
        );
        let blank = r#"{"error": {"message": "   "}}"#;
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, blank, &[]), None);
    }

    #[test]
    fn provider_explanation_is_dropped_when_it_names_provider_infrastructure() {
        // One readable sentence each, differing only in the operator-facing
        // value the provider chose to echo back.
        for message in [
            "The deployment gpt4o-prod-7f2a91be44 is not configured for this account.",
            "Model access denied for account 5f4dcc3b5aa765d61d8327deb882cf99.",
            "Request 3f8a1c2e-9b44-4d17-9a1e-77c0d2b8e451 failed validation.",
            "Route your request to https://eastus2.api.internal.example.com instead.",
            "Contact platform-oncall@example.com about this quota.",
            "The endpoint 10.42.117.8 rejected the model.",
            "Deployment prod-7 is retired.",
            "Quota exhausted for acct-123.",
            "Use region eastus2 instead.",
            "Model arn:aws:bedrock:us-east-1:481516234299:model/private is unavailable.",
        ] {
            let body = format!(r#"{{"error": {{"message": "{message}"}}}}"#);
            assert_eq!(
                rejected_detail(Dialect::OpenAiCompatible, &body, &[]),
                None,
                "relayed an identifier-bearing sentence: {message}"
            );
        }
        // Ordinary caller-actionable prose stays relayable, including the
        // punctuation and short numbers that appear in parameter complaints.
        for message in [
            "`top_p` is deprecated for this model.",
            "Unsupported value: 'input[1].status' is not one of the allowed values.",
            "max_tokens must be less than or equal to 8192, got 100000.",
            "Unsupported model 'gpt-4o-mini' for the Responses API.",
            "`v2` is not a valid value for `api_version`.",
        ] {
            let body = format!(r#"{{"error": {{"message": "{message}"}}}}"#);
            assert_eq!(
                rejected_detail(Dialect::OpenAiCompatible, &body, &[]).as_deref(),
                Some(message),
                "dropped a caller-actionable sentence: {message}"
            );
        }
    }

    #[test]
    fn relayed_explanation_collapses_interior_whitespace_runs() {
        let padded = r#"{"error": {"message": "  Unknown   parameter:\t'top_k'.  "}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, padded, &[]).as_deref(),
            Some("Unknown parameter: 'top_k'.")
        );
    }

    #[test]
    fn openai_responses_model_not_found_code_marks_a_missing_deployment() {
        // Exact 400 body captured live from api.openai.com/v1/responses (2026-09-02).
        let body = r#"{"error": {"message": "The requested model 'gpt-5-mini-does-not-exist' does not exist.",
            "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}"#;
        assert!(rejected_model_not_found(Dialect::OpenAiResponses, body));
        assert!(rejected_model_not_found(Dialect::OpenAiCompatible, body));
    }

    #[test]
    fn other_client_errors_and_dialects_are_not_missing_deployments() {
        let body = r#"{"error": {"message": "The requested model 'x' does not exist.",
            "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}"#;
        for dialect in [
            Dialect::AnthropicMessages,
            Dialect::GeminiGenerateContent,
            Dialect::BedrockConverseStream,
        ] {
            assert!(!rejected_model_not_found(dialect, body), "{dialect:?}");
        }
        let unknown = r#"{"error": {"message": "Unknown parameter: 'top_k'.",
            "type": "invalid_request_error", "param": "top_k", "code": "unknown_parameter"}}"#;
        assert!(!rejected_model_not_found(Dialect::OpenAiResponses, unknown));
        let prose =
            r#"{"error": {"message": "The requested model does not exist.", "code": null}}"#;
        assert!(!rejected_model_not_found(Dialect::OpenAiResponses, prose));
        assert!(!rejected_model_not_found(
            Dialect::OpenAiResponses,
            "model_not_found"
        ));
    }

    #[test]
    fn every_dialect_carries_an_explicit_extraction_classification() {
        for dialect in [
            Dialect::OpenAiResponses,
            Dialect::AnthropicMessages,
            Dialect::OpenAiCompatible,
            Dialect::GeminiGenerateContent,
            Dialect::BedrockConverseStream,
        ] {
            assert!(!extraction_classification(dialect).is_empty());
        }
    }
}

#[cfg(test)]
mod request_word_tests {
    use super::*;
    use crate::dialects::Dialect;

    #[test]
    fn a_sentence_naming_the_requests_own_model_is_relayed() {
        // The Anthropic client-version gate names the rejected model unquoted;
        // the caller sent that id, so redacting the sentence hid the one
        // actionable fact (upgrade the client) behind a generic 400.
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "claude-fable-5-1 requires Claude Code version 2.1.251 or later. Please upgrade Claude Code to continue."}}"#;
        assert_eq!(
            rejected_detail(Dialect::AnthropicMessages, body, &[]),
            None,
            "without the request's own words the label-shaped model id still redacts"
        );
        assert_eq!(
            rejected_detail(Dialect::AnthropicMessages, body, &["claude-fable-5-1"]).as_deref(),
            Some(
                "claude-fable-5-1 requires Claude Code version 2.1.251 or later. \
                 Please upgrade Claude Code to continue."
            )
        );
    }

    #[test]
    fn request_words_do_not_admit_other_identifiers_in_the_same_sentence() {
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "claude-fable-5-1 is retired on deployment prod-7f2a; contact your operator."}}"#;
        assert_eq!(
            rejected_detail(Dialect::AnthropicMessages, body, &["claude-fable-5-1"]),
            None,
            "an infrastructure label beside the known word still redacts the sentence"
        );
    }

    #[test]
    fn request_words_never_admit_network_or_resource_shapes() {
        // Even a caller-supplied value keeps the unambiguous network and
        // resource shapes redacted: relaying an ARN or address helps nobody.
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "Model arn:aws:bedrock:us-east-1:123:model/x is unavailable."}}"#;
        assert_eq!(
            rejected_detail(
                Dialect::AnthropicMessages,
                body,
                &["arn:aws:bedrock:us-east-1:123:model/x"],
            ),
            None
        );
    }
}
