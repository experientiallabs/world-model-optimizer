//! Bedrock ConverseStream frame mapping, mirroring the python
//! `BedrockProviderStream._decode` mapper, plus its golden-fixture tests.

use serde_json::{Map, Value};

use super::{malformed, parse_object, Normalizer};
use crate::errors::{Failure, FailureClass};
use crate::events::{bedrock_usage, require_string, require_u64, Event, ToolAccumulator};

/// The bounded single-line detail of one Bedrock exception frame.
///
/// Exception messages name the mechanism (model stream errors, service
/// unavailability) that the typed class alone cannot; the detail is attached
/// only to stream-failure classes that never relay `provider_detail` to
/// callers, so it reaches the ledger without widening the caller-facing
/// sanitization boundary.
fn bedrock_exception_detail(frame: &crate::sse::SseEvent) -> Option<String> {
    let message = serde_json::from_str::<Value>(&frame.data)
        .ok()
        .and_then(|payload| {
            payload
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let detail = super::provider_error_detail(None, message.as_deref(), &[]);
    if let Some(detail) = &detail {
        // The same structured operator line the other dialects emit, so a
        // Bedrock-declared failure is equally visible in the immediate
        // diagnostic stream.
        super::log_provider_declared_failure("bedrock_converse_stream", detail);
    }
    detail
}

impl Normalizer {
    /// Normalize one Bedrock ConverseStream frame, mirroring the python
    /// `BedrockProviderStream._decode` mapper: tool calls stream as indexed
    /// content blocks, `messageStop` retains the stop reason, and the trailing
    /// `metadata` frame flushes usage and maps the retained reason to the
    /// shared terminal outcome. Service exceptions arrive as their own frames
    /// and map to the python mapper's failure classes.
    pub(super) fn feed_bedrock(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        match frame.event.as_deref().unwrap_or("") {
            "messageStart" => Ok(Vec::new()),
            "contentBlockStart" => self.bedrock_content_start(frame),
            "contentBlockDelta" => self.bedrock_content_delta(frame),
            "contentBlockStop" => self.bedrock_content_stop(frame),
            "messageStop" => {
                let payload = parse_object(&frame.data)?;
                self.stop_reason = Some(
                    require_string(&payload, "stopReason", "Bedrock stopReason")
                        .map_err(|message| malformed(&message))?,
                );
                Ok(Vec::new())
            }
            "metadata" => self.bedrock_metadata(frame),
            "throttlingException" => Ok(vec![Event::Failed(
                Failure::new(FailureClass::Throttled, "provider throttled the request")
                    .with_provider_detail(bedrock_exception_detail(frame)),
            )]),
            "modelTimeoutException" => Ok(vec![Event::Failed(
                Failure::new(FailureClass::Timeout, "provider request timed out")
                    .with_provider_detail(bedrock_exception_detail(frame)),
            )]),
            "internalServerException"
            | "modelStreamErrorException"
            | "serviceUnavailableException" => Ok(vec![Event::Failed(
                Failure::new(FailureClass::ProviderInternal, "provider stream failed")
                    .with_provider_detail(bedrock_exception_detail(frame)),
            )]),
            "validationException" => Ok(vec![Event::Failed(Failure::new(
                FailureClass::InvalidRequest,
                "provider rejected the request",
            ))]),
            other => Err(malformed(&format!(
                "Bedrock stream emitted an unsupported event (type {})",
                super::bounded_wire_token(other),
            ))),
        }
    }

    /// Start one Bedrock tool call, or accept an empty text-block start.
    fn bedrock_content_start(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let index = require_u64(&payload, "contentBlockIndex", "Bedrock contentBlockIndex")
            .map_err(|message| malformed(&message))? as u32;
        let start = match payload.get("start") {
            None => Map::new(),
            Some(Value::Object(map)) => map.clone(),
            Some(_) => {
                return Err(malformed(
                    "Bedrock contentBlockStart.start must be an object",
                ))
            }
        };
        let raw_tool = match start.get("toolUse") {
            None | Some(Value::Null) => {
                if start.is_empty() || start.contains_key("reasoningContent") {
                    return Ok(Vec::new());
                }
                return Err(malformed(&format!(
                    "Bedrock content block start is unsupported (key {})",
                    super::bounded_wire_token(start.keys().next().map_or("", String::as_str)),
                )));
            }
            Some(value) => value,
        };
        let tool = raw_tool
            .as_object()
            .ok_or_else(|| malformed("Bedrock toolUse start must be an object"))?;
        if self.tools.contains_key(&index) {
            return Err(malformed("Bedrock stream repeated a tool-call start"));
        }
        let call_id = require_string(tool, "toolUseId", "Bedrock toolUseId")
            .map_err(|message| malformed(&message))?;
        let name = require_string(tool, "name", "Bedrock tool name")
            .map_err(|message| malformed(&message))?;
        self.reserve_tool_entry(index)?;
        self.tools
            .insert(index, ToolAccumulator::new(call_id.clone(), name.clone()));
        Ok(vec![Event::ToolCallStarted {
            index,
            call_id,
            name,
            namespace: None,
            caller: None,
        }])
    }

    /// Normalize one Bedrock text or raw tool-input fragment.
    fn bedrock_content_delta(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let index = require_u64(&payload, "contentBlockIndex", "Bedrock contentBlockIndex")
            .map_err(|message| malformed(&message))? as u32;
        let delta = payload
            .get("delta")
            .and_then(Value::as_object)
            .ok_or_else(|| malformed("Bedrock contentBlockDelta.delta must be an object"))?;
        if let Some(Value::String(text)) = delta.get("text") {
            if text.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![Event::TextDelta(text.clone())]);
        }
        let raw_tool = match delta.get("toolUse") {
            None | Some(Value::Null) => {
                // A reasoning model streams its thinking as its own indexed
                // block ahead of the answer text. The canonical event stream
                // carries answer text and tool calls, so the thinking block
                // and its deltas are accepted and dropped instead of failing
                // the stream.
                if delta.contains_key("reasoningContent") {
                    return Ok(Vec::new());
                }
                return Err(malformed(&format!(
                    "Bedrock content block delta is unsupported (key {})",
                    super::bounded_wire_token(delta.keys().next().map_or("", String::as_str)),
                )));
            }
            Some(value) => value,
        };
        if !self.tools.contains_key(&index) {
            return Err(malformed("Bedrock emitted arguments before a tool start"));
        }
        let tool_delta = raw_tool
            .as_object()
            .ok_or_else(|| malformed("Bedrock toolUse delta must be an object"))?;
        let fragment = match tool_delta.get("input") {
            Some(Value::String(fragment)) => fragment.clone(),
            _ => return Err(malformed("Bedrock tool input delta must be text")),
        };
        if fragment.is_empty() {
            return Ok(Vec::new());
        }
        self.reserve_tool_bytes(fragment.len())?;
        let tool = self.tools.get_mut(&index).expect("tool just checked");
        tool.raw_arguments.push_str(&fragment);
        Ok(vec![Event::ToolArgumentsDelta {
            index,
            delta: fragment,
        }])
    }

    /// Complete one open Bedrock tool call at its content-block stop.
    fn bedrock_content_stop(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let index = require_u64(&payload, "contentBlockIndex", "Bedrock contentBlockIndex")
            .map_err(|message| malformed(&message))? as u32;
        let Some(mut tool) = self.tools.remove(&index) else {
            return Ok(Vec::new());
        };
        let mut events = Vec::new();
        // The stop reason arrives in the following messageStop, so a fragment
        // left open by the output budget cannot be told from garbage yet.
        self.complete_tool_deferring_failure(index, &mut tool, &mut events);
        Ok(events)
    }

    /// Flush Bedrock usage and, once the stop reason is retained, terminate.
    fn bedrock_metadata(&mut self, frame: &crate::sse::SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let usage = bedrock_usage(payload.get("usage")).map_err(|message| malformed(&message))?;
        let mut events = vec![Event::Usage(usage)];
        if let Some(reason) = self.stop_reason.take() {
            events.push(self.bedrock_terminal(&reason));
        }
        Ok(events)
    }

    /// Map the retained Bedrock stop reason to one terminal gateway event.
    fn bedrock_terminal(&mut self, reason: &str) -> Event {
        let truncated = matches!(reason, "max_tokens" | "model_context_window_exceeded");
        if let Err(failure) = self.resolve_deferred_tool_failure(truncated) {
            self.tools.clear();
            return Event::Failed(failure);
        }
        if !self.tools.is_empty() {
            self.tools.clear();
            return Event::Failed(Failure::new(
                FailureClass::MalformedResponse,
                "provider stream ended with an incomplete tool call",
            ));
        }
        match reason {
            "end_turn" | "stop_sequence" | "tool_use" => Event::Completed,
            "max_tokens" | "model_context_window_exceeded" => Event::Incomplete,
            // The Bedrock stop reason names the content verdict.
            "content_filtered" | "guardrail_intervened" => Event::Failed(Failure::refusal(
                crate::stream_errors::refusal_reason(Some(reason), None),
            )),
            _ => Event::Failed(Failure::new(
                FailureClass::ProviderInternal,
                "provider ended the stream unexpectedly",
            )),
        }
    }
}

#[cfg(test)]
mod bedrock_tests {
    use super::super::{drain_stream_fixture, Dialect};
    use super::*;
    use crate::eventstream::encode_message;
    use serde_json::json;

    fn run_stream(chunks: &[Vec<u8>]) -> (Vec<Value>, Option<Failure>) {
        drain_stream_fixture(Dialect::BedrockConverseStream, chunks)
    }

    fn event(name: &str, payload: &Value) -> Vec<u8> {
        encode_message(
            &[(":message-type", "event"), (":event-type", name)],
            payload.to_string().as_bytes(),
        )
    }

    fn exception(name: &str) -> Vec<u8> {
        encode_message(
            &[(":message-type", "exception"), (":exception-type", name)],
            br#"{"message":"redacted"}"#,
        )
    }

    #[test]
    fn bedrock_golden_stream_normalizes_text_tools_usage_and_completion() {
        // Golden fixture: raw provider bytes in, exact canonical events out.
        // `native_dialect_parity_test.py` holds the python-mapper comparison.
        let chunks = vec![
            event("messageStart", &json!({"role": "assistant"})),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
            ),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"text": "lo"}}),
            ),
            event("contentBlockStop", &json!({"contentBlockIndex": 0})),
            event(
                "contentBlockStart",
                &json!({
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": "{\"city\":"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": "\"Zürich\"}"}},
                }),
            ),
            event("contentBlockStop", &json!({"contentBlockIndex": 1})),
            event("messageStop", &json!({"stopReason": "tool_use"})),
            event(
                "metadata",
                &json!({
                    "usage": {
                        "inputTokens": 9,
                        "outputTokens": 4,
                        "cacheReadInputTokens": 2,
                        "cacheWriteInputTokens": 1,
                    },
                    "metrics": {"latencyMs": 12},
                }),
            ),
        ];
        let (events, failure) = run_stream(&chunks);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "Hel"}),
                json!({"kind": "text_delta", "text": "lo"}),
                json!({"kind": "tool_call_started", "index": 1, "call_id": "call-1", "name": "lookup"}),
                json!({"kind": "tool_arguments_delta", "index": 1, "text": "{\"city\":"}),
                json!({"kind": "tool_arguments_delta", "index": 1, "text": "\"Zürich\"}"}),
                json!({
                    "kind": "tool_call_completed",
                    "index": 1,
                    "call_id": "call-1",
                    "name": "lookup",
                    "raw_arguments": "{\"city\":\"Zürich\"}",
                }),
                json!({
                    "kind": "usage",
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cached_input_tokens": 2,
                    "reasoning_tokens": null,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn bedrock_reasoning_blocks_stream_without_failing_the_answer() {
        // A reasoning model leads its turn with an indexed thinking block; only
        // the answer text reaches the canonical stream.
        let chunks = vec![
            event("messageStart", &json!({"role": "assistant"})),
            event(
                "contentBlockStart",
                &json!({"contentBlockIndex": 0, "start": {"reasoningContent": {}}}),
            ),
            event(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "a circle"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"signature": "sig"}},
                }),
            ),
            event("contentBlockStop", &json!({"contentBlockIndex": 0})),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 1, "delta": {"text": "Circle"}}),
            ),
            event("contentBlockStop", &json!({"contentBlockIndex": 1})),
            event("messageStop", &json!({"stopReason": "end_turn"})),
            event(
                "metadata",
                &json!({"usage": {"inputTokens": 9, "outputTokens": 4}}),
            ),
        ];
        let (events, failure) = run_stream(&chunks);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "Circle"}),
                json!({
                    "kind": "usage",
                    "input_tokens": 9,
                    "output_tokens": 4,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": null,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn bedrock_stop_reasons_map_to_the_python_terminal_table() {
        for (reason, expected) in [
            ("end_turn", json!({"kind": "completed"})),
            ("stop_sequence", json!({"kind": "completed"})),
            ("max_tokens", json!({"kind": "incomplete"})),
            (
                "model_context_window_exceeded",
                json!({"kind": "incomplete"}),
            ),
            (
                "guardrail_intervened",
                json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": "provider refused the request: content policy",
                    "refusal_reason": "content_policy",
                }),
            ),
            (
                "content_filtered",
                json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": "provider refused the request: content policy",
                    "refusal_reason": "content_policy",
                }),
            ),
            (
                "surprise",
                json!({
                    "kind": "failed",
                    "failure_class": "provider_internal",
                    "safe_message": "provider ended the stream unexpectedly",
                }),
            ),
        ] {
            let chunks = vec![
                event("messageStop", &json!({"stopReason": reason})),
                event(
                    "metadata",
                    &json!({"usage": {"inputTokens": 1, "outputTokens": 1}}),
                ),
            ];
            let (events, failure) = run_stream(&chunks);
            assert!(failure.is_none());
            assert_eq!(events.len(), 2, "reason {reason}");
            assert_eq!(events[1], expected, "reason {reason}");
        }
    }

    #[test]
    fn bedrock_exception_frames_map_to_python_failure_classes() {
        for (name, class) in [
            ("throttlingException", "throttled"),
            ("modelTimeoutException", "timeout"),
            ("internalServerException", "provider_internal"),
            ("modelStreamErrorException", "provider_internal"),
            ("serviceUnavailableException", "provider_internal"),
            ("validationException", "invalid_request"),
        ] {
            let chunks = vec![exception(name)];
            let (events, failure) = run_stream(&chunks);
            assert!(failure.is_none(), "exception {name}");
            assert_eq!(events.len(), 1, "exception {name}");
            assert_eq!(events[0]["kind"], "failed", "exception {name}");
            assert_eq!(events[0]["failure_class"], class, "exception {name}");
        }
    }

    #[test]
    fn bedrock_incomplete_tool_calls_fail_at_the_terminal() {
        let chunks = vec![
            event(
                "contentBlockStart",
                &json!({
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup"}},
                }),
            ),
            event("messageStop", &json!({"stopReason": "end_turn"})),
            event(
                "metadata",
                &json!({"usage": {"inputTokens": 1, "outputTokens": 1}}),
            ),
        ];
        let (events, failure) = run_stream(&chunks);
        assert!(failure.is_none());
        assert_eq!(events[0]["kind"], "tool_call_started");
        assert_eq!(events[1]["kind"], "usage");
        assert_eq!(events[2]["kind"], "failed");
        assert_eq!(events[2]["failure_class"], "malformed_response");
    }

    #[test]
    fn bedrock_tool_fragment_at_the_output_budget_is_incomplete_not_malformed() {
        let fragment = |stop_reason: &str| {
            vec![
                event(
                    "contentBlockStart",
                    &json!({
                        "contentBlockIndex": 0,
                        "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup"}},
                    }),
                ),
                event(
                    "contentBlockDelta",
                    &json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{\"city\": \"Par"}}}),
                ),
                event("contentBlockStop", &json!({"contentBlockIndex": 0})),
                event("messageStop", &json!({"stopReason": stop_reason})),
                event(
                    "metadata",
                    &json!({"usage": {"inputTokens": 1, "outputTokens": 1}}),
                ),
            ]
        };
        let (events, failure) = run_stream(&fragment("max_tokens"));
        assert!(failure.is_none());
        assert!(!events
            .iter()
            .any(|event| event["kind"] == "tool_call_completed"));
        assert_eq!(
            events.last().map(|event| event["kind"].clone()),
            Some(json!("incomplete"))
        );

        let (events, failure) = run_stream(&fragment("end_turn"));
        assert!(failure.is_none());
        let last = events.last().expect("terminal");
        assert_eq!(last["kind"], "failed");
        assert_eq!(last["failure_class"], "malformed_response");
    }

    #[test]
    fn bedrock_malformed_frames_fail_the_stream() {
        // Arguments before a tool start fail.
        let orphan = vec![event(
            "contentBlockDelta",
            &json!({"contentBlockIndex": 3, "delta": {"toolUse": {"input": "{}"}}}),
        )];
        let (_, failure) = run_stream(&orphan);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // An unsupported event name fails.
        let unknown = vec![event("mysteryEvent", &json!({}))];
        let (_, failure) = run_stream(&unknown);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // A non-text tool input delta fails.
        let bad_input = vec![
            event(
                "contentBlockStart",
                &json!({
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "c", "name": "n"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": 4}}}),
            ),
        ];
        let (_, failure) = run_stream(&bad_input);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // A stream that closes after messageStop but before metadata fails:
        // usage never arrived, so the terminal cannot be trusted.
        let unterminated = vec![event("messageStop", &json!({"stopReason": "end_turn"}))];
        let (events, failure) = run_stream(&unterminated);
        assert!(events.is_empty());
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );
    }
}
