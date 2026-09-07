//! Gemini `streamGenerateContent` frame mapping plus its golden-fixture tests.

use serde_json::Value;

use super::{malformed, parse_object, Normalizer};
use crate::errors::{Failure, FailureClass};
use crate::events::{gemini_usage, require_string, Event, ToolAccumulator};

impl Normalizer {
    /// Normalize one Gemini `streamGenerateContent` SSE frame: reasoning parts
    /// are skipped, whole function calls expand to start/arguments/completed,
    /// and the terminal candidate flushes the latest usage before its finish
    /// reason maps to the shared completion, incomplete, refusal, or
    /// provider-internal outcome. A prompt-level block (`promptFeedback.
    /// blockReason`, delivered with no candidates at all) is the same
    /// content-free refusal a candidate-level safety finish produces.
    pub(super) fn feed_gemini(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        if let Some(error) = payload.get("error").filter(|value| !value.is_null()) {
            // Google's error envelope ({"error":{"code":503,"status":"UNAVAILABLE"}})
            // arrives as a candidate-less frame; without this branch it reads
            // as a usage-only trailer and the stream ends malformed (or, after
            // prior output, as a synthesized completion of a failed answer).
            // It is the provider declaring failure, so it takes the shared
            // retry-then-failover classification the other dialects give it,
            // with its status and message riding as the bounded ledger detail.
            let (code, message) = match error.as_object() {
                Some(error) => (
                    error.get("status").and_then(Value::as_str),
                    error.get("message").and_then(Value::as_str),
                ),
                None => (None, None),
            };
            return Ok(vec![Event::Failed(self.provider_stream_failure(
                "gemini_generate_content",
                code,
                message,
            ))]);
        }
        if let Some(raw_usage) = payload.get("usageMetadata") {
            if !raw_usage.is_null() {
                self.usage = Some(gemini_usage(raw_usage).map_err(|message| malformed(&message))?);
            }
        }
        if gemini_prompt_blocked(&payload)? {
            // A candidate-less frame naming `promptFeedback.blockReason` is the
            // provider's verdict on the PROMPT, not a usage-only trailer: no
            // candidate follows and no finishReason will name the outcome.
            // It is the same content-free refusal a candidate-level SAFETY
            // finish is, so it terminates the stream as one (no retry, no
            // failover: every lane blocks the same prompt), after the usage
            // Google reported for the prompt it counted.
            let mut events = Vec::new();
            if let Some(usage) = self.usage.take() {
                events.push(Event::Usage(usage));
            }
            // The block reason names the category the same way a candidate
            // finishReason does.
            let reason = gemini_block_reason(&payload).unwrap_or_default();
            events.push(Event::Failed(Failure::refusal(
                crate::stream_errors::refusal_reason(Some(&reason), None),
            )));
            return Ok(events);
        }
        let candidates = match payload.get("candidates") {
            // A usage-only trailer frame legitimately carries no candidates at
            // all; its usageMetadata was already folded above, so continue the
            // stream instead of rejecting the whole answer as malformed. This
            // mirrors the already-handled empty-candidates case below.
            None | Some(Value::Null) => return Ok(Vec::new()),
            Some(value) => value
                .as_array()
                .ok_or_else(|| malformed("Gemini candidates must be an array"))?,
        };
        let mut events = Vec::new();
        if candidates.is_empty() {
            return Ok(events);
        }
        if candidates.len() != 1 {
            return Err(malformed("Gemini stream must contain one candidate"));
        }
        let candidate = candidates[0]
            .as_object()
            .ok_or_else(|| malformed("Gemini candidate must be an object"))?;
        match candidate.get("content") {
            None | Some(Value::Null) => {}
            Some(content) => {
                let parts = content
                    .as_object()
                    .ok_or_else(|| malformed("Gemini candidate content must be an object"))?
                    .get("parts")
                    .and_then(Value::as_array)
                    .ok_or_else(|| malformed("Gemini candidate parts must be an array"))?;
                for raw_part in parts {
                    let part = raw_part
                        .as_object()
                        .ok_or_else(|| malformed("Gemini candidate part must be an object"))?;
                    // Reasoning parts (thought text and thought signatures)
                    // are not gateway-visible output.
                    if part.get("thought") == Some(&Value::Bool(true)) {
                        continue;
                    }
                    if let Some(call) = part.get("functionCall") {
                        if !call.is_null() {
                            events.extend(self.gemini_tool_events(call)?);
                            continue;
                        }
                    }
                    match part.get("text") {
                        Some(Value::String(text)) => {
                            if !text.is_empty() {
                                events.push(Event::TextDelta(text.clone()));
                            }
                        }
                        // A part with neither visible text nor a function call
                        // (for example a bare thought signature) carries no
                        // gateway-visible output.
                        None | Some(Value::Null) => {}
                        Some(_) => return Err(malformed("Gemini text part must be text")),
                    }
                }
            }
        }
        let finish_reason = match candidate.get("finishReason") {
            None | Some(Value::Null) => return Ok(events),
            Some(Value::String(reason)) => reason.clone(),
            Some(_) => return Err(malformed("Gemini finishReason must be text")),
        };
        if let Some(usage) = self.usage.take() {
            events.push(Event::Usage(usage));
        }
        match finish_reason.as_str() {
            "STOP" | "FINISH_REASON_UNSPECIFIED" => events.push(Event::Completed),
            "MAX_TOKENS" => events.push(Event::Incomplete),
            // The python mapper's refusal signal table: safety, copyright,
            // and sensitive-information stops are content-free refusals. The
            // finish token names the category (RECITATION, SPII, SAFETY), so
            // the caller sees which policy declined without any provider prose.
            "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "RECITATION" | "SPII"
            | "IMAGE_SAFETY" => {
                events.push(Event::Failed(Failure::refusal(
                    crate::stream_errors::refusal_reason(Some(&finish_reason), None),
                )));
            }
            _ => {
                events.push(Event::Failed(Failure::new(
                    FailureClass::ProviderInternal,
                    "provider ended the stream unexpectedly",
                )));
            }
        }
        Ok(events)
    }

    /// Expand one complete Gemini function call into the canonical tool-call
    /// lifecycle, assigning the deterministic local call-ID fallback and the
    /// canonical compact JSON argument text the python mapper produces.
    fn gemini_tool_events(&mut self, value: &Value) -> Result<Vec<Event>, Failure> {
        let call = value
            .as_object()
            .ok_or_else(|| malformed("Gemini functionCall must be an object"))?;
        let name = require_string(call, "name", "Gemini functionCall name")
            .map_err(|message| malformed(&message))?;
        let index = self.gemini_tool_index;
        self.gemini_tool_index += 1;
        let call_id = match call.get("id") {
            Some(Value::String(id)) if !id.is_empty() => id.clone(),
            _ => format!("gemini-call-{index}"),
        };
        let raw_arguments = match call.get("args") {
            None => "{}".to_string(),
            Some(arguments @ Value::Object(_)) => serde_json::to_string(arguments)
                .map_err(|_| malformed("Gemini functionCall args must be an object"))?,
            Some(_) => return Err(malformed("Gemini functionCall args must be an object")),
        };
        self.reserve_tool_bytes(raw_arguments.len())?;
        let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
        tool.raw_arguments = raw_arguments.clone();
        let completed = tool.complete().map_err(|message| malformed(&message))?;
        Ok(vec![
            Event::ToolCallStarted {
                index,
                call_id,
                name,
                namespace: None,
                caller: None,
            },
            Event::ToolArgumentsDelta {
                index,
                delta: raw_arguments,
            },
            Event::ToolCallCompleted {
                index,
                call: completed,
            },
        ])
    }
}

/// Whether `promptFeedback` reports a prompt-level block: any `blockReason`
/// other than the enum's unspecified default means Google refused the prompt
/// before generating. An absent or null `promptFeedback`, or one that only
/// carries `safetyRatings`, is not a block.
fn gemini_prompt_blocked(payload: &serde_json::Map<String, Value>) -> Result<bool, Failure> {
    let feedback = match payload.get("promptFeedback") {
        None | Some(Value::Null) => return Ok(false),
        Some(value) => value
            .as_object()
            .ok_or_else(|| malformed("Gemini promptFeedback must be an object"))?,
    };
    match feedback.get("blockReason") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(reason)) => Ok(reason != "BLOCK_REASON_UNSPECIFIED"),
        Some(_) => Err(malformed("Gemini promptFeedback.blockReason must be text")),
    }
}

/// The `promptFeedback.blockReason` token, if the frame carries one. Only
/// called after `gemini_prompt_blocked` confirmed a present, meaningful
/// reason, so a malformed shape has already been rejected.
fn gemini_block_reason(payload: &serde_json::Map<String, Value>) -> Option<String> {
    payload
        .get("promptFeedback")
        .and_then(Value::as_object)
        .and_then(|feedback| feedback.get("blockReason"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod gemini_tests {
    use super::super::{drain_stream_fixture, Dialect};
    use super::*;
    use serde_json::json;

    fn run_stream(dialect: Dialect, chunks: &[&[u8]]) -> (Vec<Value>, Option<Failure>) {
        let owned: Vec<Vec<u8>> = chunks.iter().map(|chunk| chunk.to_vec()).collect();
        drain_stream_fixture(dialect, &owned)
    }

    fn sse(payload: &Value) -> Vec<u8> {
        format!("data: {payload}\n\n").into_bytes()
    }

    #[test]
    fn gemini_golden_stream_normalizes_text_tools_usage_and_completion() {
        // Golden fixture: raw provider bytes in, exact canonical events out.
        // `native_dialect_parity_test.py` holds the python-mapper comparison.
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]})),
            sse(&json!({"candidates": [{"content": {"parts": [
                {"thought": true, "text": "hidden reasoning"},
                {"text": "lo"},
            ]}}]})),
            sse(&json!({"candidates": [{"content": {"parts": [{
                "functionCall": {
                    "id": "call-1",
                    "name": "lookup",
                    "args": {"city": "Zürich", "count": 2},
                }
            }]}}]})),
            sse(&json!({
                "candidates": [{"finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 5,
                    "cachedContentTokenCount": 2,
                    "thoughtsTokenCount": 3,
                },
            })),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        let raw_arguments = "{\"city\":\"Zürich\",\"count\":2}";
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "Hel"}),
                json!({"kind": "text_delta", "text": "lo"}),
                json!({"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "lookup"}),
                json!({"kind": "tool_arguments_delta", "index": 0, "text": raw_arguments}),
                json!({
                    "kind": "tool_call_completed",
                    "index": 0,
                    "call_id": "call-1",
                    "name": "lookup",
                    "raw_arguments": raw_arguments,
                }),
                // thoughtsTokenCount is additive on the Gemini wire, so the
                // output total carries the folded 5 + 3 and reasoning names
                // the subset.
                json!({
                    "kind": "usage",
                    "input_tokens": 11,
                    "output_tokens": 8,
                    "cached_input_tokens": 2,
                    "reasoning_tokens": 3,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn gemini_thinking_stream_folds_thoughts_into_the_terminal_usage() {
        // Frame shapes from gemini-3.7-flash streamGenerateContent?alt=sse
        // (2026-09-03): every chunk carries usageMetadata, the final chunk
        // adds the thoughtSignature part and STOP. The terminal chunk's counts
        // win over the earlier partial ones, and Google's totalTokenCount
        // (11 + 7 + 654 = 672) shows thoughts are additive, so the normalized
        // output total is 661 with 654 as the reasoning subset.
        let chunks = [
            sse(&json!({
                "candidates": [{"content": {"parts": [{"text": "Sunlight scatters off air "}], "role": "model"}, "index": 0}],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 668,
                    "thoughtsTokenCount": 654,
                },
                "modelVersion": "gemini-3.7-flash",
                "responseId": "resp-1",
            })),
            sse(&json!({
                "candidates": [{
                    "content": {"parts": [{"text": "molecules.", "thoughtSignature": "CikB"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 672,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 11}],
                    "thoughtsTokenCount": 654,
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.7-flash",
                "responseId": "resp-1",
            })),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "Sunlight scatters off air "}),
                json!({"kind": "text_delta", "text": "molecules."}),
                json!({
                    "kind": "usage",
                    "input_tokens": 11,
                    "output_tokens": 661,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 654,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn gemini_missing_call_id_uses_the_deterministic_local_fallback() {
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [
                {"functionCall": {"name": "first", "args": {}}},
                {"functionCall": {"name": "second"}},
            ]}}]})),
            sse(&json!({
                "candidates": [{"finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            })),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events[0],
            json!({"kind": "tool_call_started", "index": 0, "call_id": "gemini-call-0", "name": "first"})
        );
        assert_eq!(events[1]["text"], "{}");
        assert_eq!(
            events[3],
            json!({"kind": "tool_call_started", "index": 1, "call_id": "gemini-call-1", "name": "second"})
        );
        // Absent usage counts are zero (require_integer parity), and an
        // omitted thoughtsTokenCount stays unknown.
        assert_eq!(
            events[6],
            json!({
                "kind": "usage",
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_input_tokens": 0,
                "reasoning_tokens": null,
            })
        );
        assert_eq!(events[7], json!({"kind": "completed"}));
    }

    #[test]
    fn gemini_safety_finish_maps_to_a_categorized_refusal() {
        // Each finish token names the bounded category the caller sees, with
        // the fixed phrase in the message and never any provider prose.
        for (reason, category, phrase) in [
            ("SAFETY", "content_policy", "content policy"),
            ("PROHIBITED_CONTENT", "content_policy", "content policy"),
            ("BLOCKLIST", "content_policy", "content policy"),
            (
                "RECITATION",
                "recitation",
                "recitation of copyrighted material",
            ),
            ("SPII", "data_inspection", "data inspection"),
        ] {
            let chunks = [sse(&json!({"candidates": [{"finishReason": reason}]}))];
            let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
            let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
            assert!(failure.is_none());
            assert_eq!(
                events,
                vec![json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": format!("provider refused the request: {phrase}"),
                    "refusal_reason": category,
                })],
                "{reason}"
            );
        }
    }

    #[test]
    fn gemini_prompt_block_is_a_content_free_refusal_with_its_usage() {
        // The production shape (2026-09-04, gemini-3.7-flash and 3.8-flash
        // via streamGenerateContent?alt=sse): one frame, no candidates, the
        // block named on promptFeedback, and usageMetadata counting the
        // prompt Google processed. It must terminate the stream as a refusal
        // (400, no retry, no failover), never as a malformed stream end.
        for (reason, category, message) in [
            (
                "SAFETY",
                "content_policy",
                "provider refused the request: content policy",
            ),
            (
                "PROHIBITED_CONTENT",
                "content_policy",
                "provider refused the request: content policy",
            ),
            (
                "BLOCKLIST",
                "content_policy",
                "provider refused the request: content policy",
            ),
            // A block reason the vocabulary does not know is an unnamed refusal.
            ("OTHER", "unspecified", "provider refused the request"),
            (
                "IMAGE_SAFETY",
                "content_policy",
                "provider refused the request: content policy",
            ),
        ] {
            let chunks = [sse(&json!({
                "promptFeedback": {
                    "blockReason": reason,
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}
                    ],
                },
                "usageMetadata": {"promptTokenCount": 42, "totalTokenCount": 42},
            }))];
            let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
            let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
            assert!(
                failure.is_none(),
                "{reason}: a prompt block is a terminal, not a failure"
            );
            assert_eq!(
                events,
                vec![
                    json!({
                        "kind": "usage",
                        "input_tokens": 42,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "reasoning_tokens": null,
                    }),
                    json!({
                        "kind": "failed",
                        "failure_class": "refusal",
                        "safe_message": message,
                        "refusal_reason": category,
                    }),
                ],
                "{reason}"
            );
        }
    }

    #[test]
    fn gemini_prompt_block_without_usage_is_still_a_refusal() {
        let chunks = [sse(&json!({"promptFeedback": {"blockReason": "SAFETY"}}))];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![json!({
                "kind": "failed",
                "failure_class": "refusal",
                "safe_message": "provider refused the request: content policy",
                "refusal_reason": "content_policy",
            })]
        );
        // A refusal is the model's verdict on the prompt: neither redialed on
        // the same deployment nor failed over to a sibling lane.
        let refusal = Failure::refusal(crate::errors::RefusalReason::ContentPolicy);
        assert!(!refusal.retryable_same_deployment);
        assert!(!refusal.failover_eligible);
    }

    #[test]
    fn gemini_unspecified_block_reason_and_ratings_only_feedback_are_not_blocks() {
        // Ordinary answers carry promptFeedback.safetyRatings (and some carry
        // the enum default) alongside real candidates; those must keep
        // flowing as content.
        for feedback in [
            json!({"safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}]}),
            json!({"blockReason": "BLOCK_REASON_UNSPECIFIED"}),
        ] {
            let chunks = [
                sse(&json!({
                    "promptFeedback": feedback,
                    "candidates": [{"content": {"parts": [{"text": "fine"}]}}],
                })),
                sse(&json!({"candidates": [{"finishReason": "STOP"}]})),
            ];
            let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
            let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
            assert!(failure.is_none());
            assert_eq!(
                events,
                vec![
                    json!({"kind": "text_delta", "text": "fine"}),
                    json!({"kind": "completed"}),
                ]
            );
        }
    }

    #[test]
    fn gemini_error_envelope_is_classified_by_what_google_said() {
        // Google's own error envelope on the stream (verified shape for a
        // 503 UNAVAILABLE "overloaded"): a provider-declared failure. It is
        // classified by its content like every other dialect's: an overloaded
        // model is a THROTTLE (fail over, advertise Retry-After; redialing the
        // same saturated rung buys nothing), never a malformed stream end, and
        // never a completion when output preceded it.
        let envelope = json!({
            "error": {
                "code": 503,
                "message": "The model is overloaded. Please try again later.",
                "status": "UNAVAILABLE",
            }
        });
        let failed = json!({
            "kind": "failed",
            "failure_class": "throttled",
            "safe_message": "provider throttled the request; retry after the delay in the Retry-After header",
        });
        let alone = [sse(&envelope)];
        let refs: Vec<&[u8]> = alone.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(events, vec![failed.clone()]);
        let after_output = [
            sse(&json!({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]})),
            sse(&envelope),
        ];
        let refs: Vec<&[u8]> = after_output.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![json!({"kind": "text_delta", "text": "partial"}), failed]
        );
        // A genuine provider fault keeps the retry-then-failover shape.
        let internal = json!({
            "error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}
        });
        let refs = [sse(&internal)];
        let refs: Vec<&[u8]> = refs.iter().map(Vec::as_slice).collect();
        let (events, _failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert_eq!(
            events,
            vec![json!({
                "kind": "failed",
                "failure_class": "provider_internal",
                "safe_message": "provider stream failed",
            })]
        );
        let classified = crate::stream_errors::stream_failure(
            crate::stream_errors::StreamErrorKind::ProviderInternal,
            None,
        );
        assert_eq!(classified.failure_class, FailureClass::ProviderInternal);
        assert!(classified.retryable_same_deployment);
        assert!(classified.failover_eligible);
    }

    #[test]
    fn gemini_max_tokens_is_incomplete_and_unknown_reasons_fail() {
        let incomplete = [sse(
            &json!({"candidates": [{"finishReason": "MAX_TOKENS"}]}),
        )];
        let refs: Vec<&[u8]> = incomplete.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(events, vec![json!({"kind": "incomplete"})]);

        let unknown = [sse(
            &json!({"candidates": [{"finishReason": "MALFUNCTION"}]}),
        )];
        let refs: Vec<&[u8]> = unknown.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![json!({
                "kind": "failed",
                "failure_class": "provider_internal",
                "safe_message": "provider ended the stream unexpectedly",
            })]
        );
    }

    #[test]
    fn gemini_malformed_frame_before_content_is_a_retryable_abnormal_end() {
        // A malformed frame arriving before any content is a Gemini abnormal
        // end with nothing to salvage: there is no partial answer, so it is
        // reclassified from a hard malformed reject to a retryable transport
        // failure (retry the lane, then fail over) rather than being accepted.
        // The frames are still rejected — none is silently taken as content.
        let cases: [Value; 3] = [
            // A non-text text part (python: parts.text must be a string).
            json!({"candidates": [{"content": {"parts": [{"text": 5}]}}]}),
            // Null function-call args (python: args must decode to a dict).
            json!({"candidates": [{"content": {"parts": [
                {"functionCall": {"name": "x", "args": null}}
            ]}}]}),
            // Two candidates in one frame.
            json!({"candidates": [{}, {}]}),
        ];
        for payload in &cases {
            let chunk = sse(payload);
            let refs: Vec<&[u8]> = [chunk.as_slice()].to_vec();
            let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
            assert!(events.is_empty(), "a rejected frame emits no content");
            let failure = failure.expect("a malformed pre-content frame must not succeed");
            assert_eq!(failure.failure_class, FailureClass::Transport);
            assert!(failure.retryable_same_deployment);
            assert!(failure.failover_eligible);
        }
    }

    #[test]
    fn gemini_usage_only_close_without_content_still_fails_malformed() {
        // A stream that closes having emitted NO content at all (here only a
        // usage-only trailer, then a clean EOF) still fails malformed: there is
        // no real answer to complete, so the clean-end tolerance does not apply
        // and the abnormal-end recovery never engages (nothing terminated it).
        let empty = [sse(&json!({
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 0},
        }))];
        let refs: Vec<&[u8]> = empty.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(events.is_empty());
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );
    }

    #[test]
    fn gemini_malformed_frame_after_content_recovers_as_incomplete() {
        // Content emitted, then a structurally malformed frame: the real answer
        // is preserved and the turn ends `incomplete` (an early-termination
        // finish reason), never discarded as malformed. Last-seen usage is
        // folded so delivered tokens still bill at settle.
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]})),
            sse(&json!({"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3}})),
            // A non-text text part after content: malformed mid-stream frame.
            sse(&json!({"candidates": [{"content": {"parts": [{"text": 5}]}}]})),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(
            failure.is_none(),
            "a partial answer is recovered, not failed"
        );
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "partial"}),
                json!({
                    "kind": "usage",
                    "input_tokens": 9,
                    "output_tokens": 3,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": null,
                }),
                json!({"kind": "incomplete"}),
            ]
        );
    }

    #[test]
    fn gemini_usage_only_trailer_frame_folds_usage_without_failing() {
        // A trailer frame that carries only usageMetadata (no candidates array)
        // must not fail: its usage is folded and the stream continues to its
        // real terminal candidate, which flushes the last-seen usage.
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})),
            sse(&json!({"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}})),
            sse(&json!({"candidates": [{"finishReason": "STOP"}]})),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "hi"}),
                json!({
                    "kind": "usage",
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": null,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn gemini_clean_end_after_content_completes_with_folded_usage() {
        // Gemini sometimes ends the stream right after its last content frame
        // with no finishReason frame; the relay/drain closes it as a normal
        // completion (folding the last-seen usage) rather than throwing the
        // real answer away as malformed.
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [{"text": "done"}]}}]})),
            sse(&json!({"usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1}})),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "done"}),
                json!({
                    "kind": "usage",
                    "input_tokens": 4,
                    "output_tokens": 1,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": null,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn gemini_frames_after_the_terminal_candidate_are_ignored() {
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        let terminal = crate::sse::SseEvent {
            event: None,
            data: json!({"candidates": [{"finishReason": "STOP"}]}).to_string(),
        };
        let trailing = crate::sse::SseEvent {
            event: None,
            data: json!({"candidates": [{"content": {"parts": [{"text": "late"}]}}]}).to_string(),
        };
        let events = normalizer.feed(&terminal).expect("terminal frame");
        assert!(events.iter().any(Event::is_terminal));
        assert!(normalizer.saw_terminal());
        assert!(normalizer.feed(&trailing).expect("ignored").is_empty());
    }
}
