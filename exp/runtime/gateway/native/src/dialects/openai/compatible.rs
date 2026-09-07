//! The OpenAI-compatible Chat Completions frame mapping (split from
//! `openai.rs` for the module line budget): incremental deltas, tool-call
//! fragments, exposure-gated reasoning content, and the `[DONE]` terminal.

use serde_json::Value;

use super::super::{
    finish_open_tools, finish_open_tools_truncated, malformed, parse_object, Normalizer,
};
use crate::errors::Failure;
use crate::events::{openai_compatible_usage, require_string, require_u64, Event, ToolAccumulator};

impl Normalizer {
    pub(in crate::dialects) fn feed_openai_compatible(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        if frame.data == "[DONE]" {
            let finish = self.finish_reason.as_deref();
            // A tool call cut off by the output budget (finish_reason=length,
            // arguments still an open JSON fragment) is the provider's honest
            // truncation, not a malformed stream: it surfaces as Incomplete
            // with the truncated call dropped, exactly what the caller must
            // act on (raise max_tokens), never as a 502. Live shape: Tencent
            // TokenHub glm-5.3 at max_tokens=32 streamed `{"` + `city` then
            // finished with length (staging, 2026-09-03). Any other finish
            // keeps the strict contract: unparsable arguments are malformed.
            let mut events = if finish == Some("length") {
                finish_open_tools_truncated(&mut self.tools)?
            } else {
                finish_open_tools(&mut self.tools)?
            };
            if let Some(usage) = self.usage.take() {
                events.push(Event::Usage(usage));
            }
            if self.refusal_seen || matches!(finish, Some("content_filter" | "safety")) {
                // A `content_filter`/`safety` finish reason names the category;
                // a bare visible-refusal delta names none (Unspecified).
                let reason = match finish {
                    Some(code @ ("content_filter" | "safety")) => {
                        crate::stream_errors::refusal_reason(Some(code), None)
                    }
                    _ => crate::errors::RefusalReason::Unspecified,
                };
                events.push(Event::Failed(Failure::refusal(reason)));
            } else if finish == Some("length") {
                events.push(Event::Incomplete);
            } else {
                events.push(Event::Completed);
            }
            return Ok(events);
        }
        let payload = parse_object(&frame.data)?;
        if let Some(error) = payload.get("error").filter(|value| !value.is_null()) {
            // An OpenAI-compatible relay declaring failure inside the stream
            // names the mechanism only here; the bounded detail rides the
            // failure into the ledger.
            let (code, message) = match error.as_object() {
                Some(error) => (
                    error.get("code").and_then(|value| {
                        value
                            .as_str()
                            .map(str::to_string)
                            .or_else(|| value.as_i64().map(|numeric| numeric.to_string()))
                    }),
                    error.get("message").and_then(Value::as_str).map(|message| {
                        // An aggregator's generic sentence yields to the
                        // upstream provider's own (OpenRouter metadata.raw).
                        crate::param_attribution::upstream_relayed_message(
                            &Value::Object(error.clone()),
                            message,
                        )
                        .unwrap_or_else(|| message.to_string())
                    }),
                ),
                None => (None, None),
            };
            return Ok(vec![Event::Failed(self.provider_stream_failure(
                "openai_compatible",
                code.as_deref(),
                message.as_deref(),
            ))]);
        }
        let mut events = Vec::new();
        if let Some(raw_usage) = payload.get("usage") {
            if !raw_usage.is_null() {
                self.usage = Some(
                    openai_compatible_usage(raw_usage).map_err(|message| malformed(&message))?,
                );
            }
        }
        let choices = payload
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("OpenAI-compatible choices must be an array"))?;
        if choices.is_empty() {
            return Ok(events);
        }
        if choices.len() != 1 {
            return Err(malformed(
                "OpenAI-compatible stream must contain one choice",
            ));
        }
        let choice = choices[0]
            .as_object()
            .ok_or_else(|| malformed("OpenAI-compatible choice must be an object"))?;
        let delta = choice
            .get("delta")
            .and_then(Value::as_object)
            .ok_or_else(|| malformed("OpenAI-compatible delta must be an object"))?;
        if let Some(Value::String(content)) = delta.get("content") {
            if !content.is_empty() {
                events.push(Event::TextDelta(content.clone()));
            }
        }
        if let Some(Value::String(refusal)) = delta.get("refusal") {
            self.refusal_seen = true;
            events.push(Event::RefusalDelta(refusal.clone()));
        }
        if let Some(route_sha256) = self.reasoning_content_route_sha256.clone() {
            if let Some(value) = delta.get("reasoning_content") {
                let reasoning = match value {
                    Value::Null => None,
                    Value::String(text) => Some(text),
                    _ => return Err(malformed("Fireworks reasoning_content delta must be text")),
                };
                if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
                    self.reserve_summary_bytes(reasoning.len())?;
                    events.push(Event::ReasoningContentDelta {
                        route_sha256,
                        delta: reasoning.clone(),
                    });
                }
            }
        }
        if let Some(raw_tools) = delta.get("tool_calls") {
            if !raw_tools.is_null() {
                let items = raw_tools
                    .as_array()
                    .ok_or_else(|| malformed("OpenAI-compatible tool_calls must be an array"))?;
                for value in items {
                    let item = value.as_object().ok_or_else(|| {
                        malformed("OpenAI-compatible tool call must be an object")
                    })?;
                    let index = require_u64(item, "index", "OpenAI-compatible tool index")
                        .map_err(|message| malformed(&message))?
                        as u32;
                    let function =
                        item.get("function")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                malformed("OpenAI-compatible tool function must be an object")
                            })?;
                    if let Some(tool) = self.tools.get(&index) {
                        // An identity is only restated when it is non-empty:
                        // DashScope (Qwen) argument deltas carry `"id": ""`
                        // (documented shape, live 2026-09-03), and an empty
                        // placeholder names nothing, so only a different
                        // NON-EMPTY id or name is a stream that changed identity.
                        if let Some(Value::String(repeated_id)) = item.get("id") {
                            if !repeated_id.is_empty() && repeated_id != &tool.call_id {
                                return Err(malformed(
                                    "OpenAI-compatible stream changed a tool-call ID",
                                ));
                            }
                        }
                        if let Some(Value::String(repeated_name)) = function.get("name") {
                            if !repeated_name.is_empty() && repeated_name != &tool.name {
                                return Err(malformed(
                                    "OpenAI-compatible stream changed a tool-call name",
                                ));
                            }
                        }
                    } else {
                        let call_id = require_string(item, "id", "OpenAI-compatible tool ID")
                            .map_err(|message| malformed(&message))?;
                        let name = require_string(function, "name", "OpenAI-compatible tool name")
                            .map_err(|message| malformed(&message))?;
                        self.reserve_tool_entry(index)?;
                        self.tools
                            .insert(index, ToolAccumulator::new(call_id.clone(), name.clone()));
                        events.push(Event::ToolCallStarted {
                            index,
                            call_id,
                            name,
                            namespace: None,
                            caller: None,
                        });
                    }
                    if let Some(fragment) = function.get("arguments") {
                        if !fragment.is_null() {
                            let raw_fragment = match fragment {
                                Value::String(text) => text.clone(),
                                _ => {
                                    return Err(malformed(
                                        "OpenAI-compatible argument delta must be text",
                                    ))
                                }
                            };
                            self.reserve_tool_bytes(raw_fragment.len())?;
                            let tool = self.tools.get_mut(&index).expect("tool just ensured");
                            tool.raw_arguments.push_str(&raw_fragment);
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: raw_fragment,
                            });
                        }
                    }
                }
            }
        }
        if let Some(Value::String(finish)) = choice.get("finish_reason") {
            self.finish_reason = Some(finish.clone());
            if matches!(finish.as_str(), "content_filter" | "safety") && !self.refusal_seen {
                self.refusal_seen = true;
                events.push(Event::RefusalDelta(String::new()));
            }
        }
        Ok(events)
    }
}
