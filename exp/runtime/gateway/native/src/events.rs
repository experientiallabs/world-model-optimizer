//! Provider-neutral stream events, the Rust mirror of `GatewayEvent`.
//!
//! # Usage contract
//!
//! Every usage mapper in this module emits OpenAI subset semantics:
//! `reasoning_tokens` (when known) counts a SUBSET of `output_tokens`, and
//! `cached_input_tokens` a subset of `input_tokens`. Settlement prices the
//! reasoning subset at the reasoning rate and the remainder of `output_tokens`
//! at the output rate, so a wire that reports reasoning OUTSIDE its output
//! total would bill every reasoning token at zero unless the mapper folds it
//! back in. Per wire:
//!
//! - OpenAI-shaped wires (Responses via `openai_usage`, Chat Completions via
//!   `openai_compatible_usage`): OpenAI, OpenRouter, DeepSeek, and Fireworks
//!   report reasoning inside the output total; xAI (native and relayed by
//!   Azure Foundry) reports it outside on both wires. The provider's own
//!   `total_tokens` decides: `input + output` is the subset shape and is
//!   forwarded as reported, `input + output + reasoning` is the additive shape
//!   and folds (`fold_openai_shaped_reasoning`). Without a decisive total, a
//!   reasoning count above the output total is impossible under subset
//!   semantics and folds.
//! - Gemini (`gemini_usage`): `thoughtsTokenCount` is additive by Google's
//!   definition (`totalTokenCount` = prompt + candidates + thoughts), so it is
//!   folded into `output_tokens` unconditionally.
//! - Anthropic Messages and Bedrock Converse: thinking is billed inside the
//!   provider's `output_tokens` and no separate count is published, so
//!   `reasoning_tokens` stays `None` and the total is forwarded as reported.
//!
//! A fold whose total leaves the persistable ledger range is a provider
//! contract violation and fails the stream; totals are never clamped.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::Failure;

/// Normalized token usage mirroring `GatewayUsage` semantics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    /// Cache-write tokens inside the input total, present only when the
    /// provider reported a nonzero count (Anthropic-only today). The ledger
    /// keeps billing the folded input total; this leg exists so callers see
    /// their prompt being cached (Claude Code displays it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    pub fn has_token_counts(&self) -> bool {
        self.input_tokens.is_some() && self.output_tokens.is_some()
    }
}

/// One completed tool call with provider-order raw argument text.
#[derive(Debug, Clone)]
pub struct CompletedToolCall {
    pub call_id: String,
    pub name: String,
    /// Nested tool tree (Responses `namespace`) that declared this call,
    /// preserved verbatim through retention and the client stream because
    /// the provider rejects a namespaced call replayed without it.
    pub namespace: Option<String>,
    /// Opaque SDK 3.0 `caller` attribution (for example
    /// `{"type": "program", "id": ...}`) naming the program that invoked
    /// this call; carried verbatim like `namespace` so the item
    /// round-trips exactly as the provider emitted it.
    pub caller: Option<Value>,
    pub provider_item_id: Option<String>,
    pub provider_status: Option<ProviderOutputItemStatus>,
    /// Raw provider-order argument text: a validated JSON object for
    /// function calls, freeform text for custom (freeform) tool calls.
    pub raw_arguments: String,
    /// Whether this is a freeform custom tool call (Responses-only).
    pub custom: bool,
}

/// Provider-owned Responses output-item kind whose identity must remain exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOutputItemKind {
    Reasoning,
    FunctionCall,
    CustomToolCall,
    Message,
}

/// Provider-owned Responses item lifecycle status preserved byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOutputItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

impl ProviderOutputItemStatus {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            "incomplete" => Some(Self::Incomplete),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
        }
    }
}

/// Optional phase attached to an OpenAI Responses assistant message item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderAssistantMessagePhase {
    Commentary,
    FinalAnswer,
}

impl ProviderAssistantMessagePhase {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "commentary" => Some(Self::Commentary),
            "final_answer" => Some(Self::FinalAnswer),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Commentary => "commentary",
            Self::FinalAnswer => "final_answer",
        }
    }
}

/// One ordered provider-neutral stream event.
#[derive(Debug, Clone)]
pub enum Event {
    TextDelta(String),
    RefusalDelta(String),
    /// One text delta for a specific provider-owned assistant message item.
    ProviderTextDelta {
        output_index: u32,
        item_id: String,
        delta: String,
    },
    /// One refusal delta for a specific provider-owned assistant message item.
    ProviderRefusalDelta {
        output_index: u32,
        item_id: String,
        delta: String,
    },
    /// Reserve a public Responses slot at the provider's item-start boundary.
    ProviderOutputItemStarted {
        output_index: u32,
        item_id: Option<String>,
        kind: ProviderOutputItemKind,
        status: Option<ProviderOutputItemStatus>,
        phase: Option<ProviderAssistantMessagePhase>,
    },
    /// Close one provider-owned output item with its exact lifecycle metadata.
    ProviderOutputItemCompleted {
        output_index: u32,
        item_id: Option<String>,
        kind: ProviderOutputItemKind,
        status: Option<ProviderOutputItemStatus>,
        phase: Option<ProviderAssistantMessagePhase>,
    },
    ReasoningSummaryDelta {
        output_index: u32,
        summary_index: u32,
        item_id: String,
        delta: String,
    },
    /// Verbatim Anthropic extended-thinking text for one provider block.
    ThinkingDelta {
        index: u32,
        delta: String,
    },
    /// Opaque cryptographic signature closing one Anthropic thinking block;
    /// it must round-trip byte-exact or the provider rejects the replay.
    ThinkingSignature {
        index: u32,
        signature: String,
    },
    /// One complete opaque Anthropic redacted-thinking block.
    RedactedThinking {
        index: u32,
        data: String,
    },
    /// One opaque OpenAI Responses encrypted reasoning payload, keyed by its
    /// provider output-item index.
    EncryptedReasoning {
        output_index: u32,
        item_id: String,
        encrypted_content: String,
    },
    /// Opaque Fireworks Chat reasoning, bound to the exact issuing route.
    ReasoningContentDelta {
        route_sha256: String,
        delta: String,
    },
    ToolCallStarted {
        index: u32,
        call_id: String,
        name: String,
        /// Nested tool tree (Responses `namespace`) that declared this call;
        /// present only on native Responses streams and preserved verbatim
        /// because the provider rejects a namespaced call replayed without it.
        namespace: Option<String>,
        /// Opaque SDK 3.0 `caller` attribution carried verbatim like
        /// `namespace`.
        caller: Option<Value>,
    },
    ToolArgumentsDelta {
        index: u32,
        delta: String,
    },
    ToolCallCompleted {
        index: u32,
        call: CompletedToolCall,
    },
    /// One provider-executed Anthropic server tool invocation opening
    /// (`server_tool_use`); the provider runs the tool itself, so these
    /// never become client tool calls or affect the tool-use stop reason.
    ServerToolUseStarted {
        index: u32,
        call_id: String,
        name: String,
    },
    /// Raw provider-order input fragment for one open server tool use.
    ServerToolArgumentsDelta {
        index: u32,
        delta: String,
    },
    /// One completed server tool invocation with its validated input text.
    ServerToolUseCompleted {
        index: u32,
        call: CompletedToolCall,
    },
    /// One whole verbatim Anthropic server-tool result content block
    /// (`web_search_tool_result`), carried as compact JSON text: the result
    /// arrives complete in its start frame and must reach the caller intact.
    ServerToolResult {
        index: u32,
        block: String,
    },
    /// One OpenAI Responses hosted-tool output item opening
    /// (`web_search_call`, `mcp_call`, `code_interpreter_call`, ...). The
    /// provider executes the tool itself and owns the item's shape, so the
    /// whole item is carried verbatim as compact JSON: the caller (and its
    /// next-turn echo) must see exactly what the provider produced.
    HostedToolItemStarted {
        output_index: u32,
        item_id: String,
        item_type: String,
        item: String,
    },
    /// One verbatim per-type lifecycle or delta frame for an open hosted
    /// tool item (`response.web_search_call.searching`,
    /// `response.mcp_call_arguments.delta`, ...), carried as compact JSON.
    /// The Responses encoder re-stamps only the public output index and
    /// sequence number; every other payload field passes through untouched.
    HostedToolItemProgress {
        output_index: u32,
        item_id: String,
        event_type: String,
        payload: String,
    },
    /// One completed hosted tool item with its final verbatim JSON, from the
    /// provider's `response.output_item.done` (or the last-seen item when the
    /// terminal response arrived first).
    HostedToolItemCompleted {
        output_index: u32,
        item_id: String,
        item_type: String,
        item: String,
    },
    /// One whole verbatim OpenAI Responses output-text annotation
    /// (`response.output_text.annotation.added`: URL citations from hosted
    /// web search), carried as compact JSON and attached to the open
    /// provider-owned assistant message item.
    ProviderTextAnnotation {
        output_index: u32,
        item_id: String,
        annotation: String,
    },
    /// Provider text content-block boundary on the Anthropic wire. Emitted
    /// before that block's first text delta so the Messages encoder can
    /// mirror the provider's block structure (citations attach per block);
    /// encoders without a block concept ignore it.
    TextBlockStarted {
        index: u32,
    },
    /// One whole verbatim citation object attached to the open Anthropic
    /// text block (`citations_delta`), carried as compact JSON text.
    CitationDelta {
        index: u32,
        citation: String,
    },
    Usage(Usage),
    Completed,
    Incomplete,
    /// The gateway cut the stream at one of the caller's stop sequences on a
    /// wire that has no stop field (OpenAI Responses). Settles as completed;
    /// the Messages encoder reports `stop_sequence` with this exact value.
    StoppedAtSequence(String),
    /// Anthropic `pause_turn` terminal: the provider paused a long-running
    /// server-tool turn and expects the caller to resend the conversation to
    /// continue it. Settlement treats it like a completed turn; the Messages
    /// encoder must preserve the stop reason or the caller never resumes.
    PausedTurn,
    Failed(Failure),
}

impl Event {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Completed
                | Event::Incomplete
                | Event::StoppedAtSequence(_)
                | Event::PausedTurn
                | Event::Failed(_)
        )
    }

    /// Whether this event carries the first visible model output, used to
    /// stamp time-to-first-token. A content, refusal, reasoning, or tool-argument
    /// delta counts only when it carries at least one character: an empty delta
    /// (a role-establishing or empty refusal frame) is not a visible token and
    /// must not stamp TTFT early. A tool-call start is itself the first token of
    /// a tool-only turn, so it counts even before any arguments stream. Purely
    /// structural frames are excluded so TTFT is not stamped early: the Responses
    /// `ProviderOutputItemStarted` reserves a slot at the item-start boundary
    /// *before* the first delta arrives, and the opaque reasoning-carrier frames
    /// (`ThinkingSignature`, `RedactedThinking`, `EncryptedReasoning`) never lead
    /// a turn on their own. Usage, item-close, and lifecycle/terminal frames are
    /// not output tokens either.
    pub fn is_output_token(&self) -> bool {
        match self {
            Event::TextDelta(text) | Event::RefusalDelta(text) => !text.is_empty(),
            Event::ProviderTextDelta { delta, .. }
            | Event::ProviderRefusalDelta { delta, .. }
            | Event::ReasoningSummaryDelta { delta, .. }
            | Event::ThinkingDelta { delta, .. }
            | Event::ReasoningContentDelta { delta, .. }
            | Event::ToolArgumentsDelta { delta, .. }
            | Event::ServerToolArgumentsDelta { delta, .. } => !delta.is_empty(),
            Event::ToolCallStarted { .. }
            | Event::ServerToolUseStarted { .. }
            // A hosted tool item start is the provider beginning visible
            // work, the same first-token signal as a tool-call start.
            | Event::HostedToolItemStarted { .. } => true,
            _ => false,
        }
    }
}

/// Render one event as the content-bearing JSON object used by dialect parity
/// fixtures: the same field vocabulary the fixture-event parser accepts, plus
/// the failure class and safe message for terminal failures.
pub fn simplified_event(event: &Event) -> Value {
    match event {
        Event::TextDelta(text) => serde_json::json!({"kind": "text_delta", "text": text}),
        Event::RefusalDelta(text) => serde_json::json!({"kind": "refusal_delta", "text": text}),
        Event::ProviderTextDelta {
            output_index,
            item_id,
            delta,
        } => serde_json::json!({
            "kind": "provider_text_delta",
            "output_index": output_index,
            "item_id": item_id,
            "text": delta,
        }),
        Event::ProviderRefusalDelta {
            output_index,
            item_id,
            delta,
        } => serde_json::json!({
            "kind": "provider_refusal_delta",
            "output_index": output_index,
            "item_id": item_id,
            "text": delta,
        }),
        Event::ProviderOutputItemStarted {
            output_index,
            item_id,
            kind,
            status,
            phase,
        } => {
            let mut payload = serde_json::json!({
                "kind": "provider_output_item_started",
                "output_index": output_index,
                "item_type": match kind {
                    ProviderOutputItemKind::Reasoning => "reasoning",
                    ProviderOutputItemKind::FunctionCall => "function_call",
                    ProviderOutputItemKind::CustomToolCall => "custom_tool_call",
                    ProviderOutputItemKind::Message => "message",
                },
            });
            add_provider_item_metadata(&mut payload, item_id, *status, *phase);
            payload
        }
        Event::ProviderOutputItemCompleted {
            output_index,
            item_id,
            kind,
            status,
            phase,
        } => {
            let mut payload = serde_json::json!({
                "kind": "provider_output_item_completed",
                "output_index": output_index,
                "item_type": match kind {
                ProviderOutputItemKind::Reasoning => "reasoning",
                ProviderOutputItemKind::FunctionCall => "function_call",
                ProviderOutputItemKind::CustomToolCall => "custom_tool_call",
                ProviderOutputItemKind::Message => "message",
                },
            });
            add_provider_item_metadata(&mut payload, item_id, *status, *phase);
            payload
        }
        Event::ReasoningSummaryDelta {
            output_index,
            summary_index,
            item_id,
            delta,
        } => serde_json::json!({
            "kind": "reasoning_summary_delta",
            "output_index": output_index,
            "summary_index": summary_index,
            "item_id": item_id,
            "text": delta,
        }),
        Event::ThinkingDelta { index, delta } => serde_json::json!({
            "kind": "thinking_delta",
            "index": index,
            "text": delta,
        }),
        Event::ThinkingSignature { index, signature } => serde_json::json!({
            "kind": "thinking_signature",
            "index": index,
            "signature": signature,
        }),
        Event::RedactedThinking { index, data } => serde_json::json!({
            "kind": "redacted_thinking",
            "index": index,
            "data": data,
        }),
        Event::EncryptedReasoning {
            output_index,
            item_id,
            encrypted_content,
        } => serde_json::json!({
            "kind": "encrypted_reasoning",
            "output_index": output_index,
            "item_id": item_id,
            "encrypted_content": encrypted_content,
        }),
        Event::ReasoningContentDelta {
            route_sha256,
            delta,
        } => serde_json::json!({
            "kind": "reasoning_content_delta",
            "route_sha256": route_sha256,
            "text": delta,
        }),
        Event::ToolCallStarted {
            index,
            call_id,
            name,
            namespace,
            caller,
        } => {
            let mut payload = serde_json::json!({
                "kind": "tool_call_started",
                "index": index,
                "call_id": call_id,
                "name": name,
            });
            if let Some(namespace) = namespace {
                payload["namespace"] = Value::String(namespace.clone());
            }
            if let Some(caller) = caller {
                payload["caller"] = caller.clone();
            }
            payload
        }
        Event::ToolArgumentsDelta { index, delta } => serde_json::json!({
            "kind": "tool_arguments_delta",
            "index": index,
            "text": delta,
        }),
        Event::ToolCallCompleted { index, call } => {
            let mut payload = serde_json::json!({
                "kind": "tool_call_completed",
                "index": index,
                "call_id": call.call_id,
                "name": call.name,
                "raw_arguments": call.raw_arguments,
            });
            if let Some(namespace) = &call.namespace {
                payload["namespace"] = Value::String(namespace.clone());
            }
            if let Some(caller) = &call.caller {
                payload["caller"] = caller.clone();
            }
            if let Some(item_id) = &call.provider_item_id {
                payload["item_id"] = Value::String(item_id.clone());
            }
            if let Some(status) = call.provider_status {
                payload["status"] = Value::String(status.as_str().to_string());
            }
            payload
        }
        Event::ServerToolUseStarted {
            index,
            call_id,
            name,
        } => serde_json::json!({
            "kind": "server_tool_use_started",
            "index": index,
            "call_id": call_id,
            "name": name,
        }),
        Event::ServerToolArgumentsDelta { index, delta } => serde_json::json!({
            "kind": "server_tool_arguments_delta",
            "index": index,
            "text": delta,
        }),
        Event::ServerToolUseCompleted { index, call } => serde_json::json!({
            "kind": "server_tool_use_completed",
            "index": index,
            "call_id": call.call_id,
            "name": call.name,
            "raw_arguments": call.raw_arguments,
        }),
        Event::ServerToolResult { index, block } => serde_json::json!({
            "kind": "server_tool_result",
            "index": index,
            "block": block,
        }),
        Event::HostedToolItemStarted {
            output_index,
            item_id,
            item_type,
            item,
        } => serde_json::json!({
            "kind": "hosted_tool_item_started",
            "output_index": output_index,
            "item_id": item_id,
            "item_type": item_type,
            "item": item,
        }),
        Event::HostedToolItemProgress {
            output_index,
            item_id,
            event_type,
            payload,
        } => serde_json::json!({
            "kind": "hosted_tool_item_progress",
            "output_index": output_index,
            "item_id": item_id,
            "event_type": event_type,
            "payload": payload,
        }),
        Event::HostedToolItemCompleted {
            output_index,
            item_id,
            item_type,
            item,
        } => serde_json::json!({
            "kind": "hosted_tool_item_completed",
            "output_index": output_index,
            "item_id": item_id,
            "item_type": item_type,
            "item": item,
        }),
        Event::ProviderTextAnnotation {
            output_index,
            item_id,
            annotation,
        } => serde_json::json!({
            "kind": "provider_text_annotation",
            "output_index": output_index,
            "item_id": item_id,
            "annotation": annotation,
        }),
        Event::TextBlockStarted { index } => serde_json::json!({
            "kind": "text_block_started",
            "index": index,
        }),
        Event::CitationDelta { index, citation } => serde_json::json!({
            "kind": "citation_delta",
            "index": index,
            "citation": citation,
        }),
        Event::Usage(usage) => {
            let mut payload = serde_json::json!({
                "kind": "usage",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            });
            if let Some(creation) = usage.cache_creation_input_tokens {
                payload["cache_creation_input_tokens"] = serde_json::json!(creation);
            }
            payload
        }
        // A stop-sequence cut is a completed turn to every python consumer
        // (guardrails, retention); only the public encoders name the sequence.
        Event::Completed | Event::StoppedAtSequence(_) => serde_json::json!({"kind": "completed"}),
        Event::Incomplete => serde_json::json!({"kind": "incomplete"}),
        Event::PausedTurn => serde_json::json!({"kind": "paused_turn"}),
        Event::Failed(failure) => {
            let mut value = serde_json::json!({
                "kind": "failed",
                "failure_class": failure.failure_class.as_str(),
                "safe_message": failure.safe_message,
            });
            if let Some(reason) = failure.refusal_reason {
                value["refusal_reason"] = serde_json::json!(reason.as_str());
            }
            value
        }
    }
}

fn add_provider_item_metadata(
    payload: &mut Value,
    item_id: &Option<String>,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
) {
    if let Some(item_id) = item_id {
        payload["item_id"] = Value::String(item_id.clone());
    }
    if let Some(status) = status {
        payload["status"] = Value::String(status.as_str().to_string());
    }
    if let Some(phase) = phase {
        payload["phase"] = Value::String(phase.as_str().to_string());
    }
}

/// Whether one hosted Responses item type names a tool INVOCATION.
///
/// Only invocations join the ledger's tool names: the hosted union also
/// carries results (`*_call_output`), approvals, listings, and opaque
/// conversation items (`additional_tools`, `compaction`), and recording one
/// of those would report a tool call that never occurred.
pub fn hosted_item_type_is_invocation(item_type: &str) -> bool {
    item_type.ends_with("_call")
}

/// Validate one raw tool-argument accumulation as a single JSON object.
///
/// The parse-failure reason carries serde's positional description (token
/// category and line/column, never input bytes), so an unparsable shape is
/// diagnosable from the boundary log without ever logging payload.
pub fn require_json_object_text(raw: &str) -> Result<(), String> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(_)) => Ok(()),
        Ok(_) => Err("streamed tool arguments must decode to an object".to_string()),
        Err(error) => Err(format!(
            "streamed tool arguments are not valid JSON: {error}"
        )),
    }
}

/// Accumulated per-stream state for one incrementally emitted function call.
#[derive(Debug, Clone)]
pub struct ToolAccumulator {
    pub call_id: String,
    pub name: String,
    /// Nested tool tree (Responses `namespace`) that declared this call.
    pub namespace: Option<String>,
    /// Opaque SDK 3.0 `caller` attribution carried verbatim.
    pub caller: Option<Value>,
    pub provider_item_id: Option<String>,
    pub provider_status: Option<ProviderOutputItemStatus>,
    pub raw_arguments: String,
    pub completed: bool,
    pub custom: bool,
    /// Whether this is a provider-executed Anthropic server tool
    /// (`server_tool_use`), whose lifecycle events stay on the dedicated
    /// server-tool variants and never count toward the tool-use stop reason.
    pub server: bool,
}

/// Opaque tool IDs share the Python model bound, including signature carriers.
const MAXIMUM_TOOL_CALL_ID_CHARACTERS: usize = 65_536;

impl ToolAccumulator {
    pub fn new(call_id: String, name: String) -> Self {
        Self {
            call_id,
            name,
            namespace: None,
            caller: None,
            provider_item_id: None,
            provider_status: None,
            raw_arguments: String::new(),
            completed: false,
            custom: false,
            server: false,
        }
    }

    pub fn complete(&self) -> Result<CompletedToolCall, String> {
        if !self.custom {
            // Custom (freeform) tool input is opaque text by contract; only
            // function arguments must parse as one JSON object.
            require_json_object_text(&self.raw_arguments)?;
        }
        // Mirror the python ToolCall model constraints so both engines accept
        // exactly the same provider tool-call streams (a call the python
        // engine rejects must not become client-visible history here).
        if self.call_id.is_empty()
            || self.call_id.chars().count() > MAXIMUM_TOOL_CALL_ID_CHARACTERS
            || self.name.is_empty()
            || self.name.chars().count() > 256
            || self
                .namespace
                .as_ref()
                .is_some_and(|namespace| namespace.is_empty() || namespace.chars().count() > 256)
            || self.raw_arguments.chars().count() > 4_000_000
        {
            return Err("streamed tool call is incomplete".to_string());
        }
        Ok(CompletedToolCall {
            call_id: self.call_id.clone(),
            name: self.name.clone(),
            namespace: self.namespace.clone(),
            caller: self.caller.clone(),
            provider_item_id: self.provider_item_id.clone(),
            provider_status: self.provider_status,
            raw_arguments: self.raw_arguments.clone(),
            custom: self.custom,
        })
    }
}

mod usage;
pub use usage::*;

#[cfg(test)]
mod tests;
