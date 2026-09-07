//! Public Anthropic Messages encoding, the Rust mirror of
//! `exp.runtime.anthropic_protocol.encoding` (`MessagesSseEncoder` and
//! `completed_messages_body`) and of the Anthropic error envelope in
//! `exp.runtime.anthropic_protocol.errors`.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use crate::dialects::MAXIMUM_RETAINED_OUTPUT_BYTES;
use crate::encode::{compact_json, stable_public_id};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};

const REFUSAL_MESSAGE: &str = "provider refused the request";

/// The sanitized failure for provider refusals on this surface, mirroring
/// `refusal_failure` in the python encoder.
pub fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, REFUSAL_MESSAGE)
}

/// Render one sanitized public error as the Anthropic error envelope,
/// mirroring `anthropic_error_body`: status decides the Anthropic type
/// first, then the OpenAI envelope type, and a present `param` pointer is
/// folded into the message text.
pub fn anthropic_error_body(error: &PublicError) -> Value {
    let error_type = match error.status_code {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ if error.error_type == "invalid_request_error" => "invalid_request_error",
        _ => "api_error",
    };
    let message = match &error.param {
        Some(param) if !param.is_empty() => format!("{} (param: {param})", error.message),
        _ => error.message.clone(),
    };
    let mut body = json!({
        "type": "error",
        "error": {"type": error_type, "message": message},
    });
    // A refusal carries its bounded category on the Anthropic envelope too, so
    // a Messages caller reads the same machine-readable reason as a Chat one.
    if let Some(reason) = error.refusal_reason {
        body["error"]["refusal_reason"] = json!(reason.as_str());
    }
    body
}

fn invalid_provider_stream(message: &str) -> PublicError {
    PublicError::new(502, "invalid_provider_stream", message, "api_error")
}

/// Map the terminal outcome to the Anthropic stop reason. Server tool use is
/// provider-executed and deliberately never yields `tool_use`; a paused
/// server-tool turn must keep `pause_turn` so the caller resumes it.
pub(super) fn stop_reason(terminal: &Event, saw_tool_use: bool) -> &'static str {
    match terminal {
        Event::Incomplete => "max_tokens",
        Event::PausedTurn => "pause_turn",
        // A gateway-emulated stop cut the visible text: the caller's sequence
        // ended the turn, exactly as Anthropic reports a native match.
        Event::StoppedAtSequence(_) => "stop_sequence",
        _ if saw_tool_use => "tool_use",
        _ => "end_turn",
    }
}

/// The matched stop sequence for the `stop_sequence` field, or null.
pub(super) fn stop_sequence_value(terminal: &Event) -> Value {
    match terminal {
        Event::StoppedAtSequence(sequence) => Value::String(sequence.clone()),
        _ => Value::Null,
    }
}

/// The Anthropic usage shape from `messages_usage`: cached reads come back
/// out of the normalized input total, and unknown usage reports zero counts
/// because the Anthropic shape requires both fields.
pub(super) fn messages_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return json!({"input_tokens": 0, "output_tokens": 0}),
    };
    let cached = usage.cached_input_tokens.unwrap_or(0);
    let creation = usage.cache_creation_input_tokens.unwrap_or(0);
    let mut body = Map::new();
    // Both cache legs come back out of the folded ledger total so callers
    // see the provider's own shape: input_tokens excludes cached reads and
    // cache writes, each reported on its own leg.
    body.insert(
        "input_tokens".to_string(),
        json!(usage
            .input_tokens
            .unwrap_or(0)
            .saturating_sub(cached)
            .saturating_sub(creation)),
    );
    body.insert(
        "output_tokens".to_string(),
        json!(usage.output_tokens.unwrap_or(0)),
    );
    if cached > 0 {
        body.insert("cache_read_input_tokens".to_string(), json!(cached));
    }
    if creation > 0 {
        body.insert("cache_creation_input_tokens".to_string(), json!(creation));
    }
    Value::Object(body)
}

/// Frame one named, compact, UTF-8-preserving Anthropic SSE event.
fn event_frame(name: &str, payload: &Value) -> String {
    format!("event: {name}\ndata: {}\n\n", compact_json(payload))
}

/// Frame one terminal Anthropic `error` SSE event for a sanitized failure.
fn error_frame(failure: &Failure) -> String {
    event_frame("error", &anthropic_error_body(&failure.public_error()))
}

/// The block families one Messages stream schedules, with their provider
/// grouping index where content arrives incrementally.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Tool(u32),
    Thinking(u32),
    Redacted,
    /// Provider-executed server tool use, streamed like a tool block but
    /// re-emitted as `server_tool_use` and excluded from the tool_use stop.
    ServerTool(u32),
    /// One whole verbatim server-tool result block (arrives complete).
    ServerResult,
}

/// One scheduled content block, buffered until it can stream in order.
///
/// Anthropic SSE streams content blocks strictly sequentially and a closed
/// index cannot reopen, while the canonical stream may legally interleave
/// parallel tool calls and trailing text. Blocks are therefore scheduled in
/// start order: the earliest block streams live, later blocks accumulate
/// their content in `pending` until every earlier block has closed.
struct PendingBlock {
    kind: BlockKind,
    pending: String,
    /// Thinking only: the opaque signature flushes as one `signature_delta`
    /// immediately before the block closes.
    pending_signature: String,
    /// Redacted only: the whole opaque payload travels in the start frame.
    redacted_data: Option<String>,
    /// Server result only: the whole verbatim block (validated compact JSON
    /// text) travels in the start frame.
    server_result_block: Option<String>,
    /// Text only: verbatim citation objects (validated compact JSON text),
    /// each flushed as one `citations_delta` while the block is open.
    pending_citations: Vec<String>,
    anthropic_index: Option<u32>,
}

impl PendingBlock {
    fn new(kind: BlockKind) -> Self {
        Self {
            kind,
            pending: String::new(),
            pending_signature: String::new(),
            redacted_data: None,
            server_result_block: None,
            pending_citations: Vec::new(),
            anthropic_index: None,
        }
    }
}

/// Stateful Anthropic Messages SSE encoder with one open block and one
/// terminal, emitting byte-identical frames to the python encoder.
///
/// Blocks are scheduled in start order (see [`PendingBlock`]): the earliest
/// block streams live while content for later blocks buffers within the
/// gateway's bounded retained-output budget, so interleaved parallel tool
/// calls and deferred completions encode as a valid strictly sequential
/// Anthropic lifecycle with the same block order as the non-streaming
/// aggregation.
pub struct MessagesSseEncoder {
    message_id: String,
    model: String,
    started: bool,
    terminal: bool,
    draining: bool,
    next_block_index: u32,
    blocks: Vec<PendingBlock>,
    open_position: Option<usize>,
    next_unopened: usize,
    buffered_bytes: usize,
    tool_identities: HashMap<u32, (String, String)>,
    tool_arguments: HashMap<u32, String>,
    tool_completed: HashSet<u32>,
    server_identities: HashMap<u32, (String, String)>,
    server_arguments: HashMap<u32, String>,
    server_completed: HashSet<u32>,
    saw_tool_use: bool,
    refusal_seen: bool,
    usage: Option<Usage>,
    ignored_parameters: Vec<String>,
}

impl MessagesSseEncoder {
    pub fn new(request_id: &str, model: &str) -> Self {
        Self::new_with_ignored(request_id, model, Vec::new())
    }

    /// Build an encoder that discloses controls omitted by route shaping,
    /// mirroring `ChatSseEncoder::new_with_ignored`.
    pub fn new_with_ignored(
        request_id: &str,
        model: &str,
        ignored_parameters: Vec<String>,
    ) -> Self {
        Self {
            message_id: stable_public_id("msg", request_id),
            model: model.to_string(),
            ignored_parameters,
            started: false,
            terminal: false,
            draining: false,
            next_block_index: 0,
            blocks: Vec::new(),
            open_position: None,
            next_unopened: 0,
            buffered_bytes: 0,
            tool_identities: HashMap::new(),
            tool_arguments: HashMap::new(),
            tool_completed: HashSet::new(),
            server_identities: HashMap::new(),
            server_arguments: HashMap::new(),
            server_completed: HashSet::new(),
            saw_tool_use: false,
            refusal_seen: false,
            usage: None,
        }
    }

    /// Emit the `message_start` and `ping` lifecycle events once.
    pub fn start(&mut self) -> Result<Vec<String>, PublicError> {
        if self.started {
            return Err(invalid_provider_stream(
                "Messages stream was started more than once.",
            ));
        }
        self.started = true;
        let mut message = json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        });
        // Same body-level disclosure as the Chat and Responses encoders: the
        // Anthropic envelope has no field for it, and the official SDK
        // tolerates extra keys, so a dropped control (an empty-ladder
        // `output_config.effort`, a dropped beta token) is never silent.
        disclose_ignored_parameters(&mut message, &self.ignored_parameters);
        Ok(vec![
            event_frame(
                "message_start",
                &json!({"type": "message_start", "message": message}),
            ),
            event_frame("ping", &json!({"type": "ping"})),
        ])
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Encode one ordered normalized provider event into zero or more frames.
    pub fn feed(&mut self, event: &Event) -> Result<Vec<String>, PublicError> {
        if !self.started {
            return Err(invalid_provider_stream(
                "Messages stream must be started before provider events.",
            ));
        }
        if self.terminal {
            return Err(invalid_provider_stream(
                "Messages stream received an event after its terminal.",
            ));
        }
        match event {
            Event::TextDelta(text) => self.text_delta(text),
            Event::ProviderTextDelta { delta, .. } => self.text_delta(delta),
            Event::RefusalDelta(_) => {
                // There is no Anthropic refusal block; the refusal is
                // reported as one sanitized terminal error instead.
                self.refusal_seen = true;
                Ok(Vec::new())
            }
            Event::ProviderRefusalDelta { .. } => {
                self.refusal_seen = true;
                Ok(Vec::new())
            }
            // OpenAI-only reasoning shapes have no Messages representation.
            Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::EncryptedReasoning { .. }
            | Event::ReasoningContentDelta { .. } => Ok(Vec::new()),
            Event::ThinkingDelta { index, delta } => self.thinking_delta(*index, delta),
            Event::ThinkingSignature { index, signature } => {
                self.thinking_signature(*index, signature)
            }
            Event::RedactedThinking { data, .. } => self.redacted_thinking(data),
            Event::ToolCallStarted {
                index,
                call_id,
                name,
                ..
            } => self.tool_started(*index, call_id, name),
            Event::ToolArgumentsDelta { index, delta } => self.tool_arguments_delta(*index, delta),
            Event::ToolCallCompleted { index, call } => {
                if call.custom {
                    // Custom tools only enter through a Responses request,
                    // which never encodes on the Messages surface.
                    return Err(invalid_provider_stream(
                        "Messages cannot represent a custom tool call.",
                    ));
                }
                // Some upstream dialects (OpenAI-compatible streams) emit
                // every tool completion only at their terminal sentinel, and
                // parallel tool calls may interleave, so completion verifies
                // against the accumulated state and the scheduler closes the
                // block once it is the open one.
                let identity = self.tool_identities.get(index);
                if identity.is_none() || self.tool_completed.contains(index) {
                    return Err(invalid_provider_stream(
                        "Messages tool completion omitted its started tool call.",
                    ));
                }
                let streamed = self.tool_arguments.get(index).cloned().unwrap_or_default();
                if identity != Some(&(call.call_id.clone(), call.name.clone()))
                    || streamed != call.raw_arguments
                {
                    return Err(invalid_provider_stream(
                        "Messages tool completion changed streamed identity or bytes.",
                    ));
                }
                self.tool_completed.insert(*index);
                Ok(self.advance())
            }
            Event::TextBlockStarted { .. } => {
                // A provider text-block boundary starts a fresh caller block
                // so citations attach to the block they belong to.
                self.blocks.push(PendingBlock::new(BlockKind::Text));
                Ok(self.advance())
            }
            Event::CitationDelta { citation, .. } => self.citation_delta(citation),
            Event::ServerToolUseStarted {
                index,
                call_id,
                name,
            } => self.server_tool_started(*index, call_id, name),
            Event::ServerToolArgumentsDelta { index, delta } => {
                self.server_tool_arguments_delta(*index, delta)
            }
            Event::ServerToolUseCompleted { index, call } => {
                let identity = self.server_identities.get(index);
                if identity.is_none() || self.server_completed.contains(index) {
                    return Err(invalid_provider_stream(
                        "Messages server tool completion omitted its started tool use.",
                    ));
                }
                let streamed = self
                    .server_arguments
                    .get(index)
                    .cloned()
                    .unwrap_or_default();
                if identity != Some(&(call.call_id.clone(), call.name.clone()))
                    || streamed != call.raw_arguments
                {
                    return Err(invalid_provider_stream(
                        "Messages server tool completion changed streamed identity or bytes.",
                    ));
                }
                self.server_completed.insert(*index);
                Ok(self.advance())
            }
            Event::ServerToolResult { block, .. } => self.server_tool_result(block),
            // Hosted tool items enter only through Responses-native tool
            // declarations, which never admit on the Messages surface.
            Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. } => Err(invalid_provider_stream(
                "Messages cannot represent a provider-hosted Responses tool item.",
            )),
            // OpenAI text annotations have no Messages representation; the
            // text itself streams through its delta events.
            Event::ProviderTextAnnotation { .. } => Ok(Vec::new()),
            Event::Usage(usage) => {
                if usage.has_token_counts() {
                    self.usage = Some(usage.clone());
                }
                Ok(Vec::new())
            }
            Event::Completed
            | Event::Incomplete
            | Event::StoppedAtSequence(_)
            | Event::PausedTurn => {
                self.terminal = true;
                if self.refusal_seen {
                    return Ok(vec![error_frame(&refusal_failure())]);
                }
                self.draining = true;
                let mut frames = self.advance();
                frames.push(event_frame(
                    "message_delta",
                    &json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason(event, self.saw_tool_use),
                            "stop_sequence": stop_sequence_value(event),
                        },
                        "usage": messages_usage(self.usage.as_ref()),
                    }),
                ));
                frames.push(event_frame(
                    "message_stop",
                    &json!({"type": "message_stop"}),
                ));
                Ok(frames)
            }
            Event::Failed(failure) => {
                self.terminal = true;
                Ok(vec![error_frame(failure)])
            }
        }
    }

    /// Schedule one text delta on the last text block, buffering as needed.
    fn text_delta(&mut self, delta: &str) -> Result<Vec<String>, PublicError> {
        let needs_new_block = match self.blocks.last() {
            Some(block) => block.kind != BlockKind::Text,
            None => true,
        };
        if needs_new_block {
            self.blocks.push(PendingBlock::new(BlockKind::Text));
        }
        let position = self.blocks.len() - 1;
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Schedule one thinking text delta on its provider-indexed block.
    fn thinking_delta(&mut self, index: u32, delta: &str) -> Result<Vec<String>, PublicError> {
        let position = self.thinking_position(index);
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Retain one opaque signature fragment; it flushes at block close.
    fn thinking_signature(
        &mut self,
        index: u32,
        signature: &str,
    ) -> Result<Vec<String>, PublicError> {
        let position = self.thinking_position(index);
        self.buffered_bytes = self.buffered_bytes.saturating_add(signature.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        self.blocks[position].pending_signature.push_str(signature);
        Ok(self.advance())
    }

    /// Schedule one complete redacted-thinking block at its arrival position.
    fn redacted_thinking(&mut self, data: &str) -> Result<Vec<String>, PublicError> {
        self.buffered_bytes = self.buffered_bytes.saturating_add(data.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        let mut block = PendingBlock::new(BlockKind::Redacted);
        block.redacted_data = Some(data.to_string());
        self.blocks.push(block);
        Ok(self.advance())
    }

    /// Find or schedule the thinking block for one provider index.
    fn thinking_position(&mut self, index: u32) -> usize {
        if let Some(position) = self
            .blocks
            .iter()
            .position(|block| block.kind == BlockKind::Thinking(index))
        {
            return position;
        }
        self.blocks
            .push(PendingBlock::new(BlockKind::Thinking(index)));
        self.blocks.len() - 1
    }

    /// Attach one verbatim citation to the newest text block, creating one
    /// when the citation leads its block's content.
    fn citation_delta(&mut self, citation: &str) -> Result<Vec<String>, PublicError> {
        self.buffered_bytes = self.buffered_bytes.saturating_add(citation.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        if serde_json::from_str::<Value>(citation).is_err() {
            return Err(invalid_provider_stream(
                "Messages citation was not valid JSON.",
            ));
        }
        let position = match self
            .blocks
            .iter()
            .rposition(|block| block.kind == BlockKind::Text)
        {
            Some(position) => position,
            None => {
                self.blocks.push(PendingBlock::new(BlockKind::Text));
                self.blocks.len() - 1
            }
        };
        self.blocks[position]
            .pending_citations
            .push(citation.to_string());
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Schedule one server_tool_use block at its start position.
    fn server_tool_started(
        &mut self,
        tool_index: u32,
        call_id: &str,
        name: &str,
    ) -> Result<Vec<String>, PublicError> {
        if self.server_identities.contains_key(&tool_index) {
            return Err(invalid_provider_stream(
                "A Messages server tool index was started twice.",
            ));
        }
        self.server_identities
            .insert(tool_index, (call_id.to_string(), name.to_string()));
        self.server_arguments.insert(tool_index, String::new());
        self.blocks
            .push(PendingBlock::new(BlockKind::ServerTool(tool_index)));
        Ok(self.advance())
    }

    /// Schedule one raw server-tool input fragment behind earlier blocks.
    fn server_tool_arguments_delta(
        &mut self,
        tool_index: u32,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        if !self.server_identities.contains_key(&tool_index) {
            return Err(invalid_provider_stream(
                "Messages server tool arguments arrived before its start.",
            ));
        }
        if self.server_completed.contains(&tool_index) {
            return Err(invalid_provider_stream(
                "Messages server tool arguments arrived after completion.",
            ));
        }
        self.server_arguments
            .get_mut(&tool_index)
            .expect("started server tool has accumulated arguments")
            .push_str(delta);
        let position = self
            .blocks
            .iter()
            .position(|block| block.kind == BlockKind::ServerTool(tool_index))
            .expect("started server tool has a scheduled block");
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Schedule one whole verbatim server-tool result block at its position.
    fn server_tool_result(&mut self, block: &str) -> Result<Vec<String>, PublicError> {
        self.buffered_bytes = self.buffered_bytes.saturating_add(block.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        if serde_json::from_str::<Value>(block).is_err() {
            return Err(invalid_provider_stream(
                "Messages server tool result was not valid JSON.",
            ));
        }
        let mut pending = PendingBlock::new(BlockKind::ServerResult);
        pending.server_result_block = Some(block.to_string());
        self.blocks.push(pending);
        Ok(self.advance())
    }

    /// Schedule one tool_use block at its start position.
    fn tool_started(
        &mut self,
        tool_index: u32,
        call_id: &str,
        name: &str,
    ) -> Result<Vec<String>, PublicError> {
        if self.tool_identities.contains_key(&tool_index) {
            return Err(invalid_provider_stream(
                "A Messages tool-call index was started twice.",
            ));
        }
        self.tool_identities
            .insert(tool_index, (call_id.to_string(), name.to_string()));
        self.tool_arguments.insert(tool_index, String::new());
        self.saw_tool_use = true;
        self.blocks
            .push(PendingBlock::new(BlockKind::Tool(tool_index)));
        Ok(self.advance())
    }

    /// Schedule one raw argument fragment, buffering behind earlier blocks.
    fn tool_arguments_delta(
        &mut self,
        tool_index: u32,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        if !self.tool_identities.contains_key(&tool_index) {
            return Err(invalid_provider_stream(
                "Messages tool arguments arrived before tool-call start.",
            ));
        }
        if self.tool_completed.contains(&tool_index) {
            return Err(invalid_provider_stream(
                "Messages tool arguments arrived after completion.",
            ));
        }
        self.tool_arguments
            .get_mut(&tool_index)
            .expect("started tool has accumulated arguments")
            .push_str(delta);
        let position = self
            .blocks
            .iter()
            .position(|block| block.kind == BlockKind::Tool(tool_index))
            .expect("started tool has a scheduled block");
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Retain one content fragment for its block within the bounded budget.
    fn buffer(&mut self, position: usize, delta: &str) -> Result<(), PublicError> {
        self.buffered_bytes = self.buffered_bytes.saturating_add(delta.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        self.blocks[position].pending.push_str(delta);
        Ok(())
    }

    /// Close and open blocks in start order as far as the stream allows.
    ///
    /// A text block closes once a later block exists (or at drain); a tool
    /// block closes only after its verified completion. Opening a block
    /// assigns the next sequential Anthropic index and flushes its buffered
    /// content as one delta.
    fn advance(&mut self) -> Vec<String> {
        let mut frames = Vec::new();
        loop {
            if let Some(position) = self.open_position {
                let block = &self.blocks[position];
                let last = position == self.blocks.len() - 1;
                let closable = self.draining
                    || match block.kind {
                        BlockKind::Text
                        | BlockKind::Thinking(_)
                        | BlockKind::Redacted
                        | BlockKind::ServerResult => !last,
                        BlockKind::Tool(tool_index) => self.tool_completed.contains(&tool_index),
                        BlockKind::ServerTool(tool_index) => {
                            self.server_completed.contains(&tool_index)
                        }
                    };
                if !closable {
                    return frames;
                }
                self.flush_citations(position, &mut frames);
                self.flush_signature(position, &mut frames);
                let block = &self.blocks[position];
                frames.push(event_frame(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": block.anthropic_index}),
                ));
                self.open_position = None;
                continue;
            }
            if self.next_unopened >= self.blocks.len() {
                return frames;
            }
            let position = self.next_unopened;
            self.open_position = Some(position);
            self.next_unopened += 1;
            let anthropic_index = self.next_block_index;
            self.next_block_index += 1;
            let block = &mut self.blocks[position];
            block.anthropic_index = Some(anthropic_index);
            let content_block = match block.kind {
                BlockKind::Text => json!({"type": "text", "text": ""}),
                // The SDK thinking block type requires both fields, so the
                // start frame carries their empty forms like the provider.
                BlockKind::Thinking(_) => {
                    json!({"type": "thinking", "thinking": "", "signature": ""})
                }
                BlockKind::Redacted => {
                    let data = block.redacted_data.take().unwrap_or_default();
                    self.buffered_bytes = self.buffered_bytes.saturating_sub(data.len());
                    json!({"type": "redacted_thinking", "data": data})
                }
                BlockKind::Tool(tool_index) => {
                    let identity = self
                        .tool_identities
                        .get(&tool_index)
                        .expect("scheduled tool has an identity");
                    json!({
                        "type": "tool_use",
                        "id": identity.0,
                        "name": identity.1,
                        "input": {},
                    })
                }
                BlockKind::ServerTool(tool_index) => {
                    let identity = self
                        .server_identities
                        .get(&tool_index)
                        .expect("scheduled server tool has an identity");
                    json!({
                        "type": "server_tool_use",
                        "id": identity.0,
                        "name": identity.1,
                        "input": {},
                    })
                }
                BlockKind::ServerResult => {
                    let raw = block.server_result_block.take().unwrap_or_default();
                    self.buffered_bytes = self.buffered_bytes.saturating_sub(raw.len());
                    serde_json::from_str(&raw)
                        .expect("scheduled server result was validated as JSON")
                }
            };
            frames.push(event_frame(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": anthropic_index,
                    "content_block": content_block,
                }),
            ));
            self.flush_open(&mut frames);
        }
    }

    /// Emit each retained verbatim citation as one `citations_delta` frame.
    fn flush_citations(&mut self, position: usize, frames: &mut Vec<String>) {
        let block = &mut self.blocks[position];
        if block.kind != BlockKind::Text || block.pending_citations.is_empty() {
            return;
        }
        let anthropic_index = block.anthropic_index;
        for citation in block.pending_citations.drain(..) {
            self.buffered_bytes = self.buffered_bytes.saturating_sub(citation.len());
            let parsed: Value =
                serde_json::from_str(&citation).expect("retained citation was validated as JSON");
            frames.push(event_frame(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": anthropic_index,
                    "delta": {"type": "citations_delta", "citation": parsed},
                }),
            ));
        }
    }

    /// Emit one closing `signature_delta` for a thinking block, if retained.
    fn flush_signature(&mut self, position: usize, frames: &mut Vec<String>) {
        let block = &mut self.blocks[position];
        if !matches!(block.kind, BlockKind::Thinking(_)) || block.pending_signature.is_empty() {
            return;
        }
        frames.push(event_frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": block.anthropic_index,
                "delta": {"type": "signature_delta", "signature": block.pending_signature},
            }),
        ));
        self.buffered_bytes = self
            .buffered_bytes
            .saturating_sub(block.pending_signature.len());
        block.pending_signature.clear();
    }

    /// Emit the open block's buffered content as one delta, if any.
    /// Citations flush first: on the provider wire a block's citations can
    /// precede the text they cover, and the accumulated block is identical
    /// either way.
    fn flush_open(&mut self, frames: &mut Vec<String>) {
        let Some(position) = self.open_position else {
            return;
        };
        self.flush_citations(position, frames);
        let block = &mut self.blocks[position];
        if block.pending.is_empty() {
            return;
        }
        let delta = match block.kind {
            BlockKind::Text => json!({"type": "text_delta", "text": block.pending}),
            BlockKind::Thinking(_) => json!({"type": "thinking_delta", "thinking": block.pending}),
            // Redacted and server-result blocks carry their whole payload in
            // the start frame and never buffer deltas.
            BlockKind::Redacted | BlockKind::ServerResult => return,
            BlockKind::Tool(_) | BlockKind::ServerTool(_) => {
                json!({"type": "input_json_delta", "partial_json": block.pending})
            }
        };
        frames.push(event_frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": block.anthropic_index,
                "delta": delta,
            }),
        ));
        self.buffered_bytes = self.buffered_bytes.saturating_sub(block.pending.len());
        block.pending.clear();
    }
}

mod aggregate;

pub use aggregate::{completed_messages_body, completed_messages_body_with_ignored};

/// Attach the `x-experiential-ignored-parameters` disclosure to one message
/// object when any control was dropped; an empty list adds nothing.
pub(super) fn disclose_ignored_parameters(message: &mut Value, ignored_parameters: &[String]) {
    if ignored_parameters.is_empty() {
        return;
    }
    message
        .as_object_mut()
        .expect("Anthropic message is an object")
        .insert(
            "x-experiential-ignored-parameters".to_string(),
            json!(ignored_parameters),
        );
}

#[cfg(test)]
mod tests;
