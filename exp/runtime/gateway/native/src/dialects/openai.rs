//! OpenAI frame mappings: the Responses SSE dialect and the Chat-Completions
//! compatible dialect, mirroring the python event mappers.

use serde_json::Value;

use super::{
    bounded_wire_token, complete_streamed_tool, complete_streamed_tool_truncated,
    finish_open_tools, finish_open_tools_truncated, malformed, optional_text, parse_object,
    provider_error_detail, refusal_failure, Normalizer,
};
use crate::errors::{Failure, FailureClass};
use crate::events::{
    openai_usage, require_bounded_string, require_string, require_u64, Event,
    ProviderAssistantMessagePhase, ProviderOutputItemKind, ProviderOutputItemStatus,
    ToolAccumulator,
};

const MAXIMUM_OPENAI_ID_CHARS: usize = 256;

mod hosted;
use hosted::{is_openai_hosted_item_type, is_openai_hosted_progress_event};

fn openai_identity(
    object: &serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<String, Failure> {
    require_bounded_string(object, key, label, MAXIMUM_OPENAI_ID_CHARS)
        .map_err(|message| malformed(&message))
}

fn optional_openai_identity(
    object: &serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Option<String>, Failure> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => openai_identity(object, key, label).map(Some),
    }
}

fn optional_openai_caller(
    object: &serde_json::Map<String, Value>,
    label: &str,
) -> Result<Option<Value>, Failure> {
    match object.get("caller") {
        None | Some(Value::Null) => Ok(None),
        Some(value @ Value::Object(_)) => Ok(Some(value.clone())),
        Some(_) => Err(malformed(&format!("{label} must be an object"))),
    }
}

fn openai_status(
    object: &serde_json::Map<String, Value>,
) -> Result<Option<ProviderOutputItemStatus>, Failure> {
    match object.get("status") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => ProviderOutputItemStatus::from_str(value)
            .map(Some)
            .ok_or_else(|| malformed("OpenAI output item status is invalid")),
        Some(_) => Err(malformed("OpenAI output item status must be text")),
    }
}

fn openai_message_phase(
    object: &serde_json::Map<String, Value>,
) -> Result<Option<ProviderAssistantMessagePhase>, Failure> {
    match object.get("phase") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => ProviderAssistantMessagePhase::from_str(value)
            .map(Some)
            .ok_or_else(|| malformed("OpenAI assistant message phase is invalid")),
        Some(_) => Err(malformed("OpenAI assistant message phase must be text")),
    }
}

fn openai_index(
    object: &serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<u32, Failure> {
    let value = require_u64(object, key, label).map_err(|message| malformed(&message))?;
    u32::try_from(value).map_err(|_| malformed(&format!("{label} exceeds the supported range")))
}

impl Normalizer {
    fn reconcile_openai_tool_arguments(
        &mut self,
        index: u32,
        final_arguments: &str,
        events: &mut Vec<Event>,
    ) -> Result<(), Failure> {
        let streamed = self
            .tools
            .get(&index)
            .ok_or_else(|| malformed("OpenAI function call completed before its start"))?
            .raw_arguments
            .clone();
        if !streamed.is_empty() && streamed != final_arguments {
            return Err(malformed("OpenAI tool argument fragments changed at done"));
        }
        if streamed.is_empty() && !final_arguments.is_empty() {
            self.reserve_tool_bytes(final_arguments.len())?;
            self.tools
                .get_mut(&index)
                .expect("tool just checked")
                .raw_arguments
                .push_str(final_arguments);
            events.push(Event::ToolArgumentsDelta {
                index,
                delta: final_arguments.to_string(),
            });
        }
        Ok(())
    }

    pub(super) fn feed_openai_responses(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        if frame.data == "[DONE]" {
            return Err(malformed(
                "OpenAI Responses stream ended before a terminal event",
            ));
        }
        let payload = parse_object(&frame.data)?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| frame.event.clone())
            .unwrap_or_default();
        let mut events = Vec::new();
        match event_type.as_str() {
            "response.output_text.delta" => {
                let output_index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                let item_id = openai_identity(&payload, "item_id", "OpenAI message item ID")?;
                if self.bind_openai_output_item(
                    output_index,
                    ProviderOutputItemKind::Message,
                    Some(item_id.clone()),
                )? {
                    events.push(Event::ProviderOutputItemStarted {
                        output_index,
                        item_id: Some(item_id.clone()),
                        kind: ProviderOutputItemKind::Message,
                        status: None,
                        phase: None,
                    });
                }
                let delta = optional_text(&payload, "delta", "OpenAI text delta")?;
                if !delta.is_empty() {
                    events.push(Event::ProviderTextDelta {
                        output_index,
                        item_id,
                        delta,
                    });
                }
            }
            "response.refusal.delta" => {
                let output_index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                let item_id = openai_identity(&payload, "item_id", "OpenAI message item ID")?;
                if self.bind_openai_output_item(
                    output_index,
                    ProviderOutputItemKind::Message,
                    Some(item_id.clone()),
                )? {
                    events.push(Event::ProviderOutputItemStarted {
                        output_index,
                        item_id: Some(item_id.clone()),
                        kind: ProviderOutputItemKind::Message,
                        status: None,
                        phase: None,
                    });
                }
                let delta = optional_text(&payload, "delta", "OpenAI refusal delta")?;
                self.refusal_seen = true;
                events.push(Event::ProviderRefusalDelta {
                    output_index,
                    item_id,
                    delta,
                });
            }
            "response.output_text.annotation.added" => {
                events.extend(self.openai_text_annotation(&payload)?);
            }
            "response.reasoning_summary_text.delta" => {
                let output_index =
                    openai_index(&payload, "output_index", "OpenAI reasoning output_index")?;
                let summary_index =
                    openai_index(&payload, "summary_index", "OpenAI reasoning summary_index")?;
                let item_id = openai_identity(&payload, "item_id", "OpenAI reasoning item ID")?;
                let delta = optional_text(&payload, "delta", "OpenAI reasoning summary delta")?;
                if delta.is_empty() {
                    if self
                        .openai_output_items
                        .get(&output_index)
                        .is_some_and(|identity| {
                            identity != &(ProviderOutputItemKind::Reasoning, Some(item_id.clone()))
                        })
                    {
                        return Err(malformed(
                            "OpenAI output item changed identity or type during streaming",
                        ));
                    }
                    return Ok(events);
                }
                if self.bind_openai_output_item(
                    output_index,
                    ProviderOutputItemKind::Reasoning,
                    Some(item_id.clone()),
                )? {
                    events.push(Event::ProviderOutputItemStarted {
                        output_index,
                        item_id: Some(item_id.clone()),
                        kind: ProviderOutputItemKind::Reasoning,
                        status: None,
                        phase: None,
                    });
                }
                let key = (output_index, summary_index);
                self.reserve_summary_entry(key)?;
                self.reserve_summary_bytes(delta.len())?;
                self.reasoning_summaries
                    .entry(key)
                    .or_default()
                    .push_str(&delta);
                events.push(Event::ReasoningSummaryDelta {
                    output_index,
                    summary_index,
                    item_id,
                    delta,
                });
            }
            "response.reasoning_summary_text.done" => {
                let output_index =
                    openai_index(&payload, "output_index", "OpenAI reasoning output_index")?;
                let summary_index =
                    openai_index(&payload, "summary_index", "OpenAI reasoning summary_index")?;
                let item_id = openai_identity(&payload, "item_id", "OpenAI reasoning item ID")?;
                let final_text = require_string(&payload, "text", "OpenAI reasoning summary text")
                    .map_err(|message| malformed(&message))?;
                if self.bind_openai_output_item(
                    output_index,
                    ProviderOutputItemKind::Reasoning,
                    Some(item_id.clone()),
                )? {
                    events.push(Event::ProviderOutputItemStarted {
                        output_index,
                        item_id: Some(item_id.clone()),
                        kind: ProviderOutputItemKind::Reasoning,
                        status: None,
                        phase: None,
                    });
                }
                let key = (output_index, summary_index);
                let streamed = self
                    .reasoning_summaries
                    .get(&key)
                    .map(String::as_str)
                    .unwrap_or("");
                if !streamed.is_empty() && streamed != final_text {
                    return Err(malformed(
                        "OpenAI reasoning summary fragments changed at done",
                    ));
                }
                if streamed.is_empty() && !final_text.is_empty() {
                    self.reserve_summary_entry(key)?;
                    self.reserve_summary_bytes(final_text.len())?;
                    self.reasoning_summaries.insert(key, final_text.clone());
                    events.push(Event::ReasoningSummaryDelta {
                        output_index,
                        summary_index,
                        item_id,
                        delta: final_text,
                    });
                }
            }
            "response.output_item.added" => {
                let item = payload
                    .get("item")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("OpenAI output item must be an object"))?;
                let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                let index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                if is_openai_hosted_item_type(item_type) {
                    events.extend(self.openai_hosted_item_added(index, item_type, item)?);
                    return Ok(events);
                }
                let status = openai_status(item)?;
                if item_type == "function_call" {
                    if self.tools.contains_key(&index) {
                        return Err(malformed("OpenAI stream repeated a tool-call start"));
                    }
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| malformed("OpenAI function call ID must be text"))?;
                    if call_id.is_empty() || call_id.chars().count() > MAXIMUM_OPENAI_ID_CHARS {
                        return Err(malformed(
                            "OpenAI function call ID must contain between 1 and 256 characters",
                        ));
                    }
                    let call_id = call_id.to_string();
                    let name = openai_identity(item, "name", "OpenAI function call name")?;
                    let namespace = optional_openai_identity(
                        item,
                        "namespace",
                        "OpenAI function call namespace",
                    )?;
                    let caller = optional_openai_caller(item, "OpenAI function call caller")?;
                    let item_id =
                        optional_openai_identity(item, "id", "OpenAI function call item ID")?;
                    if !self.bind_openai_output_item(
                        index,
                        ProviderOutputItemKind::FunctionCall,
                        item_id.clone(),
                    )? {
                        return Err(malformed("OpenAI stream repeated an output-item start"));
                    }
                    self.reserve_tool_entry(index)?;
                    let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
                    tool.namespace = namespace.clone();
                    tool.caller = caller.clone();
                    tool.provider_item_id = item_id.clone();
                    tool.provider_status = status;
                    events.push(Event::ProviderOutputItemStarted {
                        output_index: index,
                        item_id,
                        kind: ProviderOutputItemKind::FunctionCall,
                        status,
                        phase: None,
                    });
                    events.push(Event::ToolCallStarted {
                        index,
                        call_id,
                        name,
                        namespace,
                        caller,
                    });
                    if let Some(Value::String(initial)) = item.get("arguments") {
                        if !initial.is_empty() {
                            self.reserve_tool_bytes(initial.len())?;
                            tool.raw_arguments.push_str(initial);
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: initial.clone(),
                            });
                        }
                    }
                    self.tools.insert(index, tool);
                } else if item_type == "custom_tool_call" {
                    // A freeform (custom) tool call: identical lifecycle to a
                    // function call, with opaque `input` text instead of JSON
                    // `arguments` (stream captured live 2026-08-30).
                    if self.tools.contains_key(&index) {
                        return Err(malformed("OpenAI stream repeated a tool-call start"));
                    }
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| malformed("OpenAI custom tool call ID must be text"))?;
                    if call_id.is_empty() || call_id.chars().count() > MAXIMUM_OPENAI_ID_CHARS {
                        return Err(malformed(
                            "OpenAI custom tool call ID must contain between 1 and 256 characters",
                        ));
                    }
                    let call_id = call_id.to_string();
                    let name = openai_identity(item, "name", "OpenAI custom tool call name")?;
                    let namespace = optional_openai_identity(
                        item,
                        "namespace",
                        "OpenAI custom tool call namespace",
                    )?;
                    let caller = optional_openai_caller(item, "OpenAI custom tool call caller")?;
                    let item_id =
                        optional_openai_identity(item, "id", "OpenAI custom tool call item ID")?;
                    if !self.bind_openai_output_item(
                        index,
                        ProviderOutputItemKind::CustomToolCall,
                        item_id.clone(),
                    )? {
                        return Err(malformed("OpenAI stream repeated an output-item start"));
                    }
                    self.reserve_tool_entry(index)?;
                    let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
                    tool.custom = true;
                    tool.namespace = namespace.clone();
                    tool.caller = caller.clone();
                    tool.provider_item_id = item_id.clone();
                    tool.provider_status = status;
                    events.push(Event::ProviderOutputItemStarted {
                        output_index: index,
                        item_id,
                        kind: ProviderOutputItemKind::CustomToolCall,
                        status,
                        phase: None,
                    });
                    events.push(Event::ToolCallStarted {
                        index,
                        call_id,
                        name,
                        namespace,
                        caller,
                    });
                    if let Some(Value::String(initial)) = item.get("input") {
                        if !initial.is_empty() {
                            self.reserve_tool_bytes(initial.len())?;
                            tool.raw_arguments.push_str(initial);
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: initial.clone(),
                            });
                        }
                    }
                    self.tools.insert(index, tool);
                } else if item_type == "reasoning" {
                    let item_id = openai_identity(item, "id", "OpenAI reasoning item ID")?;
                    if !self.bind_openai_output_item(
                        index,
                        ProviderOutputItemKind::Reasoning,
                        Some(item_id.clone()),
                    )? {
                        return Err(malformed("OpenAI stream repeated an output-item start"));
                    }
                    events.push(Event::ProviderOutputItemStarted {
                        output_index: index,
                        item_id: Some(item_id),
                        kind: ProviderOutputItemKind::Reasoning,
                        status,
                        phase: None,
                    });
                } else if item_type == "message" {
                    let item_id = openai_identity(item, "id", "OpenAI message item ID")?;
                    let phase = openai_message_phase(item)?;
                    if !self.bind_openai_output_item(
                        index,
                        ProviderOutputItemKind::Message,
                        Some(item_id.clone()),
                    )? {
                        return Err(malformed("OpenAI stream repeated an output-item start"));
                    }
                    events.push(Event::ProviderOutputItemStarted {
                        output_index: index,
                        item_id: Some(item_id),
                        kind: ProviderOutputItemKind::Message,
                        status,
                        phase,
                    });
                } else {
                    return Err(malformed(&format!(
                        "OpenAI stream emitted an unsupported output item (type {})",
                        bounded_wire_token(item_type),
                    )));
                }
            }
            "response.function_call_arguments.delta" => {
                let index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                let item_id = openai_identity(&payload, "item_id", "OpenAI function call item ID")?;
                let delta = optional_text(&payload, "delta", "OpenAI argument delta")?;
                self.reserve_tool_bytes(delta.len())?;
                let tool = self
                    .tools
                    .get_mut(&index)
                    .ok_or_else(|| malformed("provider emitted arguments before a tool start"))?;
                if tool.provider_item_id.as_deref() != Some(item_id.as_str()) {
                    return Err(malformed(
                        "OpenAI function call changed identity during arguments",
                    ));
                }
                tool.raw_arguments.push_str(&delta);
                events.push(Event::ToolArgumentsDelta { index, delta });
            }
            "response.function_call_arguments.done" => {
                let index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                let item_id = openai_identity(&payload, "item_id", "OpenAI function call item ID")?;
                let tool = self
                    .tools
                    .get(&index)
                    .ok_or_else(|| malformed("OpenAI function call completed before its start"))?;
                if tool.provider_item_id.as_deref() != Some(item_id.as_str()) {
                    return Err(malformed(
                        "OpenAI function call changed identity at arguments done",
                    ));
                }
                let arguments =
                    require_string(&payload, "arguments", "OpenAI function call arguments")
                        .map_err(|message| malformed(&message))?;
                self.reconcile_openai_tool_arguments(index, &arguments, &mut events)?;
            }
            "response.custom_tool_call_input.delta" => {
                let index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                let item_id =
                    openai_identity(&payload, "item_id", "OpenAI custom tool call item ID")?;
                let delta = optional_text(&payload, "delta", "OpenAI input delta")?;
                self.reserve_tool_bytes(delta.len())?;
                let tool = self
                    .tools
                    .get_mut(&index)
                    .ok_or_else(|| malformed("provider emitted input before a tool start"))?;
                if !tool.custom || tool.provider_item_id.as_deref() != Some(item_id.as_str()) {
                    return Err(malformed(
                        "OpenAI custom tool call changed identity during input",
                    ));
                }
                tool.raw_arguments.push_str(&delta);
                events.push(Event::ToolArgumentsDelta { index, delta });
            }
            "response.custom_tool_call_input.done" => {
                let index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                let item_id =
                    openai_identity(&payload, "item_id", "OpenAI custom tool call item ID")?;
                let tool = self.tools.get(&index).ok_or_else(|| {
                    malformed("OpenAI custom tool call completed before its start")
                })?;
                if !tool.custom || tool.provider_item_id.as_deref() != Some(item_id.as_str()) {
                    return Err(malformed(
                        "OpenAI custom tool call changed identity at input done",
                    ));
                }
                let input = require_string(&payload, "input", "OpenAI custom tool call input")
                    .map_err(|message| malformed(&message))?;
                self.reconcile_openai_tool_arguments(index, &input, &mut events)?;
            }
            "response.output_item.done" => {
                let index = openai_index(&payload, "output_index", "OpenAI output_index")?;
                if !self.openai_completed_output_items.insert(index) {
                    return Err(malformed(
                        "OpenAI stream repeated an output-item completion",
                    ));
                }
                let item = payload
                    .get("item")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("OpenAI completed output item must be an object"))?;
                let done_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                if is_openai_hosted_item_type(done_type) {
                    events.extend(self.openai_hosted_item_done(index, done_type, item)?);
                    return Ok(events);
                }
                let status = openai_status(item)?;
                match item.get("type").and_then(Value::as_str) {
                    Some("reasoning") => {
                        let item_id = openai_identity(item, "id", "OpenAI reasoning item ID")?;
                        if self.bind_openai_output_item(
                            index,
                            ProviderOutputItemKind::Reasoning,
                            Some(item_id.clone()),
                        )? {
                            events.push(Event::ProviderOutputItemStarted {
                                output_index: index,
                                item_id: Some(item_id.clone()),
                                kind: ProviderOutputItemKind::Reasoning,
                                status: None,
                                phase: None,
                            });
                        }
                        match item.get("encrypted_content") {
                            Some(Value::String(encrypted)) if !encrypted.is_empty() => {
                                self.reserve_summary_bytes(encrypted.len())?;
                                events.push(Event::EncryptedReasoning {
                                    output_index: index,
                                    item_id: item_id.clone(),
                                    encrypted_content: encrypted.clone(),
                                });
                            }
                            None | Some(Value::Null) | Some(Value::String(_)) => {}
                            Some(_) => {
                                return Err(malformed(
                                    "OpenAI encrypted reasoning content must be text",
                                ))
                            }
                        }
                        events.push(Event::ProviderOutputItemCompleted {
                            output_index: index,
                            item_id: Some(item_id),
                            kind: ProviderOutputItemKind::Reasoning,
                            status,
                            phase: None,
                        });
                    }
                    Some("message") => {
                        let item_id = openai_identity(item, "id", "OpenAI message item ID")?;
                        let phase = openai_message_phase(item)?;
                        if self.bind_openai_output_item(
                            index,
                            ProviderOutputItemKind::Message,
                            Some(item_id.clone()),
                        )? {
                            events.push(Event::ProviderOutputItemStarted {
                                output_index: index,
                                item_id: Some(item_id.clone()),
                                kind: ProviderOutputItemKind::Message,
                                status: None,
                                phase,
                            });
                        }
                        events.push(Event::ProviderOutputItemCompleted {
                            output_index: index,
                            item_id: Some(item_id),
                            kind: ProviderOutputItemKind::Message,
                            status,
                            phase,
                        });
                    }
                    Some("function_call") => {
                        let item_id =
                            optional_openai_identity(item, "id", "OpenAI function call item ID")?;
                        self.bind_openai_output_item(
                            index,
                            ProviderOutputItemKind::FunctionCall,
                            item_id.clone(),
                        )?;
                        let call_id = openai_identity(item, "call_id", "OpenAI function call ID")?;
                        let name = openai_identity(item, "name", "OpenAI function call name")?;
                        let namespace = optional_openai_identity(
                            item,
                            "namespace",
                            "OpenAI function call namespace",
                        )?;
                        let caller = optional_openai_caller(item, "OpenAI function call caller")?;
                        let arguments =
                            require_string(item, "arguments", "OpenAI function call arguments")
                                .map_err(|message| malformed(&message))?;
                        let tool = self.tools.get(&index).ok_or_else(|| {
                            malformed("OpenAI function call completed before its start")
                        })?;
                        if tool.provider_item_id != item_id
                            || tool.call_id != call_id
                            || tool.name != name
                            || tool.namespace != namespace
                            || tool.caller != caller
                        {
                            return Err(malformed(
                                "OpenAI function call changed identity at completion",
                            ));
                        }
                        self.reconcile_openai_tool_arguments(index, &arguments, &mut events)?;
                        self.tools
                            .get_mut(&index)
                            .expect("tool just checked")
                            .provider_status = status;
                        events.push(Event::ProviderOutputItemCompleted {
                            output_index: index,
                            item_id,
                            kind: ProviderOutputItemKind::FunctionCall,
                            status,
                            phase: None,
                        });
                        let tool = self.tools.get_mut(&index).expect("tool just checked");
                        if status == Some(ProviderOutputItemStatus::Incomplete) {
                            // The provider itself marked this call truncated
                            // by the output budget: gpt-6-astra ends the item
                            // with status "incomplete" and the partial
                            // argument bytes (captured live 2026-09-05), so a
                            // mid-fragment call is the caller's budget to
                            // raise, never a malformed stream.
                            complete_streamed_tool_truncated(index, tool, &mut events)?;
                        } else {
                            complete_streamed_tool(index, tool, &mut events)?;
                        }
                    }
                    Some("custom_tool_call") => {
                        let item_id = optional_openai_identity(
                            item,
                            "id",
                            "OpenAI custom tool call item ID",
                        )?;
                        self.bind_openai_output_item(
                            index,
                            ProviderOutputItemKind::CustomToolCall,
                            item_id.clone(),
                        )?;
                        let call_id =
                            openai_identity(item, "call_id", "OpenAI custom tool call ID")?;
                        let name = openai_identity(item, "name", "OpenAI custom tool call name")?;
                        let namespace = optional_openai_identity(
                            item,
                            "namespace",
                            "OpenAI custom tool call namespace",
                        )?;
                        let caller =
                            optional_openai_caller(item, "OpenAI custom tool call caller")?;
                        let input = require_string(item, "input", "OpenAI custom tool call input")
                            .map_err(|message| malformed(&message))?;
                        let tool = self.tools.get(&index).ok_or_else(|| {
                            malformed("OpenAI custom tool call completed before its start")
                        })?;
                        if !tool.custom
                            || tool.provider_item_id != item_id
                            || tool.call_id != call_id
                            || tool.name != name
                            || tool.namespace != namespace
                            || tool.caller != caller
                        {
                            return Err(malformed(
                                "OpenAI custom tool call changed identity at completion",
                            ));
                        }
                        self.reconcile_openai_tool_arguments(index, &input, &mut events)?;
                        self.tools
                            .get_mut(&index)
                            .expect("tool just checked")
                            .provider_status = status;
                        events.push(Event::ProviderOutputItemCompleted {
                            output_index: index,
                            item_id,
                            kind: ProviderOutputItemKind::CustomToolCall,
                            status,
                            phase: None,
                        });
                        let tool = self.tools.get_mut(&index).expect("tool just checked");
                        complete_streamed_tool(index, tool, &mut events)?;
                    }
                    _ => {
                        return Err(malformed(&format!(
                            "OpenAI stream emitted an unsupported completed output item (type {})",
                            bounded_wire_token(done_type),
                        )))
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                let response = payload
                    .get("response")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("OpenAI terminal response must be an object"))?;
                let is_incomplete = event_type == "response.incomplete"
                    || response.get("status").and_then(Value::as_str) == Some("incomplete");
                let terminal_item_status = if is_incomplete {
                    ProviderOutputItemStatus::Incomplete
                } else {
                    ProviderOutputItemStatus::Completed
                };
                let unfinished: Vec<_> = self
                    .openai_output_items
                    .iter()
                    .filter(|(index, _)| !self.openai_completed_output_items.contains(index))
                    .map(|(index, (kind, item_id))| (*index, *kind, item_id.clone()))
                    .collect();
                for (output_index, kind, item_id) in unfinished {
                    if kind == ProviderOutputItemKind::FunctionCall {
                        if let Some(tool) = self.tools.get_mut(&output_index) {
                            tool.provider_status = Some(terminal_item_status);
                        }
                    }
                    self.openai_completed_output_items.insert(output_index);
                    events.push(Event::ProviderOutputItemCompleted {
                        output_index,
                        item_id,
                        kind,
                        status: Some(terminal_item_status),
                        phase: None,
                    });
                }
                events.extend(self.openai_sweep_hosted_items());
                // Every incomplete terminal is the provider declaring it cut
                // the output early, so a call still open mid-fragment is
                // legitimately partial for ANY of its reasons: a budget cut
                // surfaces Incomplete (the Chat lane's finish_reason=length
                // contract), and a content_filter/safety or unknown-reason
                // cut still reaches its own refusal or provider_internal
                // terminal below instead of being pre-empted here as a
                // malformed 502 (which would misclassify a refusal and churn
                // the ladder). Only a COMPLETED response keeps the strict
                // argument contract, so corruption in a served answer stays
                // fail-closed.
                events.extend(if is_incomplete {
                    finish_open_tools_truncated(&mut self.tools)?
                } else {
                    finish_open_tools(&mut self.tools)?
                });
                if let Some(usage) =
                    openai_usage(response.get("usage")).map_err(|message| malformed(&message))?
                {
                    events.push(Event::Usage(usage));
                }
                if !is_incomplete {
                    if self.refusal_seen {
                        events.push(Event::Failed(refusal_failure()));
                    } else {
                        events.push(Event::Completed);
                    }
                } else {
                    let details = response
                        .get("incomplete_details")
                        .and_then(Value::as_object)
                        .ok_or_else(|| malformed("OpenAI incomplete details must be an object"))?;
                    let reason = details
                        .get("reason")
                        .and_then(Value::as_str)
                        .ok_or_else(|| malformed("OpenAI incomplete reason must be text"))?;
                    if reason == "max_output_tokens" {
                        events.push(Event::Incomplete);
                    } else if reason == "content_filter" || reason == "safety" {
                        // The incomplete reason IS the category token.
                        events.push(Event::Failed(Failure::refusal(
                            crate::stream_errors::refusal_reason(Some(reason), None),
                        )));
                    } else {
                        // The undocumented reason is the only diagnostic this
                        // terminal carries; it rides as the bounded detail.
                        events.push(Event::Failed(
                            Failure::new(
                                FailureClass::ProviderInternal,
                                "provider ended the stream incompletely",
                            )
                            .with_retry(true, true)
                            .with_provider_detail(provider_error_detail(Some(reason), None, &[])),
                        ));
                    }
                }
            }
            "response.failed" => {
                // A failed terminal still reports the usage the provider
                // billed for the processed legs; fold it so settlement
                // accounts the charged tokens instead of zero. Its error
                // object names the mechanism, so a bounded detail rides the
                // failure into the ledger (2026-09-05: gpt-6-astra kills
                // ~120ms post-dispatch were undiagnosable without it).
                let mut code = None;
                let mut message = None;
                if let Some(response) = payload.get("response").and_then(Value::as_object) {
                    if let Some(usage) = openai_usage(response.get("usage"))
                        .map_err(|message| malformed(&message))?
                    {
                        events.push(Event::Usage(usage));
                    }
                    if let Some(error) = response.get("error").and_then(Value::as_object) {
                        code = error.get("code").and_then(error_code_text);
                        message = error.get("message").and_then(Value::as_str);
                    }
                }
                events.push(Event::Failed(self.provider_stream_failure(
                    "openai_responses",
                    code.as_deref(),
                    message,
                )));
            }
            "error" => {
                // The in-stream error frame is the provider declaring its own
                // failure mid-stream, mirroring the Anthropic dialect's error
                // event, never a stream that "ended without a terminal event".
                // Its code and message classify the failure (a throttle, the
                // caller's input, or the provider) and ride it as the bounded
                // ledger detail. The documented shape puts code/message at the
                // top level; a nested `error` object is read as the fallback
                // so an undocumented envelope never degrades to an opaque
                // "provider stream failed".
                let nested = payload.get("error").and_then(Value::as_object);
                // A code is a string in the documented shape; an aggregator
                // may send the upstream status as a number, which is just as
                // classifiable (429, 400) and must not be dropped.
                let code = payload
                    .get("code")
                    .and_then(error_code_text)
                    .or_else(|| {
                        nested
                            .and_then(|error| error.get("code"))
                            .and_then(error_code_text)
                    })
                    .or_else(|| {
                        nested
                            .and_then(|error| error.get("type"))
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    });
                let message = payload.get("message").and_then(Value::as_str).or_else(|| {
                    nested
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                });
                events.push(Event::Failed(self.provider_stream_failure(
                    "openai_responses",
                    code.as_deref(),
                    message,
                )));
            }
            other if is_openai_hosted_progress_event(other) => {
                events.extend(self.openai_hosted_progress(other, &payload)?);
            }
            _ => {}
        }
        Ok(events)
    }
}

/// One provider error code as text: a string as-is, a numeric status (an
/// aggregator relaying the upstream 429/400) rendered so it still classifies.
fn error_code_text(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_string)
        .or_else(|| value.as_i64().map(|numeric| numeric.to_string()))
}

mod compatible;

#[cfg(test)]
#[path = "openai/normalizer_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "openai/identity_tests.rs"]
mod identity_tests;

#[cfg(test)]
#[path = "openai/hosted_tests.rs"]
mod hosted_tests;

#[cfg(test)]
#[path = "openai/truncation_tests.rs"]
mod truncation_tests;
