//! Inline tests for `encode_messages`, split into a submodule file so the
//! implementation stays within the repository line budget.

use super::*;
use crate::events::CompletedToolCall;

#[test]
fn usage_reports_cached_reads_out_of_the_input_total() {
    let usage = Usage {
        input_tokens: Some(10),
        output_tokens: Some(4),
        cached_input_tokens: Some(3),
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({"input_tokens": 7, "output_tokens": 4, "cache_read_input_tokens": 3})
    );
    assert_eq!(
        messages_usage(None),
        json!({"input_tokens": 0, "output_tokens": 0})
    );
}

#[test]
fn error_body_folds_param_and_maps_status_first() {
    let mut error = PublicError::new(
        400,
        "invalid_parameter",
        "Invalid value.",
        "invalid_request_error",
    );
    error.param = Some("top_k".to_string());
    assert_eq!(
        anthropic_error_body(&error),
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid value. (param: top_k)",
            },
        })
    );
    let throttled = PublicError::new(429, "unavailable_route", "Throttled.", "api_error");
    assert_eq!(
        anthropic_error_body(&throttled)["error"]["type"],
        json!("rate_limit_error")
    );
}

#[test]
fn a_text_less_refusal_is_an_invalid_request_error_on_the_messages_surface() {
    // The Messages envelope maps by status first, so the refusal's 400 lands
    // on Anthropic's own invalid_request_error type instead of the api_error
    // a 502 routing failure would have produced.
    let refused = Failure::new(FailureClass::Refusal, "provider refused the request");
    assert_eq!(
        anthropic_error_body(&refused.public_error()),
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "provider refused the request",
                // A refusal with no named reason still carries the category,
                // as `unspecified`, on the Anthropic envelope.
                "refusal_reason": "unspecified",
            },
        })
    );
}

#[test]
fn completed_body_orders_text_before_tool_use_blocks() {
    let events = vec![
        Event::TextDelta("hi".to_string()),
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{\"b\":1,\"a\":2}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"b\":1,\"a\":2}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert!(aggregated.failure.is_none());
    assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
    assert_eq!(aggregated.body["content"][0]["type"], json!("text"));
    // preserve_order keeps the provider's key order in the parsed input.
    assert_eq!(
        compact_json(&aggregated.body["content"][1]["input"]),
        "{\"b\":1,\"a\":2}"
    );
    assert_eq!(aggregated.tool_names, vec!["search".to_string()]);
}

#[test]
fn completed_body_preserves_interleaved_block_order() {
    let events = vec![
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::TextDelta("after ".to_string()),
        Event::TextDelta("the tool".to_string()),
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["type"], json!("tool_use"));
    assert_eq!(
        aggregated.body["content"][1],
        json!({"type": "text", "text": "after the tool"})
    );
}

#[test]
fn deferred_tool_completion_keeps_the_started_block_position() {
    // OpenAI-compatible streams complete every tool only at [DONE], so
    // text may arrive between the tool's arguments and its completion.
    let events = vec![
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::TextDelta("after".to_string()),
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["type"], json!("tool_use"));
    assert_eq!(
        aggregated.body["content"][1],
        json!({"type": "text", "text": "after"})
    );
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(
            encoder
                .feed(event)
                .expect("streams the deferred completion"),
        );
    }
    assert!(frames.last().expect("terminal").contains("message_stop"));
}

#[test]
fn interleaved_parallel_tools_stream_strictly_sequential_blocks() {
    // Tool A streams live through the interleaving; tool B's fragment
    // buffers and flushes as one delta after A's block closes.
    let events = vec![
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-a".to_string(),
            name: "alpha".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{\"a\": ".to_string(),
        },
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 1,
            call_id: "call-b".to_string(),
            name: "beta".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 1,
            delta: "{\"b\": 2}".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "1}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-a".to_string(),
                name: "alpha".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"a\": 1}".to_string(),
                custom: false,
            },
        },
        Event::ToolCallCompleted {
            index: 1,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-b".to_string(),
                name: "beta".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"b\": 2}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams the interleaving"));
    }
    let names: Vec<&str> = frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("named frame")
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // The buffered tool-B fragment flushes as one input_json_delta on
    // Anthropic block index 1 after block 0 closes.
    assert!(frames[7].contains("\"index\":1"));
    assert!(frames[7].contains("{\\\"b\\\": 2}"));
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["id"], json!("call-a"));
    assert_eq!(aggregated.body["content"][1]["id"], json!("call-b"));
}

#[test]
fn refusal_content_aggregates_as_a_sanitized_failure() {
    let events = vec![Event::RefusalDelta("no".to_string()), Event::Completed];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    let failure = aggregated.failure.expect("refusal failure");
    assert_eq!(failure.failure_class, FailureClass::Refusal);
}

#[test]
fn thinking_blocks_stream_in_valid_anthropic_order() {
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "step ".to_string(),
        },
        Event::ThinkingDelta {
            index: 0,
            delta: "one".to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: "sig==".to_string(),
        },
        Event::RedactedThinking {
            index: 1,
            data: "opaque==".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams thinking"));
    }
    let names: Vec<&str> = frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("named frame")
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // The thinking block opens with the SDK-required empty fields, streams
    // its text, and closes with one signature_delta before its stop.
    assert!(frames[2].contains("{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}"));
    assert!(frames[3].contains("{\"type\":\"thinking_delta\",\"thinking\":\"step \"}"));
    assert!(frames[5].contains("{\"type\":\"signature_delta\",\"signature\":\"sig==\"}"));
    assert!(frames[7].contains("{\"type\":\"redacted_thinking\",\"data\":\"opaque==\"}"));
    assert!(frames[10].contains("{\"type\":\"text_delta\",\"text\":\"answer\"}"));
}

#[test]
fn completed_body_carries_thinking_blocks_verbatim_and_in_order() {
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "step one".to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: "sig==".to_string(),
        },
        Event::RedactedThinking {
            index: 1,
            data: "opaque==".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert!(aggregated.failure.is_none());
    assert_eq!(
        aggregated.body["content"],
        json!([
            {"type": "thinking", "thinking": "step one", "signature": "sig=="},
            {"type": "redacted_thinking", "data": "opaque=="},
            {"type": "text", "text": "answer"},
        ])
    );
    assert_eq!(aggregated.body["stop_reason"], json!("end_turn"));
}

#[test]
fn interleaved_thinking_between_tool_blocks_keeps_sequential_indices() {
    // Interleaved thinking may arrive between tool calls; blocks stay
    // strictly sequential in start order.
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "plan".to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: "sig-a".to_string(),
        },
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-1".to_string(),
            name: "search".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams interleaving"));
    }
    assert!(frames.last().expect("terminal").contains("message_stop"));
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["type"], json!("thinking"));
    assert_eq!(aggregated.body["content"][1]["type"], json!("tool_use"));
    assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
}

/// The captured live WebSearch lifecycle (2026-08-31): server tool use with
/// streamed input, one whole verbatim result block, then a cited answer.
fn web_search_events() -> Vec<Event> {
    let result_block = json!({
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{
            "type": "web_search_result",
            "title": "Python versions",
            "url": "https://www.python.org/doc/versions/",
            "encrypted_content": "EtGzBA==",
            "page_age": "March 12, 2026",
        }],
        "caller": {"type": "direct"},
    });
    let citation = json!({
        "type": "web_search_result_location",
        "cited_text": "Python 3.14.7, released on 5 August 2026",
        "url": "https://www.python.org/doc/versions/",
        "title": "Python versions",
        "encrypted_index": "Eo8BCg==",
    });
    vec![
        Event::ServerToolUseStarted {
            index: 0,
            call_id: "srvtoolu_1".to_string(),
            name: "web_search".to_string(),
        },
        Event::ServerToolArgumentsDelta {
            index: 0,
            delta: "{\"query\": \"current stable Python\"}".to_string(),
        },
        Event::ServerToolUseCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "srvtoolu_1".to_string(),
                name: "web_search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{\"query\": \"current stable Python\"}".to_string(),
                custom: false,
            },
        },
        Event::ServerToolResult {
            index: 1,
            block: compact_json(&result_block),
        },
        Event::TextBlockStarted { index: 2 },
        Event::CitationDelta {
            index: 2,
            citation: compact_json(&citation),
        },
        Event::TextDelta("The current stable Python version is 3.14.7.".to_string()),
        Event::Completed,
    ]
}

#[test]
fn server_tool_blocks_stream_in_valid_anthropic_order() {
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &web_search_events() {
        frames.extend(encoder.feed(event).expect("streams server tools"));
    }
    let names: Vec<&str> = frames
        .iter()
        .map(|frame| {
            frame
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("named frame")
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert!(
        frames[2].contains(
            "{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"web_search\",\"input\":{}}"
        ),
        "server tool start frame: {}",
        frames[2]
    );
    assert!(frames[3]
        .contains("{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\": \\\"current stable Python\\\"}\"}"));
    // The whole verbatim result block rides its start frame, caller field
    // included.
    assert!(frames[5].contains("\"type\":\"web_search_tool_result\""));
    assert!(frames[5].contains("\"caller\":{\"type\":\"direct\"}"));
    // Citations flush on the text block before its text, mirroring the
    // provider's own delta order.
    assert!(frames[8].contains("\"type\":\"citations_delta\""));
    assert!(frames[8].contains("\"cited_text\":\"Python 3.14.7, released on 5 August 2026\""));
    assert!(frames[9].contains("\"type\":\"text_delta\""));
    // Server tool use is provider-executed: the turn still ends end_turn.
    assert!(frames[11].contains("\"stop_reason\":\"end_turn\""));
}

#[test]
fn completed_body_carries_server_tool_blocks_and_citations() {
    let aggregated =
        completed_messages_body("request-abc", "coding", &web_search_events()).expect("aggregates");
    assert!(aggregated.failure.is_none());
    assert_eq!(aggregated.body["stop_reason"], json!("end_turn"));
    assert_eq!(aggregated.tool_names, vec!["web_search".to_string()]);
    let content = aggregated.body["content"].as_array().expect("content");
    assert_eq!(content.len(), 3);
    assert_eq!(
        content[0],
        json!({
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "current stable Python"},
        })
    );
    assert_eq!(content[1]["type"], json!("web_search_tool_result"));
    assert_eq!(content[1]["caller"], json!({"type": "direct"}));
    assert_eq!(
        content[2]["text"],
        json!("The current stable Python version is 3.14.7.")
    );
    assert_eq!(
        content[2]["citations"][0]["cited_text"],
        json!("Python 3.14.7, released on 5 August 2026")
    );
}

#[test]
fn paused_turn_keeps_its_stop_reason_on_both_paths() {
    let events = vec![
        Event::ServerToolUseStarted {
            index: 0,
            call_id: "srvtoolu_1".to_string(),
            name: "web_search".to_string(),
        },
        Event::ServerToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ServerToolUseCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "srvtoolu_1".to_string(),
                name: "web_search".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::PausedTurn,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams pause"));
    }
    assert!(encoder.saw_terminal());
    let message_delta = frames
        .iter()
        .find(|frame| frame.starts_with("event: message_delta"))
        .expect("message_delta frame");
    assert!(message_delta.contains("\"stop_reason\":\"pause_turn\""));
    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["stop_reason"], json!("pause_turn"));
    assert!(!aggregated.incomplete);
}

#[test]
fn usage_reports_both_cache_legs_out_of_the_folded_input_total() {
    // A live cached turn folds input as uncached + read + write for the
    // ledger; callers get the provider's own shape back, with each cache
    // leg on its own field (Claude Code displays cache_creation on turn 1).
    let usage = Usage {
        input_tokens: Some(45_543),
        output_tokens: Some(9),
        cached_input_tokens: Some(0),
        cache_creation_input_tokens: Some(45_338),
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({
            "input_tokens": 205,
            "output_tokens": 9,
            "cache_creation_input_tokens": 45_338,
        })
    );
}

#[test]
fn ignored_generation_controls_are_disclosed_by_both_messages_encoders() {
    // A dropped `output_config.effort` on an empty-ladder Claude (or a
    // dropped beta token) must reach the caller on the Messages surface the
    // same way Chat and Responses disclose it: a body-level key on the
    // message object, both on `message_start` and on the aggregated body.
    let ignored = vec![
        "reasoning_effort".to_string(),
        "anthropic-beta.claude-code-20250219".to_string(),
    ];
    let mut stream = MessagesSseEncoder::new_with_ignored("request-abc", "coding", ignored.clone());
    let frames = stream.start().expect("stream start must encode");
    let message_start = frames
        .iter()
        .find(|frame| frame.starts_with("event: message_start"))
        .expect("message_start frame");
    assert!(message_start.contains(
        "\"x-experiential-ignored-parameters\":[\"reasoning_effort\",\"anthropic-beta.claude-code-20250219\"]"
    ));

    let events = vec![Event::TextDelta("hi".to_string()), Event::Completed];
    let aggregated =
        completed_messages_body_with_ignored("request-abc", "coding", &events, &ignored)
            .expect("aggregates");
    assert_eq!(
        aggregated.body["x-experiential-ignored-parameters"],
        json!(["reasoning_effort", "anthropic-beta.claude-code-20250219"])
    );

    // Nothing dropped, nothing disclosed: the plain envelope stays byte-identical.
    let mut plain = MessagesSseEncoder::new("request-abc", "coding");
    assert!(!plain
        .start()
        .expect("plain start must encode")
        .concat()
        .contains("x-experiential-ignored-parameters"));
    let plain_body = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert!(plain_body
        .body
        .get("x-experiential-ignored-parameters")
        .is_none());
}
