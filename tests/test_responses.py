"""Native stream reconstruction preserves provider data without DSPy types."""

from copy import deepcopy

import pytest

from openai_codex_auth import CodexProtocolError, CodexResponse
from openai_codex_auth.responses import ResponseBuilder, has_output


def completed(output=None, **metadata):
    response = {
        "id": "resp_test",
        "model": "gpt-test",
        "status": "completed",
        **metadata,
    }
    if output is not None:
        response["output"] = output
    return {"type": "response.completed", "response": response}


def text_event(index, kind, text, content_index=0):
    return {
        "type": f"response.output_text.{kind}",
        "output_index": index,
        "content_index": content_index,
        "text" if kind == "done" else "delta": text,
    }


def test_done_and_delta_are_merged_per_output_and_content_index():
    builder = ResponseBuilder()
    for event in [
        text_event(0, "delta", "discarded"),
        text_event(0, "done", "first"),
        text_event(0, "delta", "third", content_index=1),
        text_event(2, "delta", "sec"),
        text_event(2, "delta", "ond"),
        completed([]),
    ]:
        builder.add(event)
    result = CodexResponse(builder.build())
    assert result.output_text == "firstthirdsecond"
    assert len(result.output) == 2
    assert result.model == "gpt-test"
    assert result.usage == {}


def test_full_final_response_and_metadata_survive_unchanged():
    output = [
        {
            "type": "reasoning",
            "id": "rs1",
            "encrypted_content": "opaque",
            "summary": [{"type": "summary_text", "text": "think", "future": 7}],
        },
        {
            "type": "function_call",
            "id": "fc1",
            "name": "weather",
            "call_id": "call1",
            "arguments": '{"city":"Austin"}',
            "status": "completed",
            "provider_extension": {"flag": True},
        },
        {
            "type": "message",
            "id": "msg1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "answer",
                    "annotations": [
                        {"type": "url_citation", "url": "https://example.com"}
                    ],
                    "logprobs": [{"token": "answer"}],
                }
            ],
        },
        {
            "type": "web_search_call",
            "id": "search1",
            "status": "completed",
            "action": {"type": "search", "query": "news"},
        },
    ]
    event = completed(
        output,
        usage={"input_tokens": 9, "output_tokens_details": {"reasoning_tokens": 2}},
        created_at=123,
        service_tier="priority",
        metadata={"a": "b"},
        future_field={"nested": [1, 2]},
    )
    original = deepcopy(event)
    builder = ResponseBuilder()
    builder.add(event)
    response = builder.build()
    assert response == event["response"]
    assert event == original
    assert CodexResponse(response).output_text == "answer"
    assert has_output(response)


def test_item_skeleton_deltas_and_final_partial_response_merge():
    builder = ResponseBuilder()
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg1",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
                "provider_added": True,
            },
        },
        text_event(0, "delta", "hello"),
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "reasoning",
                "id": "rs1",
                "encrypted_content": "opaque",
                "summary": [{"type": "summary_text", "text": "why"}],
            },
        },
        completed([{"type": "reasoning", "id": "rs1", "status": "completed"}]),
    ]
    for event in events:
        builder.add(event)
    output = builder.build()["output"]
    assert output[0]["content"][0]["text"] == "hello"
    assert output[0]["content"][0]["annotations"] == []
    assert output[0]["provider_added"] is True
    assert output[1] == {
        "type": "reasoning",
        "id": "rs1",
        "encrypted_content": "opaque",
        "summary": [{"type": "summary_text", "text": "why"}],
        "status": "completed",
    }


def test_reasoning_and_function_arguments_reconstruct_without_losing_fields():
    builder = ResponseBuilder()
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc1",
                "name": "weather",
                "call_id": "call1",
                "arguments": "",
                "future": 3,
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc1",
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc1",
            "arguments": '{"city":"Austin"}',
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 1,
            "summary_index": 0,
            "item_id": "rs1",
            "delta": "discarded",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "output_index": 1,
            "summary_index": 0,
            "text": "first",
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 1,
            "summary_index": 1,
            "delta": "second",
        },
        completed(),
    ]
    for event in events:
        builder.add(event)
    output = builder.build()["output"]
    assert output[0]["arguments"] == '{"city":"Austin"}'
    assert output[0]["call_id"] == "call1"
    assert output[0]["future"] == 3
    assert output[1]["id"] == "rs1"
    assert [part["text"] for part in output[1]["summary"]] == ["first", "second"]


def test_refusal_and_native_tool_output_count_as_final_output():
    builder = ResponseBuilder()
    builder.add(
        {
            "type": "response.refusal.done",
            "output_index": 0,
            "content_index": 0,
            "refusal": "I cannot comply",
        }
    )
    builder.add(completed())
    response = builder.build()
    assert has_output(response)
    assert CodexResponse(response).output_text == ""
    assert has_output({"output": [{"type": "web_search_call"}]})
    assert not has_output(
        {"output": [{"type": "reasoning", "summary": [{"text": "thinking"}]}]}
    )
    assert not has_output({"output": [{"type": "message", "content": [{"text": ""}]}]})


@pytest.mark.parametrize(
    "event",
    [
        [],
        {},
        {"type": 1},
        {"type": ""},
        {"type": "response.completed", "response": []},
        completed(status="failed"),
        completed(model=""),
        completed(output={}),
        completed(output=["text"]),
        completed(output=[{}]),
        completed(output=[{"type": "message", "content": "bad"}]),
        completed(output=[{"type": "message", "id": []}]),
        completed(usage=[]),
        {"type": "response.failed"},
        {"type": "response.incomplete"},
        {"type": "error"},
        {
            "type": "response.output_text.delta",
            "output_index": -1,
            "content_index": 0,
            "delta": "bad",
        },
        {
            "type": "response.output_text.delta",
            "output_index": True,
            "content_index": 0,
            "delta": "bad",
        },
        {"type": "response.output_text.delta", "output_index": 0, "delta": "bad"},
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": {"text": "bad"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": [],
        },
    ],
)
def test_malformed_native_events_fail_explicitly(event):
    with pytest.raises(CodexProtocolError):
        ResponseBuilder().add(event)


def test_stream_end_without_completed_response_fails():
    builder = ResponseBuilder()
    builder.add(text_event(0, "done", "partial"))
    with pytest.raises(CodexProtocolError, match="before response.completed"):
        builder.build()


@pytest.mark.parametrize(
    "part_event,text_event_prefix,index_name,item_type,container",
    [
        (
            "response.content_part",
            "response.output_text",
            "content_index",
            "message",
            "content",
        ),
        (
            "response.reasoning_summary_part",
            "response.reasoning_summary_text",
            "summary_index",
            "reasoning",
            "summary",
        ),
    ],
)
def test_done_parts_outrank_text_fragments(
    part_event, text_event_prefix, index_name, item_type, container
):
    builder = ResponseBuilder()
    for event in [
        {
            "type": f"{part_event}.added",
            "output_index": 0,
            index_name: 0,
            "part": {
                "type": "output_text" if item_type == "message" else "summary_text",
                "text": "",
            },
        },
        {
            "type": f"{text_event_prefix}.delta",
            "output_index": 0,
            index_name: 0,
            "delta": "partial",
        },
        {
            "type": f"{part_event}.done",
            "output_index": 0,
            index_name: 0,
            "part": {
                "type": "output_text" if item_type == "message" else "summary_text",
                "text": "complete answer",
                "metadata": {"keep": True},
            },
        },
        completed(),
    ]:
        builder.add(event)
    part = builder.build()["output"][0][container][0]
    assert part["text"] == "complete answer"
    assert part["metadata"] == {"keep": True}
