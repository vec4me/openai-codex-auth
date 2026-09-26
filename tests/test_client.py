"""Codex client integration tests using a synthetic HTTP backend."""

import asyncio
import json
import subprocess
import sys
from copy import deepcopy

import httpx
import pytest

from openai_codex_auth import (
    CodexAuth,
    CodexClient,
    CodexEmptyResponseError,
    CodexError,
    CodexHTTPError,
    CodexProtocolError,
    responses_http,
)
from openai_codex_auth import client as client_module

TOKEN = "synthetic-secret-token"
MODEL = "gpt-test"


def final_response(text="answer"):
    return {
        "id": "resp1",
        "model": MODEL,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        "service_tier": "priority",
        "metadata": {"test": "native"},
    }


def sse_response(response=None, events=None):
    events = list(events or [])
    events.append(
        {
            "type": "response.completed",
            "response": response if response is not None else final_response(),
        }
    )
    content = ": keepalive\n\n" + "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        content=content,
    )


def mock_http(monkeypatch, handler):
    sync_client = httpx.Client
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        responses_http.httpx,
        "Client",
        lambda **kwargs: sync_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        responses_http.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )


def invoke(client, asynchronous=False, **kwargs):
    options = {"model": MODEL, "input": "hello", **kwargs}
    if asynchronous:
        return asyncio.run(client.acreate(**options))
    return client.create(**options)


@pytest.fixture(autouse=True)
def no_retry_delays(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    async def noop(seconds):
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", noop)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_http_request_and_response_preserve_fields(monkeypatch, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return sse_response()

    mock_http(monkeypatch, handler)
    client = CodexClient(
        api_key=TOKEN,
        account_id="acct1",
        api_base="https://example.com/codex?route=a",
        user_agent="test-client",
        headers={
            "X-Custom": "custom",
            "authorization": "wrong",
            "USER-AGENT": "wrong",
            "ChatGPT-Account-ID": "wrong",
        },
    )
    tools = [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]
    original_tools = deepcopy(tools)
    result = invoke(
        client,
        asynchronous,
        timeout=17.0,
        connect_timeout=4.0,
        max_retries=0,
        service_tier="FAST",
        max_output_tokens=9,
        reasoning={"effort": "low", "summary": "auto"},
        tools=tools,
    )
    assert result.response == final_response()
    assert result.output_text == "answer"
    assert result.usage["total_tokens"] == 7
    assert tools == original_tools
    request = requests[0]
    assert str(request.url) == "https://example.com/codex/responses?route=a"
    body = json.loads(request.content)
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert body["model"] == MODEL
    assert body["instructions"] == "You are a helpful assistant."
    assert body["stream"] is True and body["store"] is False
    assert body["service_tier"] == "priority"
    assert "max_output_tokens" not in body
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert body["tools"] == tools
    assert TOKEN not in request.content.decode()
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["user-agent"] == "test-client"
    assert request.headers["chatgpt-account-id"] == "acct1"
    assert request.headers["x-custom"] == "custom"
    assert request.extensions["timeout"] == {
        "connect": 4.0,
        "read": 17.0,
        "write": 17.0,
        "pool": 17.0,
    }


@pytest.mark.parametrize("asynchronous", [False, True])
def test_none_instructions_are_omitted(monkeypatch, asynchronous):
    seen = []
    mock_http(monkeypatch, lambda request: seen.append(request) or sse_response())

    invoke(
        CodexClient(api_key=TOKEN, account_id="acct1"),
        asynchronous,
        instructions=None,
        max_retries=0,
    )

    assert "instructions" not in json.loads(seen[0].content)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_list_input_and_httpx_timeout_are_preserved(monkeypatch, asynchronous):
    seen = []
    mock_http(monkeypatch, lambda request: seen.append(request) or sse_response())
    input = [{"type": "function_call_output", "call_id": "call1", "output": "sunny"}]
    timeout = httpx.Timeout(connect=2, read=None, write=3, pool=4)
    invoke(
        CodexClient(api_key=TOKEN, account_id="acct1"),
        asynchronous,
        input=input,
        timeout=timeout,
        max_retries=0,
    )
    assert json.loads(seen[0].content)["input"] == input
    assert seen[0].extensions["timeout"] == timeout.as_dict()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_status_reads_auth_for_every_attempt(monkeypatch, asynchronous, status):
    class Auth(CodexAuth):
        def __init__(self):
            self.reads = 0

        def token(self):
            self.reads += 1
            return f"synthetic-token-{self.reads}"

        def account_id(self):
            return "acct1"

    auth = Auth()
    seen = []

    def handler(request):
        seen.append(request)
        return (
            httpx.Response(status, json={"error": {"message": "busy"}})
            if len(seen) < 3
            else sse_response()
        )

    mock_http(monkeypatch, handler)
    result = invoke(CodexClient(auth), asynchronous, max_retries=2)
    assert result.output_text == "answer"
    assert auth.reads == 3
    assert [request.headers["authorization"] for request in seen] == [
        f"Bearer synthetic-token-{index}" for index in (1, 2, 3)
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retries_network_and_empty_completions(monkeypatch, asynchronous):
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadError("temporary", request=request)
        if len(seen) == 2:
            return sse_response(
                {
                    "status": "completed",
                    "model": MODEL,
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "thinking"}],
                        }
                    ],
                }
            )
        return sse_response()

    mock_http(monkeypatch, handler)
    result = invoke(
        CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, max_retries=2
    )
    assert result.output_text == "answer"
    assert len(seen) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exhausted_network_error_has_no_secret_or_exception_chain(
    monkeypatch, asynchronous
):
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ConnectError(f"failed with {TOKEN}", request=request)

    mock_http(monkeypatch, handler)
    with pytest.raises(CodexError) as raised:
        invoke(
            CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, max_retries=1
        )
    assert len(seen) == 2
    assert TOKEN not in str(raised.value)
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_empty_response_error_omits_provider_content(monkeypatch, asynchronous):
    seen = []
    mock_http(
        monkeypatch,
        lambda request: (
            seen.append(request)
            or sse_response(
                {
                    "model": MODEL,
                    "status": "completed",
                    "output": [],
                    "metadata": {"private": "omit-me"},
                }
            )
        ),
    )
    with pytest.raises(CodexEmptyResponseError, match="2 attempt") as raised:
        invoke(
            CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, max_retries=1
        )
    assert len(seen) == 2
    assert "omit-me" not in str(raised.value)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_http_errors_parse_redact_body_and_do_not_retry_bad_requests(
    monkeypatch, asynchronous
):
    seen = []
    body = {
        "error": {
            "message": f"invalid {TOKEN}",
            "details": {TOKEN: [TOKEN]},
            "type": "invalid_request_error",
        }
    }
    mock_http(
        monkeypatch,
        lambda request: seen.append(request) or httpx.Response(400, json=body),
    )
    with pytest.raises(CodexHTTPError) as raised:
        invoke(CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous)
    assert len(seen) == 1
    assert TOKEN not in str(raised.value)
    assert TOKEN not in json.dumps(raised.value.body)
    assert raised.value.error["type"] == "invalid_request_error"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "content,content_type",
    [
        ("data: not-json\n\n", "text/event-stream"),
        ("data: []\n\n", "text/event-stream"),
        ("data: {}\n\n", "text/event-stream"),
        ("data: [DONE]\n\n", "text/event-stream"),
        ('data: {"type":"response.failed"}\n\n', "text/event-stream"),
        ('data: {"type":"response.incomplete"}\n\n', "text/event-stream"),
        ('event: wrong\ndata: {"type":"response.created"}\n\n', "text/event-stream"),
        (
            f'data: {{"type":"response.output_text.delta","delta":"{TOKEN}"}}\n\n',
            "text/event-stream",
        ),
        (f'data: {{"type":"{TOKEN}"', "text/event-stream"),
        ("{}", "application/json"),
    ],
)
def test_malformed_or_incomplete_sse_does_not_retry(
    monkeypatch, asynchronous, content, content_type
):
    seen = []
    mock_http(
        monkeypatch,
        lambda request: (
            seen.append(request)
            or httpx.Response(
                200, headers={"content-type": content_type}, content=content
            )
        ),
    )
    with pytest.raises(CodexProtocolError) as raised:
        invoke(CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous)
    assert len(seen) == 1
    assert TOKEN not in str(raised.value)
    assert raised.value.__context__ is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_multiline_sse_and_unterminated_final_event(monkeypatch, asynchronous):
    event = {"type": "response.completed", "response": final_response()}
    pretty_json = json.dumps(event, indent=2)
    content = "id: event-id\nretry: 500\n: heartbeat\n" + "\n".join(
        f"data: {line}" for line in pretty_json.splitlines()
    )
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=content
        ),
    )
    result = invoke(
        CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, max_retries=0
    )
    assert result.output_text == "answer"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_valid_sse_without_content_type_header(monkeypatch, asynchronous):
    response = sse_response()
    del response.headers["content-type"]
    mock_http(monkeypatch, lambda request: response)
    result = invoke(
        CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, max_retries=0
    )
    assert result.output_text == "answer"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "parameters", [{"metadata": {"debug": (TOKEN,)}}, {"metadata": {1: TOKEN}}]
)
def test_python_only_json_shapes_never_reach_transport(
    monkeypatch, asynchronous, parameters
):
    mock_http(monkeypatch, lambda request: pytest.fail("invalid JSON reached network"))
    with pytest.raises(ValueError, match="native JSON objects and lists"):
        invoke(
            CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, **parameters
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "openai/gpt-test"},
        {"model": ""},
        {"model": "gpt-test "},
        {"input": {}},
        {"input": ["hello"]},
        {"timeout": False},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": httpx.Timeout(-1)},
        {"connect_timeout": -1},
        {"transport": "http"},
        {"idle_timeout": 300},
        {"max_retries": True},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"stream": False},
        {"store": True},
        {"max_tokens": 4},
        {"max_completion_tokens": 5},
        {"messages": []},
        {"reasoning_effort": "low"},
        {"response_format": {}},
        {"num_retries": 2},
        {"api_key": "in-body"},
        {"cache": False},
        {"rollout_id": 1},
        {"temperature": float("nan")},
        {"input": TOKEN},
    ],
)
def test_invalid_controls_and_framework_contracts_fail_before_network(
    monkeypatch, overrides
):
    mock_http(
        monkeypatch, lambda request: pytest.fail("invalid request reached network")
    )
    with pytest.raises(ValueError):
        invoke(CodexClient(api_key=TOKEN, account_id="acct1"), **overrides)


def test_standalone_import_does_not_import_dspy_or_litellm():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from openai_codex_auth import CodexClient; assert 'dspy' not in sys.modules; assert 'litellm' not in sys.modules; print('standalone')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "standalone"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retry_delays_double_to_documented_cap(monkeypatch, asynchronous):
    delays = []
    seen = []

    async def asleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(client_module.time, "sleep", delays.append)
    monkeypatch.setattr(client_module.asyncio, "sleep", asleep)
    mock_http(
        monkeypatch,
        lambda request: (
            seen.append(request)
            or httpx.Response(503, json={"error": {"message": "busy"}})
        ),
    )
    with pytest.raises(CodexHTTPError):
        invoke(
            CodexClient(api_key=TOKEN, account_id="acct1"), asynchronous, max_retries=7
        )
    assert len(seen) == 8
    assert delays == [0.5, 1, 2, 4, 8, 8, 8]
