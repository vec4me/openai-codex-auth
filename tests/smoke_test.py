"""Check an installed core wheel without test-only dependencies or real requests."""

import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx

import openai_codex_auth


def main() -> None:
    for framework in ("dspy", "litellm"):
        assert importlib.util.find_spec(framework) is None, (
            f"Run the core smoke test in an isolated install without {framework}"
        )
    assert openai_codex_auth.DEFAULT_CODEX_API_BASE.startswith("https://chatgpt.com/")
    assert openai_codex_auth.DEFAULT_AUTH_PATH.name == "auth.json"
    assert callable(openai_codex_auth.getauthtoken)
    assert openai_codex_auth.codex_headers(
        "synthetic-token", account_id="acct_smoke"
    )["chatgpt-account-id"] == "acct_smoke"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            openai_codex_auth.CodexAuth(Path(tmp) / "missing.json").token()
        except openai_codex_auth.CodexAuthError as exc:
            assert "codex login" in str(exc)
        else:
            raise AssertionError("missing credential must raise CodexAuthError")

    native_input = [
        {"role": "user", "content": [{"type": "input_text", "text": "Reply with OK."}]}
    ]
    response = {
        "id": "resp_smoke",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": "completed",
        "metadata": {"smoke": "preserved"},
        "output": [
            {
                "type": "message",
                "id": "msg_smoke",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    request_count = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.method == "POST"
        assert str(request.url) == "https://example.invalid/codex/responses"
        assert request.headers["authorization"] == "Bearer synthetic-token"
        assert request.headers["chatgpt-account-id"] == "acct_smoke"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-test"
        assert payload["input"] == native_input
        assert payload["instructions"] == "Be concise."
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["store"] is False
        assert payload["stream"] is True
        assert "synthetic-token" not in request.content.decode()
        event = {"type": "response.completed", "response": response}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    sync_client = httpx.Client
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handle_request)

    def mock_sync_client(*args, **kwargs):
        return sync_client(*args, **kwargs, transport=transport)

    def mock_async_client(*args, **kwargs):
        return async_client(*args, **kwargs, transport=transport)

    client = openai_codex_auth.CodexClient(
        api_base="https://example.invalid/codex",
        api_key="synthetic-token",
        account_id="acct_smoke",
    )
    parameters = {
        "model": "gpt-test",
        "input": native_input,
        "instructions": "Be concise.",
        "reasoning": {"effort": "low"},
        "max_retries": 0,
    }
    with (
        patch.object(httpx, "Client", mock_sync_client),
        patch.object(httpx, "AsyncClient", mock_async_client),
    ):
        results = [client.create(**parameters), asyncio.run(client.acreate(**parameters))]

    assert request_count == 2
    for result in results:
        assert isinstance(result, openai_codex_auth.CodexResponse)
        assert result.model == "gpt-test"
        assert result.output_text == "OK"
        assert result.output == response["output"]
        assert result.usage == response["usage"]
        assert result.response["metadata"] == {"smoke": "preserved"}
    print("Core distribution smoke passed: sync/async HTTP, no DSPy or LiteLLM.")


if __name__ == "__main__":
    main()
