# openai-codex-auth

A standalone Python client for the ChatGPT Codex backend. It reads the Codex
CLI's ChatGPT login, refreshes credentials, sends HTTP requests, and
reconstructs completed Responses output from streaming events.

The core has no DSPy or LiteLLM dependency. Its runtime dependencies are
Requests for credential refresh and HTTPX for HTTP transport.
[`dspy-codex-auth`](https://github.com/hrbatra/dspy-codex-auth) imports this
client to provide an optional DSPy adapter.

## Install and login

```bash
uv add openai-codex-auth
codex login
```

Choose "Sign in with ChatGPT". This package requires a file-backed login at
`~/.codex/auth.json`, or an explicit auth-file path. It does not read the OS
credential store. See [Codex authentication](https://learn.chatgpt.com/docs/auth#login-caching).
No Pi installation or plugin is needed.

## Synchronous and asynchronous calls

```python
from openai_codex_auth import CodexClient

client = CodexClient()
response = client.create(
    model="gpt-5.5",
    input="Explain what a Python generator does in one sentence.",
    reasoning={"effort": "low", "summary": "auto"},
)
print(response.output_text)
print(response.usage)
```

Use the same client with `await client.acreate(...)` in asynchronous code:

```python
response = await client.acreate(
    model="gpt-5.5",
    input="What is an async iterator?",
)
```

Use bare model IDs such as `gpt-5.5`. `input` accepts a string, encoded as a user
message, or a list of native Responses input items. Pass instructions through
`instructions=` or set `instructions=None` to omit the request field. Provider
fields such as `reasoning`, `text`, and `tools` pass directly. The client
returns function calls; your application executes tools and supplies results
as input for subsequent calls. It does not provide a tool-execution loop.

`CodexResponse` exposes `output_text`, `output`, `model`, and `usage`. Its
`response` dictionary preserves provider metadata and the reconstructed output,
including reasoning summaries and function calls. Both methods consume the
stream and return the completed response.

## Transport and request behavior
```python
response = client.create(
    model="gpt-5.5",
    input="Hello",
    timeout=300.0,
    connect_timeout=10.0,
    max_retries=3,
)
```

- `timeout` defaults to 300 seconds and `connect_timeout` defaults to 10
  seconds. The client also accepts an `httpx.Timeout` for phase-specific
  deadlines, including explicit `None` values to disable particular deadlines.
- `max_retries=3` allows three additional attempts for recoverable network
  failures, HTTP/backend 429/5xx responses, and completions without meaningful
  output. Set `max_retries=0` to disable retries. Delays start at 0.5 seconds and
  double up to 8 seconds. Malformed or unfinished streams and other backend
  failures raise errors.
- Empty output and reasoning-only output without an answer are retried;
  refusals and other native output items are returned to the caller.
- The backend requires `stream=True` and `store=False`; the client enforces
  both. It removes `max_output_tokens`, which this backend rejects, and maps
  `service_tier="fast"` to `"priority"`. There is no client output-token cap.
- The client reconstructs missing output from streamed text, reasoning, and
  tool-call events. A stream ending before completion raises an error.
- `usage` is an empty dictionary when the provider omits usage; it does not
  imply zero tokens were used.

Native provider input and options are distinct from DSPy's chat-message and
LM keyword interface. Use the DSPy adapter for `codex/...` model names,
`messages`, `response_format`, and `reasoning_effort`.

## Credentials and custom clients

```python
client = CodexClient(auth="/path/to/codex-auth.json", user_agent="my-app/1.0")
```

Credentials are read for each request and refreshed when necessary. Refreshes
take an exclusive file lock. Missing or non-ChatGPT credentials raise
`CodexAuthError` with login instructions. `CodexClient` also accepts an explicit
bearer credential through `api_key=` and its account through `account_id=`;
these are Codex subscription credentials.

Existing auth-only helpers remain available:

```python
from openai_codex_auth import CodexAuth, codex_headers, getauthtoken

auth = CodexAuth()
token = auth.token()
headers = codex_headers(token, account_id=auth.account_id())
```

HTTP requests use the core's originator and configured User-Agent. Bearer
credentials stay in authentication headers, outside request bodies.

Set authentication, originator, and User-Agent through their explicit client
options. Conflicting entries in `headers=` are ignored in favor of those
options and the required protocol headers. HTTP errors expose their status and
credential-redacted provider payload through `CodexHTTPError.status_code` and
`CodexHTTPError.body`.

## Development and release

```bash
uv sync --locked --dev
uv run pytest
uv run ruff check .
uv build --no-sources
```

Tests use synthetic credentials and mocked transports. See
[RELEASING.md](RELEASING.md) for publishing instructions and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
