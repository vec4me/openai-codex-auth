# openai-codex-auth

Use the Codex CLI's ChatGPT login as a bearer credential, so OpenAI model calls
bill to a ChatGPT (Codex) subscription instead of an API key.

`codex login` stores tokens in `~/.codex/auth.json`. This package reads them,
refreshes the access token when it has expired, and writes the result back in
the CLI's format so the two stay in sync. One runtime dependency: `requests`.

## Install

```bash
uv add openai-codex-auth
codex login   # once, choosing "Sign in with ChatGPT"
```

## Usage

```python
import openai_codex_auth

auth = openai_codex_auth.CodexAuth()          # or CodexAuth("/path/to/auth.json")
token = auth.token()                            # refreshed if expired
headers = openai_codex_auth.codex_headers(token, account_id=auth.account_id(), originator="my_app")
base_url = openai_codex_auth.DEFAULT_CODEX_API_BASE
```

Send Responses API requests to `base_url` with `Authorization: Bearer <token>`
plus `headers`. The backend requires `stream: true`, rejects `system` role
input items (use `instructions` or `developer`), and rejects
`max_output_tokens`.

`openai_codex_auth.getauthtoken()` is the one-call form of the above.

Refreshes take an exclusive lock on the file, so parallel processes sharing
one login do not race each other. A missing file, or a CLI signed in with an
API key rather than ChatGPT, raises `CodexAuthError` naming `codex login`.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv build --no-sources
```

## Release

See [RELEASING.md](RELEASING.md).
