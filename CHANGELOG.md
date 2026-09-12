# Changelog

## 0.2.0 - 2026-09-11

### Added

- Standalone `CodexClient.create()` and `acreate()` with HTTP and WebSocket
  transports, shared authentication, bounded retries, and exact structured
  model-not-found transport selection.
- `CodexResponse` with native provider metadata, usage, text, reasoning, and
  function-call output reconstructed from streamed events.
- Protocol/error validation and transport tests extracted from
  `dspy-codex-auth`. The core imports neither DSPy nor LiteLLM.

### Changed

- Runtime dependencies now include HTTPX and WebSockets alongside Requests.
  Existing auth-only helpers remain available.

## 0.1.0 - 2026-09-03

### Added

- `CodexAuth`: reads the Codex CLI's `~/.codex/auth.json`, refreshes the
  access token under a file lock when it has expired, and writes the refreshed
  tokens back in the CLI's format.
- `codex_headers()` and `DEFAULT_CODEX_API_BASE` for presenting the credential
  to the Codex backend.

Replaces the OAuth login flow and Pi-format credential store that previously
lived in `dspy-codex-auth`; `codex login` is now the only login path.
