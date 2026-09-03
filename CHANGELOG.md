# Changelog

## 0.1.0 - 2026-09-03

### Added

- `CodexAuth`: reads the Codex CLI's `~/.codex/auth.json`, refreshes the
  access token under a file lock when it has expired, and writes the refreshed
  tokens back in the CLI's format.
- `codex_headers()` and `DEFAULT_CODEX_API_BASE` for presenting the credential
  to the Codex backend.

Replaces the OAuth login flow and Pi-format credential store that previously
lived in `dspy-codex-auth`; `codex login` is now the only login path.
