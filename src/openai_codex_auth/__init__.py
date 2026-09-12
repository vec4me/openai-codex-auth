"""Use the Codex CLI's ChatGPT login as a bearer credential.

Run ``codex login`` once. This module reads the tokens the CLI stores in
``~/.codex/auth.json``, refreshes the access token when it has expired, and
writes the result back in the CLI's own format so both stay in sync.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_AUTH_PATH = Path("~/.codex/auth.json").expanduser()
DEFAULT_CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "openai_codex_auth"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
_ACCOUNT_CLAIM = "https://api.openai.com/auth"
_EXPIRY_LEEWAY_SECONDS = 60


class CodexAuthError(RuntimeError):
    """No usable ChatGPT credential in the Codex CLI auth file."""


def _jwt_claims(token: str) -> dict[str, Any]:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def _is_fresh(access_token: str) -> bool:
    return _jwt_claims(access_token).get("exp", 0) > time.time() + _EXPIRY_LEEWAY_SECONDS


class CodexAuth:
    """Access token and account id from the Codex CLI's ``auth.json``."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_AUTH_PATH):
        self.path = Path(path).expanduser()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            raise CodexAuthError(
                f"No Codex credential at {self.path}. Run `codex login`."
            ) from None
        tokens = data.get("tokens") or {}
        if not tokens.get("access_token") or not tokens.get("refresh_token"):
            raise CodexAuthError(
                f"{self.path} has no ChatGPT login (auth_mode="
                f"{data.get('auth_mode')!r}). Run `codex login` and sign in with ChatGPT."
            )
        return data

    def token(self) -> str:
        """The current access token, refreshed first if it has expired."""
        access = self._read()["tokens"]["access_token"]
        if _is_fresh(access):
            return access
        return self._refresh()["tokens"]["access_token"]

    def account_id(self) -> str:
        tokens = self._read()["tokens"]
        return tokens.get("account_id") or _jwt_claims(tokens["access_token"])[
            _ACCOUNT_CLAIM
        ]["chatgpt_account_id"]

    def _refresh(self) -> dict[str, Any]:
        with self.path.open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            data = json.loads(handle.read())
            tokens = data["tokens"]
            if _is_fresh(tokens["access_token"]):
                return data  # another process refreshed while we waited for the lock
            response = requests.post(
                CODEX_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens["refresh_token"],
                    "client_id": CODEX_CLIENT_ID,
                },
                timeout=30,
            )
            response.raise_for_status()
            fresh = response.json()
            tokens["access_token"] = fresh["access_token"]
            tokens["refresh_token"] = fresh.get("refresh_token") or tokens["refresh_token"]
            if fresh.get("id_token"):
                tokens["id_token"] = fresh["id_token"]
            data["last_refresh"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            handle.seek(0)
            handle.truncate()
            json.dump(data, handle, indent=2)
            return data


def getauthtoken(path: str | os.PathLike[str] = DEFAULT_AUTH_PATH) -> str:
    return CodexAuth(path).token()


def codex_headers(
    token: str,
    *,
    account_id: str | None = None,
    originator: str = DEFAULT_CODEX_ORIGINATOR,
    extra_headers: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Headers the Codex backend expects alongside ``Authorization: Bearer``."""
    headers = {
        "chatgpt-account-id": account_id
        or _jwt_claims(token)[_ACCOUNT_CLAIM]["chatgpt_account_id"],
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
    }
    if extra_headers:
        headers.update({str(key): str(value) for key, value in extra_headers.items()})
    return headers


__all__ = [
    "CODEX_CLIENT_ID",
    "CODEX_TOKEN_URL",
    "DEFAULT_AUTH_PATH",
    "DEFAULT_CODEX_API_BASE",
    "DEFAULT_CODEX_ORIGINATOR",
    "CodexAuth",
    "CodexAuthError",
    "codex_headers",
    "getauthtoken",
]

# Authentication is defined before loading the client, which reuses these helpers.
from .client import (  # noqa: E402
    DEFAULT_CODEX_INSTRUCTIONS,
    DEFAULT_CODEX_USER_AGENT,
    CodexClient,
    CodexTransport,
)
from .responses import (  # noqa: E402
    CodexEmptyResponseError,
    CodexError,
    CodexHTTPError,
    CodexProtocolError,
    CodexResponse,
)

__all__ += [
    "DEFAULT_CODEX_INSTRUCTIONS",
    "DEFAULT_CODEX_USER_AGENT",
    "CodexClient",
    "CodexTransport",
    "CodexResponse",
    "CodexError",
    "CodexHTTPError",
    "CodexProtocolError",
    "CodexEmptyResponseError",
]
