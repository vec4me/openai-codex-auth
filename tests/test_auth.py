from __future__ import annotations

import base64
import json
import time
import tomllib
from pathlib import Path

import pytest

import openai_codex_auth
from openai_codex_auth import CodexAuth, CodexAuthError, codex_headers


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _jwt(*, exp: float, account_id: str = "acct_jwt") -> str:
    claims = {"exp": int(exp), "https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    return ".".join([_b64url({"alg": "none"}), _b64url(claims), "sig"])


def _write_auth(path: Path, access: str, *, account_id: str | None = "acct_file") -> None:
    tokens = {"access_token": access, "refresh_token": "refresh-a", "id_token": "id-a"}
    if account_id:
        tokens["account_id"] = account_id
    path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": tokens, "last_refresh": "old"}))


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fresh_token_is_returned_without_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(openai_codex_auth.requests, "post", lambda *a, **k: pytest.fail("refreshed"))
    access = _jwt(exp=time.time() + 3600)
    _write_auth(tmp_path / "auth.json", access)

    assert CodexAuth(tmp_path / "auth.json").token() == access


def test_expired_token_is_refreshed_and_written_back(tmp_path, monkeypatch):
    posted = []
    new_access = _jwt(exp=time.time() + 3600)

    def fake_post(url, *, data, timeout):
        posted.append((url, data))
        return _Response({"access_token": new_access, "refresh_token": "refresh-b", "id_token": "id-b"})

    monkeypatch.setattr(openai_codex_auth.requests, "post", fake_post)
    path = tmp_path / "auth.json"
    _write_auth(path, _jwt(exp=time.time() - 1))

    assert CodexAuth(path).token() == new_access

    assert posted == [
        (
            openai_codex_auth.CODEX_TOKEN_URL,
            {"grant_type": "refresh_token", "refresh_token": "refresh-a", "client_id": openai_codex_auth.CODEX_CLIENT_ID},
        )
    ]
    saved = json.loads(path.read_text())
    assert saved["auth_mode"] == "chatgpt"  # untouched fields survive
    assert saved["tokens"] == {"access_token": new_access, "refresh_token": "refresh-b", "id_token": "id-b", "account_id": "acct_file"}
    assert saved["last_refresh"].endswith("Z") and saved["last_refresh"] != "old"


def test_missing_file_points_to_codex_login(tmp_path):
    with pytest.raises(CodexAuthError, match="codex login"):
        CodexAuth(tmp_path / "missing.json").token()


def test_api_key_login_is_rejected(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "sk-x", "tokens": None}))

    with pytest.raises(CodexAuthError, match="auth_mode='apikey'"):
        CodexAuth(path).token()


def test_account_id_prefers_file_then_jwt(tmp_path):
    access = _jwt(exp=time.time() + 3600, account_id="acct_jwt")
    path = tmp_path / "auth.json"

    _write_auth(path, access, account_id="acct_file")
    assert CodexAuth(path).account_id() == "acct_file"

    _write_auth(path, access, account_id=None)
    assert CodexAuth(path).account_id() == "acct_jwt"


def test_codex_headers():
    token = _jwt(exp=0, account_id="acct_jwt")

    assert codex_headers(token) == {
        "chatgpt-account-id": "acct_jwt",
        "OpenAI-Beta": "responses=experimental",
        "originator": "openai_codex_auth",
    }
    custom = codex_headers(token, account_id="acct_x", originator="my_app", extra_headers={"X-Trace": 7})
    assert custom["chatgpt-account-id"] == "acct_x"
    assert custom["originator"] == "my_app"
    assert custom["X-Trace"] == "7"


def test_runtime_dependency_is_requests_only():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())

    assert [d.split(">=")[0] for d in pyproject["project"]["dependencies"]] == ["requests"]
