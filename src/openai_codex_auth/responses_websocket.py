"""Codex Responses WebSocket protocol client."""

from __future__ import annotations

import asyncio
import json
import math
import ssl
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from websockets.asyncio.client import connect as _async_connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidStatus
from websockets.sync.client import connect as _sync_connect

DEFAULT_CODEX_WEBSOCKET_BETA = "responses_websockets=2026-02-06"
DEFAULT_CODEX_WEBSOCKET_ORIGINATOR = "codex_cli_rs"
DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT = 10.0
DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT = 300.0
DEFAULT_CODEX_WEBSOCKET_USER_AGENT = "openai-codex-auth"

_PROTECTED_HEADER_NAMES = {
    "authorization",
    "openai-beta",
    "originator",
    "session-id",
    "thread-id",
    "user-agent",
    "x-client-request-id",
}


class CodexWebSocketError(RuntimeError):
    """Base error for the Codex Responses WebSocket transport."""


class CodexWebSocketConnectionError(CodexWebSocketError):
    """A recoverable network failure interrupted the WebSocket transport."""


class CodexWebSocketProtocolError(CodexWebSocketError):
    """The peer sent a frame that violates the expected protocol."""


class CodexWebSocketTimeoutError(TimeoutError, CodexWebSocketError):
    """The WebSocket connection or event stream exceeded its timeout."""


class CodexWebSocketResponseError(CodexWebSocketError):
    """The Codex backend returned a structured WebSocket error event."""

    def __init__(
        self,
        *,
        status_code: int | None,
        message: str,
        error_type: str | None,
        param: str | None,
        code: Any,
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        details = [
            f"status={status_code}" if status_code is not None else None,
            f"type={error_type}" if error_type else None,
            f"param={param}" if param else None,
            f"code={code}" if code is not None else None,
        ]
        rendered_details = ", ".join(detail for detail in details if detail)
        prefix = "Codex WebSocket response error"
        if rendered_details:
            prefix = f"{prefix} ({rendered_details})"
        super().__init__(f"{prefix}: {message}")


@dataclass(frozen=True, slots=True)
class WebSocketResult:
    """Responses events and the terminal response from one WebSocket call."""

    events: list[dict[str, Any]]
    response: dict[str, Any]


def codex_websocket_url(api_base: str) -> str:
    """Convert a Codex HTTP API base URL to its Responses WebSocket URL."""
    parsed = urlsplit(api_base)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme.lower())
    if scheme is None:
        raise ValueError("Codex api_base must use HTTP or HTTPS")
    if not parsed.netloc:
        raise ValueError("Codex api_base must include a host")
    if parsed.fragment:
        raise ValueError("Codex api_base must not include a URL fragment")

    path = f"{parsed.path.rstrip('/')}/responses"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def _validate_timeout(name: str, value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return timeout


def _handshake_headers(
    headers: Mapping[str, Any] | None,
    *,
    api_key: str,
    user_agent: str,
) -> dict[str, str]:
    request_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    handshake_headers = {
        str(key): str(value)
        for key, value in (headers or {}).items()
        if str(key).lower() not in _PROTECTED_HEADER_NAMES
    }
    handshake_headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": DEFAULT_CODEX_WEBSOCKET_BETA,
            "originator": DEFAULT_CODEX_WEBSOCKET_ORIGINATOR,
            "User-Agent": user_agent,
            "x-client-request-id": request_id,
            "session-id": session_id,
            "thread-id": request_id,
        }
    )
    return handshake_headers


def _contains_secret(value: Any, secret: str) -> bool:
    if not secret:
        return False
    if isinstance(value, str):
        return secret in value
    if isinstance(value, Mapping):
        return any(
            _contains_secret(key, secret) or _contains_secret(item, secret)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _serialize_request(request: dict[str, Any], *, api_key: str) -> str:
    if "api_key" in request:
        raise ValueError("api_key must not appear in a WebSocket request frame")
    if _contains_secret(request, api_key):
        raise ValueError("bearer credential must not appear in a WebSocket frame")

    payload = dict(request)
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Codex WebSocket request model must be a non-empty string")
    if "/" in model:
        raise ValueError("Codex WebSocket request model must be a bare model ID")
    payload["type"] = "response.create"
    frame = json.dumps(payload, separators=(",", ":"))
    if api_key and api_key in frame:
        raise ValueError("bearer credential must not appear in a WebSocket frame")
    return frame


def _decode_event(raw_message: str | bytes, *, api_key: str) -> dict[str, Any]:
    if not isinstance(raw_message, str):
        raise CodexWebSocketProtocolError(
            "Codex WebSocket returned an unexpected binary frame"
        )
    invalid_json = False
    try:
        event = json.loads(raw_message)
    except json.JSONDecodeError:
        invalid_json = True
        event = None
    if invalid_json:
        raise CodexWebSocketProtocolError(
            "Codex WebSocket frame must contain valid JSON"
        )
    if not isinstance(event, dict):
        raise CodexWebSocketProtocolError("Codex WebSocket event must be a JSON object")
    if not isinstance(event.get("type"), str):
        raise CodexWebSocketProtocolError(
            "Codex WebSocket event must include a string type"
        )
    if event["type"] != "error" and _contains_secret(event, api_key):
        raise CodexWebSocketProtocolError(
            "Codex WebSocket event contained the bearer credential"
        )
    return event


def _redact(value: Any, secret: str) -> str:
    rendered = str(value)
    if secret:
        rendered = rendered.replace(secret, "<redacted>")
    return rendered


def _status_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _raise_response_error(event: dict[str, Any], *, api_key: str) -> None:
    raw_error = event.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    raw_message = error.get("message", "unknown backend error")
    error_type = error.get("type")
    param = error.get("param")
    code = error.get("code")
    raise CodexWebSocketResponseError(
        status_code=_status_code(event.get("status", event.get("status_code"))),
        message=_redact(raw_message, api_key),
        error_type=_redact(error_type, api_key)
        if isinstance(error_type, str)
        else None,
        param=_redact(param, api_key) if isinstance(param, str) else None,
        code=_redact(code, api_key) if code is not None else None,
    )


def _terminal_response(
    event: dict[str, Any],
    *,
    api_key: str,
) -> dict[str, Any] | None:
    event_type = event["type"]
    if event_type == "error":
        _raise_response_error(event, api_key=api_key)
    if event_type in {"response.failed", "response.incomplete"}:
        response = event.get("response")
        if not isinstance(response, dict):
            raise CodexWebSocketProtocolError(
                f"{event_type} must include a response object"
            )
        raw_error = response.get("error")
        error = raw_error if isinstance(raw_error, dict) else {}
        _raise_response_error(
            {
                "error": {
                    "message": f"Codex WebSocket returned {event_type}",
                    "type": event_type,
                    **error,
                }
            },
            api_key=api_key,
        )
    if event_type != "response.completed":
        return None

    response = event.get("response")
    if not isinstance(response, dict):
        raise CodexWebSocketProtocolError(
            "response.completed must include a response object"
        )
    if response.get("status") != "completed":
        raise CodexWebSocketProtocolError(
            "response.completed response status must be completed"
        )
    model = response.get("model")
    if not isinstance(model, str) or not model:
        raise CodexWebSocketProtocolError(
            "response.completed response model must be a non-empty string"
        )
    if "output" in response and not isinstance(response["output"], list):
        raise CodexWebSocketProtocolError(
            "response.completed response output must be a list"
        )
    return response


def _connection_options(
    websocket_url: str,
    *,
    headers: dict[str, str],
    connect_timeout: float,
) -> dict[str, Any]:
    ssl_context = None
    if urlsplit(websocket_url).scheme == "wss":
        ssl_context = ssl.create_default_context(cafile=requests.certs.where())
    return {
        "ssl": ssl_context,
        "additional_headers": headers,
        "user_agent_header": None,
        "open_timeout": connect_timeout,
    }


def _receive_sync_event(
    connection: Any,
    *,
    idle_timeout: float,
    api_key: str,
) -> dict[str, Any]:
    timed_out = False
    closed = False
    connection_lost = False
    try:
        raw_message = connection.recv(timeout=idle_timeout)
    except TimeoutError:
        timed_out = True
        raw_message = None
    except ConnectionClosed as exc:
        closed = True
        connection_lost = isinstance(exc, ConnectionClosedError) and exc.rcvd is None
        raw_message = None

    if timed_out:
        raise CodexWebSocketTimeoutError(
            f"Codex WebSocket received no event for {idle_timeout} seconds"
        )
    if connection_lost:
        raise CodexWebSocketConnectionError(
            "Codex WebSocket connection lost before response.completed"
        )
    if closed:
        raise CodexWebSocketProtocolError(
            "Codex WebSocket closed before response.completed"
        )
    return _decode_event(raw_message, api_key=api_key)


async def _receive_async_event(
    connection: Any,
    *,
    idle_timeout: float,
    api_key: str,
) -> dict[str, Any]:
    timed_out = False
    closed = False
    connection_lost = False
    try:
        async with asyncio.timeout(idle_timeout):
            raw_message = await connection.recv()
    except TimeoutError:
        timed_out = True
        raw_message = None
    except ConnectionClosed as exc:
        closed = True
        connection_lost = isinstance(exc, ConnectionClosedError) and exc.rcvd is None
        raw_message = None

    if timed_out:
        raise CodexWebSocketTimeoutError(
            f"Codex WebSocket received no event for {idle_timeout} seconds"
        )
    if connection_lost:
        raise CodexWebSocketConnectionError(
            "Codex WebSocket connection lost before response.completed"
        )
    if closed:
        raise CodexWebSocketProtocolError(
            "Codex WebSocket closed before response.completed"
        )
    return _decode_event(raw_message, api_key=api_key)


def _sync_response(
    websocket_url: str,
    *,
    options: dict[str, Any],
    frame: str,
    idle_timeout: float,
    api_key: str,
) -> WebSocketResult:
    events: list[dict[str, Any]] = []
    with _sync_connect(websocket_url, **options) as connection:
        connection.send(frame)
        while True:
            event = _receive_sync_event(
                connection,
                idle_timeout=idle_timeout,
                api_key=api_key,
            )
            events.append(event)
            response = _terminal_response(event, api_key=api_key)
            if response is not None:
                return WebSocketResult(events=events, response=response)


async def _async_response(
    websocket_url: str,
    *,
    options: dict[str, Any],
    frame: str,
    idle_timeout: float,
    api_key: str,
) -> WebSocketResult:
    events: list[dict[str, Any]] = []
    async with _async_connect(websocket_url, **options) as connection:
        await connection.send(frame)
        while True:
            event = await _receive_async_event(
                connection,
                idle_timeout=idle_timeout,
                api_key=api_key,
            )
            events.append(event)
            response = _terminal_response(event, api_key=api_key)
            if response is not None:
                return WebSocketResult(events=events, response=response)


def websocket_response(
    request: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    headers: Mapping[str, Any] | None,
    user_agent: str = DEFAULT_CODEX_WEBSOCKET_USER_AGENT,
    connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
) -> WebSocketResult:
    """Send one synchronous Codex Responses request over WebSocket."""
    connect_timeout = _validate_timeout("connect_timeout", connect_timeout)
    idle_timeout = _validate_timeout("idle_timeout", idle_timeout)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("api_key must be a non-empty string")
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise ValueError("user_agent must be a non-empty string")
    websocket_url = codex_websocket_url(api_base)
    frame = _serialize_request(request, api_key=api_key)
    handshake_headers = _handshake_headers(
        headers, api_key=api_key, user_agent=user_agent
    )
    options = _connection_options(
        websocket_url,
        headers=handshake_headers,
        connect_timeout=connect_timeout,
    )

    connect_timed_out = False
    connection_failed = False
    handshake_failed = False
    handshake_status: int | None = None
    network_failed = False
    try:
        return _sync_response(
            websocket_url,
            options=options,
            frame=frame,
            idle_timeout=idle_timeout,
            api_key=api_key,
        )
    except CodexWebSocketError:
        raise
    except TimeoutError:
        connect_timed_out = True
    except InvalidStatus as exc:
        handshake_failed = True
        handshake_status = _status_code(exc.response.status_code)
    except ssl.SSLError:
        connection_failed = True
    except OSError:
        network_failed = True
    except Exception:
        connection_failed = True

    if connect_timed_out:
        raise CodexWebSocketTimeoutError(
            f"Codex WebSocket connection exceeded {connect_timeout} seconds"
        )
    if handshake_failed:
        raise CodexWebSocketResponseError(
            status_code=handshake_status,
            message="Codex WebSocket handshake rejected",
            error_type=None,
            param=None,
            code=None,
        )
    if network_failed:
        raise CodexWebSocketConnectionError("Codex WebSocket connection failed")
    if connection_failed:
        raise CodexWebSocketError("Codex WebSocket connection failed")
    raise AssertionError("unreachable")


async def awebsocket_response(
    request: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    headers: Mapping[str, Any] | None,
    user_agent: str = DEFAULT_CODEX_WEBSOCKET_USER_AGENT,
    connect_timeout: float = DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT,
    idle_timeout: float = DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT,
) -> WebSocketResult:
    """Send one asynchronous Codex Responses request over WebSocket."""
    connect_timeout = _validate_timeout("connect_timeout", connect_timeout)
    idle_timeout = _validate_timeout("idle_timeout", idle_timeout)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("api_key must be a non-empty string")
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise ValueError("user_agent must be a non-empty string")
    websocket_url = codex_websocket_url(api_base)
    frame = _serialize_request(request, api_key=api_key)
    handshake_headers = _handshake_headers(
        headers, api_key=api_key, user_agent=user_agent
    )
    options = _connection_options(
        websocket_url,
        headers=handshake_headers,
        connect_timeout=connect_timeout,
    )

    connect_timed_out = False
    connection_failed = False
    handshake_failed = False
    handshake_status: int | None = None
    network_failed = False
    try:
        return await _async_response(
            websocket_url,
            options=options,
            frame=frame,
            idle_timeout=idle_timeout,
            api_key=api_key,
        )
    except CodexWebSocketError:
        raise
    except TimeoutError:
        connect_timed_out = True
    except InvalidStatus as exc:
        handshake_failed = True
        handshake_status = _status_code(exc.response.status_code)
    except ssl.SSLError:
        connection_failed = True
    except OSError:
        network_failed = True
    except Exception:
        connection_failed = True

    if connect_timed_out:
        raise CodexWebSocketTimeoutError(
            f"Codex WebSocket connection exceeded {connect_timeout} seconds"
        )
    if handshake_failed:
        raise CodexWebSocketResponseError(
            status_code=handshake_status,
            message="Codex WebSocket handshake rejected",
            error_type=None,
            param=None,
            code=None,
        )
    if network_failed:
        raise CodexWebSocketConnectionError("Codex WebSocket connection failed")
    if connection_failed:
        raise CodexWebSocketError("Codex WebSocket connection failed")
    raise AssertionError("unreachable")


__all__ = [
    "DEFAULT_CODEX_WEBSOCKET_BETA",
    "DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT",
    "DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT",
    "DEFAULT_CODEX_WEBSOCKET_ORIGINATOR",
    "DEFAULT_CODEX_WEBSOCKET_USER_AGENT",
    "CodexWebSocketConnectionError",
    "CodexWebSocketError",
    "CodexWebSocketProtocolError",
    "CodexWebSocketResponseError",
    "CodexWebSocketTimeoutError",
    "WebSocketResult",
    "awebsocket_response",
    "codex_websocket_url",
    "websocket_response",
]
