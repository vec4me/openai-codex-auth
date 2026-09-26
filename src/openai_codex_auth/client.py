"""Framework-independent authenticated Codex Responses client."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from copy import deepcopy
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import DEFAULT_CODEX_API_BASE, DEFAULT_CODEX_ORIGINATOR, CodexAuth, codex_headers
from .responses import (
    CodexEmptyResponseError,
    CodexError,
    CodexHTTPError,
    CodexResponse,
    ResponseBuilder,
    contains_secret,
    has_output,
)
from .responses_http import ahttp_response, http_response
from .responses_websocket import (
    CodexWebSocketConnectionError,
    CodexWebSocketResponseError,
    CodexWebSocketTimeoutError,
    awebsocket_response,
    websocket_response,
)

DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."
DEFAULT_CODEX_USER_AGENT = "openai-codex-auth"
type CodexTransport = Literal["auto", "http", "websocket"]

_CLIENT_ONLY_PARAMETERS = frozenset(
    {
        "api_base",
        "api_key",
        "account_id",
        "chatgpt_account_id",
        "auth",
        "auth_storage",
        "auth_provider",
        "headers",
        "extra_headers",
        "originator",
        "user_agent",
        "messages",
        "model_type",
        "use_developer_role",
        "response_format",
        "reasoning_effort",
        "model_reasoning_effort",
        "reasoning_summary",
        "model_reasoning_summary",
        "max_tokens",
        "max_completion_tokens",
        "cache",
        "caching",
        "rollout_id",
        "num_retries",
        "retry_strategy",
        "request_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
        "stream_timeout",
        "codex_transport",
        "codex_websocket_connect_timeout",
        "codex_websocket_idle_timeout",
        "_dspy_codex_transport_controls",
        "type",
    }
)
_PROTECTED_HEADERS = frozenset(
    {
        "authorization",
        "chatgpt-account-id",
        "originator",
        "openai-beta",
        "user-agent",
        "content-type",
        "accept",
    }
)


def _positive_timeout(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _http_timeout(
    value: float | httpx.Timeout, connect_timeout: float
) -> httpx.Timeout:
    if isinstance(value, httpx.Timeout):
        for phase, deadline in value.as_dict().items():
            if deadline is not None:
                _positive_timeout(f"timeout.{phase}", deadline)
        return value
    return httpx.Timeout(_positive_timeout("timeout", value), connect=connect_timeout)


def _request(
    model: str,
    input: str | list[dict[str, Any]],
    instructions: str | None,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(model, str)
        or not model
        or model != model.strip()
        or "/" in model
    ):
        raise ValueError(
            "model must be a nonempty bare Codex model ID without a provider prefix"
        )
    if not isinstance(input, str) and (
        not isinstance(input, list) or any(not isinstance(item, dict) for item in input)
    ):
        raise ValueError(
            "input must be a string or a list of native Responses input objects"
        )
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("instructions must be a string or None")
    invalid = _CLIENT_ONLY_PARAMETERS.intersection(parameters)
    if invalid:
        raise ValueError(
            f"Unsupported client or framework parameters: {', '.join(sorted(invalid))}"
        )
    if "stream" in parameters and parameters["stream"] is not True:
        raise ValueError("Codex requires stream=True")
    if "store" in parameters and parameters["store"] is not False:
        raise ValueError("Codex requires store=False")
    request = deepcopy(parameters)
    request.pop("max_output_tokens", None)
    if isinstance(request.get("service_tier"), str):
        tier = request["service_tier"].lower()
        if tier == "fast":
            request["service_tier"] = "priority"
        elif tier in {"priority", "flex"}:
            request["service_tier"] = tier
    request.update(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": input}]}]
        if isinstance(input, str)
        else deepcopy(input),
        stream=True,
        store=False,
    )
    if instructions is not None:
        request["instructions"] = instructions
    _validate_json(request)
    try:
        json.dumps(request, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError(
            "Codex request parameters must be finite JSON values"
        ) from None
    return request


def _validate_json(value: Any) -> None:
    """Reject Python-only shapes before serialization can silently convert them."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("Codex request parameters must use native JSON objects and lists")


def _model_not_found(error: CodexHTTPError, model: str) -> bool:
    return error.status_code == 404 and error.body == {
        "error": {
            "message": f"Model not found {model}",
            "type": "invalid_request_error",
            "param": "model",
            "code": None,
        }
    }


def _retryable(error: Exception) -> bool:
    if isinstance(
        error, (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)
    ):
        return True
    if isinstance(error, CodexHTTPError):
        return error.status_code == 429 or 500 <= error.status_code <= 599
    if isinstance(error, CodexWebSocketResponseError):
        return error.status_code == 429 or (
            error.status_code is not None and 500 <= error.status_code <= 599
        )
    if isinstance(error, (CodexWebSocketConnectionError, CodexWebSocketTimeoutError)):
        return True
    return isinstance(error, CodexEmptyResponseError)


def _raise_final(error: Exception) -> None:
    if isinstance(error, httpx.TransportError):
        raise CodexError("Codex HTTP connection or stream failed") from None
    raise error


class CodexClient:
    """Use Codex CLI auth for native Responses requests, without an ML framework.

    Auth is read on each attempt. ``auto`` tries HTTP and falls back to WebSocket
    only for the exact structured model-not-found HTTP 404. ``max_retries`` counts
    additional attempts for network failures, HTTP/backend 429 or 5xx, and empty
    or reasoning-only completions. Delays are 0.5, 1, 2, ... seconds, capped at 8.
    Malformed or unfinished responses are not retried. Additional headers cannot
    override authentication, protocol, or the explicit ``user_agent``.
    """

    def __init__(
        self,
        auth: CodexAuth | str | os.PathLike[str] | None = None,
        *,
        api_base: str = DEFAULT_CODEX_API_BASE,
        api_key: str | None = None,
        account_id: str | None = None,
        headers: dict[str, str] | None = None,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
        user_agent: str | None = None,
    ) -> None:
        if not isinstance(api_base, str):
            raise ValueError("api_base must be an HTTP or HTTPS URL")
        parsed = urlsplit(api_base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.fragment
        ):
            raise ValueError("api_base must be an HTTP or HTTPS URL without a fragment")
        if parsed.username or parsed.password:
            raise ValueError("api_base must not contain credentials")
        for name, value in (
            ("api_key", api_key),
            ("account_id", account_id),
            ("user_agent", user_agent),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string when provided")
        if not isinstance(originator, str) or not originator:
            raise ValueError("originator must be a nonempty string")
        if headers is not None and (
            not isinstance(headers, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in headers.items()
            )
        ):
            raise ValueError("headers must be a dictionary of strings")
        self.auth = (
            auth
            if isinstance(auth, CodexAuth)
            else CodexAuth()
            if auth is None
            else CodexAuth(auth)
        )
        self.api_base = api_base
        self.api_key = api_key
        self.account_id = account_id
        self.headers = dict(headers or {})
        self.originator = originator
        self.user_agent = user_agent or DEFAULT_CODEX_USER_AGENT
        self._url = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"{parsed.path.rstrip('/')}/responses",
                parsed.query,
                "",
            )
        )

    def _credentials(self, request: dict[str, Any]) -> tuple[str, dict[str, str]]:
        token = self.api_key if self.api_key is not None else self.auth.token()
        account_id = self.account_id
        if account_id is None and self.api_key is None:
            account_id = self.auth.account_id()
        if contains_secret(request, token):
            raise ValueError(
                "bearer credential must not appear in a Codex request body"
            )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _PROTECTED_HEADERS
        }
        headers.update(
            codex_headers(token, account_id=account_id, originator=self.originator)
        )
        headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent,
                "Accept": "text/event-stream",
            }
        )
        return token, headers

    @staticmethod
    def _validate_controls(
        transport: CodexTransport,
        timeout: float | httpx.Timeout,
        connect_timeout: float,
        idle_timeout: float,
        max_retries: int,
    ) -> tuple[httpx.Timeout, float, float]:
        if not isinstance(transport, str) or transport not in {
            "auto",
            "http",
            "websocket",
        }:
            raise ValueError("transport must be one of 'auto', 'http', or 'websocket'")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a nonnegative integer")
        connect_timeout = _positive_timeout("connect_timeout", connect_timeout)
        idle_timeout = _positive_timeout("idle_timeout", idle_timeout)
        return _http_timeout(timeout, connect_timeout), connect_timeout, idle_timeout

    def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        instructions: str | None = DEFAULT_CODEX_INSTRUCTIONS,
        transport: CodexTransport = "auto",
        timeout: float | httpx.Timeout = 300.0,
        connect_timeout: float = 10.0,
        idle_timeout: float = 300.0,
        max_retries: int = 3,
        **parameters: Any,
    ) -> CodexResponse:
        """Return complete native output; stream=True/store=False are required.

        String input is encoded as one native user message. Retry delays start
        at 0.5 seconds and double up to 8 seconds; retries share one budget across
        HTTP and WebSocket. Auto routing can add one model-not-found HTTP probe.
        ``service_tier='fast'`` becomes ``'priority'``. ``max_output_tokens`` is
        omitted because this Codex backend does not support output-token caps.
        A scalar HTTP timeout applies to each non-connect phase; an explicit
        ``httpx.Timeout`` controls all HTTP phases. WebSocket timeouts are separate.
        """
        http_timeout, connect_timeout, idle_timeout = self._validate_controls(
            transport, timeout, connect_timeout, idle_timeout, max_retries
        )
        request = _request(model, input, instructions, parameters)
        selected = "websocket" if transport == "websocket" else "http"
        attempt = 0
        while True:
            token, headers = self._credentials(request)
            try:
                if selected == "http":
                    response = http_response(
                        request,
                        url=self._url,
                        headers=headers,
                        timeout=http_timeout,
                        api_key=token,
                    )
                else:
                    result = websocket_response(
                        request,
                        api_base=self.api_base,
                        api_key=token,
                        headers=headers,
                        user_agent=self.user_agent,
                        connect_timeout=connect_timeout,
                        idle_timeout=idle_timeout,
                    )
                    builder = ResponseBuilder()
                    for event in result.events:
                        builder.add(event)
                    builder.complete(result.response)
                    response = builder.build()
                if not has_output(response):
                    raise CodexEmptyResponseError(
                        f"Codex completed without final output after {attempt + 1} attempt(s)"
                    )
                return CodexResponse(response=response, transport=selected)
            except Exception as error:
                if (
                    isinstance(error, CodexHTTPError)
                    and selected == "http"
                    and transport == "auto"
                    and _model_not_found(error, model)
                ):
                    selected = "websocket"
                    continue
                failure = error
            # Raise outside the handler so transport exceptions cannot expose
            # credential-bearing requests through exception chaining.
            if attempt >= max_retries or not _retryable(failure):
                _raise_final(failure)
            time.sleep(min(0.5 * 2 ** min(attempt, 4), 8.0))
            attempt += 1

    async def acreate(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        instructions: str | None = DEFAULT_CODEX_INSTRUCTIONS,
        transport: CodexTransport = "auto",
        timeout: float | httpx.Timeout = 300.0,
        connect_timeout: float = 10.0,
        idle_timeout: float = 300.0,
        max_retries: int = 3,
        **parameters: Any,
    ) -> CodexResponse:
        """Async equivalent of :meth:`create`, including its retry and timeout policy."""
        http_timeout, connect_timeout, idle_timeout = self._validate_controls(
            transport, timeout, connect_timeout, idle_timeout, max_retries
        )
        request = _request(model, input, instructions, parameters)
        selected = "websocket" if transport == "websocket" else "http"
        attempt = 0
        while True:
            token, headers = await asyncio.to_thread(self._credentials, request)
            try:
                if selected == "http":
                    response = await ahttp_response(
                        request,
                        url=self._url,
                        headers=headers,
                        timeout=http_timeout,
                        api_key=token,
                    )
                else:
                    result = await awebsocket_response(
                        request,
                        api_base=self.api_base,
                        api_key=token,
                        headers=headers,
                        user_agent=self.user_agent,
                        connect_timeout=connect_timeout,
                        idle_timeout=idle_timeout,
                    )
                    builder = ResponseBuilder()
                    for event in result.events:
                        builder.add(event)
                    builder.complete(result.response)
                    response = builder.build()
                if not has_output(response):
                    raise CodexEmptyResponseError(
                        f"Codex completed without final output after {attempt + 1} attempt(s)"
                    )
                return CodexResponse(response=response, transport=selected)
            except Exception as error:
                if (
                    isinstance(error, CodexHTTPError)
                    and selected == "http"
                    and transport == "auto"
                    and _model_not_found(error, model)
                ):
                    selected = "websocket"
                    continue
                failure = error
            # Raise outside the handler so transport exceptions cannot expose
            # credential-bearing requests through exception chaining.
            if attempt >= max_retries or not _retryable(failure):
                _raise_final(failure)
            await asyncio.sleep(min(0.5 * 2 ** min(attempt, 4), 8.0))
            attempt += 1
