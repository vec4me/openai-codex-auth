"""HTTPX streaming transport for native Codex Responses JSON events."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .responses import (
    CodexHTTPError,
    CodexProtocolError,
    ResponseBuilder,
    contains_secret,
    redact_secret,
)


class _SSEDecoder:
    def __init__(self, api_key: str) -> None:
        self._data: list[str] = []
        self._event: str | None = None
        self._api_key = api_key
        self.ended = False

    def feed(self, line: str) -> dict[str, Any] | None:
        if line:
            if line.startswith(":"):
                return None
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                self._data.append(value)
            elif field == "event":
                self._event = value
            return None
        if not self._data:
            self._event = None
            return None
        payload = "\n".join(self._data)
        event_name = self._event
        self._data = []
        self._event = None
        if payload == "[DONE]":
            self.ended = True
            return None
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            event = None
        if event is None:
            raise CodexProtocolError("Codex SSE event must contain valid JSON")
        if not isinstance(event, dict):
            raise CodexProtocolError("Codex SSE event must be a JSON object")
        if not isinstance(event.get("type"), str) or not event["type"]:
            raise CodexProtocolError("Codex SSE event must include a string type")
        if event_name and event_name != event["type"]:
            raise CodexProtocolError(
                "Codex SSE event name conflicts with JSON event type"
            )
        if contains_secret(event, self._api_key):
            raise CodexProtocolError("Codex SSE event contained the bearer credential")
        return event


def _raise_http_error(response: httpx.Response, api_key: str) -> None:
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        body = None
    raise CodexHTTPError(response.status_code, redact_secret(body, api_key))


def _validate_content_type(response: httpx.Response) -> None:
    # The Codex backend can omit Content-Type on an otherwise valid SSE stream.
    if "content-type" not in response.headers:
        return
    content_type = (
        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type != "text/event-stream":
        raise CodexProtocolError("Codex HTTP response must use text/event-stream")


def http_response(
    request: dict[str, Any],
    *,
    url: str,
    headers: dict[str, str],
    timeout: httpx.Timeout,
    api_key: str,
) -> dict[str, Any]:
    builder = ResponseBuilder()
    decoder = _SSEDecoder(api_key)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, headers=headers, json=request) as response:
            if response.is_error:
                response.read()
                _raise_http_error(response, api_key)
            _validate_content_type(response)
            for line in response.iter_lines():
                event = decoder.feed(line)
                if event is not None:
                    builder.add(event)
                    if event["type"] == "response.completed":
                        return builder.build()
                if decoder.ended:
                    return builder.build()
            last_event = decoder.feed("")
            if last_event is not None:
                builder.add(last_event)
    return builder.build()


async def ahttp_response(
    request: dict[str, Any],
    *,
    url: str,
    headers: dict[str, str],
    timeout: httpx.Timeout,
    api_key: str,
) -> dict[str, Any]:
    builder = ResponseBuilder()
    decoder = _SSEDecoder(api_key)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, headers=headers, json=request
        ) as response:
            if response.is_error:
                await response.aread()
                _raise_http_error(response, api_key)
            _validate_content_type(response)
            async for line in response.aiter_lines():
                event = decoder.feed(line)
                if event is not None:
                    builder.add(event)
                    if event["type"] == "response.completed":
                        return builder.build()
                if decoder.ended:
                    return builder.build()
            last_event = decoder.feed("")
            if last_event is not None:
                builder.add(last_event)
    return builder.build()
