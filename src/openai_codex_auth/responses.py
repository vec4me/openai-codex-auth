"""Native Responses event validation and lossless output reconstruction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


class CodexError(RuntimeError):
    """A Codex request could not produce a completed response."""


class CodexProtocolError(CodexError):
    """The backend returned malformed events or an unfinished response."""


class CodexEmptyResponseError(CodexError):
    """The backend completed without usable output after all attempts."""


class CodexHTTPError(CodexError):
    """An HTTP error with its parsed, credential-redacted provider body."""

    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = status_code
        self.body = body
        raw_error = body.get("error") if isinstance(body, dict) else None
        self.error = raw_error if isinstance(raw_error, dict) else None
        super().__init__(f"Codex HTTP request failed with status {status_code}")


@dataclass(frozen=True, slots=True)
class CodexResponse:
    """A native provider response with reconstructed output and metadata intact."""

    response: dict[str, Any]
    transport: Literal["http", "websocket"]

    @property
    def model(self) -> str:
        return self.response["model"]

    @property
    def output(self) -> list[dict[str, Any]]:
        return self.response.get("output", [])

    @property
    def usage(self) -> dict[str, Any]:
        return self.response.get("usage") or {}

    @property
    def output_text(self) -> str:
        return "".join(
            part["text"]
            for item in self.output
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text" and isinstance(part.get("text"), str)
        )


def contains_secret(value: Any, secret: str) -> bool:
    if not secret:
        return False
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(
            contains_secret(key, secret) or contains_secret(item, secret)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_secret(item, secret) for item in value)
    return False


def redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "<redacted>") if secret else value
    if isinstance(value, dict):
        return {
            redact_secret(key, secret): redact_secret(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secret(item, secret) for item in value]
    return value


def _index(event: dict[str, Any], field: str) -> int:
    index = event.get(field)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CodexProtocolError(
            f"Response event {field} must be a nonnegative integer"
        )
    return index


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CodexProtocolError(f"{name} must be a JSON object")
    return value


def _objects(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CodexProtocolError(f"{name} must be a list of JSON objects")
    return value


def _item(value: Any) -> dict[str, Any]:
    item = _object(value, "Response output item")
    if not isinstance(item.get("type"), str) or not item["type"]:
        raise CodexProtocolError("Response output item must include a string type")
    if "id" in item and not isinstance(item["id"], str):
        raise CodexProtocolError("Response output item id must be a string")
    for field in ("content", "summary"):
        if field in item:
            _objects(item[field], f"Response output item {field}")
    return item


def _merge_item(previous: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    merged = {**deepcopy(previous), **deepcopy(latest)}
    for field in ("content", "summary"):
        if field not in previous or field not in latest:
            continue
        parts = deepcopy(previous[field])
        for index, part in enumerate(latest[field]):
            if index < len(parts):
                parts[index] = {**parts[index], **deepcopy(part)}
            else:
                parts.append(deepcopy(part))
        merged[field] = parts
    return merged


class ResponseBuilder:
    """Consume provider JSON events and reconstruct each missing output part."""

    def __init__(self) -> None:
        self._items: dict[int, dict[str, Any]] = {}
        self._done_items: set[int] = set()
        self._parts: dict[tuple[int, str, int], dict[str, Any]] = {}
        self._done_parts: set[tuple[int, str, int]] = set()
        self._deltas: dict[tuple[int, str, int, str], list[str]] = {}
        self._done: dict[tuple[int, str, int, str], str] = {}
        self._arguments: dict[int, list[str]] = {}
        self._arguments_done: dict[int, str] = {}
        self._response: dict[str, Any] | None = None

    def _event_item(
        self, event: dict[str, Any], item_type: str
    ) -> tuple[int, dict[str, Any]]:
        index = _index(event, "output_index")
        item = self._items.setdefault(index, {"type": item_type})
        if item["type"] != item_type:
            raise CodexProtocolError(
                "Conflicting output item types at one output_index"
            )
        if "item_id" in event:
            if not isinstance(event["item_id"], str):
                raise CodexProtocolError("Response event item_id must be a string")
            item.setdefault("id", event["item_id"])
        if item_type == "message":
            item.setdefault("role", "assistant")
        return index, item

    def add(self, event: dict[str, Any]) -> None:
        event = _object(event, "Response event")
        kind = event.get("type")
        if not isinstance(kind, str) or not kind:
            raise CodexProtocolError(
                "Response event must include a nonempty string type"
            )
        if kind in {"error", "response.failed", "response.incomplete"}:
            raise CodexProtocolError(f"Codex stream terminated with {kind}")
        if kind == "response.completed":
            self.complete(event.get("response"))
            return
        if kind in {"response.output_item.added", "response.output_item.done"}:
            index = _index(event, "output_index")
            self._items[index] = _merge_item(
                self._items.get(index, {}), _item(event.get("item"))
            )
            if kind.endswith(".done"):
                self._done_items.add(index)
            return
        if kind in {"response.content_part.added", "response.content_part.done"}:
            index, _ = self._event_item(event, "message")
            key = (index, "content", _index(event, "content_index"))
            self._parts[key] = {
                **self._parts.get(key, {}),
                **deepcopy(_object(event.get("part"), "Content part")),
            }
            if kind.endswith(".done"):
                self._done_parts.add(key)
            return
        if kind in {
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            index, _ = self._event_item(event, "reasoning")
            key = (index, "summary", _index(event, "summary_index"))
            self._parts[key] = {
                **self._parts.get(key, {}),
                **deepcopy(_object(event.get("part"), "Reasoning part")),
            }
            if kind.endswith(".done"):
                self._done_parts.add(key)
            return
        if kind in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            index, item = self._event_item(event, "function_call")
            field = "delta" if kind.endswith(".delta") else "arguments"
            value = event.get(field)
            if not isinstance(value, str):
                raise CodexProtocolError(
                    f"Function call event {field} must be a string"
                )
            if field == "delta":
                self._arguments.setdefault(index, []).append(value)
            else:
                self._arguments_done[index] = value
            for name in ("name", "call_id"):
                if name in event:
                    item[name] = event[name]
            return
        text_events = {
            "response.output_text": (
                "message",
                "content",
                "content_index",
                "output_text",
                "text",
            ),
            "response.refusal": (
                "message",
                "content",
                "content_index",
                "refusal",
                "refusal",
            ),
            "response.reasoning_summary_text": (
                "reasoning",
                "summary",
                "summary_index",
                "summary_text",
                "text",
            ),
            "response.reasoning_text": (
                "reasoning",
                "content",
                "content_index",
                "reasoning_text",
                "text",
            ),
        }
        prefix, _, suffix = kind.rpartition(".")
        if prefix not in text_events or suffix not in {"delta", "done"}:
            return  # Other native event types do not change reconstructed text.
        item_type, container, index_field, part_type, text_field = text_events[prefix]
        index, _ = self._event_item(event, item_type)
        part_index = _index(event, index_field)
        part_key = (index, container, part_index)
        self._parts.setdefault(part_key, {"type": part_type})
        value = event.get("delta" if suffix == "delta" else text_field)
        if not isinstance(value, str):
            raise CodexProtocolError("Response text event must contain string text")
        text_key = (*part_key, text_field)
        if suffix == "done":
            self._done[text_key] = value
        else:
            self._deltas.setdefault(text_key, []).append(value)

    def complete(self, response: Any) -> None:
        response = _object(response, "response.completed response")
        if response.get("status") != "completed":
            raise CodexProtocolError("response.completed status must be completed")
        if not isinstance(response.get("model"), str) or not response["model"]:
            raise CodexProtocolError(
                "response.completed model must be a nonempty string"
            )
        if "output" in response:
            for item in _objects(response["output"], "Response output"):
                _item(item)
        if response.get("usage") is not None:
            _object(response["usage"], "Response usage")
        self._response = deepcopy(response)

    def build(self) -> dict[str, Any]:
        if self._response is None:
            raise CodexProtocolError(
                "Codex response stream ended before response.completed"
            )
        items = deepcopy(self._items)
        parts = deepcopy(self._parts)
        for key in self._deltas.keys() | self._done.keys():
            if key[:3] in self._done_parts and key[3] in parts[key[:3]]:
                continue
            text = self._done[key] if key in self._done else "".join(self._deltas[key])
            parts[key[:3]][key[3]] = text
        for (index, container, part_index), part in sorted(parts.items()):
            content = items[index].setdefault(container, [])
            while len(content) <= part_index:
                content.append({})
            # Added item skeletons precede deltas; completed item fields win.
            content[part_index] = (
                {**part, **content[part_index]}
                if index in self._done_items
                else {**content[part_index], **part}
            )
        for index in self._arguments.keys() | self._arguments_done.keys():
            value = (
                self._arguments_done[index]
                if index in self._arguments_done
                else "".join(self._arguments[index])
            )
            if not items[index].get("arguments"):
                items[index]["arguments"] = value
        item_ids = {item["id"]: index for index, item in items.items() if "id" in item}
        for index, item in enumerate(self._response.get("output", [])):
            index = item_ids.get(item.get("id"), index)
            items[index] = _merge_item(items.get(index, {}), item)
        return {**self._response, "output": [item for _, item in sorted(items.items())]}


def has_output(response: dict[str, Any]) -> bool:
    """Reasoning alone is not a final answer; accept all other native output types."""
    for item in response["output"]:
        if item["type"] == "reasoning":
            continue
        if item["type"] != "message":
            return True
        if any(
            part.get("text") or part.get("refusal") for part in item.get("content", [])
        ):
            return True
    return False
