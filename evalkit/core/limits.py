"""How big one recorded value, and one span, may be.

Sizes here are UTF-8 bytes: what the receiver and the proxy in front of it count, and what a value costs
on the wire. A value over its limit keeps its start and its end around a ``…[cut N bytes]…`` marker,
because a conversation's oldest messages and its instructions are at the start of a request and its
newest ones at the end; the value's original size is recorded as ``evalkit.truncated.<field>``. The
TypeScript SDK cuts by the same rule, so a value reads alike from either.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# The most one value keeps by default: the most one ingest request may carry through the proxy.
DEFAULT_VALUE_LIMIT_BYTES = 16 * 1024 * 1024

TRUNCATED_PREFIX = "evalkit.truncated."

VALUE_LIMIT_ENV_VARS = ("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT")

# A value shorter than this is never cut while a span is fitted to the span limit: the long values are
# what make a span too large, and a short one cut leaves nothing readable.
_FIT_FLOOR = 1024


def resolve_value_limit(explicit: Optional[int]) -> int:
    """The per-value limit: the option when given, else the default, never more than the OpenTelemetry
    attribute value length limit when the environment sets one."""
    limit = DEFAULT_VALUE_LIMIT_BYTES
    if explicit is not None:
        try:
            if int(explicit) > 0:
                limit = int(explicit)
        except (TypeError, ValueError):
            pass
    for name in VALUE_LIMIT_ENV_VARS:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            env = int(raw)
        except ValueError:
            continue
        if env > 0:
            limit = min(limit, env)
            break
    return limit


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def exceeds(text: str, limit: int) -> bool:
    """Whether ``text`` is over ``limit`` bytes, encoding it only when its length cannot tell."""
    # One character is at least one UTF-8 byte and at most four.
    if len(text) > limit:
        return True
    if len(text) * 4 <= limit:
        return False
    return utf8_len(text) > limit


def _prefix_within(data: bytes, size: int) -> str:
    return data[:max(0, size)].decode("utf-8", "ignore")


def _suffix_within(data: bytes, size: int) -> str:
    if size <= 0:
        return ""
    return data[len(data) - size:].decode("utf-8", "ignore")


def _marker(cut: int) -> str:
    return f"…[cut {cut} bytes]…"


def cut_middle(text: str, limit: int) -> Tuple[str, Optional[int]]:
    """``text`` within ``limit`` bytes, and its original size in bytes when it had to be cut (else None)."""
    if not isinstance(text, str) or not exceeds(text, limit):
        return text, None
    data = text.encode("utf-8", "surrogatepass")
    total = len(data)
    # The marker is sized for the largest count it could name, so the result never exceeds the limit.
    keep = limit - len(_marker(total).encode("utf-8"))
    if keep <= 0:
        return _prefix_within(data, limit), total
    head = _prefix_within(data, (keep + 1) // 2)
    tail = _suffix_within(data, keep // 2)
    removed = total - utf8_len(head) - utf8_len(tail)
    return head + _marker(removed) + tail, total


def cut_end(text: str, limit: int) -> str:
    """``text`` within ``limit`` bytes, keeping its start and ending in ``…`` when cut."""
    if not isinstance(text, str) or not exceeds(text, limit):
        return text
    data = text.encode("utf-8", "surrogatepass")
    return _prefix_within(data, max(0, limit - 3)) + "…"


def _mark(attrs: Dict[str, Any], field: str, original: int) -> None:
    # The first cut names the value's size as it was recorded; a later fit of the same value must not
    # replace it with the size of an already shortened copy.
    attrs.setdefault(TRUNCATED_PREFIX + field, original)


def _is_marker_key(key: Any) -> bool:
    return isinstance(key, str) and key.startswith(TRUNCATED_PREFIX)


def cap_envelope_values(envelope: Any, limit: int) -> None:
    """Cut every string value of a span over ``limit`` bytes: prompt, completion, attributes, events."""
    if limit <= 0:
        return
    marks: Dict[str, int] = {}
    for name in ("prompt", "completion", "status_message"):
        value = getattr(envelope, name, None)
        if isinstance(value, str) and exceeds(value, limit):
            cut, original = cut_middle(value, limit)
            setattr(envelope, name, cut)
            if original is not None:
                marks["statusMessage" if name == "status_message" else name] = original
    attrs = envelope.attributes
    if isinstance(attrs, dict):
        for key, value in list(attrs.items()):
            if isinstance(value, str) and not _is_marker_key(key) and exceeds(value, limit):
                cut, original = cut_middle(value, limit)
                attrs[key] = cut
                if original is not None:
                    marks[str(key)] = original
    for event in (envelope.events or ()):
        event_attrs = getattr(event, "attributes", None)
        if not isinstance(event_attrs, dict):
            continue
        for key, value in list(event_attrs.items()):
            if isinstance(value, str) and not _is_marker_key(key) and exceeds(value, limit):
                cut, original = cut_middle(value, limit)
                event_attrs[key] = cut
                if original is not None:
                    _mark(event_attrs, str(key), original)
    if marks:
        if not isinstance(envelope.attributes, dict):
            envelope.attributes = {}
        for field, original in marks.items():
            _mark(envelope.attributes, field, original)


def _string_size(value: Any) -> int:
    return len(value) if isinstance(value, str) else 0


def _value_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value) + 2
    if isinstance(value, (list, tuple)):
        return 2 + sum(_value_size(v) + 1 for v in value)
    if isinstance(value, dict):
        return 2 + sum(len(str(k)) + 4 + _value_size(v) for k, v in value.items())
    return 16


def estimate_envelope_bytes(envelope: Any) -> int:
    """About how many bytes a span takes, from the lengths of its strings: cheap enough to run on every
    span, close enough to budget a queue by."""
    size = 384
    for name in ("prompt", "completion", "status_message", "operation"):
        size += _string_size(getattr(envelope, name, None))
    attrs = getattr(envelope, "attributes", None)
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            size += len(str(key)) + 4 + _value_size(value)
    for event in (getattr(envelope, "events", None) or ()):
        size += 64 + len(getattr(event, "name", "") or "")
        event_attrs = getattr(event, "attributes", None)
        if isinstance(event_attrs, dict):
            for key, value in event_attrs.items():
                size += len(str(key)) + 4 + _value_size(value)
    return size


class _Slot:
    """One string of a span that may be shortened to fit it, and where its cut is recorded."""

    __slots__ = ("container", "key", "marks", "field", "is_attr")

    def __init__(self, container: Any, key: str, marks: Optional[Dict[str, Any]], field: str, is_attr: bool) -> None:
        self.container = container
        self.key = key
        self.marks = marks
        self.field = field
        self.is_attr = is_attr

    def get(self) -> Any:
        if self.is_attr:
            return self.container.get(self.key)
        return getattr(self.container, self.key, None)

    def set(self, value: str) -> None:
        if self.is_attr:
            self.container[self.key] = value
        else:
            setattr(self.container, self.key, value)


def _dict_slots(d: Dict[str, Any]) -> Tuple[List[_Slot], Dict[str, Any]]:
    attrs = d.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    slots: List[_Slot] = []
    for key in ("prompt", "completion", "statusMessage"):
        if isinstance(d.get(key), str):
            slots.append(_Slot(d, key, attrs, key, True))
    for key, value in attrs.items():
        if isinstance(value, str) and not _is_marker_key(key):
            slots.append(_Slot(attrs, key, attrs, str(key), True))
    for event in (d.get("events") or ()):
        event_attrs = event.get("attributes") if isinstance(event, dict) else None
        if isinstance(event_attrs, dict):
            for key, value in event_attrs.items():
                if isinstance(value, str) and not _is_marker_key(key):
                    slots.append(_Slot(event_attrs, key, event_attrs, str(key), True))
    return slots, attrs


def _envelope_slots(envelope: Any) -> Tuple[List[_Slot], Dict[str, Any]]:
    attrs = envelope.attributes if isinstance(envelope.attributes, dict) else {}
    slots: List[_Slot] = []
    for name, field in (("prompt", "prompt"), ("completion", "completion"), ("status_message", "statusMessage")):
        if isinstance(getattr(envelope, name, None), str):
            slots.append(_Slot(envelope, name, attrs, field, False))
    for key, value in attrs.items():
        if isinstance(value, str) and not _is_marker_key(key):
            slots.append(_Slot(attrs, key, attrs, str(key), True))
    for event in (envelope.events or ()):
        event_attrs = getattr(event, "attributes", None)
        if isinstance(event_attrs, dict):
            for key, value in event_attrs.items():
                if isinstance(value, str) and not _is_marker_key(key):
                    slots.append(_Slot(event_attrs, key, event_attrs, str(key), True))
    return slots, attrs


def shrink_strings(slots: List[_Slot], excess: int) -> bool:
    """Take about ``excess`` bytes out of the longest strings, longest first, each keeping its start and
    end. Returns whether anything was cut."""
    sized = []
    for slot in slots:
        value = slot.get()
        if isinstance(value, str) and len(value) > _FIT_FLOOR:
            sized.append((utf8_len(value), slot))
    sized.sort(key=lambda pair: pair[0], reverse=True)
    changed = False
    for size, slot in sized:
        if excess <= 0:
            break
        target = max(_FIT_FLOOR, size - excess)
        if target >= size:
            continue
        cut, original = cut_middle(slot.get(), target)
        if original is None:
            continue
        slot.set(cut)
        if slot.marks is not None:
            _mark(slot.marks, slot.field, original)
        excess -= size - utf8_len(cut)
        changed = True
    return changed


def fit_envelope(envelope: Any, limit: int) -> int:
    """Shorten the longest strings of a span so that it comes to about ``limit`` bytes; returns the
    span's estimated size afterwards."""
    size = estimate_envelope_bytes(envelope)
    if size <= limit:
        return size
    slots, attrs = _envelope_slots(envelope)
    # A little under the limit, as the estimate counts characters and the encoded span adds escapes.
    shrink_strings(slots, size - limit + limit // 16)
    if attrs and envelope.attributes is not attrs:
        envelope.attributes = attrs
    return estimate_envelope_bytes(envelope)


def fit_dict(d: Dict[str, Any], excess: int) -> bool:
    """Shorten the longest strings of an encoded span by about ``excess`` bytes."""
    slots, attrs = _dict_slots(d)
    changed = shrink_strings(slots, excess)
    if changed and attrs and d.get("attributes") is not attrs:
        d["attributes"] = attrs
    return changed
