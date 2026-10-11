from __future__ import annotations

import json
from typing import Any

# Content-block types that carry the message's readable text.
_TEXT_TYPES = frozenset({"text", "input_text", "output_text"})
# Blocks whose payload is binary or a pointer to it: named, never inlined.
_MEDIA_TYPES = frozenset({"image", "image_url", "input_image", "document", "file", "input_file", "audio", "input_audio"})
_MEDIA_KEYS = ("image", "document", "video")


def compact_json(value: Any) -> str:
    """JSON the way the rest of the pipeline writes it: no padding, non-ASCII kept as is."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_json_text(value: Any) -> Any:
    """A string that holds a JSON list or object, parsed; anything else as it came.

    OpenTelemetry attributes are scalars, so frameworks serialise message content to JSON text.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except ValueError:
        return value


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        text = getattr(block, "text", None)
        return text if isinstance(text, str) else ""
    kind = block.get("type")
    if (kind is None or kind in _TEXT_TYPES) and isinstance(block.get("text"), str):
        return block["text"]
    if kind in _TEXT_TYPES and isinstance(block.get("content"), str):
        return block["content"]
    if kind in _MEDIA_TYPES:
        return f"[{kind}]"
    media = next((key for key in _MEDIA_KEYS if key in block), None)
    if media:
        return f"[{media}]"
    try:
        return compact_json(block)
    except Exception:
        return ""


def message_content_text(content: Any) -> str:
    """A chat message's content as the text a person reads.

    Providers accept content as a string or as a list of blocks (OpenAI parts, Anthropic and Bedrock
    blocks, which is what agent frameworks such as Strands send). Text blocks become their text, a tool
    use or other structured block its JSON, and an image or file a ``[type]`` marker; formatting the
    list itself would store Python's repr of it, which no reader can parse.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(t for t in (_block_text(b) for b in content) if t)
    if isinstance(content, dict):
        return _block_text(content)
    return str(content)
