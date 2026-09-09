from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MalformedFile(Exception):
    pass


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedFile(f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}); fix it before installing") from exc
    if not isinstance(data, dict):
        raise MalformedFile(f"{path} must contain a JSON object")
    return data


def write_atomic(path: Path, text: str, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        elif path.exists():
            os.chmod(tmp, os.stat(path).st_mode & 0o777)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, data: Dict[str, Any]) -> None:
    write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def set_nested(data: Dict[str, Any], keys: List[str], value: Any) -> None:
    cursor = data
    for key in keys[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[keys[-1]] = value


def get_nested(data: Dict[str, Any], keys: List[str]) -> Tuple[bool, Any]:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return False, None
        cursor = cursor[key]
    return True, cursor


def delete_nested(data: Dict[str, Any], keys: List[str]) -> None:
    parents: List[Dict[str, Any]] = []
    cursor: Any = data
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            return
        parents.append(cursor)
        cursor = cursor[key]
    if isinstance(cursor, dict):
        cursor.pop(keys[-1], None)
    for parent, key in zip(reversed(parents), reversed(keys[:-1])):
        if isinstance(parent.get(key), dict) and not parent[key]:
            del parent[key]


def replace_marker_block(text: str, begin: str, end: str, block: Optional[str]) -> str:
    lines = text.splitlines()
    start = stop = -1
    for index, line in enumerate(lines):
        if line.strip() == begin and start < 0:
            start = index
        elif line.strip() == end and start >= 0:
            stop = index
            break
    if start >= 0 and stop >= 0:
        del lines[start : stop + 1]
        if start > 0 and lines[start - 1].strip() == "" and (start >= len(lines) or lines[start].strip() == ""):
            del lines[start - 1]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if block is not None:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.extend([begin, *block.splitlines(), end])
    out = "\n".join(lines)
    return out + "\n" if out else ""


def has_toml_table_outside(text: str, table: str, begin: str, end: str) -> bool:
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == begin:
            inside = True
        elif stripped == end:
            inside = False
        elif not inside and stripped.startswith(f"[{table}]"):
            return True
    return False
