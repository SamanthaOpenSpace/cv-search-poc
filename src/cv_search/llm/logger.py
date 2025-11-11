from __future__ import annotations
import json
import uuid
import threading
import contextvars
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cv_search.config.settings import REPO_ROOT

_RUN_DIR = contextvars.ContextVar("cv_search_llm_run_dir", default=None)
_DEFAULT_BASE = contextvars.ContextVar("cv_search_llm_default_base", default=None)
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

def set_run_dir(run_dir: str | Path):
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    return _RUN_DIR.set(str(p))

def reset_run_dir(token):
    _RUN_DIR.reset(token)

def _default_base_dir() -> Path:
    cached = _DEFAULT_BASE.get()
    if cached:
        path = Path(cached)
        path.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = REPO_ROOT / "runs" / timestamp / "llm"
    path.mkdir(parents=True, exist_ok=True)
    _DEFAULT_BASE.set(str(path))
    return path


def _resolve_log_path() -> Path:
    run_dir = _RUN_DIR.get()
    base = Path(run_dir) if run_dir else _default_base_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "llm.prompts.md"


def _format_section(title: str, body: str | None) -> str:
    if not body:
        return ""
    return f"### {title}\n\n{body.strip()}\n\n"


def _format_json_block(data: Any) -> str:
    if data in (None, {}, []):
        return ""
    return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```\n\n"


def _format_messages(messages: list[dict]) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content")
        parts.append(f"#### Message {index} – {role}\n")
        if isinstance(content, str):
            parts.append(content.strip() + "\n\n")
        else:
            parts.append(_format_json_block(content))
    return "".join(parts)


def _build_entry(entry: dict[str, Any]) -> str:
    header = f"## Chat {entry['id']}\n\n"
    metadata = [
        f"- Timestamp: {entry['ts']}",
        f"- Provider: {entry['provider']}",
        f"- Model: {entry['model']}",
    ]
    if entry.get("run_dir"):
        metadata.append(f"- Run Directory: {entry['run_dir']}")
    meta = entry.get("meta") or {}
    meta_block = _format_json_block(meta)
    usage_block = _format_json_block(entry.get("usage"))
    request_messages = _format_messages(entry["request"]["messages"])
    request_section = _format_section("Request", request_messages)
    response_section = _format_section(
        "Response",
        "```json\n" + entry["response"]["content"].strip() + "\n```",
    )
    pieces = [header, "\n".join(metadata) + "\n\n"]
    if meta_block:
        pieces.append(_format_section("Meta", meta_block))
    if request_section:
        pieces.append(request_section)
    if response_section:
        pieces.append(response_section)
    if usage_block:
        pieces.append(_format_section("Usage", usage_block))
    pieces.append("---\n\n")
    return "".join(pieces)

def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]

def log_chat(*, messages: list[dict], model: str, response_content: str, provider: str, usage: dict | None = None, meta: dict | None = None) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "meta": meta or {},
        "request": {"messages": messages},
        "response": {"content": response_content},
        "usage": usage,
        "run_dir": _RUN_DIR.get(),
        "id": str(uuid.uuid4()),
    }
    path = _resolve_log_path()
    lock = _lock_for(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_build_entry(entry))
