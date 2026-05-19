from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES: dict[str, str] = {
    "critic_system": "critic_system.txt",
    "supervisor_system": "supervisor_system.txt",
}

_PROMPT_CACHE: dict[str, tuple[int, str]] = {}
_PROMPT_CACHE_LOCK = RLock()


class PromptNotFoundError(KeyError):
    pass


class InvalidPromptContentError(ValueError):
    pass


def list_prompt_names() -> list[str]:
    return sorted(PROMPT_FILES.keys())


def resolve_prompt_path(prompt_name: str) -> Path:
    filename = PROMPT_FILES.get(prompt_name)
    if not filename:
        raise PromptNotFoundError(prompt_name)
    return PROMPT_DIR / filename


def get_prompt_metadata(prompt_name: str) -> dict:
    path = resolve_prompt_path(prompt_name)
    if not path.exists():
        raise PromptNotFoundError(prompt_name)
    stat = path.stat()
    return {
        "name": prompt_name,
        "filename": PROMPT_FILES[prompt_name],
        "updated_at": int(stat.st_mtime),
        "size": stat.st_size,
    }


def list_prompt_metadata() -> list[dict]:
    items: list[dict] = []
    for prompt_name in list_prompt_names():
        try:
            items.append(get_prompt_metadata(prompt_name))
        except PromptNotFoundError:
            logger.warning("Prompt file missing: %s", prompt_name)
    return items


def clear_prompt_cache() -> None:
    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE.clear()


def read_prompt(prompt_name: str, use_cache: bool = True) -> str:
    path = resolve_prompt_path(prompt_name)
    if not path.exists():
        raise PromptNotFoundError(prompt_name)

    stat = path.stat()
    version = stat.st_mtime_ns

    with _PROMPT_CACHE_LOCK:
        cached = _PROMPT_CACHE.get(prompt_name)
        if use_cache and cached and cached[0] == version:
            return cached[1]

    content = path.read_text(encoding="utf-8")
    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE[prompt_name] = (version, content)
    return content


def write_prompt(prompt_name: str, content: str) -> dict:
    normalized = str(content or "").replace("\r\n", "\n")
    if not normalized.strip():
        raise InvalidPromptContentError("Prompt content cannot be empty.")

    path = resolve_prompt_path(prompt_name)
    if not path.exists():
        raise PromptNotFoundError(prompt_name)

    path.write_text(normalized, encoding="utf-8")
    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE.pop(prompt_name, None)
    return get_prompt_metadata(prompt_name)