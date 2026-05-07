from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = _env("FACERECALL_API_KEY") or _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set FACERECALL_API_KEY (preferred) or OPENAI_API_KEY in .env"
        )
    base_url = _env("FACERECALL_API_BASE") or _env("OPENAI_API_BASE")
    timeout = float(_env("FACERECALL_HTTP_TIMEOUT", "300") or "300")
    kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def llm_model() -> str:
    return _env("FACERECALL_LLM_MODEL", "gpt-4o") or "gpt-4o"


def image_model() -> str:
    return _env("FACERECALL_IMAGE_MODEL", "openai/gpt-5.4-image-2") or "openai/gpt-5.4-image-2"


def prompts_dir() -> str:
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent / "prompts")


def read_prompt(name: str) -> str:
    from pathlib import Path
    p = Path(prompts_dir()) / name
    return p.read_text(encoding="utf-8")
