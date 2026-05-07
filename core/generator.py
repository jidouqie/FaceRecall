from __future__ import annotations

import base64
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Union

import os

from ._client import get_client, image_model


IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
DEFAULT_SIZE = "1024x1024"

# Static default – used as fallback when env vars are not set.
DEFAULT_IMAGE_MODEL = "openai/gpt-5.4-image-2"
IMAGE_VARIANT_COUNT = 2
MODEL_SLOT_SEPARATOR = "#"
_DEFAULT_IMAGE_MODELS: list[str] = [
    DEFAULT_IMAGE_MODEL if i == 1 else f"{DEFAULT_IMAGE_MODEL}{MODEL_SLOT_SEPARATOR}{i}"
    for i in range(1, IMAGE_VARIANT_COUNT + 1)
]

# Kept for backward compatibility with any code that imports IMAGE_MODELS
IMAGE_MODELS: list[str] = _DEFAULT_IMAGE_MODELS

_STATIC_MODEL_LABELS: dict[str, str] = {
    "openai/gpt-5.4-image-2": "GPT-5.4 Image",
    "openai/gpt-5-image": "GPT-5 Image",
    "openai/gpt-5-image-mini": "GPT-5 Image Mini",
    "google/gemini-3-pro-image-preview": "Gemini 3 Pro Image",
    "google/gemini-3.1-flash-image-preview": "Gemini 3.1 Flash Image",
    "google/gemini-2.5-flash-image": "Gemini 2.5 Flash Image",
    "realistic-photo": "真人画像",
}


def get_image_models() -> list[str]:
    """Return two generation slots backed by the same active image model."""
    model = (
        os.environ.get("FACERECALL_IMAGE_MODEL")
        or os.environ.get("FACERECALL_IMAGE_MODEL_1")
        or DEFAULT_IMAGE_MODEL
    ).strip()
    model = model or DEFAULT_IMAGE_MODEL
    return [
        model if i == 1 else f"{model}{MODEL_SLOT_SEPARATOR}{i}"
        for i in range(1, IMAGE_VARIANT_COUNT + 1)
    ]


def _api_model_name(model: str) -> str:
    return model.split(MODEL_SLOT_SEPARATOR, 1)[0]


def model_label(model: str) -> str:
    base, _, slot = model.partition(MODEL_SLOT_SEPARATOR)
    label = _STATIC_MODEL_LABELS.get(base, base)
    return f"{label} #{slot}" if slot else label


def MODEL_LABELS_dynamic() -> dict[str, str]:
    return {m: model_label(m) for m in get_image_models()}


# Alias kept for old code
MODEL_LABELS = _STATIC_MODEL_LABELS


def is_chat_image_model(model: str) -> bool:
    """Models that generate images via chat completions (not images.generate).

    OpenRouter-style names contain a '/' (provider/model) and always go through
    chat/completions — OpenRouter does not expose the images.generate endpoint.
    Pure OpenAI model names (no '/') use images.generate unless they're Gemini.
    """
    model = _api_model_name(model)
    if "/" in model:          # OpenRouter provider/model format
        return True
    return "gemini" in model.lower()


def _detect_image_suffix(b64: str) -> str:
    """Sniff image format from base64 magic bytes."""
    try:
        head = base64.b64decode(b64[:64], validate=False)
    except Exception:
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and b"WEBP" in head[:16]:
        return "webp"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return "png"


def _save_b64(b64: str, suffix: str | None = None) -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if suffix is None:
        suffix = _detect_image_suffix(b64)
    name = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.{suffix}"
    path = IMAGES_DIR / name
    path.write_bytes(base64.b64decode(b64))
    return str(path)


def _save_url(url: str) -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.png"
    path = IMAGES_DIR / name
    urllib.request.urlretrieve(url, str(path))
    return str(path)


def _save_response(data) -> str:
    b64 = getattr(data, "b64_json", None)
    if b64:
        return _save_b64(b64)
    url = getattr(data, "url", None)
    if url:
        return _save_url(url)
    raise RuntimeError("Image API returned neither b64_json nor url")


def _extract_from_message(message) -> str:
    """Extract image from a chat completion message.

    Handles three formats:
    1. OpenRouter: image in message.model_extra['images'][*]['image_url']['url']
    2. Inline data-URL or plain image URL in message.content (text)
    3. Structured content list with image_url parts
    """
    # 1. OpenRouter-specific: images in model_extra
    extra = getattr(message, "model_extra", None) or {}
    for img in extra.get("images") or []:
        url = (img.get("image_url") or {}).get("url", "")
        if url.startswith("data:image/"):
            m = re.match(r'data:image/(\w+);base64,([A-Za-z0-9+/=\n]+)', url)
            if m:
                return _save_b64(m.group(2).replace("\n", ""), m.group(1).lower())
        elif url:
            return _save_url(url)

    content = message.content

    # 2. Structured content list (some OpenAI-compatible APIs)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:image/"):
                    m = re.match(r'data:image/(\w+);base64,([A-Za-z0-9+/=\n]+)', url)
                    if m:
                        return _save_b64(m.group(2).replace("\n", ""), m.group(1).lower())
                elif url:
                    return _save_url(url)
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

    text = content or ""

    # 3. Inline data-URL or plain URL in text
    m2 = re.search(r'data:image/(\w+);base64,([A-Za-z0-9+/=\n]+)', text)
    if m2:
        return _save_b64(m2.group(2).replace("\n", ""), m2.group(1).lower())
    url_m = re.search(r'https?://\S+\.(?:png|jpg|jpeg|webp)', text)
    if url_m:
        return _save_url(url_m.group(0))

    raise RuntimeError(f"No image found in response. content={repr(text[:200])}, extra_keys={list(extra.keys())}")


def _img_to_data_url(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


# ── per-model generate / edit ────────────────────────────────────────────────

IMAGE_TIMEOUT = 300  # seconds per image call (gpt-image-2 can queue up to ~4 min)


def generate_one(model: str, prompt: str) -> str:
    client = get_client()
    api_model = _api_model_name(model)
    if is_chat_image_model(model):
        resp = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=IMAGE_TIMEOUT,
        )
        return _extract_from_message(resp.choices[0].message)
    else:
        resp = client.images.generate(
            model=api_model, prompt=prompt, size=DEFAULT_SIZE, n=1, timeout=IMAGE_TIMEOUT,
        )
        return _save_response(resp.data[0])


def edit_one(model: str, prompt: str, ref_image: str) -> str:
    client = get_client()
    api_model = _api_model_name(model)
    if is_chat_image_model(model):
        data_url = _img_to_data_url(ref_image)
        resp = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            timeout=IMAGE_TIMEOUT,
        )
        return _extract_from_message(resp.choices[0].message)
    else:
        with open(ref_image, "rb") as f:
            resp = client.images.edit(
                model=api_model, image=f, prompt=prompt, size=DEFAULT_SIZE, n=1, timeout=IMAGE_TIMEOUT,
            )
        return _save_response(resp.data[0])


# ── multi-model parallel ─────────────────────────────────────────────────────

def generate_all_models(
    prompt: str,
    ref_per_model: dict[str, str] | None = None,
    on_model_done: Callable[[str, Union[str, Exception]], None] | None = None,
    models: list[str] | None = None,
) -> dict[str, Union[str, Exception]]:
    """Run all image models in parallel. Each model's result is reported via
    `on_model_done(model, result)` AS SOON AS it finishes — so callers can update
    UI state per-model without waiting for the slowest one. The returned dict is
    only complete after all models finish (kept for backward compatibility)."""
    active_models = models if models is not None else get_image_models()
    results: dict[str, Union[str, Exception]] = {}
    lock = threading.Lock()

    def _run(model: str) -> None:
        try:
            if ref_per_model and model in ref_per_model:
                path = edit_one(model, prompt, ref_per_model[model])
            else:
                path = generate_one(model, prompt)
            outcome: Union[str, Exception] = path
        except Exception as e:
            outcome = e
        with lock:
            results[model] = outcome
        if on_model_done is not None:
            try:
                on_model_done(model, outcome)
            except Exception:
                pass

    threads = [threading.Thread(target=_run, args=(m,), daemon=True) for m in active_models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ── backward-compat wrappers (cli.py) ───────────────────────────────────────

def text_to_image(prompt: str) -> str:
    return generate_one(image_model(), prompt)


def edit_with_refs(prompt: str, ref_images: list[str]) -> str:
    if not ref_images:
        raise ValueError("edit_with_refs requires at least one reference image")
    return edit_one(image_model(), prompt, ref_images[-1])
