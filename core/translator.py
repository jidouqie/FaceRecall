from __future__ import annotations

import base64
from pathlib import Path

from ._client import get_client, llm_model, read_prompt


def _image_to_data_url(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def call_translator(previous_image: str, feedback: str) -> str:
    """把目击者反馈翻译成 GPT-image-2 edit prompt(英文字符串)。"""
    client = get_client()
    system = read_prompt("translator_system.md")
    image_url = _image_to_data_url(previous_image)
    resp = client.chat.completions.create(
        model=llm_model(),
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"目击者反馈:{feedback}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()
