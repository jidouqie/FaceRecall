from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ._client import get_client, llm_model, read_prompt
from .feature_matrix import FeatureMatrix


MIN_GUIDE_ANSWERS = 18
MAX_GUIDE_ANSWERS = 30


@dataclass
class Conflict:
    field: str
    values: list[str]
    resolution_question: str


@dataclass
class AnchorRequest:
    category: str
    n: int


@dataclass
class GuiderOutput:
    next_question: Optional[str]
    feature_matrix_delta: dict[str, Any] = field(default_factory=dict)
    conflicts_detected: list[Conflict] = field(default_factory=list)
    ready_to_generate: bool = False
    anchor_request: Optional[AnchorRequest] = None
    raw: str = ""


def _build_user_message(feature_matrix: FeatureMatrix, history: list[dict[str, str]]) -> str:
    fm_json = json.dumps(feature_matrix.to_json_dict(), ensure_ascii=False, indent=2)
    hist_lines: list[str] = []
    for h in history[-30:]:
        speaker = "目击者" if h["role"] == "user" else "你(上一轮)"
        hist_lines.append(f"{speaker}: {h['content']}")
    history_text = "\n".join(hist_lines) if hist_lines else "(尚无对话)"
    witness_count = sum(1 for h in history if h["role"] == "user")
    progress = (
        f"当前已回答问题数：{witness_count} 个（建议至少 {MIN_GUIDE_ANSWERS}，最多 {MAX_GUIDE_ANSWERS}）。"
        f"{'已达到建议问题数，但必须完成复述确认后才能生图。' if witness_count >= MIN_GUIDE_ANSWERS else f'还需至少 {MIN_GUIDE_ANSWERS - witness_count} 个问题。'}"
        f"{'⚠️ 即将到达上限，请尽快复述确认并收尾。' if witness_count >= 25 else ''}"
    )
    return (
        f"## 当前特征矩阵\n```json\n{fm_json}\n```\n\n"
        f"## 进度提示\n{progress}\n\n"
        f"## 对话历史\n{history_text}\n\n"
        f"## 你的任务\n根据上面的状态,输出符合规范的 JSON。只输出 JSON,不要 markdown 代码块包裹,不要任何说明。"
    )


def _parse_output(raw: str) -> GuiderOutput:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)

    conflicts = [
        Conflict(
            field=c.get("field", ""),
            values=list(c.get("values", [])),
            resolution_question=c.get("resolution_question", ""),
        )
        for c in (data.get("conflicts_detected") or [])
    ]
    anchor = None
    ar = data.get("anchor_request")
    if isinstance(ar, dict) and ar.get("category"):
        anchor = AnchorRequest(category=str(ar["category"]), n=int(ar.get("n", 4)))

    return GuiderOutput(
        next_question=data.get("next_question"),
        feature_matrix_delta=data.get("feature_matrix_delta") or {},
        conflicts_detected=conflicts,
        ready_to_generate=bool(data.get("ready_to_generate", False)),
        anchor_request=anchor,
        raw=raw,
    )


def call_guider(feature_matrix: FeatureMatrix, history: list[dict[str, str]]) -> GuiderOutput:
    client = get_client()
    system = read_prompt("guider_system.md")
    user = _build_user_message(feature_matrix, history)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = client.chat.completions.create(
        model=llm_model(),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    raw = resp.choices[0].message.content or ""
    try:
        return _parse_output(raw)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Retry once: some gateways occasionally wrap or truncate JSON despite response_format
        retry = client.chat.completions.create(
            model=llm_model(),
            messages=messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "刚才的输出不是合法 JSON。请只输出符合规范的 JSON 对象，不要任何 markdown 包裹。"},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        retry_raw = retry.choices[0].message.content or ""
        try:
            return _parse_output(retry_raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return GuiderOutput(
                next_question="抱歉，刚刚走神了。能再描述一下你印象最深的面部特征吗？",
                feature_matrix_delta={},
                conflicts_detected=[],
                ready_to_generate=False,
                anchor_request=None,
                raw=retry_raw or raw,
            )
