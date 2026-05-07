from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


CORE_FIELDS = (
    "memory_highlights",
    "observation_conditions",
    "age_range",
    "gender",
    "body_type",
    "expression",
    "face_shape",
    "face_proportions",
    "hair",
    "hairline",
    "eyebrows",
    "eyes",
    "eye_spacing",
    "glasses",
    "nose",
    "mouth",
    "chin",
)

ALL_FIELDS = CORE_FIELDS + (
    "forehead",
    "ears",
    "beard",
    "skin",
    "neck_shoulders",
    "summary_confirmation",
)


@dataclass
class FeatureValue:
    value: str
    confidence: Confidence

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureValue:
        return cls(value=str(d["value"]), confidence=Confidence(d["confidence"]))


@dataclass
class FeatureMatrix:
    fields: dict[str, FeatureValue] = field(default_factory=dict)
    distinctive_marks: list[FeatureValue] = field(default_factory=list)

    @classmethod
    def empty(cls) -> FeatureMatrix:
        return cls()

    def apply_delta(self, delta: dict[str, Any]) -> None:
        if not delta:
            return
        for k, v in delta.items():
            if v is None:
                continue
            if k == "distinctive_marks":
                items = v if isinstance(v, list) else [v]
                self.distinctive_marks = [
                    FeatureValue.from_dict(item) if isinstance(item, dict) else FeatureValue(str(item), Confidence.LOW)
                    for item in items
                    if item
                ]
                continue
            if isinstance(v, dict) and "value" in v and "confidence" in v:
                self.fields[k] = FeatureValue.from_dict(v)

    def core_fill_ratio(self) -> float:
        filled = sum(1 for k in CORE_FIELDS if k in self.fields)
        return filled / len(CORE_FIELDS)

    def to_image_prompt(self) -> str:
        """生成首版简笔画像 prompt(英文,GPT-image-2 能理解)。
        low confidence 的特征用 'possibly'/'roughly' 软化。"""
        parts: list[str] = []
        parts.append(
            "Create a simple black-and-white facial line drawing of a single person, "
            "front-facing, centered, plain white background. Use clean bold sketch lines, "
            "minimal detail, no color, no shading, no photorealistic texture. "
            "Emphasize the most recognizable facial features and silhouette so the face is easy to compare from memory."
        )

        order = [
            "memory_highlights",
            "age_range",
            "gender",
            "body_type",
            "expression",
            "face_shape",
            "face_proportions",
            "forehead",
            "hairline",
            "hair",
            "eyebrows",
            "eyes",
            "eye_spacing",
            "glasses",
            "nose",
            "mouth",
            "ears",
            "chin",
            "beard",
            "skin",
            "neck_shoulders",
        ]
        for k in order:
            if k not in self.fields:
                continue
            fv = self.fields[k]
            phrase = fv.value
            if fv.confidence == Confidence.LOW:
                phrase = f"possibly {phrase}"
            elif fv.confidence == Confidence.MEDIUM:
                phrase = f"roughly {phrase}"
            parts.append(f"{k.replace('_', ' ')}: {phrase}.")

        if self.distinctive_marks:
            marks = "; ".join(
                f"{m.value}" + (" (uncertain)" if m.confidence == Confidence.LOW else "")
                for m in self.distinctive_marks
            )
            parts.append(f"Distinctive marks: {marks}.")

        parts.append(
            "Keep it as a clear simple sketch, not a realistic photo. "
            "No background scene, no clothing details, no text, no watermark."
        )
        return " ".join(parts)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fields": {k: asdict(v) for k, v in self.fields.items()},
            "distinctive_marks": [asdict(m) for m in self.distinctive_marks],
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> FeatureMatrix:
        fm = cls()
        for k, v in (d.get("fields") or {}).items():
            fm.fields[k] = FeatureValue.from_dict(v)
        fm.distinctive_marks = [
            FeatureValue.from_dict(m) for m in (d.get("distinctive_marks") or [])
        ]
        return fm
