from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .feature_matrix import FeatureMatrix


SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"


class TurnRole(str, Enum):
    WITNESS = "witness"
    LLM = "llm"
    IMAGE = "image"


class CaseStatus(str, Enum):
    ACTIVE = "active"
    CONVERGED = "converged"
    ABANDONED = "abandoned"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Turn:
    id: str
    seq: int
    role: TurnRole
    content: str
    image_path: Optional[str] = None
    parent_image: Optional[str] = None
    model_name: Optional[str] = None
    timestamp: str = field(default_factory=_now)

    @classmethod
    def witness(cls, seq: int, content: str) -> Turn:
        return cls(id=_new_id(), seq=seq, role=TurnRole.WITNESS, content=content)

    @classmethod
    def llm(cls, seq: int, content: str) -> Turn:
        return cls(id=_new_id(), seq=seq, role=TurnRole.LLM, content=content)

    @classmethod
    def image(cls, seq: int, image_path: str, parent_image: Optional[str] = None,
              prompt: str = "", model_name: Optional[str] = None) -> Turn:
        return cls(
            id=_new_id(),
            seq=seq,
            role=TurnRole.IMAGE,
            content=prompt,
            image_path=image_path,
            parent_image=parent_image,
            model_name=model_name,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn:
        return cls(
            id=d["id"],
            seq=d["seq"],
            role=TurnRole(d["role"]),
            content=d["content"],
            image_path=d.get("image_path"),
            parent_image=d.get("parent_image"),
            model_name=d.get("model_name"),
            timestamp=d.get("timestamp", _now()),
        )


@dataclass
class Case:
    id: str
    title: str
    created_at: str
    status: CaseStatus
    turns: list[Turn] = field(default_factory=list)
    feature_matrix: FeatureMatrix = field(default_factory=FeatureMatrix.empty)
    final_image: Optional[str] = None

    @classmethod
    def new(cls, title: str) -> Case:
        return cls(
            id=_new_id(),
            title=title,
            created_at=_now(),
            status=CaseStatus.ACTIVE,
        )

    def append_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def next_seq(self) -> int:
        return (self.turns[-1].seq + 1) if self.turns else 0

    def last_image(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t.role == TurnRole.IMAGE and t.image_path:
                return t.image_path
        return None

    def last_image_by_model(self, model_name: str) -> Optional[str]:
        for t in reversed(self.turns):
            if t.role == TurnRole.IMAGE and t.image_path and t.model_name == model_name:
                return t.image_path
        return None

    def image_rounds(self) -> list[list[Turn]]:
        """把 IMAGE turn 按连续块分组,每块是一个生成回合。"""
        rounds: list[list[Turn]] = []
        current: list[Turn] = []
        for t in self.turns:
            if t.role == TurnRole.IMAGE:
                current.append(t)
            else:
                if current:
                    rounds.append(current)
                    current = []
        if current:
            rounds.append(current)
        return rounds

    def witness_history(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for t in self.turns:
            if t.role == TurnRole.WITNESS:
                out.append({"role": "user", "content": t.content})
            elif t.role == TurnRole.LLM:
                out.append({"role": "assistant", "content": t.content})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "status": self.status.value,
            "turns": [t.to_dict() for t in self.turns],
            "feature_matrix": self.feature_matrix.to_json_dict(),
            "final_image": self.final_image,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        return cls(
            id=d["id"],
            title=d["title"],
            created_at=d["created_at"],
            status=CaseStatus(d["status"]),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            feature_matrix=FeatureMatrix.from_json_dict(d.get("feature_matrix", {})),
            final_image=d.get("final_image"),
        )


def save_case(case: Case) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{case.id}.json"
    path.write_text(json.dumps(case.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_case(case_id: str) -> Case:
    path = SESSIONS_DIR / f"{case_id}.json"
    return Case.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_cases() -> list[dict[str, Any]]:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            case = load_case(f.stem)
            rounds = case.image_rounds()
            last_imgs = []
            if rounds:
                last_imgs = ["/images/" + Path(t.image_path).name for t in rounds[-1] if t.image_path]
            cases.append({
                "id": case.id,
                "title": case.title,
                "created_at": case.created_at,
                "status": case.status.value,
                "version_count": len(rounds),
                "thumbnails": last_imgs[:3],
            })
        except Exception:
            continue
    return cases
