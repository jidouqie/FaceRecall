"""FaceRecall Phase 0 本地 Web 测试服务。

启动:
    cd prototype
    source .venv/bin/activate
    python server.py
然后打开 http://localhost:8787
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv, set_key as dotenv_set_key
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.session import Case, Turn, TurnRole, CaseStatus, save_case, load_case, list_cases
from core.feature_matrix import FeatureMatrix
from core.guider import call_guider
from core.translator import call_translator
from core.generator import (
    generate_all_models, generate_one, edit_one,
    get_image_models, model_label, is_chat_image_model,
)
from openai import OpenAI

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)
REALISTIC_MODEL_SLOT = "realistic-photo"
REALISTIC_PROMPT = (
    "Transform the reference simple black-and-white facial line drawing into a realistic, "
    "natural-looking front-facing head-and-shoulders portrait photo. Preserve the same face "
    "shape, facial proportions, hairstyle outline, eyebrow shape, eye shape and spacing, nose "
    "shape, mouth shape, chin, expression, and any distinctive marks from the sketch. Use a "
    "plain neutral background and even natural lighting. Do not beautify, stylize, change the "
    "person's identity, add accessories, add text, or add a watermark."
)

app = FastAPI()
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# jobs[case_id] = {model: {"status": "pending"|"ready"|"error", "path": str, "error": str}}
_jobs: dict[str, dict[str, dict]] = {}
_case_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _case_lock(case_id: str) -> threading.Lock:
    with _locks_lock:
        if case_id not in _case_locks:
            _case_locks[case_id] = threading.Lock()
        return _case_locks[case_id]


# ── request models ───────────────────────────────────────────────────────────

class NewCaseBody(BaseModel):
    title: str

class AnswerBody(BaseModel):
    answer: str

class FeedbackBody(BaseModel):
    feedback: str
    ref_image: str | None = None  # "/images/xxx.png" — 用户选中的基准图

class RealisticBody(BaseModel):
    ref_image: str

class EditTurnBody(BaseModel):
    seq: int
    new_answer: str

class RetryModelBody(BaseModel):
    model: str

class SettingsBody(BaseModel):
    api_base: str = ""
    api_key: str = ""
    llm_model: str = ""
    image_model: str = ""

class TestModelBody(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model: str
    model_type: str = "llm"   # "llm" | "image"

class ListModelsBody(BaseModel):
    api_base: str = ""
    api_key: str = ""


# ── helpers ──────────────────────────────────────────────────────────────────

def _guider_result(out, case: Case) -> dict:
    return {
        "question": out.next_question,
        "conflicts": [
            {"field": c.field, "question": c.resolution_question}
            for c in out.conflicts_detected
        ],
        "anchor_request": (
            {"category": out.anchor_request.category, "n": out.anchor_request.n}
            if out.anchor_request else None
        ),
        "fill_ratio": round(case.feature_matrix.core_fill_ratio(), 2),
        "ready": out.ready_to_generate,
    }


def _img_url(path: str) -> str:
    return "/images/" + Path(path).name


def _job_snapshot(case_id: str) -> list[dict]:
    job = _jobs.get(case_id, {})
    models = job.get("_models") or get_image_models()
    out = []
    for m in models:
        slot = job.get(m, {"status": "idle"})
        out.append({
            "model": m,
            "label": model_label(m),
            "status": slot.get("status", "idle"),
            "image_url": _img_url(slot["path"]) if slot.get("path") else None,
            "error": slot.get("error"),
        })
    return out


def _all_settled(case_id: str) -> bool:
    job = _jobs.get(case_id, {})
    models = job.get("_models") or get_image_models()
    return all(job.get(m, {}).get("status") in ("ready", "error") for m in models)


def _current_version(case: Case) -> int:
    return len(case.image_rounds()) + 1


def _is_case_image_path(path: str | None) -> bool:
    if not path:
        return False
    try:
        return Path(path).resolve().parent == IMAGES_DIR.resolve()
    except Exception:
        return False


def _delete_case_images(paths: set[str]) -> None:
    for path in paths:
        if not _is_case_image_path(path):
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _clear_generation_history(case: Case) -> set[str]:
    """Keep Q&A and feature matrix; remove generated images and later feedback."""
    image_paths = {
        t.image_path
        for t in case.turns
        if t.role == TurnRole.IMAGE and t.image_path
    }
    first_image_index = next(
        (i for i, t in enumerate(case.turns) if t.role == TurnRole.IMAGE),
        None,
    )
    if first_image_index is None:
        return set()

    cut = first_image_index
    if cut > 0:
        prev = case.turns[cut - 1]
        if prev.role == TurnRole.LLM and "生成" in prev.content:
            cut -= 1
    case.turns = case.turns[:cut]
    case.final_image = None
    case.status = CaseStatus.ACTIVE
    return {p for p in image_paths if p}


def _job_is_current(case_id: str, generation_id: str) -> bool:
    return _jobs.get(case_id, {}).get("_generation_id") == generation_id


def _commit_image_result(case_id: str, generation_id: str, model: str,
                         image_path: str, ref_path: str | None, prompt: str) -> None:
    """Atomically persist a generated image and update job status, gated by generation_id."""
    lock = _case_lock(case_id)
    with lock:
        if not _job_is_current(case_id, generation_id):
            return
        c = load_case(case_id)
        c.append_turn(Turn.image(c.next_seq(), image_path,
                                 parent_image=ref_path,
                                 prompt=prompt, model_name=model))
        save_case(c)
        _jobs[case_id][model] = {"status": "ready", "path": image_path}


def _commit_image_error(case_id: str, generation_id: str, model: str, error: str) -> None:
    lock = _case_lock(case_id)
    with lock:
        if not _job_is_current(case_id, generation_id):
            return
        _jobs[case_id][model] = {"status": "error", "error": error}


def _resolve_case_image(case: Case, url: str) -> str:
    """Resolve an /images/X.png URL to absolute path, validating it belongs to this case."""
    if not url:
        raise HTTPException(400, "缺少图片引用")
    name = Path(url).name
    candidate = str(IMAGES_DIR / name)
    valid = {t.image_path for t in case.turns if t.role == TurnRole.IMAGE and t.image_path}
    if candidate not in valid:
        raise HTTPException(400, f"图片不属于当前会话：{url}")
    if not Path(candidate).exists():
        raise HTTPException(400, f"图片文件不存在：{url}")
    return candidate


def _ensure_no_active_job(case_id: str) -> None:
    job = _jobs.get(case_id, {})
    models = job.get("_models") or []
    if any(job.get(m, {}).get("status") == "pending" for m in models):
        raise HTTPException(409, "上一次生成尚未结束，请稍候")


def _trigger_regeneration(case: Case, prompt: str) -> None:
    """Clear past generation history and start a fresh generation. Caller must hold case lock."""
    old_image_paths = _clear_generation_history(case)
    _delete_case_images(old_image_paths)
    case.append_turn(Turn.llm(case.next_seq(), "正在重新生成 2 张候选画像,请稍候……"))
    save_case(case)
    _start_generate(case, prompt)


# ── background tasks ─────────────────────────────────────────────────────────

def _start_generate(case: Case, prompt: str, ref_per_model: dict[str, str] | None = None) -> None:
    case_id = case.id
    active_models = get_image_models()
    expected_version = len(case.image_rounds()) + 1
    generation_id = uuid.uuid4().hex
    _jobs[case_id] = {m: {"status": "pending"} for m in active_models}
    _jobs[case_id]["_version"] = expected_version
    _jobs[case_id]["_prompt"] = prompt
    _jobs[case_id]["_ref_per_model"] = ref_per_model or {}
    _jobs[case_id]["_models"] = active_models
    _jobs[case_id]["_generation_id"] = generation_id

    def _on_done(model: str, result) -> None:
        if isinstance(result, Exception):
            _commit_image_error(case_id, generation_id, model, str(result))
            return
        prev = ref_per_model.get(model) if ref_per_model else None
        _commit_image_result(case_id, generation_id, model, result, prev, prompt)

    def _run() -> None:
        generate_all_models(prompt, ref_per_model=ref_per_model,
                            on_model_done=_on_done, models=active_models)

    threading.Thread(target=_run, daemon=True).start()


def _start_realistic(case: Case, ref_path: str) -> None:
    case_id = case.id
    generation_id = uuid.uuid4().hex
    expected_version = len(case.image_rounds()) + 1
    api_model = get_image_models()[0]
    _jobs[case_id] = {REALISTIC_MODEL_SLOT: {"status": "pending"}}
    _jobs[case_id]["_version"] = expected_version
    _jobs[case_id]["_prompt"] = REALISTIC_PROMPT
    _jobs[case_id]["_ref_per_model"] = {REALISTIC_MODEL_SLOT: ref_path}
    _jobs[case_id]["_models"] = [REALISTIC_MODEL_SLOT]
    _jobs[case_id]["_generation_id"] = generation_id

    def _run() -> None:
        try:
            result = edit_one(api_model, REALISTIC_PROMPT, ref_path)
        except Exception as e:
            _commit_image_error(case_id, generation_id, REALISTIC_MODEL_SLOT, str(e))
            return
        _commit_image_result(case_id, generation_id, REALISTIC_MODEL_SLOT,
                             result, ref_path, REALISTIC_PROMPT)

    threading.Thread(target=_run, daemon=True).start()


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "app.html")


@app.get("/cases")
def get_cases():
    return list_cases()


@app.post("/case/new")
def new_case(body: NewCaseBody):
    case = Case.new(title=body.title.strip() or "未命名")
    out = call_guider(case.feature_matrix, [])
    case.feature_matrix.apply_delta(out.feature_matrix_delta)
    if out.next_question:
        case.append_turn(Turn.llm(case.next_seq(), out.next_question))
    save_case(case)
    return {"case_id": case.id, **_guider_result(out, case)}


@app.post("/case/{case_id}/answer")
def submit_answer(case_id: str, body: AnswerBody):
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "会话不存在")

    case.append_turn(Turn.witness(case.next_seq(), body.answer.strip()))
    witness_seq = case.turns[-1].seq
    out = call_guider(case.feature_matrix, case.witness_history())
    case.feature_matrix.apply_delta(out.feature_matrix_delta)

    fill = case.feature_matrix.core_fill_ratio()
    witness_count = sum(1 for t in case.turns if t.role.value == "witness")

    # 硬顶 30 题：超过则强制生图，不管 LLM 怎么说
    force_by_limit = witness_count >= 30
    # 正常触发：必须由引导器完成复述确认后明确放行
    normal_trigger = out.ready_to_generate

    if force_by_limit or normal_trigger:
        msg = "好的,面部特征描述已足够详细。正在生成 2 张初版画像,请稍候……"
        case.append_turn(Turn.llm(case.next_seq(), msg))
        save_case(case)
        prompt = case.feature_matrix.to_image_prompt()
        _start_generate(case, prompt)
        return {"status": "generating", "message": msg, "fill_ratio": round(fill, 2), "witness_seq": witness_seq}

    if out.next_question:
        case.append_turn(Turn.llm(case.next_seq(), out.next_question))
    save_case(case)
    return {"status": "question", "witness_seq": witness_seq, **_guider_result(out, case)}


@app.get("/case/{case_id}/poll")
def poll(case_id: str):
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "会话不存在")

    results = _job_snapshot(case_id)
    settled = _all_settled(case_id)
    any_ready = any(r["status"] == "ready" for r in results)
    # Use tracked version so frontend gets consistent ID before images are saved
    version = _jobs.get(case_id, {}).get("_version") or len(case.image_rounds())

    return {
        "settled": settled,
        "any_ready": any_ready,
        "version": version,
        "results": results,
    }


@app.post("/case/{case_id}/feedback")
def submit_feedback(case_id: str, body: FeedbackBody):
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "会话不存在")

    if not body.feedback.strip():
        final_image = case.last_image()
        if body.ref_image:
            final_image = _resolve_case_image(case, body.ref_image)
        case.status = CaseStatus.CONVERGED
        case.final_image = final_image
        save_case(case)
        return {
            "status": "converged",
            "final_image_url": _img_url(final_image) if final_image else None,
        }

    _ensure_no_active_job(case_id)

    # 确定基准图：用户选中的那张，所有模型都从它出发
    if body.ref_image:
        ref_path = _resolve_case_image(case, body.ref_image)
    else:
        ref_path = case.last_image()
    if not ref_path:
        raise HTTPException(400, "还没有图可以编辑")

    case.append_turn(Turn.witness(case.next_seq(), body.feedback.strip()))
    save_case(case)

    active_models = get_image_models()
    generation_id = uuid.uuid4().hex

    # Translate once using the selected ref, then edit all models from that same ref
    def _run_edit() -> None:
        try:
            edit_prompt = call_translator(previous_image=ref_path, feedback=body.feedback.strip())
        except Exception as e:
            for m in active_models:
                _commit_image_error(case_id, generation_id, m, f"翻译失败:{e}")
            return

        ref_per_model = {model: ref_path for model in active_models}

        def _on_done(model: str, result) -> None:
            if isinstance(result, Exception):
                _commit_image_error(case_id, generation_id, model, str(result))
                return
            _commit_image_result(case_id, generation_id, model, result, ref_path, edit_prompt)

        generate_all_models(edit_prompt, ref_per_model=ref_per_model,
                            on_model_done=_on_done, models=active_models)

    _jobs[case_id] = {m: {"status": "pending"} for m in active_models}
    _jobs[case_id]["_version"] = len(case.image_rounds()) + 1
    _jobs[case_id]["_models"] = active_models
    _jobs[case_id]["_generation_id"] = generation_id
    threading.Thread(target=_run_edit, daemon=True).start()
    return {"status": "generating"}


@app.post("/case/{case_id}/realistic")
def generate_realistic(case_id: str, body: RealisticBody):
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "会话不存在")

    ref_path = _resolve_case_image(case, body.ref_image)
    _ensure_no_active_job(case_id)

    _start_realistic(case, ref_path)
    return {"status": "generating", "version": _jobs[case_id]["_version"]}


@app.post("/case/{case_id}/edit-turn")
def edit_turn(case_id: str, body: EditTurnBody):
    """修改某条目击者回答，重建特征矩阵；若已有画像则自动重新生成。"""
    _ensure_no_active_job(case_id)
    lock = _case_lock(case_id)
    with lock:
        try:
            case = load_case(case_id)
        except FileNotFoundError:
            raise HTTPException(404, "会话不存在")

        target = next(
            (t for t in case.turns if t.seq == body.seq and t.role == TurnRole.WITNESS), None
        )
        if not target:
            raise HTTPException(404, "找不到该回答")

        target.content = body.new_answer.strip()

        # Rebuild feature matrix with one guider call over the full (updated) history
        case.feature_matrix = FeatureMatrix.empty()
        out = call_guider(case.feature_matrix, case.witness_history())
        case.feature_matrix.apply_delta(out.feature_matrix_delta)
        save_case(case)

        if case.image_rounds():
            prompt = case.feature_matrix.to_image_prompt()
            _trigger_regeneration(case, prompt)
            return {"status": "regenerating", "fill_ratio": round(case.feature_matrix.core_fill_ratio(), 2)}

    return {
        "status": "ok",
        "fill_ratio": round(case.feature_matrix.core_fill_ratio(), 2),
        **_guider_result(out, case),
    }


@app.post("/case/{case_id}/retry-model")
def retry_model_endpoint(case_id: str, body: RetryModelBody):
    """单独重试一个超时或出错的模型，不影响其他模型状态。"""
    if body.model not in get_image_models():
        raise HTTPException(400, "无效模型")
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "会话不存在")

    job = _jobs.get(case_id) or {}
    prompt = job.get("_prompt") or case.feature_matrix.to_image_prompt()
    if not prompt:
        raise HTTPException(400, "找不到生成参数，请点「重新生成」")

    ref_per_model: dict[str, str] = job.get("_ref_per_model") or {}
    ref = ref_per_model.get(body.model)
    model = body.model
    generation_id = job.get("_generation_id")
    if not generation_id:
        # No tracked generation context — start a fresh one for this retry
        generation_id = uuid.uuid4().hex
        _jobs[case_id] = {
            "_prompt": prompt,
            "_ref_per_model": ref_per_model,
            "_version": len(case.image_rounds()) + 1,
            "_models": [model],
            "_generation_id": generation_id,
        }
    _jobs[case_id][model] = {"status": "pending"}

    def _run():
        try:
            path = edit_one(model, prompt, ref) if ref else generate_one(model, prompt)
        except Exception as e:
            _commit_image_error(case_id, generation_id, model, str(e))
            return
        _commit_image_result(case_id, generation_id, model, path, ref, prompt)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "retrying"}


@app.post("/case/{case_id}/regenerate")
def regenerate(case_id: str):
    """清空旧图片/反馈历史，复用当前 feature_matrix 重新抽 2 张初版候选图。"""
    lock = _case_lock(case_id)
    with lock:
        try:
            case = load_case(case_id)
        except FileNotFoundError:
            raise HTTPException(404, "会话不存在")

        prompt = case.feature_matrix.to_image_prompt()
        if not prompt.strip():
            raise HTTPException(400, "特征还不足以生成画像，请先回答问题")

        _trigger_regeneration(case, prompt)
    return {"status": "generating", "version": _jobs[case_id]["_version"]}


@app.get("/case/{case_id}/history")
def case_history(case_id: str):
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "会话不存在")

    rounds = case.image_rounds()
    version_data = []
    for i, rnd in enumerate(rounds, 1):
        imgs = []
        for t in rnd:
            if t.image_path:
                imgs.append({
                    "model": t.model_name or "unknown",
                    "label": model_label(t.model_name or "未知"),
                    "image_url": _img_url(t.image_path),
                })
        version_data.append({"version": i, "images": imgs})

    # Chat turns for display (exclude image turns); include seq for witness (needed for edit)
    chat_turns = [
        {"role": t.role.value, "content": t.content or "", "seq": t.seq}
        for t in case.turns
        if t.role.value in ("llm", "witness") and t.content
    ]

    # Stage: "feedback" if images exist, "qa" otherwise (mid-Q&A or stuck-generating)
    stage = "feedback" if rounds else "qa"

    return {
        "id": case.id,
        "title": case.title,
        "status": case.status.value,
        "created_at": case.created_at,
        "versions": version_data,
        "chat_turns": chat_turns,
        "stage": stage,
        "final_image_url": _img_url(case.final_image) if case.final_image else None,
    }


@app.get("/settings")
def get_settings():
    return {
        "api_base": os.environ.get("FACERECALL_API_BASE", ""),
        "api_key": os.environ.get("FACERECALL_API_KEY", ""),
        "llm_model": os.environ.get("FACERECALL_LLM_MODEL", "gpt-4o"),
        "image_model": get_image_models()[0],
    }


@app.post("/settings")
def save_settings(body: SettingsBody):
    updates = {
        "FACERECALL_API_BASE": body.api_base.strip(),
        "FACERECALL_API_KEY": body.api_key.strip(),
        "FACERECALL_LLM_MODEL": body.llm_model.strip(),
        "FACERECALL_IMAGE_MODEL": body.image_model.strip(),
    }
    for k, v in updates.items():
        dotenv_set_key(str(ENV_FILE), k, v)
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]
    from core._client import get_client
    get_client.cache_clear()
    return {"status": "ok"}


@app.post("/settings/list-models")
def list_models(body: ListModelsBody):
    """Fetch available models from the configured API endpoint."""
    try:
        api_key = body.api_key.strip() or os.environ.get("FACERECALL_API_KEY", "")
        api_base = body.api_base.strip() or os.environ.get("FACERECALL_API_BASE") or None
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=10)
        models = client.models.list()
        ids = sorted(m.id for m in models.data)
        return {"models": ids}
    except Exception as e:
        raise HTTPException(400, f"拉取失败：{e}")


@app.post("/settings/test-model")
def test_model(body: TestModelBody):
    """Quick connectivity test for a single model."""
    try:
        api_key = body.api_key.strip() or os.environ.get("FACERECALL_API_KEY", "")
        api_base = body.api_base.strip() or os.environ.get("FACERECALL_API_BASE") or None
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=15)
        if body.model_type == "llm":
            resp = client.chat.completions.create(
                model=body.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return {"ok": True, "info": f"✅ 连通 ({resp.model})"}
        else:
            # For image models, verify via model list (avoids image generation cost)
            model_ids = [m.id for m in client.models.list().data]
            if body.model in model_ids:
                return {"ok": True, "info": "✅ 模型存在"}
            # If not in list, attempt a minimal chat test for gemini-style models
            if is_chat_image_model(body.model):
                client.chat.completions.create(
                    model=body.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    timeout=15,
                )
                return {"ok": True, "info": "✅ 连通"}
            return {"ok": False, "info": f"⚠️ 模型不在列表中"}
    except Exception as e:
        return {"ok": False, "info": f"❌ {str(e)[:80]}"}


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    def _open_browser():
        import time; time.sleep(1.2)
        webbrowser.open("http://localhost:8787")
    threading.Thread(target=_open_browser, daemon=True).start()
    print("FaceRecall  →  http://localhost:8787")
    uvicorn.run("server:app", host="127.0.0.1", port=8787, reload=False)
