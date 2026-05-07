"""FaceRecall Phase 0 命令行原型。

跑法:
    python cli.py --case-name "测试1"

环境变量(.env 或 shell):
    FACERECALL_API_BASE   老板网关或 https://api.openai.com/v1
    FACERECALL_API_KEY    API key
    FACERECALL_LLM_MODEL  默认 gpt-4o
    FACERECALL_IMAGE_MODEL 默认 gpt-image-2-all
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from core.feature_matrix import FeatureMatrix
from core.guider import call_guider
from core.translator import call_translator
from core.generator import text_to_image, edit_with_refs
from core.session import Case, Turn, CaseStatus, save_case


load_dotenv(Path(__file__).resolve().parent / ".env")

console = Console()

MAX_GUIDER_TURNS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceRecall Phase 0 prototype")
    p.add_argument("--case-name", required=True, help="本次画像的名字(自取)")
    return p.parse_args()


def run_guidance(case: Case) -> None:
    """阶段 1:引导问答直到 ready_to_generate。"""
    fm = case.feature_matrix
    for _ in range(MAX_GUIDER_TURNS):
        try:
            out = call_guider(feature_matrix=fm, history=case.witness_history())
        except Exception as e:
            console.print(f"[red]Guider 调用失败:{e}[/red]")
            raise

        fm.apply_delta(out.feature_matrix_delta)

        for c in out.conflicts_detected:
            console.print(Panel(
                f"字段:{c.field}\n冲突值:{c.values}\n追问:{c.resolution_question}",
                title="检测到冲突",
                border_style="red",
            ))

        if out.anchor_request:
            console.print(
                f"[yellow]建议给目击者看 {out.anchor_request.category} 类参考图(n={out.anchor_request.n})。"
                f" Phase 0 暂不实接 AnchorBank,请目击者继续口述。[/yellow]"
            )

        if out.ready_to_generate:
            console.print(f"[green]✓ 引导阶段完成,核心维度填充率:{fm.core_fill_ratio():.0%}[/green]")
            return

        question = out.next_question or "请继续描述这张面孔的特征。"
        case.append_turn(Turn.llm(case.next_seq(), question))
        console.print(Panel(question, title=f"问 (回合 {case.next_seq()})", border_style="blue"))
        try:
            answer = input("答: ").strip()
        except EOFError:
            answer = ""
        if not answer:
            console.print("[yellow]空回答,提前结束引导阶段。[/yellow]")
            return
        case.append_turn(Turn.witness(case.next_seq(), answer))

    console.print("[yellow]达到引导轮数上限,强制进入生图阶段。[/yellow]")


def run_initial_generation(case: Case) -> str:
    prompt = case.feature_matrix.to_image_prompt()
    console.print(Panel(prompt, title="首版文生图 prompt", border_style="cyan"))
    console.print("[green]生成初版画像中...[/green]")
    img = text_to_image(prompt)
    case.append_turn(Turn.image(case.next_seq(), img, prompt=prompt))
    console.print(f"[green]初版已保存:{img}[/green]")
    return img


def run_iteration(case: Case) -> None:
    """阶段 3:目击者反馈 → Translator → 图生图,直到目击者说收敛。"""
    while True:
        console.print()
        try:
            feedback = input("反馈(空回车 = 收敛并导出): ").strip()
        except EOFError:
            feedback = ""
        if not feedback:
            return

        prev = case.last_image()
        if not prev:
            console.print("[red]找不到上一版图,无法迭代。[/red]")
            return

        console.print("[green]翻译反馈中...[/green]")
        try:
            edit_prompt = call_translator(previous_image=prev, feedback=feedback)
        except Exception as e:
            console.print(f"[red]Translator 失败:{e}[/red]")
            continue
        console.print(Panel(edit_prompt, title="Edit prompt", border_style="cyan"))

        case.append_turn(Turn.witness(case.next_seq(), feedback))
        console.print("[green]图生图迭代中...[/green]")
        try:
            new_img = edit_with_refs(edit_prompt, [prev])
        except Exception as e:
            console.print(f"[red]edit_with_refs 失败:{e}[/red]")
            continue
        case.append_turn(Turn.image(case.next_seq(), new_img, parent_image=prev, prompt=edit_prompt))
        console.print(f"[green]新版已保存:{new_img}[/green]")


def main() -> int:
    args = parse_args()
    console.print(Panel(
        f"标题:{args.case_name}\n用途:民间寻人 / 记忆辅助。\n"
        "免责:本工具不构成司法证据,不替代专业画像。",
        title="FaceRecall Phase 0",
        border_style="magenta",
    ))

    case = Case.new(title=args.case_name)
    try:
        run_guidance(case)
        run_initial_generation(case)
        run_iteration(case)
        case.status = CaseStatus.CONVERGED
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断,会话标记 abandoned。[/yellow]")
        case.status = CaseStatus.ABANDONED
    except Exception:
        console.print("[red]发生未捕获异常:[/red]")
        console.print(traceback.format_exc())
        case.status = CaseStatus.ABANDONED
    finally:
        case.final_image = case.last_image()
        path = save_case(case)
        console.print(f"\n[bold green]会话已保存:{path}[/bold green]")
        if case.final_image:
            console.print(f"[bold green]最终画像:{case.final_image}[/bold green]")

    return 0 if case.status == CaseStatus.CONVERGED else 1


if __name__ == "__main__":
    sys.exit(main())
