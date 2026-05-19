"""
test_src/solar_pipeline.py
Solar Pro 단독 파이프라인 (비교 실험용).

[파이프라인 흐름]
  1. Solar Pro — 동화 기획 (1회차만)
  2. Solar Pro — 동화 본문 생성 (hint 반영)
  3. Solar Pro — 맥락 기반 평가 (6개 항목 + 신체 안전 즉시-FAIL)
  4. FAIL → Solar Pro가 수정 지시(hint) 생성 → 재생성 → 3번부터 반복
  5. 최대 MAX_ATTEMPTS회 시도 후 마지막 동화 무조건 출력

세이프가드(카나나 8B)는 비교 대상에서 제외.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

from src.generator import check_bias
from src.evaluator import SolarEvaluator
from test_src.solar_generator import SolarGenerator

load_dotenv()
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 10


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass
class AttemptRecord:
    attempt: int
    plan: str
    story: str
    eval_result: dict
    passed: bool
    eval_summary: str


@dataclass
class SolarPipelineResult:
    user_request: str
    final_story: str
    final_plan: str
    passed: bool
    total_attempts: int
    attempts: List[AttemptRecord] = field(default_factory=list)


# ── 파이프라인 ────────────────────────────────────────────────────────────────
class SolarOnlyPipeline:
    """Solar Pro 단독 생성-평가-재작성 루프."""

    def __init__(
        self,
        generator: SolarGenerator,
        evaluator: SolarEvaluator,
        few_shot_text: str = "",
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.few_shot_text = few_shot_text

    def run(self, user_request: str) -> SolarPipelineResult:
        biased = check_bias(user_request)
        if biased:
            raise ValueError(
                f"입력에 편향을 유발할 수 있는 단어가 포함되어 있습니다: '{biased}'\n"
                "특정 성별(공주/왕자/아들/딸), 인종 등의 단어 없이 상황을 설명해 주세요."
            )

        _print_header(user_request)

        records: List[AttemptRecord] = []
        rewrite_hint = ""
        final_plan   = ""
        final_story  = ""
        passed       = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            _print_attempt_header(attempt)

            # ── Step 1: 동화 기획 (Solar Pro, 1회차만) ───────────────────────
            if attempt == 1:
                print("\n🧩 [Step 1] Solar Pro — 동화 기획 생성 중...")
                final_plan = self.generator.plan(user_request)
                print(final_plan)

            # ── Step 2: 동화 본문 생성 ────────────────────────────────────────
            print(f"\n📝 [Step 2] Solar Pro — 동화 {'재' if attempt > 1 else ''}생성 중...")
            story = self.generator.generate(
                final_plan,
                rewrite_hint,
                few_shot_examples=self.few_shot_text,
            )
            final_story = story

            print("\n📖 [생성된 동화]")
            print(story)
            print(f"\n  글자 수 (공백 제외): {len(story.replace(' ', ''))}자")

            # ── Step 3: 평가 (Solar Pro) ──────────────────────────────────────
            print("\n🧠 [Step 3] 평가 — Solar Pro (맥락 기반, 6개 항목 + 신체 안전)")
            eval_result, passed, eval_summary = self.evaluator.evaluate(
                story, flagged_sentences=[], user_request=user_request
            )
            print(eval_summary)

            records.append(AttemptRecord(
                attempt=attempt,
                plan=final_plan,
                story=story,
                eval_result=eval_result,
                passed=passed,
                eval_summary=eval_summary,
            ))

            if passed:
                print(f"\n✅ {attempt}회 시도에서 합격!")
                break

            if attempt < MAX_ATTEMPTS:
                rewrite_hint = self.evaluator.build_rewrite_hint(eval_result)
                print(f"\n🔄 수정 지시 생성 완료 → Solar Pro 재생성 시작 (시도 {attempt + 1}/{MAX_ATTEMPTS})")
                print(f"  수정 지시:\n{rewrite_hint}")
            else:
                print(f"\n⚠ 최대 시도 횟수({MAX_ATTEMPTS}회) 초과. 마지막 동화를 최종 출력합니다.")

        return SolarPipelineResult(
            user_request=user_request,
            final_story=final_story,
            final_plan=final_plan,
            passed=passed,
            total_attempts=len(records),
            attempts=records,
        )


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────
def _print_header(user_request: str):
    print("\n" + "=" * 70)
    print("  🌟 Solar Pro 단독 동화 생성 시스템 (비교 실험용)")
    print("=" * 70)
    print("  Planner:   Solar Pro")
    print("  Generator: Solar Pro")
    print("  Evaluator: Solar Pro")
    print("  Safeguard: 없음 (비교 실험)")
    print(f"  최대 시도: {MAX_ATTEMPTS}회")
    print(f"\n📝 사용자 요청:\n  {user_request}")
    print("=" * 70)


def _print_attempt_header(attempt: int):
    print(f"\n{'─' * 70}")
    print(f"  🎯 시도 {attempt} / {MAX_ATTEMPTS}")
    print(f"{'─' * 70}")


def print_final_result(result: SolarPipelineResult):
    print("\n" + "=" * 70)
    print("  🏁 최종 결과 (Solar Pro 단독)")
    print("=" * 70)
    print(f"  상태:    {'✅ 합격' if result.passed else '❌ 불합격 (최대 시도 초과)'}")
    print(f"  총 시도: {result.total_attempts}회")
    print("\n🧩 [최종 동화 기획]")
    print(result.final_plan)
    print("\n📖 [최종 동화]")
    print("─" * 70)
    print(result.final_story)
    print("─" * 70)
    print(f"  글자 수 (공백 제외): {len(result.final_story.replace(' ', ''))}자")

    if result.attempts:
        print("\n📊 [시도별 점수 히스토리]")
        for rec in result.attempts:
            avg    = rec.eval_result.get("average_score", "-")
            status = "✅ PASS" if rec.passed else "❌ FAIL"
            phys   = "🚨신체위반" if rec.eval_result.get("physical_safety_violation") else ""
            print(f"  시도 {rec.attempt}: 평균 {avg}점 → {status} {phys}")

    print("=" * 70)
