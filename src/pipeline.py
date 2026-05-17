"""
pipeline.py
생성 → 1차 평가(카나나 세이프가드) → 2차 평가(Solar Pro) → 카나나 재생성 루프.

[파이프라인 흐름]
  1. 편향 단어 검사
  2. Solar Pro — 동화 기획 (1회차만)
  3. 카나나 1.5 8B — 동화 본문 생성 (hint 반영)
  4. 카나나 세이프가드 8B — 문장 단위 1차 평가 (S1~S7 태깅)
  5. Solar Pro — 맥락 기반 2차 평가 (6개 항목 + 신체 안전 즉시-FAIL)
  6. FAIL → Solar가 수정 지시(hint) 생성 → 카나나 재생성 → 4번부터 반복
  7. 최대 4회 시도 후 마지막 동화 무조건 출력

[합격 기준]
  - 신체_안전 위반 없음 AND 평균 4.5점 이상 AND 항목별 최저 4.0점 이상
"""

import logging
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

from src.generator import FairyTaleGenerator, check_bias
from src.safeguard import KananaSafeguard
from src.evaluator import SolarEvaluator

load_dotenv()
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 4


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass
class AttemptRecord:
    attempt: int
    plan: str
    story: str
    generator_model: str
    flagged_sentences: List[dict]
    eval_result: dict
    passed: bool
    eval_summary: str


@dataclass
class PipelineResult:
    user_request: str
    generator_model: str
    final_story: str
    final_plan: str
    passed: bool
    total_attempts: int
    attempts: List[AttemptRecord] = field(default_factory=list)


# ── 파이프라인 ────────────────────────────────────────────────────────────────
class FairyTalePipeline:
    """
    동화 생성-평가-재작성 피드백 루프 파이프라인.

    Args:
        generator     : FairyTaleGenerator 인스턴스 (카나나 1.5 8B)
        safeguard     : KananaSafeguard 인스턴스
        evaluator     : SolarEvaluator 인스턴스
        few_shot_text : 데이터셋에서 로드한 참고 동화 텍스트 (없으면 빈 문자열)
    """

    def __init__(
        self,
        generator: FairyTaleGenerator,
        safeguard: KananaSafeguard,
        evaluator: SolarEvaluator,
        few_shot_text: str = "",
    ):
        self.generator = generator
        self.safeguard = safeguard
        self.evaluator = evaluator
        self.few_shot_text = few_shot_text

    def run(self, user_request: str) -> PipelineResult:
        # ── Step 0: 편향 단어 검사 ───────────────────────────────────────────
        biased = check_bias(user_request)
        if biased:
            raise ValueError(
                f"입력에 편향을 유발할 수 있는 단어가 포함되어 있습니다: '{biased}'\n"
                "특정 성별(공주/왕자/아들/딸), 인종 등의 단어 없이 상황을 설명해 주세요."
            )

        _print_header(user_request, self.generator.model_id)

        records: List[AttemptRecord] = []
        rewrite_hint = ""
        final_plan   = ""
        final_story  = ""
        passed       = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            _print_attempt_header(attempt)

            # ── Step 1: 동화 기획 (카나나, 1회차만) ──────────────────────────
            if attempt == 1:
                print(f"\n🧩 [Step 1] 카나나({self.generator.model_id.split('/')[-1]}) — 동화 기획 생성 중...")
                final_plan = self.generator.plan(user_request)
                print(final_plan)

            # ── Step 2: 동화 본문 생성 (카나나, 매 회차) ─────────────────────
            print(
                f"\n📝 [Step 2] 카나나({self.generator.model_id.split('/')[-1]}) "
                f"— 동화 {'재' if attempt > 1 else ''}생성 중..."
            )
            story = self.generator.generate(
                final_plan,
                rewrite_hint,
                few_shot_examples=self.few_shot_text,
            )
            generator_label = self.generator.model_id.split("/")[-1]
            final_story = story

            print("\n📖 [생성된 동화]")
            print(story)
            char_count = len(story.replace(" ", ""))
            print(f"\n  글자 수 (공백 제외): {char_count}자")

            # ── 길이 하드체크 — 범위 외 시 평가 없이 즉시 재생성 ─────────────
            LENGTH_MIN, LENGTH_MAX = 700, 1200
            if not (LENGTH_MIN <= char_count <= LENGTH_MAX):
                direction = "미달" if char_count < LENGTH_MIN else "초과"
                print(f"\n⚠ 글자 수 {direction} ({char_count}자) — 평가 건너뜀")
                if attempt < MAX_ATTEMPTS:
                    rewrite_hint = (
                        f"이전 동화가 {char_count}자로 길이 {direction}입니다. "
                        f"반드시 {LENGTH_MIN}자 이상 {LENGTH_MAX}자 이하로 작성하세요 (공백 제외)."
                    )
                    print(f"🔄 재생성 시작 (시도 {attempt + 1}/{MAX_ATTEMPTS})")
                    continue
                else:
                    print(f"⚠ 최대 시도 횟수({MAX_ATTEMPTS}회) 초과. 마지막 동화를 출력합니다.")
                    break

            # ── Step 3: 1차 평가 (카나나 세이프가드) ─────────────────────────
            print("\n🔍 [Step 3] 1차 평가 — 카나나 세이프가드 8B")
            sentences, flagged = self.safeguard.evaluate_story(story)
            if flagged:
                print(f"  ⚠ 요주의 문장 {len(flagged)}개 발견:")
                for f in flagged:
                    print(
                        f"    [{f['idx'] + 1}번] {f['category']}({f['desc']}): {f['sentence']}"
                    )
            else:
                print("  ✅ 모든 문장 1차 평가 통과")

            # ── Step 4: 2차 평가 (Solar Pro) ─────────────────────────────────
            print("\n🧠 [Step 4] 2차 평가 — Solar Pro (맥락 기반, 6개 항목 + 신체 안전)")
            eval_result, passed, eval_summary = self.evaluator.evaluate(
                story, flagged, user_request=user_request
            )
            print(eval_summary)

            records.append(AttemptRecord(
                attempt=attempt,
                plan=final_plan,
                story=story,
                generator_model=generator_label,
                flagged_sentences=flagged,
                eval_result=eval_result,
                passed=passed,
                eval_summary=eval_summary,
            ))

            if passed:
                print(f"\n✅ {attempt}회 시도에서 합격!")
                break

            if attempt < MAX_ATTEMPTS:
                rewrite_hint = self.evaluator.build_rewrite_hint(eval_result)
                print(
                    f"\n🔄 수정 지시 생성 완료 → 카나나 재생성 시작 (시도 {attempt + 1}/{MAX_ATTEMPTS})"
                )
                print(f"  수정 지시:\n{rewrite_hint}")
            else:
                print(
                    f"\n⚠ 최대 시도 횟수({MAX_ATTEMPTS}회) 초과. "
                    "마지막 동화를 최종 출력합니다."
                )

        return PipelineResult(
            user_request=user_request,
            generator_model=self.generator.model_id,
            final_story=final_story,
            final_plan=final_plan,
            passed=passed,
            total_attempts=len(records),
            attempts=records,
        )


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────
def _print_header(user_request: str, model_id: str):
    print("\n" + "=" * 70)
    print("  🌟 LLM 기반 아동 동화 생성 시스템")
    print("=" * 70)
    print(f"  Planner:    {model_id.split('/')[-1]}")
    print(f"  Generator:  {model_id.split('/')[-1]}")
    print(f"  Safeguard:  kanana-safeguard-8b")
    print(f"  Evaluator:  Solar Pro")
    print(f"  최대 시도:  {MAX_ATTEMPTS}회")
    print(f"\n📝 사용자 요청:\n  {user_request}")
    print("=" * 70)


def _print_attempt_header(attempt: int):
    print(f"\n{'─' * 70}")
    print(f"  🎯 시도 {attempt} / {MAX_ATTEMPTS}")
    print(f"{'─' * 70}")


def print_final_result(result: PipelineResult):
    """최종 결과를 보기 좋게 출력한다."""
    print("\n" + "=" * 70)
    print("  🏁 최종 결과")
    print("=" * 70)
    print(f"  상태:      {'✅ 합격' if result.passed else '❌ 불합격 (최대 시도 초과)'}")
    print(f"  Generator: {result.generator_model.split('/')[-1]}")
    print(f"  총 시도:   {result.total_attempts}회")
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