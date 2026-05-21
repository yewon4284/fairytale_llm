"""
pipeline.py
생성 → 1차 평가(카나나 세이프가드) → 2차 평가(Solar) → 재작성 루프를 관리한다.

[재작성 모드 — RewriteMode]

  MODE A: KANANA_REWRITE  (카나나 재생성 모드)
  ─────────────────────────────────────────────
  Solar가 평가 후 "무엇을 어떻게 고쳐라"는 지시(hint)를 내리고,
  카나나가 그 hint를 반영해 동화를 처음부터 다시 생성한다.

  흐름:
    Solar → 기획 → 카나나 → 동화 생성
        ↓ FAIL
    Solar → 수정 지시(hint) → 카나나 → 동화 재생성 (최대 4회)

  장점: 카나나의 한국어 텍스트 생성 능력 활용, 생태계 일관성 유지
  단점: hint 해석 오류 가능성, 루프마다 카나나 추론 비용 발생

  MODE B: SOLAR_REWRITE  (Solar 직접 수정 모드)
  ──────────────────────────────────────────────
  Solar가 평가 후 동화를 직접 수정해 새 버전을 생성한다.
  카나나는 1회차 생성에만 참여하고 이후 루프에는 관여하지 않는다.

  흐름:
    Solar → 기획 → 카나나 → 동화 생성 (1회차만)
        ↓ FAIL
    Solar → 이전 동화 + 평가 결과 → 직접 수정 동화 생성 (2~4회차)

  장점: 평가 결과를 즉시 반영 가능, hint 해석 오류 없음, 루프 1단계 단축
  단점: Solar API 호출 비용 증가, Solar의 한국어 동화 생성 품질이 카나나 대비 낮을 수 있음

[실험 설계]
  두 모드를 동일한 요청에 실행하여 결과를 비교하면
  "카나나 재생성 vs Solar 직접 수정" 성능 차이를 정량적으로 측정할 수 있다.
  → 논문/보고서의 Ablation Study로 활용 가능

[루프 정책]
  - 최대 4회 시도
  - 합격(평균 4.5점 이상 AND 항목별 최저 4.0점 이상) 시 즉시 반환
  - 4회 모두 불합격이더라도 마지막 동화를 무조건 출력 (결과 저장 없음)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv

from src.generator import FairyTaleGenerator, check_bias
from src.safeguard import KananaSafeguard
from src.evaluator import SolarEvaluator

load_dotenv()
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 4


# ── 재작성 모드 ──────────────────────────────────────────────────────────────
class RewriteMode(str, Enum):
    KANANA_REWRITE = "kanana_rewrite"
    """
    카나나 재생성 모드 (MODE A).
    Solar가 수정 지시(hint)를 내리고, 카나나가 hint를 반영해 동화를 재생성한다.
    """

    SOLAR_REWRITE = "solar_rewrite"
    """
    Solar 직접 수정 모드 (MODE B).
    Solar가 이전 동화와 평가 결과를 보고 직접 수정한 동화를 생성한다.
    카나나는 1회차에만 참여한다.
    """


DEFAULT_MODE = RewriteMode.KANANA_REWRITE
# Solar가 수정 지시(hint)를 작성하고, 카나나 1.5 8B가 hint를 반영해 동화를 재생성


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass
class AttemptRecord:
    attempt: int
    plan: str
    story: str
    generator_model: str        # 이번 시도에서 동화를 생성한 모델 이름
    flagged_sentences: List[dict]
    eval_result: dict
    passed: bool
    eval_summary: str


@dataclass
class PipelineResult:
    user_request: str
    rewrite_mode: str
    generator_model: str
    final_story: str
    final_plan: str
    passed: bool
    total_attempts: int
    best_attempt: int = 1
    attempts: List[AttemptRecord] = field(default_factory=list)


# ── 파이프라인 ────────────────────────────────────────────────────────────────
class FairyTalePipeline:
    """
    동화 생성-평가-재작성 피드백 루프 파이프라인.

    Args:
        generator     : FairyTaleGenerator 인스턴스
        safeguard     : KananaSafeguard 인스턴스
        evaluator     : SolarEvaluator 인스턴스
        rewrite_mode  : RewriteMode.KANANA_REWRITE 또는 RewriteMode.SOLAR_REWRITE
        few_shot_text : 데이터셋에서 로드한 참고 동화 텍스트 (퓨샷, 없으면 빈 문자열)
    """

    def __init__(
        self,
        generator: FairyTaleGenerator,
        safeguard: KananaSafeguard,
        evaluator: SolarEvaluator,
        rewrite_mode: RewriteMode = DEFAULT_MODE,
        few_shot_text: str = "",
    ):
        self.generator = generator
        self.safeguard = safeguard
        self.evaluator = evaluator
        self.rewrite_mode = rewrite_mode
        self.few_shot_text = few_shot_text

    def run(self, user_request: str) -> PipelineResult:
        # ── 편향 단어 검사 ────────────────────────────────────────────────────
        biased = check_bias(user_request)
        if biased:
            raise ValueError(
                f"입력에 편향을 유발할 수 있는 단어가 포함되어 있습니다: '{biased}'\n"
                f"특정 성별(공주/왕자/아들/딸), 인종 등의 단어 없이 상황을 설명해 주세요."
            )

        _print_header(user_request, self.rewrite_mode, self.generator.model_id)

        records: List[AttemptRecord] = []
        rewrite_hint = ""       # MODE A 전용: 카나나에 전달할 수정 지시
        previous_story = ""     # MODE B 전용: Solar에 전달할 이전 동화
        final_plan = ""
        final_story = ""
        passed = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            _print_attempt_header(attempt, self.rewrite_mode)

            # ── Step 1: 동화 기획 (Solar, 1회차만) ───────────────────────────
            if attempt == 1:
                print("\n🧩 [Step 1] Solar — 동화 기획 생성 중...")
                final_plan = self.evaluator.plan(user_request)
                print(final_plan)

            # ── Step 2: 동화 본문 생성 ────────────────────────────────────────
            if attempt == 1 or self.rewrite_mode == RewriteMode.KANANA_REWRITE:
                # MODE A: 매 회차 카나나가 (hint 반영하여) 생성
                # MODE B: 1회차만 카나나 생성, 이후는 Solar가 직접 생성
                print(f"\n📝 [Step 2] 카나나({self.generator.model_id.split('/')[-1]}) — 동화 생성 중...")
                story = self.generator.generate(
                    final_plan,
                    rewrite_hint,
                    few_shot_examples=self.few_shot_text,
                )
                generator_label = self.generator.model_id.split("/")[-1]

            else:
                # MODE B 2회차 이후: Solar가 이전 동화를 직접 수정
                print(f"\n✏️  [Step 2] Solar — 이전 동화 직접 수정 중... (시도 {attempt})")
                story = self.evaluator.rewrite_story(
                    plan=final_plan,
                    previous_story=previous_story,
                    eval_result=records[-1].eval_result,
                )
                generator_label = "solar-pro (직접 수정)"

            final_story = story
            previous_story = story
            print("\n📖 [생성된 동화]")
            print(story)
            print(f"\n  글자 수 (공백 제외): {len(story.replace(' ', ''))}자")
            print(f"  생성 주체: {generator_label}")

            # ── Step 3: 1차 평가 (카나나 세이프가드) ─────────────────────────
            print("\n🔍 [Step 3] 1차 평가 — 카나나 세이프가드")
            sentences, flagged = self.safeguard.evaluate_story(story)
            if flagged:
                print(f"  ⚠ 요주의 문장 {len(flagged)}개 발견:")
                for f in flagged:
                    print(f"    [{f['idx']+1}번] {f['category']}({f['desc']}): {f['sentence']}")
            else:
                print("  ✅ 모든 문장 1차 평가 통과")

            # ── Step 4: 2차 평가 (Solar) ──────────────────────────────────────
            # 세이프가드 태깅 결과를 항상 Solar에 전달한다.
            # S3/S5/S6이 태깅된 경우 Solar 프롬프트 규칙에 의해 맥락 무관 FAIL 처리된다.
            print("\n🧠 [Step 4] 2차 평가 — Solar API (맥락 기반)")
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
                if self.rewrite_mode == RewriteMode.KANANA_REWRITE:
                    # MODE A: Solar가 수정 지시를 hint로 만들어 카나나에 전달
                    rewrite_hint = self.evaluator.build_rewrite_hint(eval_result)
                    print(f"\n🔄 [MODE A] 수정 지시 생성 완료 → 카나나 재생성 시작 (시도 {attempt+1})")
                else:
                    # MODE B: 이전 동화를 그대로 보존, Solar가 다음 루프에서 직접 수정
                    print(f"\n🔄 [MODE B] Solar 직접 수정 시작 (시도 {attempt+1})")
            else:
                print(f"\n⚠ 최대 시도 횟수({MAX_ATTEMPTS}회) 초과.")

        # ── 최고 점수 동화 선택 ───────────────────────────────────────────────
        # 합격한 시도가 있으면 첫 번째 합격본, 없으면 평균 점수가 가장 높은 시도 선택
        best_record = max(
            records,
            key=lambda r: (
                r.passed,  # 합격 여부 우선
                r.eval_result.get("average_score", 0),  # 그 다음 점수
            )
        )
        final_story = best_record.story
        best_attempt = best_record.attempt

        if best_attempt != records[-1].attempt:
            print(f"\n💡 시도 {best_attempt}의 동화가 최고 점수 "
                  f"({best_record.eval_result.get('average_score', 0):.2f}점)로 선택됨 "
                  f"(마지막 시도 {records[-1].attempt}번 대신)")

        return PipelineResult(
            user_request=user_request,
            rewrite_mode=self.rewrite_mode.value,
            generator_model=self.generator.model_id,
            final_story=final_story,
            final_plan=final_plan,
            passed=passed,
            total_attempts=len(records),
            attempts=records,
            best_attempt=best_attempt,
        )


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────

def _print_header(user_request: str, mode: RewriteMode, model_id: str):
    mode_label = {
        RewriteMode.KANANA_REWRITE: "MODE A — 카나나 재생성 (Solar 지시 → 카나나 재작성)",
        RewriteMode.SOLAR_REWRITE:  "MODE B — Solar 직접 수정 (Solar가 동화 직접 수정)",
    }[mode]

    print("\n" + "=" * 70)
    print("  🌟 LLM 기반 아동 동화 생성 시스템")
    print("=" * 70)
    print(f"  재작성 모드: {mode_label}")
    print(f"  Generator:  {model_id.split('/')[-1]}")
    print(f"  최대 시도:  {MAX_ATTEMPTS}회")
    print(f"\n📝 사용자 요청:\n  {user_request}")
    print("=" * 70)


def _print_attempt_header(attempt: int, mode: RewriteMode):
    mode_tag = "A" if mode == RewriteMode.KANANA_REWRITE else "B"
    print(f"\n{'─' * 70}")
    print(f"  🎯 [MODE {mode_tag}] 시도 {attempt} / {MAX_ATTEMPTS}")
    print(f"{'─' * 70}")


def print_final_result(result: PipelineResult):
    """최종 결과를 보기 좋게 출력한다."""
    print("\n" + "=" * 70)
    print("  🏁 최종 결과")
    print("=" * 70)
    print(f"  상태:       {'✅ 합격' if result.passed else '❌ 불합격 (최대 시도 초과)'}")
    print(f"  재작성 모드: {result.rewrite_mode}")
    print(f"  Generator:  {result.generator_model.split('/')[-1]}")
    print(f"  총 시도:    {result.total_attempts}회")
    print(f"  최종 선택:  시도 {result.best_attempt}번 (최고 점수)")
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
            avg = rec.eval_result.get("average_score", "-")
            status = "✅ PASS" if rec.passed else "❌ FAIL"
            best_mark = " ⭐ 최종 선택" if rec.attempt == result.best_attempt else ""
            print(f"  시도 {rec.attempt} ({rec.generator_model}): 평균 {avg}점 → {status}{best_mark}")

    print("=" * 70)