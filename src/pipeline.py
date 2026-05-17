"""
pipeline.py
생성 → 1차 평가(카나나 세이프가드) → 2차 평가(Solar) → 재작성 루프를 관리한다.

[재작성 모드 — RewriteMode]

  MODE A: KANANA_REWRITE  (카나나 재생성 모드)
  Solar가 수정 지시(hint)를 내리고, 카나나가 hint를 반영해 동화를 재생성한다.

  MODE B: SOLAR_REWRITE  (Solar 직접 수정 모드)
  Solar가 이전 동화와 평가 결과를 보고 직접 수정한 동화를 생성한다.
  카나나는 1회차에만 참여한다.

[루프 정책]
  - 최대 4회 시도
  - 합격(평균 4.5점 이상 AND 항목별 최저 4.0점 이상) 시 즉시 반환
  - 4회 모두 불합격이더라도 최고 점수 동화를 출력
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
LENGTH_MIN   = 700
LENGTH_MAX   = 1800


# ── 재작성 모드 ──────────────────────────────────────────────────────────────
class RewriteMode(str, Enum):
    KANANA_REWRITE = "kanana_rewrite"
    SOLAR_REWRITE  = "solar_rewrite"


DEFAULT_MODE = RewriteMode.KANANA_REWRITE


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

    def __init__(
        self,
        generator: FairyTaleGenerator,
        safeguard: KananaSafeguard,
        evaluator: SolarEvaluator,
        rewrite_mode: RewriteMode = DEFAULT_MODE,
        few_shot_text: str = "",
    ):
        self.generator    = generator
        self.safeguard    = safeguard
        self.evaluator    = evaluator
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
        rewrite_hint   = ""
        previous_story = ""
        final_plan     = ""
        final_story    = ""
        passed         = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            _print_attempt_header(attempt, self.rewrite_mode)

            # ── Step 1: 동화 기획 (Solar, 1회차만) ───────────────────────────
            if attempt == 1:
                print("\n🧩 [Step 1] Solar — 동화 기획 생성 중...")
                final_plan = self.evaluator.plan(user_request)
                print(final_plan)

            # ── Step 2: 동화 본문 생성 ────────────────────────────────────────
            if attempt == 1 or self.rewrite_mode == RewriteMode.KANANA_REWRITE:
                print(f"\n📝 [Step 2] 카나나({self.generator.model_id.split('/')[-1]}) — 동화 {'재' if attempt > 1 else ''}생성 중...")
                story = self.generator.generate(
                    final_plan,
                    rewrite_hint,
                    few_shot_examples=self.few_shot_text,
                )
                generator_label = self.generator.model_id.split("/")[-1]
            else:
                print(f"\n✏️  [Step 2] Solar — 이전 동화 직접 수정 중... (시도 {attempt})")
                story = self.evaluator.rewrite_story(
                    plan=final_plan,
                    previous_story=previous_story,
                    eval_result=records[-1].eval_result,
                )
                generator_label = "solar-pro (직접 수정)"

            final_story    = story
            previous_story = story

            print("\n📖 [생성된 동화]")
            print(story)
            char_count = len(story.replace(" ", ""))
            print(f"\n  글자 수 (공백 제외): {char_count}자")
            print(f"  생성 주체: {generator_label}")

            # ── 길이 하드체크 ─────────────────────────────────────────────────
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
            print("\n🔍 [Step 3] 1차 평가 — 카나나 세이프가드")
            sentences, flagged = self.safeguard.evaluate_story(story)
            if flagged:
                print(f"  ⚠ 요주의 문장 {len(flagged)}개 발견:")
                for f in flagged:
                    print(f"    [{f['idx']+1}번] {f['category']}({f['desc']}): {f['sentence']}")
            else:
                print("  ✅ 모든 문장 1차 평가 통과")

            # ── Step 4: 2차 평가 (Solar) ──────────────────────────────────────
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
                    rewrite_hint = self.evaluator.build_rewrite_hint(eval_result)
                    print(f"\n🔄 [MODE A] 수정 지시 생성 완료 → 카나나 재생성 시작 (시도 {attempt+1})")
                else:
                    print(f"\n🔄 [MODE B] Solar 직접 수정 시작 (시도 {attempt+1})")
            else:
                print(f"\n⚠ 최대 시도 횟수({MAX_ATTEMPTS}회) 초과.")

        # ── 최고 점수 동화 선택 ───────────────────────────────────────────────
        if records:
            best_record = max(
                records,
                key=lambda r: (
                    r.passed,
                    r.eval_result.get("average_score", 0),
                )
            )
            final_story  = best_record.story
            best_attempt = best_record.attempt

            if best_attempt != records[-1].attempt:
                print(
                    f"\n💡 시도 {best_attempt}의 동화가 최고 점수 "
                    f"({best_record.eval_result.get('average_score', 0):.2f}점)로 선택됨"
                )
        else:
            best_attempt = 1

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
        RewriteMode.KANANA_REWRITE: "MODE A — 카나나 재생성",
        RewriteMode.SOLAR_REWRITE:  "MODE B — Solar 직접 수정",
    }[mode]
    print("\n" + "=" * 70)
    print("  🌟 LLM 기반 아동 동화 생성 시스템")
    print("=" * 70)
    print(f"  재작성 모드: {mode_label}")
    print(f"  Generator:  {model_id.split('/')[-1]}")
    print(f"  최대 시도:  {MAX_ATTEMPTS}회")
    print(f"  글자 수:    {LENGTH_MIN}~{LENGTH_MAX}자")
    print(f"\n📝 사용자 요청:\n  {user_request}")
    print("=" * 70)


def _print_attempt_header(attempt: int, mode: RewriteMode):
    mode_tag = "A" if mode == RewriteMode.KANANA_REWRITE else "B"
    print(f"\n{'─' * 70}")
    print(f"  🎯 [MODE {mode_tag}] 시도 {attempt} / {MAX_ATTEMPTS}")
    print(f"{'─' * 70}")


def print_final_result(result: PipelineResult):
    print("\n" + "=" * 70)
    print("  🏁 최종 결과")
    print("=" * 70)
    print(f"  상태:        {'✅ 합격' if result.passed else '❌ 불합격 (최대 시도 초과)'}")
    print(f"  재작성 모드: {result.rewrite_mode}")
    print(f"  Generator:   {result.generator_model.split('/')[-1]}")
    print(f"  총 시도:     {result.total_attempts}회")
    print(f"  최종 선택:   시도 {result.best_attempt}번 (최고 점수)")
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
            best_mark = " ⭐ 최종 선택" if rec.attempt == result.best_attempt else ""
            print(f"  시도 {rec.attempt} ({rec.generator_model}): 평균 {avg}점 → {status}{best_mark}")

    print("=" * 70)