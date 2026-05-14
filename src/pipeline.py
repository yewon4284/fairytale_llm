"""
pipeline.py
생성 → 세이프가드 → Solar 평가 → Solar 첨삭 → 재평가 루프

흐름:
  [1] 카나나로 동화 생성 (150단어 미만이면 토큰 늘려서 재시도, 반복 소모 안 함)
  [2] 세이프가드 전체 1회 검사 (심각한 유해 콘텐츠 안전망)
  [3] Solar 맥락 평가 (5개 항목, 하드룰 포함)
  [4] FAIL 시 Solar가 직접 첨삭 (원본 흐름 유지, 문제 문장만 수정)
  [5] 첨삭본 재평가
  [6] 여전히 FAIL이면 카나나 재생성 (fallback)
"""

import re
import time
from dataclasses import dataclass

from .generator import FairyTaleGenerator, TARGET_MIN_WORDS
from .safeguard import KananaSafeguard
from .evaluator import SolarEvaluator, EvaluationResult
from .data_loader import load_fairy_tales, get_calibration_sample, get_few_shot_samples, build_few_shot_block

MAX_ITERATIONS = 3
PASS_AVERAGE = 4.0


@dataclass
class PipelineResult:
    final_tale: str
    passed: bool
    evaluation_history: list[EvaluationResult]
    total_time_sec: float


class FairyTalePipeline:
    def __init__(
        self,
        generator: FairyTaleGenerator,
        safeguard: KananaSafeguard,
        evaluator: SolarEvaluator,
        data_dir: str = "data",
    ):
        self.generator = generator
        self.safeguard = safeguard
        self.evaluator = evaluator

        # 데이터셋에서 캘리브레이션 샘플 + few-shot 예시 로드
        tales = load_fairy_tales(data_dir)
        self.calibration = get_calibration_sample(tales)
        if self.calibration:
            print(f"[Pipeline] 캘리브레이션 샘플: '{self.calibration['title']}'")

        # few-shot: 의사소통 분류 동화 3개, 각 300자
        # (대화 중심, 짧은 문장으로 동화체 문체 가이드)
        few_shot_samples = get_few_shot_samples(tales, n=3)
        self.few_shot_example = build_few_shot_block(few_shot_samples, chars_per_sample=300)
        if few_shot_samples:
            titles = [s["title"] for s in few_shot_samples]
            print(f"[Pipeline] few-shot 예시: {titles}")

    def run(self, user_request: str) -> PipelineResult:
        start_time = time.time()
        evaluation_history = []
        current_request = user_request
        final_tale = ""
        passed = False

        print(f"\n{'='*60}")
        print(f"[Pipeline] 시작: {user_request[:80]}")
        print(f"{'='*60}")

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'─'*60}")
            print(f"[반복 {iteration}/{MAX_ITERATIONS}] 카나나 동화 생성")
            print(f"{'─'*60}")

            # ── [1] 동화 생성 ──────────────────────────────────────────
            raw_tale = self._generate_with_retry(current_request)
            word_count = self.generator.count_words(raw_tale)
            print(f"\n[생성된 동화 - {word_count}단어]\n{raw_tale}\n")

            # ── [2] 세이프가드 (전체 1회) ──────────────────────────────
            sg_result = self.safeguard.check(raw_tale)
            if not sg_result["is_safe"]:
                print(f"[Pipeline] ⛔ 세이프가드 차단: {sg_result['category']} → 재생성")
                current_request = user_request
                continue

            # ── [3] Solar 평가 ─────────────────────────────────────────
            evaluation = self.evaluator.evaluate(
                fairy_tale_text=raw_tale,
                original_request=user_request,
                calibration_sample=self.calibration,
            )
            evaluation_history.append(evaluation)
            self._print_evaluation(evaluation, label=f"반복 {iteration} - 1차 평가")

            if evaluation.passed:
                final_tale = raw_tale
                passed = True
                print(f"\n✅ 통과! (평균 {evaluation.average}점, {iteration}회)")
                break

            # ── [4] Solar 직접 첨삭 ────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[반복 {iteration}/{MAX_ITERATIONS}] Solar 직접 첨삭")
            print(f"{'─'*60}")

            rewritten = self.evaluator.rewrite_tale(
                original_tale=raw_tale,
                evaluation=evaluation,
                original_request=user_request,
            )
            print(f"\n[첨삭된 동화 - {self.generator.count_words(rewritten)}단어]\n{rewritten}\n")

            # ── [5] 첨삭본 재평가 ──────────────────────────────────────
            eval2 = self.evaluator.evaluate(
                fairy_tale_text=rewritten,
                original_request=user_request,
                calibration_sample=self.calibration,
            )
            evaluation_history.append(eval2)
            self._print_evaluation(eval2, label=f"반복 {iteration} - 첨삭 후 평가")

            if eval2.passed:
                final_tale = rewritten
                passed = True
                print(f"\n✅ 첨삭 후 통과! (평균 {eval2.average}점, {iteration}회)")
                break

            # ── [6] 여전히 FAIL → 카나나 재생성 (fallback) ────────────
            final_tale = rewritten
            if iteration < MAX_ITERATIONS:
                print(f"\n[Pipeline] 첨삭 후도 FAIL → 카나나 재생성")
                current_request = self.evaluator.build_rewrite_prompt(
                    original_request=user_request,
                    evaluation=eval2,
                )
            else:
                print(f"\n⚠️  최대 반복 도달. 마지막 결과 사용.")

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("=== 최종 결과 ===")
        print(f"{'='*60}")
        print(f"통과 여부: {'✅ PASS' if passed else '⚠️ BEST_EFFORT'}")
        print(f"총 평가 횟수: {len(evaluation_history)}회")
        print(f"소요 시간: {total_time:.1f}초")
        print(f"\n[최종 동화]\n{final_tale}")

        return PipelineResult(
            final_tale=final_tale,
            passed=passed,
            evaluation_history=evaluation_history,
            total_time_sec=round(total_time, 1),
        )

    def _generate_with_retry(self, request: str, max_retries: int = 2) -> str:
        """
        동화를 생성합니다.
        TARGET_MIN_WORDS(200) 미만이면 토큰을 늘리고 길이 명시 요청을 추가해서 재시도합니다.
        iteration 횟수를 소모하지 않습니다.
        """
        tokens = 1024
        result = ""
        for attempt in range(1, max_retries + 1):
            # 재시도 시 길이 요청을 명시적으로 추가
            req = request if attempt == 1 else (
                request + "\n\n(중요: 동화를 최소 200단어 이상으로 충분히 길게 써주세요. "
                "도입-갈등-반성-사과-마무리 각 단계를 빠짐없이 전개하세요.)"
            )
            result = self.generator.generate(
                user_request=req,
                max_new_tokens=tokens,
                few_shot_example=self.few_shot_example,
            )
            wc = self.generator.count_words(result)
            if wc >= TARGET_MIN_WORDS:
                return result
            print(f"[Pipeline] ⚠️  동화 짧음 ({wc}단어, 시도 {attempt}/{max_retries})"
                  f" → 토큰 {tokens}→{tokens+512}, 길이 요청 추가")
            tokens += 512
        return result

    @staticmethod
    def _print_evaluation(ev: EvaluationResult, label: str = "평가"):
        # 캐릭터-행동 맵
        cmap = ev.character_action_map or {}
        if cmap:
            print("\n[캐릭터-행동 맵]")
            for char, actions in cmap.items():
                acts = " → ".join(str(a) for a in actions) if isinstance(actions, list) else str(actions)
                print(f"  {char}: {acts}")

        # 요청 반영 여부
        rm = ev.request_match or {}
        if rm:
            icon = "✅" if rm.get("is_present") else "❌"
            print(f"\n[요청 반영] {icon} {rm.get('requested_bad_action','')} → {rm.get('note','')}")

        # 갈등 원인 분석
        conflict = getattr(ev, 'conflict_analysis', None)
        if conflict:
            both = conflict.get("both_at_fault")
            both_apo = conflict.get("both_apologized")
            icon = "✅" if (not both or both_apo) else "❌"
            print(f"[갈등 원인] {icon} {conflict.get('cause', '')} "
                  f"| 양측 잘못={'예' if both else '아니오'} "
                  f"| 양측 사과={'예' if both_apo else '아니오' if both_apo is False else 'N/A'}")
            if conflict.get("fault_details"):
                print(f"           {conflict['fault_details']}")

        # 사과 구조
        ap = ev.apology_check or {}
        if ap:
            icon = "❌" if ap.get("is_reversed") else "✅"
            print(f"[사과 구조] {icon} 가해자={ap.get('bad_actor','?')} | "
                  f"먼저 사과={ap.get('who_apologized_first','?')} | "
                  f"계기={ap.get('apology_trigger','none')}")

        # 점수
        reasons = ev.score_reasons or {}
        print(f"\n[{label}]")
        for k, v in ev.scores.items():
            bar = "★" * v + "☆" * (5 - v)
            reason = f"  # {reasons[k]}" if k in reasons else ""
            print(f"  {k:22s}: {bar} ({v}/5){reason}")
        print(f"  {'평균':22s}: {ev.average:.1f}/5.0  {'✅ PASS' if ev.passed else '❌ FAIL'}")
        print(f"  피드백: {ev.overall_feedback}")
        if ev.rewrite_instruction:
            print(f"  수정 지시: {ev.rewrite_instruction[:200]}")