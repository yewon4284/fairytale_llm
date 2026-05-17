"""
server.py
FastAPI 로컬 서버 — React 프론트엔드와 파이프라인을 연결한다.

실행:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

엔드포인트:
    POST /generate        동화 생성 요청
    GET  /status/{job_id} 생성 진행 상태 조회
    GET  /result/{job_id} 최종 결과 조회
    GET  /images/{filename} 생성된 이미지 제공
"""

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="동화 생성 시스템")


@app.on_event("startup")
async def startup_clear_memory():
    """서버 시작 시 GPU 메모리 초기화."""
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            free = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"GPU 메모리 초기화 완료 — 여유: {free:.1f} GB")
    except Exception as e:
        logger.warning(f"GPU 메모리 초기화 실패: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 전역 상태 ─────────────────────────────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=1)  # 모델 추론은 직렬 처리
jobs: Dict[str, dict] = {}  # job_id → 상태 dict

# 파이프라인 컴포넌트 (최초 요청 시 한 번만 로딩)
_pipeline = None
_pipeline_lock = asyncio.Lock()


# ── 요청/응답 모델 ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    request: str


class GenerateResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str          # pending | running | done | error
    step: Optional[str]  # 현재 진행 단계 설명
    progress: int        # 0~100


class ResultResponse(BaseModel):
    job_id: str
    status: str
    plan: str
    story: str
    passed: bool
    total_attempts: int
    images: list         # 이미지 URL 리스트
    scores: list         # 시도별 점수 히스토리


# ── 파이프라인 로딩 ───────────────────────────────────────────────────────────
def _load_pipeline():
    global _pipeline
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("UPSTAGE_API_KEY 환경변수가 없습니다.")

    from src.generator import FairyTaleGenerator, KANANA_15_8B
    from src.safeguard import KananaSafeguard
    from src.evaluator import SolarEvaluator
    from src.pipeline import FairyTalePipeline

    generator = FairyTaleGenerator(model_id=KANANA_15_8B)
    safeguard = KananaSafeguard()
    evaluator = SolarEvaluator(api_key=api_key)

    _pipeline = FairyTalePipeline(
        generator=generator,
        safeguard=safeguard,
        evaluator=evaluator,
    )
    logger.info("파이프라인 로딩 완료")


# ── 생성 작업 (별도 스레드에서 실행) ─────────────────────────────────────────
def _run_generation(job_id: str, user_request: str):
    global _pipeline

    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["step"] = "모델 로딩 중..."
        jobs[job_id]["progress"] = 5

        if _pipeline is None:
            _load_pipeline()

        jobs[job_id]["step"] = "Solar Pro — 동화 기획 중..."
        jobs[job_id]["progress"] = 15

        result = _pipeline.run(user_request)

        # 이미지 생성 시도 (image_generator 모듈이 있는 경우)
        images = []
        scenes = []
        try:
            import gc
            import torch
            from image_generator import FairyTaleImageGenerator
            api_key = os.getenv("UPSTAGE_API_KEY")
            img_gen = FairyTaleImageGenerator(api_key=api_key)

            # ── SDXL 로딩 전 Generator·Safeguard를 CPU로 내려 GPU 확보 ──
            jobs[job_id]["step"] = "GPU 메모리 확보 중..."
            jobs[job_id]["progress"] = 70
            logger.info("SDXL 로딩 전 Generator/Safeguard CPU 오프로드 시작")

            gen_model  = _pipeline.generator.model
            safe_model = _pipeline.safeguard.model

            # device_map="auto" 모델은 .cpu()가 불완전 → 재로딩으로 완전 해제
            del _pipeline.generator.model
            del _pipeline.safeguard.model
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU 메모리 확보 완료")

            jobs[job_id]["step"] = "Solar Pro — 장면 프롬프트 추출 중..."
            jobs[job_id]["progress"] = 75

            paths, scenes = img_gen.generate(result.final_story, result.final_plan)

            jobs[job_id]["step"] = "삽화 저장 완료"
            jobs[job_id]["progress"] = 95

            # 로컬 경로면 파일명만 추출, Imgur URL이면 그대로 사용
            images = [
                p if p.startswith("http") else os.path.basename(p)
                for p in paths
            ]
            logger.info(f"이미지 {len(images)}장 생성 완료")

            # ── SDXL 해제 후 Generator·Safeguard 다시 로딩 ──
            del img_gen.pipe
            img_gen.pipe = None
            torch.cuda.empty_cache()
            gc.collect()

            logger.info("Generator/Safeguard GPU 재로딩 중...")
            _pipeline.generator._load_model()
            _pipeline.safeguard._load_model()
            logger.info("Generator/Safeguard GPU 재로딩 완료")

        except ImportError as e:
            logger.warning(f"image_generator.py 로드 실패 — 삽화 없이 진행: {e}")
        except Exception as e:
            logger.exception(f"이미지 생성 실패 (계속 진행): {e}")
        finally:
            # 이미지 생성 성공·실패 무관하게 모델 복구
            import gc, torch
            if not hasattr(_pipeline.generator, 'model') or _pipeline.generator.model is None:
                logger.warning("Generator 모델 없음 → 재로딩")
                _pipeline.generator._load_model()
            if not hasattr(_pipeline.safeguard, 'model') or _pipeline.safeguard.model is None:
                logger.warning("Safeguard 모델 없음 → 재로딩")
                _pipeline.safeguard._load_model()

        jobs[job_id].update({
            "status": "done",
            "step": "완료",
            "progress": 100,
            "plan": result.final_plan,
            "story": result.final_story,
            "passed": result.passed,
            "total_attempts": result.total_attempts,
            "images": images,
            "scores": [
                {
                    "attempt": r.attempt,
                    "avg": r.eval_result.get("average_score", 0),
                    "passed": r.passed,
                }
                for r in result.attempts
            ],
        })

    except ValueError as e:
        jobs[job_id].update({
            "status": "error",
            "step": str(e),
            "progress": 0,
        })
    except Exception as e:
        logger.exception(f"생성 오류: {e}")
        jobs[job_id].update({
            "status": "error",
            "step": f"오류: {str(e)}",
            "progress": 0,
        })


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    from src.generator import check_bias
    biased = check_bias(req.request)
    if biased:
        raise HTTPException(
            status_code=400,
            detail=f"편향 단어 감지: '{biased}'. 성별·인종 단어 없이 입력해 주세요.",
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "step": "대기 중...",
        "progress": 0,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_generation, job_id, req.request)

    return GenerateResponse(job_id=job_id, message="생성 시작됨")


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    j = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=j["status"],
        step=j.get("step", ""),
        progress=j.get("progress", 0),
    )


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    j = jobs[job_id]
    if j["status"] != "done":
        raise HTTPException(status_code=202, detail=f"아직 완료되지 않았습니다: {j['status']}")
    return ResultResponse(
        job_id=job_id,
        status=j["status"],
        plan=j.get("plan", ""),
        story=j.get("story", ""),
        passed=j.get("passed", False),
        total_attempts=j.get("total_attempts", 0),
        images=j.get("images", []),
        scores=j.get("scores", []),
    )


@app.get("/images/{filename}")
async def serve_image(filename: str):
    path = os.path.join("outputs", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(path)


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.get("/")
async def root():
    """프론트엔드 index.html 제공."""
    index_path = os.path.join("frontend", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html 파일이 없습니다.")
    return FileResponse(index_path)