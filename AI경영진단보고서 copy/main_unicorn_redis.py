# Standard Library Imports
import os
import time
import json
import logging
import asyncio
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Dict, Any, List, Optional, Tuple
from typing_extensions import TypedDict

# Third Party Imports
import pandas as pd
import redis.asyncio as redis
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from tenacity import retry, stop_after_attempt, wait_exponential
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Local Application Imports
from AI경영진단보고서.config_bk import OPENAI_API_KEY, PROMPTS_DIR
from utils import load_prompt, validate_input_data
from utils.logger import get_logger
from utils.time_utils import get_access_time
from utils.validation import validate_input_data, DataValidationError
from database.f_read_pg_sql import fetch_company_data
from analysis import (
    determine_strength_weakness,
    create_analysis_chain,
    merge_analysis_results,
    AnalysisChainError,
    TemplateError,
    PreprocessingError
)

# Logger 설정
logger = get_logger(__name__)

# 분석 유형 정의


class AnalysisType(str, Enum):
    STRENGTH = "strength"  # 강점 분석
    WEAKNESS = "weakness"  # 약점 분석
    INSIGHT = "insight"  # 통찰 분석
    TOTAL = "total"  # 전체 분석


# 캐시 키 생성 함수
def generate_cache_key(analysis_type: AnalysisType, company_name: str, access_time: datetime) -> str:
    """캐시 키 생성 함수"""
    return f"{analysis_type}:{company_name}:{access_time.strftime('%Y-%m-%d')}"

# Redis 클라이언트 초기화
# redis_client = None


# FastAPI 앱 초기화 전, Redis 클라이언트 초기화 부분에 추가
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True,
    max_connections=20,  # 동시 연결 수 제한
    socket_timeout=5,    # 연결 시도 제한 시간
    retry_on_timeout=True,
    health_check_interval=30
)

# 외부 라이브러리 로깅 레벨 설정
for logger_name in ["httpx", "asyncio", "watchfiles.main", "urllib3", "preprocessing", "analysis", "validation", "utils"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# FastAPI 앱 초기화
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# LLM 초기화
common_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,  # 자유도 0
    max_tokens=300,
    streaming=True,  # streaming 활성화
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=OPENAI_API_KEY
)

insight_llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=1.2,  # 창의성 증가
    max_tokens=500,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=OPENAI_API_KEY
)


@app.on_event("startup")
async def startup_event():
    global redis_client
    try:
        # Redis client 초기화 및 연결 테스트
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)}")
        redis_client = None


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if redis_client:
        await redis_client.close()

# FastAPI 앱 초기화 후, 라우트 정의 전에 추가


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} Process Time: {process_time:.2f}s")
    return response


@app.get('/')
async def read_root() -> Dict[str, str]:
    """루트 엔드포인트"""
    return {"message": "Welcome to the Business Analysis API!"}


async def get_cached_data(key: str):
    try:
        data = await redis_client.get(key)
        if data:
            logger.info(f"[Cache] Cache hit - Key: {key}")
            return json.loads(data)
        logger.info(f"[Cache] Cache miss - key: {key}")
        return None
    except redis.RedisError as e:
        logger.error(f"[Cache] Redis error occured: {str(e)}")
        return None

# 캐시 데이터 저장 함수


async def set_cached_data(key: str, data: Any, expire_time: int = 86400):
    """캐시 데이터 저장 함수"""
    try:
        await redis_client.set(key, json.dumps(data), ex=expire_time)
        logger.info(f"[Cache] Data stored - Key: {key}")
    except redis.RedisError as e:
        logger.error(f"[Cache] Failed to store data: {str(e)}")


@retry(
    stop=stop_after_attempt(3),  # 최대 3번 시도
    wait=wait_exponential(
        multiplier=1,  # 첫 시도 후 1초
        min=2,  # 최소 대기 시간
        max=10  # 최대 대기 시간
    )
)
async def calculate_strength_weakness(
    company_name: str,
    access_time: datetime,
    analysis_type: AnalysisType
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """강점과 약점 분석을 수행하고 캐시를 관리하는 함수

    Args:
        company_name (str): 분석할 회사 이름
        access_time (datetime): 분석 시점
        analysis_type (AnalysisType): 분석 유형 (STRENGTH/WEAKNESS/INSIGHT/TOTAL)

    Returns:
        Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str], Optional[str]]:
            - DataFrame: 회사 데이터
            - strength: 강점 지표명
            - weakness: 약점 지표명
            - strength_result: 강점 분석 결과
            - weakness_result: 약점 분석 결과
    """

    # Redis 캐시 확인
    if redis_client:
        try:
            # INSIGHT 분석의 경우 strength와 weakness 모두 필요
            if analysis_type == AnalysisType.INSIGHT:
                # 캐시 키 생성
                strength_key = generate_cache_key(
                    AnalysisType.STRENGTH, company_name, access_time)
                weakness_key = generate_cache_key(
                    AnalysisType.WEAKNESS, company_name, access_time)

                # 캐시 데이터 조회
                logger.info(
                    f"[Cache] Checking cache for analysis type: {analysis_type}")
                cached_strength = await get_cached_data(strength_key)
                cached_weakness = await get_cached_data(weakness_key)

                # 두 캐시가 모두 존재하는 경우
                if cached_strength and cached_weakness:
                    logger.info(
                        f"[Cache] Cache hit - Using {analysis_type} data")
                    return (
                        pd.DataFrame.from_dict(cached_strength[0]),  # 회사 데이터
                        cached_strength[1],  # 강점 지표명
                        cached_weakness[1],  # 약점 지표명
                        cached_strength[2],  # 강점 분석 결과
                        cached_weakness[2]   # 약점 분석 결과
                    )
                logger.info(
                    f"[Cache] Cache miss - No data found for {analysis_type}")

            # 단일 분석(STRENGTH/WEAKNESS) 또는 TOTAL 분석
            else:
                cache_key = generate_cache_key(
                    analysis_type, company_name, access_time)
                cached_data = await get_cached_data(cache_key)

                if cached_data:
                    logger.info(
                        f"[Cache] Cache hit - Using {analysis_type} data")
                    return (
                        pd.DataFrame.from_dict(cached_data[0]),
                        cached_data[1] if analysis_type == AnalysisType.STRENGTH else None,
                        cached_data[1] if analysis_type == AnalysisType.WEAKNESS else None,
                        cached_data[2] if analysis_type == AnalysisType.STRENGTH else None,
                        cached_data[2] if analysis_type == AnalysisType.WEAKNESS else None
                    )
                logger.info(
                    f"[Cache] Cache miss - No data found for {analysis_type}")

        except redis.RedisError as e:
            logger.error(f"[Cache] Redis error occurred: {str(e)}")

    # 캐시 미스 시 새로운 분석 수행
    try:
        from utils import generate_sample_data
        # 샘플 데이터 생성 및 필터링
        df_company_info = await asyncio.to_thread(generate_sample_data, access_time)
        company_data = df_company_info[df_company_info['기업명'] == company_name]

        # 회사 데이터가 없는 경우
        if company_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Data for company {company_name} not found."
            )

        # 강점/약점 지표 결정
        strength_weakness = determine_strength_weakness(
            df_company_info, company_name)
        strength = None
        weakness = None
        strength_result = None
        weakness_result = None

        # 강점 분석 수행
        if analysis_type in [AnalysisType.STRENGTH, AnalysisType.TOTAL, AnalysisType.INSIGHT]:
            strength = strength_weakness['strength']
            strength_chain = await create_analysis_chain(
                strength, True, common_llm, df_company_info, company_name, access_time
            )
            strength_result = ""
            async for chunk in strength_chain.astream(df_company_info):
                strength_result += chunk

        # 약점 분석 수행
        if analysis_type in [AnalysisType.WEAKNESS, AnalysisType.TOTAL, AnalysisType.INSIGHT]:
            weakness = strength_weakness['weakness']
            weakness_chain = await create_analysis_chain(
                weakness, False, common_llm, df_company_info, company_name, access_time
            )
            weakness_result = ""
            async for chunk in weakness_chain.astream(df_company_info):
                weakness_result += chunk

        # 결과 캐싱
        if redis_client:
            # DataFrame을 JSON 직렬화 가능한 형태로 변환
            company_data_dict = company_data.to_dict()
            for key in company_data_dict:
                if isinstance(company_data_dict[key], dict):
                    for k, v in company_data_dict[key].items():
                        if isinstance(v, pd.Timestamp):
                            company_data_dict[key][k] = v.strftime('%Y-%m-%d')

            # 캐시 저장
            cache_key = generate_cache_key(
                analysis_type, company_name, access_time)
            cache_data = (company_data_dict, strength, weakness,
                          strength_result, weakness_result)
            await set_cached_data(cache_key, cache_data)

        return (company_data, strength, weakness, strength_result, weakness_result)

    except Exception as e:
        logger.error(f"[Error] Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/analyze/{company_name}/{analysis_type}')
@limiter.limit("5/minute")  # 분당 5회로 제한
async def analyze_company(
    request: Request,
    company_name: str,
    analysis_type: AnalysisType = Path(
        ...,
        description="Analysis type: strength, weakness, insight, or total"
    )
) -> StreamingResponse:
    """기업 분석을 수행하고 스트리밍 응답을 반환하는 End Point"""

    async def generate_analysis() -> AsyncIterator[str]:
        # 현재 시간을 기준으로 접근 시간 설정
        access_time = get_access_time()
        logger.info(
            f"[Analysis] Starting {analysis_type} analysis for company: {company_name}")

        try:
            # 강점/약점 분석 수행 및 결과 받기
            company_data, strength, weakness, strength_result, weakness_result = \
                await calculate_strength_weakness(company_name, access_time, analysis_type)
            # 분석 타입에 따른 결과 반환
            if analysis_type == AnalysisType.STRENGTH and strength_result:
                # 강점 분석만 요청된 경우
                yield strength_result
            elif analysis_type == AnalysisType.WEAKNESS and weakness_result:
                # 약점 분석만 요청된 경우
                yield weakness_result
            elif analysis_type == AnalysisType.TOTAL:
                # 전체 분석이 요청된 경우
                if strength_result:
                    yield strength_result
                if weakness_result:
                    yield weakness_result
            elif analysis_type == AnalysisType.INSIGHT:
                # 통찰 분석이 요청된 경우
                # strength와 weakness 결과가 모두 필요한지 확인
                if strength_result is None or weakness_result is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Both strength and weakness analysis are required for insight generation"
                    )
                # 통찰 분석 시작
                logger.info("[Final] Starting insight analysis...")

                # 통찰 분석을 위한 prompt template 로드
                insight_prompt_path = os.path.join(
                    PROMPTS_DIR, "insight_template.txt")
                analysis_prompt = await asyncio.to_thread(load_prompt, insight_prompt_path)

                # 통찰 분석 체인 구성
                final_chain = (
                    RunnableLambda(lambda x: {
                        'info': merge_analysis_results(
                            x['strength'], x['weakness']
                        ),
                        'strength_name': strength,
                        'weakness_name': weakness
                    })  # 입력 데이터 전처리
                    # Prompt template 적용
                    | PromptTemplate.from_template(analysis_prompt)
                    | insight_llm  # LLM 연계
                    | StrOutputParser()  # 결과 파싱
                )

                # 통찰 분석 실행 및 결과 스트리밍
                async for chunk in final_chain.astream({
                    'strength': strength_result,
                    'weakness': weakness_result
                }):
                    yield chunk

        except Exception as e:
            # 에러 발생 시 로깅 및 예외 처리
            logger.error(f"[Error] Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            )
    # 스트리밍 응답 반환
    return StreamingResponse(generate_analysis(), media_type="text/event-stream")


@app.get('/query-data/{no_com}')
async def query_data(no_com: int, months: int = 12) -> List[Dict[str, Any]]:
    """회사 데이터를 조회합니다."""
    access_time = get_access_time()
    logger.info(f"[Query] Starting data query for company {no_com}")

    try:
        df = fetch_company_data(company_id=no_com)

        if df.empty:
            logger.error(f"[Error] No data found for company {no_com}")
            raise HTTPException(
                status_code=404,
                detail=f"No data found for company {no_com}"
            )

        validate_input_data(
            df,
            required_columns=['기업명', '날짜', '업종', '총자산', '매출액'],
            company_col='기업명'
        )

        logger.info(
            f"[Query] Data query completed successfully for company {no_com}")
        return df.to_dict(orient='records')

    except DataValidationError as e:
        logger.error(f"[Error] Data validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Data validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[Error] Query error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=["analysis", "database", "preprocessing", "utils"],
        workers=(2 * os.cpu_count() + 1)
    )
