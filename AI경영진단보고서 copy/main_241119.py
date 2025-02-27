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
import platform
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
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
from config_bk import settings
from utils import load_prompt
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


# 캐시 키 생성 함수
def generate_cache_key(company_name: str, access_time: datetime) -> str:
    """캐시 키 생성 함수"""
    return f"{company_name}:{access_time.strftime('%Y-%m-%d')}"


# Redis 클라이언트 초기화 변경
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
    socket_timeout=5,
    retry_on_timeout=True,
    health_check_interval=30
)

# 외부 라이브러리 로깅 레벨 설정
for logger_name in ["httpx", "asyncio", "watchfiles.main", "urllib3", "preprocessing", "analysis", "validation", "utils"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup 이벤트 로직
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

    yield

    # shutdown 이벤트 로직
    if redis_client:
        await redis_client.close()

# FastAPI 앱 초기화 시 lifespan 매니저 추가
app = FastAPI(lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# LLM 초기화
common_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,  # 자유도 0
    max_tokens=300,
    streaming=True,  # streaming 활성화
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=settings.OPENAI_API_KEY
)

insight_llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=1.2,  # 창의성 증가
    max_tokens=500,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=settings.OPENAI_API_KEY
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} Process Time: {process_time:.2f}s")
    return response


async def get_cached_data(key: str):
    """캐시 데이터를 조회하는 함수"""
    try:
        data = await redis_client.get(key)
        if data:
            # 캐시 히트 시에는 로그를 최소화
            logger.info(f"[Cache] Cache hit - Key: {key}")
            return json.loads(data)
        # 캐시 미스 시에만 상세 로그
        if redis_client:
            logger.info(
                f"[Cache] Cache miss - Initializing new analysis for key: {key}")
        return None
    except redis.RedisError as e:
        logger.error(f"[Cache] Redis error occured: {str(e)}")
        return None


class AnalysisCache(TypedDict):
    company_data: Dict  # index 0: 회사 데이터
    analysis_status: Dict[str, bool]  # index 1: 각 분석 타입별 수행 여부
    strength: str  # index 2: 강점 분석 항목
    weakness: str  # index 3: 약점 분석 항목
    strength_result: Optional[str]  # index 4: 강점 분석 결과
    weakness_result: Optional[str]  # index 5: 약점 분석 결과
    insight_result: Optional[str]  # index 6: 통찰 분석 결과

# Timestamp 직렬화 함수


def serialize_datetime(obj):
    '''
    Timestamp 객체를 문자열로 직렬화하는 함수
    사용: json.dumps(data, default=serialize_datatime)
    '''
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f"Type {type(obj)} not serializable")

# 캐시 데이터 저장 함수


async def set_cached_data(
    key: str,
    data: Dict,
    analysis_type: AnalysisType,
    expire_time: int = settings.CACHE_EXPIRE_TIME
):
    """캐시 데이터를 저장하는 함수

    Args:
        key: 캐시 키
        data: 저장할 데이터
        analysis_type: 분석 타입
        expire_time: 캐시 만료 시간 (기본 24시간)
    """

    if not redis_client:
        logger.warning("[Cache] Redis client not available")
        return

    try:
        # 기존 캐시 데이터 조회 및 기본 구조 생성
        cached_data = await get_cached_data(key)

        if not cached_data:
            # 새로운 캐시데이터 생성
            cached_data = {
                'company_data': None,
                'analysis_status': {
                    'strength': False,
                    'weakness': False,
                    'insight': False
                },
                'strength': None,
                'weakness': None,
                'strength_result': None,
                'weakness_result': None,
                'insight_result': None
            }
            # 현재 상태 로깅
            # logger.info(
            #     f"[Cache] Initializing new cache - Key: {key}, "
            #     f"Strength: {cached_data['analysis_status']['strength']}, "
            #     f"Weakness: {cached_data['analysis_status']['weakness']}, "
            #     f"Insight: {cached_data['analysis_status']['insight']}"
            # )

        else:
            logger.info(f"[Cache] Updating existing cache - Key: {key}")

        # company_data가 없는 경우에만 업데이트
        if not cached_data['company_data'] and data.get('company_data'):
            cached_data['company_data'] = data['company_data']

        # 분석 타입별 결과 업데이트
        match analysis_type:
            case AnalysisType.STRENGTH:
                cached_data.update({
                    'strength': data.get('strength'),
                    'strength_result': data.get('strength_result'),
                    'analysis_status': {
                        **cached_data['analysis_status'],
                        'strength': True
                    }
                })
            case AnalysisType.WEAKNESS:
                cached_data.update({
                    'weakness': data.get('weakness'),
                    'weakness_result': data.get('weakness_result'),
                    'analysis_status': {
                        **cached_data['analysis_status'],
                        'weakness': True
                    }
                })
            case AnalysisType.INSIGHT:
                # INSIGHT 수행 시 strength와 weakness 결과도 함께 저장
                cached_data.update({
                    'strength': data.get('strength'),
                    'strength_result': data.get('strength_result'),
                    'weakness': data.get('weakness'),
                    'weakness_result': data.get('weakness_result'),
                    'insight_result': data.get('insight_result'),
                    'analysis_status': {
                        **cached_data['analysis_status'],
                        'strength': True,
                        'weakness': True,
                        'insight': True
                    }
                })

        # 업데이트된 상태 로깅
        logger.info(
            f"[Cache] Updated Status - Key: {key}, "
            f"Strength: {cached_data['analysis_status']['strength']}, "
            f"Weakness: {cached_data['analysis_status']['weakness']}, "
            f"Insight: {cached_data['analysis_status']['insight']}"
        )

        # Redis에 저장
        await redis_client.set(
            key,
            json.dumps(cached_data, default=serialize_datetime),
            ex=expire_time
        )
        logger.info(f"[Cache] Data stored - Key: {key}")

    except Exception as e:
        logger.error(f"[Cache] Failed to process data: {str(e)}")


# 데이터 유효성 검증 함수 추가
def validate_cache_data(data: Dict) -> bool:
    if not isinstance(data, dict):
        return False
    required_fields = {'company_data', 'analysis_status'}
    return all(field in data for field in required_fields)


@retry(
    stop=stop_after_attempt(3),  # 최대 3번 시도
    wait=wait_exponential(
        multiplier=1,  # 첫 시도 후 1초
        min=2,  # 최소 대기 시간
        max=10  # 최대 대기 시간
    )
)
async def perform_new_analysis(
    company_name: str,
    access_time: datetime,
    analysis_type: AnalysisType,
    cached_data: Optional[Dict] = None
) -> Dict:
    """새로운 분석을 수행하는 함수"""
    try:
        # 데이터 생성 또는 캐시된 데이터 사용
        if cached_data and cached_data.get('company_data'):
            company_data = pd.DataFrame.from_dict(cached_data['company_data'])
            df_company_info = company_data
        else:
            from utils import generate_sample_data
            df_company_info = await asyncio.to_thread(generate_sample_data, access_time)
            company_data = df_company_info[df_company_info['기업명']
                                           == company_name]

        if company_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Data for company {company_name} not found."
            )

        # DataFrame을 딕셔너리로 변환하기 전에 Timestamp 처리
        company_data_dict = company_data.to_dict()
        for key in company_data_dict:
            if isinstance(company_data_dict[key], dict):
                for k, v in company_data_dict[key].items():
                    if isinstance(v, pd.Timestamp):
                        company_data_dict[key][k] = v.strftime('%Y-%m-%d')

        # 결과 초기화
        result = {
            'company_data': company_data.to_dict(),
            'analysis_status': {
                'strength': False,
                'weakness': False,
                'insight': False
            }
        }

        # 캐시된 데이터 확인 및 활용
        strength_exists = cached_data['analysis_status'].get(
            'strength', False) if cached_data else False
        weakness_exists = cached_data['analysis_status'].get(
            'weakness', False) if cached_data else False

        # strength 분석이 필요한 경우
        if analysis_type in [AnalysisType.STRENGTH, AnalysisType.INSIGHT]:
            if strength_exists:
                strength = cached_data.get('strength')
                strength_result = cached_data.get('strength_result')
                logger.info("[Cache] Using cached strength analysis")
            else:
                strength_weakness = determine_strength_weakness(
                    df_company_info, company_name)
                strength = strength_weakness['strength']
                strength_chain = await create_analysis_chain(
                    strength, True, common_llm, df_company_info, company_name, access_time
                )
                logger.info("[Strength] Starting strength analysis...")
                strength_result = ""
                async for chunk in strength_chain.astream(df_company_info):
                    strength_result += chunk

            result.update({
                'strength': strength,
                'strength_result': strength_result,
                'analysis_status': {**result['analysis_status'], 'strength': True}
            })

        # weakness 분석이 필요한 경우
        if analysis_type in [AnalysisType.WEAKNESS, AnalysisType.INSIGHT]:
            if weakness_exists:
                weakness = cached_data.get('weakness')
                weakness_result = cached_data.get('weakness_result')
                logger.info("[Cache] Using cached weakness analysis")
            else:
                if not 'strength_weakness' in locals():
                    strength_weakness = determine_strength_weakness(
                        df_company_info, company_name)
                weakness = strength_weakness['weakness']
                weakness_chain = await create_analysis_chain(
                    weakness, False, common_llm, df_company_info, company_name, access_time
                )
                logger.info("[Weakness] Starting weakness analysis...")
                weakness_result = ""
                async for chunk in weakness_chain.astream(df_company_info):
                    weakness_result += chunk

            result.update({
                'weakness': weakness,
                'weakness_result': weakness_result,
                'analysis_status': {**result['analysis_status'], 'weakness': True}
            })

        # insight 분석이 필요한 경우
        if analysis_type == AnalysisType.INSIGHT:
            insight_prompt_path = os.path.join(
                settings.PROMPTS_DIR, "insight_template.txt")
            analysis_prompt = await asyncio.to_thread(load_prompt, insight_prompt_path)
            final_chain = (
                RunnableLambda(lambda x: {
                    'info': merge_analysis_results(x['strength'], x['weakness']),
                    'strength_name': result.get('strength'),
                    'weakness_name': result.get('weakness')
                })
                | PromptTemplate.from_template(analysis_prompt)
                | insight_llm
                | StrOutputParser()
            )

            logger.info("[Insight] Starting insight analysis...")
            insight_result = ""
            async for chunk in final_chain.astream({
                'strength': result.get('strength_result', ''),
                'weakness': result.get('weakness_result', '')
            }):
                insight_result += chunk

            result.update({
                'insight_result': insight_result,
                'analysis_status': {**result['analysis_status'], 'insight': True}
            })

        # 결과 캐싱
        if redis_client:
            cache_key = generate_cache_key(company_name, access_time)
            await set_cached_data(cache_key, result, analysis_type)

        return result

    except Exception as e:
        logger.error(f"[Error] Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(
        f"[Error] HTTP {exc.status_code}: {exc.detail} | Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )


if __name__ == "__main__":
    if platform.system() == "Windows":
        # Windows 환경에서는 Uvicorn 사용
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            reload_dirs=["analysis", "database", "preprocessing", "utils"],
            # workers=(2 * os.cpu_count() + 1)
        )
    else:
        # Linux/Mac 환경에서는 Gunicorn 사용
        import multiprocessing
        from gunicorn.app.base import BaseApplication

        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                config = {key: value for key, value in self.options.items()
                          if key in self.cfg.settings and value is not None}
                for key, value in config.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            'bind': '0.0.0.0:8000',
            'workers': 2 * multiprocessing.cpu_count() + 1,
            'worker_class': 'uvicorn.workers.UvicornWorker',
            'reload': True,
            'reload_extra_files': ['analysis', 'database', 'preprocessing', 'utils'],
            'loglevel': 'info'
        }

        StandaloneApplication(app, options).run()
