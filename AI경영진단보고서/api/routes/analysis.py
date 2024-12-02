# FastAPI 관련
from fastapi import APIRouter, Path, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse

# 타입 힌트
from typing import AsyncIterator, Dict, Optional, Tuple
from enum import Enum

# 비동기 및 유틸리티
import asyncio
import os
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# 데이터 처리
import pandas as pd

# 내부 모듈
from core.dependencies import get_llm_clients, limiter, get_cache
from core.config import settings
from core.cache import CacheManager

# 유틸리티
from utils.logger import get_logger
from utils.time_utils import get_access_time
from utils.load_prompt import load_prompt

# 분석 관련
from analysis import (
    determine_strength_weakness,
    create_analysis_chain,
    merge_analysis_results,
    create_summary_chain,
    AnalysisChainError,
    TemplateError,
    PreprocessingError
)

# LangChain 컴포넌트
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

# Logger 초기화
logger = get_logger(__name__)

# 분석 타입 정의


class AnalysisType(str, Enum):
    STRENGTH = "strength"
    WEAKNESS = "weakness"
    INSIGHT = "insight"


# 라우터 설정
router = APIRouter(
    prefix="/analyze",
    tags=["analysis"]
)

# 메인 엔드포인트


@router.get('/{company_name}/{analysis_type}')
@limiter.limit(settings.API_RATE_LIMIT)
async def analyze_company(
    request: Request,
    company_name: str,
    analysis_type: AnalysisType = Path(
        ...,
        description="Analysis type: strength, weakness, insight"
    ),
    strength_metric: Optional[str] = None,  # 사용자 강점 지정
    weakness_metric: Optional[str] = None,  # 사용자 약점 지정
    cache: CacheManager = Depends(get_cache),
    llm_clients: Tuple[ChatOpenAI, ChatOpenAI,
                       ChatOpenAI] = Depends(get_llm_clients)
) -> StreamingResponse:  # streaming이 필요 없으면 "-> Dict" 로 변환
    """기업 분석을 수행하고 스트리밍 응답을 반환하는 End Point"""

    ''' 
    아래 부분은 수정 필요
    '''
    # 분석 시작 시간 설정
    access_time = get_access_time()

    # # 데이터 생성을 상위 레벨로 이동
    # from utils import generate_sample_data
    # df_company_info = await asyncio.to_thread(generate_sample_data, access_time)

    sample_data_path = os.path.join(settings.BASE_DIR, 'sample.parquet')
    df_company_info = pd.read_parquet(sample_data_path)

    # llm 초기화
    common_llm, summary_llm, insight_llm = llm_clients

    # streaming이 필요 없으면 "Dict"로 변환
    async def generate_analysis() -> AsyncIterator[str]:
        """분석 결과를 생성하는 비동기 제너레이터"""
        try:
            logger.info(
                f"[Analysis] Starting {analysis_type} analysis for company: {company_name}")

            # 캐시 키 생성 및 데이터 조회
            cache_key = cache.generate_cache_key(company_name, access_time)
            cache_key = cache.generate_cache_key(company_name, access_time)
            cached_data = await cache.get(cache_key, analysis_type,
                                          strength_metric if analysis_type == "strength"
                                          else weakness_metric if analysis_type == "weakness"
                                          else f"{strength_metric}/{weakness_metric}" if analysis_type == "insight"
                                          else None
                                          )
            '''
            cached_data = {
                'strength': {
                    'indicator': None,          # 강점 지표 항목
                    'detailed_result': None,   # 상세 분석 결과
                    'summary': None             # 요약 분석 결과
                },
                'weakness': {
                    'indicator': None,          # 약점 지표 항목
                    'detailed_result': None,   # 상세 분석 결과
                    'summary': None             # 요약 분석 결과
                },
                'insight': {
                    'indicator': None           # strength_indicator + '_' + weakness_indicator
                    'summary': None              # 통찰 분석 결과
                }
            }
            '''

            # 캐시 데이터 검증
            if cached_data and not cache.validate_cache_data(cached_data):
                logger.warning(
                    f"[Cache] Invalid cache data structure - Key: {cache_key}")
                cached_data = None  # 유효하지 않은 캐시 데이터는 무시

            # 캐시된 데이터가 있고, 해당 분석의 indicator가 존재하는 경우에만 캐시 사용
            if cached_data:
                if (analysis_type == AnalysisType.STRENGTH and
                    cached_data['strength']['indicator'] == strength_metric) or \
                    (analysis_type == AnalysisType.WEAKNESS and
                     cached_data['weakness']['indicator'] == weakness_metric) or \
                    (analysis_type == AnalysisType.INSIGHT and
                        cached_data['insight']['indicator'] == f"{strength_metric}_{weakness_metric}"):
                    result = await get_cached_result(cached_data, analysis_type)
                    if result:
                        yield result
                        return

            # 새로운 분석 수행
            result = await perform_new_analysis(
                df_company_info, company_name, access_time, analysis_type, strength_metric, weakness_metric, cached_data, common_llm, summary_llm, insight_llm, cache
            )

            # 분석 결과 반환
            yield await get_analysis_result(result, analysis_type)

        except HTTPException as e:
            logger.error(f"[Error] HTTP Exception: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"[Error] Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            )

    return StreamingResponse(
        generate_analysis(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        # return await generate_analysis() streaming 필요 없을 시
    )


async def get_cached_result(cached_data: Dict, analysis_type: AnalysisType) -> Optional[str]:
    """캐시된 결과 반환"""
    if analysis_type == AnalysisType.STRENGTH:
        logger.info(
            f"[Cache] Using cached strength analysis for {cached_data['strength']['indicator']}")
        return cached_data['strength']['summary']
    elif analysis_type == AnalysisType.WEAKNESS:
        logger.info(
            f"[Cache] Using cached weakness analysis for {cached_data['weakness']['indicator']}")
        return cached_data['weakness']['summary']
    elif analysis_type == AnalysisType.INSIGHT:
        logger.info(f"[Cache] Using cached insight analysis")
        return cached_data['insight']['summary']
    return None


async def get_analysis_result(result: Dict, analysis_type: AnalysisType) -> str:
    """분석 결과 추출"""
    if analysis_type == AnalysisType.STRENGTH:
        return result['strength']['summary']
    elif analysis_type == AnalysisType.WEAKNESS:
        return result['weakness']['summary']
    elif analysis_type == AnalysisType.INSIGHT:
        if result['insight']['summary']:
            return result['insight']['summary']
        raise HTTPException(
            status_code=500,
            detail="Failed to generate insight analysis"
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,  # 원래 예외를 그대로 발생
    before_sleep=lambda retry_state: logger.warning(  # 재시도 전 로그
        f"Retry attempt {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    )
)
async def perform_new_analysis(
    df_company_info: pd.DataFrame,
    company_name: str,
    access_time: datetime,
    analysis_type: AnalysisType,
    strength_metric: Optional[str] = None,  # 사용자 강점 지정
    weakness_metric: Optional[str] = None,  # 사용자 약점 지정
    cached_data: Optional[Dict] = None,
    common_llm: Optional[ChatOpenAI] = None,
    summary_llm: Optional[ChatOpenAI] = None,
    insight_llm: Optional[ChatOpenAI] = None,
    cache: Optional[CacheManager] = None
) -> Dict:
    """새로운 분석을 수행하는 함수"""

    cache_key = cache.generate_cache_key(company_name, access_time)

    try:
        # 회사 데이터 조회
        company_data = df_company_info[df_company_info['기업명']
                                       == company_name]

        if company_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Data for company {company_name} not found."
            )
        result = {}

        # 캐시된 데이터 확인
        strength_exists = (cached_data and
                           cached_data['strength']['indicator'] == strength_metric)
        weakness_exists = (cached_data and
                           cached_data['weakness']['indicator'] == weakness_metric)

        # strength 분석이 필요한 경우
        if analysis_type in [AnalysisType.STRENGTH, AnalysisType.INSIGHT]:
            if strength_exists:
                strength = cached_data['strength']['indicator']
                strength_result = cached_data['strength']['detailed_result']
                strength_summary = cached_data['strength']['summary']
            else:
                strength_weakness = determine_strength_weakness(
                    df_company_info,
                    company_name,
                    strength_metric,
                    weakness_metric
                )
                strength = strength_weakness['strength']
                strength_chain = await create_analysis_chain(
                    strength, True, common_llm, df_company_info, company_name, access_time
                )
                logger.info(
                    f"\n\n[Strength; Detailed] Starting strength analysis for {strength}")
                strength_result = ""
                async for chunk in strength_chain.astream(df_company_info):
                    strength_result += chunk

                logger.info(
                    f"\n\n[Strength; Summary] Starting strength analysis for {strength}")
                summary_chain = await create_summary_chain(llm=summary_llm, indicator=strength)
                strength_summary = ""
                async for chunk in summary_chain.astream({"detailed_result": strength_result}):
                    strength_summary += chunk

            result['strength'] = {
                'indicator': strength,
                'detailed_result': strength_result,
                'summary': strength_summary
            }

            if cache:
                await cache.set(cache_key, {'strength': result['strength']}, "strength")

        # weakness 분석이 필요한 경우
        if analysis_type in [AnalysisType.WEAKNESS, AnalysisType.INSIGHT]:
            if weakness_exists:
                weakness = cached_data['weakness']['indicator']
                weakness_result = cached_data['weakness']['detailed_result']
                weakness_summary = cached_data['weakness']['summary']
            else:
                if not 'strength_weakness' in locals():
                    strength_weakness = determine_strength_weakness(
                        df_company_info,
                        company_name,
                        strength_metric,
                        weakness_metric
                    )
                weakness = strength_weakness['weakness']
                weakness_chain = await create_analysis_chain(
                    weakness, False, common_llm, df_company_info, company_name, access_time
                )
                logger.info(
                    f"\n\n[Weakness; Detailed] Starting weakness analysis for {weakness}")
                weakness_result = ""
                async for chunk in weakness_chain.astream(df_company_info):
                    weakness_result += chunk

                logger.info(
                    f"\n\n[Weakness; Summary] Starting weakness analysis for {weakness}")
                summary_chain = await create_summary_chain(llm=summary_llm, indicator=weakness)
                weakness_summary = ""
                async for chunk in summary_chain.astream({"detailed_result": weakness_result}):
                    weakness_summary += chunk

            result['weakness'] = {
                'indicator': weakness,
                'detailed_result': weakness_result,
                'summary': weakness_summary
            }

            if cache:
                await cache.set(cache_key, {'weakness': result['weakness']}, "weakness")

        # insight 분석이 필요한 경우
        if analysis_type == AnalysisType.INSIGHT:
            insight_prompt_path = os.path.join(
                settings.PROMPTS_DIR, "insight_template.txt")
            analysis_prompt = await asyncio.to_thread(load_prompt, insight_prompt_path)
            final_chain = (
                RunnableLambda(lambda x: {
                    'info': merge_analysis_results(x['strength'], x['weakness']),
                    'strength_name': result['strength']['indicator'],
                    'weakness_name': result['weakness']['indicator']
                })
                | PromptTemplate.from_template(analysis_prompt)
                | insight_llm
                | StrOutputParser()
            )

            logger.info(
                f"\n\n[Insight] Starting insight analysis by using strength({result['strength']['indicator']}) and weakness({result['weakness']['indicator']})")
            insight_result = ""
            async for chunk in final_chain.astream({
                'strength': result['strength']['detailed_result'],
                'weakness': result['weakness']['detailed_result']
            }):
                insight_result += chunk

            result['insight'] = {
                'indicator': f"{result['strength']['indicator']}/{result['weakness']['indicator']}",
                'summary': insight_result
            }

            # 결과 캐싱
            if cache:
                await cache.set(cache_key, {'insight': result['insight']}, "insight")

        return result

    except Exception as e:
        logger.error(f"[Error] Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
