# FastAPI 관련
from sklearn import metrics
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
# @limiter.limit(settings.API_RATE_LIMIT)
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

            # 특정 분석 유형과 메트릭으로 캐시 데이터 조회
            cached_data = await cache.get(
                company_name,
                analysis_type.value,  # AnalysisType enum의 값
                strength_metric if analysis_type == AnalysisType.STRENGTH
                else weakness_metric if analysis_type == AnalysisType.WEAKNESS
                else f"{strength_metric}/{weakness_metric}"
            )
            '''
            cached_data = {
                'strength': {
                    'analysis_metric': None,          # 강점 지표 항목
                    'detailed_result': None,   # 상세 분석 결과
                    'summary': None             # 요약 분석 결과
                },
                'weakness': {
                    'analysis_metric': None,          # 약점 지표 항목
                    'detailed_result': None,   # 상세 분석 결과
                    'summary': None             # 요약 분석 결과
                },
                'insight': {
                    'analysis_metric': None           # strength_metric + '_' + weakness_metric
                    'summary': None              # 통찰 분석 결과
                }
            }
            '''

            # 캐시 데이터 검증
            if cached_data and cache.validate_cache_data(cached_data):
                logger.info(
                    f"[Cache] Successfully loaded cached data for company: {company_name}")

                if analysis_type == AnalysisType.INSIGHT:
                    insight_exists = cached_data['insight'][
                        'analysis_metric'] == f"{strength_metric}/{weakness_metric}"
                    strength_exists = cached_data['strength']['analysis_metric'] == strength_metric
                    weakness_exists = cached_data['weakness']['analysis_metric'] == weakness_metric

                    if insight_exists:
                        # 캐시된 insight 결과 사용
                        logger.info(f"[Cache] Using cached insight analysis")
                        yield cached_data['insight']['summary']
                        return
                    elif strength_exists and weakness_exists:
                        # strength와 weakness는 있지만 insight는 없는 경우
                        logger.info(
                            f"[Cache] Both metrics exist. Performing insight analysis")
                        result = await perform_insight_analysis(company_name, cached_data, insight_llm, cache)
                        yield result['insight']['summary']
                        return
                    else:
                        # 부분적으로만 캐시가 있는 경우
                        result = {}
                        if strength_exists:
                            result['strength'] = cached_data['strength']
                            logger.info(
                                f"[Cache] Using cached strength analysis: {strength_metric}")
                        if weakness_exists:
                            result['weakness'] = cached_data['weakness']
                            logger.info(
                                f"[Cache] Using cached weakness analysis: {weakness_metric}")

                        # 없는 분석만 수행
                        result = await perform_new_analysis(
                            df_company_info,
                            company_name,
                            analysis_type,
                            strength_metric,
                            weakness_metric,
                            result,
                            common_llm,
                            summary_llm,
                            insight_llm,
                            cache
                        )
                        yield result['insight']['summary']
                        return

                elif analysis_type == AnalysisType.STRENGTH:
                    if cached_data['strength']['analysis_metric'] == strength_metric:
                        yield cached_data['strength']['summary']
                        return

                elif analysis_type == AnalysisType.WEAKNESS:
                    if cached_data['weakness']['analysis_metric'] == weakness_metric:
                        yield cached_data['weakness']['summary']
                        return

            # 새로운 분석 수행
            result = await perform_new_analysis(
                df_company_info,
                company_name,
                analysis_type,
                strength_metric,
                weakness_metric,
                None,
                common_llm,
                summary_llm,
                insight_llm,
                cache
            )
            yield await get_analysis_result(result, analysis_type)

        except Exception as e:
            logger.error(f"[Error] Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        generate_analysis(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    # return await generate_analysis() streaming 필요 없을 시


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
                           cached_data.get('strength', {}).get('analysis_metric') == strength_metric)
        weakness_exists = (cached_data and
                           cached_data.get('weakness', {}).get('analysis_metric') == weakness_metric)

        # strength 분석
        if analysis_type in [AnalysisType.STRENGTH, AnalysisType.INSIGHT]:
            if strength_exists:
                result['strength'] = cached_data['strength']
                logger.info(
                    f"[Cache] Using cached strength analysis: {strength_metric}")
            else:
                metrics = determine_strength_weakness(
                    df_company_info,
                    company_name,
                    strength_metric=strength_metric,
                    analysis_type="strength"
                )
                strength = metrics['strength']
                strength_chain = await create_analysis_chain(
                    strength, True, common_llm, df_company_info, company_name, settings.ACCESS_TIME
                )
                logger.info(
                    f"\n\n[Strength; Detailed] Starting strength analysis for {strength}")
                strength_result = ""
                async for chunk in strength_chain.astream(df_company_info):
                    strength_result += chunk

                logger.info(
                    f"\n\n[Strength; Summary] Starting strength analysis for {strength}")
                summary_chain = await create_summary_chain(llm=summary_llm, analysis_metric=strength)
                strength_summary = ""
                async for chunk in summary_chain.astream({"detailed_result": strength_result}):
                    strength_summary += chunk

                result['strength'] = {
                    'analysis_metric': strength,
                    'detailed_result': strength_result,
                    'summary': strength_summary
                }

                if cache:
                    await cache.set(company_name, {'strength': result['strength']}, "strength")

        # weakness 분석
        if analysis_type in [AnalysisType.WEAKNESS, AnalysisType.INSIGHT]:
            if weakness_exists:
                result['weakness'] = cached_data['weakness']
                logger.info(
                    f"[Cache] Using cached weakness analysis: {weakness_metric}")
            else:
                metrics = determine_strength_weakness(
                    df_company_info,
                    company_name,
                    weakness_metric=weakness_metric,
                    analysis_type='weakness'
                )
                weakness = metrics['weakness']
                weakness_chain = await create_analysis_chain(
                    weakness, False, common_llm, df_company_info, company_name, settings.ACCESS_TIME
                )
                logger.info(
                    f"\n\n[Weakness; Detailed] Starting weakness analysis for {weakness}")
                weakness_result = ""
                async for chunk in weakness_chain.astream(df_company_info):
                    weakness_result += chunk

                logger.info(
                    f"\n\n[Weakness; Summary] Starting weakness analysis for {weakness}")
                summary_chain = await create_summary_chain(llm=summary_llm, analysis_metric=weakness)
                weakness_summary = ""
                async for chunk in summary_chain.astream({"detailed_result": weakness_result}):
                    weakness_summary += chunk

                result['weakness'] = {
                    'analysis_metric': weakness,
                    'detailed_result': weakness_result,
                    'summary': weakness_summary
                }

                if cache:
                    await cache.set(company_name, {'weakness': result['weakness']}, "weakness")

        # insight 분석
        if analysis_type == AnalysisType.INSIGHT:
            insight_result = await perform_insight_analysis(company_name, result, insight_llm, cache)
            result['insight'] = insight_result['insight']

        return result

    except Exception as e:
        logger.error(f"[Error] Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def perform_insight_analysis(company_name: str, data: Dict, insight_llm: ChatOpenAI, cache: CacheManager) -> Dict:
    """Insight 분석만 수행하는 함수"""
    try:

        # data 유효성 검증 추가
        if not data or 'strength' not in data or 'weakness' not in data:
            raise ValueError("Invalid data structure for insight analysis")

        insight_prompt_path = os.path.join(
            settings.PROMPTS_DIR, "insight_template.txt")
        analysis_prompt = await asyncio.to_thread(load_prompt, insight_prompt_path)

        final_chain = (
            RunnableLambda(lambda x: {
                'info': merge_analysis_results(x['strength'], x['weakness']),
                'strength_metric': data['strength']['analysis_metric'],
                'weakness_metric': data['weakness']['analysis_metric']
            })
            | PromptTemplate.from_template(analysis_prompt)
            | insight_llm
            | StrOutputParser()
        )

        logger.info(
            f"\n\n[Insight] Starting insight analysis using strength({data['strength']['analysis_metric']}) and weakness({data['weakness']['analysis_metric']})")

        insight_result = ""
        async for chunk in final_chain.astream({
            'strength': data['strength']['detailed_result'],
            'weakness': data['weakness']['detailed_result']
        }):
            insight_result += chunk

        result = {
            'strength': data['strength'],
            'weakness': data['weakness'],
            'insight': {
                'analysis_metric': f"{data['strength']['analysis_metric']}/{data['weakness']['analysis_metric']}",
                'summary': insight_result
            }
        }

        if cache:
            await cache.set(company_name, result, "insight")

        return result

    except Exception as e:
        logger.error(f"[Error] Insight analysis failed: {str(e)}")
        raise
