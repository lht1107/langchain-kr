from typing import Tuple
import os
from enum import Enum
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
import uvicorn
import pandas as pd
from datetime import datetime
from typing import AsyncIterator, Dict, Any, Optional, List, Tuple
import asyncio
from utils import load_prompt
from utils.logger import get_logger
from utils.time_utils import get_access_time
from database import f_read_pg_sql, generate_sql_query
from analysis import (
    determine_strength_weakness,
    create_analysis_chain,
    merge_analysis_results,
    AnalysisChainError,
    TemplateError,
    PreprocessingError
)
from utils.validation import validate_input_data, DataValidationError
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from AI경영진단보고서.config_bk import OPENAI_API_KEY, PROMPTS_DIR
import logging
from typing_extensions import TypedDict
from database.f_read_pg_sql import fetch_company_data
from functools import lru_cache

# Logger 설정
logger = get_logger(__name__)

# 분석 유형 정의


class AnalysisType(str, Enum):
    STRENGTH = "strength"
    WEAKNESS = "weakness"
    INSIGHT = "insight"
    TOTAL = "total"


# 외부 라이브러리 로깅 레벨 설정
for logger_name in ["httpx", "asyncio", "watchfiles.main", "urllib3", "preprocessing", "analysis", "validation", "utils"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# FastAPI 앱 초기화
app = FastAPI()

# LLM 초기화
common_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    max_tokens=300,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=OPENAI_API_KEY
)

insight_llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=1.2,
    max_tokens=500,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=OPENAI_API_KEY
)


@app.get('/')
async def read_root() -> Dict[str, str]:
    """루트 엔드포인트"""
    return {"message": "Welcome to the Business Analysis API!"}


@lru_cache(maxsize=100)
def calculate_strength_weakness(company_name: str, access_time: str) -> Tuple[str, str, str, str]:
    """
    특정 회사의 strength와 weakness를 계산하고 분석 결과를 캐시에 저장합니다.
    Returns:
        - strength: 강점 항목
        - weakness: 약점 항목
        - strength_result: 강점 분석 결과
        - weakness_result: 약점 분석 결과
    """
    df_company_info = fetch_company_data(company_name=company_name)
    if df_company_info.empty:
        raise ValueError(f"No data found for company {company_name}")

    # 강점과 약점 항목 결정
    strength_weakness = determine_strength_weakness(
        df_company_info, company_name)
    strength = strength_weakness['strength']
    weakness = strength_weakness['weakness']

    # 강점 분석 결과 생성
    strength_chain = create_analysis_chain(
        strength, True, common_llm, df_company_info, company_name, access_time)
    strength_result = ''.join(
        chunk async for chunk in strength_chain.astream(df_company_info))

    # 약점 분석 결과 생성
    weakness_chain = create_analysis_chain(
        weakness, False, common_llm, df_company_info, company_name, access_time)
    weakness_result = ''.join(
        chunk async for chunk in weakness_chain.astream(df_company_info))

    return strength, weakness, strength_result, weakness_result


@app.get('/analyze/{company_name}/{analysis_type}')
async def analyze_company(
    company_name: str,
    analysis_type: AnalysisType = Path(
        ..., description="Analysis type: strength, weakness, insight, or total")
) -> StreamingResponse:
    """기업 분석을 수행하고 스트리밍 응답을 반환합니다."""

    async def generate_analysis() -> AsyncIterator[str]:
        access_time = get_access_time()
        logger.info(
            f"[Analysis] Starting {analysis_type} analysis for company: {company_name}")

        try:
            # 데이터 준비
            from utils import generate_sample_data
            df_company_info = await asyncio.to_thread(
                generate_sample_data,
                access_time
            )
            logger.debug(f"[Data] Generated sample data for {company_name}")

            company_data = df_company_info[df_company_info['기업명']
                                           == company_name]
            if company_data.empty:
                logger.error(
                    f"[Error] No data found for company {company_name}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Data for company {company_name} not found."
                )

            # 강점과 약점 결정
            strength_weakness = await asyncio.to_thread(
                determine_strength_weakness,
                df_company_info,
                company_name
            )
            strength = strength_weakness['strength']
            weakness = strength_weakness['weakness']
            logger.info(
                f"[Analysis] Identified strength: {strength}, weakness: {weakness}")

            # 분석 시작 (insight 제외)
            # if analysis_type != AnalysisType.INSIGHT:
            #     yield f"[Analysis Start] Analyzing {company_name}\n"

            # 강점 분석
            strength_result = ""
            if analysis_type in [AnalysisType.STRENGTH, AnalysisType.TOTAL]:
                try:
                    logger.info("[Strength] Starting strength analysis...")

                    if analysis_type == AnalysisType.TOTAL:
                        yield "\n[Strength Analysis]\n"
                    strength_output = []

                    strength_chain = await create_analysis_chain(
                        strength, True, common_llm,
                        df_company_info, company_name, access_time
                    )

                    async for chunk in strength_chain.astream(company_data):
                        yield chunk
                        strength_output.append(chunk)

                    strength_result = ''.join(strength_output)
                    logger.info("[Strength] Analysis completed successfully")
                except AnalysisChainError as e:
                    logger.error(
                        f"[Error] Failed to create strength analysis chain: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            # 강점 분석 (silent - insight용)
            elif analysis_type == AnalysisType.INSIGHT:
                try:
                    logger.info(
                        "[Strength] Running silent strength analysis...")
                    strength_output = []
                    strength_chain = await create_analysis_chain(
                        strength, True, common_llm,
                        df_company_info, company_name, access_time
                    )
                    async for chunk in strength_chain.astream(company_data):
                        strength_output.append(chunk)
                    strength_result = ''.join(strength_output)
                except AnalysisChainError as e:
                    logger.error(
                        f"[Error] Failed to create strength analysis chain: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            # 약점 분석
            weakness_result = ""
            if analysis_type in [AnalysisType.WEAKNESS, AnalysisType.TOTAL]:
                try:
                    logger.info("[Weakness] Starting weakness analysis...")
                    if analysis_type == AnalysisType.TOTAL:
                        yield "\n[Weakness Analysis]\n"
                    weakness_output = []

                    weakness_chain = await create_analysis_chain(
                        weakness, False, common_llm,
                        df_company_info, company_name, access_time
                    )

                    async for chunk in weakness_chain.astream(company_data):
                        yield chunk
                        weakness_output.append(chunk)

                    weakness_result = ''.join(weakness_output)
                    logger.info("[Weakness] Analysis completed successfully")
                except AnalysisChainError as e:
                    logger.error(
                        f"[Error] Failed to create weakness analysis chain: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            # 약점 분석 (silent - insight용)
            elif analysis_type == AnalysisType.INSIGHT:
                try:
                    logger.info(
                        "[Weakness] Running silent weakness analysis...")
                    weakness_output = []
                    weakness_chain = await create_analysis_chain(
                        weakness, False, common_llm,
                        df_company_info, company_name, access_time
                    )
                    async for chunk in weakness_chain.astream(company_data):
                        weakness_output.append(chunk)
                    weakness_result = ''.join(weakness_output)
                except AnalysisChainError as e:
                    logger.error(
                        f"[Error] Failed to create weakness analysis chain: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            # 최종 분석 (insight 또는 total)
            if analysis_type in [AnalysisType.INSIGHT, AnalysisType.TOTAL]:
                try:
                    logger.info("[Final] Starting final analysis...")
                    if analysis_type == AnalysisType.TOTAL:
                        yield "\n[Final Analysis]\n"

                    insight_prompt_path = os.path.join(
                        PROMPTS_DIR, "insight_template.txt")
                    analysis_prompt = await asyncio.to_thread(load_prompt, insight_prompt_path)

                    final_chain = (
                        RunnableLambda(lambda x: {
                            'info': merge_analysis_results(
                                x['strength'],
                                x['weakness']
                            ),
                            'strength_name': strength,
                            'weakness_name': weakness
                        })
                        | PromptTemplate.from_template(analysis_prompt)
                        | insight_llm
                        | StrOutputParser()
                    )

                    async for chunk in final_chain.astream({
                        'strength': strength_result,
                        'weakness': weakness_result
                    }):
                        yield chunk

                    logger.info(
                        f"[Complete] Analysis completed successfully for {company_name}")
                except FileNotFoundError as e:
                    logger.error(
                        f"[Error] Failed to load insight template: {str(e)}")
                    raise HTTPException(
                        status_code=500, detail="Failed to load analysis template")

            # 완료 메시지 (insight 제외)
            if analysis_type != AnalysisType.INSIGHT:
                yield "\n[Analysis Complete]"

        except Exception as e:
            logger.error(f"[Error] Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            )

    return StreamingResponse(
        generate_analysis(),
        media_type="text/event-stream"
    )


@app.get('/query-data/{no_com}')
async def query_data(no_com: int, months: int = 12) -> List[Dict[str, Any]]:
    """회사 데이터를 조회합니다."""
    access_time = get_access_time().strftime('%Y-%m-%d')
    logger.info(f"[Query] Starting data query for company {no_com}")

    try:
        # 회사 데이터 조회를 위한 API 호출
        df = fetch_company_data(company_id=no_com)  # company_id로 회사 조회

        if df.empty:
            logger.error(f"[Error] No data found for company {no_com}")
            raise HTTPException(
                status_code=404,
                detail=f"No data found for company {no_com}"
            )

        # 데이터 검증
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
        reload_dirs=["analysis", "database", "preprocessing", "utils"]
    )
