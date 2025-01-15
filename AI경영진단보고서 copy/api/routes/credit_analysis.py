from core.dependencies import get_llm_clients, limiter, get_cache
from typing import Dict, Tuple, AsyncIterator
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from enum import Enum
import os
import json
import pandas as pd
from datetime import datetime
import joblib
from tenacity import retry, stop_after_attempt, wait_exponential
import xgboost as xgb
from pydantic import BaseModel, Field

from core.config import settings
from core.cache import CacheManager
from utils.logger import get_logger
from utils.parser import output_parser
from database.sqlite_credit_cache import SQLiteCreditCache
from analysis.credit_analyzer import ShapAnalyzer
from prompts.current_credit_prompt import create_current_analysis_chain
from prompts.hypothetical_credit_prompt import create_hypothetical_analysis_chain
from langchain_openai import ChatOpenAI
from analysis import (
    create_analysis_chain,
    merge_analysis_results,
    create_summary_chain,
)

import numpy as np

# Logger 초기화
logger = get_logger(__name__)

# 라우터 설정
router = APIRouter(
    prefix="/credit_analysis",
    tags=["credit_analysis"]
)

# 모델과 데이터 로드
X_test = pd.read_pickle(os.path.join(settings.DATA_PATH, 'X_test.pkl'))
X_test.reset_index(inplace=True, drop=True)
y_test = pd.read_pickle(os.path.join(settings.DATA_PATH, 'y_test.pkl'))
y_test.reset_index(inplace=True, drop=True)

booster = xgb.Booster()
booster.load_model(os.path.join(settings.DATA_PATH, 'best_model.json'))
model = xgb.XGBClassifier()
model._Booster = booster
model.n_classes_ = len(np.unique(y_test))

explainer = joblib.load(os.path.join(settings.DATA_PATH, 'explainer.pkl'))
shap_values = np.load(os.path.join(settings.DATA_PATH, 'shap_values.npy'))

var_definition = pd.read_pickle(os.path.join(
    settings.DATA_PATH, 'var_definition.pkl'))
var_dict = var_definition.set_index("변수명")['label'].to_dict()

# SHAP 분석기 초기화
analyzer = ShapAnalyzer(model, X_test, var_dict,
                        explainer=explainer, shap_values=shap_values)


class CreditAnalysisType(str, Enum):
    CURRENT = "current"
    HYPOTHETICAL = "hypothetical"


class CreditAnalysisRequest(BaseModel):
    company_index: str = Field(..., description="Company index number")
    analysis_type: CreditAnalysisType = Field(
        ..., description="Analysis type: current or hypothetical")


@router.post("/", include_in_schema=False)
@limiter.limit(settings.API_RATE_LIMIT)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} after {retry_state.outcome.exception()}",
    ),
)
async def analyze_credit(
    request: Request,
    analysis_request: CreditAnalysisRequest,
    llm_clients: Tuple = Depends(get_llm_clients),
) -> StreamingResponse:
    """
    AI 신용분석 수행. company_index, analysis_type 등을 JSON으로 받는다.

    예:
    {
        "company_index": "10",
        "analysis_type": "current"
    }
    """
    async def generate_analysis() -> AsyncIterator[str]:
        try:
            company_index = analysis_request.company_index
            analysis_type = analysis_request.analysis_type

            company_name = f"Company_{company_index}"
            logger.info(
                f"[Credit Analysis] {analysis_type.value} analysis requested for {company_name}")

            credit_cache = SQLiteCreditCache()
            cached_data = await credit_cache.get(company_name, analysis_type)

            if cached_data:
                logger.info(
                    f"[Cache] Using cached {analysis_type} analysis for {company_name}")
                yield json.dumps(cached_data) + "\n"
                return

            # SHAP Analyzer 초기화
            analyzer = ShapAnalyzer(
                model, X_test, var_dict, explainer=explainer, shap_values=shap_values
            )

            common_llm, summary_llm, insight_llm, credit_llm = llm_clients
            result_to_cache = None

            if analysis_type == CreditAnalysisType.CURRENT:
                logger.info(
                    f"[Current Analysis] Performing SHAP analysis for {company_name}")
                current_analysis = analyzer.get_shap_analysis(
                    int(company_index))

                current_chain = await create_current_analysis_chain(credit_llm)
                logger.info(
                    f"\n\n[Current Chain] Starting chain for current analysis of {company_name}")

                final_current = None
                async for chunk in current_chain.astream(current_analysis):
                    if isinstance(chunk, dict):
                        logger.debug(
                            f"[Current Chain Chunk] {json.dumps(chunk)}")
                        yield json.dumps(chunk) + "\n"
                        final_current = chunk

                if final_current:
                    result_to_cache = {
                        "detailed_analysis": final_current.get("detailed_analysis", ""),
                        "final_report": final_current.get("final_report", "")
                    }

            elif analysis_type == CreditAnalysisType.HYPOTHETICAL:
                logger.info(
                    f"[Hypothetical Analysis] Generating scenarios for {company_name}")
                hypothetical_analysis = analyzer.generate_hypothetical_results(
                    int(company_index))

                if len(hypothetical_analysis['scenarios']) == 0:
                    logger.warning(
                        f"[Hypothetical Analysis] No improvement scenarios for {company_name}")
                    result_to_cache = {
                        "detailed_analysis": "No improvement scenarios found for this company.",
                        "final_report": "### 개선 포인트 분석\n\n향상 시나리오가 발견되지 않았습니다."
                    }
                    yield json.dumps(result_to_cache) + "\n"
                else:
                    hypothetical_chain = await create_hypothetical_analysis_chain(credit_llm)
                    logger.info(
                        f"[Hypothetical Chain] Starting chain for hypothetical analysis")

                    final_hypothetical = None
                    async for chunk in hypothetical_chain.astream(hypothetical_analysis):
                        if isinstance(chunk, dict):
                            logger.debug(
                                f"[Hypothetical Chain Chunk] {json.dumps(chunk)}")
                            yield json.dumps(chunk) + "\n"
                            final_hypothetical = chunk

                    if final_hypothetical:
                        result_to_cache = {
                            "detailed_analysis": final_hypothetical.get("detailed_analysis", ""),
                            "final_report": final_hypothetical.get("final_report", "")
                        }

            # 캐시 저장
            if result_to_cache:
                await credit_cache.set(company_name, result_to_cache, analysis_type)
                logger.info(
                    f"[Cache] Stored {analysis_type} analysis for {company_name}")

        except Exception as e:
            logger.error(f"[Error] Credit analysis failed: {str(e)}")
            yield json.dumps({"error": str(e)}) + "\n"
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        generate_analysis(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
