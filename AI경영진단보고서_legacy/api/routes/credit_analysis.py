from core.dependencies import get_llm_clients, limiter
from typing import Dict, Tuple, AsyncIterator
from fastapi import APIRouter, Path, Request, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, Tuple, AsyncIterator
from enum import Enum
import os
import json
import pandas as pd
from datetime import datetime
import joblib
from tenacity import retry, stop_after_attempt, wait_exponential
import xgboost as xgb
from core.dependencies import get_llm_clients, limiter, get_cache
from core.config import settings
from core.cache import CacheManager
from utils.logger import get_logger
from utils.parser import output_parser
from database.sqlite_credit_cache import SQLiteCreditCache
from analysis.credit_analyzer import ShapAnalyzer
from prompts.current_credit_prompt import create_current_analysis_chain
from prompts.hypothetical_credit_prompt import create_hypothetical_analysis_chain
from langchain_openai import ChatOpenAI
from core.config import settings
import numpy as np
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)


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

# 분석 타입 정의


class CreditAnalysisType(str, Enum):
    CURRENT = "current"
    HYPOTHETICAL = "hypothetical"


# 라우터 설정
router = APIRouter(
    prefix="/credit_analysis",
    tags=["credit_analysis"]
)


@router.get("")
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
    company_index: str = Query(..., description="Company index number"),
    analysis_type: str = Query(...,
                               description="Analysis type: current or hypothetical"),
    llm_clients: Tuple = Depends(get_llm_clients),
) -> StreamingResponse:
    """신용 분석 수행 및 스트리밍 응답 반환"""

    async def generate_analysis() -> AsyncIterator[str]:
        try:
            company_name = f"Company_{company_index}"
            logger.info(
                f"[Credit Analysis] {analysis_type} analysis requested for {company_name}")

            credit_cache = SQLiteCreditCache()
            cached_data = await credit_cache.get(company_name, analysis_type)

            if cached_data:
                logger.info(
                    f"[Cache] Using cached {analysis_type} analysis for {company_name}")
                yield json.dumps({"cached_data": cached_data}) + "\n"
                return

            analyzer = ShapAnalyzer(
                model, X_test, var_dict, explainer=explainer, shap_values=shap_values
            )
            _, _, _, credit_llm = llm_clients

            result_to_cache = None  # 캐시에 저장할 데이터

            if analysis_type == "current":
                logger.info(
                    f"[Current Analysis] Performing SHAP analysis for {company_name}")
                current_analysis = analyzer.get_shap_analysis(
                    int(company_index))

                current_chain = await create_current_analysis_chain(credit_llm)
                logger.info(
                    f"[Current Chain] Starting LLM chain for current analysis of {company_name}")

                final_current = None
                async for chunk in current_chain.astream(current_analysis):
                    if isinstance(chunk, dict):
                        logger.debug(
                            f"[Current Chain Chunk] {json.dumps(chunk)}")
                        yield json.dumps(chunk) + "\n"
                        final_current = chunk

                # 캐시에 저장할 데이터 구성
                if final_current:
                    result_to_cache = {
                        "detailed_analysis": final_current.get("detailed_analysis", ""),
                        "final_report": final_current.get("final_report", "")
                    }

            elif analysis_type == "hypothetical":
                logger.info(
                    f"[Hypothetical Analysis] Generating scenarios for {company_name}")
                hypothetical_analysis = analyzer.generate_hypothetical_results(
                    int(company_index))

                if len(hypothetical_analysis['scenarios']) == 0:
                    logger.warning(
                        f"[Hypothetical Analysis] No improvement scenarios for {company_name}")
                    result_to_cache = {
                        "detailed_analysis": "No significant improvement opportunities were identified for this company.",
                        "final_report": "### 개선 포인트 분석\n\n부도확률을 증가시키는 top 5 요인을 중심으로 특별한 개선 포인트가 발견되지 않았습니다."
                    }
                    yield json.dumps(result_to_cache) + "\n"
                else:
                    hypothetical_chain = await create_hypothetical_analysis_chain(
                        credit_llm)
                    logger.info(
                        f"[Hypothetical Chain] Starting LLM chain for hypothetical analysis")

                    final_hypothetical = None
                    async for chunk in hypothetical_chain.astream(hypothetical_analysis):
                        if isinstance(chunk, dict):
                            logger.debug(
                                f"[Hypothetical Chain Chunk] {json.dumps(chunk)}")
                            yield json.dumps(chunk) + "\n"
                            final_hypothetical = chunk

                    # 캐시에 저장할 데이터 구성
                    if final_hypothetical:
                        result_to_cache = {
                            "detailed_analysis": final_hypothetical.get("detailed_analysis", ""),
                            "final_report": final_hypothetical.get("final_report", "")
                        }

            # 캐시에 저장
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
