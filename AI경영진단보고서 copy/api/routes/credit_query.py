import os
from fastapi import APIRouter, HTTPException, Request
from typing import Dict
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pydantic import BaseModel, Field
from core.config import settings
from analysis.credit_analyzer import ShapAnalyzer
from utils.logger import get_logger
from enum import Enum

# Logger 초기화
logger = get_logger(__name__)

# 라우터 설정
router = APIRouter(
    prefix="/credit_query",
    tags=["credit_query"]
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


class CreditQueryRequest(BaseModel):
    company_index: str = Field(..., description="Company index number")
    analysis_type: CreditAnalysisType = Field(
        ..., description="Analysis type: current or hypothetical")


@router.post("/", include_in_schema=False, response_model=Dict)
async def analyze_query(
    request: Request,
    query_request: CreditQueryRequest
):
    """
    company_index와 analysis_type(current/hypothetical)에 따른 분석 결과를 반환.
    예: POST /credit_query
    {
        "company_index": "10",
        "analysis_type": "current"
    }
    """
    try:
        company_index = query_request.company_index
        analysis_type = query_request.analysis_type

        logger.info(
            f"[Query] Received {analysis_type.value} request for Company_{company_index}")

        if not company_index.isdigit():
            raise ValueError("company_index must be a digit.")

        idx = int(company_index)
        if idx < 0 or idx >= len(X_test):
            raise ValueError("Invalid company_index range.")

        if analysis_type == CreditAnalysisType.CURRENT:
            logger.info(
                f"[Current Analysis] Perform analysis for Company_{company_index}")
            current_analysis = analyzer.get_shap_analysis(idx)
            return {
                "analysis_type": CreditAnalysisType.CURRENT.value,
                "current_analysis": current_analysis
            }

        elif analysis_type == CreditAnalysisType.HYPOTHETICAL:
            logger.info(
                f"[Hypothetical Analysis] Perform scenario generation for Company_{company_index}")
            hypothetical_analysis = analyzer.generate_hypothetical_results(idx)

            if not hypothetical_analysis.get("scenarios"):
                logger.warning(
                    f"[Hypothetical] No scenarios found for Company_{company_index}")
                return {
                    "analysis_type": CreditAnalysisType.HYPOTHETICAL.value,
                    "message": "No improvement scenarios found."
                }

            return {
                "analysis_type": CreditAnalysisType.HYPOTHETICAL.value,
                "hypothetical_analysis": hypothetical_analysis
            }

    except ValueError as e:
        logger.error(f"[Query] Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Query] Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
