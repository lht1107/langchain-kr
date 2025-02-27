import os
from fastapi import APIRouter, Query, HTTPException, Request, Depends
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from core.config import settings
from analysis.credit_analyzer import ShapAnalyzer
from utils.logger import get_logger
from core.dependencies import get_llm_clients

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


@router.get("", response_model=Dict)
async def analyze_query(
    request: Request,
    company_index: str = Query(..., description="Company index number"),
    analysis_type: str = Query(...,
                               description="Analysis type: current or hypothetical")
):
    """
    company_index와 analysis_type(current/hypothetical)을 기반으로 분석 결과 반환
    """
    try:
        logger.info(
            f"Received query for {analysis_type} analysis of Company_{company_index}")

        # 분석 타입에 따라 처리
        if analysis_type == "current":
            logger.info(
                f"Calculating current_analysis for Company_{company_index}")
            current_analysis = analyzer.get_shap_analysis(int(company_index))
            return {"analysis_type": "current", "current_analysis": current_analysis}

        elif analysis_type == "hypothetical":
            logger.info(
                f"Calculating hypothetical_analysis for Company_{company_index}")
            hypothetical_analysis = analyzer.generate_hypothetical_results(
                int(company_index))

            # 시나리오가 없는 경우 기본 메시지 반환
            if len(hypothetical_analysis.get("scenarios", [])) == 0:
                logger.warning(
                    f"No improvement scenarios for Company_{company_index}")
                return {
                    "analysis_type": "hypothetical",
                    "message": "No significant improvement opportunities identified."
                }

            return {"analysis_type": "hypothetical", "hypothetical_analysis": hypothetical_analysis}

        else:
            logger.error(f"Invalid analysis type: {analysis_type}")
            raise HTTPException(
                status_code=400, detail="Invalid analysis type. Must be 'current' or 'hypothetical'.")

    except ValueError as e:
        logger.error(f"Invalid company index: {e}")
        raise HTTPException(status_code=400, detail="Invalid company index")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
