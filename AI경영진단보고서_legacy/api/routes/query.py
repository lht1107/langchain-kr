# FastAPI 관련
from fastapi import APIRouter, Path, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import pandas as pd
import os

# 내부 모듈
from core.dependencies import get_cache
from core.cache import CacheManager
from preprocessing import (
    preprocess_growth_data,
    preprocess_profitability_data,
    preprocess_partner_stability_data,
    preprocess_financial_stability_data,
    preprocess_hr_data,
    preprocess_cashflow_data
)
from utils.logger import get_logger
from core.config import settings

# Logger 초기화
logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# 지표별 전처리 함수 매핑
preprocess_functions = {
    'growth': preprocess_growth_data,
    'profitability': preprocess_profitability_data,
    'partner_stability': preprocess_partner_stability_data,
    'financial_stability': preprocess_financial_stability_data,
    'hr': preprocess_hr_data,
    'cashflow': preprocess_cashflow_data
}


@router.get('/{company_name}/{analysis_type}')
async def get_company_metrics(
    company_name: str,
    analysis_type: str = Path(...,
                              description="Analysis type: strength or weakness"),
    strength_metric: Optional[str] = None,
    weakness_metric: Optional[str] = None,
    cache: CacheManager = Depends(get_cache)
) -> Dict:
    """회사의 지표 데이터 조회"""
    try:
        # 분석 유형에 따른 메트릭 결정
        analysis_metric = (
            strength_metric if analysis_type == "strength"
            else weakness_metric if analysis_type == "weakness"
            else None
        )

        # 메트릭 유효성 검증
        if not analysis_metric or analysis_metric not in preprocess_functions:
            logger.error(f"[Query] Invalid analysis_metric: {analysis_metric}")
            raise HTTPException(
                status_code=400, detail="Invalid analysis_metric")

        # 데이터 로드 및 전처리
        logger.info(
            f"[Query] Processing {analysis_metric} data for company: {company_name}")
        df_company_info = pd.read_parquet(
            os.path.join(settings.BASE_DIR, 'sample.parquet'))

        # 전처리 함수 실행
        result = preprocess_functions[analysis_metric](
            df_company_info,
            company_name,
            settings.ACCESS_TIME
        )

        logger.info(f"[Query] Successfully processed data for {company_name}")

        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"[Query] Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
