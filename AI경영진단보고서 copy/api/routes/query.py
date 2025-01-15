import os
import asyncio
from typing import Dict
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from core.dependencies import get_cache
from core.cache import CacheManager
from core.config import settings
from utils.logger import get_logger

# 아래는 예시로, preprocess_* 함수들을 한 dict에 매핑한다고 가정
from preprocessing import (
    preprocess_growth_data,
    preprocess_profitability_data,
    preprocess_partner_stability_data,
    preprocess_financial_stability_data,
    preprocess_hr_data,
    preprocess_cashflow_data
)
from pydantic import BaseModel

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)

PREPROCESS_FUNCTIONS = {
    "growth": preprocess_growth_data,
    "profitability": preprocess_profitability_data,
    "partner_stability": preprocess_partner_stability_data,
    "financial_stability": preprocess_financial_stability_data,
    "hr": preprocess_hr_data,
    "cashflow": preprocess_cashflow_data
}


class QueryRequest(BaseModel):
    company_name: str
    metric: str


async def get_latest_access_time_from_data(data_path: str) -> pd.Timestamp:
    """Parquet or CSV 파일을 읽어 'date' 또는 'updated_at' 등의 컬럼 최대값을 Timestamp로 반환."""
    try:
        # 예: sample.parquet 에서 'date' 컬럼의 최댓값
        df = pd.read_parquet(data_path)

        # 우선순위: 'updated_at' → 'date' → '날짜' → 없으면 현재 시점
        for col in ['updated_at', 'date', '날짜']:
            if col in df.columns:
                max_val = pd.to_datetime(df[col]).max()
                logger.info(
                    f"[Query] Determined latest_access_time from column '{col}': {max_val}")
                return max_val

        # 위 컬럼이 없으면 현재 시각
        logger.warning(
            "[Query] No date column found, using now() as access_time")
        return pd.Timestamp.now()

    except Exception as e:
        logger.error(
            f"[Query] Failed to load data for latest access time: {str(e)}")
        return pd.Timestamp.now()  # fallback


@router.post("/", include_in_schema=False)
async def query_company_metrics(
    request: QueryRequest,
    cache: CacheManager = Depends(get_cache)
) -> JSONResponse:
    """
    Fetch and preprocess company metrics based on the request payload.

    Args:
        request (QueryRequest): The JSON request payload.
        cache (CacheManager): Cache dependency.

    Returns:
        JSONResponse: Preprocessed metric data.
    """
    try:
        company_name = request.company_name
        metric = request.metric

        # Validate the metric
        if metric not in PREPROCESS_FUNCTIONS:
            logger.error(f"[Query] Invalid metric: {metric}")
            raise HTTPException(
                status_code=400, detail="Invalid metric"
            )

        logger.info(f"[Query] Fetching data for {metric} of {company_name}")

        # Load company data (Parquet)
        sample_data_path = os.path.join(settings.DATA_PATH, 'sample.parquet')
        if not os.path.exists(sample_data_path):
            raise HTTPException(
                status_code=500,
                detail=f"Sample data file not found at {sample_data_path}"
            )

        # DataFrame 로드
        df_company_info = pd.read_parquet(sample_data_path)

        # ★ 동적으로 access_time 결정
        latest_access_time = await get_latest_access_time_from_data(sample_data_path)

        # Preprocess data using the mapped function
        preprocess_function = PREPROCESS_FUNCTIONS[metric]
        # 여기서 preprocess_xxx_data(
        #    df: pd.DataFrame, company_name: str, access_time: pd.Timestamp
        # ) -> Dict[str, Any] 형태
        result = preprocess_function(
            df_company_info, company_name, latest_access_time
        )

        logger.info(
            f"[Query] Successfully processed {metric} data for {company_name}")

        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )

    except KeyError as key_error:
        logger.error(f"[Query] Missing key in data processing: {key_error}")
        raise HTTPException(
            status_code=400, detail=f"Invalid key: {key_error}"
        )

    except FileNotFoundError as file_error:
        logger.error(f"[Query] File not found: {file_error}")
        raise HTTPException(status_code=500, detail="Data file not found")

    except Exception as e:
        logger.error(f"[Query] Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
