# FastAPI 관련
from fastapi import APIRouter, Path, HTTPException, Depends
from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import asyncio
import numpy as np

# 내부 모듈
from core.dependencies import get_cache
from core.cache import CacheManager
from preprocessing import (
    preprocess_growth_data,
    preprocess_profitability_data,
    preprocess_partner_stability_data,
    preprocess_financial_stability_data
)
from utils import generate_sample_data
from utils.time_utils import get_access_time
from utils.logger import get_logger
from core.config import settings
import os

# Logger 초기화
logger = get_logger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["query"]
)

# 지표별 전처리 함수 매핑
preprocess_functions = {
    'growth': preprocess_growth_data,
    'profitability': preprocess_profitability_data,
    'partner_stability': preprocess_partner_stability_data,
    'financial_stability': preprocess_financial_stability_data
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
        # today = datetime.now()
        access_time = get_access_time()

        # 이번 달의 캐시 데이터 조회
        cache_key = cache.generate_cache_key(company_name, access_time)
        cached_data = await cache.get(
            cache_key,
            analysis_type,
            strength_metric if analysis_type == "strength"
            else weakness_metric if analysis_type == "weakness"
            else None
        )
        logger.info(f"[Query] Checking cache for key: {cache_key}")

        if not cached_data:
            logger.warning(f"[Query] No cache data found for {cache_key}")
            raise HTTPException(status_code=404, detail="Cache data not found")

        # 지표 일치 여부 검증
        if analysis_type == "strength":
            if cached_data['strength']['indicator'] != strength_metric:
                logger.warning(
                    f"[Query] Indicator mismatch: expected {strength_metric}")
                raise HTTPException(
                    status_code=404, detail="Indicator mismatch")
            indicator = strength_metric
        elif analysis_type == "weakness":
            if cached_data['weakness']['indicator'] != weakness_metric:
                logger.warning(
                    f"[Query] Indicator mismatch: expected {weakness_metric}")
                raise HTTPException(
                    status_code=404, detail="Indicator mismatch")
            indicator = weakness_metric
        else:
            logger.error(f"[Query] Invalid analysis type: {analysis_type}")
            raise HTTPException(
                status_code=400, detail="Invalid analysis type")

        if indicator not in preprocess_functions:
            logger.error(f"[Query] Invalid indicator: {indicator}")
            raise HTTPException(status_code=400, detail="Invalid indicator")

        # 기업 데이터 생성 및 전처리
        logger.info(f"[Query] Processing {indicator} data for {company_name}")
        # df_company_info = await asyncio.to_thread(
        #     generate_sample_data,
        #     access_time
        # )
        sample_data_path = os.path.join(settings.BASE_DIR, 'sample.parquet')
        df_company_info = pd.read_parquet(sample_data_path)

        # 전처리 함수가 이미 직렬화된 데이터를 반환
        result = preprocess_functions[indicator](
            df_company_info,
            company_name,
            access_time
        )

        logger.info(f"[Query] Successfully processed data for {company_name}")

        # JSON 응답에 한글 인코딩 설정 추가
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"[Query] Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
