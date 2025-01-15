import requests
import os
from typing import Optional, Dict
import pandas as pd
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# 환경 변수를 사용하여 API URL 및 인증 정보 설정
DB_API_URL = f"http://{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/company_data"
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def fetch_company_data(company_id: Optional[int] = None, company_name: Optional[str] = None) -> pd.DataFrame:
    """
    API를 통해 특정 회사 데이터를 조회하여 DataFrame으로 반환하는 함수.

    Parameters:
    - company_id (Optional[int]): 조회하려는 회사의 ID
    - company_name (Optional[str]): 조회하려는 회사의 이름

    Returns:
    - pd.DataFrame: API로부터 받은 회사 데이터가 포함된 DataFrame
    """
    if not company_id and not company_name:
        raise ValueError("Either company_id or company_name must be provided.")

    params = {}
    if company_id:
        params['company_id'] = company_id
    if company_name:
        params['company_name'] = company_name

    try:
        response = requests.get(DB_API_URL, params=params,
                                auth=(DB_USERNAME, DB_PASSWORD))
        response.raise_for_status()  # 요청 성공 여부 확인
        data = response.json()

        # JSON 데이터를 DataFrame으로 변환
        df = pd.DataFrame([data]) if data else pd.DataFrame()
        logger.info("Company data fetched successfully via API.")
        return df

    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        raise RuntimeError(f"API error: {e}")
