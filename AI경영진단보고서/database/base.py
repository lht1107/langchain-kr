# 기본 모듈
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import json
from datetime import datetime

# 데이터 처리
import pandas as pd

# 내부 모듈
from core.config import settings


class BaseCache(ABC):
    """회사 저장소의 기본 인터페이스를 정의하는 추상 클래스"""

    def serialize_datetime(self, obj: Any) -> str:
        """Timestamp 객체를 문자열로 직렬화"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime(settings.TIMESTAMP)
        return obj

    @abstractmethod
    async def get(self, company_name: str, analysis_type: str = None, analysis_metric: str = None) -> Optional[Dict]:
        """캐시에서 데이터 조회

        Args:
            company_name: 회사명
            analysis_type: 분석 유형 (strength/weakness/insight)
            analysis_metric: 분석 지표

        Returns:
            조회된 데이터 또는 None
        """
        pass

    @abstractmethod
    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        """회사에 데이터 저장

        Args:
            company_name: 저장할 회사 키
            value: 저장할 데이터
            analysis_type: 분석 유형 (strength/weakness/insight)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """회사 연결 종료"""
        pass
