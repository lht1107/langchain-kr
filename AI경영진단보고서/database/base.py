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
    """캐시 저장소의 기본 인터페이스를 정의하는 추상 클래스"""

    def serialize_datetime(self, obj: Any) -> str:
        """Timestamp 객체를 문자열로 직렬화"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime(settings.TIMESTAMP)
        return obj

    @abstractmethod
    async def get(self, key: str, analysis_type: str = None, indicator: str = None) -> Optional[Dict]:
        """캐시에서 데이터 조회

        Args:
            key: 조회할 캐시 키
            analysis_type: 분석 유형 (strength/weakness/insight)
            indicator: 분석 지표

        Returns:
            조회된 데이터 또는 None
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Dict, analysis_type: str) -> None:
        """캐시에 데이터 저장

        Args:
            key: 저장할 캐시 키
            value: 저장할 데이터
            analysis_type: 분석 유형 (strength/weakness/insight)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """캐시 연결 종료"""
        pass
