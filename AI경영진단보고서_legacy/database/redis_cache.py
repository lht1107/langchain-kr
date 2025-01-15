# 기본 모듈
import json
from typing import Dict, Optional

# Redis 관련
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

# 내부 모듈
from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)


class RedisCache(BaseCache):
    """Redis 캐시 구현 클래스"""

    def __init__(self):
        """Redis 캐시 초기화"""
        self._pool = None
        self.settings = settings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Redis connection attempt {retry_state.attempt_number} failed. Retrying..."
        )
    )
    async def _get_connection(self) -> redis.Redis:
        """Redis 연결 풀 생성 및 반환

        Returns:
            redis.Redis: Redis 연결 객체

        Raises:
            Exception: 연결 실패 시
        """
        if not self._pool:
            self._pool = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                decode_responses=True,
                max_connections=self.settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=self.settings.REDIS_SOCKET_TIMEOUT,
                retry_on_timeout=self.settings.REDIS_RETRY_ON_TIMEOUT,
                health_check_interval=self.settings.REDIS_HEALTH_CHECK_INTERVAL
            )
            await self._pool.ping()  # 연결 확인
        return self._pool

    async def get(self, key: str, analysis_type: str = None, indicator: str = None) -> Optional[Dict]:
        """캐시에서 데이터 조회"""
        try:
            conn = await self._get_connection()
            data = await conn.get(key)

            if not data:
                return None

            result = json.loads(data)

            # 특정 분석 유형과 지표가 지정된 경우 해당 결과만 반환
            if analysis_type and indicator:
                if analysis_type in result and result[analysis_type].get('indicator') == indicator:
                    return {
                        'strength': {'indicator': None, 'detailed_result': None, 'summary': None},
                        'weakness': {'indicator': None, 'detailed_result': None, 'summary': None},
                        'insight': {'indicator': None, 'summary': None},
                        analysis_type: result[analysis_type]
                    }
                return None

            return result

        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None

    async def set(self, key: str, value: Dict, analysis_type: str) -> None:
        """캐시에 데이터 저장"""
        try:
            conn = await self._get_connection()

            # 기존 데이터가 있으면 업데이트
            existing_data = await self.get(key)
            if existing_data:
                existing_data[analysis_type] = value[analysis_type]
                value = existing_data

            await conn.set(
                key,
                json.dumps(value, default=self.serialize_datetime)
            )
            logger.debug(
                f"[Redis] Successfully stored {analysis_type} data for key: {key}")

        except Exception as e:
            logger.error(f"Redis set error: {str(e)}")

    async def close(self) -> None:
        """Redis 연결 종료

        Raises:
            Exception: 연결 종료 실패 시
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
