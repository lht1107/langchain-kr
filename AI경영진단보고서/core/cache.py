# 타입 힌트
from typing import List, Dict, Optional

# 시간 관련
from datetime import datetime

# 내부 모듈
from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)


class CacheManager:
    """캐시 저장소 관리 클래스

    여러 저장소(Redis, SQLite, PostgreSQL)를 통합 관리하는 매니저 클래스
    """

    def __init__(self):
        """캐시 매니저 초기화"""
        self.storages: List[BaseCache] = []
        self._initialize_storages()

    def _initialize_storages(self) -> None:
        """설정에 따라 캐시 저장소들을 초기화"""
        try:
            # Redis 저장소 초기화
            if settings.CACHE_TYPE in ['redis', 'all']:
                from database.redis_cache import RedisCache
                self.storages.append(RedisCache())
                logger.info("[Cache] Redis storage initialized")

            # SQLite 저장소 초기화
            if settings.CACHE_TYPE in ['sqlite', 'all']:
                from database.sqlite_cache import SQLiteCache
                self.storages.append(SQLiteCache())
                logger.info("[Cache] SQLite storage initialized")

            # PostgreSQL 저장소 초기화
            if settings.CACHE_TYPE in ['postgresql', 'all']:
                from database.postgresql_cache import PostgreSQLCache
                self.storages.append(PostgreSQLCache())
                logger.info("[Cache] PostgreSQL storage initialized")

            if not self.storages:
                logger.warning(
                    f"[Cache] No storage initialized for type: {settings.CACHE_TYPE}")

        except Exception as e:
            logger.error(f"[Cache] Failed to initialize storages: {str(e)}")
            raise

    def generate_cache_key(self, company_name: str, access_time: datetime) -> str:
        """캐시 키 생성 함수"""
        base_key = f"{company_name}:{access_time.strftime(settings.TIMESTAMP)}"
        return base_key

    def validate_cache_data(self, data: Dict) -> bool:
        try:
            if not isinstance(data, dict):
                return False

            required_fields = {'strength', 'weakness', 'insight'}
            if not all(field in data for field in required_fields):
                return False

            # strength와 weakness 검증
            required_analysis_fields = {
                'indicator', 'detailed_result', 'summary'}
            for analysis_type in ['strength', 'weakness']:
                if not all(field in data[analysis_type] for field in required_analysis_fields):
                    return False

            # insight 검증
            if not all(field in data['insight'] for field in ['indicator', 'summary']):
                return False

            return True
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    async def get(self, key: str, analysis_type: str = None, indicator: str = None):
        """캐시 데이터 조회"""
        for storage in self.storages:
            try:
                result = await storage.get(key, analysis_type, indicator)
                if result and self.validate_cache_data(result):
                    logger.debug(f"[Cache] Hit - Key: {key}")
                    return result
            except Exception as e:
                logger.error(
                    f"[Cache] Error getting data from {storage.__class__.__name__}: {str(e)}")

        logger.info(f"[Cache] Miss - Key: {key}")
        return None

    async def set(self, key: str, data: Dict, analysis_type: str = None) -> None:
        """캐시 데이터 저장"""
        try:
            cached_data = await self.get(key) or self._create_empty_cache()

            # 분석 결과 업데이트
            if analysis_type:
                self._update_analysis_data(cached_data, data, analysis_type)

            # 저장소 업데이트
            for storage in self.storages:
                try:
                    await storage.set(key, cached_data, analysis_type)
                    logger.debug(
                        f"[Cache] Stored in {storage.__class__.__name__}")
                except Exception as e:
                    logger.error(
                        f"[Cache] Error setting data in {storage.__class__.__name__}: {str(e)}")

            # 단일 통합 로그 메시지
            logger.info(
                f"\n[Cache] Analysis completed - Key: {key} | "
                f"Type: {analysis_type or 'N/A'} | "
                f"Indicators: [S: {cached_data['strength']['indicator']}, "
                f"W: {cached_data['weakness']['indicator']}, "
                f"I: {cached_data['insight']['indicator']}]"
            )

        except Exception as e:
            logger.error(f"[Cache] Failed to set cached data: {str(e)}")
            raise

    def _update_analysis_data(self, cached_data: Dict, new_data: Dict, analysis_type: str) -> None:
        """분석 결과 업데이트"""
        match analysis_type:
            case "strength":
                cached_data['strength'] = new_data['strength']
            case "weakness":
                cached_data['weakness'] = new_data['weakness']
            case "insight":
                cached_data['insight'] = new_data['insight']

    def _create_empty_cache(self) -> Dict:
        """빈 캐시 데이터 구조 생성"""
        return {
            'strength': {
                'indicator': None,
                'detailed_result': None,
                'summary': None
            },
            'weakness': {
                'indicator': None,
                'detailed_result': None,
                'summary': None
            },
            'insight': {
                'indicator': None,
                'summary': None
            }
        }

    async def close(self) -> None:
        """모든 저장소 연결 종료"""
        for storage in self.storages:
            try:
                if hasattr(storage, 'close'):
                    await storage.close()
                    logger.info(f"[Cache] Closed {storage.__class__.__name__}")
            except Exception as e:
                logger.error(
                    f"[Cache] Error closing {storage.__class__.__name__}: {str(e)}")
