# postgresql_cache.py
from typing import Dict, Optional
import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential
from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)


class PostgreSQLCache(BaseCache):
    """PostgreSQL 기반 캐시 구현"""

    def __init__(self):
        self._pool = None
        self.settings = settings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"[PostgreSQL] Connection retry attempt {retry_state.attempt_number}"
        )
    )
    async def _get_pool(self) -> asyncpg.Pool:
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                self.settings.CONNECTION_STRING,
                min_size=5,
                max_size=20
            )
            await self._init_db()
        return self._pool

    async def _init_db(self):
        """캐시 테이블 초기화"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # 분석 결과 테이블
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_TABLE_NAME} (
                    cache_key TEXT,                
                    analysis_type TEXT,            
                    indicator TEXT,                
                    detailed_result TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 피드백 테이블
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_FEEDBACK_NAME} (
                    id SERIAL PRIMARY KEY,
                    cache_key TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 캐시 키 인덱스
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                ON {settings.SQLITE_TABLE_NAME}
                (cache_key, analysis_type, indicator, created_at)
                WHERE cache_key IS NOT NULL
            """)

    async def get(self, key: str, analysis_type: str = None, indicator: str = None) -> Optional[Dict]:
        """캐시 데이터 조회"""
        try:
            today = datetime.now()
            company_name = key.split(':')[0]
            current_month = f"{today.year}-{today.month:02d}"

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # 기본 쿼리 구성
                base_query = f"""
                    SELECT analysis_type, indicator, detailed_result, summary
                    FROM {settings.SQLITE_TABLE_NAME}
                    WHERE cache_key LIKE $1
                """
                params = [f"{company_name}:{current_month}-%"]
                param_count = 1

                # THRESHOLD 이후인 경우
                if today.day >= settings.THRESHOLD:
                    param_count += 1
                    base_query += f" AND EXTRACT(DAY FROM created_at) >= ${param_count}"
                    params.append(settings.THRESHOLD)

                # 특정 분석 유형과 지표가 지정된 경우
                if analysis_type and indicator:
                    param_count += 2
                    base_query += f" AND analysis_type = ${param_count-1} AND indicator = ${param_count}"
                    params.extend([analysis_type, indicator])

                base_query += " ORDER BY created_at DESC LIMIT 1"

                row = await conn.fetchrow(base_query, *params)

                if not row:
                    return None

                result = {
                    'strength': {'indicator': None, 'detailed_result': None, 'summary': None},
                    'weakness': {'indicator': None, 'detailed_result': None, 'summary': None},
                    'insight': {'indicator': None, 'summary': None}
                }

                analysis_type = row['analysis_type']
                if analysis_type in ['strength', 'weakness']:
                    result[analysis_type] = {
                        'indicator': row['indicator'],
                        'detailed_result': row['detailed_result'],
                        'summary': row['summary']
                    }
                elif analysis_type == 'insight':
                    result['insight'] = {
                        'indicator': row['indicator'],
                        'summary': row['summary']
                    }

                return result

        except Exception as e:
            logger.error(f"[PostgreSQL] Get error for key {key}: {str(e)}")
            return None

    async def set(self, key: str, value: Dict, analysis_type: str) -> None:
        """캐시 데이터 저장"""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    if analysis_type in ['strength', 'weakness']:
                        await conn.execute(
                            f"""INSERT INTO {settings.SQLITE_TABLE_NAME}
                            (cache_key, analysis_type, indicator, detailed_result, summary)
                            VALUES ($1, $2, $3, $4, $5)""",
                            key,
                            analysis_type,
                            value[analysis_type]['indicator'],
                            value[analysis_type]['detailed_result'],
                            value[analysis_type]['summary']
                        )
                    elif analysis_type == 'insight':
                        await conn.execute(
                            f"""INSERT INTO {settings.SQLITE_TABLE_NAME}
                            (cache_key, analysis_type, indicator, summary)
                            VALUES ($1, $2, $3, $4)""",
                            key,
                            analysis_type,
                            value['insight']['indicator'],
                            value['insight']['summary']
                        )

                    logger.debug(
                        f"[PostgreSQL] Successfully stored {analysis_type} data for key: {key}")

        except Exception as e:
            logger.error(f"[PostgreSQL] Set error for key {key}: {str(e)}")

    async def close(self):
        """커넥션 풀 종료"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("[PostgreSQL] Connection pool closed")
