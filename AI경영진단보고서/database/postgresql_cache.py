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
        self._pool: Optional[asyncpg.pool.Pool] = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"[PostgreSQL] Connection retry attempt {retry_state.attempt_number}"
        )
    )
    async def _get_pool(self) -> asyncpg.pool.Pool:
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                dsn=settings.POSTGRESQL_CONNECTION_STRING,
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
                CREATE TABLE IF NOT EXISTS {settings.POSTGRESQL_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_metric TEXT NOT NULL,
                    detailed_result TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 피드백 테이블
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.POSTGRESQL_FEEDBACK_NAME} (
                    id SERIAL PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_metric TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 인덱스 생성 (존재하지 않을 경우에만)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_analysis_lookup 
                ON {settings.POSTGRESQL_TABLE_NAME} 
                (company_name, created_at DESC, analysis_type, analysis_metric)
            """)

    async def get(
        self,
        company_name: str,
        analysis_type: Optional[str] = None,
        analysis_metric: Optional[str] = None
    ) -> Optional[Dict]:
        """캐시에서 데이터 조회"""

        try:
            today = datetime.now()
            current_month = f"{today.year}-{today.month:02d}"
            prev_month = f"{today.year}-{today.month-1:02d}" if today.month > 1 else f"{today.year-1}-12"

            result = {
                'strength': {'analysis_metric': None, 'detailed_result': None, 'summary': None},
                'weakness': {'analysis_metric': None, 'detailed_result': None, 'summary': None},
                'insight': {'analysis_metric': None, 'summary': None}
            }

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                if today.day >= settings.THRESHOLD:
                    base_query = f"""
                        SELECT analysis_type, analysis_metric, detailed_result, summary
                        FROM {settings.POSTGRESQL_TABLE_NAME}
                        WHERE company_name = $1 
                        AND TO_CHAR(created_at, 'YYYY-MM') = $2
                        AND EXTRACT(DAY FROM created_at) >= $3
                    """
                    params = [company_name, current_month, settings.THRESHOLD]
                else:
                    base_query = f"""
                        SELECT analysis_type, analysis_metric, detailed_result, summary
                        FROM {settings.POSTGRESQL_TABLE_NAME}
                        WHERE company_name = $1 AND (
                            TO_CHAR(created_at, 'YYYY-MM') = $2 OR
                            (TO_CHAR(created_at, 'YYYY-MM') = $3 AND EXTRACT(DAY FROM created_at) >= $4)
                        )
                    """
                    params = [company_name, current_month,
                              prev_month, settings.THRESHOLD]

                if analysis_type == 'insight' and analysis_metric:
                    strength_metric, weakness_metric = analysis_metric.split(
                        '/')
                    base_query += """ 
                        AND (
                            (analysis_type = 'strength' AND analysis_metric = $5) OR
                            (analysis_type = 'weakness' AND analysis_metric = $6) OR
                            (analysis_type = 'insight' AND analysis_metric = $7)
                        )
                    """
                    params.extend(
                        [strength_metric, weakness_metric, analysis_metric])
                elif analysis_type and analysis_metric:
                    base_query += " AND analysis_type = $5 AND analysis_metric = $6"
                    params.extend([analysis_type, analysis_metric])

                base_query += " ORDER BY created_at DESC"

                rows = await conn.fetch(base_query, *params)

                if not rows:
                    return None

                for row in rows:
                    analysis_type_row, metric, detailed, summary = row['analysis_type'], row[
                        'analysis_metric'], row['detailed_result'], row['summary']
                    if analysis_type_row in ['strength', 'weakness']:
                        result[analysis_type_row] = {
                            'analysis_metric': metric,
                            'detailed_result': detailed,
                            'summary': summary
                        }
                    elif analysis_type_row == 'insight':
                        result['insight'] = {
                            'analysis_metric': metric,
                            'summary': summary
                        }

                return result

        except Exception as e:
            logger.error(
                f"[PostgreSQL] Get error for company {company_name}: {str(e)}")
            return None

    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        """캐시 데이터 저장"""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    if analysis_type in ['strength', 'weakness']:
                        await conn.execute(
                            f"""INSERT INTO {settings.POSTGRESQL_TABLE_NAME}
                            (company_name, analysis_type, analysis_metric, detailed_result, summary)
                            VALUES ($1, $2, $3, $4, $5)""",
                            company_name,
                            analysis_type,
                            value[analysis_type]['analysis_metric'],
                            value[analysis_type]['detailed_result'],
                            value[analysis_type]['summary']
                        )
                    elif analysis_type == 'insight':
                        await conn.execute(
                            f"""INSERT INTO {settings.POSTGRESQL_TABLE_NAME}
                            (company_name, analysis_type, analysis_metric, summary)
                            VALUES ($1, $2, $3, $4)""",
                            company_name,
                            analysis_type,
                            value['insight']['analysis_metric'],
                            value['insight']['summary']
                        )

                    logger.debug(
                        f"[PostgreSQL] Successfully stored {analysis_type} data for company: {company_name}"
                    )

        except Exception as e:
            logger.error(
                f"[PostgreSQL] Set error for company {company_name}: {str(e)}")
            raise

    def _create_empty_cache(self) -> Dict:
        """빈 캐시 데이터 구조 생성"""
        return {
            'strength': {
                'analysis_metric': None,
                'detailed_result': None,
                'summary': None
            },
            'weakness': {
                'analysis_metric': None,
                'detailed_result': None,
                'summary': None
            },
            'insight': {
                'analysis_metric': None,
                'summary': None
            }
        }

    async def close(self) -> None:
        """커넥션 풀 종료"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("[PostgreSQL] Connection pool closed")
