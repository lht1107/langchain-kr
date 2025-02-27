import asyncpg
from typing import Dict, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class PostgreSQLCreditCache(BaseCache):
    """PostgreSQL-based implementation for credit consulting cache."""

    def __init__(self, table_name: str = None, feedback_table_name: str = None):
        """Initialize the PostgreSQLCreditCache.

        Args:
            table_name: Optional custom table name
            feedback_table_name: Optional custom feedback table name
        """

        self._pool: Optional[asyncpg.pool.Pool] = None
        self.table_name = table_name or settings.CREDIT_TABLE_NAME
        self.feedback_table_name = feedback_table_name or settings.CREDIT_FEEDBACK_NAME
        self.index_name = settings.CREDIT_CACHE_INDEX

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"[PostgreSQLCredit] Connection retry attempt {retry_state.attempt_number}"
        )
    )
    async def _get_pool(self) -> asyncpg.pool.Pool:
        """Establish a connection pool to the PostgreSQL database."""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                dsn=settings.CONNECTION_STRING,
                min_size=5,
                max_size=20
            )
            await self._init_db()
        return self._pool

    async def _init_db(self):
        """Initialize credit consulting tables and indices."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Create main credit consulting table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    detailed_analysis TEXT,
                    final_report TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create feedback table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.feedback_table_name} (
                    id SERIAL PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for efficient querying
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.index_name}
                ON {self.table_name}
                (company_name, created_at DESC, analysis_type)
            """)

    async def get(
        self,
        company_name: str,
        analysis_type: str
    ) -> Optional[Dict]:
        """Retrieve cached credit consulting data."""
        if not company_name or not analysis_type:
            raise ValueError("Company name and analysis type are required")

        try:
            today = datetime.now()
            current_month = f"{today.year}-{today.month:02d}"
            prev_month = (f"{today.year}-{today.month-1:02d}"
                          if today.month > 1 else f"{today.year-1}-12")

            result = {
                'detailed_analysis': None,
                'final_report': None
            }

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                base_query = f"""
                    SELECT detailed_analysis, final_report
                    FROM {self.table_name}
                    WHERE company_name = $1 
                    AND analysis_type = $2
                    AND (
                        (TO_CHAR(created_at, 'YYYY-MM') = $3 AND EXTRACT(DAY FROM created_at) >= $4)
                        OR 
                        (TO_CHAR(created_at, 'YYYY-MM') = $5 AND EXTRACT(DAY FROM created_at) >= $6)
                    )
                    ORDER BY created_at DESC LIMIT 1
                """
                params = [
                    company_name,
                    analysis_type,
                    current_month if today.day >= settings.THRESHOLD else prev_month,
                    settings.THRESHOLD,
                    current_month,
                    settings.THRESHOLD
                ]

                row = await conn.fetchrow(base_query, *params)
                if row:
                    result.update({
                        'detailed_analysis': row['detailed_analysis'],
                        'final_report': row['final_report']
                    })
                    return result
                return None

        except Exception as e:
            logger.error(f"[PostgreSQLCredit] Get error: {str(e)}")
            return None

    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        """Store credit consulting data."""
        if not all([company_name, value, analysis_type]):
            raise ValueError("Missing required parameters for cache storage")

        required_fields = {'detailed_analysis', 'final_report'}
        if not all(field in value for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        f"""INSERT INTO {self.table_name}
                           (company_name, analysis_type, detailed_analysis, final_report)
                           VALUES ($1, $2, $3, $4)""",
                        company_name,
                        analysis_type,
                        value['detailed_analysis'],
                        value['final_report']
                    )
                    logger.debug(
                        f"[PostgreSQLCredit] Stored {analysis_type} analysis for: {company_name}")
        except Exception as e:
            logger.error(f"[PostgreSQLCredit] Set error: {str(e)}")
            raise

    def _create_empty_cache(self) -> Dict:
        """Create an empty cache structure."""
        return {
            'detailed_analysis': None,
            'final_report': None
        }

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("[PostgreSQLCredit] Connection pool closed")
