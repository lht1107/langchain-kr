import asyncpg
import threading
from typing import Dict, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class PostgreSQLCache(BaseCache):
    """PostgreSQL-based implementation of a cache with support for asynchronous operations."""

    _pool_creation_lock = threading.Lock()

    def __init__(self, table_name: str = None, feedback_table_name: str = None,
                 index_name: str = None, feedback_index_name: str = None):
        """Initialize the PostgreSQLCache.

        Args:
            table_name: Optional custom table name.
            feedback_table_name: Optional custom feedback table name.
            index_name: Optional custom index name for analysis results.
            feedback_index_name: Optional custom index name for feedback.
        """
        self._pool: Optional[asyncpg.pool.Pool] = None
        self.schema = settings.DB_SCHEMA
        self.table_name = f"{self.schema}.{table_name or settings.TABLE_NAME}"
        self.feedback_table_name = f"{self.schema}.{feedback_table_name or settings.FEEDBACK_NAME}"
        self.index_name = index_name or settings.CACHE_INDEX
        self.feedback_index_name = feedback_index_name or settings.FEEDBACK_CACHE_INDEX

        connection_string = settings.CONNECTION_STRING
        masked_connection_string = connection_string.replace(
            connection_string.split(":")[2].split("@")[0], "******"
        )
        logger.info(
            f"[PostgreSQLCache] Using connection string: {masked_connection_string}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"[PostgreSQL] Connection retry attempt {retry_state.attempt_number}"
        )
    )
    async def initialize_pool(self) -> None:
        """Initialize a connection pool."""
        if not self._pool or self._pool.is_closing():
            with PostgreSQLCache._pool_creation_lock:  # 락 획득
                # 락 내에서 다시 확인하여 다른 스레드가 생성하지 않았는지 확인
                if not self._pool or self._pool.is_closing():
                    try:
                        logger.info(
                            "[PostgreSQLCache] Creating connection pool.")
                        self._pool = await asyncpg.create_pool(
                            dsn=settings.CONNECTION_STRING,
                            min_size=settings.POSTGRES_POOL_MIN_SIZE,
                            max_size=settings.POSTGRES_POOL_MAX_SIZE
                        )
                        logger.info(
                            "[PostgreSQLCache] Connection pool created successfully.")
                    except Exception as e:
                        logger.error(
                            f"[PostgreSQLCache] Failed to create connection pool: {str(e)}")
                        raise

    async def _get_pool(self) -> asyncpg.pool.Pool:
        """Return a connection pool, assuming initialize_pool in action.."""
        if not self._pool or self._pool.is_closing():
            await self.initialize_pool()
        return self._pool

    def _create_empty_cache(self) -> Dict:
        """
        Create an empty cache structure.

        Returns:
            Dict: Empty cache structure.
        """
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
                'detailed_result': None,
                'summary': None
            }
        }

    async def get(self, company_name: str, analysis_type: str, analysis_metric: str) -> Optional[Dict]:
        """
        Retrieve analysis data from PostgreSQL cache based on company and analysis parameters.

        Args:
            company_name (str): Name of the company to retrieve data for
            analysis_type (str): Type of analysis (strength, weakness, or insight)
            analysis_metric (str): Metric type for analysis (e.g., growth, profitability)
                                For insight, format should be 'strength_metric/weakness_metric'

        Returns:
            Optional[Dict]: Dictionary containing analysis results or None if no data found
        """
        try:
            today = datetime.now()
            current_month = f"{today.year}{today.month:02d}"
            prev_month = f"{today.year}{today.month-1:02d}" if today.month > 1 else f"{today.year-1}12"
            threshold = int(settings.THRESHOLD)
            result = self._create_empty_cache()

            pool = await self._get_pool()
            if pool is None:
                raise asyncpg.PostgresConnectionError(
                    "Failed to create connection pool")

            async with pool.acquire() as conn:
                async with conn.transaction():
                    if today.day >= settings.THRESHOLD:
                        base_query = f"""
                            SELECT type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy
                            FROM {self.table_name}
                            WHERE nm_comp = $1 
                            AND substr(at_created, 1, 6) = $2
                            AND CAST(substr(at_created, 7, 2) AS INTEGER) >= $3
                        """
                        params = [company_name, current_month, threshold]
                    else:
                        base_query = f"""
                            SELECT type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy
                            FROM {self.table_name}
                            WHERE nm_comp = $1 AND (
                                (substr(at_created, 1, 6) = $2) OR
                                (substr(at_created, 1, 6) = $3 AND CAST(substr(at_created, 7, 2) AS INTEGER)>= $4)
                            )
                        """
                        params = [company_name, current_month,
                                  prev_month, threshold]

                    if analysis_type == 'insight':
                        strength_metric, weakness_metric = analysis_metric.split(
                            '/')
                        base_query += f"""
                            AND ((type_analy = 'strength' AND type_analy_metric = ${len(params) + 1}) OR
                                (type_analy = 'weakness' AND type_analy_metric = ${len(params) + 2}) OR
                                (type_analy = 'insight' AND type_analy_metric = ${len(params) + 3}))
                            ORDER BY at_created DESC
                        """
                        params.extend(
                            [strength_metric, weakness_metric, analysis_metric])
                    else:
                        base_query += f"""
                            AND type_analy = ${len(params) + 1}
                            AND type_analy_metric = ${len(params) + 2}
                            ORDER BY at_created DESC
                        """
                        params.extend([analysis_type, analysis_metric])

                    rows = await conn.fetch(base_query, *params)
                    if not rows:
                        return None

                    for row in rows:
                        result[row['type_analy']] = {
                            'analysis_metric': row['type_analy_metric'],
                            'detailed_result': row['rslt_dtl_analy'],
                            'summary': row['sumry_analy']
                        }
                    return result

        except asyncpg.PostgresConnectionError as e:
            logger.error(f"[PostgreSQL] Connection error: {str(e)}")
            self._pool = None
            return None
        except Exception as e:
            logger.error(f"[PostgreSQL] Get error: {str(e)}")
            return None

    async def set(self, nm_comp: str, value: Dict, type_analy: str) -> None:
        try:
            pool = await self._get_pool()
            if pool is None:
                raise asyncpg.PostgresConnectionError(
                    "Failed to create connection pool")

            async with pool.acquire() as conn:
                async with conn.transaction():
                    if type_analy in ['strength', 'weakness']:
                        update_query = f"""
                            UPDATE {self.table_name}
                            SET rslt_dtl_analy = $4,
                                sumry_analy = $5,
                                at_created = to_char(now(), 'YYYYMMDDHH24MISS')
                            WHERE nm_comp = $1 
                            AND type_analy = $2 
                            AND type_analy_metric = $3
                        """
                        update_result = await conn.execute(
                            update_query,
                            nm_comp,
                            type_analy,
                            value[type_analy]['analysis_metric'],
                            value[type_analy]['detailed_result'],
                            value[type_analy]['summary']
                        )

                        if update_result == "UPDATE 0":
                            insert_query = f"""
                                INSERT INTO {self.table_name}
                                (nm_comp, type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy, at_created)
                                VALUES ($1, $2, $3, $4, $5, to_char(now(), 'YYYYMMDDHH24MISS'))
                            """
                            await conn.execute(
                                insert_query,
                                nm_comp,
                                type_analy,
                                value[type_analy]['analysis_metric'],
                                value[type_analy]['detailed_result'],
                                value[type_analy]['summary']
                            )
                    else:
                        update_query = f"""
                            UPDATE {self.table_name}
                            SET sumry_analy = $4,
                                at_created = to_char(now(), 'YYYYMMDDHH24MISS')
                            WHERE nm_comp = $1 
                            AND type_analy = $2 
                            AND type_analy_metric = $3
                        """
                        update_result = await conn.execute(
                            update_query,
                            nm_comp,
                            type_analy,
                            value['insight']['analysis_metric'],
                            value['insight']['summary']
                        )

                        if update_result == "UPDATE 0":
                            insert_query = f"""
                                INSERT INTO {self.table_name}
                                (nm_comp, type_analy, type_analy_metric, sumry_analy, at_created)
                                VALUES ($1, $2, $3, $4, to_char(now(), 'YYYYMMDDHH24MISS'))
                            """
                            await conn.execute(
                                insert_query,
                                nm_comp,
                                type_analy,
                                value['insight']['analysis_metric'],
                                value['insight']['summary']
                            )

            logger.info(
                f"[PostgreSQLCache] Successfully set data for {nm_comp}, {type_analy}")

        except asyncpg.PostgresConnectionError as e:
            logger.error(f"[PostgreSQL] Connection error: {str(e)}")
            self._pool = None
            raise

        except Exception as e:
            logger.error(
                f"[PostgreSQLCache] Set error for company {nm_comp}: {str(e)}")
            raise

    async def close(self) -> None:
        """애플리케이션 종료 시 호출하여 커넥션 풀을 닫음."""
        if self._pool:
            try:
                await self._pool.close()
                logger.info("[PostgreSQL] Connection pool closed")
            except Exception as e:
                logger.error(
                    f"[PostgreSQLCache] Error while closing connection pool: {str(e)}")
