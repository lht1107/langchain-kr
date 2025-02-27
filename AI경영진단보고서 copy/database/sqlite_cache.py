from typing import Dict, Optional
from datetime import datetime

from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteCache(BaseCache):
    """SQLite-based implementation of a cache with asynchronous support."""

    def __init__(self, db_path: str = None, table_name: str = None,
                 feedback_table_name: str = None, index_name: str = None, feedback_index_name: str = None):
        """Initialize the SQLiteCache using settings-defined configurations or provided parameters."""

        # 분석 결과와 피드백 관련 설정
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self.table_name = table_name or settings.TABLE_NAME  # 분석 결과 테이블 이름
        self.feedback_table_name = feedback_table_name or settings.FEEDBACK_NAME  # 피드백 테이블 이름
        self.index_name = index_name or settings.CACHE_INDEX  # 분석 결과 인덱스 이름
        self.feedback_index_name = feedback_index_name or settings.FEEDBACK_CACHE_INDEX  # 피드백 인덱스 이름
        import sqlite3
        import aiosqlite
        self.sqlite3 = sqlite3
        self.aiosqlite = aiosqlite
        self._init_db()

    def _init_db(self):
        """Initialize the cache and feedback tables if they do not exist."""
        with self.sqlite3.connect(self.db_path) as conn:
            # Analysis results table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, -- 순번
                    nm_comp TEXT NOT NULL, -- 회사 이름
                    type_analy TEXT NOT NULL, -- 분석 유형 (strength, weakness, insight)
                    type_analy_metric TEXT NOT NULL, -- 분석 지표 유형 (growth, profitability, partner_stability, financial_stability, hr, cashflow)
                    rslt_dtl_analy TEXT NULL, -- 상세 분석 결과
                    sumry_analy TEXT NOT NULL, -- 분석 요약
                    at_created TEXT DEFAULT (strftime('%Y%m%d%H%M%S', 'now')) NOT NULL, -- 생성일자
                    dt_insert TEXT NULL, -- 입력일시
                    dt_modify TEXT NULL -- 수정일시
                )
            """)

            # Feedback table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.feedback_table_name} (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, -- 순번
                    nm_comp TEXT NOT NULL, -- 회사 이름
                    type_feedback TEXT NOT NULL, -- 피드백 유형 (개선사항, 오류신고, 기타)
                    type_analy TEXT NOT NULL, -- 분석 유형 (strength, weakness, insight)
                    type_analy_metric TEXT NOT NULL, -- 분석 지표 유형 (growth, profitability, partner_stability, financial_stability, hr, cashflow)
                    conts_feedback TEXT NULL, -- 피드백 내용
                    at_created TEXT DEFAULT (strftime('%Y%m%d%H%M%S', 'now')) NULL, -- 생성일자
                    dt_insert TEXT NULL, -- 입력일시
                    dt_modify TEXT NULL -- 수정일시
                )
            """)

            # Create index for analysis results
            try:
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.index_name} 
                    ON {self.table_name} 
                    (nm_comp, at_created DESC, type_analy, type_analy_metric)
                """)

                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.feedback_index_name} 
                    ON {self.feedback_table_name} 
                    (nm_comp, at_created DESC, type_analy, type_analy_metric)
                """)
            except self.sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    raise e

    async def get(self, company_name: str, analysis_type: str, analysis_metric: str) -> Optional[Dict]:
        try:
            today = datetime.now()
            current_month = f"{today.year}{today.month:02d}"
            prev_month = f"{today.year}{today.month-1:02d}" if today.month > 1 else f"{today.year-1}12"

            result = self._create_empty_cache()

            async with self.aiosqlite.connect(self.db_path) as conn:
                # 날짜 조건 설정
                if today.day >= settings.THRESHOLD:
                    base_query = f"""
                        SELECT type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy
                        FROM {self.table_name}
                        WHERE nm_comp = ? 
                        AND substr(at_created, 1, 6) = ?
                        AND substr(at_created, 7, 2) >= ?
                    """
                    params = [company_name, current_month,
                              str(settings.THRESHOLD)]
                else:
                    base_query = f"""
                        SELECT type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy
                        FROM {self.table_name}
                        WHERE nm_comp = ? AND (
                            (substr(at_created, 1, 6) = ?) OR
                            (substr(at_created, 1, 6) = ? AND substr(at_created, 7, 2) >= ?)
                        )
                    """
                    params = [company_name, current_month,
                              prev_month, str(settings.THRESHOLD)]

                # 분석 타입에 따른 조건 추가
                if analysis_type == 'insight':
                    strength_metric, weakness_metric = analysis_metric.split(
                        '/')
                    base_query += f"""
                        AND ((type_analy = 'strength' AND type_analy_metric = ?) OR
                            (type_analy = 'weakness' AND type_analy_metric = ?) OR
                            (type_analy = 'insight' AND type_analy_metric = ?))
                        ORDER BY at_created DESC
                    """
                    params.extend(
                        [strength_metric, weakness_metric, analysis_metric])
                else:
                    base_query += """
                        AND type_analy = ? 
                        AND type_analy_metric = ?
                        ORDER BY at_created DESC
                    """
                    params.extend([analysis_type, analysis_metric])

                async with conn.execute(base_query, params) as cursor:
                    rows = await cursor.fetchall()

                    if not rows:
                        return None

                    # 결과 매핑
                    for row in rows:
                        analysis_type_row, metric, detailed, summary = row
                        result[analysis_type_row] = {
                            'analysis_metric': metric,
                            'detailed_result': detailed,
                            'summary': summary
                        }

                    # logger.info(result)
                    return result

        except Exception as e:
            logger.error(f"[SQLite] Get error: {str(e)}")
            return None

    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        """
        Store analysis data in the cache.

        Args:
            company_name (str): Name of the company (stored as nm_comp).
            value (Dict): Data to store.
            analysis_type (str): Type of analysis (stored as type_analy).
        """
        try:
            async with self.aiosqlite.connect(self.db_path) as conn:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    if analysis_type in ['strength', 'weakness']:
                        await conn.execute(
                            f"""INSERT INTO {self.table_name}
                            (nm_comp, type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy)
                            VALUES (?, ?, ?, ?, ?)""",
                            (company_name,
                             analysis_type,
                             value[analysis_type]['analysis_metric'],
                             value[analysis_type]['detailed_result'],
                             value[analysis_type]['summary'])
                        )
                    elif analysis_type == 'insight':
                        await conn.execute(
                            f"""INSERT INTO {self.table_name}
                            (nm_comp, type_analy, type_analy_metric, rslt_dtl_analy, sumry_analy)
                            VALUES (?, ?, ?, ?, ?)""",
                            (company_name,
                             analysis_type,
                             value['insight']['analysis_metric'],
                             None,
                             value['insight']['summary'])
                        )

                    await conn.commit()
                    logger.debug(
                        f"[SQLite] Successfully stored {analysis_type} data for company: {company_name}")

                except Exception as e:
                    await conn.rollback()
                    logger.error(f"[SQLite] Transaction failed: {str(e)}")
                    raise e

        except Exception as e:
            logger.error(f"[SQLite] Set error: {str(e)}")
            raise

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
                'summary': None
            }
        }

    async def close(self) -> None:
        """
        Placeholder for explicit cleanup. SQLite does not require explicit closure.
        """
        logger.info("[SQLiteCache] No explicit close action required.")
