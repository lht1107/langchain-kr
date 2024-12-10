# 기본 모듈
import sqlite3
import json
from typing import Dict, Optional
from datetime import datetime

# 비동기 지원
import aiosqlite

# 내부 모듈
from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteCache(BaseCache):
    """SQLite 기반 캐시 구현"""

    def __init__(self):
        self.db_path = settings.SQLITE_DB_PATH
        self._init_db()

    def _init_db(self):
        """캐시 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            # 분석 결과 테이블
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_metric TEXT NOT NULL,
                    detailed_result TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 피드백 테이블
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_FEEDBACK_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_metric TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 인덱스가 없을 때만 생성
            try:
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_analysis_lookup 
                    ON {settings.SQLITE_TABLE_NAME} 
                    (company_name, created_at DESC, analysis_type, analysis_metric)
                """)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    raise e

    async def get(
        self,
        company_name: str,
        analysis_type: str,
        analysis_metric: str
    ) -> Optional[Dict]:
        try:
            today = datetime.now()
            current_month = f"{today.year}-{today.month:02d}"
            prev_month = f"{today.year}-{today.month-1:02d}" if today.month > 1 else f"{today.year-1}-12"

            result = {
                'strength': {'analysis_metric': None, 'detailed_result': None, 'summary': None},
                'weakness': {'analysis_metric': None, 'detailed_result': None, 'summary': None},
                'insight': {'analysis_metric': None, 'summary': None}
            }

            async with aiosqlite.connect(self.db_path) as conn:
                if today.day >= settings.THRESHOLD:
                    base_query = f"""
                        SELECT analysis_type, analysis_metric, detailed_result, summary
                        FROM {settings.SQLITE_TABLE_NAME}
                        WHERE company_name = ? 
                        AND strftime('%Y-%m', created_at) = ?
                        AND strftime('%d', created_at) >= ?
                    """
                    params = [company_name, current_month,
                              str(settings.THRESHOLD)]
                else:
                    base_query = f"""
                        SELECT analysis_type, analysis_metric, detailed_result, summary
                        FROM {settings.SQLITE_TABLE_NAME}
                        WHERE company_name = ? AND (
                            (strftime('%Y-%m', created_at) = ?) OR
                            (strftime('%Y-%m', created_at) = ? AND strftime('%d', created_at) >= ?)
                        )
                    """
                    params = [company_name, current_month,
                              prev_month, str(settings.THRESHOLD)]

                if analysis_type == 'insight':
                    strength_metric, weakness_metric = analysis_metric.split(
                        '/')
                    base_query += """ 
                        AND ((analysis_type = 'strength' AND analysis_metric = ?) OR
                            (analysis_type = 'weakness' AND analysis_metric = ?) OR
                            (analysis_type = 'insight' AND analysis_metric = ?))
                    """
                    params.extend(
                        [strength_metric, weakness_metric, analysis_metric])
                else:
                    base_query += " AND analysis_type = ? AND analysis_metric = ?"
                    params.extend([analysis_type, analysis_metric])

                base_query += " ORDER BY created_at DESC"

                async with conn.execute(base_query, params) as cursor:
                    rows = await cursor.fetchall()

                    if not rows:
                        return None

                    for row in rows:
                        analysis_type_row, metric, detailed, summary = row
                        result[analysis_type_row] = {
                            'analysis_metric': metric,
                            'detailed_result': detailed,
                            'summary': summary
                        }

                    return result

        except Exception as e:
            logger.error(f"[SQLite] Get error: {str(e)}")
            return None

    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    if analysis_type in ['strength', 'weakness']:
                        await conn.execute(
                            f"""INSERT INTO {settings.SQLITE_TABLE_NAME}
                            (company_name, analysis_type, analysis_metric, detailed_result, summary)
                            VALUES (?, ?, ?, ?, ?)""",
                            (company_name,
                             analysis_type,
                             value[analysis_type]['analysis_metric'],
                             value[analysis_type]['detailed_result'],
                             value[analysis_type]['summary'])
                        )
                    elif analysis_type == 'insight':
                        await conn.execute(
                            f"""INSERT INTO {settings.SQLITE_TABLE_NAME}
                            (company_name, analysis_type, analysis_metric, detailed_result, summary)
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
        """연결 종료 (컨텍스트 매니저로 관리되므로 별도 처리 불필요)"""
        pass
