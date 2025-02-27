# 기본 모듈
import sqlite3
from typing import Dict, Optional
from datetime import datetime

# 비동기 지원
import aiosqlite

# 내부 모듈
from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteCreditCache(BaseCache):
    """SQLite 기반 신용 컨설팅 캐시 구현"""

    def __init__(self):
        self.db_path = settings.SQLITE_CREDIT_DB_PATH
        self._init_db()

    def _init_db(self):
        """캐시 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            # 신용 분석 결과 테이블
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_CREDIT_TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    detailed_analysis TEXT,
                    final_report TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 피드백 테이블
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_CREDIT_FEEDBACK_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 인덱스 생성
            try:
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {settings.SQLITE_CREDIT_CACHE_INDEX} 
                    ON {settings.SQLITE_CREDIT_TABLE_NAME} 
                    (company_name, created_at DESC, analysis_type)
                """)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    raise e

    async def get(
        self,
        company_name: str,
        analysis_type: str
    ) -> Optional[Dict]:
        """캐시된 신용 분석 결과 조회"""
        try:
            today = datetime.now()
            current_month = f"{today.year}-{today.month:02d}"
            prev_month = f"{today.year}-{today.month-1:02d}" if today.month > 1 else f"{today.year-1}-12"

            result = {
                'detailed_analysis': None,
                'final_report': None
            }

            async with aiosqlite.connect(self.db_path) as conn:
                if today.day >= settings.THRESHOLD:
                    base_query = f"""
                        SELECT analysis_type, detailed_analysis, final_report
                        FROM {settings.SQLITE_CREDIT_TABLE_NAME}
                        WHERE company_name = ? 
                        AND analysis_type = ?
                        AND strftime('%Y-%m', created_at) = ?
                        AND strftime('%d', created_at) >= ?
                    """
                    params = [company_name, analysis_type,
                              current_month, str(settings.THRESHOLD)]
                else:
                    base_query = f"""
                        SELECT analysis_type, detailed_analysis, final_report
                        FROM {settings.SQLITE_CREDIT_TABLE_NAME}
                        WHERE company_name = ?
                        AND analysis_type = ?
                        AND (
                            (strftime('%Y-%m', created_at) = ?) OR
                            (strftime('%Y-%m', created_at) = ? AND strftime('%d', created_at) >= ?)
                        )
                    """
                    params = [company_name, analysis_type,
                              current_month, prev_month, str(settings.THRESHOLD)]

                base_query += " ORDER BY created_at DESC LIMIT 1"

                async with conn.execute(base_query, params) as cursor:
                    row = await cursor.fetchone()

                    if not row:
                        return None

                    _, detailed_analysis, final_report = row
                    result['detailed_analysis'] = detailed_analysis
                    result['final_report'] = final_report

                    return result

        except Exception as e:
            logger.error(f"[SQLite] Get credit cache error: {str(e)}")
            return None

    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        """신용 분석 결과 저장"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    await conn.execute(
                        f"""INSERT INTO {settings.SQLITE_CREDIT_TABLE_NAME}
                           (company_name, analysis_type, detailed_analysis, final_report)
                           VALUES (?, ?, ?, ?)""",
                        (
                            company_name,
                            analysis_type,
                            value['detailed_analysis'],
                            value['final_report']
                        )
                    )
                    await conn.commit()
                    logger.debug(
                        f"[SQLite] Successfully stored {analysis_type} credit analysis for company: {company_name}")
                except Exception as e:
                    await conn.rollback()
                    logger.error(f"[SQLite] Transaction failed: {str(e)}")
                    raise e
        except Exception as e:
            logger.error(f"[SQLite] Set credit cache error: {str(e)}")
            raise

    def _create_empty_cache(self) -> Dict:
        """빈 캐시 데이터 구조 생성"""
        return {
            'detailed_analysis': None,
            'final_report': None
        }

    async def close(self) -> None:
        """연결 종료 (컨텍스트 매니저로 관리되므로 별도 처리 불필요)"""
        pass
