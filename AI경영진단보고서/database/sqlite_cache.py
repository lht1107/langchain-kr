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
                    cache_key TEXT,                
                    analysis_type TEXT,            
                    indicator TEXT,                
                    detailed_result TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 피드백 테이블
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.SQLITE_FEEDBACK_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    indicator TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 인덱스가 없을 때만 생성
            try:
                conn.execute(f"""
                    CREATE INDEX idx_cache_lookup ON {settings.SQLITE_TABLE_NAME}
                    (cache_key, analysis_type, indicator, created_at)
                """)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    raise e

    async def get(self, key: str, analysis_type: str = None, indicator: str = None) -> Optional[Dict]:
        """캐시 데이터 조회"""
        try:
            today = datetime.now()
            company_name = key.split(':')[0]
            current_month = f"{today.year}-{today.month:02d}"

            async with aiosqlite.connect(self.db_path) as conn:
                # 기본 쿼리 구성
                base_query = f"""
                    SELECT analysis_type, indicator, detailed_result, summary
                    FROM {settings.SQLITE_TABLE_NAME}
                    WHERE cache_key LIKE ?
                """
                params = [f"{company_name}:{current_month}-%"]

                # THRESHOLD 이후인 경우
                if today.day >= settings.THRESHOLD:
                    base_query += " AND strftime('%d', created_at) >= ?"
                    params.append(f"{settings.THRESHOLD:02d}")

                # 특정 분석 유형과 지표가 지정된 경우
                if analysis_type and indicator:
                    base_query += " AND analysis_type = ? AND indicator = ?"
                    params.extend([analysis_type, indicator])

                base_query += " ORDER BY created_at DESC LIMIT 1"

                async with conn.execute(base_query, params) as cursor:
                    rows = await cursor.fetchall()

                if not rows:
                    return None

                result = {
                    'strength': {'indicator': None, 'detailed_result': None, 'summary': None},
                    'weakness': {'indicator': None, 'detailed_result': None, 'summary': None},
                    'insight': {'indicator': None, 'summary': None}
                }

                for row in rows:
                    analysis_type, indicator, detailed, summary = row
                    if analysis_type in ['strength', 'weakness']:
                        result[analysis_type] = {
                            'indicator': indicator,
                            'detailed_result': detailed,
                            'summary': summary
                        }
                    elif analysis_type == 'insight':
                        result['insight'] = {
                            'indicator': indicator,
                            'summary': summary
                        }

                return result

        except Exception as e:
            logger.error(f"SQLite get error: {str(e)}")
            return None

    async def set(self, key: str, value: Dict, analysis_type: str) -> None:
        """캐시 데이터 저장"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    if analysis_type in ['strength', 'weakness']:
                        await conn.execute(
                            f"""INSERT INTO {settings.SQLITE_TABLE_NAME}
                            (cache_key, analysis_type, indicator, detailed_result, summary)
                            VALUES (?, ?, ?, ?, ?)""",
                            (key,
                             analysis_type,
                             value[analysis_type]['indicator'],
                             value[analysis_type]['detailed_result'],
                             value[analysis_type]['summary'])
                        )
                    elif analysis_type == 'insight':
                        await conn.execute(
                            f"""INSERT INTO {settings.SQLITE_TABLE_NAME}
                            (cache_key, analysis_type, indicator, summary)
                            VALUES (?, ?, ?, ?)""",
                            (key,
                             analysis_type,
                             value['insight']['indicator'],
                             value['insight']['summary'])
                        )

                    await conn.commit()
                    logger.debug(
                        f"[SQLite] Successfully stored {analysis_type} data for key: {key}")

                except Exception as e:
                    await conn.rollback()
                    logger.error(f"[SQLite] Transaction failed: {str(e)}")
                    raise e

        except Exception as e:
            logger.error(f"SQLite set error: {str(e)}")

    async def close(self) -> None:
        """연결 종료 (컨텍스트 매니저로 관리되므로 별도 처리 불필요)"""
        pass
