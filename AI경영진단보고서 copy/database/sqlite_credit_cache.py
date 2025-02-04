from typing import Dict, Optional
from datetime import datetime

from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteCreditCache(BaseCache):
    """SQLite-based implementation for credit consulting cache."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        feedback_table_name: Optional[str] = None,
        index_name: Optional[str] = None
    ):
        """Initialize SQLiteCreditCache using settings-defined configurations."""

        self.db_path = db_path or settings.SQLITE_CREDIT_DB_PATH
        self.table_name = table_name or settings.CREDIT_TABLE_NAME
        self.feedback_table_name = feedback_table_name or settings.CREDIT_FEEDBACK_NAME
        self.index_name = index_name or settings.CREDIT_CACHE_INDEX

        # Validate table names
        if not all([self.table_name, self.feedback_table_name, self.index_name]):
            raise ValueError("Table names and index name cannot be empty")
        import sqlite3
        import aiosqlite
        self.sqlite3 = sqlite3
        self.aiosqlite = aiosqlite
        self._init_db()

    def _init_db(self):
        """Initialize credit consulting cache and feedback tables."""
        try:
            with self.sqlite3.connect(self.db_path) as conn:
                # Create main credit consulting table
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        detailed_analysis TEXT,
                        final_report TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create feedback table
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.feedback_table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT NOT NULL,
                        feedback_type TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        feedback_text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create index for efficient querying
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.index_name} 
                    ON {self.table_name} 
                    (company_name, created_at DESC, analysis_type)
                """)
                conn.commit()

        except self.sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                logger.error(
                    f"[SQLiteCreditCache] Database initialization failed: {str(e)}")
                raise
        except Exception as e:
            logger.error(
                f"[SQLiteCreditCache] Unexpected error during initialization: {str(e)}")
            raise

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

            async with self.aiosqlite.connect(self.db_path) as conn:
                base_query = f"""
                    SELECT analysis_type, detailed_analysis, final_report
                    FROM {self.table_name}
                    WHERE company_name = ? 
                    AND analysis_type = ?
                    AND (
                        (strftime('%Y-%m', created_at) = ? AND strftime('%d', created_at) >= ?)
                        OR 
                        (strftime('%Y-%m', created_at) = ? AND strftime('%d', created_at) >= ?)
                    )
                    ORDER BY created_at DESC LIMIT 1
                """
                params = [
                    company_name,
                    analysis_type,
                    current_month if today.day >= settings.THRESHOLD else prev_month,
                    str(settings.THRESHOLD),
                    current_month,
                    str(settings.THRESHOLD)
                ]

                async with conn.execute(base_query, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        _, detailed_analysis, final_report = row
                        result.update({
                            'detailed_analysis': detailed_analysis,
                            'final_report': final_report
                        })
                        return result
                    return None

        except Exception as e:
            logger.error(f"[SQLiteCreditCache] Get error: {str(e)}")
            return None

    async def set(self, company_name: str, value: Dict, analysis_type: str) -> None:
        """Store credit consulting data."""
        if not all([company_name, value, analysis_type]):
            raise ValueError("Missing required parameters for cache storage")

        required_fields = {'detailed_analysis', 'final_report'}
        if not all(field in value for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        try:
            async with self.aiosqlite.connect(self.db_path) as conn:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    await conn.execute(
                        f"""INSERT INTO {self.table_name}
                           (company_name, analysis_type, detailed_analysis, final_report)
                           VALUES (?, ?, ?, ?)""",
                        (company_name, analysis_type,
                         value['detailed_analysis'], value['final_report'])
                    )
                    await conn.commit()
                    logger.debug(
                        f"[SQLiteCreditCache] Stored {analysis_type} analysis for: {company_name}")
                except Exception as e:
                    await conn.rollback()
                    logger.error(
                        f"[SQLiteCreditCache] Transaction failed: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"[SQLiteCreditCache] Set error: {str(e)}")
            raise

    def _create_empty_cache(self) -> Dict:
        """Create an empty cache structure."""
        return {
            'detailed_analysis': None,
            'final_report': None
        }

    async def close(self) -> None:
        """Cleanup placeholder."""
        logger.info("[SQLiteCreditCache] No explicit close action required")
