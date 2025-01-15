import importlib
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Dict

from core.dependencies import get_cache, limiter
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


class Feedback(BaseModel):
    company_name: str
    feedback_type: str
    analysis_type: str
    analysis_metric: str
    feedback_text: str

    class Config:
        from_attributes = True

    @field_validator("company_name")
    def validate_company_name(cls, value):
        if not value.strip():
            raise ValueError("company_name 필드는 비어 있을 수 없습니다.")
        return value

    @field_validator("feedback_type")
    def validate_feedback_type(cls, value):
        allowed_types = ["개선사항", "오류신고", "기타"]
        if value not in allowed_types:
            raise ValueError(
                f"유효하지 않은 feedback_type: {value}. 허용되는 값: {allowed_types}")
        return value

    @field_validator("analysis_type")
    def validate_analysis_type(cls, value):
        if not value.strip():
            raise ValueError("analysis_type 필드는 비어 있을 수 없습니다.")
        return value

    @field_validator("feedback_text")
    def validate_feedback_length(cls, value):
        if not value.strip():
            raise ValueError("feedback_text 필드는 비어 있을 수 없습니다.")
        if len(value) > 1000:
            raise ValueError("feedback_text는 1000자를 초과할 수 없습니다.")
        return value


async def save_to_postgresql(feedback: Feedback) -> int:
    """PostgreSQL에 피드백 저장"""
    try:
        # 동적 import
        asyncpg = importlib.import_module('asyncpg')
        conn = await asyncpg.connect(settings.CONNECTION_STRING)
        query = f"""
            INSERT INTO {settings.DB_SCHEMA}.{settings.FEEDBACK_NAME}
            (nm_comp, type_feedback, type_analy, type_analy_metric, conts_feedback, at_created)
            VALUES ($1, $2, $3, $4, $5, to_char(now(), 'YYYYMMDDHH24MISS'))
            RETURNING seq
        """
        feedback_id = await conn.fetchval(
            query,
            feedback.company_name.strip(),
            feedback.feedback_type.strip(),
            feedback.analysis_type.strip(),
            feedback.analysis_metric.strip(),
            feedback.feedback_text.strip()
        )
        await conn.close()
        return feedback_id
    except Exception as e:
        logger.error(f"[PostgreSQL] Feedback save error: {str(e)}")
        raise


def save_to_sqlite(feedback: Feedback) -> int:
    """SQLite에 피드백 저장"""
    try:
        # 동적 import
        sqlite3 = importlib.import_module('sqlite3')
        with sqlite3.connect(settings.SQLITE_DB_PATH, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {settings.FEEDBACK_NAME}
                (nm_comp, type_feedback, type_analy, type_analy_metric, conts_feedback, at_created)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                feedback.company_name.strip(),
                feedback.feedback_type.strip(),
                feedback.analysis_type.strip(),
                feedback.analysis_metric.strip(),
                feedback.feedback_text.strip(),
                datetime.now().strftime('%Y%m%d%H%M%S')
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"[SQLite] Feedback save error: {str(e)}")
        raise


@router.post("")
@limiter.limit(settings.API_RATE_LIMIT)
async def create_feedback(feedback: Feedback, request: Request) -> Dict:
    """피드백을 저장하는 End Point"""
    try:
        logger.info(
            f"[Feedback] Saving feedback for company: {feedback.company_name}")

        # DB 타입에 따라 저장 함수 선택
        if settings.DB_TYPE.lower() == 'postgresql':
            feedback_id = await save_to_postgresql(feedback)
        elif settings.DB_TYPE.lower() == 'sqlite':
            feedback_id = save_to_sqlite(feedback)
        else:
            raise ValueError(f"Unsupported DB_TYPE: {settings.DB_TYPE}")

        logger.info(
            f"[Feedback] Successfully saved feedback with ID {feedback_id}")
        return {
            "status": "success",
            "message": "피드백이 성공적으로 저장되었습니다.",
            "feedback_id": feedback_id
        }

    except ImportError as ie:
        logger.error(f"[Import Error] {str(ie)}")
        raise HTTPException(
            status_code=500,
            detail=f"데이터베이스 모듈을 불러올 수 없습니다: {str(ie)}"
        )

    except Exception as e:
        logger.error(f"[Error] Unexpected error: {str(e)}")
