from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3
from typing import Dict

from core.dependencies import limiter
from core.config import settings
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)

# 라우터 설정
router = APIRouter(
    prefix="/feedback",
    tags=["feedback"]
)

# 피드백 모델 정의


class Feedback(BaseModel):
    company_name: str
    feedback_type: str
    analysis_type: str
    analysis_metric: str
    feedback_text: str

    class Config:
        from_attributes = True


@router.post("")
@limiter.limit(settings.API_RATE_LIMIT)
async def create_feedback(feedback: Feedback, request: Request) -> Dict:  # request 파라미터 추가
    """피드백을 저장하는 End Point"""
    try:
        logger.info(
            f"[Feedback] Saving feedback for company: {feedback.company_name}")

        with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {settings.SQLITE_FEEDBACK_NAME} 
                (company_name, feedback_type, analysis_type, analysis_metric, feedback_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                feedback.company_name,
                feedback.feedback_type,
                feedback.analysis_type,
                feedback.analysis_metric,
                feedback.feedback_text,
                datetime.now()
            ))
            conn.commit()

        return {
            "status": "success",
            "message": "피드백이 성공적으로 저장되었습니다."
        }

    except sqlite3.Error as e:
        logger.error(f"[Error] Database error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"데이터베이스 오류: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[Error] Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"예상치 못한 오류가 발생했습니다: {str(e)}"
        )
