from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3

from core.dependencies import limiter
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/credit_feedback",
    tags=["credit_feedback"]
)


class CreditFeedback(BaseModel):
    company_name: str
    feedback_type: str  # 개선사항/오류신고/기타
    analysis_type: str  # current/hypothetical
    feedback_text: str


@router.post("")
@limiter.limit(settings.API_RATE_LIMIT)
async def create_credit_feedback(feedback: CreditFeedback, request: Request):
    try:
        logger.info(
            f"[Credit Feedback] Saving feedback for company: {feedback.company_name}")

        with sqlite3.connect(settings.SQLITE_CREDIT_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {settings.SQLITE_CREDIT_FEEDBACK_NAME} 
                (company_name, feedback_type, analysis_type, feedback_text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                feedback.company_name,
                feedback.feedback_type,
                feedback.analysis_type,
                feedback.feedback_text,
                datetime.now()
            ))
            conn.commit()
        logger.info(
            f"[Credit Feedback] Feedback saved successfully for (feedback.company_name)")
        return {"status": "success", "message": "피드백이 성공적으로 저장되었습니다."}

    except sqlite3.OperationalError as e:
        logger.error(f"OperationalError: {str(e)}")
        raise HTTPException(
            status_code=500, detail="OperationalError: Could not save feedback")
    except sqlite3.DatabaseError as e:
        logger.error(f"DatabaseError: {str(e)}")
        raise HTTPException(
            status_code=500, detail="DatabaseError: Could not save feedback")
    except Exception as e:
        logger.error(f"[Error] Credit feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unknown error occurred")
