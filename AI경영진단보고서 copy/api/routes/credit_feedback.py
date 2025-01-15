from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import importlib

from core.dependencies import limiter
from core.config import settings
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)

# 라우터 설정
router = APIRouter(
    prefix="/credit_feedback",
    tags=["credit_feedback"]
)


class CreditFeedback(BaseModel):
    company_name: str = Field(..., description="Company name")
    feedback_type: str = Field(..., description="Feedback type: 개선사항/오류신고/기타")
    analysis_type: str = Field(...,
                               description="Analysis type: current/hypothetical")
    feedback_text: str = Field(..., description="Feedback text")

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
                f"유효하지 않은 feedback_type: {value}. 허용되는 값: {allowed_types}"
            )
        return value

    @field_validator("analysis_type")
    def validate_analysis_type(cls, value):
        allowed_types = ["current", "hypothetical"]
        if value not in allowed_types:
            raise ValueError(
                f"유효하지 않은 analysis_type: {value}. 허용되는 값: {allowed_types}"
            )
        return value

    @field_validator("feedback_text")
    def validate_feedback_text(cls, value):
        if not value.strip():
            raise ValueError("feedback_text 필드는 비어 있을 수 없습니다.")
        if len(value) > 1000:
            raise ValueError("feedback_text는 1000자를 초과할 수 없습니다.")
        return value


@router.post("/", include_in_schema=False)
@limiter.limit(settings.API_RATE_LIMIT)
async def create_credit_feedback(feedback: CreditFeedback, request: Request):
    """
    신용 분석 피드백 생성 엔드포인트
    {
        "company_name": "Company_10",
        "feedback_type": "개선사항",
        "analysis_type": "current",
        "feedback_text": "이 부분이 궁금합니다."
    }
    """
    try:
        sqlite3 = importlib.import_module('sliqte3')
        logger.info(
            f"[Credit Feedback] Saving feedback for company: {feedback.company_name}")

        with sqlite3.connect(settings.SQLITE_CREDIT_DB_PATH) as conn:
            cursor = conn.cursor()

            # 테이블 존재 여부 확인 및 생성
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.CREDIT_FEEDBACK_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """)

            # 데이터 삽입
            cursor.execute(f"""
                INSERT INTO {settings.CREDIT_FEEDBACK_NAME} 
                (company_name, feedback_type, analysis_type, feedback_text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                feedback.company_name.strip(),
                feedback.feedback_type.strip(),
                feedback.analysis_type.strip(),
                feedback.feedback_text.strip(),
                datetime.now()
            ))
            conn.commit()

            feedback_id = cursor.lastrowid

        logger.info(
            f"[Credit Feedback] Feedback saved successfully for {feedback.company_name} with ID {feedback_id}")
        return {
            "status": "success",
            "message": "피드백이 성공적으로 저장되었습니다.",
            "feedback_id": feedback_id
        }

    except sqlite3.OperationalError as e:
        logger.error(f"[Credit Feedback] OperationalError: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="OperationalError: Could not save feedback"
        )
    except sqlite3.DatabaseError as e:
        logger.error(f"[Credit Feedback] DatabaseError: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="DatabaseError: Could not save feedback"
        )
    except ValueError as ve:
        logger.error(f"[Credit Feedback] Validation Error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=f"입력 데이터 오류: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"[Credit Feedback] Unknown error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Unknown error occurred"
        )
