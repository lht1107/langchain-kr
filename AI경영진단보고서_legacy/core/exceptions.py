# FastAPI 관련
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

# 기본 모듈
import traceback
from typing import Dict, Any
from enum import IntEnum

# 내부 모듈
from core.config import settings
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)


class HTTPStatus(IntEnum):
    """HTTP 상태 코드 정의"""

    # 2xx: 성공
    OK = 200                    # 요청 성공
    CREATED = 201              # 리소스 생성 성공
    ACCEPTED = 202             # 요청 접수됨
    NO_CONTENT = 204          # 성공했지만 반환할 콘텐츠 없음

    # 4xx: 클라이언트 에러
    BAD_REQUEST = 400         # 잘못된 요청
    UNAUTHORIZED = 401        # 인증 필요
    FORBIDDEN = 403           # 권한 없음
    NOT_FOUND = 404          # 리소스를 찾을 수 없음
    METHOD_NOT_ALLOWED = 405  # 허용되지 않은 HTTP 메서드
    CONFLICT = 409           # 리소스 충돌
    TOO_MANY_REQUESTS = 429  # 요청 횟수 초과

    # 5xx: 서버 에러
    INTERNAL_SERVER_ERROR = 500  # 서버 내부 오류
    BAD_GATEWAY = 502           # 게이트웨이 오류
    SERVICE_UNAVAILABLE = 503   # 서비스 일시적 사용 불가
    GATEWAY_TIMEOUT = 504       # 게이트웨이 시간 초과


def format_error_response(
    status_code: int,
    message: str,
    extra: Dict[str, Any] = None
) -> Dict:
    """에러 응답의 포맷을 통일하는 함수

    Args:
        status_code: HTTP 상태 코드
        message: 에러 메시지
        extra: 추가 컨텍스트 정보 (선택사항)

    Returns:
        포맷팅된 에러 응답 딕셔너리
    """
    response = {
        "status": "error",
        "code": status_code,
        "message": message
    }

    if extra:
        response.update(extra)
    return response


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP 예외를 처리하는 핸들러

    Args:
        request: FastAPI 요청 객체
        exc: 발생한 HTTP 예외

    Returns:
        형식화된 JSON 에러 응답
    """
    # 에러 컨텍스트 수집
    error_context = {
        "path": request.url.path,
        "method": request.method,
        "client_host": request.client.host if request.client else None,
    }

    # 구조화된 로깅
    logger.error(
        f"[Error] HTTP {exc.status_code}: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            **error_context
        }
    )

    # JSON 응답 반환
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            extra={"context": error_context}
        ),
        headers=exc.headers,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """일반적인 예외를 처리하는 핸들러

    Args:
        request: FastAPI 요청 객체
        exc: 발생한 예외

    Returns:
        형식화된 JSON 에러 응답
    """
    # 에러 컨텍스트 수집
    error_context = {
        "path": request.url.path,
        "method": request.method,
        "client_host": request.client.host if request.client else None,
        "error_type": exc.__class__.__name__
    }

    # 개발 환경에서 추가 정보 포함
    if settings.ENV == "development":
        error_context.update({
            "traceback": traceback.format_exc(),
            "error_details": str(exc)
        })

    # 상세 로깅
    logger.exception(
        f"[Error] Unexpected error: {str(exc)}",
        extra=error_context
    )

    # JSON 응답 반환
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content=format_error_response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            message="An internal server error occurred",
            extra={"context": error_context} if settings.ENV == "development" else None
        )
    )
