# 기본 모듈
import time
from typing import Callable

# FastAPI 관련
from fastapi import Request, Response

# 내부 모듈
from utils.logger import get_logger
from core.config import settings  # 설정값 추가

# Logger 초기화
logger = get_logger(__name__)

# 상수 정의
SLOW_REQUEST_THRESHOLD = 20.0  # 초
ERROR_STATUS_CODE = 500
WARNING_STATUS_CODE = 400


async def log_requests(request: Request, call_next: Callable) -> Response:
    """HTTP 요청에 대한 로깅 미들웨어

    모든 HTTP 요청에 대한 로깅을 처리하고 성능 모니터링을 수행합니다.

    Args:
        request: FastAPI 요청 객체
        call_next: 다음 미들웨어 또는 라우트 핸들러

    Returns:
        FastAPI 응답 객체

    Raises:
        Exception: 요청 처리 중 발생한 모든 예외
    """
    request_id = str(time.time())  # 요청 식별자 추가
    start_time = time.time()

    # 요청 정보 구성
    request_info = {
        'request_id': request_id,
        'method': request.method,
        'path': request.url.path,
        'client_host': request.client.host if request.client else None,
        'user_agent': request.headers.get('user-agent'),
    }

    try:
        # 요청 시작 로깅
        logger.info(
            f"[{request_id}] Incoming request | "
            f"Method: {request_info['method']} | "
            f"Path: {request_info['path']}"
        )

        # 요청 처리
        response = await call_next(request)
        process_time = time.time() - start_time

        # 응답 정보 추가
        request_info.update({
            'status_code': response.status_code,
            'process_time': f"{process_time:.2f}s"
        })

        # 응답 상태에 따른 로그 레벨 조정
        log_message = (
            f"[{request_id}] Request completed | "
            f"Path: {request_info['path']} | "
            f"Status: {request_info['status_code']} | "
            f"Time: {request_info['process_time']}"
        )

        if response.status_code >= ERROR_STATUS_CODE:
            logger.error(log_message)
        elif response.status_code >= WARNING_STATUS_CODE:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # 성능 모니터링
        if process_time > SLOW_REQUEST_THRESHOLD:
            logger.warning(
                f"[{request_id}] Slow request detected | "
                f"Path: {request_info['path']} | "
                f"Time: {request_info['process_time']}"
            )

        return response

    except Exception as e:
        # 예외 처리 및 로깅
        process_time = time.time() - start_time
        error_info = {
            **request_info,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'process_time': f"{process_time:.2f}s"
        }

        logger.error(
            f"[{request_id}] Request failed | "
            f"Path: {error_info['path']} | "
            f"Error: {error_info['error_type']}: {error_info['error_message']} | "
            f"Time: {error_info['process_time']}"
        )
        raise
