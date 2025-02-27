# main.py

from fastapi import FastAPI, HTTPException

# Rate Limiting 관련
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# 내부 모듈
from core import (
    limiter,
    log_requests,
    http_exception_handler,
    general_exception_handler,
    run_server,
    lifespan
)
from api.routes import (
    analysis_router,
    query_router,
    health_router,
    feedback_router,
    credit_router,
    credit_feedback_router,
    credit_query_router
)
from utils.logger import get_logger
import warnings

# Suppress specific XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost.core')

# Logger 초기화
logger = get_logger(__name__)

# FastAPI 애플리케이션 설정
app = FastAPI(
    title="Business Analysis API",
    description="AI경영진단보고서 분석을 위한 API",
    version="1.0.0",
    lifespan=lifespan  # Application 생명주기 관리
)

# API 요청 제한 설정
app.state.limiter = limiter  # API 호출 횟수 제한
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# HTTP 요청/응답 로깅 미들웨어
app.middleware("http")(log_requests)

# API 라우터 등록
app.include_router(health_router)    # 상태 체크 엔드포인트
app.include_router(analysis_router)  # 분석 관련 엔드포인트
app.include_router(query_router)     # 쿼리 관련 엔드포인트
app.include_router(feedback_router)  # 피드백 관련 엔드포인트
app.include_router(credit_router)    # 신용 분석 엔드포인트 추가
app.include_router(credit_feedback_router)
app.include_router(credit_query_router)

# 전역 예외 처리 설정
app.add_exception_handler(HTTPException, http_exception_handler)      # HTTP 예외
app.add_exception_handler(Exception, general_exception_handler)       # 일반 예외

# 서버 실행
if __name__ == "__main__":
    run_server(app)
