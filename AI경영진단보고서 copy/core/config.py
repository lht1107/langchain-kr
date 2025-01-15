import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from typing import ClassVar, Dict, List, Optional

# Load environment variables from the .env file
load_dotenv()


class DBConfigMixin(BaseModel):
    """DB 관련 공통 설정"""
    TABLE_NAME: ClassVar[str] = "ai_mdr_analy_rslt"
    FEEDBACK_NAME: ClassVar[str] = "ai_mdr_feedback "
    CACHE_INDEX: ClassVar[str] = "ai_mdr_analy_rslt_nm_comp_idx"
    FEEDBACK_CACHE_INDEX: ClassVar[str] = "ai_mdr_feedback_nm_comp_idx"
    # 아래는 신용분석 Agent table 정의 (legacy; table을 postgresql에 정의 후 다시 정리할 필요)
    CREDIT_TABLE_NAME: ClassVar[str] = "CREDIT_CONSULTING"
    CREDIT_FEEDBACK_NAME: ClassVar[str] = "USER_FEEDBACK"
    CREDIT_CACHE_INDEX: ClassVar[str] = "idx_cache_index"


class BaseConfig(BaseSettings, DBConfigMixin):
    """공통 설정 클래스"""
    ENV: str = os.getenv("ENV", "development").lower()
    PC: str = os.getenv("PC", "local").lower()

    # 기본 경로 설정
    BASE_DIR: ClassVar[str] = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    PROMPTS_DIR: ClassVar[str] = os.path.join(BASE_DIR, "prompts") if PC == "local" else os.path.join(
        "C:\\DuzonBizon\\SmartAPlus\\AI경영진단보고서_local", "prompts")
    DATA_PATH: ClassVar[str] = os.path.join(BASE_DIR, "data")
    LOG_DIR: ClassVar[str] = os.path.join(BASE_DIR, "logs")
    LOG_FILE_PATH: ClassVar[str] = os.path.join(LOG_DIR, "application.log")

    # API 설정
    API_RATE_LIMIT: str = "5/minute"
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    PROXY_URL: str = os.getenv("PROXY_URL", "http://10.160.11.15:8500")
    PROXY_ENABLED: bool = os.getenv("PROXY_ENABLED", "false").lower() == "true"
    CLIENT_SIZE: int = 5  # 클라이언트 풀에서 관리할 OpenAI API 클라이언트의 최대 개수를 지정하는 값 (재사용)

    # 데이터베이스 타입 설정
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")

    # PostgreSQL 풀 설정
    POSTGRES_POOL_MIN_SIZE: int = 2
    POSTGRES_POOL_MAX_SIZE: int = 5
    DB_USERNAME: str = os.getenv("DB_USERNAME", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_SCHEMA: str = os.getenv("DB_SCHEMA", "cbsch")
    DB_NAME: str = os.getenv("DB_NAME", "trdb_test")

    # SQLite 설정
    SQLITE_DB_PATH: str = os.path.join(
        BASE_DIR, "database", "sqlite_cache.db")
    SQLITE_CREDIT_DB_PATH: str = os.path.join(
        BASE_DIR, "database", "sqlite_credit_cache.db")

    # 메트릭스 설정: 6가지
    METRICS: ClassVar[List[str]] = [
        "growth", "profitability", "partner_stability",
        "financial_stability", "hr", "cashflow"
    ]

    # 지표별 입력되는 main key: 예를 들어, annual_revenue의 value에는 연도별 매출액 관련 정보가 들어가있어야함함
    REQUIRED_TAGS: ClassVar[Dict[str, List[str]]] = {
        "growth": [
            "{latest_year_month}", "{annual_revenue}", "{annual_assets}",
            "{monthly_revenue}", "{monthly_growth}"
        ],
        "profitability": [
            "{latest_year_month}", "{annual_profit}", "{annual_margins}",
            "{monthly_profit}", "{monthly_margins}"
        ],
        "partner_stability": [
            "{latest_year_month}", "{annual_top5_sales}", "{monthly_top5_sales}",
            "{annual_top5_purchase}", "{monthly_top5_purchase}"
        ],
        "financial_stability": [
            "{latest_year_month}", "{annual_borrowings}", "{annual_liquidity}",
            "{monthly_borrowings}", "{monthly_liquidity}"
        ],
        "hr": [
            "{monthly_employee_metrics}", "{annual_employee_metrics}",
            "{tenure_distribution}", "{age_distribution}",
            "{average_tenure_by_age}", "{average_salary_by_age}",
            "{monthly_salary_and_revenue_metrics}"
        ],
        "cashflow": [
            "{operating_activities}", "{investing_activities}",
            "{financing_activities}", "{monthly_cashflow}"
        ]
    }

    # 기본 설정
    TIMESTAMP: ClassVar[str] = "%Y-%m-%d"
    THRESHOLD: ClassVar[int] = 26

    SERVER_HOST: str = os.getenv("HOST", "localhost")
    SERVER_SCHEME: str = os.getenv("SCHEME", "http")
    SERVER_PORT: int = int(os.getenv("PORT", 8000))
    SERVER_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SLOW_REQUEST_THRESHOLD: float = float(
        os.getenv("SLOW_REQUEST_THRESHOLD", 20.0))

    # 워커 설정
    WORKERS_PER_CORE: int = 2
    WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"

    # 리로드 설정
    RELOAD_DIRS: ClassVar[List[str]] = [
        "analysis", "database", "preprocessing", "utils",
        "core", "api", "prompts"
    ]

    @property
    def WORKERS_COUNT(self) -> int:
        """총 워커 수 계산"""
        import multiprocessing
        return self.WORKERS_PER_CORE * multiprocessing.cpu_count() + 1

    def to_safe_dict(self) -> Dict:
        """민감 정보를 제외한 설정 반환"""
        return self.dict(exclude={
            "OPENAI_API_KEY",
            "DB_PASSWORD"
        })

    @property
    def CONNECTION_STRING(self) -> str:
        """데이터베이스 연결 문자열 생성"""
        if self.DB_TYPE == "postgresql":
            return (
                f"postgresql://{self.DB_USERNAME}:{self.DB_PASSWORD}@"
                f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            )
        elif self.DB_TYPE == "sqlite":
            return f"sqlite:///{self.SQLITE_DB_PATH}"
        return ""

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=True,
        extra='ignore'
    )


class DevelopmentConfig(BaseConfig):
    """개발 환경 설정"""
    # 서버 설정
    RELOAD_ENABLED: bool = True


class ProductionConfig(BaseConfig):
    """운영 환경 설정"""
    # 서버 설정
    SERVER_HOST: str = os.getenv("HOST", "your-production-domain.com")
    SERVER_SCHEME: str = "https"
    SERVER_PORT: int = 443
    SERVER_LOG_LEVEL: str = "WARNING"
    RELOAD_ENABLED: bool = False


def get_settings() -> BaseConfig:
    """환경에 따른 설정 인스턴스 반환"""
    env = os.getenv("ENV", "development").lower()
    return ProductionConfig() if env == "production" else DevelopmentConfig()


# 설정 인스턴스 생성
settings = get_settings()
