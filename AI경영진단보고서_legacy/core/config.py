import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import ClassVar, Dict, List
from datetime import datetime
import pandas as pd

# Load environment variables from the .env file
load_dotenv()


class Settings(BaseSettings):

    # 환경 설정 추가
    ENV: str = "development"  # 'production'
    ACCESS_TIME: ClassVar[datetime] = pd.Timestamp(datetime(2024, 11, 26))

    # 캐시 설정
    CACHE_TYPE: str = "sqlite"  # 'redis', 'sqlite', 'postgresql', 'all'
    CACHE_CREDIT_TYPE: str = 'sqlite'

    # 프롬프트 폴더 지정
    BASE_DIR: ClassVar[str] = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))  # 모든 instance가 공유
    PROMPTS_DIR: ClassVar[str] = os.path.join(BASE_DIR, "prompts")
    DATA_PATH: ClassVar[str] = os.path.join(BASE_DIR, 'data')

    # 로그 파일 경로 설정
    LOG_DIR: ClassVar[str] = os.path.join(BASE_DIR, 'logs')
    LOG_FILE_PATH: ClassVar[str] = os.path.join(LOG_DIR, 'application.log')

    # SQLite 설정
    SQLITE_DB_PATH: str = os.path.join(BASE_DIR, 'database', 'sqlite_cache.db')
    SQLITE_TABLE_NAME: ClassVar[str] = "AI_COMMENTS"
    SQLITE_FEEDBACK_NAME: ClassVar[str] = 'USER_FEEDBACK'
    SQLITE_CACHE_INDEX: ClassVar[str] = 'idx_cache_index'

    SQLITE_CREDIT_DB_PATH: str = os.path.join(
        BASE_DIR, 'database', 'sqlite_credit_cache.db')
    SQLITE_CREDIT_CACHE_INDEX: ClassVar[str] = 'idx_cache_index'
    SQLITE_CREDIT_TABLE_NAME: ClassVar[str] = "CREDIT_CONSULTING"
    SQLITE_CREDIT_FEEDBACK_NAME: ClassVar[str] = 'USER_FEEDBACK'

    # PostgreSQL 설정
    DB_USERNAME: str = os.getenv("DB_USERNAME")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: str = os.getenv("DB_PORT")
    DB_NAME: str = os.getenv("DB_NAME")

    # Redis 설정
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_RETRY_ON_TIMEOUT: bool = True
    REDIS_HEALTH_CHECK_INTERVAL: int = 30
    CACHE_EXPIRE_TIME: int = 86400

    # API 설정
    API_RATE_LIMIT: str = "5/minute"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    METRICS: ClassVar[List[str]] = ['growth', 'profitability',
                                    'partner_stability', 'financial_stability', 'hr', 'cashflow']

    REQUIRED_TAGS: ClassVar[Dict[str, List[str]]] = {
        "growth": [
            "{latest_year_month}",
            "{annual_revenue}",
            "{annual_assets}",
            "{monthly_revenue}",
            "{monthly_growth}"
        ],
        "profitability": [
            "{latest_year_month}",
            "{annual_profit}",
            "{annual_margins}",
            "{monthly_profit}",
            "{monthly_margins}"
        ],
        "partner_stability": [
            "{latest_year_month}",
            "{annual_top5_sales}",
            "{monthly_top5_sales}",
            "{annual_top5_purchase}",
            "{monthly_top5_purchase}"
        ],
        "financial_stability": [
            "{latest_year_month}",
            "{annual_borrowings}",
            "{annual_liquidity}",
            "{monthly_borrowings}",
            "{monthly_liquidity}"
        ],
        "hr": [
            '{monthly_employee_metrics}',
            '{annual_employee_metrics}',
            '{tenure_distribution}',
            '{age_distribution}',
            '{average_tenure_by_age}',
            '{average_salary_by_age}',
            '{monthly_salary_and_revenue_metrics}'
        ],
        'cashflow': [
            '{operating_activities}',
            '{investing_activities}',
            '{financing_activities}',
            '{monthly_cashflow}'
        ]
    }

    # TIMESTAMP format
    TIMESTAMP: ClassVar[str] = '%Y-%m-%d'
    THRESHOLD: ClassVar[int] = 26

    # 서버 설정 추가
    SERVER_HOST: str = os.getenv("HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("PORT", 8000))
    SERVER_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # 서버 워커 설정
    WORKERS_PER_CORE: int = 2  # CPU 코어당 워커 수
    WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"

    # 자동 리로드 설정
    RELOAD_ENABLED: bool = True  # 개발 환경에서만 True로 설정
    RELOAD_DIRS: ClassVar[List[str]] = [
        "analysis",
        "database",
        "preprocessing",
        "utils",
        "core",
        "api"
    ]

    @property
    def WORKERS_COUNT(self) -> int:
        """워커 수 계산"""
        import multiprocessing
        return self.WORKERS_PER_CORE * multiprocessing.cpu_count() + 1

    @property
    def SERVER_CONFIG(self) -> Dict:
        """서버 설정 반환"""
        return {
            'host': self.SERVER_HOST,
            'port': self.SERVER_PORT,
            'log_level': self.SERVER_LOG_LEVEL,
            'reload': self.RELOAD_ENABLED if self.ENV == 'development' else False,
            'reload_dirs': self.RELOAD_DIRS if self.ENV == 'development' else None,
            'workers': self.WORKERS_COUNT if self.ENV == 'production' else 1
        }

    @property
    def CONNECTION_STRING(self):
        return f'postgresql+psycopg2://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'

    class Config:
        env_file = ".env"


# 설정 instance 생성
settings = Settings()
