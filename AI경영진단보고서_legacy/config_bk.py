import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import ClassVar

# Load environment variables from the .env file
load_dotenv()


class Settings(BaseSettings):

    # 데이터베이스 설정
    DB_USERNAME: str = os.getenv("DB_USERNAME")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: str = os.getenv("DB_PORT")
    DB_NAME: str = os.getenv("DB_NAME")

    # Redis 설정
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    REDIS_MAX_CONNECTIONS: int = 20
    CACHE_EXPIRE_TIME: int = 86400

    # API 설정
    API_RATE_LIMIT: str = "5/minute"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # 프롬프트 폴더 지정
    BASE_DIR: ClassVar[str] = os.path.dirname(
        os.path.abspath(__file__))  # 모든 instance가 공유
    PROMPTS_DIR: ClassVar[str] = os.path.join(BASE_DIR, "prompts")

    # 로그 파일 경로 설정
    LOG_DIR: ClassVar[str] = os.path.join(BASE_DIR, 'logs')
    LOG_FILE_PATH: ClassVar[str] = os.path.join(LOG_DIR, 'application.log')

    class Config:
        env_file = ".env"

    @property
    def CONNECTION_STRING(self):
        return f'postgresql+psycopg2://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'


# Define required tags for each analysis type
REQUIRED_TAGS = {
    "growth": ["{latest_year_month}", "{annual_revenue}", "{annual_assets}", "{monthly_revenue}", "{monthly_growth}"],
    "profitability": ["{latest_year_month}", "{annual_profit}", "{annual_margins}", "{monthly_profit}", "{monthly_margins}"],
    "partner_stability": ["{latest_year_month}", "{annual_top5_sales}", "{monthly_top5_sales}", "{annual_top5_purchase}", "{monthly_top5_purchase}"]
}

# 설정 instance 생성
settings = Settings()
