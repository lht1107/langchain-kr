# 타입 힌트
from typing import Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager

# FastAPI 관련
from fastapi import FastAPI, Depends

# Rate Limiting 관련
from slowapi import Limiter
from slowapi.util import get_remote_address

# LangChain 관련
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 내부 모듈
from core.cache import CacheManager
from .config import settings
from utils.logger import get_logger

# Logger 초기화
logger = get_logger(__name__)

# 캐시 매니저 초기화 및 의존성 설정
cache_manager = CacheManager()


def get_cache():
    """캐시 매니저 의존성 주입을 위한 함수"""
    return cache_manager


# API 요청 제한 설정
limiter = Limiter(key_func=get_remote_address)


@dataclass
class LLMConfig:
    """LLM 설정을 위한 데이터 클래스

    Attributes:
        model_name: 사용할 모델 이름
        temperature: 생성 텍스트의 무작위성 정도 (0~2)
        max_tokens: 최대 토큰 수
        streaming: 스트리밍 응답 여부
    """
    model_name: str
    temperature: float
    max_tokens: int
    streaming: bool = True

    def __post_init__(self):
        """설정값 유효성 검증"""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


class LLMClientFactory:
    """LLM 클라이언트 생성을 위한 팩토리 클래스"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: OpenAI API 키
        """
        self.api_key = api_key
        self.callbacks = [StreamingStdOutCallbackHandler()]

    def create_client(self, config: LLMConfig) -> ChatOpenAI:
        """LLM 클라이언트 생성

        Args:
            config: LLM 설정 객체

        Returns:
            ChatOpenAI: 설정된 LLM 클라이언트
        """
        return ChatOpenAI(
            api_key=self.api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=config.streaming,
            callbacks=self.callbacks
        )


# LLM 클라이언트 전역 변수
common_llm = None    # 일반 분석용
summary_llm = None   # 요약 분석용
insight_llm = None   # 통찰 분석용


def get_llm_clients(
    factory: LLMClientFactory = Depends()
) -> Tuple[ChatOpenAI, ChatOpenAI, ChatOpenAI]:
    """LLM 클라이언트들을 생성하는 의존성 함수

    Returns:
        Tuple[ChatOpenAI, ChatOpenAI, ChatOpenAI]: (일반 분석, 요약 분석, 통찰 분석) 클라이언트
    """
    # 일반 분석용 설정
    common_config = LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0,        # 결정적 출력
        max_tokens=1000
    )

    # 요약 분석용 설정
    summary_config = LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0,        # 결정적 출력
        max_tokens=500
    )

    # 통찰 분석용 설정
    insight_config = LLMConfig(
        model_name="gpt-4o",
        temperature=1.2,      # 창의적 출력
        max_tokens=500
    )

    return (
        factory.create_client(common_config),
        factory.create_client(summary_config),
        factory.create_client(insight_config)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리

    애플리케이션 시작 시 필요한 리소스를 초기화하고,
    종료 시 리소스를 정리하는 컨텍스트 매니저
    """
    global common_llm, summary_llm, insight_llm

    # LLM 팩토리 초기화 및 의존성 등록
    llm_factory = LLMClientFactory(settings.OPENAI_API_KEY)
    app.dependency_overrides[LLMClientFactory] = lambda: llm_factory

    try:
        logger.info("Starting application initialization for LLM use")

        # LLM 클라이언트 초기화
        common_llm, summary_llm, insight_llm = get_llm_clients(llm_factory)

        logger.info("Completed initialization successfully for LLM use")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        # 리소스 정리
        logger.info("Starting application shutdown...")
        await cache_manager.close()
        logger.info("Completed application shutdown")
