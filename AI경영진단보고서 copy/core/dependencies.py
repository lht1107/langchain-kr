from queue import Queue
from typing import Optional, Tuple
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from slowapi import Limiter
from slowapi.util import get_remote_address
from httpx import Client
from utils.logger import get_logger
from core.config import settings


from core.cache import CacheManager


# Logger initialization
logger = get_logger(__name__)

# Prevent duplicate log initialization messages
if not hasattr(logger, "initialized"):
    logger.info("Starting application lifespan management.")
    logger.initialized = True  # Prevent future duplicate logs

# API 요청 제한 설정
limiter = Limiter(key_func=get_remote_address)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model_name: str
    temperature: float
    max_tokens: int
    streaming: bool = True


class LLMClientPool:
    """LLM Client Pooling for Reusability"""

    def __init__(self, size: int, api_key: str, proxy_url: Optional[str] = None):
        self.size = size
        self.pool = Queue(maxsize=size)
        self.api_key = api_key
        self.proxy_url = settings.PROXY_URL if settings.PROXY_ENABLED else None
        self.callbacks = [StreamingStdOutCallbackHandler()]
        self._log_proxy_usage_done = False  # Ensures logging happens only once

        # Log and initialize clients
        self._log_proxy_usage()
        self.common_config = LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=2000
        )
        self.summary_config = LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=500
        )
        self.insight_config = LLMConfig(
            model_name="gpt-4o",
            temperature=1.2,
            max_tokens=500
        )
        self.credit_config = LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=5000
        )

        # Pre-initialize the pool with clients
        for _ in range(self.size):
            self.pool.put(self._create_client(self.common_config))

    def _log_proxy_usage(self):
        """Log proxy usage details once."""
        if not self._log_proxy_usage_done:
            logger.info(
                f"Initializing LLM client pool with {self.size} pre-configured clients.")
            if self.proxy_url:
                logger.info(f"Using proxy: {self.proxy_url}")
            else:
                logger.info("Directly connecting to OpenAI API.")
            self._log_proxy_usage_done = True

    def _create_client(self, config: LLMConfig) -> ChatOpenAI:
        """Create a single LLM client."""
        http_client = Client(proxies=self.proxy_url,
                             verify=False, follow_redirects=True) if self.proxy_url else None

        return ChatOpenAI(
            api_key=self.api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=config.streaming,
            callbacks=self.callbacks,
            client=http_client
        )

    def get_client(self, config: LLMConfig) -> ChatOpenAI:
        """Retrieve a client with a specific configuration."""
        return self._create_client(config)

    def release_client(self, client: ChatOpenAI):
        """Return a client back to the pool."""
        self.pool.put(client)


# Initialize the pool globally
client_pool = LLMClientPool(
    size=settings.CLIENT_SIZE, api_key=settings.OPENAI_API_KEY, proxy_url=settings.PROXY_URL
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Initialize CacheManager and LLM clients
    global cache_instance
    cache_instance = CacheManager()

    try:
        logger.info("Starting application lifespan management.")
        yield
    finally:
        logger.info("Shutting down application resources.")
        await cache_instance.close()


def get_llm_clients() -> Tuple[ChatOpenAI, ChatOpenAI, ChatOpenAI, ChatOpenAI]:
    """Fetch LLM clients with specific configurations."""
    try:
        # Example of different configurations based on environment
        common_client = client_pool.get_client(client_pool.common_config)
        summary_client = client_pool.get_client(client_pool.summary_config)
        insight_client = client_pool.get_client(client_pool.insight_config)
        credit_client = client_pool.get_client(client_pool.credit_config)

        return common_client, summary_client, insight_client, credit_client
    except Exception as e:
        logger.error(f"Failed to retrieve LLM clients: {e}")
        raise


# CacheManager initialization
cache_instance = CacheManager()


def get_cache() -> CacheManager:
    """
    Returns the global CacheManager instance.
    """
    return cache_instance
