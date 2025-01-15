# 기본 모듈
import os
import platform
import multiprocessing
import signal
from typing import Any, Dict

# 서버 관련
import uvicorn
from fastapi import FastAPI

# 내부 모듈
from utils.logger import get_logger
from core.config import settings

# Logger 초기화
logger = get_logger(__name__)

# Windows 환경 체크
IS_WINDOWS = platform.system() == "Windows"

# Gunicorn은 Windows가 아닐 때만 임포트
if not IS_WINDOWS:
    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        """Gunicorn 애플리케이션 래퍼 (Linux/Mac 전용)"""

        def __init__(self, app: FastAPI, options: Dict[str, Any] = None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self) -> None:
            config = {
                key: value for key, value in self.options.items()
                if key in self.cfg.settings and value is not None
            }
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self) -> FastAPI:
            return self.application


class ServerConfig:
    """서버 설정 관리 클래스"""

    def __init__(self):
        self.port = int(os.getenv("PORT", settings.SERVER_PORT))
        self.host = os.getenv("HOST", settings.SERVER_HOST)
        self.log_level = os.getenv("LOG_LEVEL", settings.SERVER_LOG_LEVEL)
        self.workers = settings.WORKERS_PER_CORE * multiprocessing.cpu_count() + 1
        self.reload = settings.ENV == "development"
        self.reload_dirs = settings.RELOAD_DIRS

    @property
    def bind(self) -> str:
        return f"{self.host}:{self.port}"


def handle_shutdown(signum: int, frame: Any) -> None:
    """서버 종료 핸들러"""
    logger.info("Initiating server shutdown...")
    logger.info("Cleaning up resources...")
    logger.info("Server shutdown complete")


def run_server(app: FastAPI) -> None:
    """FastAPI 서버 실행"""
    config = ServerConfig()

    # 종료 시그널 핸들러 등록
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        logger.info(f"Starting server on {config.bind}")
        logger.info(f"Environment: {settings.ENV}")

        if IS_WINDOWS:
            # Windows: Uvicorn
            uvicorn.run(
                "main:app",
                host=config.host,
                port=config.port,
                reload=config.reload,
                log_level=config.log_level,
                reload_dirs=config.reload_dirs if config.reload else None
            )
        else:
            # Linux/Mac: Gunicorn + Uvicorn
            options = {
                'bind': config.bind,
                'workers': config.workers,
                'worker_class': 'uvicorn.workers.UvicornWorker',
                'reload': config.reload,
                'reload_extra_files': config.reload_dirs if config.reload else None,
                'loglevel': config.log_level
            }
            StandaloneApplication(app, options).run()

    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        raise
