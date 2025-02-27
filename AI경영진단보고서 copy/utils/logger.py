# utils/logger.py

import os
import logging
from logging import Logger, StreamHandler, FileHandler, Formatter
from colorlog import ColoredFormatter
from core.config import settings


def setup_logging():
    """
    초기 로깅 설정을 수행하는 함수.
    콘솔과 파일에 로그를 출력하며, 외부 라이브러리의 로그 레벨을 조정.
    """
    # 로그 디렉토리 생성
    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR)

    # 파일 로그 포맷
    file_format = Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # 콘솔 로그 포맷 (색상 지원)
    console_format = ColoredFormatter(
        fmt="%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red,bg_white",
        },
    )

    # 파일 핸들러
    file_handler = FileHandler(settings.LOG_FILE_PATH)
    file_handler.setLevel(settings.SERVER_LOG_LEVEL)  # config에서 설정한 레벨 사용
    file_handler.setFormatter(file_format)

    # 콘솔 핸들러
    console_handler = StreamHandler()
    console_handler.setLevel(settings.SERVER_LOG_LEVEL)  # config에서 설정한 레벨 사용
    console_handler.setFormatter(console_format)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.SERVER_LOG_LEVEL)  # config에서 설정한 레벨 사용
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 외부 라이브러리 로그 레벨 조정
    noisy_loggers = [
        "httpx",
        "asyncio",
        "watchfiles.main",
        "urllib3",
        "preprocessing",
        "analysis",
        "validation",
        "utils",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> Logger:
    """
    특정 이름의 로거를 반환.

    Args:
        name (str): 로거 이름.

    Returns:
        Logger: 구성된 로거 인스턴스.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Info message example")
    """
    return logging.getLogger(name)


# 애플리케이션 시작 시 로깅 초기화
if not logging.getLogger().hasHandlers():
    setup_logging()
