import os
import logging
from core.config import settings

# 로깅 설정을 위한 초기화 함수


def setup_logging():
    """로깅 설정을 초기화하는 함수"""
    # 로그 디렉토리 생성
    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR)

    # 기본 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.LOG_FILE_PATH),  # 로그 파일로 출력
            logging.StreamHandler()  # 콘솔에도 출력
        ]
    )

    # 외부 라이브러리 로깅 레벨 설정
    noisy_loggers = [
        "httpx",
        "asyncio",
        "watchfiles.main",
        "urllib3",
        "preprocessing",
        "analysis",
        "validation",
        "utils"
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# 애플리케이션 시작 시 로깅 설정 초기화
if not logging.getLogger().hasHandlers():
    setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    로거 인스턴스를 반환하는 함수.

    Args:
        name (str): 로거의 이름.

    Returns:
        logging.Logger: 설정된 로거 인스턴스.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message.")
    """
    return logging.getLogger(name)
