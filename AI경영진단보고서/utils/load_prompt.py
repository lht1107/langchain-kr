import os
from core.config import settings


def load_prompt(file_name: str) -> str:
    """
    주어진 파일 이름에 해당하는 프롬프트 파일을 로드합니다.

    Args:
        file_name (str): 읽어올 프롬프트 파일의 이름

    Returns:
        str: 프롬프트 파일의 내용

    Raises:
        FileNotFoundError: 프롬프트 파일을 찾을 수 없는 경우 예외 발생

    Example:
        >>> load_prompt("growth_template.txt")
        "This is the content of the growth template."

    """
    file_path = os.path.join(settings.PROMPTS_DIR, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt file '{file_name}' not found at '{file_path}'.")
