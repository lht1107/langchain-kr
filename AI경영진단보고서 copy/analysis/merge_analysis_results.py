from typing import Dict

def merge_analysis_results(strength_result: str, weakness_result: str) -> str:
    """
    강점 분석 결과와 약점 분석 결과를 하나의 문자열로 병합합니다.

    Args:
        strength_result (str): 강점 분석 결과.
        weakness_result (str): 약점 분석 결과.

    Returns:
        str: 병합된 결과 문자열.
    """
    return '\n\n'.join([
        "Strength Analysis:",
        strength_result,
        "Weakness Analysis:",
        weakness_result
    ])
