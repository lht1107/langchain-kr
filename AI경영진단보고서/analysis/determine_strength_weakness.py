import random
import pandas as pd
from typing import Dict
from core.config import settings
from typing import Dict, Optional


def determine_strength_weakness(df: pd.DataFrame, company_name: str,
                                strength_metric: Optional[str] = None,
                                weakness_metric: Optional[str] = None) -> Dict:
    """
    회사의 강점과 약점을 결정하는 함수.

    Args:
        data (pd.DataFrame): 전체 회사 데이터를 담고 있는 데이터프레임.
        company_name (str): 분석 대상 회사명.

    Returns:
        Dict[str, str]: 회사의 강점과 약점 지표를 각각 나타내는 딕셔너리.
    """

    if strength_metric and weakness_metric:
        return {
            'strength': strength_metric,
            'weakness': weakness_metric
        }

    # 강점과 약점을 결정
    strength = random.choice(settings.INDICATORS)
    remaining_indicators = [
        indicator for indicator in settings.INDICATORS if indicator != strength]
    weakness = random.choice(remaining_indicators)

    return {'strength': strength, 'weakness': weakness}
