import random
import pandas as pd
from typing import Dict
from core.config import settings
from typing import Dict, Optional
from fastapi import HTTPException


def determine_strength_weakness(
    strength_metric: Optional[str] = None,
    weakness_metric: Optional[str] = None,
    analysis_type: Optional[str] = None
) -> Dict:
    """강점/약점 지표 결정 함수"""
    result = {}

    if analysis_type == "insight":
        # insight 분석 시 두 metric 모두 필수
        if not strength_metric or not weakness_metric:
            raise HTTPException(
                status_code=400,
                detail="Both strength_metric and weakness_metric are required for insight analysis"
            )
        result['strength'] = strength_metric
        result['weakness'] = weakness_metric
    elif analysis_type == "strength":
        result['strength'] = strength_metric
    elif analysis_type == "weakness":
        result['weakness'] = weakness_metric

    return result

# def determine_strength_weakness(df: pd.DataFrame, company_name: str,
#                                 strength_metric: Optional[str] = None,
#                                 weakness_metric: Optional[str] = None) -> Dict:
#     """
#     회사의 강점과 약점을 결정하는 함수.

#     Args:
#         data (pd.DataFrame): 전체 회사 데이터를 담고 있는 데이터프레임.
#         company_name (str): 분석 대상 회사명.

#     Returns:
#         Dict[str, str]: 회사의 강점과 약점 지표를 각각 나타내는 딕셔너리.
#     """

#     if strength_metric and weakness_metric:
#         return {
#             'strength': strength_metric,
#             'weakness': weakness_metric
#         }

#     # 강점과 약점을 결정
#     strength = random.choice(settings.METRICS)
#     remaining_metrics = [
#         metric for metric in settings.METRICS if metric != strength]
#     weakness = random.choice(remaining_metrics)

#     return {'strength': strength, 'weakness': weakness}
