import pandas as pd
import numpy as np

def extract_latest_month_and_previous_year(df):
    """
    DataFrame에서 최신 월과 전년 동기 월을 동적으로 추출하는 함수.
    """
    # 날짜 컬럼에서 최신 날짜와 그 전년도 같은 월을 추출
    latest_month = df["날짜"].max()  # 최신 날짜
    year, month = latest_month.split("-")
    previous_year_month = f"{int(year) - 1}-{month}"  # 전년도 같은 월

    return latest_month, previous_year_month

#%%
def calculate_three_year_average(values, months):
    """
    3년 평균을 계산하는 함수.
    """
    last_three_year_values = [
        values.get(month) for month in months if month in values
    ]
    return round(np.mean(last_three_year_values), 2)

#%%
def preprocess_financial_data(df):
    """
    DataFrame을 받아서 각 재무 비율을 분석하기 위한 형태로 변환하는 함수.
    """
    processed_data = {}

    # 최신 월과 전년 동기 월 추출
    latest_year_month, previous_year_month = (
        extract_latest_month_and_previous_year(df)
    )

    for column in df.columns[1:]:  # 첫 번째 column인 '날짜'는 제외
        metric_name = column
        values = dict(zip(df["날짜"], df[column]))  # 날짜를 키로 하여 값 매핑
        processed_data[metric_name] = {}

        # 최신 데이터
        latest_data = values[latest_year_month]

        # 전년 동기 데이터
        prev_year_data = values[previous_year_month]
        change_vs_last_year = (
            ((latest_data - prev_year_data) / prev_year_data) * 100
            if prev_year_data != 0
            else 0
        )

        # 3년 평균 계산
        last_three_year_months = [
            f'{int(latest_year_month.split("-")[0]) - i}-{latest_year_month.split("-")[1]}'
            for i in range(1, 4)
        ]
        three_year_average = calculate_three_year_average(
            values, last_three_year_months
        )
        change_vs_three_year_average = (
            ((latest_data - three_year_average) / three_year_average) * 100
            if three_year_average != 0
            else 0
        )

        # 최근 12개월 데이터 추출
        recent_12_months = {
            month: round(values[month], 2)
            for month in values
            if month
            >= f"{int(latest_year_month.split('-')[0]) - 1}-{latest_year_month.split('-')[1]}"
        }

        # JSON 형식으로 변환 및 열 이름 변경 ('전년 동기 데이터'로 열 이름 변경)
        processed_data[metric_name] = {
            "최신 데이터": round(latest_data, 2),
            "전년 동기 데이터": round(prev_year_data, 2),
            "전년 동기 대비 변화율 (%)": round(change_vs_last_year, 2),
            "직전년도 3년 평균": round(three_year_average, 2),
            "3년 평균 대비 변화율 (%)": round(change_vs_three_year_average, 2),
            "최근 12개월 데이터": recent_12_months,
        }

    return {
        "latest_year_month": latest_year_month,
        "previous_year_month": previous_year_month,  # 동적으로 설정된 전년 동기 데이터
        "processed_data": processed_data,
    }
    
#%%
def assign_chain_data(processed_financial_data):
    return {
        "growth_data": {
            "매출증가율": processed_financial_data["processed_data"]["매출증가율"],
            "총자산증가율": processed_financial_data["processed_data"]["총자산증가율"],
            "순이익증가율": processed_financial_data["processed_data"]["순이익증가율"],
        },
        "profitability_data": {
            "영업이익률": processed_financial_data["processed_data"]["영업이익률"],
            "순이익률": processed_financial_data["processed_data"]["순이익률"],
            "ROA": processed_financial_data["processed_data"]["ROA"],
        },
        "stability_data": {
            "부채비율": processed_financial_data["processed_data"]["부채비율"],
            "이자보상배율": processed_financial_data["processed_data"]["이자보상배율"],
            "자기자본비율": processed_financial_data["processed_data"]["자기자본비율"],
        },
        "productivity_data": {
            "노동생산성": processed_financial_data["processed_data"]["노동생산성"],
            "자산회전율": processed_financial_data["processed_data"]["자산회전율"],
            "재고자산회전율": processed_financial_data["processed_data"]["재고자산회전율"],
        },
        "liquidity_data": {
            "유동비율": processed_financial_data["processed_data"]["유동비율"],
            "현금흐름비율": processed_financial_data["processed_data"]["현금흐름비율"],
            "당좌비율": processed_financial_data["processed_data"]["당좌비율"],
        },
        "latest_year_month": processed_financial_data["latest_year_month"],
        "previous_year_month": processed_financial_data["previous_year_month"]
    }
# %%
