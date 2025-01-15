import pandas as pd
from typing import Dict, Any


def preprocess_cashflow_data(df: pd.DataFrame, target_company_name: str, access_time: pd.Timestamp) -> Dict[str, Any]:
    """
    현금흐름 데이터를 전처리하여 JSON 형식으로 반환
    Args:
        df: 원본 DataFrame
        target_company_name: 분석 대상 기업명
        access_time: 최신 날짜 (pd.Timestamp)
    """
    # DataFrame 복사
    df = df.copy()

    # 접근 시간 처리
    # 이미 pd.Timestamp 이므로 바로 사용
    latest_date = access_time  # - pd.DateOffset(months=1)
    latest_year_month = latest_date.strftime('%Y-%m')
    latest_year = latest_date.year

    # 타겟 회사 데이터 필터링
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['년'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    company_data = df[df['기업명'] == target_company_name].copy()
    if company_data.empty:
        raise ValueError(f"대상 회사 데이터가 없습니다: {target_company_name}")

    # 최근 3년과 현재 연도(E) 데이터 준비
    recent_years = [latest_year - i for i in range(3, 0, -1)]

    operating_activities = {'지표': ['영업활동현금흐름', '운전자금']}
    investing_activities = {'지표': ['투자활동현금흐름', '감가상각비']}
    financing_activities = {'지표': ['재무활동현금흐름', '차입금의존도']}

    # 연간 데이터 처리
    for year in recent_years:
        year_label = f"{year}년"
        year_data = company_data[(company_data['년'] == year) & (
            company_data['월'] == 12)]
        if not year_data.empty:
            row = year_data.iloc[-1]
            operating_activities[year_label] = [
                row['현금흐름']['영업활동현금흐름'],
                row['현금흐름']['운전자금']
            ]
            investing_activities[year_label] = [
                row['현금흐름']['투자활동현금흐름'],
                row['현금흐름']['감가상각비']
            ]
            financing_activities[year_label] = [
                row['현금흐름']['재무활동현금흐름'],
                row['현금흐름']['차입금의존도']
            ]

    # 현재 연도(E) 데이터 처리
    current_year_label = f"{latest_year}년(E)"
    current_year_data = company_data[
        (company_data['년'] == latest_year) & (
            company_data['월'] == latest_date.month)
    ]
    if not current_year_data.empty:
        row = current_year_data.iloc[-1]
        operating_activities[current_year_label] = [
            row['현금흐름']['영업활동현금흐름'],
            row['현금흐름']['운전자금']
        ]
        investing_activities[current_year_label] = [
            row['현금흐름']['투자활동현금흐름'],
            row['현금흐름']['감가상각비']
        ]
        financing_activities[current_year_label] = [
            row['현금흐름']['재무활동현금흐름'],
            row['현금흐름']['차입금의존도']
        ]

    # 월별 데이터 처리 (과거 12개월)
    past_12_months = pd.date_range(end=latest_date, periods=12, freq='MS')
    monthly_data = {'지표': ['영업활동현금흐름', '매출액', '현금창출율']}

    for date in reversed(past_12_months):  # 최근 날짜부터
        month_label = date.strftime('%Y-%m')
        month_data = company_data[company_data['날짜'] == date]
        if not month_data.empty:
            row = month_data.iloc[-1]
            # 매출액이 0이 아닌 경우에만 현금창출율 계산
            revenue = row['매출액'] if row['매출액'] else None
            cflow = row['현금흐름']['영업활동현금흐름']
            cash_gen_rate = (cflow / revenue * 100) if revenue else None
            monthly_data[month_label] = [cflow, revenue, cash_gen_rate]

    return {
        'latest_year_month': latest_year_month,
        'operating_activities': operating_activities,
        'investing_activities': investing_activities,
        'financing_activities': financing_activities,
        'monthly_cashflow': monthly_data
    }
