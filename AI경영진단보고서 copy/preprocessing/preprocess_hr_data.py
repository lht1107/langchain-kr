import pandas as pd
from typing import Dict, Any
import logging
from utils.logger import get_logger

logger = get_logger(__name__)


class HumanResourcesMetricsCalculator:
    """
    인적 자원 지표 계산을 위한 클래스
    """

    REQUIRED_COLUMNS = ['기업명', '날짜', '입사자 수', '퇴사자 수',
                        '임직원 수', '월평균급여액', '업종', '월매출창출액', '인적관리']

    def __init__(self, df: pd.DataFrame, company_name: str, access_time: pd.Timestamp):
        self.df = self._validate_input_data(df)
        self.company_name = company_name
        self.access_time = pd.to_datetime(access_time)
        self.latest_date = self.access_time  # - pd.DateOffset(months=1)
        self.df['date'] = pd.to_datetime(self.df['날짜'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df_company = self.df[self.df['기업명'] == self.company_name].copy()
        if self.df_company.empty:
            raise ValueError(f"회사 데이터가 없음: {self.company_name}")

    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 검증"""
        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        return df[self.REQUIRED_COLUMNS].copy()

    def calculate_monthly_employee_metrics(self) -> Dict[str, Any]:
        """최근 1년간 월별 임직원 수 및 전년 동월 대비 증감율"""
        logger.info("[HR] Calculating monthly employee metrics")
        past_12_months = pd.date_range(
            end=self.latest_date, periods=12, freq='MS'
        )
        metrics = []
        for date in past_12_months:
            current_month = self.df_company[self.df_company['date'] == date]
            prev_month = self.df_company[self.df_company['date']
                                         == date - pd.DateOffset(years=1)]

            current_employees = int(
                current_month['임직원 수'].iloc[0]) if not current_month.empty else 0
            prev_employees = int(
                prev_month['임직원 수'].iloc[0]) if not prev_month.empty else 0
            growth_rate = round(((current_employees - prev_employees) /
                                prev_employees) * 100, 2) if prev_employees else None

            metrics.append({
                '날짜': date.strftime('%Y-%m'),
                '임직원 수': current_employees,
                '전년동월대비 증감율': growth_rate
            })
        return metrics

    def calculate_annual_employee_metrics(self) -> pd.DataFrame:
        """최근 3년간 연말 입사자, 퇴사자, 임직원 수 데이터를 정리"""
        logger.info("[HR] Calculating annual employee metrics")

        # 최근 3년을 오름차순으로 정렬
        recent_years = sorted([self.latest_date.year - i for i in range(1, 4)])

        # 데이터 수집
        metrics = {
            '지표': ['연간 입사자 수', '연간 퇴사자 수', '연말 임직원 수'],
        }

        for year in recent_years:
            year_data = self.df_company[self.df_company['year'] == year]
            total_hires = year_data['입사자 수'].sum()
            total_resignations = year_data['퇴사자 수'].sum()
            employees_at_year_end = year_data.loc[
                year_data['month'] == 12, '임직원 수'
            ].iloc[-1] if not year_data.empty else 0

            metrics[f"{year}년"] = [
                int(total_hires),
                int(total_resignations),
                int(employees_at_year_end)
            ]

        # DataFrame 생성
        df = pd.DataFrame(metrics)
        return df

    def extract_hr_dictionaries(self) -> Dict[str, Any]:
        """근속기간, 나이분포도, 연령별 평균 근속기간, 연령별 평균연봉"""
        logger.info("[HR] Extracting HR-related dictionaries")
        # Assuming consistent HR data across rows
        hr_data = self.df_company['인적관리'].iloc[0]
        return {
            '근속기간': hr_data['근속기간'],
            '나이분포도': hr_data['나이 분포도'],
            '연령별 평균 근속기간': hr_data['연령별 평균 근속기간'],
            '연령별 평균 연봉': hr_data['연령별 평균 연봉']
        }

    def calculate_monthly_salary_and_revenue(self) -> Dict[str, Any]:
        """최근 1년간 월평균 급여와 업종평균 급여, 매출액 데이터"""
        logger.info("[HR] Calculating monthly salary and revenue metrics")
        past_12_months = pd.date_range(
            end=self.latest_date, periods=12, freq='MS'
        )

        metrics = []
        for date in past_12_months:
            current_month = self.df_company[self.df_company['date'] == date]
            industry_data = self.df[(self.df['업종'] == self.df_company['업종'].iloc[0]) & (
                self.df['date'] == date)]

            avg_salary = float(
                current_month['월평균급여액'].iloc[0]) if not current_month.empty else 0.0
            industry_avg_salary = industry_data['월평균급여액'].mean(
            ) if not industry_data.empty else 0.0

            avg_revenue = float(
                current_month['월매출창출액'].iloc[0]) if not current_month.empty else 0.0
            industry_avg_revenue = industry_data['월매출창출액'].mean(
            ) if not industry_data.empty else 0.0

            metrics.append({
                '날짜': date.strftime('%Y-%m'),
                '월평균급여액': avg_salary,
                '업종평균 급여': industry_avg_salary,
                '월매출창출액': avg_revenue,
                '업종평균 매출액': industry_avg_revenue
            })
        return metrics


def preprocess_hr_data(df_company_info: pd.DataFrame, company_name: str, access_time: str) -> Dict[str, Any]:
    """HR 관련 데이터 전처리 함수"""
    logger.info(
        f"[Preprocess] Starting HR data preprocessing for {company_name}")
    calculator = HumanResourcesMetricsCalculator(
        df_company_info, company_name, access_time)

    try:
        # 메트릭 계산
        monthly_employee_metrics = calculator.calculate_monthly_employee_metrics()
        annual_employee_metrics_df = calculator.calculate_annual_employee_metrics()
        hr_dicts = calculator.extract_hr_dictionaries()
        monthly_salary_and_revenue_metrics = calculator.calculate_monthly_salary_and_revenue()

        logger.info(f"{hr_dicts['연령별 평균 근속기간']}")
        # DataFrame을 직렬화 가능한 dict로 변환 (지표명 포함)
        result = {
            'latest_year_month': calculator.latest_date.strftime('%Y-%m'),
            'annual_employee_metrics': [
                {
                    '지표': row_name,
                    **annual_employee_metrics_df.loc[row_name].to_dict()
                } for row_name in annual_employee_metrics_df.index
            ],
            'average_tenure_by_age': [
                {
                    '지표': '평균근속기간',
                    '20대': hr_dicts['연령별 평균 근속기간']['20대'],
                    '30대': hr_dicts['연령별 평균 근속기간']['30대'],
                    '40대': hr_dicts['연령별 평균 근속기간']['40대'],
                    '50대 이상': hr_dicts['연령별 평균 근속기간']['50대 이상']
                }
            ],
            'average_salary_by_age': [
                {
                    '지표': '평균연봉',
                    '20대': hr_dicts['연령별 평균 연봉']['20대'],
                    '30대': hr_dicts['연령별 평균 연봉']['30대'],
                    '40대': hr_dicts['연령별 평균 연봉']['40대'],
                    '50대 이상': hr_dicts['연령별 평균 연봉']['50대 이상']
                }
            ],
            'tenure_distribution': [
                {
                    '지표': '비율',
                    **hr_dicts['근속기간']
                }
            ],
            'age_distribution': [
                {
                    '지표': '비율',
                    **hr_dicts['나이분포도']
                }
            ],
            'monthly_employee_metrics': [
                {
                    '지표': '임직원 수',
                    **{entry['날짜']: entry['임직원 수'] for entry in monthly_employee_metrics}
                },
                {
                    '지표': '전년동월대비 증감율',
                    **{entry['날짜']: entry['전년동월대비 증감율'] for entry in monthly_employee_metrics}
                }
            ],
            'monthly_salary_and_revenue_metrics': [
                {
                    '지표': '월평균급여액',
                    **{entry['날짜']: entry['월평균급여액'] for entry in monthly_salary_and_revenue_metrics}
                },
                {
                    '지표': '업종평균 급여',
                    **{entry['날짜']: entry['업종평균 급여'] for entry in monthly_salary_and_revenue_metrics}
                },
                {
                    '지표': '월매출창출액',
                    **{entry['날짜']: entry['월매출창출액'] for entry in monthly_salary_and_revenue_metrics}
                },
                {
                    '지표': '업종평균 매출액',
                    **{entry['날짜']: entry['업종평균 매출액'] for entry in monthly_salary_and_revenue_metrics}
                }
            ]
        }

        logger.info(
            f"[Preprocess] HR data preprocessing completed for {company_name}")
        return result

    except Exception as e:
        logger.error(f"[Error] Failed to preprocess HR data: {e}")
        raise
