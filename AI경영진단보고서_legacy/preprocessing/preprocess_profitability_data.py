import pandas as pd
from typing import Dict, Any
import logging
from utils.logger import get_logger
from utils.validation import validate_input_data, DataValidationError

logger = get_logger(__name__)


class ProfitabilityMetricsCalculator:
    """수익성 지표 계산을 위한 메인 클래스"""

    REQUIRED_COLUMNS = ['기업명', '날짜', '업종', '영업이익', '당기순이익', '매출액']
    COMPANY_COL, DATE_COL, OPERATING_PROFIT_COL, NET_PROFIT_COL, REVENUE_COL, INDUSTRY_COL = '기업명', '날짜', '영업이익', '당기순이익', '매출액', '업종'

    def __init__(self, df_company_info: pd.DataFrame, target_company_name: str, access_time: str, n_years: int = 3):
        logger.info(
            f"[Initialize] Starting ProfitabilityMetricsCalculator for {target_company_name}")
        self.access_datetime = pd.to_datetime(access_time)
        self.df = validate_input_data(
            df_company_info, self.REQUIRED_COLUMNS, self.COMPANY_COL)
        self.target_company_name = target_company_name
        self.n_years = n_years

        self.profitability_data = {
            'annual_profit': {},
            'annual_margins': {},
            'monthly_profit': {},
            'monthly_margins': {}
        }
        logger.debug(
            f"[Initialize] Initialization completed for {target_company_name}")

    def _prepare_data(self):
        """데이터 전처리 및 필요한 필드 추가"""
        logger.info(
            f"[Data] Starting data preparation for {self.target_company_name}")
        self.df['date'] = pd.to_datetime(self.df[self.DATE_COL])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df = self.df.fillna(
            {self.OPERATING_PROFIT_COL: 0, self.NET_PROFIT_COL: 0, self.REVENUE_COL: 0})

        # 타겟 회사 및 업종 데이터 설정
        self.df_company = self.df[self.df[self.COMPANY_COL]
                                  == self.target_company_name].reset_index(drop=True)
        # logger.info(f"Company data head: \n{self.df_company.head()}")

        if self.df_company.empty:
            error_msg = f"회사 데이터가 없음: {self.target_company_name}"
            logger.error(f"[Error] {error_msg}")
            raise DataValidationError(error_msg)
        self.target_industry = self.df_company[self.INDUSTRY_COL].iloc[0]

        # 업종 필터링
        self.df_industry = self.df[self.df[self.INDUSTRY_COL]
                                   == self.target_industry].reset_index(drop=True)

        # 가장 최근 날짜 설정
        self.latest_date = self.access_datetime - pd.DateOffset(months=1)
        self.latest_month = self.latest_date.month
        logger.info(
            f"[Data] Data preparation completed for {self.target_company_name}")

    def _calculate_margin_rate(self, profit: float, revenue: float, threshold: float = 0.0001) -> float:
        """이익률 계산"""
        logger.debug(
            f"[Calculate] Revenue below threshold (revenue: {revenue})")
        if abs(revenue) < threshold:
            logger.warning(f"매출액이 임계값보다 작음 (revenue: {revenue})")
            return 0
        return round((profit / revenue * 100), 2)

    def _process_annual_data(self):
        """과거 연도 및 연말 예상 연도 데이터 계산 및 저장"""
        logger.info(
            f"[Annual] Processing annual data for {self.target_company_name}")
        for year_offset in range(1, self.n_years + 1):
            target_year = self.latest_date.year - year_offset
            self._calculate_annual_metrics(target_year, estimate=False)
        self._calculate_annual_metrics(
            self.latest_date.year, estimate=True)  # 연말 예상 데이터
        logger.info(
            f"[Annual] Annual data processing completed for {self.target_company_name}")

    def _calculate_annual_metrics(self, year: int, estimate: bool = False):
        """연간 또는 연말 예상 데이터 계산"""
        if estimate:
            current_data = self.df_company[(self.df_company['year'] == year) & (
                self.df_company['month'] <= self.latest_month)]
            annualized_metrics = self._calculate_annualized_metrics(
                current_data)
            industry_metrics = self._calculate_annualized_metrics(
                self.df_industry[self.df_industry['year'] == year])
        else:
            current_data = self.df_company[self.df_company['year'] == year]
            prev_data = self.df_company[self.df_company['year'] == year - 1]
            industry_data = self.df_industry[self.df_industry['year'] == year]
            annualized_metrics = self._calculate_metrics(
                current_data, prev_data)
            industry_metrics = self._calculate_metrics(industry_data)

        key = f"{year}년(E)" if estimate else f"{year}년"
        self.profitability_data['annual_profit'][key] = {
            '영업이익': annualized_metrics['operating_profit'],
            '당기순이익': annualized_metrics['net_profit']
        }
        self.profitability_data['annual_margins'][key] = {
            '영업이익률': annualized_metrics['operating_profit_margin'],
            '당기순이익률': annualized_metrics['net_profit_margin'],
            '업종평균 영업이익률': industry_metrics['operating_profit_margin'],
            '업종평균 당기순이익률': industry_metrics['net_profit_margin']
        }

    def _calculate_annualized_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """연환산 지표 계산"""
        monthly_op = data[self.OPERATING_PROFIT_COL].sum()
        monthly_np = data[self.NET_PROFIT_COL].sum()
        monthly_rev = data[self.REVENUE_COL].sum()

        # 연환산 계산
        annualized_op = (monthly_op / self.latest_month) * 12
        annualized_np = (monthly_np / self.latest_month) * 12
        annualized_rev = (monthly_rev / self.latest_month) * 12

        return {
            'operating_profit': int(round(annualized_op)),
            'net_profit': int(round(annualized_np)),
            'operating_profit_margin': self._calculate_margin_rate(annualized_op, annualized_rev),
            'net_profit_margin': self._calculate_margin_rate(annualized_np, annualized_rev)
        }

    def _calculate_metrics(self, current_data: pd.DataFrame, prev_data: pd.DataFrame = None) -> Dict[str, float]:
        """연간 수익성 지표 계산"""
        current_op = current_data[self.OPERATING_PROFIT_COL].sum()
        current_np = current_data[self.NET_PROFIT_COL].sum()
        current_rev = current_data[self.REVENUE_COL].sum()

        return {
            'operating_profit': int(round(current_op)),
            'net_profit': int(round(current_np)),
            'operating_profit_margin': self._calculate_margin_rate(current_op, current_rev),
            'net_profit_margin': self._calculate_margin_rate(current_np, current_rev)
        }

    def _process_monthly_data(self):
        """최근 12개월 월별 데이터 계산 및 저장"""
        logger.info(
            f"[Monthly] Processing monthly data for {self.target_company_name}")
        past_12_months = pd.date_range(
            end=self.latest_date + pd.DateOffset(months=1), periods=12, freq='ME')
        cumulative_data = {'current': {
            'operating_profit': 0, 'net_profit': 0, 'revenue': 0}}

        for date in past_12_months:
            month_str = date.strftime('%Y-%m')
            current_month = self.df_company[self.df_company['date'].dt.strftime(
                '%Y-%m') == month_str]
            prev_month = self.df_company[self.df_company['date'].dt.strftime(
                '%Y-%m') == (date - pd.DateOffset(years=1)).strftime('%Y-%m')]

            current_op = int(current_month[self.OPERATING_PROFIT_COL].sum())
            current_np = int(current_month[self.NET_PROFIT_COL].sum())
            current_rev = int(current_month[self.REVENUE_COL].sum())
            prev_op = int(prev_month[self.OPERATING_PROFIT_COL].sum(
            )) if not prev_month.empty else 0
            prev_np = int(prev_month[self.NET_PROFIT_COL].sum(
            )) if not prev_month.empty else 0

            cumulative_data['current']['operating_profit'] += current_op
            cumulative_data['current']['net_profit'] += current_np
            cumulative_data['current']['revenue'] += current_rev

            self.profitability_data['monthly_profit'][month_str] = {
                '당월영업이익': current_op,
                '전년동월영업이익': prev_op,
                '당월당기순이익': current_np,
                '전년동월당기순이익': prev_np
            }
            self.profitability_data['monthly_margins'][month_str] = {
                '영업이익률': self._calculate_margin_rate(current_op, current_rev),
                '누적영업이익률': self._calculate_margin_rate(cumulative_data['current']['operating_profit'], cumulative_data['current']['revenue']),
                '당기순이익률': self._calculate_margin_rate(current_np, current_rev),
                '누적순이익률': self._calculate_margin_rate(cumulative_data['current']['net_profit'], cumulative_data['current']['revenue'])
            }
        logger.debug(f"[Monthly] Processed data for {month_str}")
        logger.info(
            f"[Monthly] Monthly data processing completed for {self.target_company_name}")

    def calculate_profitability_metrics(self) -> Dict[str, pd.DataFrame]:
        """수익성 지표 계산 실행"""
        logger.info(
            f"[Metrics] Starting profitability metrics calculation for {self.target_company_name}")
        self._prepare_data()
        self._process_annual_data()
        self._process_monthly_data()

        # 데이터 구조 생성
        annual_profit_data = {
            '지표': ['영업이익', '당기순이익']
        }
        annual_margins_data = {
            '지표': ['영업이익률', '당기순이익률', '업종평균 영업이익률', '업종평균 당기순이익률']
        }
        monthly_profit_data = {
            '지표': ['당월영업이익', '전년동월영업이익', '당월당기순이익', '전년동월당기순이익']
        }
        monthly_margins_data = {
            '지표': ['영업이익률', '누적영업이익률', '당기순이익률', '누적순이익률']
        }

        # # DataFrames 생성 및 날짜 순 정렬
        # annual_profit_df = pd.DataFrame(
        #     self.profitability_data['annual_profit']).sort_index(axis=1)
        # annual_margins_df = pd.DataFrame(
        #     self.profitability_data['annual_margins']).sort_index(axis=1)
        # monthly_profit_df = pd.DataFrame(
        #     self.profitability_data['monthly_profit']).sort_index(axis=1)
        # monthly_margins_df = pd.DataFrame(
        #     self.profitability_data['monthly_margins']).sort_index(axis=1)

        # 연도별/월별 데이터 추가
        for year in sorted(self.profitability_data['annual_profit'].keys()):
            annual_profit_data[year] = [
                self.profitability_data['annual_profit'][year]['영업이익'],
                self.profitability_data['annual_profit'][year]['당기순이익']
            ]
            annual_margins_data[year] = [
                self.profitability_data['annual_margins'][year]['영업이익률'],
                self.profitability_data['annual_margins'][year]['당기순이익률'],
                self.profitability_data['annual_margins'][year]['업종평균 영업이익률'],
                self.profitability_data['annual_margins'][year]['업종평균 당기순이익률']
            ]

        for month in sorted(self.profitability_data['monthly_profit'].keys()):
            monthly_profit_data[month] = [
                self.profitability_data['monthly_profit'][month]['당월영업이익'],
                self.profitability_data['monthly_profit'][month]['전년동월영업이익'],
                self.profitability_data['monthly_profit'][month]['당월당기순이익'],
                self.profitability_data['monthly_profit'][month]['전년동월당기순이익']
            ]
            monthly_margins_data[month] = [
                self.profitability_data['monthly_margins'][month]['영업이익률'],
                self.profitability_data['monthly_margins'][month]['누적영업이익률'],
                self.profitability_data['monthly_margins'][month]['당기순이익률'],
                self.profitability_data['monthly_margins'][month]['누적순이익률']
            ]

        # DataFrame 생성
        annual_profit_df = pd.DataFrame(annual_profit_data).set_index('지표')
        annual_margins_df = pd.DataFrame(annual_margins_data).set_index('지표')
        monthly_profit_df = pd.DataFrame(monthly_profit_data).set_index('지표')
        monthly_margins_df = pd.DataFrame(monthly_margins_data).set_index('지표')

        logger.info(
            f"[Metrics] Profitability metrics calculation completed for {self.target_company_name}")

        return {
            'latest_year_month': self.latest_date.strftime('%Y-%m'),
            'annual_profit': annual_profit_df,
            'annual_margins': annual_margins_df,
            'monthly_profit': monthly_profit_df,
            'monthly_margins': monthly_margins_df
        }


# def preprocess_profitability_data(df_company_info: pd.DataFrame, target_company_name: str, access_time: str) -> Dict[str, Any]:
#     """수익성 지표 데이터를 전처리하여 JSON 형식으로 변환"""
#     logger.info(f"[Profitability] Starting profitability data preprocessing for {target_company_name}")
#     calculator = ProfitabilityMetricsCalculator(df_company_info, target_company_name, access_time)
#     result = calculator.calculate_profitability_metrics()
#     # 추가 ⭐
#     logger.info(f"[Profitability] Profitability data preprocessing completed for {target_company_name}")
#     return result


def preprocess_profitability_data(df_company_info: pd.DataFrame, target_company_name: str, access_time: str) -> Dict[str, Any]:
    """수익성 지표 데이터를 전처리하여 JSON 직렬화 가능한 형태로 변환"""
    logger.info(
        f"[Profitability] Starting profitability data preprocessing for {target_company_name}")

    # ProfitabilityMetricsCalculator를 통한 데이터 계산
    calculator = ProfitabilityMetricsCalculator(
        df_company_info, target_company_name, access_time)
    metrics = calculator.calculate_profitability_metrics()

    # DataFrame을 직렬화 가능한 dict로 변환 (지표명 포함)
    result = {
        'latest_year_month': metrics['latest_year_month'],
        'annual_profit': [
            {
                '지표': row_name,
                **metrics['annual_profit'].loc[row_name].to_dict()
            } for row_name in metrics['annual_profit'].index
        ],
        'annual_margins': [
            {
                '지표': row_name,
                **metrics['annual_margins'].loc[row_name].to_dict()
            } for row_name in metrics['annual_margins'].index
        ],
        'monthly_profit': [
            {
                '지표': row_name,
                **metrics['monthly_profit'].loc[row_name].to_dict()
            } for row_name in metrics['monthly_profit'].index
        ],
        'monthly_margins': [
            {
                '지표': row_name,
                **metrics['monthly_margins'].loc[row_name].to_dict()
            } for row_name in metrics['monthly_margins'].index
        ]
    }

    logger.info(
        f"[Profitability] Profitability data preprocessing completed for {target_company_name}")
    return result
