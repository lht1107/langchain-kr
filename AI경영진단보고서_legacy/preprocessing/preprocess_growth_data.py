import pandas as pd
from typing import Dict, Any
import logging
from utils.logger import get_logger
from utils.validation import validate_input_data, DataValidationError

logger = get_logger(__name__)


class GrowthMetricsCalculator:
    """성장성 지표 계산을 위한 메인 클래스"""

    REQUIRED_COLUMNS = ['기업명', '날짜', '업종', '총자산', '매출액']
    COMPANY_COL, DATE_COL, ASSET_COL, REVENUE_COL, INDUSTRY_COL = '기업명', '날짜', '총자산', '매출액', '업종'

    def __init__(self, df_company_info: pd.DataFrame, target_company_name: str, access_time: str, n_years: int = 3):
        logger.info(
            f"[Initialize] Starting GrowthMetricsCalculator for {target_company_name}")
        self.access_datetime = access_time
        self.df = validate_input_data(
            df_company_info, self.REQUIRED_COLUMNS, self.COMPANY_COL)
        self.target_company_name = target_company_name
        self.n_years = n_years
        self.growth_data = {
            'annual_revenue': {},
            'annual_assets': {},
            'monthly_revenue': {},
            'monthly_growth': {}
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
        self.df = self.df.fillna({self.ASSET_COL: 0, self.REVENUE_COL: 0})

        # 타겟 회사의 업종 정보 가져오기 및 설정
        self.df_company = self.df[self.df[self.COMPANY_COL]
                                  == self.target_company_name].reset_index(drop=True)
        if self.df_company.empty:
            error_msg = f"회사 데이터가 없음: {self.target_company_name}"
            logger.error(f"[Error] {error_msg}")
            raise DataValidationError(error_msg)

        # 타겟 업종 설정
        self.target_industry = self.df_company[self.INDUSTRY_COL].iloc[0]
        self.df_industry = self.df[self.df[self.INDUSTRY_COL]
                                   == self.target_industry].reset_index(drop=True)
        # logger.info(f"Company data head: \n{self.df_company.head()}")

        self.latest_date = self.access_datetime - pd.DateOffset(months=1)
        logger.info(
            f"[Data] Data preparation completed for {self.target_company_name}")

    def _calculate_growth_rate(self, current: float, previous: float, threshold: float = 0.0001) -> float:
        """성장률 계산"""
        if abs(previous) < threshold:
            logger.debug(
                f"[Calculate] Previous value below threshold (previous: {previous})")
            return 0
        return round((current - previous) / previous * 100, 2)

    def _process_annual_data(self):
        """과거 연도 및 연말 예상 연도 데이터 계산 및 저장"""
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
            # 2024년 9월까지 누적 매출액을 기반으로 연환산 수행
            current_data = self.df_company[(self.df_company['year'] == year) & (
                self.df_company['month'] <= self.latest_date.month)]
            current_revenue = int(
                (current_data[self.REVENUE_COL].sum() / self.latest_date.month) * 12)
        else:
            # 기존 연도는 그대로 누적합 사용
            current_data = self.df_company[self.df_company['year'] == year]
            current_revenue = int(current_data[self.REVENUE_COL].sum())

        prev_data = self.df_company[self.df_company['year'] == year - 1]
        prev_revenue = int(
            prev_data[self.REVENUE_COL].sum() if not prev_data.empty else 0)

        # 자산 데이터 처리 (연환산 불필요)
        current_assets = int(
            current_data[self.ASSET_COL].iloc[-1] if not current_data.empty else 0)
        prev_assets = int(
            prev_data[self.ASSET_COL].iloc[-1] if not prev_data.empty else 0)

        # 산업 평균 계산
        current_industry_data = self.df_industry[self.df_industry['year'] == year]
        prev_industry_data = self.df_industry[self.df_industry['year'] == year - 1]

        industry_revenue = int(current_industry_data[self.REVENUE_COL].sum())
        prev_industry_revenue = int(prev_industry_data[self.REVENUE_COL].sum(
        ) if not prev_industry_data.empty else 0)
        industry_assets = int(current_industry_data[self.ASSET_COL].mean(
        ) if not current_industry_data.empty else 0)
        prev_industry_assets = int(
            prev_industry_data[self.ASSET_COL].mean() if not prev_industry_data.empty else 0)

        key = f"{year}년(E)" if estimate else f"{year}년"
        self.growth_data['annual_revenue'][key] = {
            '매출액': current_revenue,
            '매출액증가율': self._calculate_growth_rate(current_revenue, prev_revenue),
            '업종평균 매출액증가율': self._calculate_growth_rate(industry_revenue, prev_industry_revenue),
        }
        self.growth_data['annual_assets'][key] = {
            '총자산': current_assets,
            '총자산증가율': self._calculate_growth_rate(current_assets, prev_assets),
            '업종평균 총자산증가율': self._calculate_growth_rate(industry_assets, prev_industry_assets)
        }

    def _process_monthly_data(self):
        """최근 12개월 월별 데이터 계산 및 저장"""
        logger.info(
            f"[Monthly] Processing monthly data for {self.target_company_name}")
        past_12_months = pd.date_range(
            end=self.latest_date + pd.DateOffset(months=1), periods=12, freq='ME')
        cumulative_data = {'current': {'revenue': 0},
                           'previous': {'revenue': 0}}

        for idx, date in enumerate(past_12_months):
            month_str = date.strftime('%Y-%m')
            current_month = self.df_company[self.df_company['date'].dt.strftime(
                '%Y-%m') == month_str]
            prev_month = self.df_company[self.df_company['date'].dt.strftime(
                '%Y-%m') == (date - pd.DateOffset(years=1)).strftime('%Y-%m')]

            current_revenue = int(current_month[self.REVENUE_COL].sum())
            prev_revenue = int(
                prev_month[self.REVENUE_COL].sum() if not prev_month.empty else 0)

            cumulative_data['current']['revenue'] += current_revenue
            cumulative_data['previous']['revenue'] += prev_revenue

            self.growth_data['monthly_revenue'][month_str] = {
                '당월매출액': current_revenue,
                '전년동월매출액': prev_revenue
            }
            self.growth_data['monthly_growth'][month_str] = {
                '매출액증가율': self._calculate_growth_rate(current_revenue, prev_revenue),
                '누적매출액증가율': self._calculate_growth_rate(
                    cumulative_data['current']['revenue'],
                    cumulative_data['previous']['revenue']
                )
            }
            logger.debug(f"[Monthly] Processed data for {month_str}")
        logger.info(
            f"[Monthly] Monthly data processing completed for {self.target_company_name}")

    def calculate_growth_metrics(self) -> Dict[str, pd.DataFrame]:
        """성장성 지표 계산 실행"""
        logger.info(
            f"[Metrics] Starting growth metrics calculation for {self.target_company_name}")
        self._prepare_data()
        self._process_annual_data()
        self._process_monthly_data()

        # # DataFrames 생성 및 날짜 순 정렬
        # annual_revenue_df = pd.DataFrame(
        #     self.growth_data['annual_revenue']).sort_index(axis=1)
        # annual_assets_df = pd.DataFrame(
        #     self.growth_data['annual_assets']).sort_index(axis=1)
        # monthly_revenue_df = pd.DataFrame(
        #     self.growth_data['monthly_revenue']).sort_index(axis=1)
        # monthly_growth_df = pd.DataFrame(
        #     self.growth_data['monthly_growth']).sort_index(axis=1)

        # 데이터 구조 생성
        annual_revenue_data = {
            '지표': ['매출액', '매출액증가율', '업종평균 매출액증가율']
        }
        annual_assets_data = {
            '지표': ['총자산', '총자산증가율', '업종평균 총자산증가율']
        }
        monthly_revenue_data = {
            '지표': ['당월매출액', '전년동월매출액']
        }
        monthly_growth_data = {
            '지표': ['매출액증가율', '누적매출액증가율']
        }

        # 연도별/월별 데이터 추가
        for year in sorted(self.growth_data['annual_revenue'].keys()):
            annual_revenue_data[year] = [
                self.growth_data['annual_revenue'][year]['매출액'],
                self.growth_data['annual_revenue'][year]['매출액증가율'],
                self.growth_data['annual_revenue'][year]['업종평균 매출액증가율']
            ]
            annual_assets_data[year] = [
                self.growth_data['annual_assets'][year]['총자산'],
                self.growth_data['annual_assets'][year]['총자산증가율'],
                self.growth_data['annual_assets'][year]['업종평균 총자산증가율']
            ]

        for month in sorted(self.growth_data['monthly_revenue'].keys()):
            monthly_revenue_data[month] = [
                self.growth_data['monthly_revenue'][month]['당월매출액'],
                self.growth_data['monthly_revenue'][month]['전년동월매출액']
            ]
            monthly_growth_data[month] = [
                self.growth_data['monthly_growth'][month]['매출액증가율'],
                self.growth_data['monthly_growth'][month]['누적매출액증가율']
            ]

        # DataFrame 생성
        annual_revenue_df = pd.DataFrame(annual_revenue_data).set_index('지표')
        annual_assets_df = pd.DataFrame(annual_assets_data).set_index('지표')
        monthly_revenue_df = pd.DataFrame(monthly_revenue_data).set_index('지표')
        monthly_growth_df = pd.DataFrame(monthly_growth_data).set_index('지표')

        return {
            'latest_year_month': self.latest_date.strftime('%Y-%m'),
            'annual_revenue': annual_revenue_df,
            'annual_assets': annual_assets_df,
            'monthly_revenue': monthly_revenue_df,
            'monthly_growth': monthly_growth_df
        }

# def preprocess_growth_data(df_company_info: pd.DataFrame, target_company_name: str, access_time: str) -> Dict[str, Any]:
#     """성장성 지표 데이터를 전처리하여 JSON 형식으로 변환"""
#     logger.info(f"[Growth] Starting growth data preprocessing for {target_company_name}")
#     calculator = GrowthMetricsCalculator(df_company_info, target_company_name, access_time)
#     result = calculator.calculate_growth_metrics()
#     logger.info(f"[Growth] Growth data preprocessing completed for {target_company_name}")
#     return result


def preprocess_growth_data(df_company_info: pd.DataFrame, target_company_name: str, access_time: str) -> Dict[str, Any]:
    """성장성 지표 데이터를 전처리하여 JSON 직렬화 가능한 형태로 변환"""
    logger.info(
        f"[Growth] Starting growth data preprocessing for {target_company_name}")

    calculator = GrowthMetricsCalculator(
        df_company_info, target_company_name, access_time)
    metrics = calculator.calculate_growth_metrics()

    # DataFrame을 직렬화 가능한 dict로 변환 (지표명 포함)
    result = {
        'latest_year_month': metrics['latest_year_month'],
        'annual_revenue': [
            {
                '지표': '매출액',
                **metrics['annual_revenue'].loc['매출액'].to_dict()
            },
            {
                '지표': '매출액증가율',
                **metrics['annual_revenue'].loc['매출액증가율'].to_dict()
            },
            {
                '지표': '업종평균 매출액증가율',
                **metrics['annual_revenue'].loc['업종평균 매출액증가율'].to_dict()
            }
        ],
        'annual_assets': [
            {
                '지표': row_name,
                **metrics['annual_assets'].loc[row_name].to_dict()
            } for row_name in metrics['annual_assets'].index
        ],
        'monthly_revenue': [
            {
                '지표': row_name,
                **metrics['monthly_revenue'].loc[row_name].to_dict()
            } for row_name in metrics['monthly_revenue'].index
        ],
        'monthly_growth': [
            {
                '지표': row_name,
                **metrics['monthly_growth'].loc[row_name].to_dict()
            } for row_name in metrics['monthly_growth'].index
        ]
    }

    logger.info(
        f"[Growth] Growth data preprocessing completed for {target_company_name}")
    return result
