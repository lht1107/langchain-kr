from typing import Dict, Any
import pandas as pd
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class FinancialStabilityMetricsCalculator:
    """재무 안정성 지표 계산을 위한 메인 클래스"""

    REQUIRED_COLUMNS = [
        '기업명', '날짜', '업종', '총자산', '매출액',
        '단기차입금', '장기차입금', '유동자산', '유동부채',
        '매출채권', '매입채무', '재고자산'
    ]

    def __init__(self, df: pd.DataFrame, company_name: str, access_time: datetime, n_years: int = 3):
        self.df = self._validate_input_data(df)
        self.company_name = company_name
        self.access_time = access_time
        self.n_years = n_years
        self.stability_data = {
            'annual_borrowings': {},
            'annual_liquidity': {},
            'monthly_borrowings': {},
            'monthly_liquidity': {}
        }
        self._prepare_data()

    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 검증"""
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        return df[self.REQUIRED_COLUMNS].copy()

    def _prepare_data(self):
        """데이터 전처리"""
        self.df['date'] = pd.to_datetime(self.df['날짜'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month

        self.df_company = self.df[self.df['기업명'] == self.company_name].copy()
        if self.df_company.empty:
            raise ValueError(f"회사 데이터 없음: {self.company_name}")

        self.target_industry = self.df_company['업종'].iloc[0]
        self.df_industry = self.df[self.df['업종']
                                   == self.target_industry].copy()
        self.latest_date = self.access_time - pd.DateOffset(months=1)

    def calculate_metrics(self) -> Dict[str, Any]:
        """재무 안정성 지표 계산"""
        try:
            # 연간 지표 계산
            for year_offset in range(1, self.n_years + 1):
                target_year = self.access_time.year - year_offset
                self._calculate_annual_borrowings(target_year, estimate=False)
                self._calculate_annual_liquidity(target_year, estimate=False)

            # 현재 연도 예상치 계산
            self._calculate_annual_borrowings(
                self.access_time.year, estimate=True)
            self._calculate_annual_liquidity(
                self.access_time.year, estimate=True)

            # 월별 지표 계산
            self._process_monthly_borrowings()
            self._process_monthly_liquidity()

            return {
                'latest_year_month': self.latest_date.strftime('%Y-%m'),
                'annual_borrowings': pd.DataFrame(self.stability_data['annual_borrowings']).sort_index(axis=1),
                'annual_liquidity': pd.DataFrame(self.stability_data['annual_liquidity']).sort_index(axis=1),
                'monthly_borrowings': pd.DataFrame(self.stability_data['monthly_borrowings']).sort_index(axis=1),
                'monthly_liquidity': pd.DataFrame(self.stability_data['monthly_liquidity']).sort_index(axis=1)
            }

        except Exception as e:
            logger.error(f"재무 안정성 지표 계산 실패: {str(e)}")
            raise

    def _calculate_annual_borrowings(self, year: int, estimate: bool = False):
        """연도별 차입금 데이터 계산"""
        current_data = self.df_company[
            (self.df_company['year'] == year) &
            (self.df_company['month'] ==
             (self.latest_date.month if estimate else 12))
        ]

        if not current_data.empty:
            short_term_debt = current_data['단기차입금'].iloc[-1]
            long_term_debt = current_data['장기차입금'].iloc[-1]
            total_assets = current_data['총자산'].iloc[-1]
            debt_ratio = round((short_term_debt + long_term_debt) /
                               total_assets * 100, 2) if total_assets != 0 else 0
        else:
            short_term_debt = long_term_debt = total_assets = debt_ratio = 0

        month_filter = self.latest_date.month if estimate else 12
        current_industry_data = self.df_industry[
            (self.df_industry['year'] == year) &
            (self.df_industry['month'] == month_filter)
        ]

        individual_ratios = ((current_industry_data['단기차입금'] + current_industry_data['장기차입금']) /
                             current_industry_data['총자산'] * 100)
        industry_debt_ratio = round(individual_ratios[individual_ratios != 0].mean(
        ), 2) if not individual_ratios.empty else 0

        key = f"{year}년(E)" if estimate else f"{year}년"
        self.stability_data['annual_borrowings'][key] = {
            '단기차입금': short_term_debt,
            '장기차입금': long_term_debt,
            '차입금의존도': debt_ratio,
            '업종평균 차입금의존도': industry_debt_ratio
        }

    def _calculate_annual_liquidity(self, year: int, estimate: bool = False):
        """연도별 유동성 데이터 계산"""
        current_data = self.df_company[
            (self.df_company['year'] == year) &
            (self.df_company['month'] ==
             (self.latest_date.month if estimate else 12))
        ]

        if not current_data.empty:
            current_assets = current_data['유동자산'].iloc[-1]
            current_liabilities = current_data['유동부채'].iloc[-1]
            current_ratio = round(
                (current_assets / current_liabilities) * 100, 2) if current_liabilities != 0 else 0
        else:
            current_assets = current_liabilities = current_ratio = 0

        month_filter = self.latest_date.month if estimate else 12
        current_industry_data = self.df_industry[
            (self.df_industry['year'] == year) &
            (self.df_industry['month'] == month_filter)
        ]

        industry_ratios = (
            current_industry_data['유동자산'] / current_industry_data['유동부채'] * 100)
        industry_current_ratio = round(industry_ratios[industry_ratios != 0].mean(
        ), 2) if not industry_ratios.empty else 0

        key = f"{year}년(E)" if estimate else f"{year}년"
        self.stability_data['annual_liquidity'][key] = {
            '유동자산': current_assets,
            '유동부채': current_liabilities,
            '유동비율': current_ratio,
            '업종평균 유동비율': industry_current_ratio
        }

    def _process_monthly_borrowings(self):
        """최근 12개월 월별 차입금 데이터 계산"""
        past_12_months = pd.date_range(
            end=self.latest_date + pd.DateOffset(months=1),
            periods=12,
            freq='ME'
        )

        for date in past_12_months:
            month_str = date.strftime('%Y-%m')
            current_month = self.df_company[
                self.df_company['date'].dt.strftime('%Y-%m') == month_str
            ]

            if not current_month.empty:
                total_debt = current_month['단기차입금'].iloc[0] + \
                    current_month['장기차입금'].iloc[0]
                short_term_debt = current_month['단기차입금'].iloc[0]
                revenue = current_month['매출액'].iloc[0]
                debt_to_sales_ratio = round(
                    (total_debt / revenue) * 100, 2) if revenue != 0 else 0
            else:
                total_debt = short_term_debt = revenue = debt_to_sales_ratio = 0

            current_industry_data = self.df_industry[
                self.df_industry['date'].dt.strftime('%Y-%m') == month_str
            ]
            industry_ratios = ((current_industry_data['단기차입금'] + current_industry_data['장기차입금']) /
                               current_industry_data['매출액'] * 100)
            industry_ratio = round(industry_ratios[industry_ratios != 0].mean(
            ), 2) if not industry_ratios.empty else 0

            self.stability_data['monthly_borrowings'][month_str] = {
                '총차입금': total_debt,
                '단기차입금': short_term_debt,
                '매출액대비차입비율': debt_to_sales_ratio,
                '업종평균 매출대비차입비율': industry_ratio
            }

    def _process_monthly_liquidity(self):
        """최근 12개월 월별 유동성 데이터 계산"""
        past_12_months = pd.date_range(
            end=self.latest_date + pd.DateOffset(months=1),
            periods=12,
            freq='ME'
        )

        for date in past_12_months:
            month_str = date.strftime('%Y-%m')
            current_month = self.df_company[
                self.df_company['date'].dt.strftime('%Y-%m') == month_str
            ]

            if not current_month.empty:
                working_capital = (current_month['매출채권'].iloc[0] -
                                   current_month['매입채무'].iloc[0] +
                                   current_month['재고자산'].iloc[0])
                revenue = current_month['매출액'].iloc[0]
                working_capital_turnover = round(
                    (working_capital / revenue) * 100, 2) if revenue != 0 else 0
            else:
                working_capital = revenue = working_capital_turnover = 0

            current_industry_data = self.df_industry[
                self.df_industry['date'].dt.strftime('%Y-%m') == month_str
            ]
            industry_turnovers = ((current_industry_data['매출채권'] -
                                   current_industry_data['매입채무'] +
                                   current_industry_data['재고자산']) /
                                  current_industry_data['매출액'] * 100)
            industry_turnover = round(industry_turnovers[industry_turnovers != 0].mean(
            ), 2) if not industry_turnovers.empty else 0

            self.stability_data['monthly_liquidity'][month_str] = {
                '운전자금': working_capital,
                '운전자금회전율': working_capital_turnover,
                '업종평균 운전자금회전율': industry_turnover
            }


def preprocess_financial_stability_data(df: pd.DataFrame, company_name: str, access_time: datetime) -> Dict[str, Any]:
    """재무 안정성 데이터 전처리 함수"""
    try:
        logger.info(
            f"[Financial] Starting financial stability preprocessing for {company_name}")

        calculator = FinancialStabilityMetricsCalculator(
            df=df,
            company_name=company_name,
            access_time=access_time
        )
        metrics = calculator.calculate_metrics()

        # DataFrame을 직렬화 가능한 dict로 변환 (지표명 포함)
        result = {
            'latest_year_month': metrics['latest_year_month'],
            'annual_borrowings': [
                {
                    '지표': row_name,
                    **metrics['annual_borrowings'].loc[row_name].to_dict()
                } for row_name in metrics['annual_borrowings'].index
            ],
            'annual_liquidity': [
                {
                    '지표': row_name,
                    **metrics['annual_liquidity'].loc[row_name].to_dict()
                } for row_name in metrics['annual_liquidity'].index
            ],
            'monthly_borrowings': [
                {
                    '지표': row_name,
                    **metrics['monthly_borrowings'].loc[row_name].to_dict()
                } for row_name in metrics['monthly_borrowings'].index
            ],
            'monthly_liquidity': [
                {
                    '지표': row_name,
                    **metrics['monthly_liquidity'].loc[row_name].to_dict()
                } for row_name in metrics['monthly_liquidity'].index
            ]
        }

        logger.info(
            f"[Financial] Financial stability preprocessing completed for {company_name}")
        return result

    except Exception as e:
        logger.error(f"재무 안정성 데이터 전처리 실패: {str(e)}")
        raise
