import pandas as pd
from typing import Dict, Any
import logging
from utils.logger import get_logger
from utils.validation import validate_input_data, DataValidationError

logger = get_logger(__name__)


class PartnerStabilityMetricsCalculator:
    """거래처 안정성 지표 계산 클래스"""

    REQUIRED_COLUMNS = ['기업명', '날짜', '상위_매출처', '상위_매입처']
    COMPANY_COL, DATE_COL = '기업명', '날짜'

    def __init__(self, df: pd.DataFrame, target_company_name: str, access_time: str):
        logger.info(
            f"[Initialize] Starting PartnerStabilityMetricsCalculator for {target_company_name}")
        self.access_date = access_time
        self.df = validate_input_data(
            df, self.REQUIRED_COLUMNS, self.COMPANY_COL)
        self.target_company_name = target_company_name
        self.df[self.DATE_COL] = pd.to_datetime(self.df[self.DATE_COL])
        self.df_company = self.df[self.df[self.COMPANY_COL]
                                  == self.target_company_name]
        if self.df_company.empty:
            error_msg = f"대상 회사 {self.target_company_name} 데이터가 존재하지 않습니다."
            logger.error(f"[Error] {error_msg}")
            raise DataValidationError(error_msg)
        self.latest_date = (self.access_date -
                            pd.DateOffset(months=1)).strftime('%Y-%m')

    def _process_partner_data(self, partner_type: str) -> Dict[str, Any]:
        """거래처 데이터 처리"""
        logger.info(f"[Partner] Processing {partner_type} data started")
        start_date = pd.to_datetime(
            self.latest_date) - pd.DateOffset(months=11)

        # 기간 필터링
        logger.debug(f"[Partner] Filtering data for {partner_type}")
        period_data = self.df_company[
            (self.df_company['날짜'] >= start_date) &
            (self.df_company['날짜'] <= pd.to_datetime(
                self.latest_date) + pd.offsets.MonthEnd(1))
        ]

        # 거래처 데이터 추출
        partner_key = f'상위_{partner_type}처'
        partner_data = []
        for _, row in period_data.iterrows():
            for partner in row[partner_key]:
                partner_dict = partner.copy()
                partner_dict['날짜'] = row['날짜']
                partner_data.append(partner_dict)

        if not partner_data:
            return {}

        # 컬럼명 설정
        company_col = f'{partner_type}처_회사명'
        grade_col = f'{partner_type}처_신용등급'
        amount_col = f'{partner_type}처_{partner_type}금액'
        term_col = f'{partner_type}처_회수기일' if partner_type == '매출' else f'{partner_type}처_지급기일'

        # 데이터프레임 생성
        partner_df = pd.DataFrame(partner_data)

        # Top 5 거래처 선정
        logger.debug(f"[Partner] Selecting top 5 {partner_type} partners")
        total_by_company = partner_df.groupby(company_col)[amount_col].sum()
        top_5_companies = total_by_company.nlargest(5).index.tolist()

        # 연간 데이터 생성
        logger.debug(f"[Partner] Generating annual data for {partner_type}")
        annual_data = []
        total_amount = int(total_by_company.sum())
        top_5_amount = int(total_by_company[top_5_companies].sum())

        for company in top_5_companies:
            company_data = partner_df[partner_df[company_col] == company]
            annual_amount = int(total_by_company[company])

            annual_data.append({
                '회사명': company,
                '신용등급': company_data[grade_col].iloc[0],
                f'연간_{partner_type}금액': annual_amount,
                '거래비율': round((annual_amount / total_amount * 100), 2)
            })

        # 합계 정보 추가
        annual_data.extend([
            {
                '회사명': '상위 5개 거래처 합계',
                f'연간_{partner_type}금액': top_5_amount,
                '거래비율': round((top_5_amount / total_amount * 100), 2)
            },
            {
                '회사명': '기타 거래처 합계',
                f'연간_{partner_type}금액': int(total_amount - top_5_amount),
                '거래비율': round(((total_amount - top_5_amount) / total_amount * 100), 2)
            }
        ])

        # # 월별 데이터 생성
        # logger.debug(f"[Partner] Generating monthly data for {partner_type}")
        # monthly_data = {}
        # for company in top_5_companies:
        #     company_data = partner_df[partner_df[company_col] == company]
        #     company_monthly = []

        #     # 전체 기간의 합계 계산
        #     total_term = int(company_data[term_col].sum())
        #     annual_amount = int(total_by_company[company])

        #     # 최근 12개월 데이터 추가
        #     for month in pd.date_range(start=start_date, end=pd.to_datetime(self.latest_date) + pd.offsets.MonthEnd(1), freq='ME'):
        #         month_data = company_data[
        #             company_data['날짜'].dt.to_period(
        #                 'M') == month.to_period('M')
        #         ]

        #         if not month_data.empty:
        #             amount = int(month_data[amount_col].sum())
        #         else:
        #             amount = 0

        #         company_monthly.append({
        #             '날짜': month.strftime('%Y-%m'),
        #             f'{partner_type}금액': amount
        #         })

        #     # 평균 데이터 추가
        #     company_monthly.extend([
        #         {
        #             '날짜': f'평균{partner_type}금액',
        #             f'{partner_type}금액': round(annual_amount / 12, 2)
        #         },
        #         {
        #             '날짜': f'평균{"회수" if partner_type == "매출" else "지급"}기일',
        #             f'{partner_type}금액': round(total_term / 12, 2)
        #         }
        #     ])

        #     monthly_data[company] = company_monthly
        # logger.info(f"[Partner] Processing {partner_type} data completed")
        # return {
        #     'annual_data': annual_data,
        #     'monthly_data': monthly_data
        # }
        # 월별 데이터 생성 부분 수정
        logger.debug(f"[Partner] Generating monthly data for {partner_type}")
        monthly_data = {company: {}
                        for company in top_5_companies}  # 회사별로 딕셔너리 초기화

        # 날짜 리스트 생성
        date_range = pd.date_range(start=start_date,
                                   end=pd.to_datetime(
                                       self.latest_date) + pd.offsets.MonthEnd(1),
                                   freq='ME')
        dates = [date.strftime('%Y-%m') for date in date_range]

        # 각 회사별로 날짜 데이터를 저장
        for company in top_5_companies:
            company_data = partner_df[partner_df[company_col] == company]
            annual_amount = int(total_by_company[company])
            total_term = int(company_data[term_col].sum())

            # 월별 데이터 추가
            for date in dates:
                month_data = company_data[company_data['날짜'].dt.strftime(
                    '%Y-%m') == date]
                monthly_data[company][date] = int(
                    month_data[amount_col].sum()) if not month_data.empty else '-'

            # 평균 데이터 추가
            monthly_data[company][f'평균{partner_type}금액'] = round(
                annual_amount / 12, 2)
            monthly_data[company][f'평균{"회수" if partner_type == "매출" else "지급"}기일'] = round(
                total_term / 12, 2)

        return {
            'annual_data': annual_data,
            'monthly_data': monthly_data
        }

    def calculate_stability_metrics(self) -> Dict[str, Any]:
        """안정성 지표 계산 실행"""
        logger.info("[Metrics] Starting stability metrics calculation")
        sales_data = self._process_partner_data('매출')
        purchase_data = self._process_partner_data('매입')
        logger.info("[Metrics] Stability metrics calculation completed")
        return {
            'latest_year_month': self.latest_date,
            'annual_top5_sales': sales_data.get('annual_data', []),
            'monthly_top5_sales': sales_data.get('monthly_data', {}),
            'annual_top5_purchase': purchase_data.get('annual_data', []),
            'monthly_top5_purchase': purchase_data.get('monthly_data', {})
        }


# def preprocess_partner_stability_data(df: pd.DataFrame, company: str, access_time: str) -> Dict[str, Any]:
#     """안정성 지표 전처리 함수"""
#     logger.info(
#         f"[Preprocess] Starting partner stability preprocessing for {company}")

#     calculator = PartnerStabilityMetricsCalculator(df, company, access_time)
#     result = calculator.calculate_stability_metrics()
#     logger.info(
#         f"[Preprocess] Partner stability preprocessing completed for {company}")
#     return result

def preprocess_partner_stability_data(df: pd.DataFrame, company: str, access_time: str) -> Dict[str, Any]:
    """안정성 지표 전처리 함수"""
    logger.info(
        f"[Preprocess] Starting partner stability preprocessing for {company}")

    calculator = PartnerStabilityMetricsCalculator(df, company, access_time)
    metrics = calculator.calculate_stability_metrics()

    # 직렬화 가능한 형태로 변환
    result = {
        'latest_year_month': metrics['latest_year_month'],
        'annual_top5_sales': metrics['annual_top5_sales'],
        'monthly_top5_sales': [
            {
                '지표': date,
                **{partner: values[partner] for partner in values}
            } for date, values in metrics['monthly_top5_sales'].items()
        ],
        'annual_top5_purchase': metrics['annual_top5_purchase'],
        'monthly_top5_purchase': [
            {
                '지표': date,
                **{partner: values[partner] for partner in values}
            } for date, values in metrics['monthly_top5_purchase'].items()
        ]
    }

    logger.info(
        f"[Preprocess] Partner stability preprocessing completed for {company}")
    return result
