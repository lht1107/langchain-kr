import pandas as pd
import numpy as np
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Set a consistent random seed for reproducibility
random_state = np.random.RandomState(42)

# Define company and industry details
NUM_COMPANIES = 100
BASE_ASSET_VALUE = 30000
INDUSTRIES = ['Manufacturing', 'Retail', 'Technology', 'Healthcare', 'Finance']

company_names = [f'Company_{i + 1}' for i in range(NUM_COMPANIES)]
company_INDUSTRIES = [random_state.choice(
    INDUSTRIES) for _ in range(NUM_COMPANIES)]

# logger.debug("Initialized company and industry details.")

# Define partner pools
sales_partner_pool = list({f'SalesPartner_{i + 1}' for i in range(8)})
purchase_partner_pool = list({f'PurchasePartner_{i + 1}' for i in range(8)})

# logger.debug("Defined sales and purchase partner pools.")

# Set base employee count per company
company_base_employees = {company: random_state.randint(
    100, 300) for company in company_names}
# logger.debug("Assigned base employee count per company.")


def generate_partner_data():
    """Generate top sales and purchase partner information."""
    logger.debug("Generating partner data for sales and purchase.")

    sales_companies = list(random_state.choice(
        sales_partner_pool, 5, replace=False))
    sales_grades = list(random_state.choice(
        ['AAA', 'AA+', 'AA', 'A+', 'A', 'BBB+', 'BBB', 'BB', 'B', 'CCC-'], 5, replace=False))

    purchase_companies = list(random_state.choice(
        purchase_partner_pool, 5, replace=False))
    purchase_grades = list(random_state.choice(
        ['AAA', 'AA+', 'AA', 'A+', 'A', 'BBB+', 'BBB', 'BB', 'B', 'CCC-'], 5, replace=False))

    logger.debug("Partner data generated for sales and purchase.")

    return [
        {
            "매출처_회사명": sales_companies[i],
            "매출처_신용등급": sales_grades[i],
            "매출처_매출금액": int(random_state.uniform(1000, 10000)),
            "매출처_회수금액": int(random_state.uniform(500, 5000)),
            "매출처_회수기일": int(random_state.uniform(30, 90)),
        }
        for i in range(5)
    ], [
        {
            "매입처_회사명": purchase_companies[i],
            "매입처_신용등급": purchase_grades[i],
            "매입처_매입금액": int(random_state.uniform(1000, 10000)),
            "매입처_지급금액": int(random_state.uniform(100, 1000)),
            "매입처_지급기일": int(random_state.uniform(30, 90)),
        }
        for i in range(5)
    ]


def generate_basic_data(company, industry, date):
    """Generate basic financial data for a company at a given date."""
    logger.debug(f"Generating basic data for {company} at {date}.")

    total_assets = round(BASE_ASSET_VALUE *
                         (1 + random_state.normal(0, 0.05)), 0)
    revenue = round(random_state.uniform(1000, 10000), 0)
    operating_profit = round(revenue * random_state.uniform(-0.2, 0.2), 0)
    net_income = round(operating_profit * random_state.uniform(-0.5, 0.8), 0)
    short_term_loans = round(random_state.uniform(500, 5000), 0)
    long_term_loans = round(random_state.uniform(1000, 10000), 0)
    total_loans = short_term_loans + long_term_loans
    loan_to_sales = round((total_loans / revenue) * 100, 2)
    working_capital_turnover = round(random_state.uniform(1, 5), 2)
    operating_cash_flow = round(random_state.uniform(500, 5000), 0)
    net_cash_flow = round(operating_cash_flow - total_loans, 0)
    ar_balance = round(random_state.uniform(1000, 5000), 0)
    ap_balance = round(random_state.uniform(800, 4000), 0)
    inventory = round(random_state.uniform(2000, 10000), 0)
    current_assets = round(random_state.uniform(
        15000, 30000), 0)  # 유동자산 (1.5억원 ~ 3억원)
    current_liabilities = round(random_state.uniform(
        10000, 25000), 0)  # 유동부채 (1억원 ~ 2.5억원)

    # Generate partner data
    sales_partners, purchase_partners = generate_partner_data()

    # Calculate employees based on base count with variation
    base_employees = company_base_employees[company]
    employees = round(base_employees * (1 + random_state.uniform(-0.1, 0.1)))

    logger.debug(f"Data generation completed for {company} on {date}")

    return {
        '기업명': company,
        '업종': industry,
        '날짜': date,
        '매출액증가율': round(random_state.uniform(5, 15), 2),
        '총자산증가율': round(random_state.uniform(3, 10), 2),
        '총자산': int(total_assets),
        '매출액': int(revenue),
        '영업이익': int(operating_profit),
        '영업이익률': round(operating_profit / revenue * 100, 2) if revenue != 0 else round(random_state.uniform(-20, 20), 2),
        '당기순이익': int(net_income),
        '당기순이익률': round(net_income / revenue * 100, 2) if revenue != 0 else round(random_state.uniform(-30, 30), 2),
        '단기차입금': int(short_term_loans),
        '장기차입금': int(long_term_loans),
        '유동자산': int(current_assets),  # 유동자산 (정수형)
        '유동부채': int(current_liabilities),  # 유동부채 (정수형)
        '매출대비차입금': loan_to_sales,
        '운전자금회전율': working_capital_turnover,
        '인원수': employees,
        '월평균급여액': round(random_state.uniform(200, 500), 1),
        '월매출창출액': round(random_state.uniform(5000, 20000), 1),
        '영업활동현금흐름/매출액': round(random_state.uniform(0.05, 0.2), 2),
        '영업활동현금흐름': int(operating_cash_flow),
        '순현금흐름': int(net_cash_flow),
        '매출채권': int(ar_balance),
        '매입채무': int(ap_balance),
        '재고자산': int(inventory),
        '상위_매출처': sales_partners,
        '상위_매입처': purchase_partners
    }


def generate_sample_data(access_datetime):
    """
    Generate sample data up to one month before the specified access_datetime.

    Args:
        access_datetime (datetime): 접근 시간

    Returns:
        pd.DataFrame: 생성된 회사 정보 데이터프레임
    """
    logger.info(f"Generating sample data up to {access_datetime}.")

    # 날짜 범위 생성 (4년 전부터 한 달 전까지)
    start_date = f"{access_datetime.year - 4}-01-01"
    dates = pd.date_range(
        start=start_date,
        end=access_datetime - pd.DateOffset(months=1),
        freq='MS'
    ).strftime("%Y-%m").tolist()

    # 회사 및 업종 정보 생성
    company_names = [f'Company_{i + 1}' for i in range(NUM_COMPANIES)]
    company_industries = [random_state.choice(
        INDUSTRIES) for _ in range(NUM_COMPANIES)]

    # 데이터 생성
    basic_data = [
        generate_basic_data(company, industry, date)
        for company, industry in zip(company_names, company_industries)
        for date in dates
    ]

    df_company_info = pd.DataFrame(basic_data)
    df_company_info['날짜'] = pd.to_datetime(
        df_company_info['날짜'], format="%Y-%m")

    logger.info("Sample data generation complete.")
    return df_company_info
