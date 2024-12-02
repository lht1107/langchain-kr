from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from database.f_read_pg_sql import fetch_company_data
from utils.logger import get_logger
from fastapi import HTTPException

# Initialize logger
logger = get_logger(__name__)


def generate_sql_query(access_time: str, no_com: int, months: int, coa_list: list) -> str:
    """
    SQL 쿼리를 생성합니다.

    Args:
        access_time (str): 기준 날짜 (예: '2024-05-01')
        no_com (int): 회사 번호
        months (int): 조회할 월 수
        coa_list (list): COA 리스트

    Returns:
        str: 생성된 SQL 쿼리 문자열
    """
    # 기준 날짜 생성
    try:
        base_date = datetime.strptime(access_time, '%Y-%m-%d')
        logger.info(f"Access time converted successfully: {access_time}")
    except ValueError as e:
        logger.error(
            f"Invalid access_time format: {access_time}. Expected format is YYYY-MM-DD. Error: {str(e)}")
        raise ValueError(f"Invalid access_time format: {str(e)}")

    date_list = [(base_date - relativedelta(months=i)).strftime('%Y%m')
                 for i in range(months)]
    logger.debug(f"Date list generated: {date_list}")

    # COA 리스트를 SQL IN 절에 맞게 변환
    coa_in_clause = ','.join(f"'{coa}'" for coa in coa_list)

    # COA 이름 매핑용 sub query 작성
    coaname_subquery = f"""
        SELECT coa, coaname
        FROM wksch.coatable
        WHERE category2 IN ('자산', '부채', '자본', '수익', '비용', '원가', '현금흐름', '비율')
        AND category1 = '테크핀'
        AND coa IN ({coa_in_clause})
    """

    try:
        # COA와 COA 이름을 매핑하는 쿼리 실행
        coa_mapping = fetch_company_data(coaname_subquery)
        logger.info("Successfully fetched COA names.")
    except Exception as e:
        logger.error(f"Error while fetching COA names: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error while fetching COA names: {str(e)}")

    # Pivot 컬럼을 Python에서 동적으로 생성
    try:
        pivot_columns = ',\n'.join(
            [f"MAX(CASE WHEN combined_data.coa = '{row.coa}' THEN combined_data.mybungae END) AS \"{row.coaname}\""
             for _, row in coa_mapping.iterrows()]
        )
        logger.debug(f"Pivot columns created successfully.")
    except KeyError as e:
        logger.error(f"Error while creating pivot columns: {str(e)}")
        raise ValueError(
            f"Missing expected column while creating pivot columns: {str(e)}")

    # 모든 월별 데이터를 UNION ALL로 결합
    union_queries = []
    for i, date in enumerate(date_list, start=1):
        union_queries.append(
            f"SELECT mycoa, mybungae, '{date}' AS date "
            f"FROM wksch.mapping_jv_cf({no_com}, '{date}', '{date}', '테크핀', "
            f"ARRAY['자산', '부채', '자본', '수익', '비용', '원가', '현금흐름', '비율'], 0, 0)"
        )

    # 최종 쿼리 구성
    final_query = f"""
    WITH basic AS (
        SELECT coa, sortorder, coaname, category2, calculation_douzone, calculation_techfin
        FROM wksch.coatable
        WHERE category2 IN ('자산', '부채', '자본', '수익', '비용', '원가', '현금흐름', '비율')
        AND category1 = '테크핀'
        AND coa IN ({coa_in_clause})
    ),
    combined_f AS (
        {' UNION ALL '.join(union_queries)}
    ),
    combined_data AS (
        SELECT f.date,
            insa.ct_work,
            basic.coa,
            basic.coaname,
            f.mybungae
        FROM basic
        JOIN combined_f f ON basic.coa = f.mycoa
        LEFT JOIN dtsch.ftw_insa AS insa 
        ON insa.no_com = {no_com}
        AND insa.dm_insa = f.date
    )
    SELECT 
        date,
        ct_work,
        {pivot_columns}
    FROM combined_data
    GROUP BY date, ct_work
    ORDER BY date;
    """

    logger.info("SQL query generated successfully.")
    return final_query
