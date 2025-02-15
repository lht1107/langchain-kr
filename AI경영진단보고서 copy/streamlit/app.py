# 1. 기본 모듈 임포트
# fmt: off
import nest_asyncio
nest_asyncio.apply()
from typing import Dict, Any, Optional
import asyncio
from functools import partial
# import sqlite3
from datetime import datetime
import json
import io
import time
import requests
import pandas as pd
import streamlit as st

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core import settings
from utils.load_prompt import load_prompt 
from database.sqlite_cache import SQLiteCache
from database.postgresql_cache import PostgreSQLCache
# fmt: on

# 2. 프로젝트 루트 경로 설정

# isort: skip


def get_cache_manager():
    """DB_TYPE 설정에 따라 적절한 캐시 매니저를 반환"""
    try:
        if settings.DB_TYPE.lower() == 'sqlite':
            db_path = os.path.join(
                settings.BASE_DIR, "database", "sqlite_cache.db")
            return SQLiteCache(db_path)
        elif settings.DB_TYPE.lower() == 'postgresql':
            return PostgreSQLCache()
        else:
            raise ValueError(f"Unsupported DB_TYPE: {settings.DB_TYPE}")
    except Exception as e:
        st.error(f"데이터베이스 연결 실패: {str(e)}")
        return None


# 데이터베이스 경로 설정
db_path = os.path.join(settings.BASE_DIR, "database", "sqlite_cache.db")

# 페이지 설정
st.set_page_config(
    page_title="AI경영진단보고서 베타",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",


)

metrics_mapping = {
    'growth': {
        'title': '💹 성장성',
        'annual': ['annual_revenue', 'annual_assets'],
        'monthly': ['monthly_revenue', 'monthly_growth']
    },
    'profitability': {
        'title': '📈 수익성',
        'annual': ['annual_profit', 'annual_margins'],
        'monthly': ['monthly_profit', 'monthly_margins']
    },
    'partner_stability': {
        'title': '🤝 거래처 안정성',
        'annual': ['annual_top5_sales', 'annual_top5_purchase'],
        'monthly': ['monthly_top5_sales', 'monthly_top5_purchase']
    },
    'financial_stability': {
        'title': '💰 재무 안정성',
        'annual': ['annual_borrowings', 'annual_liquidity'],
        'monthly': ['monthly_borrowings', 'monthly_liquidity']
    },
    'hr': {
        'title': '👥 인적 관리',
        'annual': ['annual_employee_metrics', 'average_tenure_by_age', 'average_salary_by_age', 'tenure_distribution', 'age_distribution'],
        'monthly': ['monthly_employee_metrics', 'monthly_salary_and_revenue_metrics'],
    },
    'cashflow': {
        'title': '💵 현금흐름',
        'annual': ['operating_activities', 'investing_activities', 'financing_activities'],
        'monthly': ['monthly_cashflow']
    }
}


def process_dataframe(metrics_data: Dict, metric: str) -> pd.DataFrame:
    """
    데이터를 데이터프레임으로 변환하고 NaN을 "-"로 변환.
    이미 데이터프레임인 경우 그대로 반환.
    """
    if isinstance(metrics_data[metric], pd.DataFrame):
        return metrics_data[metric]  # 이미 데이터프레임인 경우 그대로 반환

    # JSON 데이터인 경우 데이터프레임으로 변환
    df = pd.DataFrame(metrics_data[metric])
    return df.fillna("-")  # NaN 값을 "-"로 변환


def get_server_url():
    """서버 URL을 구성하는 함수"""
    return f"{settings.SERVER_SCHEME}://{settings.SERVER_HOST}:{settings.SERVER_PORT}"


def submit_feedback(company_name: str, feedback_type: str, feedback_text: str,
                    analysis_type: str, analysis_metric: str) -> bool:
    """피드백 제출 함수"""
    try:
        if not feedback_text:
            st.error("피드백 내용을 입력해주세요.")
            return False

        feedback_data = {
            "company_name": company_name or "unknown",
            "feedback_type": feedback_type,
            "analysis_type": analysis_type,
            "analysis_metric": analysis_metric or "none",
            "feedback_text": feedback_text
        }

        # 서버 URL 구성
        server_url = get_server_url()

        response = requests.post(
            f"{server_url}/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        return response.status_code == 200

    except requests.exceptions.RequestException as e:
        st.error(f"피드백 전송 중 오류가 발생했습니다. 다시 시도해주세요.")
        return False

# SQLite 캐시 확인 함수


def check_cache(company_name: str, strength_metric: str = None, weakness_metric: str = None) -> Optional[Dict]:
    """
    캐시된 분석 결과를 조회하는 함수

    Args:
        company_name: 회사명
        strength_metric: 강점 분석 지표 (선택)
        weakness_metric: 약점 분석 지표 (선택)

    Returns:
        Optional[Dict]: 캐시된 분석 결과 또는 None
    """
    try:
        # 캐시 매니저 초기화
        if 'cache_manager' not in st.session_state:
            st.session_state.cache_manager = get_cache_manager()

        cache_manager = st.session_state.cache_manager

        if strength_metric and weakness_metric:
            # insight 분석 결과 조회
            result = asyncio.run(cache_manager.get(
                company_name,
                analysis_type="insight",
                analysis_metric=f"{strength_metric}/{weakness_metric}"
            ))

            if result:
                return result

            # insight 결과가 없으면 개별 strength/weakness 결과 조회
            strength_result = asyncio.run(cache_manager.get(
                company_name,
                analysis_type="strength",
                analysis_metric=strength_metric
            ))
            weakness_result = asyncio.run(cache_manager.get(
                company_name,
                analysis_type="weakness",
                analysis_metric=weakness_metric
            ))

            if strength_result and weakness_result:
                return {
                    "strength": strength_result["strength"],
                    "weakness": weakness_result["weakness"]
                }
            return None

        # company_name만 있는 경우의 조회
        return asyncio.run(cache_manager.get(company_name))

    except Exception as e:
        st.error(f"캐시 조회 오류: {str(e)}")
        return None


# async def fetch_analysis_data(cache_manager, company_name, strength_metric, weakness_metric):
#     if strength_metric and weakness_metric:
#         # Insight 분석 결과 조회
#         result = await cache_manager.get(
#             company_name,
#             analysis_type="insight",
#             analysis_metric=f"{strength_metric}/{weakness_metric}"
#         )

#         if result:
#             return result

#         # 개별 Strength/Weakness 결과 조회
#         strength_result = await cache_manager.get(
#             company_name,
#             analysis_type="strength",
#             analysis_metric=strength_metric
#         )
#         weakness_result = await cache_manager.get(
#             company_name,
#             analysis_type="weakness",
#             analysis_metric=weakness_metric
#         )

#         if strength_result and weakness_result:
#             return {
#                 "strength": strength_result["strength"],
#                 "weakness": weakness_result["weakness"]
#             }
#         return None

#     # Company_name만 있는 경우
#     return await cache_manager.get(company_name)


# def check_cache(company_name: str, strength_metric: str = None, weakness_metric: str = None):
#     try:
#         if 'cache_manager' not in st.session_state:
#             st.session_state.cache_manager = get_cache_manager()

#         cache_manager = st.session_state.cache_manager

#         result = asyncio.run(fetch_analysis_data(
#             cache_manager, company_name, strength_metric, weakness_metric
#         ))
#         return result

#     except Exception as e:
#         st.error(f"캐시 조회 오류: {str(e)}")
#         return None


@st.cache_data
def load_company_data():
    return pd.read_parquet(os.path.join(settings.DATA_PATH, 'overview_table.parquet'))


def perform_analysis(selected_company: str, cached_data: Optional[Dict]) -> Optional[Dict]:
    """새로운 분석 수행"""
    try:
        if cached_data and isinstance(cached_data, dict) and all(
            cached_data.get(k, {}).get('analysis_metric')
            for k in ['strength', 'weakness', 'insight']
        ):
            return cached_data

        strength_metric = st.session_state.strength_selector
        weakness_metric = st.session_state.weakness_selector

        server_url = get_server_url()
        response = requests.post(
            f"{server_url}/analyze/",
            json={
                "company_name": selected_company,
                "analysis_type": "insight",
                "strength_metric": strength_metric,
                "weakness_metric": weakness_metric
            },
            timeout=60
        )
        response.raise_for_status()

        if response.status_code == 200:
            # asyncio.run() 대신 직접 호출
            return check_cache(
                selected_company,
                strength_metric=strength_metric,
                weakness_metric=weakness_metric
            )

        st.error("분석은 완료되었으나 결과를 불러오는데 실패했습니다.")
        return None

    except requests.exceptions.Timeout:
        st.error("분석 시간이 초과되었습니다. 다시 시도해주세요.")
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 오류가 발생했습니다: {str(e)}")
    return None


def display_analysis_results(data, title, metric_info, company_name, is_strength=True):
    # 탭 생성
    container = st.container()
    with container:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 연간 지표",
            "📊 월간 지표",
            "📝 상세 분석",
            "🎯 분석 템플릿",
            "💡 요약 템플릿",
        ])
        st.markdown(
            """
        <style>
            div[data-testid="stHorizontalBlock"] {
                background-color: #f0f2f6;  /* 연한 회색 배경 */
                padding: 20px;  /* 내부 여백 */
                border-radius: 10px;  /* 모서리 둥글게 */
                border: 1px solid #e0e0e0;  /* 테두리 */
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* 그림자 효과 */
                width: 130%;  /* 컨테이너 너비 확장 */
                margin-left: -15%;  /* 좌측 마진으로 중앙 정렬 */
            }
            div.stTabs {
                border: 1px solid #e0e0e0;  /* 탭 컨테이너 테두리 */
                border-radius: 5px;  /* 모서리 둥글게 */
                padding: 10px;  /* 내부 여백 */
            }
            /* 탭 버튼 너비 균등 분할 */
            button[data-baseweb="tab"] {
                flex: 1 1 20%;  /* 5개 탭을 동일한 너비로 분할 */
                min-width: 0;  /* 최소 너비 제한 제거 */
                text-align: center;  /* 텍스트 중앙 정렬 */
            }
            /* 선택된 탭 스타일링 */
            button[data-baseweb="tab"][aria-selected="true"] {
                background-color: #e6f3ff;  /* 연한 파란색 배경 */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* 그림자 효과 */
                font-weight: bold !important;  /* 굵은 글씨 */
            }
        </style>
            """,
            unsafe_allow_html=True
        )

        # 서버 URL 구성
        server_url = get_server_url()

        # 연간 데이터 표시
        metrics_response = requests.post(
            f"{server_url}/query/",
            json={
                "company_name": company_name,
                "metric": st.session_state.strength_selector if is_strength else st.session_state.weakness_selector
            },
            timeout=60
        )

        metrics_data = None
        if metrics_response.status_code == 200:
            try:
                metrics_data = metrics_response.json()
            except json.JSONDecoderError:
                st.error("데이터 형식이 올바르지 않습니다.")
                return
        else:
            st.error("API 호출 실패: 데이터를 불러오지 못했습니다.")
            return

        with tab1:
            if metrics_data:
                for metric in metric_info.get('annual', []):
                    if metric in metrics_data:
                        df = process_dataframe(metrics_data, metric)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )
            # 현재 시간을 파일명에 포함
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 연간 데이터 다운로드",
                data=json.dumps(metrics_data, ensure_ascii=False, indent=2),
                file_name=f"{company_name}_annual_metrics_{timestamp}.json",
                mime="application/json",
                # 강점/약점 구분 추가
                key=f"annual_download_{timestamp}_{'strength' if is_strength else 'weakness'}"
            )
            st.caption("💡 데이터는 JSON 형식으로 저장됩니다.")

        with tab2:
            # 월간 데이터 표시
            if metrics_data:
                monthly_data = {k: v for k, v in metrics_data.items(
                ) if k in metric_info.get('monthly', [])}
                for metric in metric_info.get('monthly', []):
                    if metric in metrics_data:
                        df = process_dataframe(metrics_data, metric)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )
                # JSON 다운로드 버튼 추가
                st.download_button(
                    label="📥 월간 데이터 다운로드",
                    data=json.dumps(
                        monthly_data, ensure_ascii=False, indent=2),
                    file_name=f"{company_name}_monthly_metrics_{timestamp}.json",
                    mime="application/json",
                    # 강점/약점 구분 추가
                    key=f"monthly_download_{timestamp}_{'strength' if is_strength else 'weakness'}"
                )
                st.caption("💡 데이터는 JSON 형식으로 저장됩니다.")

        with tab3:
            # 상세 분석 결과
            st.markdown(data['detailed_result'])

        with tab4:
            # 분석 템플릿 로드 및 표시
            try:
                analysis_metric = data['analysis_metric']
                template_file = f"{analysis_metric}_template.txt"
                template_content = load_prompt(template_file)
                st.code(template_content, language='yaml')
            except FileNotFoundError as e:
                st.error(f"템플릿 파일을 찾을 수 없습니다: {str(e)}")

        with tab5:
            # 요약 템플릿 로드 및 표시
            try:
                summary_content = load_prompt("summary_template.txt")
                st.code(summary_content, language='markdown')
            except FileNotFoundError as e:
                st.error(f"요약 템플릿 파일을 찾을 수 없습니다: {str(e)}")

    time.sleep(1)  # 0.5초 간격
    # 요약 표시
    st.subheader(f"💡 {title} 요약")

    # def stream_summary():
    #     for word in data['summary'].split():
    #         yield word + " "
    #         time.sleep(0.02)

    st.write_stream(stream_summary(data['summary']))
    # st.markdown(data['summary'])


def format_summary_text(text: str) -> str:
    import re

    # 퍼센트 패턴만 먼저 처리
    percent_pattern = r'(-?\d+\.?\d*%)'
    formatted_text = re.sub(percent_pattern, r'**\1**', text)

    # 나머지 패턴들 처리
    patterns = [
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:백만|만)?\s*원)',
        r'(\d+(?:,\d{3})*\s*명)',
        r'(\d+(?:\.\d+)?\s*년)(?!도|간|말|초|부터|까지)',
        r'(\d+(?:\.\d+)?\s*배)'
    ]

    for pattern in patterns:
        formatted_text = re.sub(pattern, r'**\1**', formatted_text)

    return formatted_text


def stream_summary(text: str):
    import time

    formatted_text = format_summary_text(text)
    buffer = ""

    for char in formatted_text:
        if char == '*':
            buffer += char
            if len(buffer) >= 2 and buffer[-2:] == '**':
                yield buffer
                buffer = ""
        else:
            if buffer:
                yield buffer
                buffer = ""
            yield char
        time.sleep(0.01)

    if buffer:
        yield buffer


def generate_pdf_report(data, company_name):
    """PDF 보고서 생성"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Malgun Gothic 폰트 등록
    pdfmetrics.registerFont(TTFont('Malgun Gothic', 'malgun.ttf'))

    # 한글 지원 스타일 생성
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Korean',
        fontName='Malgun Gothic',
        fontSize=10,
        leading=16
    ))
    styles.add(ParagraphStyle(
        name='Heading',
        fontName='Malgun Gothic',
        fontSize=14,
        leading=20,
        spaceAfter=12
    ))

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []

    # 제목
    story.append(Paragraph(f"{company_name} 경영진단보고서", styles['Heading']))
    story.append(Spacer(1, 12))

    # 내용
    for section, title in [('strength', '강점'), ('weakness', '약점')]:
        # 섹션 제목
        story.append(Paragraph(title, styles['Heading']))

        # 지표 데이터 표시
        metric_info = metrics_mapping.get(data[section]['analysis_metric'], {})

        # 연간 지표
        if metric_info.get('annual'):
            story.append(Paragraph("연간 지표", styles['Korean']))

            # 서버 URL 구성
            server_url = get_server_url()

            metrics_response = requests.post(
                f"{server_url}/query/",
                json={
                    "company_name": company_name,
                    "metric": data['strength']['analysis_metric'] if section == 'strength' else data['weakness']['analysis_metric']
                },
                timeout=60
            )
            if metrics_response.status_code == 200:
                metrics_data = metrics_response.json()
                for metric in metric_info['annual']:
                    if metric in metrics_data:
                        df = pd.DataFrame(metrics_data[metric])
                        table_data = [df.columns.tolist()] + df.values.tolist()
                        table = Table(table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, -1), 'Malgun Gothic'),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 12))

        # 월간 지표
        if metric_info.get('monthly'):
            story.append(Paragraph("월간 지표", styles['Korean']))
            # 월간 지표 테이블 추가 (연간 지표와 동일한 방식)
            if metrics_data:
                monthly_data = {k: v for k, v in metrics_data.items(
                ) if k in metric_info.get('monthly', [])}
                for metric in metric_info.get('monthly', []):
                    if metric in monthly_data:
                        df = pd.DataFrame(monthly_data[metric])
                        table_data = [df.columns.tolist()] + df.values.tolist()
                        table = Table(table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, -1), 'Malgun Gothic'),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 12))

        # 요약
        story.append(Paragraph("분석 요약", styles['Korean']))
        story.append(Paragraph(data[section]['summary'], styles['Korean']))
        story.append(Spacer(1, 20))

    # 통찰 섹션
    story.append(Paragraph("통찰", styles['Heading']))
    story.append(Paragraph(data['insight']['summary'], styles['Korean']))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


def display_results(data: Dict, company_name: str):
    try:

        # 강점 분석 섹션
        time.sleep(1)
        st.toast("💪 강점 분석을 시작합니다")  # , icon="💡")
        metric_info = metrics_mapping.get(
            data['strength']['analysis_metric'], {})
        st.subheader(f"{metric_info.get('title', '')} (강점)")

        display_analysis_results(
            data['strength'],
            "강점 분석",
            metric_info,
            company_name,
            is_strength=True
        )
        st.markdown("---")

        # 약점 분석 섹션
        time.sleep(1)
        st.toast("🔍 약점 분석을 시작합니다")  # , icon="⚠️")
        metric_info = metrics_mapping.get(
            data['weakness']['analysis_metric'], {})
        st.subheader(f"{metric_info.get('title', '')} (약점)")

        display_analysis_results(
            data['weakness'],
            "약점 분석",
            metric_info,
            company_name,
            is_strength=False
        )

        st.markdown("---")

        # 최종 통찰 섹션
        time.sleep(1)
        st.toast("🎯 통찰 분석을 시작합니다")  # , icon="✨")
        st.subheader("🎯 최종 통찰")
        with st.expander("Insight Template", icon="🔮"):
            template_content = load_prompt("insight_template.txt")
            st.code(template_content, language='yaml')
        st.write_stream(stream_summary(data['insight']['summary']))

        # PDF 다운로드
        pdf_buffer = generate_pdf_report(data, company_name)
        st.divider()
        st.download_button(
            label="📥 보고서 다운로드",
            data=pdf_buffer,
            file_name=f"{company_name}_경영진단보고서.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"데이터 표시 중 오류가 발생했습니다: {str(e)}")


def display_metric_data(metrics_data: Dict, metric_info: Dict):
    """지표 데이터 표시 함수"""
    # 연간 지표 통합
    with st.expander("📈 연간 지표", expanded=False):
        for metric in metric_info.get('annual', []):
            if metric in metrics_data:
                # st.markdown(f"**{metric.replace('annual_', '')}**")
                st.dataframe(pd.DataFrame(
                    metrics_data[metric]), use_container_width=True, hide_index=True)

    # 월간 지표 통합
    with st.expander("📊 월간 지표", expanded=False):
        for metric in metric_info.get('monthly', []):
            if metric in metrics_data:
                # st.markdown(f"**{metric.replace('monthly_', '')}**")
                st.dataframe(pd.DataFrame(
                    metrics_data[metric]), use_container_width=True, hide_index=True)


def main():

    # 세션 스테이트 초기화
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = get_cache_manager()

    st.title("📊 AI경영진단보고서 베타 테스트")

    # 데이터 로드
    df = load_company_data()
    df = df.head(5)

    # 사이드바 구성
    with st.sidebar:
        # 기업 선택 섹션
        top_section = st.container(border=True)
        with top_section:
            st.header("🏢 기업 선택")
            selected_company = st.selectbox(
                "분석할 기업을 선택하세요",
                options=df['기업명'].tolist(),
                key="company_selector"
            )

            st.markdown("<br>" * 1, unsafe_allow_html=True)

            # 강점/약점 지표 선택
            st.subheader("📊 분석 지표 선택")
            strength_metric = st.selectbox(
                "강점 지표",
                options=list(metrics_mapping.keys()),
                format_func=lambda x: metrics_mapping[x]['title'],
                key="strength_selector"
            )

            weakness_metric = st.selectbox(
                "약점 지표",
                options=[m for m in metrics_mapping.keys() if m !=
                         strength_metric],
                format_func=lambda x: metrics_mapping[x]['title'],
                key="weakness_selector"
            )

            # 분석 시작 버튼
            if st.button("분석 시작", key='analyze_button'):
                st.session_state.analysis_started = True
                st.session_state.current_analysis = None

        st.divider()

        st.header("📝 피드백")

        # form으로 피드백 섹션 감싸기
        with st.form("feedback_form", clear_on_submit=False):

            analysis_type = st.selectbox(
                "분석 유형",
                options=["strength", "weakness", "insight"],
                format_func=lambda x: {
                    "strength": "강점 분석",
                    "weakness": "약점 분석",
                    "insight": "통찰 분석"
                }[x],
                key="analysis_type_select"
            )

            if "current_analysis_metric" not in st.session_state:
                st.session_state.current_analysis_metric = None

            # 현재 선택된 분석 유형의 지표 표시
            if analysis_type == "strength":
                st.session_state.current_analysis_metric = st.session_state.get(
                    'strength_selector')
                st.caption(
                    f"📊 선택된 지표: {metrics_mapping.get(st.session_state.current_analysis_metric, {}).get('title', '')}")
            elif analysis_type == "weakness":
                st.session_state.current_analysis_metric = st.session_state.get(
                    'weakness_selector')
                st.caption(
                    f"📊 선택된 지표: {metrics_mapping.get(st.session_state.current_analysis_metric, {}).get('title', '')}")
            else:
                strength_metric = st.session_state.get('strength_selector')
                weakness_metric = st.session_state.get('weakness_selector')
                st.session_state.current_analysis_metric = f"{strength_metric}/{weakness_metric}"
                st.caption(
                    f"📊 선택된 지표: 강점({metrics_mapping.get(strength_metric, {}).get('title', '')}) + 약점({metrics_mapping.get(weakness_metric, {}).get('title', '')})")

            feedback_type = st.radio(
                "피드백 유형",
                options=["개선사항", "오류신고", "기타"],
                horizontal=True,
                key="feedback_type_radio"  # 고유한 key 추가
            )

            feedback_text = st.text_area(
                "의견을 남겨주세요",
                placeholder="분석 결과나 사용성에 대한 의견을 자유롭게 작성해주세요.",
                key="feedback_text_area"  # 고유한 key 추가
            )

            # 폼 제출 버튼
            submitted = st.form_submit_button("피드백 제출")
            if submitted:
                if not st.session_state.analysis_started:
                    st.warning("먼저 분석을 시작해주세요!")
                    return

                if feedback_text:
                    if submit_feedback(
                        st.session_state.get("company_selector", ""),
                        feedback_type,
                        feedback_text,
                        analysis_type,
                        st.session_state.current_analysis_metric
                    ):
                        st.success("피드백이 성공적으로 저장되었습니다!")
                        st.balloons()
                else:
                    st.error("피드백 내용을 입력해주세요.")

    # 메인 화면 내용
    if not st.session_state.analyze_button:
        st.empty()
        st.markdown("""## 👋 AI 경영진단보고서 사용 안내""")
        col1, col2 = st.columns([6, 4])

        with col1:
            st.markdown("""
            ### 🎯 주요 기능
            1. **기업 분석**: 선택한 기업의 강점과 약점을 AI가 분석
            2. **맞춤형 통찰**: 재무/성장/수익성 등 다양한 지표 기반 분석
            3. **실시간 피드백**: 분석 결과에 대한 의견을 즉시 제출 가능

            ### 💫 사용 방법
            1. 왼쪽 사이드바에서 기업을 선택하세요
            2. 분석하고 싶은 강점과 약점 지표를 선택하세요
            3. '분석 시작' 버튼을 클릭하면 AI가 분석을 시작합니다

            ### 📊 분석 결과
            - **상세 데이터**: 연간/월간 지표 데이터 제공
            - **AI 분석**: 선택한 지표에 대한 심층 분석
            - **PDF 보고서**: 분석 결과를 보고서로 다운로드 가능
            
            ### 💾 데이터 관리
            - 🗄️ **SQLite 캐시 시스템**
                - 빠른 데이터 접근 및 조회
                - 분석 결과 자동 캐싱
            - 🔄 **데이터 갱신**
                - 매월 26일 자동 업데이트
                - 최신 데이터 유지
            - 📦 **영구 저장소**
                - 분석 이력 추적
                - 피드백 데이터 축적
            """)

        with col2:
            st.markdown("""
            ### 🔧 기술 스택

            #### 🚀 API 엔드포인트
            - 🎯 `/analyze`
                - 기업별 맞춤형 분석 실행
                - 강점/약점/통찰 결과 생성
            - 📊 `/query`
                - 기업별 상세 지표 데이터 조회
                - 연간/월간 데이터 제공
            - 📝 `/feedback`
                - 실시간 사용자 피드백 수집
                - 분석 결과별 의견 저장

            #### ⚙️ 주요 매개변수
            - 🏢 `company_name`: 분석 대상 기업 식별자
            - 💪 `strength_metric`: 강점 분석 지표 선택
            - 🔍 `weakness_metric`: 약점 분석 지표 선택
            - 📈 `analysis_type`: strength/weakness/insight 분석 유형

            > 💫 신규 분석 시, 최대 약 1분 정도 소요됩니다
            > 
            > ⚡ 이전 분석 결과는 자동으로 캐시되어 즉시 확인 가능합니다
            """)

        return

    # 분석 실행 및 결과 표시
    if st.session_state.analysis_started and st.session_state.company_selector:
        if st.session_state.current_analysis is None:
            with st.spinner(f"🔄 {st.session_state.company_selector} 분석 중..."):
                cached_data = check_cache(
                    company_name=st.session_state.company_selector,
                    strength_metric=st.session_state.strength_selector,
                    weakness_metric=st.session_state.weakness_selector
                )
                # st.write(cached_data)

                if cached_data and all(cached_data.get(k, {}).get('analysis_metric') for k in ['strength', 'weakness', 'insight']):
                    st.toast("💾 캐시된 데이터를 불러왔습니다")
                    st.session_state.current_analysis = cached_data
                else:
                    data = perform_analysis(
                        st.session_state.company_selector, None)
                    if data:
                        st.session_state.current_analysis = data

        # 저장된 분석 결과 표시
        if st.session_state.current_analysis:
            display_results(st.session_state.current_analysis,
                            st.session_state.company_selector)


if __name__ == "__main__":
    main()
