# 1. 기본 모듈 임포트
# fmt: off
from typing import Dict, Any, Optional
import sqlite3
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
# fmt: on
# 2. 프로젝트 루트 경로 설정

# isort: skip
# 3. 로컬 모듈 임포트

# 데이터베이스 경로 설정
DB_PATH = os.path.join(project_root, "database", "sqlite_cache.db")
# %%

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
    }
}


def process_dataframe(metrics_data, metric):
    df = pd.DataFrame(metrics_data[metric])
    # NaN 값을 "-"로 변환
    df = df.fillna("-")
    return df


def submit_feedback(
    cache_key: str,
    feedback_type: str,
    feedback_text: str,
    analysis_type: str,
    indicator: str
):
    try:
        if not cache_key:
            st.error("분석 결과가 없습니다. 먼저 분석을 실행해주세요.")
            return False

        response = requests.post(
            "http://127.0.0.1:8000/feedback",
            json={
                "cache_key": cache_key,
                "feedback_type": feedback_type,
                "analysis_type": analysis_type,
                "indicator": indicator,
                "feedback_text": feedback_text
            },
            timeout=10
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# SQLite 캐시 확인 함수


def check_cache(company_name: str, strength_metric: str = None, weakness_metric: str = None) -> Optional[Dict]:
    """SQLite 캐시 확인"""
    try:
        today = datetime.now()
        current_month = f"{today.year}-{today.month:02d}"

        if not os.path.exists(DB_PATH):
            st.error(f"데이터베이스 파일을 찾을 수 없습니다: {DB_PATH}")
            return None

        with sqlite3.connect(DB_PATH) as conn:
            # 기본 쿼리 구성
            base_query = f"""
                SELECT cache_key, analysis_type, indicator, detailed_result, summary
                FROM {settings.SQLITE_TABLE_NAME}
                WHERE cache_key LIKE ?
                AND analysis_type IN ('strength', 'weakness', 'insight')
            """
            params = [f"{company_name}:{current_month}-%"]

            if today.day >= settings.THRESHOLD:
                base_query += " AND strftime('%d', created_at) >= ?"
                params.append(f"{settings.THRESHOLD:02d}")

            # 특정 지표에 대한 필터링
            if strength_metric or weakness_metric:
                base_query += """ AND (
                    (analysis_type = 'strength' AND indicator = ?)
                    OR (analysis_type = 'weakness' AND indicator = ?)
                    OR (analysis_type = 'insight' AND indicator = ?)
                )"""
                insight_indicator = f"{strength_metric}/{weakness_metric}" if strength_metric and weakness_metric else None
                params.extend(
                    [strength_metric, weakness_metric, insight_indicator])

            base_query += " ORDER BY created_at DESC"

            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()

            if not rows:
                return None

            result = {
                'cache_key': rows[0][0],
                'strength': {'indicator': None, 'detailed_result': None, 'summary': None},
                'weakness': {'indicator': None, 'detailed_result': None, 'summary': None},
                'insight': {'indicator': None, 'summary': None}
            }

            # 각 분석 유형별 최신 데이터 처리
            seen_types = set()
            for row in rows:
                cache_key, analysis_type, indicator, detailed, summary = row

                if analysis_type not in seen_types:
                    seen_types.add(analysis_type)
                    if analysis_type in ['strength', 'weakness']:
                        result[analysis_type] = {
                            'indicator': indicator,
                            'detailed_result': detailed,
                            'summary': summary
                        }
                    elif analysis_type == 'insight':
                        result['insight'] = {
                            'indicator': indicator,
                            'summary': summary
                        }

                if len(seen_types) == 3:  # strength, weakness, insight 모두 찾음
                    break

            return result

    except sqlite3.Error as e:
        st.error(f"데이터베이스 오류: {str(e)}")
        return None


@st.cache_data
def load_company_data():
    return pd.read_csv('overview_table.csv')


def perform_analysis(selected_company: str, strength_metric: str = None, weakness_metric: str = None) -> Optional[Dict]:
    """새로운 분석 수행"""

    try:
        response = requests.get(
            f"http://127.0.0.1:8000/analyze/{selected_company}/insight",
            params={
                "strength_metric": strength_metric,
                "weakness_metric": weakness_metric
            },
            timeout=60
        )
        response.raise_for_status()  # 400이면 HTTPError 발생, 성공 (200)이면, 동작 無

        if response.status_code == 200 and response.content:
            try:
                return response.json()
            except json.JSONDecodeError:
                # API가 성공적으로 실행되었다면 캐시에서 데이터를 가져옴
                cached_data = check_cache(selected_company,
                                          strength_metric=strength_metric,
                                          weakness_metric=weakness_metric)
                if cached_data:
                    return cached_data
                else:
                    st.error("분석은 완료되었으나 결과를 불러오는데 실패했습니다.")
                    return None
        else:
            st.error("API 응답이 비어있습니다.")
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

        with tab1:
            # 연간 데이터 표시
            metrics_response = requests.get(
                f"http://127.0.0.1:8000/query/{company_name}/{'strength' if is_strength else 'weakness'}",
                params={
                    'strength_metric': st.session_state.strength_selector if is_strength else None,
                    'weakness_metric': st.session_state.weakness_selector if not is_strength else None
                },
                timeout=60
            )
            if metrics_response.status_code == 200:
                metrics_data = metrics_response.json()
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
            if metrics_response.status_code == 200:
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
                indicator = data['indicator']
                template_file = f"{indicator}_template.txt"
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
    st.markdown(data['summary'])


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
        metric_info = metrics_mapping.get(data[section]['indicator'], {})

        # 연간 지표
        if metric_info.get('annual'):
            story.append(Paragraph("연간 지표", styles['Korean']))
            metrics_response = requests.get(
                f"http://127.0.0.1:8000/query/{company_name}/{section}",
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
        st.toast("💪 강점 분석을 시작합니다", icon="💡")
        metric_info = metrics_mapping.get(data['strength']['indicator'], {})
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
        st.toast("🔍 약점 분석을 시작합니다", icon="⚠️")
        metric_info = metrics_mapping.get(data['weakness']['indicator'], {})
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
        st.toast("🎯 통찰 분석을 시작합니다", icon="✨")
        st.subheader("🎯 최종 통찰")
        st.markdown(data['insight']['summary'])

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
    st.title("📊 AI경영진단보고서 베타 테스트")

    # 데이터 로드
    df = load_company_data()
    df = df.head(10)

    # 사이드바에 기업 목록 표시
    with st.sidebar:
        # 상단 섹션
        top_section = st.container(border=True)
        with top_section:
            st.header("🏢 기업 선택")
            selected_company = st.selectbox(
                "분석할 기업을 선택하세요",
                options=df['기업명'].tolist(),
                key="company_selector"
            )

            # 중간 여백
            st.markdown("<br>" * 1, unsafe_allow_html=True)

            # 강점/약점 지표 선택
            st.subheader("📊 분석 지표 선택")

            # 강점 지표 선택
            strength_metric = st.selectbox(
                "강점 지표",
                options=list(metrics_mapping.keys()),
                format_func=lambda x: metrics_mapping[x]['title'],
                key="strength_selector"
            )

            # 약점 지표 선택
            weakness_metric = st.selectbox(
                "약점 지표",
                options=[m for m in metrics_mapping.keys() if m !=
                         strength_metric],
                format_func=lambda x: metrics_mapping[x]['title'],
                key="weakness_selector"
            )

            analyze_button = st.button("분석 시작")

        # 중간 여백
        # st.markdown("<br>" * 5, unsafe_allow_html=True)
        st.divider()

        # 하단 피드백 섹션
        feedback_section = st.container(border=True)
        with feedback_section:
            st.header("📝 피드백")

            # 피드백 유형 선택
            feedback_type = st.radio(
                "피드백 유형",
                options=["개선사항", "오류신고", "기타"],
                horizontal=True
            )

            # 분석 유형 선택
            analysis_type = st.selectbox(
                "분석 유형",
                options=["strength", "weakness", "insight"],
                format_func=lambda x: {
                    "strength": "강점 분석",
                    "weakness": "약점 분석",
                    "insight": "통찰 분석"
                }[x]
            )

            # 현재 선택된 분석 유형의 지표 표시
            if analysis_type in ["strength", "weakness"]:
                metric_info = metrics_mapping.get(
                    strength_metric if analysis_type == "strength" else weakness_metric,
                    {}
                )
                st.caption(f"📊 선택된 지표: {metric_info.get('title', '')}")
            elif analysis_type == "insight":
                st.caption(
                    f"📊 선택된 지표: 강점({metrics_mapping.get(strength_metric, {}).get('title', '')}) + 약점({metrics_mapping.get(weakness_metric, {}).get('title', '')})")

            # 피드백 내용
            feedback_text = st.text_area(
                "의견을 남겨주세요",
                placeholder="분석 결과나 사용성에 대한 의견을 자유롭게 작성해주세요."
            )

            submit_button = st.button("피드백 제출")

            if submit_button and feedback_text:
                if "current_cache_key" not in st.session_state:
                    st.error("먼저 분석을 실행해주세요.")
                else:
                    current_indicator = (
                        strength_metric if analysis_type == "strength"
                        else weakness_metric if analysis_type == "weakness"
                        else f"{strength_metric}/{weakness_metric}"
                    )
                    if submit_feedback(
                        st.session_state.current_cache_key,
                        feedback_type,
                        feedback_text,
                        analysis_type,
                        current_indicator
                    ):
                        st.success("피드백이 성공적으로 저장되었습니다!")
                        st.balloons()

    # 메인 화면
    if selected_company and analyze_button:
        with st.spinner(f"🔄 {selected_company} 분석 중..."):
            cached_data = check_cache(selected_company,
                                      strength_metric=strength_metric,
                                      weakness_metric=weakness_metric)

            if cached_data:
                st.session_state.current_cache_key = cached_data['cache_key']
                st.toast("💾 캐시된 데이터를 불러왔습니다", icon="✅")
                display_results(cached_data, selected_company)
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("분석을 시작합니다...")
                st.toast("🚀 분석을 시작합니다", icon="ℹ️")
                progress_bar.progress(25)

                # API 호출 시 선택된 지표 전달
                data = perform_analysis(
                    selected_company,
                    strength_metric=strength_metric,
                    weakness_metric=weakness_metric
                )

                if data:
                    progress_bar.progress(75)
                    status_text.text("분석 결과를 저장하고 표시합니다...")
                    st.toast("📊 분석 결과를 저장하고 표시합니다", icon="ℹ️")
                    display_results(data, selected_company)
                    progress_bar.progress(100)
                    st.toast("✨ 분석이 완료되었습니다", icon="✅")
                    status_text.empty()


if __name__ == "__main__":
    main()
