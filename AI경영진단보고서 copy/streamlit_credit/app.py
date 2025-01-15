# fmt: off
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict
import xgboost as xgb


# 페이지 설정
st.set_page_config(
    page_title="AI 신용분석 리포트",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
DB_PATH = os.path.join(project_root, "database", "sqlite_credit_cache.db")
from core.config import settings
from utils import load_prompt
# fmt: on
# SQLite 연결 함수


@st.cache_data(show_spinner=False)
def fetch_credit_query(company_index: str, analysis_type: str) -> dict:
    """
    Fetch credit_query results from API and cache them.
    """
    try:

        payload = {
            "company_index": company_index,
            "analysis_type": analysis_type
        }

        response = requests.post(
            f"http://127.0.0.1:8000/credit_query/",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from credit_query API: {e}")
        return {}


# 게이지 차트 함수
@st.cache_resource
def create_gauge_chart(current_analysis):
    proba = current_analysis["proba"]
    grade = current_analysis["grade"]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        title={'text': f"Credit Grade: {grade}"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'steps': [
                {'range': [0, 5], 'color': "lightgreen", 'name': 'AAA'},
                {'range': [5.01, 10], 'color': "green", 'name': 'AA'},
                {'range': [10.01, 15], 'color': "yellowgreen", 'name': 'A'},
                {'range': [15.01, 20], 'color': "yellow", 'name': 'BBB'},
                {'range': [20.01, 30], 'color': "orange", 'name': 'BB'},
                {'range': [30.01, 40], 'color': "darkorange", 'name': 'B'},
                {'range': [40.01, 50], 'color': "orangered", 'name': 'CCC'},
                {'range': [50.01, 60], 'color': "red", 'name': 'CC'},
                {'range': [60.01, 100], 'color': "darkred", 'name': 'C'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': proba
            }
        }
    ))

    fig_gauge.update_layout(
        width=400,
        height=300,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig_gauge, proba, grade

# 가로 막대 그래프 함수


@st.cache_resource
def create_bar_charts(current_analysis):
    top_increasing = current_analysis["top_increasing"]
    top_decreasing = current_analysis["top_decreasing"]

    # 상승 요인 데이터
    increasing_features = [d["label"] for d in top_increasing]
    increasing_values = [d["shap_value"] for d in top_increasing]

    # 하락 요인 데이터
    decreasing_features = [d["label"] for d in top_decreasing]
    decreasing_values = [-abs(d["shap_value"]) for d in top_decreasing]

    # 데이터를 합침
    features = decreasing_features + increasing_features
    values = decreasing_values + increasing_values

    # 정렬: 값 기준으로 (오름차순) 정렬
    sorted_data = sorted(zip(features, values), key=lambda x: x[1])
    sorted_features, sorted_values = zip(*sorted_data)

    # 툴팁 데이터 추가 (SHAP 값만 표시)
    tooltips = [f"{value:.2f}" for value in sorted_values]

    # 그래프 생성
    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker=dict(
            color=["red" if v < 0 else "blue" for v in sorted_values]
        ),
        hoverinfo="text",  # 툴팁 활성화
        text=tooltips  # 툴팁 텍스트 (특성 제외, SHAP 값만)
    ))

    fig_bar.update_layout(
        # title="Feature Impact on Default Probability",
        xaxis=dict(
            title="부도에 미치는 영향 (%p)",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2
        ),
        yaxis=dict(
            title="Features",
            automargin=True
        ),
        height=500,
        width=700,
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig_bar


# Streamlit 화면 출력


def render_current_analysis(current_analysis):
    """
    Render the current analysis in a given column using cached charts.
    """
    # 게이지 차트
    st.subheader("Credit Probability and Grade")
    fig_gauge, proba, grade = create_gauge_chart(current_analysis)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 부도 확률 및 신용등급 정보
    st.info(f"당사의 부도확률은 {proba}%이고, 신용등급은 {grade}에 해당합니다.")

    st.divider()

    # 가로 막대 그래프
    st.subheader("Top Increasing and Decreasing Factors")
    fig_bar = create_bar_charts(current_analysis)
    st.plotly_chart(fig_bar, use_container_width=True)


# Streamlit 화면 출력 함수
def render_hypothetical_analysis(hypothetical_analysis, current_analysis):
    for scenario in hypothetical_analysis["scenarios"]:
        st.subheader(f"{scenario['id']} : {scenario['label']}")

        # Overview Chart (Vertical Bar Chart)
        overview_chart = create_scenario_overview_chart(scenario)
        st.plotly_chart(overview_chart, use_container_width=True)

        # Current vs. Hypothetical Chart (Horizontal Bar Chart)
        comparison_chart = create_current_vs_hypothetical_chart(
            current_analysis, scenario
        )
        st.plotly_chart(comparison_chart, use_container_width=True)

        # Divider for separation
        st.divider()


# 1. 시나리오 개요: Before vs. After (Vertical Bar Chart)
@st.cache_data
def create_scenario_overview_chart(scenario: Dict):
    before = scenario["before"]
    after = scenario["after"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before",
        x=[scenario["label"]],
        y=[before["probability"]],
        marker=dict(color="red"),
        text=[f"Grade: {before['grade']}"],
        textposition="outside",
        hoverinfo="text"
    ))
    fig.add_trace(go.Bar(
        name="After",
        x=[scenario["label"]],
        y=[after["new_probability"]],
        marker=dict(color="green"),
        text=[f"Grade: {after['new_grade']}"],
        textposition="outside",
        hoverinfo="text"
    ))

    fig.update_layout(
        barmode="group",
        title="Default Probability: Before vs. After",
        xaxis=dict(title="Scenario"),
        yaxis=dict(title="Default Probability (%)", range=[0, 100]),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


# 2. Current vs. Hypothetical Comparison (Horizontal Bar Chart)
@st.cache_data
def create_current_vs_hypothetical_chart(current_analysis: Dict, scenario: Dict):
    current_features = {f["feature"]: f["shap_value"]
                        for f in current_analysis["feature_impacts"]}
    hypothetical_features = scenario["after"]["top_influenced_features"]

    labels = [f["label"] for f in hypothetical_features]
    current_values = [current_features.get(
        f["feature"], 0) for f in hypothetical_features]
    new_values = [f["new_shap_value"] for f in hypothetical_features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current SHAP Value",
        x=current_values,
        y=labels,
        orientation="h",
        marker=dict(color="blue"),
        text=[f"{v:.2f}" for v in current_values],
        textposition="auto",
        hoverinfo="text"
    ))
    fig.add_trace(go.Bar(
        name="New SHAP Value",
        x=new_values,
        y=labels,
        orientation="h",
        marker=dict(color="green"),
        text=[f"{v:.2f}" for v in new_values],
        textposition="auto",
        hoverinfo="text"
    ))

    fig.update_layout(
        barmode="group",
        title="Feature Impact: Current vs. Hypothetical",
        xaxis=dict(title="부도에 미치는 영향 (%p)"),
        yaxis=dict(title="Feature"),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def get_cached_analysis(company_index: str, analysis_type: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        today = datetime.now()
        current_month = f"{today.year}-{today.month:02d}"

        query = """
        SELECT detailed_analysis, final_report
        FROM CREDIT_CONSULTING
        WHERE company_name = ? 
        AND analysis_type = ?
        AND strftime('%Y-%m', created_at) = ?
        ORDER BY created_at DESC
        LIMIT 1
        """

        cursor.execute(
            query, (f"Company_{company_index}", analysis_type, current_month))
        result = cursor.fetchone()
        conn.close()

        return result
    except Exception as e:
        st.error(f"데이터베이스 조회 중 오류 발생: {str(e)}")
        return None


@st.cache_data
def sample_generate():
    # 데이터 로드 및 전처리
    X_test = pd.read_pickle(os.path.join(settings.DATA_PATH, 'X_test.pkl'))
    X_test.reset_index(inplace=True, drop=True)
    y_test = pd.read_pickle(os.path.join(settings.DATA_PATH, 'y_test.pkl'))
    y_test.reset_index(inplace=True, drop=True)

    # 모델 로드
    booster = xgb.Booster()
    booster.load_model(os.path.join(settings.DATA_PATH, 'best_model.json'))
    model = xgb.XGBClassifier()
    model._Booster = booster
    model.n_classes_ = len(np.unique(y_test))

    # 예측 및 샘플링
    probas = model.predict_proba(X_test)[:, 1]
    result_dfs = generate_sample_data(X_test, y_test, probas)
    return result_dfs[0], result_dfs[1], probas  # 세 개의 값을 반환


def generate_sample_data(X_test, y_test, probas):
    # 부도 기업 (true positive) 샘플링
    true_positive_mask = (y_test.values == 1) & (probas >= 0.5)
    true_positive_indices = X_test.index[true_positive_mask]
    result_df = pd.DataFrame({
        'index': true_positive_indices,
        'actual': y_test[true_positive_indices],
        'predicted_proba': probas[true_positive_mask]
    }).sort_values('predicted_proba', ascending=False).sample(n=5, random_state=42)

    # 생존 기업 (true negative) 샘플링
    true_negative_mask = (y_test.values == 0) & (probas < 0.5)
    true_negative_indices = X_test.index[true_negative_mask]
    result_df_0 = pd.DataFrame({
        'index': true_negative_indices,
        'actual': y_test[true_negative_indices],
        'predicted_proba': probas[true_negative_mask]
    }).sort_values('predicted_proba', ascending=False).sample(n=5, random_state=40)

    return result_df, result_df_0


def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.preview_data = sample_generate()  # 캐시된 데이터 로드
        st.session_state.analysis_requested = False
        st.session_state.status_type = '부도 기업'
        st.session_state.company_index = "unknown"  # 초기값 설정
        st.session_state.analysis_type = "unknown"  # 초기값 설정
        st.session_state.analysis_result = None

        # 기업 상태 변경 시, 분석 요청 초기화
        if "status_type" in st.session_state:
            st.session_state.analysis_requested = False


def submit_feedback(feedback_type, feedback_text):
    if not feedback_text:
        st.error("피드백 내용을 입력해주세요.")
        return

    company_index = str(st.session_state.get("company_index", "unknown"))
    analysis_type = st.session_state.get("analysis_type", "unknown")

    # # 디버깅: 전송 데이터 확인
    # feedback_data = {
    #     "company_name": company_index,
    #     "feedback_type": feedback_type,
    #     "analysis_type": analysis_type,
    #     "feedback_text": feedback_text,
    # }
    # st.write("Feedback Data:", feedback_data)  # 디버깅 출력

    try:

        payload = {
            "company_name": company_index,
            "feedback_type": feedback_type,
            "analysis_type": analysis_type,
            "feedback_text": feedback_text
        }

        with st.spinner("피드백을 제출 중입니다..."):
            response = requests.post(
                "http://127.0.0.1:8000/credit_feedback/",
                json=payload
            )
            response.raise_for_status()  # HTTP 에러 검사
            if response.status_code == 200:
                st.success("피드백이 성공적으로 저장되었습니다!")
                st.balloons()
            else:
                st.error(f"피드백 저장에 실패했습니다. 상태 코드: {response.status_code}")
    except requests.exceptions.HTTPError as e:
        st.error(f"서버 오류: {e.response.status_code} - {e.response.reason}")
    except requests.exceptions.RequestException as e:
        st.error(f"피드백 전송 중 오류가 발생했습니다: {str(e)}")


def main():

    initialize_session_state()

    # 사이드바 구성
    with st.sidebar:
        st.title("📈 신용컨설팅 설정")

        # form 밖에서 radio 버튼 구성
        status_type = st.radio(
            "기업 상태",
            options=["부도 기업", "생존 기업"],
            format_func=lambda x: "부도 기업" if x == "부도 기업" else "생존 기업",
            key='status_type',
            on_change=lambda: st.session_state.update(
                {"analysis_requested": False})
        )

        # 분석 설정을 위한 form
        with st.form("analysis_form"):  # 고유 키: "analysis_form"
            result_df, result_df_0, probas = st.session_state.preview_data
            company_indices = result_df['index'].tolist(
            ) if status_type == "부도 기업" else result_df_0['index'].tolist()

            selected_company = st.selectbox(
                "기업 선택",
                options=company_indices,
                format_func=lambda x: f"Company_{x} (예측확률: {probas[x]:.2%})"
            )

            analysis_type = st.radio(
                "분석 방식",
                options=["current", "hypothetical"],
                format_func=lambda x: "현황 분석" if x == "current" else "시뮬레이션 분석"
            )

            submitted = st.form_submit_button("분석 시작")  # `key` 제거
            if submitted:
                st.session_state.analysis_requested = True
                st.session_state.company_index = str(selected_company)
                st.session_state.analysis_type = analysis_type
                st.session_state.analysis_result = None

        if not st.session_state.get('analysis_requested', False):
            st.info("👆 분석을 시작하려면 '분석 시작' 버튼을 클릭하세요.")

        st.divider()
        st.header("📝 피드백")

        # 피드백 폼은 항상 표시
        with st.form("credit_feedback_form", clear_on_submit=True):  # 고유 키: "credit_feedback_form"
            feedback_type = st.radio(
                "피드백 유형",
                options=["개선사항", "오류신고", "기타"],
                horizontal=True,
                key="feedback_type"
            )

            feedback_text = st.text_area(
                "의견을 남겨주세요",
                placeholder="분석 결과나 사용성에 대한 의견을 자유롭게 작성해주세요.",
                key="feedback_text"
            )

            submitted_feedback = st.form_submit_button("피드백 제출")  # `key` 제거

            # 제출 시 동작
            if submitted_feedback:
                # if not st.session_state.get("analysis_requested", False):
                #     st.warning("분석 없이 피드백을 제출했습니다. 분석을 시작한 후 제출하는 것을 권장합니다.")
                submit_feedback(feedback_type, feedback_text)

    # 최종 보고서 표시 부분
    if not st.session_state.analysis_requested:
        st.title("🤖 AI기반 신용분석 리포트")
        st.markdown("""
        ### 👋 사용 방법
        1. 왼쪽 사이드바에서 분석하고 싶은 기업의 상태를 선택하세요
        2. 기업 목록에서 특정 기업을 선택하세요
        3. 분석 유형(현황/시뮬레이션)을 선택하세요
        4. '분석 시작' 버튼을 클릭하세요
        """)

        st.write('')
        st.divider()
        st.write('')

        st.success("📢 현황 분석과 시뮬레이션 분석을 통해 기업의 신용 상태를 정확히 진단하고 개선 전략을 수립할 수 있습니다.")

        # 두 개의 컬럼으로 현황 분석 및 시뮬레이션 분석 설명
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 **현황 분석**")
            st.write("""
            **현황 분석**은 특정 기업의 현재 신용 상태를 분석합니다.  
            부도 확률과 신용 등급을 확인하고, 부도 위험에 가장 큰 영향을 미치는 요인들을 분석합니다.  
            """)
            st.markdown("""
            **API 정보**:  
            - **엔드포인트**: `/credit_analysis`  
            - **파라미터**:  
                - `company_index`: 기업 인덱스 (숫자) 
                - `analysis_type`: `"current"`  
            """)

        with col2:
            st.subheader("🔄 **시뮬레이션 분석**")
            st.write("""
            **시뮬레이션 분석**은 특정 요인을 변화시켰을 때 신용 상태가 어떻게 개선되는지를 평가합니다.  
            가정된 시나리오를 통해 부도 확률 및 신용 등급 변화를 예측하고 전략적 인사이트를 제공합니다.  
            """)
            st.markdown("""
            **API 정보**:  
            - **엔드포인트**: `/credit_analysis`  
            - **파라미터**:  
                - `company_index`: 기업 인덱스 (숫자)
                - `analysis_type`: `"hypothetical"`  
            """)

    else:
        st.title("📑 최종 보고서(안)")

        with st.status("신용분석 진행 중...", expanded=True) as status:
            st.toast("🔍 분석 프로세스를 시작합니다")
            st.write("캐시 확인 중...")
            cached_result = get_cached_analysis(
                st.session_state.company_index,
                st.session_state.analysis_type
            )

            if cached_result:
                st.toast("💾 캐시된 결과를 불러옵니다")
                st.write("💾 캐시된 분석 결과를 불러옵니다")
                detailed_analysis, final_report = cached_result
                result_dict = {
                    'detailed_analysis': detailed_analysis,
                    'final_report': final_report
                }
            else:
                st.toast("🔄 새로운 분석을 시작합니다")
                st.write("🔄 새로운 분석을 시작합니다")
                try:
                    payload = {
                        "company_index": st.session_state.company_index,
                        "analysis_type": st.session_state.analysis_type
                    }

                    response = requests.post(
                        f"http://127.0.0.1:8000/credit_analysis/",
                        json=payload
                    )
                    response.raise_for_status()
                    result_dict = response.json()
                    st.toast("✨ 분석이 완료되었습니다")
                except requests.exceptions.HTTPError as e:
                    st.error(
                        f"서버 오류: {e.response.status_code} - {e.response.reason}")
                    return
                except requests.exceptions.RequestException as e:
                    st.error(f"요청 처리 중 오류 발생: {str(e)}")
                    return

        expander = st.expander("분석 내용", expanded=False)
        with expander:
            tab1, tab2 = st.tabs(["📊 상세 분석", "📝 분석 템플릿"])

            with tab1:
                # st.write(result_dict.keys())
                st.markdown(result_dict['detailed_analysis'])

            with tab2:
                try:
                    # 현재 분석 유형에 따른 템플릿 파일명 설정
                    if st.session_state.analysis_type == "current":
                        template_file = "current_credit_template.txt"
                    else:
                        template_file = "hypothetical_credit_template.txt"

                    # 템플릿 내용 로드 및 표시
                    template_content = os.path.join(
                        settings.PROMPTS_DIR, template_file)
                    st.code(load_prompt(template_content), language='yaml')

                except FileNotFoundError as e:
                    st.error(f"템플릿 파일을 찾을 수 없습니다: {str(e)}")

        # with st.expander("📜 상세 분석 보기"):
        #     if "detailed_analysis" in result_dict and result_dict["detailed_analysis"]:
        #         st.write(result_dict["detailed_analysis"])
        #     else:
        #         st.warning("상세 분석 결과가 없습니다.")

        # 2개의 열로 구성
        col1, col2 = st.columns(2)

        # 왼쪽 열: 스트리밍 보고서
        with col1:
            # st.subheader("📜 요약 보고서")
            st.subheader('')

            def stream_report():
                for line in result_dict['final_report'].split('\n'):
                    yield line + '\n'
                    time.sleep(0.2)

            st.write_stream(stream_report())
            status.update(label="분석 완료!", state="complete", expanded=False)

        # 오른쪽 열: 추가 분석 결과 표시
        with col2:
            # st.subheader("📊 시각화 결과")
            # st.write("Fetching additional analysis from credit_query API...")
            st.subheader("")

            if st.session_state.analysis_type == "current":
                # current_analysis API 호출 (캐싱)
                current_analysis = fetch_credit_query(
                    company_index=st.session_state.company_index,
                    analysis_type="current"
                ).get("current_analysis", None)

                if current_analysis:
                    # render_current_analysis로 차트와 정보를 렌더링
                    render_current_analysis(current_analysis)
                else:
                    st.error("No current analysis data available.")
            else:
                # Hypothetical 분석 데이터 가져오기
                hypothetical_analysis = fetch_credit_query(
                    company_index=st.session_state.company_index,
                    analysis_type="hypothetical"
                ).get("hypothetical_analysis", None)

                if not hypothetical_analysis:
                    st.error("No hypothetical analysis data available.")
                else:
                    # Current 분석 데이터 가져오기 (캐시 또는 API 호출)
                    current_analysis = fetch_credit_query(
                        company_index=st.session_state.company_index,
                        analysis_type="current"
                    ).get("current_analysis", None)

                    if not current_analysis:
                        st.warning(
                            "Current analysis data not available. Fetching via API...")
                        try:
                            current_analysis = fetch_credit_query(
                                company_index=st.session_state.company_index,
                                analysis_type="current"
                            ).get("current_analysis", None)
                        except Exception as e:
                            st.error(
                                f"Error fetching current analysis data: {e}")
                            current_analysis = None

                    if current_analysis:
                        render_hypothetical_analysis(
                            hypothetical_analysis, current_analysis)
                    else:
                        st.error(
                            "Current analysis data could not be retrieved, comparison is unavailable.")
            # else:
            #     st.info("No additional analysis available for non-hypothetical type.")

            # # credit_query API 호출
            # try:
            #     analysis_type = st.session_state.analysis_type
            #     payload = {
            #         "company_index": st.session_state.company_index,
            #         "analysis_type": analysis_type
            #     }

            #     response = requests.post(
            #         f"http://127.0.0.1:8000/credit_query/",
            #         json=payload
            #     )
            #     response.raise_for_status()
            #     query_result = response.json()

            #     if analysis_type == "current":
            #         st.write("### Current Analysis")
            #         st.json(query_result.get("current_analysis", {}))
            #     elif analysis_type == "hypothetical":
            #         st.write("### Hypothetical Analysis")
            #         st.json(query_result.get("hypothetical_analysis", {}))

            # except requests.exceptions.HTTPError as e:
            #     st.error(
            #         f"credit_query 서버 오류: {e.response.status_code} - {e.response.reason}")
            # except requests.exceptions.RequestException as e:
            #     st.error(f"credit_query 요청 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
