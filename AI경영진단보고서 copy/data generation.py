# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:24:02 2024

@author: lht11
"""
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()
#%%


def extract_latest_month_and_previous_year(df):
    """
    DataFrame에서 최신 월과 전년 동기 월을 동적으로 추출하는 함수.
    """
    # 날짜 컬럼에서 최신 날짜와 그 전년도 같은 월을 추출
    latest_month = df["날짜"].max()  # 최신 날짜
    year, month = latest_month.split("-")
    previous_year_month = f"{int(year) - 1}-{month}"  # 전년도 같은 월

    return latest_month, previous_year_month


def calculate_three_year_average(values, months):
    """
    3년 평균을 계산하는 함수.
    """
    last_three_year_values = [
        values.get(month) for month in months if month in values
    ]
    return np.mean(last_three_year_values)


def preprocess_financial_data(df):
    """
    DataFrame을 받아서 각 재무 비율을 분석하기 위한 형태로 변환하는 함수.
    """
    processed_data = {}

    # 최신 월과 전년 동기 월 추출
    latest_year_month, previous_year_month = (
        extract_latest_month_and_previous_year(df)
    )

    for column in df.columns[1:]:  # 첫 번째 column인 '날짜'는 제외
        metric_name = column
        values = dict(zip(df["날짜"], df[column]))  # 날짜를 키로 하여 값 매핑
        processed_data[metric_name] = {}

        # 최신 데이터
        latest_data = values[latest_year_month]

        # 전년 동기 데이터
        prev_year_data = values[previous_year_month]
        change_vs_last_year = (
            ((latest_data - prev_year_data) / prev_year_data) * 100
            if prev_year_data != 0
            else 0
        )

        # 3년 평균 계산
        last_three_year_months = [
            f'{int(latest_year_month.split("-")[0]) - i}-{latest_year_month.split("-")[1]}'
            for i in range(1, 4)
        ]
        three_year_average = calculate_three_year_average(
            values, last_three_year_months
        )
        change_vs_three_year_average = (
            ((latest_data - three_year_average) / three_year_average) * 100
            if three_year_average != 0
            else 0
        )

        # 최근 12개월 데이터 추출
        recent_12_months = {
            month: values[month]
            for month in values
            if month
            >= f"{int(latest_year_month.split('-')[0]) - 1}-{latest_year_month.split('-')[1]}"
        }

        # JSON 형식으로 변환 및 열 이름 변경 ('전년 동기 데이터'로 열 이름 변경)
        processed_data[metric_name] = {
            "최신 데이터": latest_data,
            "전년 동기 데이터": prev_year_data,
            "전년 동기 대비 변화율 (%)": change_vs_last_year,
            "직전년도 3년 평균": three_year_average,
            "3년 평균 대비 변화율 (%)": change_vs_three_year_average,
            "최근 12개월 데이터": recent_12_months,
        }

    return {
        "latest_year_month": latest_year_month,
        "previous_year_month": previous_year_month,  # 동적으로 설정된 전년 동기 데이터
        "processed_data": processed_data,
    }


# JSON으로 변환하기 위해 DataFrame 처리
processed_financial_data = preprocess_financial_data(df_financial)

# 출력 예시
import pprint

pprint.pprint(processed_financial_data)
# %%


# 각 체인에서 사용될 데이터를 분리하여 할당하는 함수
def assign_chain_data(processed_financial_data):
    return {
        "growth_data": processed_financial_data["processed_data"][
            "매출 성장률"
        ],
        "profitability_data": {
            "영업이익률": processed_financial_data["processed_data"][
                "영업이익률"
            ],
            "순이익률": processed_financial_data["processed_data"]["순이익률"],
            "ROA": processed_financial_data["processed_data"]["ROA"],
        },
        "stability_data": {
            "부채비율": processed_financial_data["processed_data"]["부채비율"],
            "이자보상배율": processed_financial_data["processed_data"][
                "이자보상배율"
            ],
            "자기자본비율": processed_financial_data["processed_data"][
                "자기자본비율"
            ],
        },
        "productivity_data": {
            "노동생산성": processed_financial_data["processed_data"][
                "노동생산성"
            ],
            "자산회전율": processed_financial_data["processed_data"][
                "자산회전율"
            ],
            "재고자산회전율": processed_financial_data["processed_data"][
                "재고자산회전율"
            ],
        },
        "liquidity_data": {
            "유동비율": processed_financial_data["processed_data"]["유동비율"],
            "현금흐름비율": processed_financial_data["processed_data"][
                "현금흐름비율"
            ],
            "당좌비율": processed_financial_data["processed_data"]["당좌비율"],
        },
        "latest_year_month": processed_financial_data["latest_year_month"],
        "previous_year_month": processed_financial_data["previous_year_month"],
    }


# Preprocessed 데이터를 각 체인에 맞게 세팅
input_data = assign_chain_data(processed_financial_data)
# %%
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# .env 파일 경로 설정
env_path = os.path.join(
    r"C:\Users\lht11\Documents\GitHub\langchain-kr", ".env"
)  # your_folder_path는 .env 파일이 있는 경로
load_dotenv(dotenv_path=env_path)

os.environ["OPENAI_API_KEY"]


# %%
from langchain.prompts import PromptTemplate

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# OpenAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# System Prompt 정의
system_prompt = """
<Persona>
	당신은 중소기업 CEO를 위한 경영진단 보고서를 작성하는 AI입니다.
</Persona>

<Instruction>
	1. 모든 분석은 최신 월({latest_year_month})을 기준으로 전년 동기({previous_year_month})와 직전 12개월 데이터를 포함하여 수행하세요.
	2. 분석은 최대한 객관적인 사실에 기반하며, 변화율과 비교 지표를 명확히 제시하세요.
	3. 각각의 분석 체인은 사실 위주의 분석을 수행하고, 경영 전략 체인은 창의적인 제안을 제공하도록 해야 합니다.
  4. 각각의 분석 체인은 간결하게 3줄 이내로 요약하세요.	
</Instruction>
"""

# 공통 LLM 설정 (자유도 낮게 설정)
common_llm = ChatOpenAI(
    temperature=0.2, model_name="gpt-4o", openai_api_key=openai_api_key
)

# %%

# 성장성 Chain
growth_template = (
    system_prompt
    + """
다음은 최신 월({latest_year_month}) 기준으로 주어진 성장성 관련 매출 성장률, 자산 성장률, 순이익 성장률 데이터입니다:
{growth_data}
	1. **최신 월**({latest_year_month}) 기준으로 **전년 동기({previous_year_month})** 대비 변화율을 해석하세요.
	2. **최근 12개월** 동안의 월별 데이터를 분석하여 **계절적 패턴**과 **월별 추이**를 설명하세요.
"""
)

growth_prompt = PromptTemplate(
    input_variables=[
        "growth_data",
        "latest_year_month",
        "previous_year_month",
    ],
    template=growth_template,
)

growth_chain = LLMChain(
    llm=common_llm, prompt=growth_prompt, output_key="growth_analysis"
)
# %%
# 수익성 Chain
profitability_template = (
    system_prompt
    + """
다음은 최신 월({latest_year_month}) 기준으로 주어진 수익성 관련 영업이익률, 순이익률, ROA 데이터입니다:
{profitability_data}
	1. **최신 월**({latest_year_month}) 기준으로 **전년 동기({previous_year_month})** 대비 변화율을 해석하세요.
	2. **최근 12개월** 동안의 월별 데이터를 분석하여 **계절적 패턴**과 **월별 추이**를 설명하세요.
"""
)

profitability_prompt = PromptTemplate(
    input_variables=[
        "profitability_data",
        "latest_year_month",
        "previous_year_month",
    ],
    template=profitability_template,
)

profitability_chain = LLMChain(
    llm=common_llm,
    prompt=profitability_prompt,
    output_key="profitability_analysis",
)

# %%
# 재무 안정성 Chain
stability_template = (
    system_prompt
    + """
다음은 최신 월({latest_year_month}) 기준으로 주어진 재무 안정성 관련 부채비율, 이자보상배율, 자기자본비율 데이터입니다:
{stability_data}
	1. **최신 월**({latest_year_month}) 기준으로 **전년 동기({previous_year_month})** 대비 변화율을 해석하세요.
	2. **최근 12개월** 동안의 월별 데이터를 분석하여 **계절적 패턴**과 **월별 추이**를 설명하세요.
"""
)

stability_prompt = PromptTemplate(
    input_variables=[
        "stability_data",
        "latest_year_month",
        "previous_year_month",
    ],
    template=stability_template,
)

stability_chain = LLMChain(
    llm=common_llm, prompt=stability_prompt, output_key="stability_analysis"
)
# %%
# 생산성 Chain
productivity_template = (
    system_prompt
    + """
다음은 최신 월({latest_year_month}) 기준으로 주어진 생산성 관련 노동생산성, 자산회전율, 재고자산회전율 데이터입니다:
{productivity_data}
	1. **최신 월**({latest_year_month}) 기준으로 **전년 동기({previous_year_month})** 대비 변화율을 해석하세요.
	2. **최근 12개월** 동안의 월별 데이터를 분석하여 **계절적 패턴**과 **월별 추이**를 설명하세요.
"""
)

productivity_prompt = PromptTemplate(
    input_variables=[
        "productivity_data",
        "latest_year_month",
        "previous_year_month",
    ],
    template=productivity_template,
)

productivity_chain = LLMChain(
    llm=common_llm,
    prompt=productivity_prompt,
    output_key="productivity_analysis",
)
# %%
# 유동성 Chain
liquidity_template = (
    system_prompt
    + """
다음은 최신 월({latest_year_month}) 기준으로 주어진 유동성 관련 유동비율, 현금흐름비율, 당좌비율 데이터입니다:
{liquidity_data}
	1. **최신 월**({latest_year_month}) 기준으로 **전년 동기({previous_year_month})** 대비 변화율을 해석하세요.
	2. **최근 12개월** 동안의 월별 데이터를 분석하여 **계절적 패턴**과 **월별 추이**를 설명하세요.
"""
)

liquidity_prompt = PromptTemplate(
    input_variables=[
        "liquidity_data",
        "latest_year_month",
        "previous_year_month",
    ],
    template=liquidity_template,
)

liquidity_chain = LLMChain(
    llm=common_llm, prompt=liquidity_prompt, output_key="liquidity_analysis"
)
# %%
# 경영 솔루션 제안 LLM (자유도 높게 설정)
solution_llm = ChatOpenAI(
    temperature=1.2, model_name="gpt-4o"
)  # 창의적인 접근 허용

solution_template = (
    system_prompt
    + """
다음은 기업의 최신 재무 성과 분석 결과입니다:
	1. 성장성 분석: {growth_analysis}
	2. 수익성 분석: {profitability_analysis}
	3. 재무 안정성 분석: {stability_analysis}
	4. 생산성 분석: {productivity_analysis}
	5. 유동성 분석: {liquidity_analysis}

이 결과를 바탕으로 기업의 전반적인 재무 상태를 평가하고, 향후 성장을 위한 경영 전략을 제안하세요.
특히 비용 효율성, 자본 활용, 리스크 관리, 그리고 새로운 시장 기회를 고려한 전략을 포함하세요.
"""
)

solution_prompt = PromptTemplate(
    input_variables=[
        "growth_analysis",
        "profitability_analysis",
        "stability_analysis",
        "productivity_analysis",
        "liquidity_analysis",
    ],
    template=solution_template,
)

solution_chain = LLMChain(
    llm=solution_llm, prompt=solution_prompt, output_key="solution"
)

# %%
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[
        growth_chain,
        profitability_chain,
        stability_chain,
        productivity_chain,
        liquidity_chain,
        solution_chain,  # 최종 경영 솔루션 및 제언 체인
    ],
    input_variables=[
        "growth_data",
        "profitability_data",
        "stability_data",
        "productivity_data",
        "liquidity_data",
        "latest_year_month",
        "previous_year_month",
    ],
    output_variables=[
        "growth_analysis",
        "profitability_analysis",
        "stability_analysis",
        "productivity_analysis",
        "liquidity_analysis",
        "solution",
    ],
)

# 체인 실행
result = overall_chain.invoke(input_data)

# 결과 출력
print("Growth Analysis:", result["growth_analysis"])
print("Profitability Analysis:", result["profitability_analysis"])
print("Stability Analysis:", result["stability_analysis"])
print("Productivity Analysis:", result["productivity_analysis"])
print("Liquidity Analysis:", result["liquidity_analysis"])
print("Solution:", result["solution"])
