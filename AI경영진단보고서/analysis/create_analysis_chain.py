import os
from typing import Dict, Tuple, Any, AsyncIterator
import pandas as pd
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from utils import load_prompt
from utils.time_utils import get_access_time
from utils.logger import get_logger
from preprocessing import (
    preprocess_growth_data,
    preprocess_profitability_data,
    preprocess_partner_stability_data,
    preprocess_financial_stability_data
)
from core.config import settings

logger = get_logger(__name__)


class AnalysisChainError(Exception):
    """분석 체인 생성 및 실행 관련 에러"""
    pass


class TemplateError(AnalysisChainError):
    """템플릿 로딩 및 검증 관련 에러"""
    pass


class PreprocessingError(AnalysisChainError):
    """데이터 전처리 관련 에러"""
    pass


async def create_analysis_chain(
    analysis_metric: str,
    is_strength: bool,
    llm: ChatOpenAI,
    df_company_info: pd.DataFrame,
    company_name: str,
    access_time: pd.Timestamp
) -> RunnableLambda:
    """
    비동기 분석 체인 생성

    Args:
        analysis_metric: 분석할 지표 (growth, profitability, partner_stability)
        is_strength: 강점 분석 여부
        llm: 언어 모델
        df_company_info: 기업 데이터 DataFrame
        company_name: 기업명
        access_time: 분석 시점

    Raises:
        ValueError: 입력 데이터 검증 실패 시
        AnalysisChainError: 분석 체인 생성 실패 시
        TemplateError: 템플릿 로딩/검증 실패 시
        PreprocessingError: 전처리 함수 설정 실패 시
    """
    try:
        logger.info(
            f"[Chain] Creating analysis chain - "
            f"Indicator: {analysis_metric}, Company: {company_name}, "
            f"Is Strength: {is_strength}"
        )

        # Step 1: Validate Input
        if not isinstance(df_company_info, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if df_company_info.empty:
            raise ValueError("DataFrame cannot be empty")

        if not company_name or not isinstance(company_name, str):
            raise ValueError("Company name must be a non-empty string")

        if not isinstance(access_time, pd.Timestamp):
            raise ValueError("access_time must be a pandas Timestamp")

        if not isinstance(is_strength, bool):
            raise ValueError("is_strength must be a boolean")

        if not analysis_metric or not isinstance(analysis_metric, str):
            raise ValueError("Indicator must be a non-empty string")

        if analysis_metric not in settings.REQUIRED_TAGS:
            raise AnalysisChainError(
                f"Invalid analysis_metric '{analysis_metric}'. Must be one of: {list(settings.REQUIRED_TAGS.keys())}"
            )

        # Step 2: Load and Validate Template
        try:
            template_path = os.path.join(
                settings.PROMPTS_DIR, f"{analysis_metric}_template.txt")
            if not os.path.exists(template_path):
                raise TemplateError(
                    f"Template file not found: {template_path}")

            base_template = await asyncio.to_thread(load_prompt, template_path)
            if not base_template or not isinstance(base_template, str):
                raise TemplateError("Template content is invalid")

            logger.debug(
                f"[Template] Successfully loaded template for {analysis_metric}")

            # Validate template
            missing_tags = [
                tag for tag in settings.REQUIRED_TAGS[analysis_metric]
                if tag not in base_template
            ]
            if missing_tags:
                logger.warning(
                    f"[Warning] Template missing required tags: {missing_tags}"
                )

        except (FileNotFoundError, PermissionError) as e:
            error_msg = f"Cannot access template file: {str(e)}"
            logger.error(f"[Error] {error_msg}")
            raise TemplateError(error_msg) from e
        except Exception as e:
            error_msg = f"Template processing failed: {str(e)}"
            logger.error(f"[Error] {error_msg}")
            raise TemplateError(error_msg) from e

        # Step 3: Construct Full Prompt
        try:
            analysis_type = "Strength" if is_strength else "Weakness"
            additional_prompt = (
                "Perform analysis on positive side."
                if is_strength else
                "Perform analysis on negative side."
            )
            full_template = (
                f"[{analysis_type} Analysis for {analysis_metric}]\n"
                f"{base_template}\n\n{additional_prompt}"
            )

            prompt = PromptTemplate.from_template(
                full_template,
                partial_variables={"latest_year_month": settings.ACCESS_TIME}
            )
        except Exception as e:
            error_msg = f"Failed to construct prompt: {str(e)}"
            logger.error(f"[Error] {error_msg}")
            raise AnalysisChainError(error_msg) from e

        # Step 4: Create Analysis Chain
        try:
            if not isinstance(llm, ChatOpenAI):
                raise ValueError("Invalid language model provided")
            analysis_chain = prompt | llm | StrOutputParser()
        except Exception as e:
            error_msg = f"Failed to create analysis chain: {str(e)}"
            logger.error(f"[Error] {error_msg}")
            raise AnalysisChainError(error_msg) from e

        # Step 5: Prepare Preprocessing
        try:
            preprocess_func, data_keys = get_preprocess_and_keys(
                analysis_metric)
            if not callable(preprocess_func):
                raise PreprocessingError("Invalid preprocessing function")
            if not isinstance(data_keys, list) or not data_keys:
                raise PreprocessingError("Invalid data keys")
        except ValueError as e:
            error_msg = f"Failed to get preprocessing function: {str(e)}"
            logger.error(f"[Error] {error_msg}")
            raise PreprocessingError(error_msg) from e

        # Step 6: Create Final Chain
        def process_data(data: pd.DataFrame) -> Dict[str, Any]:
            """
            데이터 전처리 함수

            Raises:
                PreprocessingError: 전처리 실패 시
            """
            try:
                if not isinstance(data, pd.DataFrame):
                    raise ValueError("Input must be a DataFrame")

                processed_data = preprocess_func(
                    data, company_name, settings.ACCESS_TIME)
                if not isinstance(processed_data, dict):
                    raise ValueError("Preprocessing must return a dictionary")

                missing_keys = [
                    key for key in data_keys if key not in processed_data]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {missing_keys}")

                return {
                    key: processed_data[key] for key in data_keys
                }
            except Exception as e:
                error_msg = f"Data preprocessing failed: {str(e)}"
                logger.error(f"[Error] {error_msg}")
                raise PreprocessingError(error_msg) from e

        # 전처리와 분석을 연결한 최종 체인 반환
        return RunnableLambda(process_data) | analysis_chain

    except (ValueError, AnalysisChainError, TemplateError, PreprocessingError):
        raise
    except Exception as e:
        error_msg = f"Unexpected error in create_analysis_chain: {str(e)}"
        logger.error(f"[Error] {error_msg}")
        raise AnalysisChainError(error_msg) from e


def get_preprocess_and_keys(analysis_metric: str) -> Tuple[callable, list]:
    """전처리 함수와 데이터 키 매핑"""

    # 전처리 함수 매핑
    preprocess_functions = {
        'growth': preprocess_growth_data,
        'profitability': preprocess_profitability_data,
        'partner_stability': preprocess_partner_stability_data,
        'financial_stability': preprocess_financial_stability_data
    }

    if analysis_metric not in settings.REQUIRED_TAGS:
        raise ValueError(f"Unsupported analysis_metric: {analysis_metric}")

    # settings에서 정의된 태그에서 중괄호 제거하여 키 목록 생성
    keys = [tag.strip('{}') for tag in settings.REQUIRED_TAGS[analysis_metric]]

    return preprocess_functions[analysis_metric], keys
