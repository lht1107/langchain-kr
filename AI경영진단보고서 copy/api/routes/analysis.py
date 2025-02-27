import asyncio
import os
from typing import AsyncIterator, Dict, Optional, Tuple
from enum import Enum
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from core.dependencies import get_llm_clients, get_cache
from core.config import settings
from core.cache import CacheManager
from utils.logger import get_logger
from utils.load_prompt import load_prompt
from analysis import (
    create_analysis_chain,
    merge_analysis_results,
    create_summary_chain,
)

# Initialize logger
logger = get_logger(__name__)


class AnalysisType(str, Enum):
    """
    Enum class to define the types of analysis available.
    """
    STRENGTH = "strength"
    WEAKNESS = "weakness"
    INSIGHT = "insight"


router = APIRouter(
    prefix="/analyze",
    tags=["analysis"]
)


def load_company_data():
    """
    Load company data from a sample .parquet file.

    Returns:
        pd.DataFrame: The loaded company data.
    """
    sample_data_path = os.path.join(settings.DATA_PATH, "sample.parquet")
    return pd.read_parquet(sample_data_path)


@router.post("/", include_in_schema=False)
async def analyze_company(
    request: Request,
    analysis_request: Dict,
    cache: CacheManager = Depends(get_cache),
    llm_clients: Tuple = Depends(get_llm_clients),
) -> StreamingResponse:
    """
    Analyze a company's strengths, weaknesses, or insights based on input parameters.

    Args:
        request (Request): The HTTP request object.
        analysis_request (Dict): The analysis parameters (company name, type, etc.).
        cache (CacheManager): Dependency for caching.
        llm_clients (Tuple): Dependency for LLM clients.

    Returns:
        StreamingResponse: A streaming response with the analysis result.
    """
    common_llm, summary_llm, insight_llm, _ = llm_clients

    # Load company data once for the request
    company_data = await asyncio.to_thread(load_company_data)

    async def generate_analysis() -> AsyncIterator[str]:
        """
        Generate the analysis as a streaming response.

        Yields:
            str: The result of the analysis.
        """
        try:
            company_name = analysis_request["company_name"]
            analysis_type = AnalysisType(analysis_request["analysis_type"])
            strength_metric = analysis_request.get("strength_metric")
            weakness_metric = analysis_request.get("weakness_metric")

            # Check for cached data
            cached_data = await cache.get(
                company_name,
                analysis_type.value,  # AnalysisType enum's value
                strength_metric if analysis_type == AnalysisType.STRENGTH
                else weakness_metric if analysis_type == AnalysisType.WEAKNESS
                else f"{strength_metric}/{weakness_metric}"
            )

            if cached_data:
                insight_exists = cached_data["insight"].get(
                    "analysis_metric") == f"{strength_metric}/{weakness_metric}"
                strength_exists = cached_data["strength"].get(
                    "analysis_metric") == strength_metric
                weakness_exists = cached_data["weakness"].get(
                    "analysis_metric") == weakness_metric

                # Return cached results if available
                if analysis_type == AnalysisType.INSIGHT:
                    if insight_exists:
                        logger.info("\n[Cache] Using cached insight analysis.")
                        yield cached_data["insight"]["summary"]
                        return
                    elif strength_exists and weakness_exists:
                        logger.info(
                            "\n[Cache] Both metrics exist. Performing insight analysis.")
                        result = await perform_insight_analysis(company_name, cached_data, insight_llm, cache)
                        yield result["insight"]["summary"]
                        return

                elif analysis_type == AnalysisType.STRENGTH and strength_exists:
                    logger.info("\n[Cache] Using cached strength analysis.")
                    yield cached_data["strength"]["summary"]
                    return

                elif analysis_type == AnalysisType.WEAKNESS and weakness_exists:
                    logger.info("\n[Cache] Using cached weakness analysis.")
                    yield cached_data["weakness"]["summary"]
                    return

            # Perform new analysis if no valid cached results exist
            result = await perform_new_analysis(
                company_name,
                analysis_type,
                strength_metric,
                weakness_metric,
                cached_data,
                company_data,
                common_llm,
                summary_llm,
                insight_llm,
                cache
            )
            yield result[analysis_type.value]["summary"]

        except Exception as e:
            logger.error(f"\n[Error] Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        generate_analysis(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def perform_new_analysis(
    company_name: str,
    analysis_type: AnalysisType,
    strength_metric: Optional[str],
    weakness_metric: Optional[str],
    cached_data: Optional[Dict],
    company_data: pd.DataFrame,
    common_llm,
    summary_llm,
    insight_llm,
    cache: CacheManager,
) -> Dict:
    """
    Perform a new analysis when no valid cached data is available.

    Args:
        company_name (str): The name of the company.
        analysis_type (AnalysisType): The type of analysis.
        strength_metric (Optional[str]): The strength metric.
        weakness_metric (Optional[str]): The weakness metric.
        cached_data (Optional[Dict]): Cached data if available.
        company_data (pd.DataFrame): Dataframe containing company information.
        common_llm, summary_llm, insight_llm: LLM models.
        cache (CacheManager): Cache manager instance.

    Returns:
        Dict: The analysis result.
    """
    try:
        company_info = company_data[company_data["기업명"] == company_name]
        if company_info.empty:
            raise HTTPException(
                status_code=404, detail=f"Data for company {company_name} not found.")

        result = {}

        # Strength Analysis
        if analysis_type in [AnalysisType.STRENGTH, AnalysisType.INSIGHT]:
            if cached_data and cached_data.get("strength", {}).get("analysis_metric") == strength_metric:
                result["strength"] = cached_data["strength"]
            else:
                result["strength"] = await analyze_metric(
                    company_name, "strength", strength_metric, common_llm, summary_llm, company_info, cache
                )
        # Weakness Analysis
        if analysis_type in [AnalysisType.WEAKNESS, AnalysisType.INSIGHT]:
            if cached_data and cached_data.get("weakness", {}).get("analysis_metric") == weakness_metric:
                result["weakness"] = cached_data["weakness"]
            else:
                result["weakness"] = await analyze_metric(
                    company_name, "weakness", weakness_metric, common_llm, summary_llm, company_info, cache
                )

        # Insight Analysis
        if analysis_type == AnalysisType.INSIGHT:
            # Validate data for Insight Analysis
            if not result.get("strength") or not result.get("weakness"):
                raise ValueError(
                    'Both strength and weakness data are required for insight')
            insight_result = await perform_insight_analysis(
                company_name, result, insight_llm, cache
            )
            result["insight"] = insight_result['insight']

        return result

    except Exception as e:
        logger.error(
            f"[Error] New analysis failed for {company_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def perform_insight_analysis(
    company_name: str, data: Dict, insight_llm, cache: CacheManager
) -> Dict:
    """
    Perform insight analysis by combining strength and weakness data.

    Args:
        company_name (str): The name of the company.
        data (Dict): Data containing strength and weakness analysis.
        insight_llm: LLM instance for insight analysis.
        cache (CacheManager): Cache manager instance.

    Returns:
        Dict: Insight analysis result.
    """
    try:
        # 데이터 유효성 확인
        if not data.get("strength") or not data.get("weakness"):
            raise ValueError(
                "Missing strength or weakness data for insight analysis.")

        prompt_path = os.path.join(
            settings.PROMPTS_DIR, "insight_template.txt")
        analysis_prompt = await asyncio.to_thread(load_prompt, prompt_path)

        final_chain = (
            RunnableLambda(lambda x: {
                "info": merge_analysis_results(x["strength"], x["weakness"]),
                "strength_metric": data["strength"]["analysis_metric"],
                "weakness_metric": data["weakness"]["analysis_metric"],
            })
            | PromptTemplate.from_template(analysis_prompt)
            | insight_llm
            | StrOutputParser()
        )

        logger.info(
            f"\n\n[Insight] Starting insight analysis using strength({data['strength']['analysis_metric']}) and weakness({data['weakness']['analysis_metric']})")

        insight_result = ""
        async for chunk in final_chain.astream({
            "strength": data["strength"]["detailed_result"],
            "weakness": data["weakness"]["detailed_result"],
        }):
            insight_result += chunk

        if not insight_result:
            raise ValueError("Insight analysis failed: No result generated.")
        result = {
            "insight": {
                "analysis_metric": f"{data['strength']['analysis_metric']}/{data['weakness']['analysis_metric']}",
                "summary": insight_result,
            }
        }

        await cache.set(company_name, result, "insight")
        logger.info(
            f"[Insight] Insight analysis completed for {company_name} on {data['strength']['analysis_metric']}/{data['weakness']['analysis_metric']}.")
        return result

    except Exception as e:
        logger.error(f"\n[Error] Insight analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Insight analysis failed")


async def analyze_metric(
    company_name: str,
    analysis_type: str,
    metric: str,
    common_llm,
    summary_llm,
    company_data,
    cache: CacheManager,
) -> Dict:
    """
    Analyze a single metric (strength or weakness).

    Args:
        company_name (str): The name of the company.
        analysis_type (str): The type of analysis ("strength" or "weakness").
        metric (str): The metric to analyze.
        common_llm, summary_llm: LLM models.
        company_data (pd.DataFrame): Company data for analysis.
        cache (CacheManager): Cache manager instance.

    Returns:
        Dict: Analysis result.
    """
    try:
        # 캐시 확인
        cached_metric = await cache.get_validated(company_name, analysis_type, metric)
        if cached_metric:
            logger.info(
                f"[Cache] Found cached {analysis_type} data for metric: {metric}")
            return cached_metric

        # 분석 체인 생성
        analysis_chain = await create_analysis_chain(
            metric, analysis_type == "strength", common_llm, company_data, company_name
        )

        # Log for the detailed analysis
        logger.info(
            f"\n\n[{analysis_type.capitalize()}; Detailed] Starting {analysis_type} analysis for {metric}"
        )
        # 상세 분석 시작
        detailed_result = ""
        async for chunk in analysis_chain.astream(company_data):
            detailed_result += chunk

        summary_chain = await create_summary_chain(summary_llm, metric)

        # Log for the summary generation
        logger.info(
            f"\n\n[{analysis_type.capitalize()}; Summary] Starting {analysis_type} analysis for {metric}"
        )
        # 요약 생성
        summary = ""
        async for chunk in summary_chain.astream({"detailed_result": detailed_result}):
            summary += chunk

        # 결과 검증
        if not detailed_result:
            raise ValueError(
                f"{analysis_type.capitalize()} analysis failed for metric {metric}: No detailed result generated.")
        if not summary:
            raise ValueError(
                f"{analysis_type.capitalize()} analysis failed for metric {metric}: No summary generated.")

        result = {
            "analysis_metric": metric,
            "detailed_result": detailed_result,
            "summary": summary,
        }

        await cache.set(company_name, {analysis_type: result}, analysis_type)
        logger.info(
            f"[{analysis_type.capitalize()}] Successfully analyzed and cached metric: {metric}")

        return result

    except Exception as e:
        logger.error(
            f"[Error] {analysis_type.capitalize()} analysis failed for {metric}: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis metric failed")
