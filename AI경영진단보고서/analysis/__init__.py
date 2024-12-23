"""
Analysis Module.

This module contains functions that are used to determine the strengths and weaknesses of a company,
create analysis chains using Langchain, and merge the analysis results for final insights.

Modules Included:
- determine_strength_weakness: Determines the strengths and weaknesses of a company.
- create_analysis_chain: Creates an analysis chain for specific indicators.
- merge_analysis_results: Merges the results of strengths and weaknesses analysis.

Usage:
from analysis import determine_strength_weakness, create_analysis_chain, merge_analysis_results
"""

from .determine_strength_weakness import determine_strength_weakness
from .create_summary_chain import create_summary_chain
from .merge_analysis_results import merge_analysis_results
from .create_analysis_chain import create_analysis_chain
from .credit_analyzer import ShapAnalyzer


class AnalysisChainError(Exception):
    """분석 체인 생성 및 실행 관련 에러"""
    pass


class TemplateError(AnalysisChainError):
    """템플릿 로딩 및 검증 관련 에러"""
    pass


class PreprocessingError(AnalysisChainError):
    """데이터 전처리 관련 에러"""
    pass


__all__ = ["determine_strength_weakness", "create_analysis_chain",
           "merge_analysis_results", "create_summary_chain", "ShapAnalyzer"]
