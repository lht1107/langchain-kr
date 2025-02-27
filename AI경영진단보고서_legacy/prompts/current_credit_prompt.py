import os
import asyncio
from langchain.prompts import PromptTemplate
from utils.parser import output_parser
from utils import load_prompt
from core.config import settings


def format_factors(factors):
    formatted_factors = []
    for factor in factors:
        factor_text = f"""
- **{factor['label']}**:
  - Value: {factor['value']}
  - Impact on Default Probability: {factor['shap_value']}%p
  - Percentile: {factor['percentile']}"""
        formatted_factors.append(factor_text)
    return "\n".join(formatted_factors)


def format_analysis(data):
    analysis_text = f"""
### Company Analysis
- **Company Name**: {data['company_name']}
- **Default Probability**: {data['proba']}%
- **Credit Grade**: {data['grade']}

### Factors Increasing Default Probability
{format_factors(data['top_increasing'])}

### Factors Decreasing Default Probability
{format_factors(data['top_decreasing'])}
"""
    return analysis_text


async def create_current_analysis_chain(llm):

    template_path = os.path.join(
        settings.PROMPTS_DIR, "current_credit_template.txt")

    current_credit_template = await asyncio.to_thread(load_prompt, template_path)

    current_analysis_prompt = PromptTemplate.from_template(
        current_credit_template)

    return {
        "format": lambda x: output_parser.get_format_instructions(),
        "analysis_results": lambda x: format_analysis(x)
    } | current_analysis_prompt | llm | output_parser
