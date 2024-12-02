import os
from typing import Dict
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from utils import load_prompt  
from utils.logger import get_logger
from preprocessing import preprocess_growth_data, preprocess_profitability_data, preprocess_partner_stability_data
from config import PROMPTS_DIR, REQUIRED_TAGS 

logger = get_logger(__name__)

def create_analysis_chain(
    indicator: str, 
    is_strength: bool, 
    llm: ChatOpenAI, 
    df: pd.DataFrame, 
    company_name: str, 
    access_time: pd.Timestamp
):
    """
    Generates an analysis chain to assess a company's strengths or weaknesses.
    """
    logger.info(f"Creating analysis chain for indicator '{indicator}' (Strength: {is_strength})")

    # Step 1: Validate Indicator
    if indicator not in REQUIRED_TAGS:
        logger.error(f"Invalid indicator '{indicator}' provided.")
        raise ValueError(f"Unknown indicator: {indicator}")
    
    # Step 2: Load and Validate Template
    try:
        template_path = os.path.join(PROMPTS_DIR, f"{indicator}_template.txt")
        if not os.path.isfile(template_path):
            logger.error(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template file for '{indicator}' not found at {template_path}")
        
        base_template = load_prompt(template_path)

        # Check required tags for the template
        missing_tags = [tag for tag in REQUIRED_TAGS[indicator] if tag not in base_template]
        if missing_tags:
            logger.warning(f"Template '{indicator}_template.txt' is missing required tags: {missing_tags}")

    except Exception as e:
        logger.error(f"Error loading template for indicator '{indicator}': {e}")
        raise

    # Step 3: Construct Full Prompt
    additional_prompt = "Perform analysis on positive side." if is_strength else "Perform analysis on negative side."
    full_template = f"{base_template}\n\n{additional_prompt}"
    prompt = PromptTemplate.from_template(full_template)
    analysis_chain = prompt | llm | StrOutputParser()
    logger.info(f"Successfully created analysis chain for indicator '{indicator}'")

    # Step 4: Assign Preprocessing Function and Data Keys
    preprocess_func, data_keys = get_preprocess_and_keys(indicator)

    # Step 5: Return Runnable Lambda with access_time passed to preprocess_func
    return RunnableLambda(lambda df: analysis_chain.invoke({
        **{key: preprocess_func(df, company_name, access_time)[key] for key in data_keys}
    }))

def get_preprocess_and_keys(indicator: str):
    """Utility to map indicator to the correct preprocessing function and data keys."""
    if indicator == 'growth':
        return preprocess_growth_data, ['latest_year_month', 'annual_revenue', 'annual_assets', 'monthly_revenue', 'monthly_growth']
    elif indicator == 'profitability':
        return preprocess_profitability_data, ['latest_year_month', 'annual_profit', 'annual_margins', 'monthly_profit', 'monthly_margins']
    elif indicator == 'partner_stability':
        return preprocess_partner_stability_data, ['latest_year_month', 'annual_top5_sales', 'monthly_top5_sales', 'annual_top5_purchase', 'monthly_top5_purchase']
    else:
        logger.error(f"Unsupported indicator type: {indicator}")
        raise ValueError(f"Unsupported indicator type: {indicator}")
