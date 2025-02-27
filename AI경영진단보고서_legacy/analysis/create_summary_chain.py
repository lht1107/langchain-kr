import os
from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import StrOutputParser
from utils.load_prompt import load_prompt
from langchain_openai import ChatOpenAI
from core.config import settings
import asyncio
from utils.logger import get_logger

logger = get_logger(__name__)


async def create_summary_chain(llm: ChatOpenAI, analysis_metric: str) -> RunnableLambda:
    """
    Summarizes a detailed analysis using an LLM and a predefined prompt.

    Parameters:
        detailed_analysis (str): The detailed analysis text to be summarized.
        llm (Any): The language model object used for summarization.
        file_name (str): Name of the summary template file.

    Returns:
        str: The summarized result as a string.

    Raises:
        FileNotFoundError: If the prompt file is not found.
    """

    logger.info(
        f"[Chain] Creating summary chain using template '{analysis_metric}'")

    # Step 1: Load the summary template
    try:
        template_path = os.path.join(
            settings.PROMPTS_DIR, "summary_template.txt")
        summary_template = await asyncio.to_thread(load_prompt, template_path)
        logger.debug("[Template] Successfully loaded template for summary")
    except FileNotFoundError as e:
        error_msg = f"Summary template file not found at {settings.PROMPTS_DIR}"
        logger.error(f"[Error] {error_msg}")
        raise FileNotFoundError(error_msg) from e
    except Exception as e:
        error_msg = "Failed to load Summary template"
        logger.error(f"[Error] {error_msg}: {str(e)}")
        raise RuntimeError(error_msg) from e

    # Step 2: Create the prompt template
    try:
        prompt = PromptTemplate.from_template(summary_template)
        logger.debug("[Chain] Prompt template created successfully")
    except Exception as e:
        error_msg = f"Failed to create Summary prompt template for '{analysis_metric}'"
        logger.error(f"[Error] {error_msg}: {str(e)}")
        raise RuntimeError(error_msg) from e

    # Step 3: Define the summarization chain
    # Create the summarization chain
    summary_chain = (
        RunnablePassthrough()  # Directly pass the detailed analysis to the next step
        | prompt  # Use the prompt to format the detailed analysis
        | llm  # Generate the summary using the language model
        | StrOutputParser()  # Parse the output into a clean string
    )

    return summary_chain
