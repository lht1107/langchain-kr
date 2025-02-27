import os
import asyncio
from langchain.prompts import PromptTemplate
from utils.parser import output_parser
from utils import load_prompt
from core.config import settings


def calculate_percentage_change(before_value, after_value):
    """Calculate percentage change between two values"""
    try:
        change = ((after_value - before_value) / before_value) * 100
        return f"{change:.1f}"
    except ZeroDivisionError:
        return "N/A"
    except TypeError:
        return "Error: Invalid numeric values"


def calculate_change(before_value, after_value):
    """Calculate percentage change between two values"""
    try:
        change = (after_value - before_value)
        return f"{change:.2f}"
    except ZeroDivisionError:
        return "N/A"
    except TypeError:
        return "Error: Invalid numeric values"


def format_influences_table(influences, key_factor):
    """
    Convert influence factors to markdown table format.
    Highlights the key factor that changed in the scenario.
    """
    table_rows = []
    table_rows.append("\n| 순위 | 요인 | 현재 값 | 변경 값 | 영향도 (%p) |")
    table_rows.append("|------|---------|----------|----------|-------------|")

    for idx, feature in enumerate(influences[:5], 1):
        feature_label = feature.get('label', 'Unknown Feature')
        delta_shap = feature.get('delta_shap', 'N/A')

        # 주요 요인은 명확히 표시
        if feature_label == key_factor['label']:
            table_rows.append(
                f"| {idx} | {feature_label} | {key_factor['before_value']} | {key_factor['after_value']} | {delta_shap} |"
            )
        else:
            # 나머지 요인은 변화 없음 처리
            table_rows.append(
                f"| {idx} | {feature_label} | {feature['new_value']} | {feature['new_value']} | {delta_shap} |"
            )
    return "\n".join(table_rows) + "\n"


def format_scenarios(scenarios):
    """Format scenario information in English"""
    formatted_scenarios = []
    for scenario in scenarios:

        key_factor = {
            'label': scenario['label'],
            'before_value': scenario['before']['value'],
            'after_value': scenario['after']['new_value']
        }

        percent_change = calculate_percentage_change(
            scenario['before']['value'],
            scenario['after']['new_value']
        )

        probability_change = calculate_change(
            scenario['before']['probability'],
            scenario['after']['new_probability']
        )

        scenario_text = f"""
        ### Scenario {scenario['id']}: {scenario['label']}

        #### Simulation Overview
        - The analysis examines changing {scenario['label']} from its current value of {scenario['before']['value']} to {scenario['after']['new_value']}.
        - This represents a {percent_change}% change from the current state.

        #### Simulation Results
        - **Default Probability Change**: {scenario['before']['probability']}% → {scenario['after']['new_probability']}% (decrease of {probability_change}%)
        - **Credit Grade Change**: {scenario['before']['grade']} → {scenario['after']['new_grade']}

        #### Analysis of Key Influencing Factors
        {format_influences_table(scenario['after']['top_influenced_features'], key_factor)}

        #### Scenario Interpretation
        - This scenario analyzes how improving {scenario['label']} affects the credit grade.
        - Assuming all other factors remain constant at their current levels, this improvement could enhance the credit grade from {scenario['before']['grade']} to {scenario['after']['new_grade']}.
        """
        formatted_scenarios.append(scenario_text)
    return "\n".join(formatted_scenarios)


async def create_hypothetical_analysis_chain(llm):

    template_path = os.path.join(
        settings.PROMPTS_DIR, "hypothetical_credit_template.txt")

    hypothetical_credit_template = await asyncio.to_thread(load_prompt, template_path)

    hypothetical_analysis_prompt = PromptTemplate.from_template(
        hypothetical_credit_template)

    return {
        "format": lambda x: output_parser.get_format_instructions(),
        "scenarios_analysis": lambda x: format_scenarios(x["scenarios"])
    } | hypothetical_analysis_prompt | llm | output_parser
