from typing import Dict
import shap
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils.logger import get_logger

logger = get_logger(__name__)


class ShapAnalyzer:
    """SHAP analysis and result interpretation class for classification models."""

    def __init__(self, model, X: pd.DataFrame, var_dict: Dict, explainer=None, shap_values=None):
        self.model = model
        self.X = X
        self.var_dict = var_dict
        if explainer is None:
            self.explainer = shap.TreeExplainer(model)
            logger.warning(
                "Creating new TreeExplainer - consider passing pre-computed explainer")
        else:
            self.explainer = explainer

        if shap_values is None:
            logger.warning(
                "Computing new SHAP values - consider passing pre-computed values")
            self.shap_values = self.explainer.shap_values(self.X)
        else:
            self.shap_values = shap_values

        self._cache = {}  # 결과 캐싱을 위한 딕셔너리
        logger.info("ShapAnalyzer initialized.")

    def get_shap_analysis(self, instance_index: int) -> Dict:
        """Perform SHAP analysis and return results."""
        self.company_name = f"Company_{instance_index}"
        try:
            # 저장된 shap_values 사용
            shap_values = self.shap_values

            # base_value 설정
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[1]
            else:
                base_value = self.explainer.expected_value

            # Calculate feature impacts
            feature_impacts = []
            instance = self.X.iloc[instance_index]

            for idx, (feature, value) in enumerate(instance.items()):
                shap_value = shap_values[instance_index][idx]
                percentile = int(stats.percentileofscore(
                    self.X[feature], float(value)))

                # .get('label', feature)
                label = self.var_dict.get(feature, feature)

                feature_impacts.append({
                    "feature": feature,
                    "label": label,
                    "value": round(float(value), 2),
                    "shap_value": round(float(shap_value*100), 2),
                    "impact": "positive" if shap_value > 0 else "negative",
                    "percentile": percentile
                })

            increasing = sorted([{
                "feature": f["feature"],
                "label": f["label"],
                "value": f["value"],
                "shap_value": f["shap_value"],
                "percentile": f["percentile"]
            } for f in feature_impacts if f["shap_value"] > 0],
                key=lambda x: x["shap_value"], reverse=True)[:5]

            decreasing = sorted([{
                "feature": f["feature"],
                "label": f["label"],
                "value": f["value"],
                "shap_value": f["shap_value"],
                "percentile": f["percentile"]
            } for f in feature_impacts if f["shap_value"] < 0],
                key=lambda x: x["shap_value"])[:5]

            # Simplified predict_proba
            instance_df = pd.DataFrame([instance])
            model_prob = self.model.predict_proba(
                instance_df)[0, 1]  # 직접 양성 클래스 확률 선택

            self.current_prob = round(model_prob * 100, 2)
            # logger.info(f"[current_prob]: {self.current_prob}")

            grade = self.map_probability_to_grade(self.current_prob)
            shap_sum = sum(shap_values[instance_index])
            shap_prob = shap_sum + base_value

            if not np.isclose(shap_prob, model_prob, rtol=1e-3):
                logger.warning(
                    f"Probability mismatch - SHAP: {shap_prob:.4f}, Model {model_prob:.4f}")

            return {
                # "base_value": base_value,
                "company_name": self.company_name,
                "proba": round(model_prob * 100, 2),
                "grade": grade,
                "feature_impacts": feature_impacts,
                "top_increasing": increasing,
                "top_decreasing": decreasing
            }
        except Exception as e:
            logger.error(f"Error in get_shap_analysis: {e}")
            raise e

    def map_probability_to_grade(self, proba: float) -> str:
        """Map default probability to credit grade with wider ranges."""
        if proba < 5:
            return "AAA"
        elif proba < 10:
            return "AA"
        elif proba < 15:
            return "A"
        elif proba < 20:
            return "BBB"
        elif proba < 30:
            return "BB"
        elif proba < 40:
            return "B"
        elif proba < 50:
            return "CCC"
        elif proba < 60:
            return "CC"
        else:
            return "C"

    def calculate_hypothetical(self, instance_index: int, feature: str, values: np.ndarray, scenario_index: int) -> Dict:
        """
        Perform hypothetical analysis by varying a single feature and capture the first grade improvement scenario.
        """
        self.grade_hierarchy = ["AAA", "AA", "A",
                                "BBB", "BB", "B", "CCC", "CC", "C"]
        try:
            # Clone the instance and create hypothetical data
            instance = self.X.iloc[instance_index].copy()
            hypothetical_data = pd.DataFrame(
                [instance.to_dict()] * len(values))
            hypothetical_data[feature] = values

            # Retrieve the feature label
            label = self.var_dict.get(feature, feature)

            # Predict probabilities and calculate grades
            proba = self.model.predict_proba(hypothetical_data)[:, 1]
            rounded_proba = [round(p * 100, 2) for p in proba]
            grades = [self.map_probability_to_grade(p) for p in rounded_proba]

            # Current grade for comparison
            current_grade = self.map_probability_to_grade(self.current_prob)

            # Create a DataFrame for filtering
            results_df = pd.DataFrame({
                'value': values,
                'probability': rounded_proba,
                'grade': grades,
                'index': range(len(values))
            })

            # Find scenarios where grade improves (i.e., grade is better than current_grade)
            current_grade_index = self.grade_hierarchy.index(current_grade)

            # Filter scenarios with better grade
            improved_df = results_df[results_df['grade'].apply(
                lambda x: self.grade_hierarchy.index(x) < current_grade_index)]

            if improved_df.empty:
                # No improvement found
                return None  # Return None if no valid scenario found

            # Select the first scenario where grade improves
            first_improvement_idx = improved_df.index.max()
            scenario_value = improved_df.loc[first_improvement_idx, 'value']
            scenario_prob = improved_df.loc[first_improvement_idx,
                                            'probability']
            scenario_grade = improved_df.loc[first_improvement_idx, 'grade']

            # Create hypothetical_data for the selected scenario
            selected_data = hypothetical_data.iloc[first_improvement_idx:first_improvement_idx + 1]
            shap_values = self.explainer.shap_values(selected_data)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            delta_shap_values = shap_values[0] - \
                self.shap_values[instance_index]

            # Use top_increasing features from get_shap_analysis
            top_increasing_features = [
                factor["feature"] for factor in self.get_shap_analysis(instance_index)["top_increasing"]
            ]

            top_features = []
            for feature_name in top_increasing_features:
                # Find the index of the feature
                if feature_name in self.X.columns:
                    idx = self.X.columns.get_loc(feature_name)
                    shap_val = shap_values[0][idx] * \
                        100  # Convert to percentage
                    # Convert to percentage
                    delta_val = delta_shap_values[idx] * 100
                    new_value = selected_data.iloc[0][feature_name]

                    f_label = self.var_dict.get(feature_name, feature_name)
                    top_features.append({
                        "feature": feature_name,
                        "label": f_label,
                        "new_value": round(float(new_value), 2),
                        "new_shap_value": round(shap_val, 2),  # New SHAP Value
                        "delta_shap": round(delta_val, 2),  # Delta SHAP
                    })

            return {
                "id": f"Scenario_{scenario_index}",
                "feature": feature,
                "label": label,
                "before": {
                    "value": round(float(instance[feature]), 2),
                    "probability": self.current_prob,
                    "grade": current_grade
                },
                "after": {
                    "new_value": round(float(scenario_value), 2),
                    "new_probability": scenario_prob,
                    "new_grade": scenario_grade,
                    "top_influenced_features": top_features
                }
            }
        except Exception as e:
            logger.error(f"Error in calculate_hypothetical: {e}")
            raise e

    def generate_hypothetical_results(self, instance_index: int) -> Dict:
        """
        Generate hypothetical results for the top 5 increasing factors,
        capturing the first grade improvement scenario for each.
        """
        try:
            # Retrieve top increasing features
            top_increasing_features = [
                factor["feature"] for factor in self.get_shap_analysis(instance_index)["top_increasing"]
            ]

            hypothetical_results = []
            scenario_index = 1  # Initialize scenario index
            for feature in top_increasing_features:
                # Create a range of values to vary the feature
                current_value = self.X.iloc[instance_index][feature]
                rg = range(-100, 101, 1)
                percentages = np.array(rg) / 100
                feature_range = current_value * (1 + percentages)

                # Calculate hypothetical scenarios
                result = self.calculate_hypothetical(
                    instance_index, feature, feature_range, scenario_index)
                if result:  # Include only if there's an improvement
                    hypothetical_results.append(result)
                    scenario_index += 1  # Increment scenario index

            return {"scenarios": hypothetical_results}
        except Exception as e:
            logger.error(f"Error in generate_hypothetical_results: {e}")
            raise e
