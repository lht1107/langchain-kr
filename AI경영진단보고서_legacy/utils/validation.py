import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)  # Use the centralized logger configuration


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_input_data(df: pd.DataFrame, required_columns: list, company_col: str) -> pd.DataFrame:
    """
    Validate the basic structure and data types in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list): List of columns that must be present.
        company_col (str): The column containing company names.

    Returns:
        pd.DataFrame: The validated DataFrame with required columns.

    Raises:
        DataValidationError: If required columns are missing or contain invalid data.
    """
    # Define columns with complex structure to be excluded from numeric validation
    complex_structure_columns = ['상위_매출처', '상위_매입처']

    # Check for missing required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # Validate company column is not entirely empty
    if df[company_col].isnull().all():
        raise DataValidationError("Company names are missing")

    # Validate data types
    for col in required_columns:
        if col in ['업종', '기업명']:  # Exclude categorical fields from numeric checks
            if not df[col].apply(lambda x: isinstance(x, str)).all():
                raise DataValidationError(
                    f"Non-string value found in column: {col}")
        elif col == '날짜':
            # Ensure date is in a recognizable datetime format
            if not pd.to_datetime(df[col], errors='coerce').notnull().all():
                raise DataValidationError(
                    f"Invalid date format in column: {col}")
        elif col not in complex_structure_columns and col != company_col:  # Only validate numeric fields
            # Check if other columns contain numeric values
            if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                raise DataValidationError(
                    f"Non-numeric value found in column: {col}")

    logger.info("[validation] Data validation completed successfully")
    return df[required_columns].copy()
