# database_connection.py
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, Dict
from core.config import settings
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Validate database configuration
if not all([settings.DB_USERNAME, settings.DB_PASSWORD, settings.DB_HOST, settings.DB_PORT, settings.DB_NAME]):
    logger.error(
        "Database configuration is missing in config.py or environment variables.")
    raise EnvironmentError(
        "Database configuration is missing. Please set the required database variables.")

# Create connection string
connection_string = f'postgresql+psycopg2://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}'


def fetch_company_data(qry: str, params: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Executes a SQL query on the PostgreSQL database and returns the result as a DataFrame.
    Args:
        qry (str): The SQL query to be executed.
        params (Optional[Dict[str, str]], optional): A dictionary of parameters for the SQL query.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the results of the SQL query.
    """
    # Create SQLAlchemy engine
    engine = create_engine(connection_string)

    # Execute query and return the result as a DataFrame
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(qry, conn, params=params)
            logger.info(f"SQL query executed successfully: {qry}")
    except Exception as e:
        logger.error(f"Error while executing the SQL query: {str(e)}")
        raise RuntimeError(f"Error while executing the SQL query: {str(e)}")

    return df
