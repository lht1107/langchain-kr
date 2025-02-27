from typing import List, Dict, Optional
from core.config import settings
from database.base import BaseCache
from utils.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential
import importlib

# Logger initialization
logger = get_logger(__name__)


class CacheManager:
    """Class to manage caching with support for multiple storage backends."""

    def __init__(self):
        """Initialize cache storages based on environment settings."""
        self.storages: List[BaseCache] = []
        self.credit_storages: List[BaseCache] = []
        self._initialize_storages()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _initialize_storages(self) -> None:
        """Initialize primary and credit storages depending on the environment."""
        try:
            db_type = settings.DB_TYPE.lower()
            logger.info(f"[DB_TYPE] {db_type}")
            if settings.ENV.lower() == "development":
                if db_type == "sqlite":
                    # from database.sqlite_cache import SQLiteCache
                    # self.storages.append(SQLiteCache())
                    sqlite_module = importlib.import_module(
                        'database.sqlite_cache')
                    self.storages.append(sqlite_module.SQLiteCache())
                    logger.info(
                        "[Cache] SQLite storage initialized for development")

                    # Credit cache 초기화
                    # from database.sqlite_credit_cache import SQLiteCreditCache
                    # self.credit_storages.append(SQLiteCreditCache())
                    sqlite_credit = importlib.import_module(
                        'database.sqlite_credit_cache')
                    self.credit_storages.append(
                        sqlite_credit.SQLiteCreditCache())
                    logger.info(
                        "[Cache] SQLiteCreditCache storage initialized for development")

                else:
                    # from database.postgresql_cache import PostgreSQLCache
                    # self.storages.append(PostgreSQLCache())
                    postgresql_module = importlib.import_module(
                        'database.postgresql_cache')
                    self.storages.append(postgresql_module.PostgreSQLCache())
                    logger.info(
                        "[Cache] PostgreSQL storage initialized for development")

                    # from database.postgresql_credit_cache import PostgreSQLCreditCache
                    # self.credit_storages.append(PostgreSQLCreditCache())
                    postgresql_module = importlib.import_module(
                        'database.postgresql_cache')
                    self.credit_storages.append(
                        postgresql_module.PostgreSQLCache())
                    logger.info(
                        "[Cache] PostgreSQL_CreditCache storage initialized for development")

            elif settings.ENV.lower() == "production":
                # from database.postgresql_cache import PostgreSQLCache
                # self.storages.append(PostgreSQLCache())
                postgresql_module = importlib.import_module(
                    'database.postgresql_cache')
                self.storages.append(postgresql_module.PostgreSQLCache())
                logger.info(
                    "[Cache] PostgreSQL storage initialized for development")

                # from database.postgresql_credit_cache import PostgreSQLCreditCache
                # self.credit_storages.append(PostgreSQLCreditCache())
                postgresql_module = importlib.import_module(
                    'database.postgresql_cache')
                self.credit_storages.append(
                    postgresql_module.PostgreSQLCache())
                logger.info(
                    "[Cache] PostgreSQL_CreditCache storage initialized for development")

            if not self.storages:
                logger.warning("[Cache] No primary storage initialized")
            if not self.credit_storages:
                logger.warning("[Cache] No credit storage initialized")

        except Exception as e:
            logger.error(f"[Cache] Failed to initialize storages: {str(e)}")
            raise

    async def get_validated(self, company_name: str, analysis_type: str = None, analysis_metric: str = None) -> Optional[Dict]:
        """
        Retrieve and validate cached data for a company.

        Args:
            company_name (str): Name of the company.
            analysis_type (str, optional): Type of analysis (e.g., strength, weakness, insight).
            analysis_metric (str, optional): Specific metric for the analysis.

        Returns:
            Optional[Dict]: Validated cached data if available, otherwise None.
        """
        for storage in self.storages:
            try:
                result = await storage.get(company_name, analysis_type, analysis_metric)
                if result and self.validate_cache_data(result):
                    logger.debug(
                        f"[Cache] Validated hit - company_name: {company_name}, analysis_type: {analysis_type}, metric: {analysis_metric}"
                    )
                    return result
            except Exception as e:
                logger.error(
                    f"[Cache] Error retrieving validated data from {storage.__class__.__name__}: {str(e)}"
                )

        logger.info(
            f"[Cache] No validated cached data found for company: {company_name}, type: {analysis_type}, metric: {analysis_metric}"
        )
        return None

    def validate_cache_data(self, data: Dict) -> bool:
        """
        Validate the structure and completeness of cached data.

        Args:
            data (Dict): Cached data to validate.

        Returns:
            bool: True if data is valid, otherwise False.
        """
        try:
            if not isinstance(data, dict):
                return False

            required_fields = {'strength', 'weakness', 'insight'}
            if not all(field in data for field in required_fields):
                return False

            required_analysis_fields = {
                'analysis_metric', 'detailed_result', 'summary'}
            for analysis_type in ['strength', 'weakness']:
                if not all(field in data[analysis_type] for field in required_analysis_fields):
                    return False

            if not all(field in data['insight'] for field in ['analysis_metric', 'summary']):
                return False

            return True
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    async def get(self, company_name: str, analysis_type: str = None, analysis_metric: str = None) -> Optional[Dict]:
        """
        Retrieve cached data for a company.

        Args:
            company_name (str): Name of the company.
            analysis_type (str, optional): Type of analysis (e.g., strength, weakness, insight).
            analysis_metric (str, optional): Specific metric for the analysis.

        Returns:
            Optional[Dict]: Cached data if available, otherwise None.
        """
        for storage in self.storages:
            try:
                result = await storage.get(company_name, analysis_type, analysis_metric)
                if result and self.validate_cache_data(result):
                    logger.debug(
                        f"[Cache] Validated hit - company_name: {company_name}")
                    return result
            except Exception as e:
                logger.error(
                    f"[Cache] Error retrieving data from {storage.__class__.__name__}: {str(e)}")

        return None

    async def set(self, company_name: str, data: Dict, analysis_type: str = None) -> None:
        """
        Store analysis results in the cache.

        Args:
            company_name (str): Name of the company.
            data (Dict): Data to store in the cache.
            analysis_type (str, optional): Type of analysis (e.g., strength, weakness, insight).
        """
        try:
            existing_data = await self.get(company_name)
            if not existing_data:
                existing_data = self._create_empty_cache()

            if analysis_type:
                if analysis_type in ['strength', 'weakness']:
                    existing_data[analysis_type] = data[analysis_type]
                elif analysis_type == 'insight':
                    if 'strength' in data:
                        existing_data['strength'] = data['strength']
                    if 'weakness' in data:
                        existing_data['weakness'] = data['weakness']
                    existing_data['insight'] = data['insight']

            for storage in self.storages:
                try:
                    await storage.set(company_name, existing_data, analysis_type)
                    logger.debug(
                        f"[Cache] Stored in {storage.__class__.__name__}")
                except Exception as e:
                    logger.error(
                        f"[Cache] Error storing data in {storage.__class__.__name__}: {str(e)}")
        except Exception as e:
            logger.error(f"[Cache] Failed to set cached data: {str(e)}")
            raise

    def _create_empty_cache(self) -> Dict:
        """
        Create an empty cache structure.

        Returns:
            Dict: Empty cache structure.
        """
        return {
            'strength': {'analysis_metric': None, 'detailed_result': None, 'summary': None},
            'weakness': {'analysis_metric': None, 'detailed_result': None, 'summary': None},
            'insight': {'analysis_metric': None, 'summary': None},
        }

    async def close(self) -> None:
        """
        Close all cache storages.
        """
        for storage in self.storages:
            try:
                if hasattr(storage, 'close'):
                    await storage.close()
                    logger.info(f"[Cache] Closed {storage.__class__.__name__}")
            except Exception as e:
                logger.error(
                    f"[Cache] Error closing {storage.__class__.__name__}: {str(e)}")
