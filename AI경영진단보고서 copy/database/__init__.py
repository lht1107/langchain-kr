from .base import BaseCache
import importlib
from core.config import settings

# 공통 클래스
__all__ = ["BaseCache"]

if settings.DB_TYPE.lower() == 'sqlite':
    SQLiteCache = importlib.import_module("database.sqlite_cache").SQLiteCache
    SQLiteCreditCache = importlib.import_module(
        "database.sqlite_credit_cache").SQLiteCreditCache
    from .sqlite_credit_cache import SQLiteCreditCache
    __all__.extend(["SQLiteCache", "SQLiteCreditCache"])
else:
    PostgreSQLCache = importlib.import_module(
        "database.postgresql_cache").PostgreSQLCache
    PostgreSQLCreditCache = importlib.import_module(
        "database.postgresql_credit_cache").PostgreSQLCreditCache

    __all__.extend(["PostgreSQLCache", "PostgreSQLCreditCache"])


# from .base import BaseCache

# from .sqlite_cache import SQLiteCache
# from .postgresql_cache import PostgreSQLCache
# from .sqlite_credit_cache import SQLiteCreditCache
# from .postgresql_credit_cache import PostgreSQLCreditCache

# __all__ = [
#     "BaseCache",
#     "SQLiteCache",
#     "PostgreSQLCache",
#     "SQLiteCreditCache",
#     "PostgreSQLCreditCache"
# ]
