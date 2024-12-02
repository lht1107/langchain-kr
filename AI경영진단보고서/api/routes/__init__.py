from .analysis import router as analysis_router
from .query import router as query_router
from .health import router as health_router
from .feedback import router as feedback_router
__all__ = ['analysis_router', 'query_router',
           'health_router', 'feedback_router']
