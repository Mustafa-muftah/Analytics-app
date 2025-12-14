from .cameras import router as cameras_router
from .analytics import router as analytics_router
from .reports import router as reports_router

__all__ = ['cameras_router', 'analytics_router', 'reports_router']