"""FastAPI entry points for health, inference, and future backend services."""

from .main import app, create_app

__all__ = ["app", "create_app"]
