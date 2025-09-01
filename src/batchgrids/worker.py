"""Celery worker entry point."""
from batchgrids.worker.celery_app import celery_app

__all__ = ["celery_app"]