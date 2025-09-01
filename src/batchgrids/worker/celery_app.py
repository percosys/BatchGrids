from celery import Celery

from batchgrids.config import settings

celery_app = Celery(
    "batchgrids",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["batchgrids.worker.tasks"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)