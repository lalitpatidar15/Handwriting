import os
from typing import Optional, Tuple

from celery import Celery
from kombu.exceptions import KombuError


BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "idp_worker",
    broker=BROKER_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=True,
    task_store_errors_even_if_ignored=False,
)


def is_celery_broker_available() -> Tuple[bool, Optional[str]]:
    try:
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=1)
        return True, None
    except (OSError, KombuError) as exc:
        return False, str(exc)

# Auto-discover tasks in workers package.
celery_app.autodiscover_tasks(["workers"])
