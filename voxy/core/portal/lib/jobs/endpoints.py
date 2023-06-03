from typing import List, Type

from django.conf import settings
from loguru import logger
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from core.portal.activity.jobs import SessionTrackingETLJob
from core.portal.compliance.jobs.door_event_aggregation import (
    DoorEventAggregationJob,
)
from core.portal.compliance.jobs.production_line_aggregation import (
    ProductionLineAggregationJob,
)
from core.portal.demos.jobs.refresh_demo_data import RefreshDemoDataJob
from core.portal.lib.jobs.base import JobBase
from core.portal.notifications.jobs.base import ScheduledNotificationJob
from core.portal.notifications.jobs.hacks.zone_pulse import ZonePulseJob
from core.portal.notifications.jobs.organization_daily_summary import (
    OrganizationDailySummaryJob,
)


def run_jobs(
    jobs: List[JobBase],
    notification_job_types: List[Type[ScheduledNotificationJob]],
) -> Response:
    all_jobs = []

    # Run notification jobs first
    for job_type in notification_job_types:
        all_jobs.extend(job_type.get_jobs_to_run(base_url=settings.BASE_URL))

    # Then run remaining jobs
    all_jobs.extend(jobs)

    success_count = 0
    fail_count = 0

    for job in all_jobs:
        try:
            job.run()
            success_count += 1
        except Exception:  # trunk-ignore(pylint/W0703): allow catching general exception
            job_type = type(job).__name__
            logger.exception(f"Job failed ({job_type})")
            fail_count += 1

    return Response(
        {
            "job_count": len(all_jobs),
            "success_count": success_count,
            "fail_count": fail_count,
        }
    )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def hourly_trigger(*args, **kwargs) -> Response:
    """Triggered every hour on the hour."""
    del args, kwargs

    return run_jobs(
        jobs=[
            RefreshDemoDataJob(),
            DoorEventAggregationJob(),
            ProductionLineAggregationJob(),
        ],
        notification_job_types=[
            OrganizationDailySummaryJob,
        ],
    )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def semihourly_trigger(*args, **kwargs) -> Response:
    """Triggered twice per hour (XX:00 and XX:30)."""
    del args, kwargs

    return run_jobs(
        jobs=[
            SessionTrackingETLJob(),
        ],
        notification_job_types=[
            ZonePulseJob,
        ],
    )
