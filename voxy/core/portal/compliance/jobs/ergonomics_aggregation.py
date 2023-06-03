from datetime import datetime
from functools import cached_property
from typing import Any, Dict

from django.db import connections, transaction
from django.utils import timezone
from loguru import logger

from core.portal.analytics.enums import AggregateGroup
from core.portal.compliance.jobs.utils import IDLookup, fetch_as_dict
from core.portal.compliance.models.ergonomics_aggregate import (
    ErgonomicsAggregate,
)
from core.portal.lib.jobs.base import JobBase

QUERY = """
SELECT
    time_bucket(INTERVAL %(group_by)s, timestamp) AS "bucket",
    camera_uuid,
    organization,
    location,
    SUM(CASE WHEN person_lift_type IS NOT NULL THEN 1 ELSE 0 END) AS "total_posture_count",
    SUM(CASE WHEN person_lift_type = 0 THEN 1 ELSE 0 END) AS "bad_posture_count",
    SUM(CASE WHEN person_lift_type = 1 THEN 1 ELSE 0 END) AS "good_posture_count",
    SUM(CASE WHEN person_lift_type = 2 THEN 1 ELSE 0 END) AS "unknown_posture_count",
    max(timestamp) AS "max_timestamp"
FROM state_state
WHERE
    timestamp BETWEEN %(start_timestamp)s AND %(end_timestamp)s
    AND person_lift_type IS NOT NULL
GROUP BY
    bucket,
    organization,
    location,
    camera_uuid
;
"""


class ErgonomicsAggregationJob(JobBase):
    """Job which performs ergonomics data aggregation."""

    @cached_property
    def id_lookup(self) -> IDLookup:
        """Instance of IDLookup for converting keys/UUIDs to foreign keys.

        Returns:
            IDLookup: IDLookup instance
        """
        return IDLookup()

    @cached_property
    def start_timestamp_hourly(self) -> datetime:
        """Start of time range filter used in Timescale hourly events query.

        Returns:
            datetime: start timestamp
        """
        try:
            latest_record = ErgonomicsAggregate.objects.latest("max_timestamp")
            return latest_record.max_timestamp.replace(
                minute=0, second=0, microsecond=0
            )
        except ErgonomicsAggregate.DoesNotExist:
            return datetime.min.replace(tzinfo=timezone.utc)

    @cached_property
    def end_timestamp(self) -> datetime:
        """End of time range filter used in Timescale events query.

        Returns:
            datetime: end timestamp
        """
        return timezone.now()

    def save_row(self, group_by: AggregateGroup, row: Dict[str, Any]) -> None:
        """Saves a row of aggregate data from Timescale.

        Args:
            group_by (AggregateGroup): how the row data was grouped
            row (Dict[str, Any]): row of aggregate data from Timescale
        """
        ErgonomicsAggregate.objects.update_or_create(
            group_by=group_by,
            group_key=row.get("bucket"),
            organization_id=self.id_lookup.get_organization_id(
                row.get("organization")
            ),
            zone_id=self.id_lookup.get_zone_id(row.get("location")),
            camera_id=self.id_lookup.get_camera_id(row.get("camera_uuid")),
            defaults={
                "max_timestamp": row.get("max_timestamp"),
                "total_posture_count": row.get("total_posture_count"),
                "good_posture_count": row.get("good_posture_count"),
                "bad_posture_count": row.get("bad_posture_count"),
                "unknown_posture_count": row.get("unknown_posture_count"),
            },
        )

    def run_hourly_aggregation(self) -> None:
        """Runs hourly aggregation."""
        try:
            logger.info("Starting hourly aggregation job")

            query_params = {
                "group_by": "1 hour",
                "start_timestamp": self.start_timestamp_hourly,
                "end_timestamp": self.end_timestamp,
            }

            with connections["state"].cursor() as cursor:
                cursor.execute(QUERY, query_params)
                rows = fetch_as_dict(cursor, query_params)

            logger.info(
                f"Generated {len(rows)} aggregate record(s) between "
                f" {self.start_timestamp_hourly} and {self.end_timestamp}"
            )

            with transaction.atomic():
                for row in rows:
                    self.save_row(AggregateGroup.HOUR, row)

            logger.info(f"Successfully saved {len(rows)} record(s)")
        # trunk-ignore(pylint/W0703): catch and log any exceptions
        except Exception:
            logger.exception("Hourly aggregation job failed")

    def run(self) -> None:
        """Runs the aggregation job."""
        self.run_hourly_aggregation()
