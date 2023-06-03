from datetime import datetime
from functools import cached_property
from typing import Any, Dict

from django.db import connections, transaction
from django.utils import timezone
from loguru import logger

from core.portal.analytics.enums import AggregateGroup
from core.portal.compliance.jobs.utils import IDLookup, fetch_as_dict
from core.portal.compliance.models.door_event_aggregate import (
    DoorEventAggregate,
)
from core.portal.lib.jobs.base import JobBase

QUERY = """
WITH
    door_open_closed_summary AS (
        SELECT
            timestamp,
            organization,
            location,
            camera_uuid,
            event_type,
            LEAD(event_type, 1) OVER (PARTITION BY camera_uuid, object_id ORDER BY timestamp ASC) AS "next_event_type",
            (LEAD(timestamp, 1) OVER (PARTITION BY camera_uuid, object_id ORDER BY timestamp ASC) - timestamp) AS "next_event_interval"
        FROM state_event
        WHERE
            timestamp BETWEEN %(start_timestamp)s  AND %(end_timestamp)s
            AND event_type IN (1, 2)
    )
SELECT
    time_bucket(INTERVAL %(group_by)s, timestamp) AS "bucket",
    camera_uuid,
    organization,
    location,
    count(*) AS "opened_count",
    count(CASE WHEN next_event_interval <= '30 seconds' THEN 1 END) AS "closed_within_30_seconds_count",
    count(CASE WHEN next_event_interval <= '1 minute' THEN 1 END) AS "closed_within_1_minute_count",
    count(CASE WHEN next_event_interval <= '5 minutes' THEN 1 END) AS "closed_within_5_minutes_count",
    count(CASE WHEN next_event_interval <= '10 minutes' THEN 1 END) AS "closed_within_10_minutes_count",
    max(timestamp) AS "max_timestamp"
FROM door_open_closed_summary
WHERE
    event_type = 1
    AND next_event_type = 2
GROUP BY
    bucket,
    organization,
    location,
    camera_uuid
;
"""


class DoorEventAggregationJob(JobBase):
    """Job which performs door event data aggregation."""

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
            latest_record = DoorEventAggregate.objects.latest("max_timestamp")
            return latest_record.max_timestamp.replace(
                minute=0, second=0, microsecond=0
            )
        except DoorEventAggregate.DoesNotExist:
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

        This operation is idempotent in the sense that if a record already
        exists for a specific group/org/zone/camera combination, the aggregate
        values of the existing row are updated. The motivation for this is to
        support processing of incomplete buckets and easy regeneration of data
        for older buckets. For example, let's say an hourly job runs at 9:30,
        we aggregate all data between 9:00-9:30, and save it in the 9:00
        bucket. At 10:00, we run the job again and regenerate the aggregate
        data between 9:00-10:00 and update the existing record for the 9:00
        bucket which was created in the previous run.

        Run 1 (9:00 bucket): 5 events (only includes data from 9:00-9:30)
        Run 2 (9:00 bucket): 10 events (includes all data between 9:00-10:00)

        Args:
            group_by (AggregateGroup): how the row data was grouped
            row (Dict[str, Any]): row of aggregate data from Timescale
        """
        DoorEventAggregate.objects.update_or_create(
            group_by=group_by,
            group_key=row.get("bucket"),
            organization_id=self.id_lookup.get_organization_id(
                row.get("organization")
            ),
            zone_id=self.id_lookup.get_zone_id(row.get("location")),
            camera_id=self.id_lookup.get_camera_id(row.get("camera_uuid")),
            defaults={
                "max_timestamp": row.get("max_timestamp"),
                "opened_count": row.get("opened_count"),
                "closed_within_30_seconds_count": row.get(
                    "closed_within_30_seconds_count"
                ),
                "closed_within_1_minute_count": row.get(
                    "closed_within_1_minute_count"
                ),
                "closed_within_5_minutes_count": row.get(
                    "closed_within_5_minutes_count"
                ),
                "closed_within_10_minutes_count": row.get(
                    "closed_within_10_minutes_count"
                ),
            },
        )

    def run_hourly_aggregation(self) -> None:
        """Runs hourly door event aggregation."""
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
        """Runs door event aggregation job."""
        self.run_hourly_aggregation()
