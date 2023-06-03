from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List

from django.db import connections
from django.utils import timezone
from loguru import logger

from core.portal.analytics.enums import AggregateGroup
from core.portal.compliance.jobs.utils import IDLookup, fetch_as_dict
from core.portal.compliance.models.production_line import ProductionLine
from core.portal.compliance.models.production_line_aggregate import (
    ProductionLineAggregate,
)
from core.portal.lib.jobs.base import JobBase
from core.portal.state.models.state import State
from core.structs.actor import ActorCategory

QUERY = """
-- Generate a list of all time buckets between requested time range.
--
-- For example, given the following start/end timestamps and an interval of '1 hour':
--
--     start: 2022-11-21 08:00:00
--       end: 2022-11-21 13:59:59
--
-- Will generate the following records:
--
--    2022-11-21 08:00:00
--    2022-11-21 09:00:00
--    2022-11-21 10:00:00
--    2022-11-21 11:00:00
--    2022-11-21 12:00:00
--    2022-11-21 13:00:00
--
WITH time_buckets AS (
    SELECT generate_series(
        date_trunc(%(series_date_trunc_unit)s, %(start_timestamp)s::timestamp),
        date_trunc(%(series_date_trunc_unit)s, %(end_timestamp)s::timestamp),
        %(group_by)s::INTERVAL
    ) AS bucket
),
-- Generate a list of all production line actor IDs
--
actors AS (
    SELECT UNNEST(%(actor_id_array)s) as actor_id
),
-- Generate a list of all time bucket * actor combinations.
--
-- For example, given the time bucket example from above and two unique actor IDs (foo, bar),
-- the resulting data will look like:
--
--    time_bucket            actor_id
--    2022-11-21 08:00:00    foo
--    2022-11-21 08:00:00    bar
--    2022-11-21 09:00:00    foo
--    2022-11-21 09:00:00    bar
--    2022-11-21 10:00:00    foo
--    2022-11-21 10:00:00    bar
--    2022-11-21 11:00:00    foo
--    2022-11-21 11:00:00    bar
--    2022-11-21 12:00:00    foo
--    2022-11-21 12:00:00    bar
--    2022-11-21 13:00:00    foo
--    2022-11-21 13:00:00    bar
--
-- 6 time buckets * 2 actors = 12 actor_time_bucket records
--
actor_time_buckets AS (
    SELECT
        timezone('utc', bucket) AS "bucket",
        actor_id
    FROM time_buckets
    CROSS JOIN actors
),
-- Now that we have all possible actor/time bucket combinations we can join
-- the state records against these buckets. State records may span across multiple
-- time buckets, so the SUM expressions below ensure that the appropriate durations
-- are attributed to the correct time buckets.
--
-- For example, given the following actor time buckets:
--
--    time_bucket    actor_id
--    8:00:00        foo
--    9:00:00        foo
--
-- And the following state message:
--
--    timestamp    end_timestamp    actor_id    motion_zone_is_in_motion
--    8:45:00      9:15:00          foo         TRUE
--    9:15:00      10:15:00         foo         FALSE
--
-- The query below should split the uptime duration between the 8:00 and 9:00 buckets,
-- and split the downtime duration between the 9:00 and 10:00 buckets:
--
--    time_bucket     actor_id    uptime_duration_s    downtime_duration_s
--    08:00:00        foo         900                  0
--    09:00:00        foo         900                  2700
--    10:00:00        foo         0                    900
--
raw_durations AS (
    SELECT
        atb.bucket,
        organization,
        location,
        camera_uuid,
        atb.actor_id AS "production_line_uuid",
        SUM(
            CASE WHEN s.motion_zone_is_in_motion IS TRUE THEN
                EXTRACT(epoch FROM (
                    LEAST((atb.bucket + %(group_by)s::interval) - '1 millisecond'::interval, s.timestamp + (s.end_timestamp - s.timestamp)) - GREATEST(atb.bucket, s.timestamp)
                ))
                ELSE 0 END
            ) AS "uptime_duration_s",
        SUM(
            CASE WHEN s.motion_zone_is_in_motion IS FALSE THEN
                EXTRACT(epoch FROM (
                    LEAST((atb.bucket + %(group_by)s::interval) - '1 millisecond'::interval, s.timestamp + (s.end_timestamp - s.timestamp)) - GREATEST(atb.bucket, s.timestamp)
                ))
                ELSE 0 END
            ) AS "downtime_duration_s",
        MAX(s.timestamp) AS "max_timestamp"
    FROM
        actor_time_buckets atb
        -- Join state messages with every time bucket they overlap with
        LEFT JOIN state_state s
            ON s.actor_category = %(production_line_actor_category_id)s
            AND s.actor_id = atb.actor_id
            AND (
                (s.timestamp >= atb.bucket AND s.timestamp < atb.bucket + %(group_by)s::interval)
                OR (s.end_timestamp >= atb.bucket AND s.end_timestamp < atb.bucket + %(group_by)s::interval)
            )
    WHERE
        -- Only include records where the state time range overlaps with our target start/end range
        (s.timestamp >= %(start_timestamp)s::timestamp AND s.timestamp < %(end_timestamp)s::timestamp)
        OR (s.end_timestamp >= %(start_timestamp)s::timestamp AND s.end_timestamp < %(end_timestamp)s::timestamp)
    GROUP BY
        bucket,
        organization,
        location,
        camera_uuid,
        atb.actor_id
)
-- The final selection has some extra logic to prevent time buckets from containing
-- more time than is possible. This can happen when state message time ranges overlap.
-- This seems to result in very small "overflow" conditions where we might see 3603
-- seconds worth of state in an hour, when the maximum possible value is 3600 seconds.
-- Here we simply trim off those extra few seconds, starting with uptime duration
-- and then capping downtime duration at the remaining possible seconds for that bucket.
--
SELECT
    bucket,
    organization,
    location,
    camera_uuid,
    production_line_uuid,
    LEAST(uptime_duration_s, EXTRACT('epoch' FROM %(group_by)s::interval)) AS "uptime_duration_s",
    LEAST(downtime_duration_s, EXTRACT('epoch' FROM %(group_by)s::interval) - LEAST(uptime_duration_s, EXTRACT('epoch' FROM %(group_by)s::interval))) AS "downtime_duration_s",
    max_timestamp
FROM raw_durations
;
"""


class ProductionLineAggregationJob(JobBase):
    """Job which performs production line data aggregation."""

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

        Raises:
            RuntimeError: when no state records exist
        """
        try:
            latest_record = ProductionLineAggregate.objects.latest(
                "max_timestamp"
            )
            return latest_record.max_timestamp.replace(
                minute=0, second=0, microsecond=0
            )
        except ProductionLineAggregate.DoesNotExist:
            # If no aggregate data exists, get the earliest timestamp
            # for a motion detection zone state message
            try:
                earliest_state_record = State.objects.filter(
                    actor_category=ActorCategory.MOTION_DETECTION_ZONE.value
                ).earliest("timestamp")
                return earliest_state_record.timestamp
            except State.DoesNotExist as exc:
                # This is considered an exceptional case, if there are no state
                # records then aggregation cannot occur. If this happens in
                # production then we want to know about it.
                raise RuntimeError("No state records found") from exc

    @cached_property
    def end_timestamp(self) -> datetime:
        """End of time range filter used in Timescale events query.

        Returns:
            datetime: end timestamp
        """
        return timezone.now()

    def bulk_create_or_update(
        self, group_by: AggregateGroup, rows: List[Dict[str, Any]]
    ) -> None:
        """Bulk create (or update) aggregate records from Timescale query results.

        Args:
            group_by (AggregateGroup): group by option
            rows (List[Dict[str, Any]]): rows of Timescale query data
        """
        unsaved_objects = [
            ProductionLineAggregate(
                group_by=group_by,
                group_key=row.get("bucket"),
                organization_id=self.id_lookup.get_organization_id(
                    row.get("organization")
                ),
                zone_id=self.id_lookup.get_zone_id(row.get("location")),
                camera_id=self.id_lookup.get_camera_id(row.get("camera_uuid")),
                production_line_id=self.id_lookup.get_production_line_id(
                    row.get("production_line_uuid")
                ),
                max_timestamp=row.get("max_timestamp"),
                uptime_duration_s=row.get("uptime_duration_s"),
                downtime_duration_s=row.get("downtime_duration_s"),
            )
            for row in rows
        ]

        ProductionLineAggregate.objects.bulk_create(
            unsaved_objects,
            update_conflicts=True,
            update_fields=[
                "max_timestamp",
                "uptime_duration_s",
                "downtime_duration_s",
            ],
            unique_fields=[
                "group_by",
                "group_key",
                "organization_id",
                "zone_id",
                "camera_id",
                "production_line_id",
            ],
        )

    def run_hourly_aggregation(self) -> None:
        """Runs hourly aggregation."""
        try:
            logger.info("Starting hourly aggregation job")

            production_line_uuids = list(
                ProductionLine.objects.values_list("uuid", flat=True)
            )

            query_params = {
                "group_by": "1 hour",
                "series_date_trunc_unit": "hour",
                "production_line_actor_category_id": ActorCategory.MOTION_DETECTION_ZONE.value,
                "actor_id_array": production_line_uuids,
                "start_timestamp": self.start_timestamp_hourly,
                "end_timestamp": self.end_timestamp,
            }

            logger.info(
                "Aggregating data between timestamps:"
                + f" {self.start_timestamp_hourly} to {self.end_timestamp}"
            )

            with connections["state"].cursor() as cursor:
                cursor.execute(QUERY, query_params)
                rows = fetch_as_dict(cursor, query_params)

            logger.info(f"Generated {len(rows)} aggregate record(s)")

            if rows:
                self.bulk_create_or_update(AggregateGroup.HOUR, rows)
                logger.info(f"Successfully saved {len(rows)} record(s)")

        # trunk-ignore(pylint/W0703): catch and log any exceptions
        except Exception:
            logger.exception("Hourly aggregation job failed")

    def run(self) -> None:
        """Runs the aggregation job."""
        self.run_hourly_aggregation()
