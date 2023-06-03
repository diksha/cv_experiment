import base64
import re
import typing as t
from datetime import datetime, timedelta

import boto3
from botocore.response import StreamingBody
from django.db import DatabaseError, transaction
from django.db.models import F, Q
from google.protobuf.any_pb2 import Any
from loguru import logger

from core.portal.lib.enums import TimeBucketWidth
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
)
from core.portal.perceived_data.models.perceived_actor_state_duration_aggregate import (
    PerceivedActorStateDurationAggregate,
)
from core.portal.perceived_data.models.processed_perceived_actor_state_batch_object import (
    ProcessedPerceivedActorStateBatchObject,
)

# trunk-ignore-all(pylint/E0611)
from core.structs.protobufs.v1.state_pb2 import State as PerceivedActorState
from protos.perception.structs.v1.actor_pb2 import ActorCategory

BUCKET_NAME = "voxel-perception-production-states-events"

DurationTimeBuckets = dict[
    str,
    dict[PerceivedActorStateDurationCategory, dict[datetime, timedelta]],
]


# trunk-ignore-all(pylint/W9011)
class PerceivedActorStateDurationBatchProcessor:
    @staticmethod
    def has_batch_been_processed(batch_key: str) -> bool:
        """Checks if the provided key of a batch object has already been processed.

        Args:
            batch_key (str): key of an object

        Returns:
            bool: False if this object has been processed. Otherwise, True.
        """
        return ProcessedPerceivedActorStateBatchObject.objects.filter(
            key__exact=batch_key
        ).exists()

    def execute(self, bucket_name: str, batch_key: str) -> None:
        """Execute the batch processor

        Args:
            bucket_name (str): The name of the bucket where the object is stored
            batch_key (str): The key to the batch object in S3
        """
        batch = self._load_batch(bucket_name=bucket_name, batch_key=batch_key)
        duration_time_buckets = self._process_batch(batch)
        self._publish_time_buckets_as_aggregates(
            batch_key=batch_key,
            duration_time_buckets=duration_time_buckets,
        )

    def _load_batch(self, bucket_name: str, batch_key: str) -> StreamingBody:
        client = boto3.client("s3")
        response = client.get_object(Bucket=bucket_name, Key=batch_key)
        return response["Body"]

    def _process_batch(self, batch: StreamingBody) -> DurationTimeBuckets:
        batch_time_buckets: DurationTimeBuckets = {}
        for serialized_message in batch.iter_lines():
            any_message = Any.FromString(base64.b64decode(serialized_message))
            # If message is a state message, unpack. Else, skip message.
            message = PerceivedActorState()
            if not any_message.Unpack(message):
                continue

            if not self._is_message_valid(message):
                continue

            message_category = self._find_message_category(message)
            if not message_category:
                continue

            # Convert integer milliseconds from epoch to datetime
            start_time = datetime.fromtimestamp(message.timestamp_ms / 1000)
            end_time = datetime.fromtimestamp(message.end_timestamp_ms / 1000)
            message_time_buckets = self._calculate_time_bucket_durations(
                start_time, end_time
            )

            # Merge the message's time bucket aggregates into the batch's
            camera_uuid = message.camera_uuid
            # First initialize the dicts if they don't exist
            if camera_uuid not in batch_time_buckets:
                batch_time_buckets.update({camera_uuid: {}})
            if message_category not in batch_time_buckets[camera_uuid]:
                batch_time_buckets[camera_uuid].update({message_category: {}})
            for (
                bucket_start_timestamp,
                duration,
            ) in message_time_buckets.items():
                existing_duration = batch_time_buckets[camera_uuid][
                    message_category
                ].get(bucket_start_timestamp, timedelta())
                batch_time_buckets[camera_uuid][message_category].update(
                    {bucket_start_timestamp: existing_duration + duration}
                )

        return batch_time_buckets

    def _is_message_valid(self, message) -> bool:
        if message.end_timestamp_ms < message.timestamp_ms:
            logger.error("Message's end timestamp is before start timestamp")
            return False

        return True

    # TODO: Rewrite to be generic
    def _find_message_category(
        self, message: PerceivedActorState
    ) -> t.Union[PerceivedActorStateDurationCategory, None]:
        if message.actor_category is ActorCategory.ACTOR_CATEGORY_PIT:
            # Extra defensive for right-now
            if message.pit_is_stationary:
                return PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME

            return PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME

        if message.actor_category is ActorCategory.ACTOR_CATEGORY_PERSON:
            if not message.person_is_associated:
                if self._is_message_from_old_buggy_perception_version(message):
                    if (
                        not message.person_reach_type
                        and not message.person_lift_type
                    ):
                        return PerceivedActorStateDurationCategory.PERSON_TIME
                else:
                    return PerceivedActorStateDurationCategory.PERSON_TIME

        return None

    @staticmethod
    def _is_message_from_old_buggy_perception_version(
        message: PerceivedActorState,
    ) -> bool:
        """Necessary for consuming messsages before v0.94.0 due
        to message replication bug.

        Args:
            message (PerceivedActorState): state message

        Returns:
            bool: is message produced by perception version < 94
        """
        if not message.HasField("run_uuid"):
            return True

        try:
            # Match everything between the first two periods
            pattern = r"\.(.*)\."
            match_object = re.search(pattern, message.run_uuid)
            perception_version = int(
                message.run_uuid[
                    match_object.start() + 1 : match_object.end() - 1
                ]
            )
        except re.error:
            return False
        except ValueError:
            return False

        return perception_version < 94

    @staticmethod
    def _calculate_time_bucket_durations(
        start_time: datetime, end_time: datetime
    ) -> dict[datetime, timedelta]:
        # Calculate datetime for the first and last time buckets
        # Round start/end datetime down to nearest hour (bucket width)
        first_time_bucket_time = start_time.replace(
            minute=0, second=0, microsecond=0
        )
        last_time_bucket_time = end_time.replace(
            minute=0, second=0, microsecond=0
        )

        time_buckets: dict[datetime, timedelta] = {}
        # If state message does not apply to multiple buckets, then return duration
        # Else, calculate first bucket's duration
        if first_time_bucket_time == last_time_bucket_time:
            return {first_time_bucket_time: end_time - start_time}

        next_time_bucket_time = first_time_bucket_time + timedelta(hours=1)
        time_buckets.update(
            {first_time_bucket_time: next_time_bucket_time - start_time}
        )

        # Calculate intermediate buckets (not first and not last)
        # Implication: Each intermediate bucket has duration = bucket_width
        # Implication: if this loop is entered, assertTrue(len(time_bucket) >= 3)
        # NOTE: In practice, a state should rarely span over 2 buckets.
        curr_time_bucket_time = next_time_bucket_time
        while curr_time_bucket_time < last_time_bucket_time:
            time_buckets.update({curr_time_bucket_time: timedelta(hours=1)})
            curr_time_bucket_time = curr_time_bucket_time + timedelta(hours=1)

        # Calculate last bucket
        # Implication: AssertEqual(curr_time_bucket_time, last_time_bucket_time)
        time_buckets.update(
            {last_time_bucket_time: end_time - last_time_bucket_time}
        )

        return time_buckets

    # TODO: Consider refactoring this out into: creation, publishing, seeing
    def _publish_time_buckets_as_aggregates(
        self,
        batch_key: str,
        duration_time_buckets: dict[
            str,
            dict[
                PerceivedActorStateDurationCategory, dict[datetime, timedelta]
            ],
        ],
    ) -> None:
        data_create = []
        query_expressions = Q()
        # TODO: Use list comprehension to flatten the list
        for (
            camera_uuid,
            category_time_buckets,
        ) in duration_time_buckets.items():
            for category, time_buckets in category_time_buckets.items():
                for (
                    time_bucket_start_timestamp,
                    duration,
                ) in time_buckets.items():
                    data_create.append(
                        PerceivedActorStateDurationAggregate(
                            time_bucket_start_timestamp=time_bucket_start_timestamp,
                            time_bucket_width=TimeBucketWidth.HOUR,
                            category=category,
                            camera_uuid=camera_uuid,
                            duration=duration,
                        )
                    )
                    query_expressions |= Q(
                        time_bucket_start_timestamp__exact=time_bucket_start_timestamp,
                        time_bucket_width__exact=TimeBucketWidth.HOUR,
                        category__exact=category,
                        camera_uuid__exact=camera_uuid,
                    )

        logger.info(f"Number of rows to insert/update: {len(data_create)}")

        try:
            with transaction.atomic():
                existing_rows = []
                rows_created = []
                # Ensure filters aren't empty.
                # Otherwise, the entire table will be retreived
                if query_expressions != Q():
                    for (
                        row
                    ) in PerceivedActorStateDurationAggregate.objects.filter(
                        query_expressions
                    ):
                        # Need to remove timezone to do look-up
                        row_timezone_normalized = (
                            row.time_bucket_start_timestamp.replace(
                                tzinfo=None
                            )
                        )
                        # NOTE: there should be an implied consistency that every row returned
                        # should exist in the duration_time_buckets
                        row.duration = (
                            F("duration")
                            + duration_time_buckets[row.camera_uuid][
                                row.category
                            ][row_timezone_normalized]
                        )
                        existing_rows.append(row)

                    PerceivedActorStateDurationAggregate.objects.bulk_update(
                        existing_rows, fields=["duration"]
                    )

                    rows_created = PerceivedActorStateDurationAggregate.objects.bulk_create(
                        data_create,
                        update_conflicts=False,
                        ignore_conflicts=True,
                        unique_fields=[
                            "time_bucket_start_timestamp",
                            "time_bucket_width",
                            "category",
                            "camera_uuid",
                        ],
                    )

                ProcessedPerceivedActorStateBatchObject.objects.create(
                    key=batch_key
                )
                logger.info(
                    f"Rows updated: {len(existing_rows)} - "
                    f"Rows created: {len(rows_created) - len(existing_rows)}"
                )
        except DatabaseError as err:
            logger.error(
                f"Failed to publish aggregates for chunk {batch_key}: {err}"
            )
