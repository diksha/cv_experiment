import typing as t

from django.db import connections
from loguru import logger

from core.portal.compliance.jobs.utils import fetch_as_dict
from core.portal.lib.enums import TimeBucketWidth
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
)
from core.portal.perceived_data.models.perceived_actor_state_duration_aggregate import (
    PerceivedActorStateDurationAggregate,
)


class PerceivedActorStateDurationETLProcess:
    def __init__(
        self,
        query: str,
        query_params: t.Mapping[str, t.Any],
        category: PerceivedActorStateDurationCategory,
    ):
        self.query: t.Final[str] = query
        self.query_params: t.Final[t.Mapping[str, t.Any]] = query_params
        self.category: t.Final[PerceivedActorStateDurationCategory] = category

    def __str__(self):
        """Generates string representation of the ETL process

        Returns:
            str: Representation of the ETL process
        """
        return f"{self.query} - {self.query_params}"

    def __extract(self) -> list[dict]:
        """Extract from the datasource

        Returns:
            list[dict]: Extracted data
        """
        db_name: t.Final[str] = "state"
        logger.info(f"Starting extraction from '{db_name}' database")

        with connections[db_name].cursor() as cursor:
            cursor.execute(sql=self.query, params=self.query_params)
            extracted_data = fetch_as_dict(
                cursor=cursor, params=self.query_params
            )

        logger.info(
            f"Completed extraction of {len(extracted_data)} entries from '{db_name}' database"
        )
        return extracted_data

    def __transform(
        self, data: list[dict[str, t.Any]]
    ) -> list[PerceivedActorStateDurationAggregate]:
        """Sanitizes and transforms the provided data into PerceivedActorStateDurationAggregate

        Args:
            data (list[dict[str, t.Any]]): Data to be transformed

        Returns:
            list[PerceivedActorStateDurationAggregate]: Transformed data as individual entries
            of PerceivedActorStateDurationAggregate
        """
        logger.info(f"Starting transform on {len(data)} entries")

        # TODO: Should this be its own helper function?
        time_bucket_width_enum: t.Final[TimeBucketWidth] = getattr(
            TimeBucketWidth, self.query_params.get("time_bucket_width").upper()
        )
        transformed_data = [
            PerceivedActorStateDurationAggregate(
                # TODO: datetime warnings
                time_bucket_start_timestamp=entry.get(
                    "time_bucket_start_timestamp"
                ),
                time_bucket_width=time_bucket_width_enum,
                category=self.category,
                camera_uuid=entry.get("camera_uuid"),
                duration=entry.get("duration"),
            )
            # Filtering out invalid camera_uuids
            for entry in data
            if entry.get("camera_uuid", "").strip()
        ]

        logger.info(
            f"Completed transform with {len(transformed_data)} entries"
        )
        return transformed_data

    def __load(self, data: list[PerceivedActorStateDurationAggregate]) -> None:
        logger.info(f"Starting load of {len(data)}")

        PerceivedActorStateDurationAggregate.objects.bulk_create(
            data,
            update_conflicts=True,
            update_fields=[
                "duration",
            ],
            unique_fields=[
                "time_bucket_start_timestamp",
                "time_bucket_width",
                "category",
                "camera_uuid",
            ],
        )
        logger.info(f"Completed loading of {len(data)} entries into database")

    def execute(self) -> None:
        """Executes the ETL process as initialized"""
        logger.info(f"Executing ETL Process: {self}")
        extracted_data = self.__extract()
        transformed_data = self.__transform(data=extracted_data)
        self.__load(data=transformed_data)
        logger.info("ETL Process has completed")
