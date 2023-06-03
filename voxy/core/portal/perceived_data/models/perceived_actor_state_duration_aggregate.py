from django.contrib import admin
from django.db import models

from core.portal.lib.enums import TimeBucketWidth
from core.portal.lib.models.base import Model
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
)


class PerceivedActorStateDurationAggregate(Model):
    """Perceived actor state duration aggregate model."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "perceived_data"
        db_table = "perceived_actor_state_duration_aggregate"

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "camera_uuid",
                    "time_bucket_start_timestamp",
                    "category",
                    "time_bucket_width",
                ],
                name="perceived_actor_state_duration_aggregate_unique_constraint",
            )
        ]

        indexes = [
            # TODO: define appropriate indices
        ]

    time_bucket_start_timestamp = models.DateTimeField(
        blank=False,
        null=False,
        help_text="Timestamp of the beginning of this time bucket.",
    )
    time_bucket_width = models.PositiveSmallIntegerField(
        blank=False,
        null=False,
        choices=TimeBucketWidth.choices,
        help_text="Width of this time bucket (hour, day, etc.)",
    )
    category = models.PositiveSmallIntegerField(
        blank=False,
        null=False,
        choices=PerceivedActorStateDurationCategory.choices,
        help_text="Category of data contained in this aggregate group.",
    )
    camera_uuid = models.CharField(
        max_length=250,
        null=False,
        blank=False,
    )
    duration = models.DurationField(
        blank=False,
        null=False,
        default=0,
        help_text="Total actor state duration.",
    )

    def __str__(self) -> str:
        """String representation of the instance.

        Returns:
            str: string representation of the instance.
        """

        return " - ".join(
            [
                self.time_bucket_start_timestamp.isoformat(),
                TimeBucketWidth(self.time_bucket_width).name,
                PerceivedActorStateDurationCategory(self.category).name,
                self.camera_uuid,
                str(self.duration),
            ]
        )


admin.site.register(PerceivedActorStateDurationAggregate)
