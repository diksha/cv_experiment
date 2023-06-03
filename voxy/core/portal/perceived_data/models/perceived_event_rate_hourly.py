from django.contrib import admin
from django.db import models

from core.portal.devices.models.camera import Camera
from core.portal.lib.models.base import Model
from core.portal.perceived_data.models.perceived_event_rate_definition import (
    PerceivedEventRateDefinition,
)


class PerceivedEventRateHourly(Model):
    """Perceived event rate hourly buckets."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "perceived_data"
        db_table = "perceived_event_rate_hourly"

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "time_bucket_start_timestamp",
                    "definition",
                    "camera",
                ],
                name="perceived_event_rate_hourly_unique_constraint",
            )
        ]

    time_bucket_start_timestamp = models.DateTimeField(
        null=False,
        blank=False,
        help_text="Timestamp of the beginning of this time bucket.",
    )
    definition = models.ForeignKey(
        PerceivedEventRateDefinition,
        on_delete=models.CASCADE,
        null=False,
        blank=False,
        related_name="hourly_perceived_event_rates",
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        null=False,
        blank=False,
        related_name="hourly_perceived_event_rates",
    )
    numerator_value = models.DecimalField(
        blank=False,
        null=False,
        default=0.0,
        # This may be overkill, but this field supports the following range:
        #     min: 000000000.0000000001
        #     max: 999999999.9999999999
        max_digits=19,
        decimal_places=10,
        help_text="Perceived event rate numerator value.",
    )
    denominator_value = models.DecimalField(
        blank=False,
        null=False,
        default=0.0,
        # This may be overkill, but this field supports the following range:
        #     min: 000000000.0000000001
        #     max: 999999999.9999999999
        max_digits=19,
        decimal_places=10,
        help_text="Perceived event rate denominator_value.",
    )

    def __str__(self) -> str:
        """String representation of the instance.

        Returns:
            str: string representation of the instance.
        """

        return " - ".join(
            [
                self.time_bucket_start_timestamp.isoformat(),
                str(self.definition),
                str(self.camera),
            ]
        )


admin.site.register(PerceivedEventRateHourly)
