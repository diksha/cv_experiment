from django.db import models

from core.portal.analytics.enums import AggregateGroup
from core.portal.api.models.organization import Organization
from core.portal.devices.models.camera import Camera
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class PPEEventAggregate(Model):
    """PPE event aggregate data."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "analytics"
        db_table = "ppe_aggregates"

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "group_key",
                    "group_by",
                    "organization",
                    "zone",
                    "camera",
                ],
                name="ppe_event_aggregate_unique_constraint",
            )
        ]

        indexes = [
            models.Index(
                # We expect the most common queries to be organization-level
                # or zone-level aggregates for a particular time range.
                fields=["-group_key", "group_by", "organization", "zone"]
            ),
        ]

    def __str__(self) -> str:
        """String representation of the instance.

        Returns:
            str: string representation of the instance.
        """

        return (
            f"{self.group_key}"
            f" - {self.group_by}"
            f" - {self.organization.name}"
            f" - {self.zone.name}"
            f" - {self.camera.uuid}"
        )

    group_key = models.DateTimeField(
        blank=False,
        null=False,
        help_text="Timestamp of the beginning of the aggregate group.",
    )
    group_by = models.CharField(
        blank=False,
        null=False,
        max_length=25,
        choices=AggregateGroup.choices,
        help_text="How this data was grouped (by hour, by day, etc.)",
    )
    max_timestamp = models.DateTimeField(
        blank=False,
        null=False,
        help_text="Max timestamp of all events contained in this aggregate group.",
    )

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        null=False,
        related_name="ppe_aggregates",
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.CASCADE,
        null=False,
        related_name="ppe_aggregates",
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        null=False,
        related_name="ppe_aggregates",
    )

    hard_hat_wearing_duration_ms = models.PositiveIntegerField(
        null=False, default=0
    )
    hard_hat_not_wearing_duration_ms = models.PositiveIntegerField(
        null=False, default=0
    )

    safety_vest_wearing_duration_ms = models.PositiveIntegerField(
        null=False, default=0
    )
    safety_vest_not_wearing_duration_ms = models.PositiveIntegerField(
        null=False, default=0
    )
