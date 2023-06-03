from typing import List

from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models
from django.db.models.query import QuerySet
from django_cte import CTEManager, CTEQuerySet

from core.portal.analytics.enums import AggregateGroup
from core.portal.api.models.organization import Organization
from core.portal.compliance.models.door_event_aggregate_filters import (
    apply_filter,
)
from core.portal.devices.models.camera import Camera
from core.portal.incidents.types import Filter
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class DoorEventAggregateQuerySet(CTEQuerySet):
    def apply_filters(
        self, filters: List[Filter], current_user: User
    ) -> QuerySet["DoorEventAggregate"]:
        """Applies all provided filters to the queryset.
        Args:
            filters (List[FilterInputType]):
                A list of FilterInputType object defines the query filters
            current_user (User): Current user
        Returns:
            queryset (QuerySet["DoorEventAggregate"]): A door event aggregate queryset
        """

        queryset: QuerySet[DoorEventAggregate] = self
        for filter_data in filters:
            queryset = apply_filter(queryset, filter_data, current_user)
        return queryset


class DefaultManager(CTEManager):
    def get_queryset(self) -> DoorEventAggregateQuerySet:
        """Get door event aggregate queryset

        Returns:
            DoorEventAggregateQuerySet: Door event aggregate queyset
        """
        return DoorEventAggregateQuerySet(self.model, using=self._db)


class DoorEventAggregate(Model):
    """Door event aggregate data."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "analytics"
        db_table = "door_event_aggregates"

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "group_key",
                    "group_by",
                    "organization",
                    "zone",
                    "camera",
                ],
                name="unique_row_per_camera_per_group_by_option",
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
        related_name="door_event_aggregates",
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.CASCADE,
        null=False,
        related_name="door_event_aggregates",
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        null=False,
        related_name="door_event_aggregates",
    )

    opened_count = models.PositiveIntegerField(null=False, default=0)
    closed_within_30_seconds_count = models.PositiveIntegerField(
        null=False, default=0
    )
    closed_within_1_minute_count = models.PositiveIntegerField(
        null=False, default=0
    )
    closed_within_5_minutes_count = models.PositiveIntegerField(
        null=False, default=0
    )
    closed_within_10_minutes_count = models.PositiveIntegerField(
        null=False, default=0
    )

    # Custom model managers
    objects = DefaultManager.from_queryset(DoorEventAggregateQuerySet)()


admin.site.register(DoorEventAggregate)
