from typing import List

from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models
from django.db.models.query import QuerySet
from django_cte import CTEManager, CTEQuerySet

from core.portal.analytics.enums import AggregateGroup
from core.portal.api.models.organization import Organization
from core.portal.compliance.models.production_line import ProductionLine
from core.portal.compliance.models.production_line_aggregate_filters import (
    apply_filter,
)
from core.portal.devices.models.camera import Camera
from core.portal.incidents.types import Filter
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class ProductionLineAggregateQuerySet(CTEQuerySet):
    def apply_filters(
        self, filters: List[Filter], current_user: User
    ) -> QuerySet["ProductionLineAggregate"]:
        """Applies all provided filters to the queryset.
        Args:
            filters (List[FilterInputType]):
                A list of FilterInputType object defines the query filters
            current_user (User): Current user
        Returns:
            queryset (QuerySet["ProductionLineAggregate"]):
                A production line aggregate queryset
        """

        queryset: QuerySet[ProductionLineAggregate] = self
        for filter_data in filters:
            queryset = apply_filter(queryset, filter_data, current_user)
        return queryset


class DefaultManager(CTEManager):
    def get_queryset(self) -> ProductionLineAggregateQuerySet:
        """Get production line aggregate queryset

        Returns:
            ProductionLineAggregateQuerySet: Production line aggregate queryset
        """
        return ProductionLineAggregateQuerySet(self.model, using=self._db)


class ProductionLineAggregate(Model):
    """Production line aggregate data."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "compliance"
        db_table = "production_line_aggregate"

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "group_key",
                    "group_by",
                    "organization",
                    "zone",
                    "camera",
                    "production_line",
                ],
                name="ergonomics_aggregate_unique_row_per_production_line_per_group_by_option",
            )
        ]
        indexes = [
            models.Index(
                # Intended for querying org-level or site-level aggregates
                fields=["-group_key", "group_by", "organization", "zone"]
            ),
            models.Index(
                # Intended for querying a specific production line
                fields=["-group_key", "group_by", "production_line"]
            ),
        ]

    def __str__(self) -> str:
        """String representation of the instance.

        Returns:
            str: string representation of the instance.
        """

        return " - ".join(
            [
                self.group_key.isoformat(),
                self.group_by,
                self.organization.name,
                self.zone.name,
                self.camera.name,
                self.production_line.name,
            ]
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
        related_name="production_line_aggregates",
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.CASCADE,
        null=False,
        related_name="production_line_aggregates",
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        null=False,
        related_name="production_line_aggregates",
    )
    production_line = models.ForeignKey(
        ProductionLine,
        on_delete=models.CASCADE,
        null=False,
        related_name="production_line_aggregates",
    )

    uptime_duration_s = models.PositiveIntegerField(null=False, default=0)
    downtime_duration_s = models.PositiveIntegerField(null=False, default=0)

    # Custom model managers
    objects = DefaultManager.from_queryset(ProductionLineAggregateQuerySet)()


admin.site.register(ProductionLineAggregate)
