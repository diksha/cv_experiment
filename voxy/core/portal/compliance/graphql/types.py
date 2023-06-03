import typing as t
from datetime import datetime

import graphene
from django.db.models import F, fields
from django.db.models.fields.json import KeyTextTransform
from django.db.models.functions import Cast
from django.db.models.query import QuerySet
from graphene_django import DjangoConnectionField

from core.portal.accounts.permissions import (
    ANALYTICS_READ,
    DOWNTIME_READ,
    INCIDENTS_READ,
)
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.analytics.enums import AggregateGroup
from core.portal.api.models.incident import Incident
from core.portal.compliance.models.production_line import (
    ProductionLine as ProductionLineModel,
)
from core.portal.incidents.graphql.types import FilterInputType, IncidentType
from core.portal.lib.graphql.exceptions import PermissionDenied


class ComplianceType(graphene.ObjectType):
    """GraphQL type for compliance types."""

    id = graphene.ID(required=True)
    key = graphene.String(required=True)
    name = graphene.String(required=True)


class DoorOpenStatsMetrics(graphene.ObjectType):
    """Door open stats group metrics"""

    open_time_duration_s = graphene.Int(
        required=True,
        description="Duration of door open time (in seconds)",
    )
    close_time_duration_s = graphene.Int(
        required=True,
        description="Duration of door close time (in seconds)",
    )
    partially_open_time_duration_s = graphene.Int(
        required=True,
        description="Duration of door partially open time (in seconds)",
    )


class DoorOpenStatsDimension(graphene.ObjectType):
    """Door open stats group dimensions."""

    datetime = graphene.DateTime(
        required=True,
        description=(
            "Door open group datetime truncated to the appropriate date part"
            " based on the group_by property (e.g. hourly groups are truncated"
            " to the hour, daily groups truncated to the day, etc.)."
            " All values are localized to the event zone's timezone."
        ),
    )
    max_timestamp = graphene.DateTime(
        required=True,
        description="Max timestamp of all events contained in this aggregate group.",
    )
    organization = graphene.Field(
        graphene.lazy_import(
            "core.portal.organizations.graphql.types.OrganizationType"
        ),
        required=True,
    )
    zone = graphene.Field(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType"),
        required=True,
    )
    camera = graphene.Field(
        graphene.lazy_import("core.portal.devices.graphql.types.CameraType"),
        required=True,
    )


class DoorOpenStatsGroup(graphene.ObjectType):
    """Door open stats group (e.g. aggregate data)"""

    class Meta:
        description = "Door open stats groups (e.g. aggregate data)"

    metrics = graphene.Field(DoorOpenStatsMetrics, required=True)
    dimensions = graphene.Field(DoorOpenStatsDimension, required=True)


class ProductionLineStatusMetrics(graphene.ObjectType):
    """Production line status group metrics"""

    uptime_duration_seconds = graphene.Int(
        required=True,
        description="Duration of production line uptime (in seconds)",
    )
    downtime_duration_seconds = graphene.Int(
        required=True,
        description="Duration of production line downtime (in seconds)",
    )
    unknown_duration_seconds = graphene.Int(
        required=True,
        description="Duration of time where production line status is unknown (in seconds)",
    )


class ProductionLineStatusDimension(graphene.ObjectType):
    """Production line status group dimensions."""

    datetime = graphene.DateTime(
        required=True,
        description=(
            "Production line group datetime truncated to the appropriate date part"
            + " based on the group_by property (e.g. hourly groups are truncated"
            + " to the hour, daily groups truncated to the day, etc.)."
            + " All values are localized to the event zone's timezone."
        ),
    )
    max_timestamp = graphene.DateTime(
        required=True,
        description="Max timestamp of all events contained in this aggregate group.",
    )
    organization = graphene.Field(
        graphene.lazy_import(
            "core.portal.organizations.graphql.types.OrganizationType"
        ),
        required=True,
    )
    zone = graphene.Field(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType"),
        required=True,
    )
    camera = graphene.Field(
        graphene.lazy_import("core.portal.devices.graphql.types.CameraType"),
        required=True,
    )


class ProductionLineStatusGroup(graphene.ObjectType):
    """Production line status group (e.g. aggregate data)"""

    class Meta:
        description = "Production line status groups (e.g. aggregate data)"

    metrics = graphene.Field(ProductionLineStatusMetrics, required=True)
    dimensions = graphene.Field(ProductionLineStatusDimension, required=True)


class ProductionLine(graphene.ObjectType):
    id = graphene.ID(required=True)
    uuid = graphene.String(required=True)
    name = graphene.String(required=True)
    camera = graphene.Field(
        graphene.lazy_import("core.portal.devices.graphql.types.CameraType"),
        required=True,
    )
    status_1h_groups = graphene.List(
        graphene.NonNull(ProductionLineStatusGroup),
        start_timestamp=graphene.DateTime(required=True),
        end_timestamp=graphene.DateTime(required=True),
        required=True,
        filters=graphene.List(FilterInputType),
    )

    # TODO: optimize this field via dataloader pattern
    #       when fetching lists of production lines
    @staticmethod
    def resolve_status_1h_groups(
        parent: "ProductionLine",
        info: graphene.ResolveInfo,
        start_timestamp: datetime,
        end_timestamp: datetime,
        filters: t.Optional[t.List[FilterInputType]] = None,
    ) -> t.List[ProductionLineStatusGroup]:
        """Resolves production lines aggregate data grouped by hour.

        Args:
            parent (ProductionLine): production line
            info (graphene.ResolveInfo): graphene context
            start_timestamp (datetime): start of data filter range
            end_timestamp (datetime): end of data filter range
            filters (OptionalList[FilterInputType]):
                A list of FilterInputType filters. Default to None

        Returns:
            List[ProductionLineStatusGroup]: list of production lines
        """

        # TODO: prevent serving too much data by validating a maximum time range for hourly groups

        production_line = ProductionLineModel.objects.get(uuid=parent.uuid)

        if not has_zone_permission(
            info.context.user, production_line.zone, ANALYTICS_READ
        ):
            return PermissionDenied(
                "You do not have permission to view production line status data."
            )

        aggregate_records = (
            production_line.production_line_aggregates.filter(
                group_by=AggregateGroup.HOUR,
                group_key__range=[start_timestamp, end_timestamp],
            )
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
            .prefetch_related("organization", "zone", "camera")
            .order_by("group_key")
        )

        return [
            ProductionLineStatusGroup(
                dimensions=ProductionLineStatusDimension(
                    datetime=record.group_key.astimezone(
                        production_line.zone.tzinfo
                    ),
                    max_timestamp=record.max_timestamp,
                    organization=record.organization,
                    zone=record.zone,
                    camera=record.camera,
                ),
                metrics=ProductionLineStatusMetrics(
                    uptime_duration_seconds=record.uptime_duration_s,
                    downtime_duration_seconds=record.downtime_duration_s,
                    unknown_duration_seconds=(
                        3600
                        - (
                            record.uptime_duration_s
                            + record.downtime_duration_s
                        )
                    ),
                ),
            )
            for record in aggregate_records
        ]

    incidents = DjangoConnectionField(
        IncidentType,
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
        order_by=graphene.String(),
    )

    @staticmethod
    def resolve_incidents(
        parent: "ProductionLine",
        info: graphene.ResolveInfo,
        *args,
        start_timestamp: t.Optional[datetime] = None,
        end_timestamp: t.Optional[datetime] = None,
        order_by: t.Optional[str] = None,
        **kwargs,
    ) -> QuerySet[IncidentType]:
        """Resolve production line incidents.

        Args:
            parent (ProductionLine): production line
            info (graphene.ResolveInfo): graphene context
            args: unused args
            start_timestamp (t.Optional[datetime], optional): start timestamp filter
            end_timestamp (t.Optional[datetime], optional): end timestamp filter
            order_by (t.Optional[str], optional): order by value
            kwargs: unused kwargs

        Returns:
            QuerySet[IncidentType]: production line incidents
        """
        del args, kwargs
        if not has_zone_permission(
            info.context.user,
            parent.zone,
            DOWNTIME_READ,
        ):
            return PermissionDenied(
                "You do not have permission to view production line details."
            )

        if not has_zone_permission(
            info.context.user,
            parent.zone,
            INCIDENTS_READ,
        ):
            return PermissionDenied(
                "You do not have permission to view incidents."
            )

        queryset = Incident.objects.filter(
            incident_type__key__iexact="PRODUCTION_LINE_DOWN",
            # TODO: find a better way to model the relationship between
            #       production lines and incidents
            data__track_uuid=parent.uuid,
        )
        if start_timestamp:
            queryset = queryset.filter(timestamp__gte=start_timestamp)
        if end_timestamp:
            queryset = queryset.filter(timestamp__lt=end_timestamp)

        # Add a calculated duration field to the queryset
        queryset = queryset.annotate(
            end_ms=Cast(
                KeyTextTransform("end_frame_relative_ms", "data"),
                fields.FloatField(),
            ),
            start_ms=Cast(
                KeyTextTransform("start_frame_relative_ms", "data"),
                fields.FloatField(),
            ),
        ).annotate(time_duration=F("end_ms") - F("start_ms"))
        # Order the queryset by the calculated duration field
        value = order_by if order_by else "-time_duration"
        queryset = queryset.order_by(value)
        return queryset
