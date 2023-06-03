import json
from datetime import date, datetime
from typing import List, Optional

import graphene
from django.contrib.auth.models import User
from django.core.cache import cache
from django.db.models import Count
from django.db.models.functions import (
    TruncDay,
    TruncHour,
    TruncMonth,
    TruncQuarter,
    TruncWeek,
    TruncYear,
)
from django.db.models.query import QuerySet
from graphene import ResolveInfo
from graphene_django import DjangoObjectType
from graphql_relay import to_global_id
from loguru import logger

from core.portal.accounts.graphql.types import UserType
from core.portal.accounts.permissions import INCIDENTS_READ
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_type import (
    CameraIncidentType as CameraIncidentTypeModel,
)
from core.portal.api.models.incident_type import (
    IncidentType as IncidentTypeModel,
)
from core.portal.api.models.incident_type import (
    OrganizationIncidentType as OrganizationIncidentTypeModel,
)
from core.portal.devices.graphql.types import CameraConfigNewModelType
from core.portal.incidents.graphql.enums import TaskStatusEnum
from core.portal.incidents.types import Filter
from core.portal.lib.graphql.enums import TimeBucketWidth
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.zones.models.zone import Zone

TIME_BUCKET_TRUNC_FN_MAP = {
    TimeBucketWidth.YEAR: TruncYear,
    TimeBucketWidth.QUARTER: TruncQuarter,
    TimeBucketWidth.MONTH: TruncMonth,
    TimeBucketWidth.WEEK: TruncWeek,
    TimeBucketWidth.DAY: TruncDay,
    TimeBucketWidth.HOUR: TruncHour,
}


class TagType(graphene.ObjectType):
    label = graphene.String()
    value = graphene.String(required=True)


class IncidentCategory(graphene.ObjectType):
    key = graphene.String()
    name = graphene.String()
    incident_types = graphene.List(
        graphene.lazy_import(
            "core.portal.incidents.graphql.types.IncidentTypeType"
        )
    )


class IncidentTypeType(DjangoObjectType):
    """Base incident type."""

    class Meta:
        model = IncidentTypeModel
        fields = ["id", "key", "name", "background_color"]

    key = graphene.String(required=True)
    name = graphene.String(required=True)
    category = graphene.String()
    background_color = graphene.String()

    def resolve_label(self, _: ResolveInfo) -> str:
        return self.value.replace("_", " ").title()


class OrganizationIncidentTypeType(DjangoObjectType):
    """Organization-specific incident type."""

    class Meta:
        model = OrganizationIncidentTypeModel
        fields = ["id", "key", "name", "background_color"]

    id = graphene.ID(required=True)
    key = graphene.String(required=True)
    name = graphene.String(required=True)
    background_color = graphene.String(required=True)


class CameraIncidentType(DjangoObjectType):
    """Camera-specific incident type."""

    class Meta:
        model = CameraIncidentTypeModel
        fields = ["id", "key", "name", "background_color", "description"]

    id = graphene.ID(required=True)
    key = graphene.String(required=True)
    name = graphene.String(required=True)
    background_color = graphene.String(required=True)
    description = graphene.String()


class IncidentType(DjangoObjectType):
    class Meta:
        model = Incident
        interfaces = [graphene.relay.Node]
        exclude = ["deleted_at", "cooldown_source", "review_status"]
        # TODO: utilize django-filter instead of custom *_filter arguments
        filter_fields: List[str] = []

    pk = graphene.Int(required=True)
    uuid = graphene.String(required=True)
    status = graphene.String()
    incident_type = graphene.Field(CameraIncidentType)
    bookmarked = graphene.Boolean(required=True)
    highlighted = graphene.Boolean(required=True)
    thumbnail_url = graphene.String(required=True)
    thumbnail_url_mrap = graphene.String(required=True)
    video_url = graphene.String(required=True)
    video_url_mrap = graphene.String(required=True)

    annotations_url = graphene.String(required=True)
    annotations_url_mrap = graphene.String(required=True)
    duration = graphene.Int()
    end_timestamp = graphene.DateTime()

    docker_image_tag = graphene.String()
    actor_ids = graphene.List(graphene.NonNull(graphene.String), required=True)
    assignees = graphene.List(
        graphene.lazy_import("core.portal.accounts.graphql.types.UserType"),
        required=True,
    )

    tags = graphene.List(TagType, required=True)
    camera_uuid = graphene.String(required=True)
    camera_config = graphene.Field(CameraConfigNewModelType)
    alerted = graphene.Boolean(required=True)

    def resolve_uuid(self, info: ResolveInfo) -> str:
        del info
        return self.uuid_wrapper

    @staticmethod
    def resolve_incident_type(
        parent: Incident, info: ResolveInfo
    ) -> Optional[CameraIncidentType]:
        def get_incident_type():
            return (
                parent.incident_type.camera_incident_types.filter(
                    camera=parent.camera,
                )
                .prefetch_related("incident_type__site_incident_types__site")
                .first()
            )

        cache_key = CameraIncidentTypeModel.cache_key(
            parent.incident_type_id,
            parent.camera_id,
        )
        return cache.get_or_set(cache_key, get_incident_type)

    def resolve_bookmarked(self, _: ResolveInfo) -> bool:
        return getattr(self, "bookmarked", False)

    @staticmethod
    def resolve_highlighted(parent: Incident, _: ResolveInfo) -> bool:
        """Resolve highlighted field.

        Args:
            parent (Incident): resolver parent object

        Returns:
            bool: true if incident is highlighted, otherwise false
        """
        return bool(parent.highlighted)

    def resolve_alerted(self, _: ResolveInfo) -> bool:
        return getattr(self, "alerted", False)

    @staticmethod
    def resolve_assignees(parent: Incident, info: ResolveInfo) -> List[User]:
        return parent.assigned_to.all()

    def resolve_tags(self, info: ResolveInfo) -> List[TagType]:
        tags = []

        if self.docker_image_tag:
            tags.append(TagType(label="image", value=self.docker_image_tag))
        if info.context.user.is_superuser:
            if self.experimental and self.incident_version:
                tags.append(
                    TagType(
                        label="incident_version", value=self.incident_version
                    )
                )

        return tags

    def resolve_camera_uuid(self, _: ResolveInfo) -> Optional[str]:
        return self.camera_uuid


class FilterInputType(graphene.InputObjectType):
    key = graphene.String()
    value_json = graphene.String()

    def to_filter_type(self) -> Filter:
        value = json.loads(self.value_json)
        if not isinstance(value, (bool, str, list)):
            raise RuntimeError(f"Invalid filter type: {type(self.value_json)}")
        return Filter(self.key, json.loads(self.value_json))

    @classmethod
    def to_filter_list(
        cls, filters: Optional[List["FilterInputType"]]
    ) -> List[Filter]:
        return [item.to_filter_type() for item in filters or []]


class IncidentAggregateMetrics(graphene.ObjectType):
    """Incident aggregate group metrics."""

    count = graphene.Int(
        required=True,
        description="Count of incidents in this group.",
    )


class IncidentAggregateDimensions(graphene.ObjectType):
    """Incident aggregate group dimensions."""

    datetime = graphene.DateTime(
        required=True,
        description=(
            "Incident aggregate group datetime truncated to the appropriate date part"
            + " based on the group_by property (e.g. hourly groups are truncated"
            + " to the hour, daily groups truncated to the day, etc.)."
            + " All values are localized to the parent zone's timezone."
        ),
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
    incident_type = graphene.Field(OrganizationIncidentTypeType, required=True)


class IncidentAggregateGroup(graphene.ObjectType):
    """Incident aggregate group."""

    class Meta:
        description = "Incident aggregate group."

    id = graphene.ID(required=True)

    @staticmethod
    def resolve_id(
        parent: "IncidentAggregateGroup",
        _: graphene.ResolveInfo,
    ) -> str:
        """Resolve object ID by generating opaque key comprised of of all dimension IDs.

        This key is intended to be opaque and primarily used for client-side cache keys,
        so it should never be passed back to the API.

        Args:
            parent (IncidentAggregateGroup): parent group instance

        Returns:
            str: unique object ID
        """
        dimension_ids = [
            parent.dimensions.datetime,
            parent.dimensions.organization.id,
            parent.dimensions.zone.id,
            parent.dimensions.camera.id,
            parent.dimensions.incident_type.id,
        ]
        delimiter = "$"
        unique_string = delimiter.join([str(value) for value in dimension_ids])
        return to_global_id(parent.__class__.__name__, unique_string)

    metrics = graphene.Field(IncidentAggregateMetrics, required=True)
    dimensions = graphene.Field(IncidentAggregateDimensions, required=True)


class SiteIncidentAnalytics(graphene.ObjectType):
    def __init__(self, site: Zone) -> None:
        super().__init__()
        self.site = site

    id = graphene.ID(required=True)
    name = graphene.String(required=True)

    @staticmethod
    def resolve_name(
        parent: "SiteIncidentAnalytics", _: graphene.ResolveInfo
    ) -> str:
        """Resolve name

        Args:
            parent (SiteIncidentAnalytics): parent object

        Returns:
            str: Site name
        """
        return parent.site.name

    @staticmethod
    def resolve_id(
        parent: "SiteIncidentAnalytics",
        _: graphene.ResolveInfo,
    ) -> str:
        """Resolve site analytics ID.

        Args:
            parent (SiteIncidentAnalytics): parent object

        Returns:
            str: ID
        """
        return to_global_id(parent.__class__.__name__, parent.site.id)

    incident_aggregate_groups = graphene.Field(
        graphene.List(graphene.NonNull(IncidentAggregateGroup)),
        start_date=graphene.Date(
            description="Start date inclusive",
        ),
        end_date=graphene.Date(
            description="End date, inclusive",
        ),
        start_timestamp=graphene.DateTime(
            description="Start timestamp, inclusive",
        ),
        end_timestamp=graphene.DateTime(
            description="End timestamp, inclusive",
        ),
        group_by=graphene.Argument(TimeBucketWidth, required=True),
        filters=graphene.List(FilterInputType),
    )

    @staticmethod
    def resolve_incident_aggregate_groups(
        parent: "SiteIncidentAnalytics",
        info: graphene.ResolveInfo,
        group_by: TimeBucketWidth,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        filters: Optional[List[FilterInputType]] = None,
    ) -> List[IncidentAggregateGroup]:
        """Resolve incident aggregate groups.

        Args:
            parent (SiteIncidentAnalytics): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (date): start date filter
            end_date (date): end date filter
            start_timestamp (datetime): start timestamp filter
            end_timestamp (datetime): end timestamp filter
            group_by (TimeBucketWidth): time bucket width
            filters (Optional[List[FilterInputType]]): additional filters

        Raises:
            RuntimeError: when invalid arguments provided

        Returns:
            List[IncidentAggregateGroup]: list of incident aggregate groups
        """

        if not parent.site or not has_zone_permission(
            info.context.user,
            parent.site,
            INCIDENTS_READ,
        ):
            return PermissionDenied(
                "You do not have permission to view incident data."
            )

        # Prefer timestamp filters over date filters
        # If dates are provided, convert them into localized start/end
        # timestamps for the requested date range
        if start_date and not start_timestamp:
            start_timestamp = datetime.combine(
                start_date, datetime.min.time()
            ).replace(tzinfo=parent.site.tzinfo)

        if end_date and not end_timestamp:
            end_timestamp = datetime.combine(
                end_date, datetime.max.time()
            ).replace(tzinfo=parent.site.tzinfo)

        if not start_timestamp and not end_timestamp:
            raise RuntimeError("Start and end filters are required.")

        time_bucket_trunc_fn = TIME_BUCKET_TRUNC_FN_MAP.get(group_by)

        if not time_bucket_trunc_fn:
            raise RuntimeError(f"Invalid group_by argument: {group_by}")

        camera_map = {item.id: item for item in parent.site.cameras.all()}
        incident_type_map = {
            item.key: item for item in parent.site.enabled_incident_types.all()
        }

        # TODO: use .aggregable_objects() instead of .objects() manager
        #       once data is backfilled
        records = (
            Incident.objects.for_site(parent.site)
            .from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .apply_filters(
                FilterInputType.to_filter_list(filters),
                info.context.user,
            )
            .annotate(
                time_bucket_start_timestamp=time_bucket_trunc_fn(
                    "timestamp",
                    tzinfo=parent.site.tzinfo,
                )
            )
            .values(
                "time_bucket_start_timestamp",
                "camera_id",
                "incident_type__key",
            )
            .annotate(count=Count("pk"))
            .order_by(
                "time_bucket_start_timestamp",
                "camera_id",
                "incident_type__key",
            )
        )

        results = []
        for record in records:
            time_bucket_start_timestamp = record.get(
                "time_bucket_start_timestamp"
            )
            camera_id = record.get("camera_id")
            camera = camera_map.get(camera_id)
            incident_type_key = record.get("incident_type__key")
            incident_type = incident_type_map.get(incident_type_key)

            # Skip records with invalid cameras or incident types
            if not camera or not incident_type:
                logger.warning(
                    f"invalid camera ID ({camera_id}) or"
                    + f" incident type ID ({incident_type_key})"
                    + f" for time bucket ({time_bucket_start_timestamp})"
                    + f" at site ({parent.site.key})"
                )
                continue

            results.append(
                IncidentAggregateGroup(
                    metrics=IncidentAggregateMetrics(
                        count=record.get("count", 0)
                    ),
                    dimensions=IncidentAggregateDimensions(
                        datetime=time_bucket_start_timestamp,
                        organization=parent.site.organization,
                        zone=parent.site,
                        camera=camera,
                        incident_type=incident_type,
                    ),
                )
            )

        return results


class TaskType(DjangoObjectType):
    class Meta:
        interfaces = [graphene.relay.Node]
        model = Incident
        exclude = ["deleted_at", "cooldown_source", "review_status"]
        fields: List[str] = []
        filter_fields: List[str] = []

    incident = graphene.Field(IncidentType)
    assigned_by = graphene.List(UserType)
    assigned_to = graphene.List(UserType)
    status = graphene.Field(TaskStatusEnum)

    @staticmethod
    def resolve_incident(
        parent: Incident, _: graphene.ResolveInfo
    ) -> Incident:
        """Resolve the task's incident field.

        Args:
            parent (Incident): parent incident.

        Returns:
            Incident: task incident.
        """
        return parent

    @staticmethod
    def resolve_assigned_by(
        parent: Incident, _: graphene.ResolveInfo
    ) -> "QuerySet[User]":
        """Resolve task assigned by user list.

        Args:
            parent (Incident): parent incident.

        Returns:
            QuerySet[User]: assigned by users.
        """
        return parent.assigned_by.all()

    @staticmethod
    def resolve_assigned_to(
        parent: Incident, _: graphene.ResolveInfo
    ) -> "QuerySet[User]":
        """Resolve task assigned to user list.

        Args:
            parent (Incident): parent incident.

        Returns:
            QuerySet[User]: assigned to users.
        """
        return parent.assigned_to.all()

    @staticmethod
    def resolve_status(
        parent: Incident, _: graphene.ResolveInfo
    ) -> TaskStatusEnum:
        """Resolve task status.

        Args:
            parent (Incident): parent incident.

        Returns:
            TaskStatusEnum: task status
        """
        if parent.status:
            return TaskStatusEnum.get(parent.status)
        return TaskStatusEnum.OPEN


class TaskStatsType(graphene.ObjectType):

    total_count = graphene.Int()
    open_count = graphene.Int()
    resolved_count = graphene.Int()

    @staticmethod
    def resolve_total_count(
        parent: "QuerySet[Incident]", _: graphene.ResolveInfo
    ) -> int:
        """Resolve total count.

        Args:
            parent (QuerySet[Incident]): parent incident queryset.

        Returns:
            int: total count.
        """
        return parent.count()

    @staticmethod
    def resolve_open_count(
        parent: "QuerySet[Incident]", _: graphene.ResolveInfo
    ) -> int:
        """Resolve open count.

        Args:
            parent (QuerySet[Incident]): parent incident queryset.

        Returns:
            int: open count.
        """
        return parent.exclude(status="resolved").count()

    @staticmethod
    def resolve_resolved_count(
        parent: "QuerySet[Incident]", _: graphene.ResolveInfo
    ) -> int:
        """Resolve resolved count.

        Args:
            parent (QuerySet[Incident]): parent incident queryset.

        Returns:
            int: resolved count.
        """
        return parent.filter(status="resolved").count()
