# trunk-ignore-all(pylint/C0302): long module ok for now
# trunk-ignore-all(pylint/R0904): too many public methods ok for now
from datetime import date as py_date
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import graphene
from django.contrib.auth.models import User
from django.db.models import Count, DateField, F, Q, Window
from django.db.models.functions import RowNumber, TruncDate, TruncDay
from django.db.models.query import QuerySet
from django_cte import With
from graphene_django import DjangoConnectionField, DjangoObjectType
from graphql_relay import to_global_id
from loguru import logger

from core.portal.accounts.graphql.types import UserType
from core.portal.accounts.permissions import (
    ANALYTICS_READ,
    CAMERAS_READ,
    INCIDENTS_READ,
)
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.activity.graphql.types import SessionCount
from core.portal.activity.services import get_site_user_session_counts
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident
from core.portal.comments.graphql.types import CommentType
from core.portal.compliance.graphql.types import ComplianceType, ProductionLine
from core.portal.demos.data.constants import DEMO_SCORE_DATA
from core.portal.devices.graphql.types import CameraType
from core.portal.incidents.enums import IncidentPriority, TaskStatus
from core.portal.incidents.graphql.types import (
    FilterInputType,
    IncidentCategory,
    IncidentType,
    IncidentTypeType,
    SiteIncidentAnalytics,
)
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.types import PageInfo
from core.portal.lib.utils.score_utils import (
    handle_if_event_score_name_organization_override,
)
from core.portal.scores.graphql.types import Score, SiteScoreStats
from core.portal.scores.services import calculate_all_site_event_scores
from core.portal.zones.models import Zone


class ZoneIncidentStats(graphene.ObjectType):
    """Incident stats at a zone."""

    total_count = graphene.Int()
    resolved_count = graphene.Int()
    open_count = graphene.Int()
    high_priority_count = graphene.Int()
    medium_priority_count = graphene.Int()
    low_priority_count = graphene.Int()


class ZoneIncidentTypeStats(graphene.ObjectType):
    """Incident type stats at a zone."""

    incident_type = graphene.Field(IncidentTypeType)
    total_count = graphene.Int()
    max_total_count = graphene.Int(
        description="Highest totalCount value of all incident type stats returned in a query. Used by clients to compare totalCount with maxTotalCount for things like bar charts with relative widths per incident type."
    )


class IncidentCategoryStats(graphene.ObjectType):
    """Incident category stats."""

    category_key = graphene.String()
    total_count = graphene.Int()


class CameraStats(graphene.ObjectType):
    """Camera-related stats at a zone."""

    camera = graphene.Field(CameraType)
    category_stats = graphene.List(IncidentCategoryStats)


class AssigneeStats(graphene.ObjectType):
    """Assignee-related stats at a zone."""

    assignee = graphene.Field(UserType)
    open_count = graphene.Int()
    resolved_count = graphene.Int()
    resolved_time_avg_minutes = graphene.Int()


class ClientPreference(graphene.ObjectType):
    """Client preference key/value pair."""

    key = graphene.String(required=True)
    value = graphene.String(required=True)


class TimeBucketIncidentTypeCount(graphene.ObjectType):
    """Incident type counts for time buckets"""

    count = graphene.Int()
    incident_type = graphene.Field(IncidentTypeType)


class IncidentFeedItemTimeBucket(graphene.ObjectType):
    """Incident feed item data scoped to a time bucket."""

    key = graphene.String(required=True)
    title = graphene.String(required=True)
    start_hour = graphene.Int(
        required=True, description="Time bucket start hour (1-24)"
    )
    end_hour = graphene.Int(
        required=True, description="Time bucket end hour (1-24)"
    )
    duration_hours = graphene.Int(
        required=True, description="Time bucket duration in hours (1-24)"
    )
    start_timestamp = graphene.DateTime(
        required=True, description="Time bucket exact start timestamp"
    )
    end_timestamp = graphene.DateTime(
        required=True, description="Time bucket exact end timestamp"
    )
    incident_count = graphene.Int(
        required=True,
        description="Total incident count within this time bucket",
    )
    incident_counts = graphene.List(
        TimeBucketIncidentTypeCount,
        description="Total incident counts by type within time bucket",
    )
    latest_incidents = graphene.List(
        IncidentType, description="Latest N incidents within this time bucket"
    )


# TODO: move these classes to the appropriate module


class EmptyRangeFeedItem(graphene.ObjectType):
    """Feed item which represents a range of days without any incidents."""

    key = graphene.String(required=True)
    title = graphene.String(required=True)
    start_date = graphene.Date(required=True)
    end_date = graphene.Date(required=True)
    day_count = graphene.Int(
        required=True,
        description="Number of days covered by this feed item (at least 1)",
    )


class DailyIncidentsFeedItem(graphene.ObjectType):
    """Feed item which contains 1 days worth of data."""

    key = graphene.String(required=True)
    date = graphene.Date(required=True)
    time_buckets = graphene.List(
        IncidentFeedItemTimeBucket,
        required=True,
        description="Time buckets covering all 24 hours of a single day, localized to the organization/zone timezone",
    )


class IncidentFeedItem(graphene.Union):
    class Meta:
        types = (DailyIncidentsFeedItem, EmptyRangeFeedItem)


class IncidentFeedItemEdge(graphene.ObjectType):
    cursor = graphene.String(required=True)
    node = graphene.Field(IncidentFeedItem)


class IncidentFeedConnection(graphene.ObjectType):
    page_info = graphene.Field(PageInfo, required=True)
    edges = graphene.List(IncidentFeedItemEdge)


def generate_time_bucket_title(
    start_timestamp: datetime, end_timestamp: datetime
) -> str:
    start_hour = start_timestamp.strftime("%-I")
    start_meridiem = start_timestamp.strftime("%p").lower()
    end_hour = end_timestamp.strftime("%-I")
    end_meridiem = end_timestamp.strftime("%p").lower()

    if start_hour == end_hour:
        # 10pm
        return f"{start_hour}{start_meridiem}"

    rounded_end_timestamp = end_timestamp + timedelta(hours=1)
    rounded_end_hour = rounded_end_timestamp.strftime("%-I")
    rounded_end_meridiem = rounded_end_timestamp.strftime("%p").lower()
    if start_meridiem == end_meridiem:
        # 10-11pm
        return f"{start_hour}-{rounded_end_hour}{rounded_end_meridiem}"
    # 10pm-12am
    return f"{start_hour}{start_meridiem}-{rounded_end_hour}{rounded_end_meridiem}"


def init_daily_incidents_feed_item_edge(
    date: str, tzinfo: ZoneInfo
) -> IncidentFeedItemEdge:
    start_datetime_string = f"{date} 00:00:00"
    start_timestamp = datetime.strptime(
        start_datetime_string, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=tzinfo)
    end_timestamp = start_timestamp + timedelta(days=1, microseconds=-1)

    time_bucket = IncidentFeedItemTimeBucket(
        key=date,
        title=generate_time_bucket_title(start_timestamp, end_timestamp),
        start_hour=0,
        end_hour=23,
        duration_hours=24,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        incident_count=0,
        latest_incidents=[],
    )

    return IncidentFeedItemEdge(
        cursor=date,
        node=DailyIncidentsFeedItem(
            key=f"feed-item-{date}",
            date=date,
            time_buckets=[time_bucket],
        ),
    )


class ZoneType(DjangoObjectType):
    class Meta:
        model = Zone
        interfaces = [graphene.relay.Node]
        fields: List[str] = ["id", "key", "name", "timezone"]

    cameras = DjangoConnectionField(CameraType)
    incident_categories = graphene.List(IncidentCategory)
    incident_types = graphene.List(IncidentTypeType)
    compliance_types = graphene.List(ComplianceType)
    incident_stats = graphene.Field(
        ZoneIncidentStats,
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
    )
    incident_type_stats = graphene.List(
        ZoneIncidentTypeStats,
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
        filters=graphene.List(FilterInputType),
    )
    camera_stats = graphene.List(
        CameraStats,
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
    )
    assignee_stats = graphene.List(
        AssigneeStats,
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
        start_date=graphene.Date(),
        end_date=graphene.Date(),
    )
    recent_comments = DjangoConnectionField(CommentType)
    latest_activity_timestamp = graphene.DateTime()

    incident_feed = graphene.Field(
        IncidentFeedConnection,
        start_date=graphene.Date(),
        end_date=graphene.Date(),
        filters=graphene.List(FilterInputType),
        # TODO: delete this obsolete arg
        time_bucket_size_hours=graphene.Int(),
        after=graphene.String(),
    )

    users = DjangoConnectionField(UserType, required=True)
    assignable_users = DjangoConnectionField(UserType, required=True)
    timezone = graphene.String(required=True)

    client_preferences = graphene.List(
        graphene.NonNull(ClientPreference), required=True
    )

    overall_score = graphene.Field(
        Score,
        start_date=graphene.Date(
            required=True,
            description="Start date, inclusive",
        ),
        end_date=graphene.Date(
            required=True,
            description="End date, inclusive",
        ),
    )
    event_scores = graphene.Field(
        graphene.List(Score),
        start_date=graphene.Date(
            required=True,
            description="Start date, inclusive",
        ),
        end_date=graphene.Date(
            required=True,
            description="End date, inclusive",
        ),
    )
    session_count = graphene.Field(
        graphene.NonNull(SessionCount),
        start_date=graphene.Date(
            required=True,
            description="Start date, inclusive",
        ),
        end_date=graphene.Date(
            required=True,
            description="End date, inclusive",
        ),
    )

    @staticmethod
    def resolve_session_count(
        parent: Zone,
        info: graphene.ResolveInfo,
        start_date: py_date,
        end_date: py_date,
    ) -> SessionCount:
        """Resolve session count
        Args:
            parent (OrganizationType): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (date): start of event date filter range
            end_date (date): end of event date filter range
        Returns:
            SessionCount: A SessionCount object
        """
        return SessionCount(
            users=get_site_user_session_counts(parent, start_date, end_date),
        )

    @staticmethod
    def resolve_client_preferences(
        parent: Zone, _: graphene.ResolveInfo
    ) -> List[ClientPreference]:
        """Resolve client preferences.

        Args:
            parent (Zone): parent site

        Returns:
            List[ClientPreference]: list of client preferences
        """
        config = parent.config or {}
        preferences = config.get("client_preferences", {})
        if isinstance(preferences, dict):
            return [
                ClientPreference(key=key, value=str(value))
                for (key, value) in preferences.items()
            ]
        return []

    is_active = graphene.Boolean(required=True)

    @staticmethod
    def resolve_is_active(parent: Zone, _: graphene.ResolveInfo) -> bool:
        """Resolves the is_active field.

        Args:
            parent (Zone): parent zone

        Returns:
            bool: true if zone is active, otherwise false
        """
        return bool(parent.active)

    @staticmethod
    def resolve_users(
        root: Zone,
        info: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> QuerySet[User]:
        return root.active_users

    @staticmethod
    def resolve_assignable_users(
        root: Zone,
        *_: None,
        **__: None,
    ) -> QuerySet[User]:
        """
        Returns all the users who can be assigned tasks in the specified zone
        Args:
            root (Zone): The relevant zone
            *_: Nothing
            **__: Nothing

        Returns:
            QuerySet[User]: A QuerySet that resolves to retrieve all the users
            in the site where is_assignable=True
        """
        return root.assignable_users

    # trunk-ignore-all(pylint/R0915): allow too-many-statements
    # trunk-ignore-all(pylint/R0912): allow too-many-branches
    @staticmethod
    def resolve_incident_feed(
        root: Zone,
        info: graphene.ResolveInfo,
        start_date: py_date,
        end_date: py_date,
        filters: Optional[List[FilterInputType]] = None,
        after: Optional[str] = None,
        time_bucket_size_hours: Optional[int] = None,
    ) -> IncidentFeedConnection:
        """Resolves incident feed requests."""
        # TODO: delete this obsolete kwarg
        del time_bucket_size_hours

        start_date_timestamp = (
            datetime.combine(start_date, datetime.min.time()).replace(
                tzinfo=root.tzinfo
            )
            if start_date
            else None
        )

        end_date_timestamp = (
            datetime.combine(end_date, datetime.max.time()).replace(
                tzinfo=root.tzinfo
            )
            if end_date
            else None
        )

        MAX_INCIDENTS_PER_DAY = 20
        # The +1 logic ensures we return at least 2 days worth of data given
        # the date truncation logic further down in this resolver which drops
        # the last day's worth of data from each response to ensure consistent
        # cursor behavior.
        MAX_INCIDENTS_PER_DB_QUERY = MAX_INCIDENTS_PER_DAY * 2 + 1
        TRUNC_DATE_FIELD_NAME = "trunc_date"
        TIME_BUCKET_INCIDENT_COUNT_FIELD_NAME = "time_bucket_incident_count"
        TIME_BUCKET_RANK_FIELD_NAME = "time_bucket_rank"
        TIME_BUCKET_RANK_LTE_FILTER = f"{TIME_BUCKET_RANK_FIELD_NAME}__lte"
        CURSOR_LT_FILTER = f"{TRUNC_DATE_FIELD_NAME}__lt"

        annotation_dict = {
            TRUNC_DATE_FIELD_NAME: TruncDate(
                "timestamp", tzinfo=root.tzinfo, output_field=DateField()
            ),
            TIME_BUCKET_RANK_FIELD_NAME: Window(
                expression=RowNumber(),
                partition_by=[F(TRUNC_DATE_FIELD_NAME)],
                order_by=F("timestamp").desc(),
            ),
            TIME_BUCKET_INCIDENT_COUNT_FIELD_NAME: Window(
                expression=Count("id"),
                partition_by=[F(TRUNC_DATE_FIELD_NAME)],
            ),
        }

        # NOTE: CTE usage required here as we can't filter by window functions
        cte = With(
            root.incidents.for_user(info.context.user)
            .from_timestamp(start_date_timestamp)
            .to_timestamp(end_date_timestamp)
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
            .annotate(**annotation_dict)
        )

        filter_dict = {
            TIME_BUCKET_RANK_LTE_FILTER: MAX_INCIDENTS_PER_DAY,
        }

        queryset = (
            cte.queryset()
            .with_cte(cte)
            .filter(**filter_dict)
            .order_by("-timestamp")
        )

        if after:
            after_filter_dict = {CURSOR_LT_FILTER: after}
            queryset = queryset.filter(**after_filter_dict)

        # Execute the query
        results = list(queryset[:MAX_INCIDENTS_PER_DB_QUERY])

        # Imperfect way of determining if more data exists beyond current batch
        maybe_has_next_page = len(results) == MAX_INCIDENTS_PER_DB_QUERY

        # This is a crude strategy for ensuring consistent cursor behavior.
        # If the database query returns multiple days worth of data, we
        # ignore the last day's data. This ensures we don't return partial days.
        # The DB query params should ensure we get at least one days worth
        # of data, so we will always be able to return at least one day.
        truncate_results_at_date = None
        if len(results) >= MAX_INCIDENTS_PER_DB_QUERY:
            first_date = getattr(results[0], TRUNC_DATE_FIELD_NAME)
            last_date = getattr(results[-1], TRUNC_DATE_FIELD_NAME)
            if first_date != last_date:
                truncate_results_at_date = last_date

        # Re-fetch incidents with pre-fetched fields and build a lookup map.
        # We do this in a separate query because django-cte appears to break
        # prefetch behavior, and without prefetching these fields the
        # serialization process leads to N+1 query behavior. This could also
        # be solved with dataloaders but graphene doesn't support them :(
        incident_lookup = {}
        incident_ids = map(lambda x: x.id, results)
        if incident_ids:
            hydrated_incidents = (
                root.incidents.for_user(info.context.user)
                .with_bookmarked_flag(info.context.user)
                .filter(id__in=incident_ids)
                .prefetch_related(
                    "camera",
                    "assigned_to",
                    "organization",
                    "incident_type",
                    "incident_type__organization_incident_types",
                )
            )
            for incident in hydrated_incidents:
                incident_lookup[incident.id] = incident

        # Generate response and fill in empty time ranges
        edge_map = {}
        last_date = None
        start_cursor = None
        end_cursor = None

        # Get incident counts grouped by time buckets
        enabled_incident_type_keys = {
            t.key: t.incident_type for t in root.enabled_incident_types.all()
        }
        aggregate_incident_count = (
            root.incidents.for_user(info.context.user)
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
            .annotate(
                day=TruncDay(
                    "timestamp",
                    tzinfo=root.tzinfo,
                )
            )
            .values("day", "incident_type__key")
            .order_by()
            .annotate(count=Count("pk"))
        )

        incident_type_counts_by_date = {}
        for incident_count in aggregate_incident_count:
            time_bucket = incident_count["day"]
            incident_type_key = enabled_incident_type_keys.get(
                incident_count["incident_type__key"], None
            )

            if (
                incident_type_counts_by_date.get(time_bucket, None)
                and incident_type_key
            ):
                incident_type_counts_by_date[time_bucket].update(
                    {incident_type_key: incident_count["count"]}
                )
            else:
                incident_type_counts_by_date[time_bucket] = {
                    incident_type_key: incident_count["count"]
                }

        for record in results:
            date = getattr(record, TRUNC_DATE_FIELD_NAME)

            if truncate_results_at_date and truncate_results_at_date == date:
                break

            start_cursor = max(start_cursor, date) if start_cursor else date
            end_cursor = min(end_cursor, date) if end_cursor else date

            if date not in edge_map:
                # If there is any gap between the current date and the
                # previously processed date, add an empty range feed item
                if last_date:
                    days_since_last_activity = (last_date - date).days
                    if days_since_last_activity > 1:
                        key = f"{date}_empty_range"
                        start_date = date + timedelta(days=1)
                        end_date = last_date - timedelta(days=1)
                        edge_map[key] = IncidentFeedItemEdge(
                            cursor=last_date,
                            node=EmptyRangeFeedItem(
                                key=key,
                                start_date=start_date,
                                end_date=end_date,
                                day_count=days_since_last_activity - 1,
                            ),
                        )

                # Initialize an edge for the current incident's date
                edge_map[date] = init_daily_incidents_feed_item_edge(
                    date, root.tzinfo
                )

            # Add the incident to the appropriate time bucket
            day_time_buckets = edge_map[date].node.time_buckets
            if len(day_time_buckets) != 1:
                logger.error(
                    f"Expected exactly 1 time bucket but found {len(day_time_buckets)} for key: {date}"
                )
                continue

            time_bucket = day_time_buckets[0]
            time_bucket_incident_counts = incident_type_counts_by_date.get(
                time_bucket.start_timestamp.replace(tzinfo=root.tzinfo),
                None,
            )
            if time_bucket_incident_counts:
                time_bucket.incident_counts = [
                    TimeBucketIncidentTypeCount(
                        count=count, incident_type=incident_type
                    )
                    for incident_type, count in time_bucket_incident_counts.items()
                ]
            time_bucket.incident_count = getattr(
                record, TIME_BUCKET_INCIDENT_COUNT_FIELD_NAME
            )
            time_bucket.latest_incidents.append(incident_lookup[record.id])
            last_date = date

        return IncidentFeedConnection(
            page_info=PageInfo(
                has_previous_page=False,
                has_next_page=maybe_has_next_page,
                start_cursor=start_cursor,
                end_cursor=end_cursor,
            ),
            edges=edge_map.values(),
        )

    highlighted_events = DjangoConnectionField(IncidentType)

    @staticmethod
    def resolve_highlighted_events(
        parent: "Zone",
        info: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> QuerySet[Incident]:
        """Resolve highlighted events for a zone.

        Args:
            parent (Zone): parent zone
            info (graphene.ResolveInfo): graphene context
            args: unused args
            kwargs: unused kwargs

        Returns:
            QuerySet[Incident]: list of events (incidents)
        """
        del args, kwargs
        if not has_zone_permission(info.context.user, parent, INCIDENTS_READ):
            return PermissionDenied(
                "You do not have permission to view events for this zone."
            )
        return parent.incidents.filter(highlighted=True).order_by("-timestamp")

    highlighted_events_count = graphene.Int(
        required=True,
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
    )

    @staticmethod
    def resolve_highlighted_events_count(
        parent: "Zone",
        info: graphene.ResolveInfo,
        *args,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> int:
        """Resolve highlighted events count for a zone.

        Args:
            parent (Zone): parent zone
            info (graphene.ResolveInfo): graphene context
            start_timestamp (datetime): start of time range filter
            end_timestamp (datetime): end of time range filter
            args: unused args
            kwargs: unused kwargs

        Returns:
            int: count of highlighted events within the specified time range
        """
        del args, kwargs
        if not has_zone_permission(info.context.user, parent, INCIDENTS_READ):
            return PermissionDenied(
                "You do not have permission to view events for this zone."
            )
        return (
            parent.incidents.from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .filter(highlighted=True)
            .count()
        )

    @staticmethod
    def resolve_cameras(
        root: Zone,
        info: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> QuerySet[CameraType]:
        del args, kwargs
        if not has_zone_permission(
            info.context.user,
            root,
            CAMERAS_READ,
        ):
            raise PermissionDenied(
                "You do not have permission to view this zone's cameras."
            )
        return root.cameras.prefetch_related("camera_incident_types").all()

    @staticmethod
    def resolve_incident_categories(
        root: Zone,
        _: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> List[IncidentCategory]:
        """Returns all incident categories which are enabled at this zone."""
        del args, kwargs
        categories = {}
        for incident_type in root.enabled_incident_types.all():
            if incident_type.category not in categories:
                categories[incident_type.category] = []
            categories[incident_type.category].append(incident_type)

        return [
            # Have to call .incident_type because enabled_incident_types returns
            # OrganizationIncidentType instead of IncidentType.
            IncidentCategory(
                key=key,
                name=key.title(),
                incident_types=[
                    incident_type.incident_type
                    for incident_type in incident_types
                ],
            )
            for key, incident_types in categories.items()
        ]

    @staticmethod
    def resolve_incident_types(
        root: Zone,
        _: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> List[IncidentTypeType]:
        del args, kwargs
        return [
            # Have to call .incident_type because enabled_incident_types returns
            # OrganizationIncidentType instead of IncidentType.
            incident_type.incident_type
            for incident_type in root.enabled_incident_types.all()
        ]

    @staticmethod
    def resolve_compliance_types(
        root: Zone,
        *_: None,
        **__: None,
    ) -> List[ComplianceType]:
        """Returns enabled compliance types for a zone.

        Args:
            root (Zone): compliance types will be scoped to this zone
            _ (None): unused args
            __ (None): unused keyword args

        Returns:
            List[ComplianceType]: list of compliance types
        """
        compliance_types = [
            ComplianceType(
                id=zone_compliance_type.id,
                key=zone_compliance_type.compliance_type.key,
                name=zone_compliance_type.name,
            )
            for zone_compliance_type in root.enabled_zone_compliance_types
        ]
        return compliance_types

    @staticmethod
    def resolve_incident_stats(
        root: Zone,
        info: graphene.ResolveInfo,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
    ) -> ZoneIncidentStats:
        aggregation_dict = {
            "total_count": Count("pk"),
            "resolved_count": Count(
                "pk", filter=Q(status=TaskStatus.RESOLVED.value)
            ),
            "open_count": Count("pk", filter=Q(status=TaskStatus.OPEN.value)),
            "high_priority_count": Count(
                "pk", filter=Q(priority=IncidentPriority.HIGH.value)
            ),
            "medium_priority_count": Count(
                "pk", filter=Q(priority=IncidentPriority.MEDIUM.value)
            ),
            "low_priority_count": Count(
                "pk", filter=Q(priority=IncidentPriority.LOW.value)
            ),
        }
        counts = (
            root.incidents.for_user(info.context.user)
            .from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .aggregate(**aggregation_dict)
        )
        return ZoneIncidentStats(**counts)

    @staticmethod
    def resolve_incident_type_stats(
        root: Zone,
        info: graphene.ResolveInfo,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        filters: Optional[List[FilterInputType]] = None,
    ) -> List[ZoneIncidentTypeStats]:
        INCIDENT_TYPE_KEY = "incident_type__key"
        TOTAL_COUNT = "total_count"
        annotation_dict = {TOTAL_COUNT: Count("pk")}
        count_qs = (
            root.incidents.for_user(info.context.user)
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
            .from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .values(INCIDENT_TYPE_KEY)
            .annotate(**annotation_dict)
            .order_by(TOTAL_COUNT)
        )

        count_map = {}
        max_total_count = 0
        for item in count_qs:
            count_map[item[INCIDENT_TYPE_KEY]] = item[TOTAL_COUNT]
            max_total_count = max(max_total_count, item[TOTAL_COUNT])

        return [
            ZoneIncidentTypeStats(
                incident_type=org_incident_type.incident_type,
                total_count=count_map.get(org_incident_type.key, 0),
                max_total_count=max_total_count,
            )
            for org_incident_type in root.enabled_incident_types
        ]

    @staticmethod
    def resolve_camera_stats(
        root: Zone,
        info: graphene.ResolveInfo,
        start_timestamp: Optional[datetime],
        end_timestamp: Optional[datetime],
    ) -> List[CameraStats]:
        CATEGORY_COL = "incident_type__category"
        CAMERA_ID_COL = "camera_id"
        GROUP_BY_COLS = [CATEGORY_COL, CAMERA_ID_COL]
        TOTAL_COUNT_COL = "total_count"

        annotation_dict = {TOTAL_COUNT_COL: Count("pk")}
        incident_count_qs = (
            root.incidents.for_user(info.context.user)
            .from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .values(*GROUP_BY_COLS)
            .annotate(**annotation_dict)
            .order_by(*GROUP_BY_COLS)
        )

        camera_map = {}
        for row in incident_count_qs:
            camera_id = row.get(CAMERA_ID_COL, None)
            if not camera_id:
                continue
            if camera_id not in camera_map:
                camera_map[camera_id] = []
            camera_map[camera_id].append(
                IncidentCategoryStats(
                    category_key=row.get(CATEGORY_COL, "Unknown"),
                    total_count=row.get(TOTAL_COUNT_COL, 0),
                )
            )

        return [
            CameraStats(
                camera=camera, category_stats=camera_map.get(camera.id)
            )
            for camera in root.cameras.all()
        ]

    @staticmethod
    def resolve_assignee_stats(
        root: Zone,
        info: graphene.ResolveInfo,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        start_date: Optional[py_date] = None,
        end_date: Optional[py_date] = None,
    ) -> List[AssigneeStats]:
        OPEN_COUNT_COL = "open_count"
        RESOLVED_COUNT_COL = "resolved_count"

        # If dates are provided, localize them to start/end
        # timestamps for the current zone
        if start_date and end_date:
            start_timestamp = datetime.combine(
                start_date, datetime.min.time()
            ).astimezone(root.tzinfo)
            end_timestamp = datetime.combine(
                end_date, datetime.max.time()
            ).astimezone(root.tzinfo)

        valid_time_filters = bool(start_timestamp) and bool(end_timestamp)

        if not valid_time_filters:
            raise RuntimeError("Start and end time filters are required")

        annotation_dict = {
            OPEN_COUNT_COL: Count(
                "assignee",
                filter=Q(
                    assignee__incident__status=TaskStatus.OPEN.value,
                    assignee__incident__timestamp__gte=start_timestamp,
                    assignee__incident__timestamp__lte=end_timestamp,
                    assignee__incident__zone_id=info.context.user.profile.site.id,
                ),
            ),
            RESOLVED_COUNT_COL: Count(
                "assignee",
                filter=Q(
                    assignee__incident__status=TaskStatus.RESOLVED.value,
                    assignee__incident__timestamp__gte=start_timestamp,
                    assignee__incident__timestamp__lte=end_timestamp,
                    assignee__incident__zone_id=info.context.user.profile.site.id,
                ),
            ),
        }
        assignee_qs = root.assignable_users.annotate(**annotation_dict)

        return [
            AssigneeStats(
                assignee=user,
                open_count=getattr(user, OPEN_COUNT_COL),
                resolved_count=getattr(user, RESOLVED_COUNT_COL),
            )
            for user in assignee_qs.all()
        ]

    @staticmethod
    def resolve_recent_comments(
        root: Zone,
        info: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> QuerySet[Comment]:
        del args, kwargs
        return Comment.objects.filter(
            incident__organization=info.context.user.profile.current_organization,
            incident__camera__zone_id__in=[info.context.user.profile.site],
        ).order_by("-created_at")

    @staticmethod
    def resolve_latest_activity_timestamp(
        root: Zone,
        info: graphene.ResolveInfo,
        *args,
        **kwargs,
    ) -> Optional[datetime]:
        del args, kwargs
        latest_activity = (
            Comment.objects.filter(
                incident__organization=info.context.user.profile.current_organization,
                incident__camera__zone_id__in=[info.context.user.profile.site],
            )
            .order_by("-created_at")
            .first()
        )

        if latest_activity:
            return latest_activity.created_at
        return None

    @staticmethod
    def resolve_timezone(
        root: Zone,
        info: graphene.ResolveInfo,
    ) -> str:
        """Resolver for zone timezone

        Args:
            root (Zone): instance of root object
            info (graphene.ResolveInfo): info object that stores context

        Returns:
            str: timezone info of the site
        """
        del info
        return root.tzinfo

    production_lines = graphene.List(
        graphene.NonNull(ProductionLine),
        filters=graphene.List(FilterInputType),
    )

    @staticmethod
    def resolve_production_lines(
        parent: "Zone",
        info: graphene.ResolveInfo,
        filters: Optional[List[FilterInputType]] = None,
    ) -> List[ProductionLine]:
        """Resolves zone production lines.

        Args:
            parent (Zone): parent zone
            info (graphene.ResolveInfo): graphene context
            filters (OptionalList[FilterInputType]): A list of FilterInputType filters.
                Default to None

        Returns:
            List[ProductionLine]: list of production lines
        """
        if not has_zone_permission(info.context.user, parent, ANALYTICS_READ):
            return PermissionDenied(
                "You do not have permission to view production line status data."
            )
        queryset = parent.production_lines.apply_filters(
            FilterInputType.to_filter_list(filters), info.context.user
        )
        return [
            ProductionLine(
                # trunk-ignore(pylint/W0212): only way to get graphene class name
                id=to_global_id(ProductionLine._meta.name, production_line.id),
                uuid=production_line.uuid,
                name=production_line.name,
                camera=production_line.camera,
            )
            for production_line in queryset.prefetch_related(
                "camera"
            ).order_by("name")
        ]

    incident_analytics = graphene.Field(
        graphene.NonNull(SiteIncidentAnalytics)
    )

    @staticmethod
    def resolve_incident_analytics(
        parent: Zone,
        info: graphene.ResolveInfo,
    ) -> SiteIncidentAnalytics:
        """Resolve incident analytics.

        Args:
            parent (Zone): parent zone
            info (graphene.ResolveInfo): graphene context

        Returns:
            IncidentAnalytics: incident analytics instance
        """
        del info
        return SiteIncidentAnalytics(site=parent)

    incidents = DjangoConnectionField(
        IncidentType,
        start_date=graphene.Date(),
        end_date=graphene.Date(),
        start_timestamp=graphene.DateTime(),
        end_timestamp=graphene.DateTime(),
        filters=graphene.List(FilterInputType),
    )

    @staticmethod
    def resolve_incidents(
        parent: Zone,
        info: graphene.ResolveInfo,
        *args,
        start_date: Optional[py_date] = None,
        end_date: Optional[py_date] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        filters: Optional[List[FilterInputType]] = None,
        **kwargs,
    ) -> QuerySet[Incident]:
        """Resolve zone incidents.

        Args:
            parent (Zone): zone
            info (graphene.ResolveInfo): graphene context
            args: unused args
            start_date (Optional[py_date]): start date filter
            end_date (Optional[py_date]): end date filter
            start_timestamp (Optional[datetime]): start timestamp filter
            end_timestamp (Optional[datetime]): end timestamp filter
            filters (Optional[List[FilterInputType]]): incident filters
            kwargs: unused kwargs

        Returns:
            QuerySet[Incident]: incident queryset which gets paginated by graphene
        """
        del args, kwargs

        if not has_zone_permission(info.context.user, parent, INCIDENTS_READ):
            return PermissionDenied(
                "You do not have permission to view incidents for this zone."
            )

        # If dates are provided, localize them to start/end
        # timestamps for the current zone
        if start_date:
            start_timestamp = datetime.combine(
                start_date, datetime.min.time()
            ).astimezone(parent.tzinfo)

        if end_date:
            end_timestamp = datetime.combine(
                end_date, datetime.max.time()
            ).astimezone(parent.tzinfo)

        queryset = (
            Incident.objects.for_site(parent)
            .from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
            .with_bookmarked_flag(info.context.user)
        )

        # TODO: add support for different order_by criteria
        return queryset.prefetch_related("camera", "incident_type").order_by(
            "-timestamp"
        )

    site_score_stats = graphene.Field(
        SiteScoreStats,
        deprecation_reason="Use 'overallScore' and 'eventScores' fields",
    )

    @staticmethod
    def resolve_site_score_stats(
        parent: Zone,
        info: graphene.ResolveInfo,
    ) -> SiteScoreStats:
        """Resolves all the site scores for the given site

        Args:
            parent (Zone): the site
            info (graphene.ResolveInfo): graphene context

        Returns:
            SiteScoreStats: site score and event scores
        """
        return SiteScoreStats(site=parent)

    @staticmethod
    def resolve_overall_score(
        parent: Zone,
        info: graphene.ResolveInfo,
        start_date: py_date,
        end_date: py_date,
    ) -> Optional[Score]:
        """Resolve site score

        Args:
            parent (Zone): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (py_date): start of event date filter range
            end_date (py_date): end of event date filter range

        Returns:
            Score: A score object
        """
        demo_data = DEMO_SCORE_DATA["sites"].get(parent.name)
        if demo_data is not None:
            return demo_data["overallScore"]

        results = calculate_all_site_event_scores(
            site_ids=[parent.id],
            start_date=start_date,
            end_date=end_date,
        )
        total = 0
        num_event_scores = 0
        for result in results:
            if result.get("site_id") == parent.id:
                total += result.get("calculated_score")
                num_event_scores += 1

        if num_event_scores == 0:
            return None

        score_value = total / num_event_scores
        return Score(value=int(score_value), label=parent.name)

    @staticmethod
    def resolve_event_scores(
        parent: Zone,
        info: graphene.ResolveInfo,
        start_date: py_date,
        end_date: py_date,
    ) -> list[Score]:
        """Resolve event type scores

        Args:
            parent (Zone): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (py_date): start of event date filter range
            end_date (py_date): end of event date filter range

        Returns:
            list[Score]: A list of incident type score object
        """
        demo_data = DEMO_SCORE_DATA["sites"].get(parent.name)
        if demo_data is not None:
            return demo_data["eventScores"]

        results = calculate_all_site_event_scores(
            site_ids=[parent.id],
            start_date=start_date,
            end_date=end_date,
        )
        event_scores = []
        for result in results:
            if result.get("site_id") == parent.id:
                event_scores.append(
                    Score(
                        label=result.get("score_name"),
                        value=result.get("calculated_score"),
                    )
                )

        handle_if_event_score_name_organization_override(
            organization_key=parent.organization.key, event_scores=event_scores
        )

        return event_scores
