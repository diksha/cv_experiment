from datetime import datetime
from typing import List, Optional

import graphene
from graphene.types.generic import GenericScalar

from core.portal.analytics.services import get_series
from core.portal.incidents.graphql.types import FilterInputType


class PriorityCountType(graphene.ObjectType):
    """Aggregate counts by priority."""

    low_priority_count = graphene.Int(required=True)
    medium_priority_count = graphene.Int(required=True)
    high_priority_count = graphene.Int(required=True)


class StatusCountType(graphene.ObjectType):
    """Aggregate counts by status."""

    open_count = graphene.Int(required=True)
    resolved_count = graphene.Int(required=True)


class IncidentTypeCountType(graphene.ObjectType):
    """Key/value pair for mapping incident types to aggregate counts."""

    key = graphene.String(required=True)
    value = graphene.Int(required=True)


class SeriesType(graphene.ObjectType):
    """Wrapper type for grouping aggregate types by key."""

    key = graphene.DateTime()
    priority_counts = graphene.Field(PriorityCountType, required=True)
    status_counts = graphene.Field(StatusCountType, required=True)
    incident_type_counts = GenericScalar(required=True)


class AnalyticsType(graphene.ObjectType):
    """Wrapper for all top-level analytics fields."""

    series = graphene.List(
        SeriesType,
        from_utc=graphene.DateTime(required=True),
        to_utc=graphene.DateTime(required=True),
        group_by=graphene.String(required=True),
        filters=graphene.List(FilterInputType),
    )

    @staticmethod
    def resolve_series(
        root,
        info: graphene.ResolveInfo,
        from_utc: datetime,
        to_utc: datetime,
        group_by: str,
        filters: Optional[List[FilterInputType]] = None,
    ):
        """Resolves the analytics series field.

        Args:
            root (AnalyticsType): parent type
            info (graphene.ResolveInfo): graphene context
            from_utc (datetime): start of time range filter
            to_utc (datetime): end of time range filter
            group_by (str): how results should be grouped (HOUR, DAY, etc.)
            filters (OptionalList[FilterInputType]):
                A list of FilterInputType filters. Default to None

        Returns:
            SeriesType: series data
        """
        return get_series(
            zone=info.context.user.profile.site,
            organization=info.context.user.profile.current_organization,
            from_utc=from_utc,
            to_utc=to_utc,
            group_by=group_by,
            current_user=info.context.user,
            filters=filters,
        )
