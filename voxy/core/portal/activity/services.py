import typing as t
from datetime import date, datetime
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.db.models.query import QuerySet

from core.portal.activity.graphql.types import (
    SessionSiteCount,
    SessionUserCount,
)
from core.portal.api.models.organization import Organization
from core.portal.zones.models.zone import Zone


def build_session_filter(
    start_date: date,
    end_date: date,
    tzinfo: ZoneInfo,
    organization_id: t.Optional[int] = None,
    site_id: t.Optional[int] = None,
    user_queryset: t.Optional[QuerySet[User]] = None,
) -> Q:
    """Build a filter for session start and end timestamps.

    Args:
        start_date (date): start of date filter range
        end_date (date): end of date filter range
        tzinfo (ZoneInfo): timezone info used to localize timestamps
        organization_id (int, optional): organization id. Defaults to None.
        site_id (int, optional): site id. Defaults to None.
        user_queryset (QuerySet[User], optional): user queryset. Defaults to None.

    Returns:
        Q: session filter
    """

    # Convert start date to localized start-of-day timestamp
    start_timestamp = datetime.combine(
        start_date, datetime.min.time()
    ).replace(tzinfo=tzinfo)

    # Convert end date to localized end-of-day timestamp
    end_timestamp = datetime.combine(end_date, datetime.max.time()).replace(
        tzinfo=tzinfo
    )

    covered_start_timestamp = Q(
        user_sessions__start_timestamp__gte=start_timestamp,
        user_sessions__start_timestamp__lte=end_timestamp,
    )

    covered_end_timestamp = Q(
        user_sessions__end_timestamp__gte=start_timestamp,
        user_sessions__end_timestamp__lte=end_timestamp,
    )

    # Build org/site filter
    org_site_filter = {}

    if organization_id:
        org_site_filter["user_sessions__organization_id"] = organization_id

    if site_id:
        org_site_filter["user_sessions__site_id"] = site_id

    if user_queryset:
        org_site_filter["user_sessions__user__in"] = user_queryset

    # Include any sessions which overlap with requested date range
    return Q(**org_site_filter) & (
        covered_start_timestamp | covered_end_timestamp
    )


def get_organization_weekly_average_site_session_counts(
    organization: Organization,
    start_date: date,
    end_date: date,
) -> t.List[SessionSiteCount]:
    """Get organization weekly average site session counts.

    Args:
        organization (Organization): organization
        start_date (date): start date filter
        end_date (date): end date filter

    Returns:
        t.List[SessionSiteCount]: list of session counts
    """
    time_delta = end_date - start_date
    day_count = max(time_delta.days, 1)
    week_count = max(int(day_count / 7), 1)

    # Build queryset to filter by active organization users,
    # primarily to exclude internal Voxel employee sessions
    user_queryset = organization.active_users.values_list("id", flat=True)

    session_filter = build_session_filter(
        start_date,
        end_date,
        organization.tzinfo,
        user_queryset=user_queryset,
    )

    return [
        SessionSiteCount(site=site, value=int(site.session_count / week_count))
        for site in organization.zones.filter(active=True)
        .annotate(
            session_count=Count(
                "user_sessions",
                filter=session_filter,
            )
        )
        .order_by("-session_count")
    ]


def get_organization_user_session_counts(
    organization: Organization,
    start_date: date,
    end_date: date,
) -> t.List[SessionUserCount]:
    """Get organization user session counts.

    Args:
        organization (Organization): organization
        start_date (date): start date filter
        end_date (date): end date filter

    Returns:
        List[SessionUserCount]: List of SessionUserCount objects
    """
    session_filter = build_session_filter(
        start_date,
        end_date,
        organization.tzinfo,
        organization_id=organization.id,
    )
    return [
        SessionUserCount(user=user, value=user.session_count)
        for user in organization.active_users.annotate(
            session_count=Count(
                "user_sessions",
                filter=session_filter,
            )
        ).order_by("-session_count")
    ]


def get_site_user_session_counts(
    site: Zone,
    start_date: date,
    end_date: date,
) -> t.List[SessionUserCount]:
    """Given a site, fetch all the site-specific session counts.

    Args:
        site (Zone): site
        start_date (date): start date filter
        end_date (date): end date filter

    Returns:
        List[SessionUserCount]: List of SessionUserCount objects
    """
    session_filter = build_session_filter(
        start_date,
        end_date,
        site.tzinfo,
        site_id=site.id,
    )
    return [
        SessionUserCount(user=user, value=user.session_count)
        for user in site.active_users.annotate(
            session_count=Count(
                "user_sessions",
                filter=session_filter,
            )
        ).order_by("-session_count")
    ]
