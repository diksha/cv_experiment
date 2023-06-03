#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import datetime
from typing import List, Optional

from django.contrib.auth.models import User
from django.db.models import Count, DateTimeField, Q
from django.db.models.functions import (
    JSONObject,
    TruncDay,
    TruncHour,
    TruncMonth,
)
from django.db.models.query import QuerySet
from django_stubs_ext import WithAnnotations

from core.portal.api.models.incident import Incident
from core.portal.api.models.organization import Organization
from core.portal.incidents.graphql.types import FilterInputType
from core.portal.zones.models.zone import Zone


def get_series(
    *,
    organization: Organization,
    zone: Zone,
    from_utc: datetime.datetime,
    to_utc: datetime.datetime,
    group_by: str,
    current_user: User,
    filters: Optional[List[FilterInputType]] = None,
) -> QuerySet[WithAnnotations[Incident]]:
    timezone = zone.tzinfo

    queryset = (
        Incident.objects.for_organization(organization, [zone])
        .from_timestamp(from_utc)
        .to_timestamp(to_utc)
    ).apply_filters(FilterInputType.to_filter_list(filters), current_user)

    if group_by == "hour":
        queryset = queryset.values(
            key=TruncHour(
                "timestamp", tzinfo=timezone, output_field=DateTimeField()
            ),
        )
    elif group_by == "day":
        queryset = queryset.values(
            key=TruncDay(
                "timestamp", tzinfo=timezone, output_field=DateTimeField()
            ),
        )
    elif group_by == "month":
        queryset = queryset.values(
            key=TruncMonth(
                "timestamp", tzinfo=timezone, output_field=DateTimeField()
            ),
        )
    else:
        raise RuntimeError(f"Invalid group_by value: {group_by}")

    grouped_types = {}
    for incident_type in zone.enabled_incident_types:
        grouped_types[incident_type.key] = Count(
            "id",
            filter=Q(incident_type__key=incident_type.key),
        )

    queryset = queryset.annotate(
        # Types
        incident_type_counts=JSONObject(**grouped_types),
        # Priority
        priority_counts=JSONObject(
            high_priority_count=Count(
                "id", filter=Q(priority=Incident.Priority.HIGH)
            ),
            medium_priority_count=Count(
                "id",
                filter=Q(priority=Incident.Priority.MEDIUM),
            ),
            low_priority_count=Count(
                "id",
                filter=Q(priority=Incident.Priority.LOW),
            ),
        ),
        # Status
        status_counts=JSONObject(
            open_count=Count(
                "id",
                filter=Q(status=Incident.Status.OPEN),
            ),
            resolved_count=Count(
                "id",
                filter=Q(status=Incident.Status.RESOLVED),
            ),
        ),
    )

    # order_by() is critical here to ensure we only group by `key`, otherwise
    # the ORM will include all other order_by values in the GROUP BY clause
    # by default and we will see unexpected results.
    # https://docs.djangoproject.com/en/3.2/topics/db/aggregation/#interaction-with-order-by
    return queryset.order_by("key")
