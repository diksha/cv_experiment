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
from typing import TYPE_CHECKING, Callable, Dict, List, Union

from django.contrib.auth.models import User
from django.db.models import Count, Exists, OuterRef, Q, Subquery
from django.db.models.query import QuerySet

from core.portal.incidents.enums import (
    AssignmentFilterOption,
    FilterKey,
    StatusFilterOption,
)
from core.portal.incidents.types import Filter, FilterValue
from core.portal.lib.graphql.utils import ids_to_primary_keys
from core.portal.lib.utils.filter.base import (
    apply_filter as generic_apply_filter,
)
from core.portal.lib.utils.filter.base import camera_filter

if TYPE_CHECKING:
    from core.portal.api.models.incident import Incident


def remove_bool_values(values: List[Union[str, bool]]) -> List[str]:
    return [x for x in values if isinstance(x, str)]


def extras_filter(
    queryset: QuerySet["Incident"],
    value: FilterValue,
    current_user: User,
) -> QuerySet["Incident"]:
    value_list = value if isinstance(value, list) else [value]
    if "BOOKMARKED" in value_list:
        bookmarked_incidents_subquery = (
            current_user.profile.starred_list.incidents.filter(
                id=OuterRef("id")
            )
        )
        queryset = queryset.annotate(
            bookmarked=Exists(Subquery(bookmarked_incidents_subquery)),
        ).filter(bookmarked=True)
    if "HIGHLIGHTED" in value_list:
        queryset = queryset.filter(highlighted=True)
    return queryset


def priority_filter(
    queryset: QuerySet["Incident"], value: FilterValue, *_: None
) -> QuerySet["Incident"]:
    value_list = value if isinstance(value, list) else [value]
    return queryset.filter(priority__in=value_list)


def is_valid_filter(
    queryset: QuerySet["Incident"], *_: None
) -> QuerySet["Incident"]:
    """
    Only returns valid incidents (valid count > 0), (invalid count == 0)

    NOTE: this is intended to be used for offline/internal purposes.
    Do not use for production purposes

    Args:
       queryset (QuerySet[Incident]): original queryset
       *_ (None): other kwargs

    Returns:
        QuerySet[Incident]: Filtered query set
    """

    return queryset.filter(valid_feedback_count__gt=0,).exclude(
        invalid_feedback_count__gt=0,
    )


def is_unsure_filter(
    queryset: QuerySet["Incident"], *_: None
) -> QuerySet["Incident"]:
    """
    Only returns unsure incidents (unsure count > 0)

    NOTE: this is intended to be used for offline/internal purposes.
    Do not use for production purposes

    Args:
       queryset (QuerySet[Incident]): original queryset
       *_ (None): other kwargs

    Returns:
        QuerySet[Incident]: Filtered query set
    """

    return queryset.filter(unsure_feedback_count__gt=0)


def is_corrupt_filter(
    queryset: QuerySet["Incident"], *_: None
) -> QuerySet["Incident"]:
    """
    Only returns corrupt incidents (corrupt > 0)

    NOTE: this is intended to be used for offline/internal purposes.
    Do not use for production purposes

    Args:
       queryset (QuerySet[Incident]): original queryset
       *_ (None): other kwargs

    Returns:
        QuerySet[Incident]: Filtered query set
    """
    return queryset.filter(corrupt_feedback_count__gt=0)


def exclude_unsure_filter(
    queryset: QuerySet["Incident"], *_: None
) -> QuerySet["Incident"]:
    """
    Only returns incidents that did not have unsure feedback (unsure count == 0)

    NOTE: this is intended to be used for offline/internal purposes.
    Do not use for production purposes

    Args:
       queryset (QuerySet[Incident]): original queryset
       *_ (None): other kwargs

    Returns:
        QuerySet[Incident]: Filtered query set
    """
    return queryset.exclude(unsure_feedback_count__gt=0)


def exclude_corrupt_filter(
    queryset: QuerySet["Incident"], *_: None
) -> QuerySet["Incident"]:
    """
    Only returns incidents that do not have a corrupt feedback
     (corrupt count == 0)

    NOTE: this is intended to be used for offline/internal purposes.
    Do not use for production purposes

    Args:
       queryset (QuerySet[Incident]): original queryset
       *_ (None): other kwargs

    Returns:
        QuerySet[Incident]: Filtered query set
    """
    return queryset.exclude(corrupt_feedback_count__gt=0)


def is_invalid_filter(
    queryset: QuerySet["Incident"], *_: None
) -> QuerySet["Incident"]:
    """
    Only returns invalid incidents (invalid count > 0), (valid count == 0)

    NOTE: this is intended to be used for offline/internal purposes.
    Do not use for production purposes

    Args:
       queryset (QuerySet[Incident]): original queryset
       *_ (None): other kwargs

    Returns:
        QuerySet[Incident]: Filtered query set
    """
    return queryset.filter(invalid_feedback_count__gt=0,).exclude(
        valid_feedback_count__gt=0,
    )


def status_filter(
    queryset: QuerySet["Incident"], value: FilterValue, *_: None
) -> QuerySet["Incident"]:
    value_list = value if isinstance(value, list) else [value]
    queryset = queryset.annotate(
        assignee_count=Count("user_incidents", distinct=True)
    )

    # TODO: get status values from enum instead of hardcoded strings
    q = Q()
    if StatusFilterOption.UNASSIGNED.value in value_list:
        q = q | Q(status="open", assignee_count=0)
    if StatusFilterOption.OPEN_AND_ASSIGNED.value in value_list:
        q = q | Q(status="open", assignee_count__gt=0)
    if StatusFilterOption.RESOLVED.value in value_list:
        q = q | Q(status="resolved")
    return queryset.filter(q)


def assignment_filter(
    queryset: QuerySet["Incident"], value: FilterValue, current_user: User
) -> QuerySet["Incident"]:
    # Ignore bool values for this filter
    if isinstance(value, bool):
        return queryset

    value_set = set(value if isinstance(value, list) else [value])

    if not value_set:
        return queryset

    queryset = queryset.annotate(
        assigned_to_count=Count(
            "user_incidents",
            filter=Q(user_incidents__assignee=current_user),
            distinct=True,
        ),
        assigned_by_count=Count(
            "user_incidents",
            filter=Q(user_incidents__assigned_by=current_user),
            distinct=True,
        ),
    )

    q = Q()
    if AssignmentFilterOption.ASSIGNED_TO_ME.value in value_set:
        q = q | Q(assigned_to_count__gt=0)
        value_set.discard(AssignmentFilterOption.ASSIGNED_TO_ME.value)
    if AssignmentFilterOption.ASSIGNED_BY_ME.value in value_set:
        q = q | Q(assigned_by_count__gt=0)
        value_set.discard(AssignmentFilterOption.ASSIGNED_BY_ME.value)

    if value_set:
        # TODO(PRO-321): don't make incident filters convert GraphQL global IDs
        # Remaining values should be user IDs, these IDs could be Django
        # primary keys or GraphQL global IDs. This is messy and the model
        # shouldn't need to know anything about GraphQL global IDs, therefore...

        # Convert any GraphQL global IDs to primary keys
        primary_keys = ids_to_primary_keys(value_set)
        q = q | Q(user_incidents__assignee__in=primary_keys)

    return queryset.filter(q)


def incident_type_filter(
    queryset: QuerySet["Incident"], value: FilterValue, current_user: User
) -> QuerySet["Incident"]:
    value_list = value if isinstance(value, list) else [value]
    return queryset.filter(incident_type__key__in=value_list)


filter_map: Dict[str, Callable] = {
    FilterKey.PRIORITY.value: priority_filter,
    FilterKey.STATUS.value: status_filter,
    FilterKey.CAMERA.value: camera_filter,
    FilterKey.ASSIGNMENT.value: assignment_filter,
    FilterKey.INCIDENT_TYPE.value: incident_type_filter,
    FilterKey.EXTRAS.value: extras_filter,
    FilterKey.IS_INVALID.value: is_invalid_filter,
    FilterKey.IS_VALID.value: is_valid_filter,
    FilterKey.IS_CORRUPT.value: is_corrupt_filter,
    FilterKey.IS_UNSURE.value: is_unsure_filter,
    FilterKey.EXCLUDE_UNSURE.value: exclude_unsure_filter,
    FilterKey.EXCLUDE_CORRUPT.value: exclude_corrupt_filter,
}


def apply_filter(
    queryset: QuerySet["Incident"], filter_data: Filter, current_user: User
) -> QuerySet["Incident"]:
    return generic_apply_filter(
        queryset, filter_data, current_user, filter_map
    )
