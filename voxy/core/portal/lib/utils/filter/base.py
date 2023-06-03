from typing import Callable, Dict

from django.contrib.auth.models import User
from django.db.models.query import QuerySet
from loguru import logger

from core.portal.incidents.types import Filter, FilterValue
from core.portal.lib.graphql.utils import ids_to_primary_keys
from core.portal.lib.utils.filter.enums import QueryModelType


def apply_filter(
    queryset: QuerySet[QueryModelType],
    filter_data: Filter,
    current_user: User,
    filter_map: Dict[str, Callable],
) -> QuerySet[QueryModelType]:
    """Apply supported QueryModelType custom filter
    Args:
        queryset(QuerySet[QueryModelType]): Supported QueryModelType queryset
        filter_data (FilterInputType): FilterInputType object defines the query filter
        current_user (User): Current user
        filter_map (Dict[str, Callable]):
            Filter mapping with supported filter as a key and callable function as a value
    Returns:
        QuerySet[QueryModelType]: A queryset
    """
    filter_func = filter_map.get(filter_data.key)
    if filter_func:
        queryset = filter_func(queryset, filter_data.value, current_user)
    else:
        logger.warning("No filter defined for key: %s", filter_data.key)
    return queryset


def camera_filter(
    queryset: QuerySet[QueryModelType], value: FilterValue, *_: None
) -> QuerySet[QueryModelType]:
    """Apply camera id filter on queryset
    Args:
        queryset (QuerySet[QueryModelType]): Supported QueryModelType queryset
        value (FilterValue): Union of filter key
        *_: Do nothing for the rest params and default to None

    Returns:
        QuerySet[QueryModelType]: Supported QueryModelType queryset after applying camera filter
    """
    # Ignore bool values for this filter
    if isinstance(value, bool):
        return queryset

    value_set = set(value if isinstance(value, list) else [value])

    if not value_set:
        return queryset

    # Remaining values should be user IDs, these IDs could be Django
    # primary keys or GraphQL global IDs. This is messy and the model
    # shouldn't need to know anything about GraphQL global IDs, therefore...

    # Convert any GraphQL global IDs to primary keys
    primary_keys = ids_to_primary_keys(value_set)
    return queryset.filter(camera_id__in=primary_keys)
