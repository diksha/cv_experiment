from typing import TYPE_CHECKING, Callable, Dict

from django.contrib.auth.models import User
from django.db.models.query import QuerySet

# TODO: Move the share type, enum, and util function into a different
# file/structure rather than tied to the Incident model.
from core.portal.incidents.enums import FilterKey
from core.portal.incidents.types import Filter
from core.portal.lib.utils.filter.base import (
    apply_filter as generic_apply_filter,
)
from core.portal.lib.utils.filter.base import camera_filter

if TYPE_CHECKING:
    from core.portal.compliance.models.production_line import ProductionLine

filter_map: Dict[str, Callable] = {
    FilterKey.CAMERA.value: camera_filter,
}


def apply_filter(
    queryset: QuerySet["ProductionLine"],
    filter_data: Filter,
    current_user: User,
) -> QuerySet["ProductionLine"]:
    """Apply production line custom filter
    Args:
        queryset(QuerySet[ProductionLine]): A production line queryset
        filter_data (FilterInputType): A FilterInputType object defines the query filter
        current_user (User): Current user
    Returns:
        QuerySet[ProductionLine]: the filtered list of production lines
    """
    return generic_apply_filter(
        queryset, filter_data, current_user, filter_map
    )
