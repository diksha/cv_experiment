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
    from core.portal.compliance.models.door_open_aggregate import (
        DoorOpenAggregate,
    )

filter_map: Dict[str, Callable] = {
    FilterKey.CAMERA.value: camera_filter,
}


def apply_filter(
    queryset: QuerySet["DoorOpenAggregate"],
    filter_data: Filter,
    current_user: User,
) -> QuerySet["DoorOpenAggregate"]:
    """Apply door open aggregate custom filter
    Args:
        queryset(QuerySet[DoorOpenAggregate]): A door open aggregate queryset
        filter_data (FilterInputType): A FilterInputType object defines the query filter
        current_user (User): Current user
    Returns:
        QuerySet[DoorOpenAggregate]: the filtered list of door open aggregat queryset
    """
    return generic_apply_filter(
        queryset, filter_data, current_user, filter_map
    )