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
    from core.portal.compliance.models.production_line_aggregate import (
        ProductionLineAggregate,
    )

filter_map: Dict[str, Callable] = {
    FilterKey.CAMERA.value: camera_filter,
}


def apply_filter(
    queryset: QuerySet["ProductionLineAggregate"],
    filter_data: Filter,
    current_user: User,
) -> QuerySet["ProductionLineAggregate"]:
    """Apply production line aggregate custom filter
    Args:
        queryset(QuerySet[ProductionLineAggregate]): A production line aggregate queryset
        filter_data (FilterInputType): A FilterInputType object defines the query filter
        current_user (User): Current user
    Returns:
        QuerySet[ProductionLineAggregate]: the filtered list of production line aggregates queryset
    """
    return generic_apply_filter(
        queryset, filter_data, current_user, filter_map
    )
