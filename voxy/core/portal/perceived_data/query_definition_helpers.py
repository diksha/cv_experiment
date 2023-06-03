from django.db.models import F, OuterRef, Q, Subquery

from core.portal.api.models.incident_type import CameraIncidentType
from core.portal.perceived_data.calculation_methods import (
    PerceivedEventRateCalculationInput,
)
from core.portal.perceived_data.enums import (
    PerceivedEventRateCalculationMethod,
)
from core.portal.perceived_data.models.perceived_event_rate_hourly import (
    PerceivedEventRateDefinition,
)


def _query_for_calculation_definitions(
    query_expressions: Q,
    calculation_method: PerceivedEventRateCalculationMethod,
) -> list[PerceivedEventRateCalculationInput]:
    return list(
        CameraIncidentType.objects.filter(
            Q(enabled__exact=True) & query_expressions
        )
        .annotate(
            definition_id=F("incident_type__perceived_event_rate_definitions")
        )
        .filter(definition_id__isnull=False)
        .annotate(
            perceived_actor_state_duration_category=Subquery(
                PerceivedEventRateDefinition.objects.filter(
                    pk=OuterRef("definition_id"),
                    calculation_method=calculation_method,
                ).values("perceived_actor_state_duration_category")
            )
        )
        .values_list(
            "camera_id",
            "definition_id",
            "incident_type_id",
            "perceived_actor_state_duration_category",
            named=True,
        )
    )


def query_for_camera_calculation_definitions(
    camera_ids: list[int],
    calculation_method: PerceivedEventRateCalculationMethod,
) -> list[PerceivedEventRateCalculationInput]:
    """Retreive all VPER definitions with the given calculation method that are relevant
    to the provided list of cameras.

    Args:
        camera_ids (list[int]): List of relevant cameras
        calculation_method (PerceivedEventRateCalculationMethod): calculation method

    Returns:
        list[PerceivedEventRateCalculationInput]: Pairs of relevant cameras and VPER definitions
    """
    return _query_for_calculation_definitions(
        query_expressions=Q(camera_id__in=camera_ids),
        calculation_method=calculation_method,
    )
