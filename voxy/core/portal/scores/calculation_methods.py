import typing as t
from datetime import datetime
from decimal import Decimal

from core.portal.perceived_data.calculation_methods import (
    PerceivedEventRateCalculationMethodMapping,
)
from core.portal.perceived_data.models.perceived_event_rate_hourly import (
    PerceivedEventRateDefinition,
)
from core.portal.scores.models.score_band_range import ScoreBandRange
from core.portal.scores.models.score_definition import ScoreDefinition


def _generate_score_from_band(value: Decimal, score_band_id: int) -> int:
    """Applies the provided value (vper) to the provided score band

    Args:
        value (Decimal): the vper
        score_band_id (int): the score band

    Returns:
        int: the score value
    """
    try:
        score_band_range = (
            ScoreBandRange.objects.only("score_value")
            .filter(
                score_band_id__exact=score_band_id,
                deleted_at__isnull=True,
                lower_bound_inclusive__gte=value,
            )
            .order_by("lower_bound_inclusive")[0:1]
            .get()
        )
    except ScoreBandRange.DoesNotExist:
        return 0

    return score_band_range.score_value


def calculate_30_day_event_score(
    perceived_event_rate_definition_id: int,
    score_band_id: int,
    camera_ids: list[int],
    date_time: datetime,
) -> t.Union[int, None]:
    """Calculates the 30 day event score

    Args:
        perceived_event_rate_definition_id (int): The score's VPER definition
        score_band_id (int): The score's band
        camera_ids (list[int]): The relevant cameras
        date_time (datetime): The datetime for the calculation

    Returns:
        t.Union[int, None]: The 30 day event score
    """
    per_definition = PerceivedEventRateDefinition.objects.get(
        pk=perceived_event_rate_definition_id
    )
    per = PerceivedEventRateCalculationMethodMapping[
        per_definition.calculation_method
    ](definition=per_definition, camera_ids=camera_ids, date_time=date_time)

    if per is None:
        return None

    score = _generate_score_from_band(value=per, score_band_id=score_band_id)

    return score


# TODO: Figure out how to encapsulate all this into one enum
CalculationMethodMapping = {
    ScoreDefinition.CalculationMethod.THIRTY_DAY_EVENT_SCORE: calculate_30_day_event_score,
}
