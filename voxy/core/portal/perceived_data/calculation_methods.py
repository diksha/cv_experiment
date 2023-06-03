import typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from django.db.models import Count, DecimalField, OuterRef, Q, Subquery, Sum
from django.db.models.fields.json import KeyTextTransform
from django.db.models.functions import Cast, Greatest, TruncHour
from loguru import logger

from core.portal.api.models.incident import Camera, Incident
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
    PerceivedEventRateCalculationMethod,
)
from core.portal.perceived_data.models.perceived_actor_state_duration_aggregate import (
    PerceivedActorStateDurationAggregate,
    TimeBucketWidth,
)
from core.portal.perceived_data.models.perceived_event_rate_hourly import (
    PerceivedEventRateDefinition,
    PerceivedEventRateHourly,
)


@dataclass
class ValueData:
    time_bucket_start_timestamp: datetime
    denominator_value: float
    numerator_value: float = 0


@dataclass
class PerceivedEventRateCalculationInput:
    camera_id: int
    definition_id: int
    incident_type_id: int
    perceived_actor_state_duration_category: PerceivedActorStateDurationCategory


def execute_hourly_discrete(
    calculation_inputs: list[PerceivedEventRateCalculationInput],
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Given pairs of camera_id and definition_ids, retrieve the relevant discrete hourly VPERs

    Args:
        calculation_inputs (list[PerceivedEventRateCalculationInput]): list of pairs to calcualte
        start_time (datetime): start time
        end_time (datetime): end time
    """
    _execute_hourly(
        calculation_inputs=calculation_inputs,
        start_time=start_time,
        end_time=end_time,
        is_discrete=True,
    )


def execute_hourly_continuous(
    calculation_inputs: list[PerceivedEventRateCalculationInput],
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Given pairs of camera_id and definition_ids, retrieve the relevant continuous hourly VPERs

    Args:
        calculation_inputs (list[PerceivedEventRateCalculationInput]): list of pairs to calcualte
        start_time (datetime): start time
        end_time (datetime): end time
    """
    _execute_hourly(
        calculation_inputs=calculation_inputs,
        start_time=start_time,
        end_time=end_time,
        is_discrete=False,
    )


# TODO(@itay): Refactor this and use `PerceivedEventRateDefinition` object as input
def _execute_hourly(
    calculation_inputs: list[PerceivedEventRateCalculationInput],
    start_time: datetime,
    end_time: datetime,
    is_discrete: bool,
) -> None:
    logger.info(f"Time Range: {start_time} -> {end_time}")

    # Create query expressions to retrieve ActorStateDurationAggregates
    query_expression = Q()
    for definition in calculation_inputs:
        query_expression |= Q(
            camera_id__exact=definition.camera_id,
            category__exact=definition.perceived_actor_state_duration_category,
        )

    state_durations = _calculate_hourly_actor_state_durations(
        query_expression, start_time, end_time
    )
    if len(state_durations) == 0:
        logger.warning("No ActorStateDurations found")
        return

    # Create the outer-left set for a LEFT JOIN
    results: list[tuple[PerceivedEventRateCalculationInput, ValueData]] = []
    for duration in state_durations:
        for calculation_input in calculation_inputs:
            if (
                duration.camera_id == calculation_input.camera_id
                and duration.category
                == calculation_input.perceived_actor_state_duration_category
            ):
                results.append(
                    (
                        calculation_input,
                        ValueData(
                            denominator_value=duration.duration.total_seconds(),
                            time_bucket_start_timestamp=duration.time_bucket_start_timestamp,
                        ),
                    )
                )

    # Create query expressions to retreive Incidents
    query_expression = Q()
    for (definition, _) in results:
        query_expression |= Q(
            camera_id__exact=definition.camera_id,
            incident_type__exact=definition.incident_type_id,
        )

    incident_aggregates = _calcualte_hourly_incident_aggregates(
        query_expression, is_discrete, start_time, end_time
    )

    # Complete the LEFT JOIN by matching Incidents with the outer table
    for aggregate in incident_aggregates.values_list(
        "camera_id",
        "incident_type_id",
        "time_bucket_start_timestamp",
        "incident_aggregate",
        named=True,
    ):
        for (definition, value) in results:
            if (
                value.time_bucket_start_timestamp
                != aggregate.time_bucket_start_timestamp
            ):
                continue
            if definition.incident_type_id != aggregate.incident_type_id:
                continue
            if definition.camera_id != aggregate.camera_id:
                continue
            value.numerator_value = aggregate.incident_aggregate

    logger.info(f"Numer of entries to write: {len(results)}")
    objs_to_create = [
        PerceivedEventRateHourly(
            time_bucket_start_timestamp=value.time_bucket_start_timestamp,
            definition_id=definition.definition_id,
            camera_id=definition.camera_id,
            numerator_value=value.numerator_value,
            denominator_value=value.denominator_value,
        )
        for (definition, value) in results
    ]
    rows_written = PerceivedEventRateHourly.objects.bulk_create(
        objs=objs_to_create,
        update_conflicts=True,
        update_fields=["numerator_value", "denominator_value"],
        unique_fields=[
            "camera",
            "definition",
            "time_bucket_start_timestamp",
        ],
    )
    logger.info(f"Numer of entries written: {len(rows_written)}")


def _calculate_hourly_actor_state_durations(
    query_expression: Q,
    start_time: datetime,
    end_time: datetime,
):
    if query_expression == Q():
        state_durations = PerceivedActorStateDurationAggregate.objects.none()
    else:
        state_durations = (
            PerceivedActorStateDurationAggregate.objects.filter(
                time_bucket_start_timestamp__gte=start_time.replace(
                    tzinfo=timezone.utc
                ),
                time_bucket_start_timestamp__lt=end_time.replace(
                    tzinfo=timezone.utc
                ),
                time_bucket_width__exact=TimeBucketWidth.HOUR,
            )
            .annotate(
                camera_id=Subquery(
                    Camera.objects.filter(uuid=OuterRef("camera_uuid")).values(
                        "id"
                    )
                )
            )
            .filter(query_expression)
            .values_list(
                "camera_id",
                "category",
                "time_bucket_start_timestamp",
                "duration",
                named=True,
            )
        )

    return state_durations


def _calcualte_hourly_incident_aggregates(
    query_expression: Q,
    is_discrete: bool,
    start_time: datetime,
    end_time: datetime,
):
    if query_expression == Q():
        incident_aggregates = Incident.objects.none()
    else:
        incident_aggregates = Incident.aggregable_objects.filter(
            Q(
                timestamp__gte=start_time.replace(tzinfo=timezone.utc),
                timestamp__lt=end_time.replace(tzinfo=timezone.utc),
            )
            & query_expression
        ).values(
            "camera",
            "incident_type",
            time_bucket_start_timestamp=TruncHour("timestamp"),
        )
    if is_discrete:
        incident_aggregates = _calculate_hourly_incident_counts(
            incident_aggregates
        )
    elif not is_discrete:
        incident_aggregates = _calculate_hourly_incident_durations(
            incident_aggregates
        )

    return incident_aggregates


def _calculate_hourly_incident_counts(incident_aggregates):
    incident_aggregates = incident_aggregates.annotate(
        incident_aggregate=Count("id")
    )
    return incident_aggregates


def _calculate_hourly_incident_durations(incident_aggregates):
    # TODO(@itay): Need to figure out how to calculate incident duration over multiple buckets
    incident_aggregates = incident_aggregates.annotate(
        # TODO(@itay): think about optimizing this so to not return NULLs.
        incident_aggregate=Sum(
            Greatest(
                Cast(
                    KeyTextTransform("end_frame_relative_ms", "data"),
                    DecimalField(max_digits=13, decimal_places=0),
                )
                - Cast(
                    KeyTextTransform("start_frame_relative_ms", "data"),
                    DecimalField(max_digits=13, decimal_places=0),
                ),
                Cast(0, DecimalField(max_digits=13, decimal_places=0)),
            )
        )
        # Convert to seconds
        / 1000
    )

    return incident_aggregates


def calculate_30_day_discrete(
    definition: PerceivedEventRateDefinition,
    camera_ids: list[int],
    date_time: datetime,
) -> t.Union[Decimal, None]:
    """Calculates a single 30-day discrete VPER with the given cameras

    Args:
        definition (PerceivedEventRateDefinition): The VPER definition
        camera_ids (list[int]): The relevant cameras
        date_time (datetime): The datetime of the VPER

    Returns:
        t.Union[Decimal, None]: A VPER or None (if error)
    """
    return _calculate_30_day_vper(
        definition=definition,
        camera_ids=camera_ids,
        date_time=date_time,
        is_discrete=True,
    )


def calculate_30_day_continuous(
    definition: PerceivedEventRateDefinition,
    camera_ids: list[int],
    date_time: datetime,
) -> t.Union[Decimal, None]:
    """Calculates a single 30-day continuous VPER with the given cameras

    Args:
        definition (PerceivedEventRateDefinition): The VPER definition
        camera_ids (list[int]): The relevant cameras
        date_time (datetime): The datetime of the VPER

    Returns:
        t.Union[Decimal, None]: A VPER or None (if error)
    """
    return _calculate_30_day_vper(
        definition=definition,
        camera_ids=camera_ids,
        date_time=date_time,
        is_discrete=False,
    )


def _calculate_30_day_vper(
    definition: PerceivedEventRateDefinition,
    camera_ids: list[int],
    date_time: datetime,
    # TODO: validate is_discrete is either HOURLY_CONTINOUS or HOURLY_DISCRETE
    is_discrete: bool,
) -> t.Union[Decimal, None]:
    """Calculates a single 30-day VPER with the given cameras

    Args:
        definition (PerceivedEventRateDefinition): The VPER definition
        camera_ids (list[int]): The relevant cameras
        date_time (datetime): The datetime of the VPER
        is_discrete (bool): If should use discrete hourly vpers

    Returns:
        t.Union[Decimal, None]: A VPER or None (if error)
    """
    calculation_method = (
        PerceivedEventRateCalculationMethod.HOURLY_DISCRETE
        if is_discrete
        else PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS
    )
    hourly_definition = PerceivedEventRateDefinition.objects.only("id").get(
        incident_type=definition.incident_type,
        perceived_actor_state_duration_category=definition.perceived_actor_state_duration_category,
        calculation_method=calculation_method,
    )
    return _calculate_with_hourly_vpers(
        hourly_definition=hourly_definition,
        camera_ids=camera_ids,
        time_from=date_time - timedelta(days=30),
        time_to=date_time,
    )


def _calculate_with_hourly_vpers(
    hourly_definition: PerceivedEventRateDefinition,
    camera_ids: list[int],
    time_from: datetime,
    time_to: datetime,
) -> t.Union[Decimal, None]:
    """Calculates a single VPER from all hourly vpers within the
    time range

    Args:
        hourly_definition (PerceivedEventRateDefinition): The hourly definition
        camera_ids (list[int]): The relevant camera_ids
        time_from (datetime): lower bound of timerange (inclusive)
        time_to (datetime): upper bound of timerange (exclusive)

    Returns:
        t.Union[Decimal, None]:A VPER or None (if error)
    """
    vper = PerceivedEventRateHourly.objects.filter(
        definition_id__exact=hourly_definition.id,
        camera__in=camera_ids,
        time_bucket_start_timestamp__gte=time_from,
        time_bucket_start_timestamp__lt=time_to,
    ).aggregate(
        total_numerator_value=Sum("numerator_value"),
        total_denominator_value=Sum("denominator_value"),
    )
    event_rate = None
    try:
        event_rate = (
            vper["total_numerator_value"] / vper["total_denominator_value"]
        )
    # TODO: Implement more robust exception handling
    # trunk-ignore(pylint/W0718)
    except Exception as exception:
        logger.error(f"There was an exception: {exception}")

    return event_rate


PerceivedEventRateCalculationMethodMapping = {
    PerceivedEventRateCalculationMethod.HOURLY_DISCRETE: (
        execute_hourly_discrete
    ),
    PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS: (
        execute_hourly_continuous
    ),
    PerceivedEventRateCalculationMethod.THIRTY_DAY_DISCRETE: calculate_30_day_discrete,
    PerceivedEventRateCalculationMethod.THIRTY_DAY_CONTINUOUS: calculate_30_day_continuous,
}
