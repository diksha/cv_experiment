from core.portal.incidents.enums import IncidentTypeKey
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
    PerceivedEventRateCalculationMethod,
)
from core.portal.perceived_data.models.perceived_event_rate_definition import (
    PerceivedEventRateDefinition,
)

definitions = {
    IncidentTypeKey.BUMP_CAP: [],
    IncidentTypeKey.BAD_POSTURE_WITH_SAFETY_UNIFORM: [],
    IncidentTypeKey.OVERREACHING_WITH_SAFETY_UNIFORM: [],
    IncidentTypeKey.N_PERSON_PED_ZONE: [],
    IncidentTypeKey.PRODUCTION_LINE_DOWN: [],
    IncidentTypeKey.NO_PED_ZONE: [
        PerceivedEventRateDefinition(
            name="No ped zone violation time over person time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PERSON_TIME,
        )
    ],
    IncidentTypeKey.NO_STOP_AT_END_OF_AISLE: [
        PerceivedEventRateDefinition(
            name="Speeding at end of aisle count over moving PIT time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME
            ),
        ),
    ],
    IncidentTypeKey.NO_STOP_AT_DOOR_INTERSECTION: [
        PerceivedEventRateDefinition(
            name="Speeding at door count over moving PIT time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME
            ),
        ),
    ],
    IncidentTypeKey.OVERREACHING: [
        PerceivedEventRateDefinition(
            name="Overreaching count over person time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PERSON_TIME
            ),
        ),
    ],
    IncidentTypeKey.Safety_Harness: [],
    IncidentTypeKey.SPILL: [],
    IncidentTypeKey.NO_STOP_AT_INTERSECTION: [
        PerceivedEventRateDefinition(
            name="Speeding at intersection count over moving PIT time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME
            ),
        ),
    ],
    IncidentTypeKey.HARD_HAT: [
        PerceivedEventRateDefinition(
            name="Hard hat violation time over person time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PERSON_TIME,
        )
    ],
    IncidentTypeKey.SAFETY_VEST: [
        PerceivedEventRateDefinition(
            name="Safety vest violation time over person time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PERSON_TIME,
        )
    ],
    IncidentTypeKey.BAD_POSTURE: [
        PerceivedEventRateDefinition(
            name="Bad lift count over person time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PERSON_TIME
            ),
        ),
    ],
    IncidentTypeKey.DOOR_VIOLATION: [],
    IncidentTypeKey.PARKING_DURATION: [
        PerceivedEventRateDefinition(
            name="Parking duration violation over stationary PIT time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME
            ),
        )
    ],
    IncidentTypeKey.PIGGYBACK: [
        PerceivedEventRateDefinition(
            name="Piggybacking count over moving PIT time",
            calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
            perceived_actor_state_duration_category=(
                PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME
            ),
        ),
    ],
    IncidentTypeKey.OPEN_DOOR_DURATION: [],
    IncidentTypeKey.MISSING_PPE: [],
    IncidentTypeKey.HIGH_VIS_HAT_OR_VEST: [],
}
