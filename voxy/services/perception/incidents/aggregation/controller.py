from core.structs.incident import Incident
from services.perception.incidents.aggregation.aggregators.actor_cooldown_duration import (
    ActorCooldownDurationAggregator,
)


class IncidentAggregationController:
    def __init__(self):
        self._incident_type_and_aggregator_map = {
            "PRODUCTION_LINE_DOWN": ActorCooldownDurationAggregator(
                epsilon_between_incidents_ms=200_000
            ),
            "OPEN_DOOR_DURATION": ActorCooldownDurationAggregator(),
            "PARKING_DURATION": ActorCooldownDurationAggregator(),
            "SPILL": ActorCooldownDurationAggregator(
                epsilon_between_incidents_ms=200_000
            ),
            "BUMP_CAP": ActorCooldownDurationAggregator(),
            "HARD_HAT": ActorCooldownDurationAggregator(),
            "SAFETY_VEST": ActorCooldownDurationAggregator(),
            "HIGH_VIS_HAT_OR_VEST": ActorCooldownDurationAggregator(),
            "NO_PED_ZONE": ActorCooldownDurationAggregator(),
            "N_PERSON_PED_ZONE": ActorCooldownDurationAggregator(),
        }

    def process(self, incident: Incident) -> Incident:
        """Process an incident.

        Args:
            incident (Incident): Input Incident

        Returns:
            Incident: Updated Incident
        """
        incident_type = incident.incident_type_id

        # Aggregate incident if aggregator exists for incident type.
        if incident_type in self._incident_type_and_aggregator_map:
            aggregator = self._incident_type_and_aggregator_map[incident_type]
            return aggregator.aggregate(incident)

        return incident
