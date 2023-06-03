from core.portal.demos.data.incident_types import (
    bad_posture,
    hard_hat,
    no_ped_zone,
    no_stop_at_intersection,
    open_door_duration,
    overreaching,
    parking_duration,
    piggyback,
    safety_vest,
    spill,
)
from core.portal.incidents.enums import IncidentTypeKey

INCIDENT_TYPE_CONFIGS = {
    IncidentTypeKey.SPILL: spill.CONFIG,
    IncidentTypeKey.BAD_POSTURE: bad_posture.CONFIG,
    IncidentTypeKey.NO_STOP_AT_INTERSECTION: no_stop_at_intersection.CONFIG,
    IncidentTypeKey.SAFETY_VEST: safety_vest.CONFIG,
    IncidentTypeKey.OVERREACHING: overreaching.CONFIG,
    IncidentTypeKey.NO_PED_ZONE: no_ped_zone.CONFIG,
    IncidentTypeKey.PARKING_DURATION: parking_duration.CONFIG,
    IncidentTypeKey.PIGGYBACK: piggyback.CONFIG,
    IncidentTypeKey.OPEN_DOOR_DURATION: open_door_duration.CONFIG,
    IncidentTypeKey.HARD_HAT: hard_hat.CONFIG,
}
