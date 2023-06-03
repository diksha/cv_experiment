from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.OPEN_DOOR_DURATION,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/3cd3483f-3c27-4440-bdd9-994fc1bb1a49",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/39a3482d-866f-4b4a-814e-daac47b45e55",
        ),
    ],
    relative_day_config={
        0: 1,
        -1: 0,
        -2: 0,
        -3: 0,
        -4: 0,
        -5: 0,
        -6: 0,
        -7: 1,
        -8: 1,
        -9: 0,
        -10: 0,
        -11: 0,
        -12: 0,
        -13: 0,
        -14: 0,
        -15: 1,
        -16: 2,
        -17: 1,
        -18: 0,
        -19: 0,
        -20: 0,
        -21: 0,
        -22: 0,
        -23: 0,
        -24: 1,
        -25: 1,
        -26: 0,
        -27: 0,
        -28: 0,
        -29: 0,
        -30: 0,
    },
)
