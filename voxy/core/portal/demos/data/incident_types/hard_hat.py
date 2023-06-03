from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.HARD_HAT,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/88aaadb8-f6c6-42a0-817b-0c992294a7dd",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/79f27d1c-51a2-4b6a-9845-7da47ac499e5",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/a3dd5e46-3f52-41b7-9c89-d73d03c47969",
        ),
    ],
    relative_day_config={
        0: 1,
        -1: 0,
        -2: 0,
        -3: 1,
        -4: 1,
        -5: 1,
        -6: 1,
        -7: 1,
        -8: 1,
        -9: 0,
        -10: 0,
        -11: 0,
        -12: 0,
        -13: 0,
        -14: 0,
        -15: 1,
        -16: 1,
        -17: 1,
        -18: 0,
        -19: 0,
        -20: 0,
        -21: 0,
        -22: 0,
        -23: 0,
        -24: 5,
        -25: 1,
        -26: 0,
        -27: 0,
        -28: 1,
        -29: 1,
        -30: 1,
    },
)
