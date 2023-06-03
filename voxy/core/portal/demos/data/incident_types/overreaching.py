from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.OVERREACHING,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/c19c680c-5cc0-44c7-9bc5-b66602aa0a23",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/346dc7f9-9787-4557-81c8-3a9ffe4e16b0",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/6f96910d-027b-4ab9-8daf-7a73e328b0ea",
        ),
    ],
    relative_day_config={
        0: 1,
        -1: 1,
        -2: 0,
        -3: 0,
        -4: 0,
        -5: 1,
        -6: 0,
        -7: 1,
        -8: 0,
        -9: 0,
        -10: 1,
        -11: 1,
        -12: 2,
        -13: 1,
        -14: 4,
        -15: 1,
        -16: 2,
        -17: 2,
        -18: 3,
        -19: 4,
        -20: 2,
        -21: 4,
        -22: 3,
        -23: 8,
        -24: 4,
        -25: 5,
        -26: 8,
        -27: 6,
        -28: 15,
        -29: 10,
        -30: 20,
    },
)
