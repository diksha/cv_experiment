from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.NO_STOP_AT_INTERSECTION,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/f94945c1-3366-409f-961d-f48e159a2460",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/2b88d399-65cd-4193-a664-f3244163880c",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/250b1287-288e-4889-8fc8-614a611dd007",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/23fe9e66-4b1a-4ca4-a07f-7d92d1ac1e02",
        ),
    ],
    relative_day_config={
        0: 21,
        -1: 18,
        -2: 15,
        -3: 11,
        -4: 10,
        -5: 3,
        -6: 8,
        -7: 9,
        -8: 6,
        -9: 6,
        -10: 6,
        -11: 2,
        -12: 5,
        -13: 6,
        -14: 7,
        -15: 2,
        -16: 0,
        -17: 2,
        -18: 0,
        -19: 4,
        -20: 0,
        -21: 4,
        -22: 1,
        -23: 1,
        -24: 1,
        -25: 4,
        -26: 2,
        -27: 5,
        -28: 2,
        -29: 1,
        -30: 5,
    },
)
