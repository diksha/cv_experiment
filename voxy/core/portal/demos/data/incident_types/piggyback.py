from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.PIGGYBACK,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/7a73456a-421c-48a1-9a79-0814c5532be4",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/91f29a36-6037-4d5b-842a-a630d304624d",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/ae96fe29-8072-4bfc-b5fc-c1c781d14553",
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
        -24: 0,
        -25: 0,
        -26: 0,
        -27: 0,
        -28: 0,
        -29: 0,
        -30: 0,
    },
)
