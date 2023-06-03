from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.NO_PED_ZONE,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/4b6db807-6fe2-4ec4-ac76-bd9359ca5c1c",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/1af91c07-8f35-4d8f-867b-76112fe91a77",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/a3018f8d-9c25-4bab-92a6-699fe916fec8",
        ),
    ],
    relative_day_config={
        0: 4,
        -1: 2,
        -2: 0,
        -3: 0,
        -4: 0,
        -5: 1,
        -6: 0,
        -7: 1,
        -8: 0,
        -9: 0,
        -10: 2,
        -11: 2,
        -12: 2,
        -13: 2,
        -14: 2,
        -15: 2,
        -16: 5,
        -17: 2,
        -18: 5,
        -19: 2,
        -20: 5,
        -21: 9,
        -22: 5,
        -23: 7,
        -24: 9,
        -25: 8,
        -26: 9,
        -27: 11,
        -28: 11,
        -29: 10,
        -30: 12,
    },
)
