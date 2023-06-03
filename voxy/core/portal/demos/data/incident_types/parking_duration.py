from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.PARKING_DURATION,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/00794699-8b46-4d56-b56d-85d6b4fd70b4",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/f0b2caec-6a24-4a72-ba83-cb4c7e76c612",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/66b5eb4f-dfd3-4b46-919f-aa531990f9c4",
        ),
    ],
    relative_day_config={
        0: 1,
        -1: 0,
        -2: 2,
        -3: 2,
        -4: 0,
        -5: 5,
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
        -27: 5,
        -28: 0,
        -29: 0,
        -30: 0,
    },
)
