from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.BAD_POSTURE,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/e54c5a13-f574-4c8f-972a-451a14be9725",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/a0db58f9-9e4b-48ca-a4d7-b7f81f8e85b9",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/95c13cbc-b342-42c3-8c9b-6bdc6b746360",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/9c939c46-af1c-4914-8c01-40ef0b77669c",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/89601f05-53a4-4624-a1df-dc19c32fadf5",
        ),
    ],
    relative_day_config={
        0: 2,
        -1: 2,
        -2: 0,
        -3: 0,
        -4: 0,
        -5: 2,
        -6: 0,
        -7: 2,
        -8: 0,
        -9: 0,
        -10: 2,
        -11: 2,
        -12: 4,
        -13: 2,
        -14: 8,
        -15: 2,
        -16: 4,
        -17: 4,
        -18: 6,
        -19: 8,
        -20: 4,
        -21: 8,
        -22: 2,
        -23: 4,
        -24: 2,
        -25: 2,
        -26: 4,
        -27: 4,
        -28: 8,
        -29: 12,
        -30: 12,
    },
)
