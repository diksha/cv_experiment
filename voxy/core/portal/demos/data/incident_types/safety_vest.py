from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.SAFETY_VEST,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/f61b2ca6-e3c4-4d62-b57e-62f905d18c5b",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/3057a62a-c66a-4dc2-abff-23a4f5ca4f9d",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/6c6d7f3b-c548-47d2-bfd0-cd21a0a97666",
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
        -23: 7,
        -24: 6,
        -25: 8,
        -26: 10,
        -27: 6,
        -28: 6,
        -29: 10,
        -30: 15,
    },
)
