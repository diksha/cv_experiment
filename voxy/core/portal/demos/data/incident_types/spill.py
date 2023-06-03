from core.portal.demos.data.types import (
    DemoIncidentTypeConfig,
    DemoSourceIncident,
)
from core.portal.incidents.enums import IncidentTypeKey

CONFIG = DemoIncidentTypeConfig(
    incident_type_key=IncidentTypeKey.SPILL,
    source_incidents=[
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/6791d0eb-872a-4f60-b240-205aedfa2a4b",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/cfefc5bf-4e99-43de-be9f-80432cddd70c",
        ),
        DemoSourceIncident(
            "https://app.voxelai.com/incidents/57bc90a2-61a5-4225-b18e-e29f549a733d",
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
        -19: 1,
        -20: 1,
        -21: 1,
        -22: 0,
        -23: 1,
        -24: 0,
        -25: 0,
        -26: 1,
        -27: 0,
        -28: 0,
        -29: 0,
        -30: 0,
    },
)
