from enum import Enum


class EdgeLifecycle(Enum):
    """The device state enum"""

    EDGE_LIFECYCLE_LIVE = "live"
    EDGE_LIFECYCLE_PROVISIONED = "provisioned"
    EDGE_LIFECYCLE_MAINTENANCE = "maintenance"
