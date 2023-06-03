from core.portal.devices.enums import EdgeLifecycle
from core.portal.utils.enum_utils import EnumConfig

STATIC_EDGE_LIFECYCLE_CONFIG_MAP = {
    EdgeLifecycle.EDGE_LIFECYCLE_LIVE: EnumConfig(
        EdgeLifecycle.EDGE_LIFECYCLE_LIVE.value,
        "The edge is live and ready to be used.",
    ),
    EdgeLifecycle.EDGE_LIFECYCLE_PROVISIONED: EnumConfig(
        EdgeLifecycle.EDGE_LIFECYCLE_PROVISIONED.value,
        "The edge is provisioned",
    ),
    EdgeLifecycle.EDGE_LIFECYCLE_MAINTENANCE: EnumConfig(
        EdgeLifecycle.EDGE_LIFECYCLE_MAINTENANCE.value,
        "The edge is in maintenance",
    ),
}
