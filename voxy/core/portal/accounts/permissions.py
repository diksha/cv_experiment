from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Permission:
    capability_key: str
    global_scope: bool = True
    organization_scope: bool = True
    zone_scope: bool = True

    @property
    def global_permission_key(self) -> Optional[str]:
        """Generates global prefix permission key string

        Returns:
            Optional[str]: None or permission key string
        """
        if self.global_scope:
            return f"global:{self.capability_key}"
        return None

    @property
    def organization_permission_key(self) -> Optional[str]:
        """Generates organization prefix permission key string

        Returns:
            Optional[str]: None or permission key string
        """
        if self.organization_scope:
            return f"organization:{self.capability_key}"
        return None

    @property
    def zone_permission_key(self) -> Optional[str]:
        """Generates zone prefix permission key string

        Returns:
            Optional[str]: None or zone key string
        """
        if self.zone_scope:
            return f"zone:{self.capability_key}"
        return None


# ****************************************************************************
# Pages
# ****************************************************************************
PAGE_DASHBOARD = Permission("page:dashboard")
PAGE_EXECUTIVE_DASHBOARD = Permission("page:executive_dashboard")
PAGE_ANALYTICS = Permission("page:analytics")
PAGE_INCIDENTS = Permission("page:incidents")
PAGE_INCIDENT_DETAILS = Permission("page:incident_details")
PAGE_REVIEW_QUEUE = Permission("page:review_queue")
PAGE_REVIEW_HISTORY = Permission("page:review_history")
PAGE_ACCOUNT = Permission("page:account")
PAGE_ACCOUNT_CAMERAS = Permission("page:account_cameras")

# ****************************************************************************
# Incidents
# ****************************************************************************
INCIDENTS_READ = Permission("incidents:read")
INCIDENTS_ASSIGN = Permission("incidents:assign")
INCIDENTS_UNASSIGN = Permission("incidents:unassign")
INCIDENTS_RESOLVE = Permission("incidents:resolve")
INCIDENTS_REOPEN = Permission("incidents:reopen")
INCIDENTS_COMMENT = Permission("incidents:comment")
INCIDENTS_BOOKMARK = Permission("incidents:bookmark")
INCIDENTS_HIGHLIGHT = Permission("incidents:highlight")
INCIDENTS_DOWNLOAD_VIDEO = Permission("incidents:download_video")
INCIDENT_PUBLIC_LINK_CREATE = Permission("incidents:public_link_create")
INCIDENT_DETAILS_READ = Permission("incident_details:read")
EXPERIMENTAL_INCIDENTS_READ = Permission("experimental_incidents:read")

# ****************************************************************************
# Incident Feedback & Review Queue
# ****************************************************************************
INCIDENT_FEEDBACK_CREATE = Permission("incident_feedback:create")
INCIDENT_FEEDBACK_READ = Permission("incident_feedback:read")
REVIEW_QUEUE_READ = Permission(
    "review_queue:read", organization_scope=False, zone_scope=False
)

# ****************************************************************************
# Analytics
# ****************************************************************************
ANALYTICS_READ = Permission("analytics:read")

# ****************************************************************************
# Users
# ****************************************************************************
USERS_INVITE = Permission("users:invite")
USERS_REMOVE = Permission("users:remove")
USERS_UPDATE_PROFILE = Permission("users:update_profile")
USERS_UPDATE_ROLE = Permission("users:update_role")
USERS_UPDATE_SITE = Permission("users:update_site")
USERS_READ = Permission("users:read")
REVIEWER_ACCOUNT_CREATE = Permission("reviewer_account:create")
REVIEWER_ACCOUNTS_UPDATE_ROLE = Permission("reviewer_account:update_role")
# ****************************************************************************
# Zones
# ****************************************************************************

CAMERAS_RENAME = Permission("cameras:rename")
CAMERAS_READ = Permission("cameras:read")

# ****************************************************************************
# Downtime
# ****************************************************************************

DOWNTIME_READ = Permission("downtime:read")

# ****************************************************************************
# Uncategorized
# ****************************************************************************
TOOLBOX = Permission("toolbox", organization_scope=False, zone_scope=False)
SELF_SWITCH_ORGANIZATION = Permission(
    "self:switch_organization", zone_scope=False
)
SELF_SWITCH_SITE = Permission("self:switch_site")
SELF_UPDATE_ROLE = Permission("self:update_role")
SELF_UPDATE_PROFILE = Permission(
    "self:update_profile", organization_scope=False, zone_scope=False
)
INTERNAL_SITE_ACCESS = Permission(
    "self:internal_site_access", organization_scope=False, zone_scope=False
)
