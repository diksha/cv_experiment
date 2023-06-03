from dataclasses import dataclass
from typing import List

from django.utils import timezone
from loguru import logger

from core.portal.accounts import permissions
from core.portal.accounts.models.role import Role
from core.portal.accounts.models.role_permission import RolePermission

# TODO: Remove legacy role
EXTERNAL_ORGANIZATION_STAFF = "external:organization:staff"

# Static roles
EXTERNAL_MANAGER = "external:manager"
EXTERNAL_ADMIN = "external:admin"
EXTERNAL_ORGANIZATION_EXECUTIVE = "external:organization_executive"
INTERNAL_REVIEWER = "internal:reviewer"
INTERNAL_REVIEW_MANAGER = "internal:review_manager"
INTERNAL_SUPERUSER = "internal:superuser"


@dataclass(frozen=True)
class RoleConfig:
    key: str
    name: str
    visible_to_customers: bool
    permission_keys: List[str]


STATIC_ROLES_CONFIG_MAP = {
    EXTERNAL_MANAGER: RoleConfig(
        EXTERNAL_MANAGER,
        "Manager",
        True,
        [
            # Pages
            permissions.PAGE_DASHBOARD.global_permission_key,
            permissions.PAGE_INCIDENTS.global_permission_key,
            permissions.PAGE_INCIDENT_DETAILS.global_permission_key,
            permissions.PAGE_ANALYTICS.global_permission_key,
            permissions.PAGE_ACCOUNT.global_permission_key,
            # Incidents
            permissions.INCIDENTS_READ.zone_permission_key,
            permissions.INCIDENTS_ASSIGN.zone_permission_key,
            permissions.INCIDENTS_UNASSIGN.zone_permission_key,
            permissions.INCIDENTS_RESOLVE.zone_permission_key,
            permissions.INCIDENTS_REOPEN.zone_permission_key,
            permissions.INCIDENTS_COMMENT.zone_permission_key,
            permissions.INCIDENTS_BOOKMARK.zone_permission_key,
            permissions.INCIDENTS_DOWNLOAD_VIDEO.zone_permission_key,
            permissions.INCIDENT_PUBLIC_LINK_CREATE.zone_permission_key,
            permissions.INCIDENT_DETAILS_READ.zone_permission_key,
            # Incident feedback
            permissions.INCIDENT_FEEDBACK_CREATE.zone_permission_key,
            # Cameras
            permissions.CAMERAS_READ.zone_permission_key,
            # Analytics
            permissions.ANALYTICS_READ.zone_permission_key,
            # Downtime
            permissions.DOWNTIME_READ.zone_permission_key,
            # Users
            permissions.USERS_READ.zone_permission_key,
            # Self
            permissions.SELF_SWITCH_SITE.zone_permission_key,
            permissions.SELF_UPDATE_PROFILE.global_permission_key,
        ],
    ),
    EXTERNAL_ADMIN: RoleConfig(
        EXTERNAL_ADMIN,
        "Admin",
        True,
        [
            # Pages
            permissions.PAGE_DASHBOARD.global_permission_key,
            permissions.PAGE_INCIDENTS.global_permission_key,
            permissions.PAGE_INCIDENT_DETAILS.global_permission_key,
            permissions.PAGE_ANALYTICS.global_permission_key,
            permissions.PAGE_ACCOUNT.global_permission_key,
            permissions.PAGE_ACCOUNT_CAMERAS.global_permission_key,
            # Incidents
            permissions.INCIDENTS_READ.zone_permission_key,
            permissions.INCIDENTS_ASSIGN.zone_permission_key,
            permissions.INCIDENTS_UNASSIGN.zone_permission_key,
            permissions.INCIDENTS_RESOLVE.zone_permission_key,
            permissions.INCIDENTS_REOPEN.zone_permission_key,
            permissions.INCIDENTS_COMMENT.zone_permission_key,
            permissions.INCIDENTS_BOOKMARK.zone_permission_key,
            permissions.INCIDENTS_DOWNLOAD_VIDEO.zone_permission_key,
            permissions.INCIDENT_PUBLIC_LINK_CREATE.zone_permission_key,
            permissions.INCIDENT_DETAILS_READ.zone_permission_key,
            # Incident feedback
            permissions.INCIDENT_FEEDBACK_CREATE.zone_permission_key,
            # Cameras
            permissions.CAMERAS_READ.zone_permission_key,
            permissions.CAMERAS_RENAME.zone_permission_key,
            # Analytics
            permissions.ANALYTICS_READ.zone_permission_key,
            # Downtime
            permissions.DOWNTIME_READ.zone_permission_key,
            # Users
            permissions.USERS_INVITE.zone_permission_key,
            permissions.USERS_REMOVE.zone_permission_key,
            permissions.USERS_UPDATE_ROLE.zone_permission_key,
            permissions.USERS_UPDATE_SITE.zone_permission_key,
            permissions.USERS_READ.zone_permission_key,
            # Self
            permissions.SELF_SWITCH_SITE.zone_permission_key,
            permissions.SELF_UPDATE_PROFILE.global_permission_key,
        ],
    ),
    EXTERNAL_ORGANIZATION_EXECUTIVE: RoleConfig(
        EXTERNAL_ORGANIZATION_EXECUTIVE,
        "Organization Executive",
        # TODO: set `True`, show to customers once we have role "hierarchy" access control
        False,
        [
            # Pages
            permissions.PAGE_DASHBOARD.global_permission_key,
            permissions.PAGE_INCIDENTS.global_permission_key,
            permissions.PAGE_INCIDENT_DETAILS.global_permission_key,
            permissions.PAGE_ANALYTICS.global_permission_key,
            permissions.PAGE_ACCOUNT.global_permission_key,
            permissions.PAGE_ACCOUNT_CAMERAS.global_permission_key,
            permissions.PAGE_EXECUTIVE_DASHBOARD.global_permission_key,
            # Incidents
            permissions.INCIDENTS_READ.organization_permission_key,
            permissions.INCIDENTS_ASSIGN.organization_permission_key,
            permissions.INCIDENTS_UNASSIGN.organization_permission_key,
            permissions.INCIDENTS_RESOLVE.organization_permission_key,
            permissions.INCIDENTS_REOPEN.organization_permission_key,
            permissions.INCIDENTS_COMMENT.organization_permission_key,
            permissions.INCIDENTS_BOOKMARK.organization_permission_key,
            permissions.INCIDENTS_DOWNLOAD_VIDEO.organization_permission_key,
            permissions.INCIDENT_PUBLIC_LINK_CREATE.organization_permission_key,
            permissions.INCIDENT_DETAILS_READ.organization_permission_key,
            # Incident feedback
            permissions.INCIDENT_FEEDBACK_CREATE.organization_permission_key,
            # Cameras
            permissions.CAMERAS_READ.organization_permission_key,
            permissions.CAMERAS_RENAME.organization_permission_key,
            # Analytics
            permissions.ANALYTICS_READ.organization_permission_key,
            # Downtime
            permissions.DOWNTIME_READ.organization_permission_key,
            # Users
            permissions.USERS_INVITE.organization_permission_key,
            permissions.USERS_REMOVE.organization_permission_key,
            permissions.USERS_UPDATE_ROLE.organization_permission_key,
            permissions.USERS_UPDATE_SITE.organization_permission_key,
            permissions.USERS_READ.organization_permission_key,
            # Self
            permissions.SELF_SWITCH_SITE.organization_permission_key,
            permissions.SELF_UPDATE_PROFILE.global_permission_key,
        ],
    ),
    INTERNAL_REVIEWER: RoleConfig(
        INTERNAL_REVIEWER,
        "Reviewer",
        False,
        [
            # Pages
            permissions.PAGE_REVIEW_QUEUE.global_permission_key,
            permissions.PAGE_INCIDENT_DETAILS.global_permission_key,
            permissions.PAGE_ACCOUNT.global_permission_key,
            # Incidents
            permissions.INCIDENT_DETAILS_READ.global_permission_key,
            # Incident feedback
            permissions.INCIDENT_FEEDBACK_CREATE.global_permission_key,
            permissions.INCIDENT_FEEDBACK_READ.global_permission_key,
            permissions.REVIEW_QUEUE_READ.global_permission_key,
            # Other
            permissions.INTERNAL_SITE_ACCESS.global_permission_key,
            permissions.SELF_UPDATE_PROFILE.global_permission_key,
        ],
    ),
    INTERNAL_REVIEW_MANAGER: RoleConfig(
        INTERNAL_REVIEW_MANAGER,
        "Review Manager",
        False,
        [
            # Pages
            permissions.PAGE_REVIEW_QUEUE.global_permission_key,
            permissions.PAGE_REVIEW_HISTORY.global_permission_key,
            permissions.PAGE_INCIDENT_DETAILS.global_permission_key,
            permissions.PAGE_ACCOUNT.global_permission_key,
            # Incidents
            permissions.INCIDENT_DETAILS_READ.global_permission_key,
            permissions.EXPERIMENTAL_INCIDENTS_READ.global_permission_key,
            # Incident feedback
            permissions.INCIDENT_FEEDBACK_CREATE.global_permission_key,
            permissions.INCIDENT_FEEDBACK_READ.global_permission_key,
            permissions.REVIEW_QUEUE_READ.global_permission_key,
            # Users
            permissions.REVIEWER_ACCOUNTS_UPDATE_ROLE.global_permission_key,
            # Other
            permissions.INTERNAL_SITE_ACCESS.global_permission_key,
            permissions.SELF_UPDATE_PROFILE.global_permission_key,
        ],
    ),
    INTERNAL_SUPERUSER: RoleConfig(
        INTERNAL_SUPERUSER,
        "Superuser",
        False,
        [
            # Pages
            permissions.PAGE_DASHBOARD.global_permission_key,
            permissions.PAGE_EXECUTIVE_DASHBOARD.global_permission_key,
            permissions.PAGE_ANALYTICS.global_permission_key,
            permissions.PAGE_INCIDENTS.global_permission_key,
            permissions.PAGE_INCIDENT_DETAILS.global_permission_key,
            permissions.PAGE_REVIEW_QUEUE.global_permission_key,
            permissions.PAGE_REVIEW_HISTORY.global_permission_key,
            permissions.PAGE_ACCOUNT.global_permission_key,
            permissions.PAGE_ACCOUNT_CAMERAS.global_permission_key,
            # Incidents
            permissions.INCIDENTS_READ.global_permission_key,
            permissions.INCIDENTS_ASSIGN.global_permission_key,
            permissions.INCIDENTS_UNASSIGN.global_permission_key,
            permissions.INCIDENTS_RESOLVE.global_permission_key,
            permissions.INCIDENTS_REOPEN.global_permission_key,
            permissions.INCIDENTS_COMMENT.global_permission_key,
            permissions.INCIDENTS_BOOKMARK.global_permission_key,
            permissions.INCIDENTS_HIGHLIGHT.global_permission_key,
            permissions.INCIDENTS_DOWNLOAD_VIDEO.global_permission_key,
            permissions.INCIDENT_PUBLIC_LINK_CREATE.global_permission_key,
            permissions.EXPERIMENTAL_INCIDENTS_READ.global_permission_key,
            permissions.INCIDENT_DETAILS_READ.global_permission_key,
            # Incident feedback
            permissions.INCIDENT_FEEDBACK_CREATE.global_permission_key,
            permissions.INCIDENT_FEEDBACK_READ.global_permission_key,
            permissions.REVIEW_QUEUE_READ.global_permission_key,
            # Analytics
            permissions.ANALYTICS_READ.global_permission_key,
            # Cameras
            permissions.CAMERAS_READ.global_permission_key,
            permissions.CAMERAS_RENAME.global_permission_key,
            # Users
            permissions.USERS_INVITE.global_permission_key,
            permissions.USERS_REMOVE.global_permission_key,
            permissions.USERS_UPDATE_ROLE.global_permission_key,
            permissions.USERS_UPDATE_SITE.global_permission_key,
            permissions.USERS_READ.global_permission_key,
            permissions.REVIEWER_ACCOUNT_CREATE.global_permission_key,
            permissions.REVIEWER_ACCOUNTS_UPDATE_ROLE.global_permission_key,
            # Self
            permissions.SELF_SWITCH_ORGANIZATION.global_permission_key,
            permissions.SELF_SWITCH_SITE.global_permission_key,
            permissions.SELF_UPDATE_ROLE.global_permission_key,
            # Downtime
            permissions.DOWNTIME_READ.global_permission_key,
            # Zone
            permissions.CAMERAS_READ.global_permission_key,
            # Other
            permissions.TOOLBOX.global_permission_key,
            permissions.INTERNAL_SITE_ACCESS.global_permission_key,
            permissions.SELF_UPDATE_PROFILE.global_permission_key,
        ],
    ),
}


def sync_static_roles():
    """Ensures static roles exist in the database."""

    for role_config in STATIC_ROLES_CONFIG_MAP.values():
        logger.info(f"Syncing role: {role_config.key}")
        role = Role.objects.filter(key=role_config.key).first()

        if role:
            logger.info(f"Role exists: {role_config.key}")
        else:
            logger.warning(f"Creating role: {role_config.key}")
            role = Role.objects.create(
                key=role_config.key,
                name=role_config.name,
                visible_to_customers=role_config.visible_to_customers,
            )

        desired_permission_keys = set(role_config.permission_keys)
        current_permissions = role.role_permissions.filter(
            removed_at__isnull=True
        )

        for cp in current_permissions:
            if cp.permission_key in desired_permission_keys:
                # Permission already exists, remove from create list
                desired_permission_keys.remove(cp.permission_key)
            else:
                # Permission doesn't belong, mark it as removed
                logger.warning(
                    f"Removing undesired permission ({cp.permission_key}) from role: {role_config.key}"
                )
                cp.removed_at = timezone.now()
                cp.save()

        for permission_key in desired_permission_keys:
            logger.warning(
                f"Assigning desired permission ({permission_key}) to role: {role_config.key}"
            )
            RolePermission.objects.create(
                role=role,
                permission_key=permission_key,
            )
        logger.info(f"Finished syncing role: {role_config.key}")
