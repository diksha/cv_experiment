from dataclasses import dataclass
from typing import Dict, List, Optional

from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone
from loguru import logger

from core.portal.accounts import roles
from core.portal.accounts.models.role import Role
from core.portal.accounts.models.user_role import UserRole
from core.portal.api.models.organization import Organization
from core.portal.zones.models.zone import Zone


@dataclass(frozen=True)
class TestUserConfig:
    username: str
    email: str
    first_name: str
    last_name: str
    is_superuser: bool
    role_keys: List[str]
    auth0_id_map: Dict[str, str]


SUPERUSER = TestUserConfig(
    "superuser@example.com",
    "superuser@example.com",
    "Sue",
    "Superuser",
    True,
    [roles.INTERNAL_SUPERUSER],
    {
        # https://manage.auth0.com/dashboard/us/voxeldev/users/YXV0aDAlN0M2MWVjNTJlOGEzY2YyNTAwNjhiOGNlZmU
        "development": "auth0|61ec52e8a3cf250068b8cefe",
        "test": "auth0|61ec52e8a3cf250068b8cefe",
        # https://manage.auth0.com/dashboard/us/voxelstaging/users/YXV0aDAlN0M2MmY0OTE3ODIzZWM3NDQxYzYyNzkyYjg
        "staging": "auth0|62f4917823ec7441c62792b8",
    },
)

SITE_ADMIN = TestUserConfig(
    "site_admin@example.com",
    "site_admin@example.com",
    "Shawn",
    "Site Admin",
    False,
    [roles.EXTERNAL_ADMIN],
    {
        # https://manage.auth0.com/dashboard/us/voxeldev/users/YXV0aDAlN0M2MWVjNTU4Yzg2OGZkNDAwNjk4MzJmYmU
        "development": "auth0|61ec558c868fd40069832fbe",
        "test": "auth0|61ec558c868fd40069832fbe",
        # https://manage.auth0.com/dashboard/us/voxelstaging/users/YXV0aDAlN0M2MmY0OTE5ZDE4MDNmZGY5NmY0ZTE5NTk
        "staging": "auth0|62f4919d1803fdf96f4e1959",
    },
)

SITE_MANAGER = TestUserConfig(
    "site_manager@example.com",
    "site_manager@example.com",
    "Cindy",
    "Site Manager",
    False,
    [roles.EXTERNAL_MANAGER],
    {
        # https://manage.auth0.com/dashboard/us/voxeldev/users/YXV0aDAlN0M2MmYzMTAzYTczMzY3OTU1NDEyZDcyZGM
        "development": "auth0|62f3103a73367955412d72dc",
        "test": "auth0|62f3103a73367955412d72dc",
        # https://manage.auth0.com/dashboard/us/voxelstaging/users/YXV0aDAlN0M2MmY0OTFiYTI3NmUyOGViMzhjZWQ2Y2U
        "staging": "auth0|62f491ba276e28eb38ced6ce",
    },
)

REVIEWER = TestUserConfig(
    "reviewer@example.com",
    "reviewer@example.com",
    "Jane",
    "Reviewer",
    False,
    [roles.INTERNAL_REVIEWER],
    {
        # https://manage.auth0.com/dashboard/us/voxeldev/users/YXV0aDAlN0M2MWVjNTVhYThiN2UzMjAwNmFlNWJjOGM
        "development": "auth0|61ec55aa8b7e32006ae5bc8c",
        "test": "auth0|61ec55aa8b7e32006ae5bc8c",
        # https://manage.auth0.com/dashboard/us/voxelstaging/users/YXV0aDAlN0M2MmY0OTFkMGU1NzhkOTg1ZmU0MTczMzY
        "staging": "auth0|62f491d0e578d985fe417336",
    },
)

REVIEW_MANAGER = TestUserConfig(
    "review_manager@example.com",
    "review_manager@example.com",
    "Harold",
    "Review Manager",
    False,
    [roles.INTERNAL_REVIEW_MANAGER],
    {
        # https://manage.auth0.com/dashboard/us/voxeldev/users/YXV0aDAlN0M2MWVjNTVjNjg2OGZkNDAwNjk4MzJmY2U
        "development": "auth0|61ec55c6868fd40069832fce",
        "test": "auth0|61ec55c6868fd40069832fce",
        # https://manage.auth0.com/dashboard/us/voxelstaging/users/YXV0aDAlN0M2MmY0OTFmMGE4ODAxNjc5ZDdkNmQwMjI
        "staging": "auth0|62f491f0a8801679d7d6d022",
    },
)

TEST_USER_CONFIGS = [
    SUPERUSER,
    SITE_ADMIN,
    SITE_MANAGER,
    REVIEWER,
    REVIEW_MANAGER,
]


def sync_test_users():
    for user_config in TEST_USER_CONFIGS:
        sync_test_user(user_config)


def sync_test_user(
    user_config: TestUserConfig,
    organizations: Optional[List[Organization]] = None,
    zones: Optional[List[Zone]] = None,
) -> User:
    if settings.PRODUCTION:
        raise RuntimeError("Please don't run this in production")

    logger.info(f"Syncing test user: {user_config.username}")

    # Ensure user exists
    try:
        user = User.objects.get(username=user_config.username)
        logger.info(f"Test user exists: {user_config.username}")
    except User.DoesNotExist:
        logger.warning(f"Creating test user: {user_config.username}")
        user = User.objects.create(username=user_config.username)

    # Sync user attributes
    user.email = user_config.email
    user.first_name = user_config.first_name
    user.last_name = user_config.last_name
    user.is_staff = user_config.is_superuser
    user.is_superuser = user_config.is_superuser
    user.save()

    # Sync Auth0 ID
    auth0_id = user_config.auth0_id_map.get(settings.ENVIRONMENT)
    if auth0_id:
        user.profile.data = {"auth0_id": auth0_id}
        user.profile.save()
    else:
        logger.error(
            f"No Auth0 ID defined for environment: {settings.ENVIRONMENT}"
        )

    sync_test_user_roles(user, user_config.role_keys)
    sync_test_user_organizations(user, organizations)
    sync_test_user_zones(user, zones)
    logger.info("Done")


def sync_test_user_roles(user: User, role_keys: List[str]) -> None:
    expected_role_keys = set(role_keys)
    for user_role in user.user_roles.filter(removed_at__isnull=True):
        if user_role.role.key in expected_role_keys:
            expected_role_keys.remove(user_role.role.key)
        else:
            logger.warning(
                f"Removing undesired role from test user: {user.username} - {user_role.role.key}"
            )
            user_role.removed_at = timezone.now()
            user_role.save()

    for role_key in expected_role_keys:
        logger.warning(
            f"Assigning desired role to test user: {user.username} - {role_key}"
        )
        UserRole.objects.create(
            user=user,
            role=Role.objects.get(key=role_key),
        )


def sync_test_user_organizations(
    user: User, organizations: List[Organization]
) -> None:
    # Only sync if values are provided
    if organizations:
        expected_organizations = set(organizations)
        user.profile.organization = organizations[0]
        user.profile.save()

        for org in user.organizations.all():
            if org in expected_organizations:
                expected_organizations.remove(org)
            else:
                logger.warning(
                    f"Removing user from undesired organization: {user.username} - {org.key}"
                )
                org.users.remove(user)

        for org in expected_organizations:
            logger.warning(
                f"Adding test user to desired organization: {user.username} - {org.key}"
            )
            org.users.add(user)


def sync_test_user_zones(user: User, zones: List[Zone]) -> None:
    # Only sync if values are provided
    if zones:
        expected_zones = set(zones)
        user.profile.site = zones[0]
        user.profile.save()

        for zone in user.zones.all():
            if zone in expected_zones:
                expected_zones.remove(zone)
            else:
                logger.warning(
                    f"Removing user from undesired zone: {user.username} - {zone.key}"
                )
                zone.users.remove(user)

        for zone in expected_zones:
            logger.warning(
                f"Adding test user to desired zone: {user.username} - {zone.key}"
            )
            zone.users.add(user)
