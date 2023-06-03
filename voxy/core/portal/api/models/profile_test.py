import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from core.portal.accounts.models.role import Role
from core.portal.testing.factories import (
    RoleFactory,
    RolePermissionFactory,
    UserFactory,
    UserRoleFactory,
)

ACTIVE_PERMISSION_PREFIX = "active"
INACTIVE_PERMISSION_PREFIX = "inactive"


@pytest.fixture(name="role_with_active_permissions_1")
def _role_with_active_permissions_1() -> Role:
    role = RoleFactory()
    for _ in range(3):
        role.role_permissions.add(RolePermissionFactory())
    return role


@pytest.fixture(name="role_with_active_permissions_2")
def _role_with_active_permissions_2() -> Role:
    role = RoleFactory()
    for _ in range(3):
        role.role_permissions.add(RolePermissionFactory())
    return role


@pytest.fixture(name="role_with_inactive_permissions")
def _role_with_inactive_permissions() -> Role:
    role = RoleFactory()
    for _ in range(3):
        role.role_permissions.add(
            RolePermissionFactory(removed_at=timezone.now())
        )
    return role


@pytest.mark.django_db
def test_profile_auto_created_for_new_user() -> None:
    u = User.objects.create()
    assert u.profile is not None


@pytest.mark.django_db
def test_permissions_only_includes_active_roles_permissions(
    role_with_active_permissions_1: Role,
    role_with_active_permissions_2: Role,
) -> None:
    user = UserFactory()

    # Active role
    UserRoleFactory(user=user, role=role_with_active_permissions_1)

    # Inactive role
    UserRoleFactory(
        user=user,
        role=role_with_active_permissions_2,
        removed_at=timezone.now(),
    )

    expected_permissions = {
        rp.permission_key
        for rp in role_with_active_permissions_1.role_permissions.all()
    }

    assert set(user.profile.permissions) == expected_permissions


@pytest.mark.django_db
def test_permissions_excludes_inactive_permissions(
    role_with_inactive_permissions: Role,
) -> None:
    user = UserFactory()
    UserRoleFactory(user=user, role=role_with_inactive_permissions)

    assert len(user.profile.permissions) == 0
