import pytest

from core.portal.accounts.models.role import Role
from core.portal.accounts.roles import (
    STATIC_ROLES_CONFIG_MAP,
    sync_static_roles,
)


def verify_role_state():
    for role_config in STATIC_ROLES_CONFIG_MAP.values():
        role = Role.objects.get(key=role_config.key)
        role.name = role_config.name
        role.visible_to_customers = role_config.visible_to_customers
        expected_permission_keys = set(role_config.permission_keys)
        actual_permission_keys = set(
            role.role_permissions.filter(removed_at__isnull=True).values_list(
                "permission_key", flat=True
            )
        )
        assert expected_permission_keys == actual_permission_keys


@pytest.mark.django_db
def test_sync_static_roles() -> None:
    # Start with empty roles table
    assert Role.objects.count() == 0

    # Sync roles and verify
    sync_static_roles()
    verify_role_state()

    # Sync roles again and verify roles are not duplicated, etc.
    sync_static_roles()
    verify_role_state()

    # Slightly modify all roles
    for role in Role.objects.all():
        role.name += "zzz"
        role.visible_to_customers = not role.visible_to_customers
        role.save()
        for role_permission in role.role_permissions.all():
            role_permission.permission_key += "zzz"
            role_permission.save()

    # Sync roles and verify
    sync_static_roles()
    verify_role_state()
