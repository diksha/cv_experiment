import mock
import pytest

from core.portal.accounts.middleware import PermissionsMiddleware
from core.portal.testing.factories import (
    RoleFactory,
    RolePermissionFactory,
    UserFactory,
    UserRoleFactory,
)


@pytest.mark.django_db
def test_permissions_middleware() -> None:
    """Permissions middleware happy path test."""

    user = UserFactory()
    role = RoleFactory()
    UserRoleFactory(user=user, role=role)

    expected_permission_keys = [
        "foo",
        "bar",
        "buzz",
        "baz",
    ]

    for key in expected_permission_keys:
        RolePermissionFactory(role=role, permission_key=key)

    get_response_mock = mock.Mock()
    permissions_middleware = PermissionsMiddleware(
        get_response=get_response_mock
    )
    permissions_middleware(mock.Mock(user=user))
    positional_args = get_response_mock.call_args[0]
    mocked_request = positional_args[0]
    calculated_permissions = mocked_request.user.permissions
    assert mocked_request.user.permissions is not None
    assert set(calculated_permissions) == set(expected_permission_keys)
