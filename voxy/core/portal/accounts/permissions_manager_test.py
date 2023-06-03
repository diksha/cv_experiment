import pytest

from core.portal.accounts.permissions import Permission
from core.portal.accounts.permissions_manager import (
    has_global_permission,
    has_organization_permission,
    has_zone_permission,
)
from core.portal.testing.factories import (
    OrganizationFactory,
    UserFactory,
    ZoneFactory,
)


@pytest.mark.django_db
def test_has_zone_permission_with_org_permission():
    user = UserFactory()
    zone = ZoneFactory(key="test")

    zone.organization.users.set([user])
    user.permissions = ["organization:test"]

    assert has_zone_permission(user, zone, Permission("test")) is True


@pytest.mark.django_db
def test_has_zone_permission_with_zone_permission():
    user = UserFactory()
    zone = ZoneFactory(key="test")

    zone.users.set([user])
    user.permissions = ["zone:test"]

    assert has_zone_permission(user, zone, Permission("test")) is True


@pytest.mark.django_db
def test_has_zone_permission_with_global_permission():
    user = UserFactory()
    zone = ZoneFactory(key="test")

    user.permissions = ["global:test"]

    assert has_zone_permission(user, zone, Permission("test")) is True


@pytest.mark.django_db
def test_does_not_have_zone_permission():
    user = UserFactory()
    zone = ZoneFactory(key="test")

    user.permissions = ["zone:test", "organization:test"]

    assert has_zone_permission(user, zone, Permission("test")) is False


@pytest.mark.django_db
def test_has_org_permission_with_org_permission():
    user = UserFactory()
    org = OrganizationFactory()

    org.users.set([user])
    user.permissions = ["organization:test"]

    assert has_organization_permission(user, org, Permission("test")) is True


@pytest.mark.django_db
def test_has_org_permission_with_global_permission():
    user = UserFactory()
    org = OrganizationFactory()

    user.permissions = ["global:test"]

    assert has_organization_permission(user, org, Permission("test")) is True


@pytest.mark.django_db
def test_does_not_have_org_permission():
    user = UserFactory()
    org = OrganizationFactory()

    user.permissions = ["organization:test"]

    assert has_organization_permission(user, org, Permission("test")) is False


@pytest.mark.django_db
def test_has_global_permission_with_global_permission():
    user = UserFactory()

    user.permissions = ["global:test"]

    assert has_global_permission(user, Permission("test")) is True


@pytest.mark.django_db
def test_does_not_have_global_permission():
    user = UserFactory()

    user.permissions = ["organization:test"]

    assert has_global_permission(user, Permission("test")) is False
