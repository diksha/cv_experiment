import pytest

from core.portal.testing.factories import (
    UserFactory,
    ZoneFactory,
    ZoneUserFactory,
)


@pytest.mark.django_db
def test_zone_assignable_users_returns_expected_users() -> None:
    """Verifies assignable_users property returns expected users"""
    zone = ZoneFactory()
    assignable_users = [UserFactory(), UserFactory()]
    not_assignable_users = [UserFactory(), UserFactory()]

    for user in assignable_users:
        ZoneUserFactory(zone=zone, user=user, is_assignable=True)

    for user in not_assignable_users:
        ZoneUserFactory(zone=zone, user=user, is_assignable=False)

    expected_user_ids = {u.id for u in assignable_users}
    actual_user_ids = {u.id for u in zone.assignable_users.all()}

    assert expected_user_ids == actual_user_ids
    assert zone.users.count() == len(assignable_users + not_assignable_users)


@pytest.mark.django_db
def test_zones_returns_user_zones() -> None:
    """Verifies user.zones property returns user zones."""
    # Zones
    foo_zone = ZoneFactory(key="foo")
    bar_zone = ZoneFactory(key="bar")
    ZoneFactory(key="baz")

    # User
    user = UserFactory(username="foo1")
    ZoneUserFactory(zone=foo_zone, user=user)
    ZoneUserFactory(zone=bar_zone, user=user)

    assert user.zones.count() == 2
    assert foo_zone in user.zones.all()
    assert bar_zone in user.zones.all()


@pytest.mark.django_db
def test_users_only_returns_zone_users() -> None:
    """Verifies zone.users property returns zone users."""
    # Unassigned users
    UserFactory(username="a")
    UserFactory(username="b")
    UserFactory(username="c")

    # Foo zone/users
    foo_zone = ZoneFactory(key="foo")
    foo1 = UserFactory(username="foo1")
    foo2 = UserFactory(username="foo2")
    ZoneUserFactory(zone=foo_zone, user=foo1)
    ZoneUserFactory(zone=foo_zone, user=foo2)

    # Bar zone/users
    bar_zone = ZoneFactory(key="bar")
    bar1 = UserFactory(username="bar1")
    bar2 = UserFactory(username="bar2")
    bar3 = UserFactory(username="bar3")
    ZoneUserFactory(zone=bar_zone, user=bar1)
    ZoneUserFactory(zone=bar_zone, user=bar2)
    ZoneUserFactory(zone=bar_zone, user=bar3)

    assert foo_zone.users.filter(username__contains="foo").count() == 2
    assert bar_zone.users.filter(username__contains="bar").count() == 3
