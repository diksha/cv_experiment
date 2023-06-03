from typing import Callable

import pytest

from core.portal.testing.factories import (
    CameraFactory,
    CameraIncidentTypeFactory,
    IncidentTypeFactory,
    OrganizationFactory,
)

# trunk-ignore-all(pylint/E1101): need to access 'add' member methods in this test


@pytest.mark.django_db
def test_only_returns_enabled_types() -> None:
    """Test that enabled_incident_types only returns enabled types."""
    incident_type_1 = IncidentTypeFactory(key="foo")
    incident_type_2 = IncidentTypeFactory(key="bar")
    incident_type_3 = IncidentTypeFactory(key="bazz")

    cam = CameraFactory()
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_1,
        enabled=True,
    )
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_2,
        enabled=True,
    )
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_3,
        enabled=False,
    )
    org = OrganizationFactory(cameras=[cam])
    org.incident_types.add(incident_type_1)
    org.incident_types.add(incident_type_2)
    org.incident_types.add(incident_type_3)

    assert len(org.enabled_incident_types) == 2

    for incident_type in org.enabled_incident_types:
        assert incident_type.enabled
        assert incident_type.name is not None


@pytest.mark.django_db
def test_incident_type_data_is_prefetched(
    django_assert_num_queries: Callable,
) -> None:
    """Test that incident type data is prefetched for org incident types.

    Args:
        django_assert_num_queries (Callable): django ORM assertion helper
    """
    cam = CameraFactory()
    keys = ["a", "b", "c", "d", "e", "f"]
    org = OrganizationFactory(cameras=[cam])

    for key in keys:
        incident_type = IncidentTypeFactory(key=key)

        cam.incident_types.add(incident_type)
        org.incident_types.add(incident_type)

    # Only 2 query should be executed when related model fields are accessed
    with django_assert_num_queries(2):
        for incident_type in org.enabled_incident_types:
            assert incident_type.key in keys
            assert incident_type.name is not None


@pytest.mark.django_db
def test_returns_unique_and_enabled_camera_types() -> None:
    """Test that unique and enabled camera types are returned"""
    incident_type_1 = IncidentTypeFactory(key="foo")
    incident_type_2 = IncidentTypeFactory(key="bar")
    incident_type_3 = IncidentTypeFactory(key="bazz")

    cam = CameraFactory()
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_1,
        enabled=True,
    )
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_2,
        enabled=True,
    )
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_3,
        enabled=False,
    )

    cam2 = CameraFactory()
    CameraIncidentTypeFactory(
        camera=cam2,
        incident_type=incident_type_1,
        enabled=True,
    )

    # non-org camera
    cam3 = CameraFactory()
    CameraIncidentTypeFactory(
        camera=cam3,
        incident_type=incident_type_1,
        enabled=True,
    )

    org = OrganizationFactory(cameras=[cam, cam2])
    org.incident_types.add(incident_type_1)
    org.incident_types.add(incident_type_2)
    org.incident_types.add(incident_type_3)

    assert len(org.enabled_incident_types) == 2

    for incident_type in org.enabled_incident_types:
        assert incident_type.enabled
        assert incident_type.name is not None


@pytest.mark.django_db
def test_returns_incident_types_from_only_one_org() -> None:
    """Test that types from other orgs are not returned"""
    incident_type_1 = IncidentTypeFactory(key="foo")

    cam = CameraFactory()
    CameraIncidentTypeFactory(
        camera=cam,
        incident_type=incident_type_1,
        enabled=True,
    )

    org = OrganizationFactory(cameras=[cam])
    org2 = OrganizationFactory()

    org.incident_types.add(incident_type_1)
    org2.incident_types.add(incident_type_1)

    assert len(org.enabled_incident_types) == 1

    for incident_type in org.enabled_incident_types:
        assert incident_type.organization == org
        assert incident_type.enabled
        assert incident_type.name is not None
