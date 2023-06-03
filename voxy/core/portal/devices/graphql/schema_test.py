import json

import graphene
import mock
import pytest
from django.contrib.auth.models import User
from graphene.test import Client

from core.portal.api.models.organization import Organization
from core.portal.devices.graphql.schema import (
    CameraConfigNewMutations,
    CameraQueries,
)
from core.portal.devices.models.camera import Camera
from core.portal.testing.factories import (
    CameraFactory,
    OrganizationFactory,
    UserFactory,
)


@pytest.fixture(name="organization_foo")
def _organization_foo():
    """Organization fixture (Foo).

    Returns:
        Organization: organization
    """
    return OrganizationFactory(key="FOO", name="Foo")


@pytest.fixture(name="organization_bar")
def _organization_bar():
    """Organization fixture (Bar).

    Returns:
        Organization: organization
    """
    return OrganizationFactory(key="BAR", name="bar")


@pytest.fixture(name="user1")
def _user1(organization_foo: Organization):
    """User fixture (Foo org).

    Args:
        organization_foo (Organization): foo organization

    Returns:
        User: user
    """
    return UserFactory(
        username="admin",
        email="admin1@example.com",
        is_staff=True,
        is_superuser=True,
        profile__organization=organization_foo,
    )


@pytest.fixture(name="user2")
def _user2():
    """User fixture (no org).

    Returns:
        User: user
    """
    return UserFactory(
        username="admin2",
        email="admin2@example.com",
        profile__organization=None,
    )


@pytest.fixture(name="user3")
def _user3(organization_bar: Organization):
    """User fixture (Bar org).

    Args:
        organization_bar (Organization): bar organization

    Returns:
        User: user
    """
    return UserFactory(
        username="admin3",
        email="admin3@example.com",
        is_staff=True,
        is_superuser=True,
        profile__organization=organization_bar,
    )


@pytest.fixture(name="camera1")
def _camera1(organization_foo: Organization):
    """Camera fixture (Foo org).

    Args:
        organization_foo (Organization): organization

    Returns:
        Camera: camera
    """
    return CameraFactory(uuid="1", name="name", organization=organization_foo)


@pytest.fixture(name="camera2")
def _camera2(organization_bar: Organization):
    """Camera fixture (Bar org).

    Args:
        organization_bar (Organization): organization

    Returns:
        Camera: camera
    """
    return CameraFactory(
        uuid="2", name="anothername", organization=organization_bar
    )


@pytest.mark.django_db
def test_camera_config_queries(camera1: Camera) -> None:
    """Test camera config happy path.

    Args:
        camera1 (Camera): test camera
    """
    client = Client(
        graphene.Schema(query=CameraQueries, mutation=CameraConfigNewMutations)
    )
    door_json = json.dumps(
        json.dumps([{"polygon": [[0, 0]], "orientation": "front_door"}])
    )
    query = """
    mutation CreateCameraConfig($uuid: String!, $doors:JSONString) {
        cameraConfigNewCreate(
            uuid: $uuid,
            doors: $doors
        ) {
            cameraConfigNew {
                doors
            }
        }
    }
    """
    qvars = {"uuid": "1", "doors": door_json}
    # Test camera created
    executed = client.execute(query, variables=qvars)
    executed = client.execute(
        """
        {
            cameraConfigNew(uuid:"1", version:1) {
                doors
            }
        }
        """
    )
    assert executed == {
        "data": {
            "cameraConfigNew": {
                "doors": '"[{\\"polygon\\": [[0, 0]], \\"orientation\\": \\"front_door\\"}]"'
            }
        }
    }


@pytest.mark.django_db
@mock.patch("core.portal.devices.graphql.schema.ORG_KEY_ALLOWLIST", ["FOO"])
def test_camera_queries_foo(
    user1: User,
    camera1: Camera,
    camera2: Camera,
) -> None:
    """Test cameras field returns expected cameras when present.

    Args:
        user1 (User): test user
        camera1 (Camera): test camera
        camera2 (Camera): test camera
    """

    class TestContext:
        user = user1

    context = TestContext()
    client = Client(
        graphene.Schema(query=CameraQueries, mutation=CameraConfigNewMutations)
    )
    executed = client.execute(
        """
        {
            cameras {
                uuid
            }
        }
        """,
        context=context,
    )
    assert executed == {"data": {"cameras": [{"uuid": "1"}]}}


@pytest.mark.django_db
def test_camera_queries_none(
    user2: User, camera1: Camera, camera2: Camera
) -> None:
    """Test cameras field returns empty list when no cameras present.

    Args:
        user2 (User): test user
        camera1 (Camera): test camera
        camera2 (Camera): test camera
    """

    class TestContext:
        user = user2

    context = TestContext()
    client = Client(
        graphene.Schema(query=CameraQueries, mutation=CameraConfigNewMutations)
    )
    executed = client.execute(
        """
        {
            cameras {
                uuid
            }
        }
        """,
        context=context,
    )
    assert executed == {"data": {"cameras": []}}


@pytest.mark.django_db
def test_camera_queries_bar(
    user3: User, camera1: Camera, camera2: Camera
) -> None:
    """Test cameras field returns expected cameras when present.

    Args:
        user3 (User): test user
        camera1 (Camera): test camera
        camera2 (Camera): test camera
    """
    query = """
    {
        cameras {
            uuid
        }
    }
    """
    expected_response = {"data": {"cameras": [{"uuid": "1"}, {"uuid": "2"}]}}

    # Graphene
    class GrapheneContext:
        user = user3

    client = Client(
        graphene.Schema(query=CameraQueries, mutation=CameraConfigNewMutations)
    )
    graphene_response = client.execute(query, context=GrapheneContext())
    assert graphene_response == expected_response
