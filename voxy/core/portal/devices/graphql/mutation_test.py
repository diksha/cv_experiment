import json

import graphene
import pytest
from graphene.test import Client

from core.portal.devices.graphql.schema import (
    CameraConfigNewMutations,
    CameraQueries,
)
from core.portal.devices.models.camera import Camera


@pytest.mark.django_db
def test_camera_config_mutations() -> None:
    """Test that camera config create mutation creates new camera config."""
    client = Client(
        graphene.Schema(query=CameraQueries, mutation=CameraConfigNewMutations)
    )
    Camera(uuid="1", name="name").save()
    # Test camera created
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
    executed = client.execute(query, variables=qvars)
    assert executed == {
        "data": {
            "cameraConfigNewCreate": {
                "cameraConfigNew": {
                    "doors": '"[{\\"polygon\\": [[0, 0]], \\"orientation\\": \\"front_door\\"}]"'
                }
            }
        }
    }

    # Test database updated
    door_json = json.dumps(
        json.dumps([{"polygon": [[1, 1]], "orientation": "front_door"}])
    )
    query = """
    mutation CreateCameraConfig($uuid: String!, $doors:JSONString) {
        cameraConfigNewCreate(
            uuid: $uuid,
            doors: $doors
        ) {
            cameraConfigNew {
                doors,version
            }
        }
    }
    """
    qvars = {"uuid": "1", "doors": door_json}
    executed = client.execute(query, variables=qvars)
    assert executed == {
        "data": {
            "cameraConfigNewCreate": {
                "cameraConfigNew": {
                    "doors": '"[{\\"polygon\\": [[1, 1]], \\"orientation\\": \\"front_door\\"}]"',
                    "version": 2,
                }
            }
        }
    }
