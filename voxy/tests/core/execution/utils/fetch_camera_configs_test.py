import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from requests import Session
from requests.models import Response

from core.execution.utils.fetch_camera_config_lib import _fetch_camera_configs


class FetchCameraConfigsTest(unittest.TestCase):
    """Unittest supports test automation, sharing of setup and shutdown code
    for tests, aggregation of tests into collections, and independence of
    the tests from the reporting framework.

    The patch() decorator / context manager makes it easy to mock classes or
    objects in a module under test. The object you specify will be replaced with
    a mock (or other object) during the test and restored when the test ends.

    Args:
        unittest (_type_): A test case is the individual unit of testing.
        It checks for a specific response to a particular set of inputs.
        unittest provides a base class, TestCase, which may be used to
        create new test cases.
    """

    @patch(
        "core.execution.utils.fetch_camera_config_lib.get_secret_from_aws_secret_manager"
    )
    @patch.object(Session, "post")
    def test_fetch_camera_config(
        self, mock_session: MagicMock, mock_get_secrets: MagicMock
    ) -> None:
        """Magicmock replaces any nontrivial API call or object creation with
        a mock call or object. This allows you to fully define the behavior of
        the call and avoid creating real objects.

        Args:
            mock_session (MagicMock): _description_
            mock_get_secrets (MagicMock): _description_
        """

        with tempfile.TemporaryDirectory() as tempdir:
            camera_config_dir = os.path.join(tempdir, "configs/cameras/")
            os.makedirs(camera_config_dir)
            os.environ["BUILD_WORKSPACE_DIRECTORY"] = tempdir
            mock_get_secrets.return_value = '{\
                "auth_url": "auth_url",\
                "client_id": "client_id",\
                "client_secret": "secret",\
                "audience": "audience",\
                "host": "host",\
                }'
            response1 = Response()
            response1.status_code = 200
            response1._content = (  # trunk-ignore(pylint/W0212)
                b'{"access_token" : "token"}'
            )

            response2 = Response()
            response2.status_code = 200
            mapping = json.dumps([{"door_id": 1}])
            resp_content = f'{{"data" : {{\
                "cameraConfigNew": {{\
                    "doors": {json.dumps(mapping)}\
                    }}\
                }} }}'
            response2._content = (  # trunk-ignore(pylint/W0212)
                resp_content.encode("ascii")
            )
            response_ = [response1]
            for _ in range(328):
                response_.append(response2)
            mock_session.side_effect = response_
            self.assertEqual(
                _fetch_camera_configs(
                    ["configs/cameras/americold/modesto/0001/cha.yaml"]
                ),
                {
                    "americold/modesto/0001/cha": {
                        "doors": [{"door_id": 1}],
                        "version": 3,
                        "nextDoorId": 2,
                    }
                },
            )
