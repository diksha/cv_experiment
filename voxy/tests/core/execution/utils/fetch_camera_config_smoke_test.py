import json
import unittest

from core.execution.utils.fetch_camera_config_lib import _fetch_camera_configs


class FetchCameraConfigsNightlyTest(unittest.TestCase):
    def test_fetch_camera_config(self) -> None:
        with open("configs/cameras/camera_config.json", encoding="utf-8") as f:
            camera_config_file = json.loads(f.read())
        with open("configs/cameras/cameras", encoding="utf-8") as f:
            cameras = []
            for line in f:
                cameras.append(line.rstrip("\n"))
            camera_config = _fetch_camera_configs(cameras)
            assert camera_config == camera_config_file
