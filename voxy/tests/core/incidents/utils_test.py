import unittest

from core.incidents.utils import CameraConfig


class UtilsTest(unittest.TestCase):
    def testCameraConfigPresent(self) -> None:
        camera_config = CameraConfig("americold/modesto/0001/cha", 720, 1024)
        self.assertNotEqual(camera_config.version, None)

    def testCameraConfigNotPresent(self) -> None:
        self.assertRaises(
            Exception, CameraConfig, "americold/modesto/not/present", 720, 1024
        )
