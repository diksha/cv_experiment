import unittest
from unittest.mock import patch

from core.incidents.utils import DoorCameraConfig
from core.perception.door.state_classifier_vision import (
    VanillaResnetClassifier,
)
from core.structs.attributes import Point, Polygon

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class InterpolationTest(unittest.TestCase):
    @patch(
        "core.perception.door.state_classifier_vision.VanillaResnetClassifier.load_model"
    )
    def test_pad_polygon_top(self, mock_load_method) -> None:
        """Test padding logic for doors.

        Args:
            mock_load_method (MagicMock): load model method
        """
        vanilla_resnet_classifier = VanillaResnetClassifier(
            camera_uuid="uuid",
            model_path="model_path",
            config={},
            gpu_runtime=GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL,
            triton_server_url="triton.server.name",
        )
        door = DoorCameraConfig(
            polygon=Polygon(
                vertices=[
                    Point(0.0, 10.0),
                    Point(15.0, 5.5),
                    Point(2.1, 6.4),
                    Point(8.0, 4.5),
                    Point(1.8, 0),
                ]
            ),
            orientation="side_door",
            door_id=1,
            door_type="freezer",
        )
        self.assertEqual(
            vanilla_resnet_classifier.pad_polygon(door).to_dict(),
            {
                "vertices": [
                    {"x": 0, "y": 0, "z": None},
                    {"x": 45.0, "y": 0, "z": None},
                    {"x": 45.0, "y": 40.0, "z": None},
                    {"x": 0, "y": 40.0, "z": None},
                ]
            },
        )

    @patch(
        "core.perception.door.state_classifier_vision.VanillaResnetClassifier.load_model"
    )
    def test_pad_polygon_center(self, mock_load_method) -> None:
        """Test padding logic for doors.

        Args:
            mock_load_method (MagicMock): load model method
        """
        vanilla_resnet_classifier = VanillaResnetClassifier(
            camera_uuid="uuid",
            model_path="model_path",
            config={},
            gpu_runtime=GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL,
            triton_server_url="triton.server.name",
        )
        door = DoorCameraConfig(
            polygon=Polygon(
                vertices=[
                    Point(40.0, 35.0),
                    Point(45.0, 40.5),
                    Point(35.1, 38.4),
                    Point(38.0, 34.5),
                    Point(31.8, 40),
                ]
            ),
            orientation="side_door",
            door_id=1,
            door_type="freezer",
        )
        self.assertEqual(
            vanilla_resnet_classifier.pad_polygon(door).to_dict(),
            {
                "vertices": [
                    {"x": 1.8000000000000007, "y": 4.5, "z": None},
                    {"x": 75.0, "y": 4.5, "z": None},
                    {"x": 75.0, "y": 70.5, "z": None},
                    {"x": 1.8000000000000007, "y": 70.5, "z": None},
                ]
            },
        )
