# import unit test
import unittest

import numpy as np
import torch

from core.perception.pose.vit_pose import ViTPoseModel
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon, Pose
from core.structs.frame import Frame

# trunk-ignore-begin(pylint/E0611)
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)
from third_party.vit_pose.configs.ViTPose_base_coco_256x192 import (
    model as model_cfg,
)
from third_party.vit_pose.models.model import ViTPose


def generate_dummy_model() -> str:
    """
    Generates dummy untrained vit pose model

    Returns:
        str: the path to the model
    """
    model = ViTPose(model_cfg).eval()
    sample_batch_size = 1
    n_channels = 3
    model_input_size = [192, 256]

    trace_inputs = torch.zeros(
        sample_batch_size, n_channels, *model_input_size, device="cpu"
    )
    tmp_model_name = "tmp_vit_pose.pt"
    torch.jit.trace(model, trace_inputs).save(tmp_model_name)
    return tmp_model_name


class ViTPoseTest(unittest.TestCase):
    def setUp(self):

        self.model = ViTPoseModel(
            model_path=generate_dummy_model(),
            gpu_runtime=GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL,
            triton_server_url="localhost:8001",
        )

    def _get_dummy_polygon(self):
        polygon_corners = [
            Point(0, 0),
            Point(0, 100),
            Point(100, 100),
            Point(100, 0),
        ]
        return Polygon(polygon_corners)

    def test_model(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # other required fields:
        frame_struct = Frame(
            actors=[
                Actor(
                    track_id=0,
                    category=ActorCategory.PERSON,
                    polygon=self._get_dummy_polygon(),
                )
            ],
            frame_number=0,
            frame_width=1920,
            frame_height=1080,
            relative_timestamp_s=0,
            relative_timestamp_ms=0,
            epoch_timestamp_ms=0,
        )
        frame_struct = self.model(frame, frame_struct)
        self.assertEqual(len(frame_struct.actors), 1)
        self.assertTrue(isinstance(frame_struct.actors[0].pose, Pose))
