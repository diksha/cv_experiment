import copy
import typing
import unittest

import cv2
import numpy as np
import torch
from PIL import Image

from core.incidents.utils import CameraConfig, DoorCameraConfig
from core.perception.inference.transforms.registry import get_transforms
from core.structs.actor import Actor, ActorCategory, DoorOrientation, DoorState
from core.structs.attributes import Point, Polygon, RectangleXYWH
from lib.ml.inference.tasks.door_state.vanilla_resnet.utils import (
    crop_and_recolor_door,
    extact_door_configs,
    extract_preprocessed_door_image,
    pad_polygon,
    post_process_prediction,
)


class DoorVanillaResnetInferenceUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")
        self.image_shape = (480, 960)  # height, width
        self.camera_uuid = "americold/modesto/0001/cha"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state2class = {"closed": 0, "open": 1, "partially_open": 2}
        rgb_range = 255
        n_color_channels = 3
        self.image = np.random.randint(
            rgb_range,
            size=(self.image_shape[0], self.image_shape[1], n_color_channels),
            dtype=np.uint8,
        )
        self.preprocessing_transforms = {"bgr2rgb": True}
        self.training_transforms = get_transforms(
            [
                {
                    "name": "Resize",
                    "params": {
                        "size": [224, 224],
                    },
                },
                {
                    "name": "ToTensor",
                    "params": {},
                },
                {
                    "name": "Normalize",
                    "params": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            ]
        )
        self.postprocessing_transforms = {"padding": 30}
        self.prediction = torch.rand(len(self.state2class))

    def _get_image(self) -> np.ndarray:
        """Returns copy of test image
        Returns:
            np.ndarray: copy of test image
        """
        return copy.deepcopy(self.image)

    def _get_prediction(self) -> torch.tensor:
        """Returns copy of test prediction tensor
        Returns:
            torch.tensor: copy of prediction tensor
        """
        return copy.deepcopy(self.prediction)

    def _legacy_get_doors(self) -> typing.List[DoorCameraConfig]:
        """Legacy get doors from camera config
        Returns:
            typing.List[DoorCameraConfig]: list of doors from testing
            camera config
        """
        camera_config = CameraConfig(
            self.camera_uuid, self.image_shape[0], self.image_shape[1]
        )
        return camera_config.doors

    def _legacy_crop_and_recolor_door(
        self, frame: np.ndarray, door: DoorCameraConfig
    ) -> np.ndarray:
        """Legacy crop and recolor doors from state_classifier_vision
        Args:
            frame (np.ndarray): input image
            door (DoorCameraConfig): door to crop
        Returns:
            np.ndarray: cropped door image
        """
        rect = RectangleXYWH.from_polygon(door.polygon)
        cropped_image = frame[
            max(0, rect.top_left_vertice.y) : min(
                frame.shape[0], rect.top_left_vertice.y + rect.h
            ),
            max(0, rect.top_left_vertice.x) : min(
                frame.shape[1], rect.top_left_vertice.x + rect.w
            ),
        ]
        # Converting image to rgb
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        return cropped_image

    def _legacy_pad_polygon(
        self, door: DoorCameraConfig, door_polygon_pad: int
    ) -> Polygon:
        """Legacy pad polygon
        Args:
            door (DoorCameraConfig): door to pad
            door_polygon_pad (int): amount to pad
        Returns:
            Polygon: padded door polygon
        """
        door_polygon = copy.deepcopy(door.polygon)

        return Polygon(
            vertices=[
                Point(
                    max(
                        door_polygon.get_top_left().x - door_polygon_pad,
                        0,
                    ),
                    max(
                        door_polygon.get_top_left().y - door_polygon_pad,
                        0,
                    ),
                ),
                Point(
                    door_polygon.get_top_right().x + door_polygon_pad,
                    max(
                        door_polygon.get_top_right().y - door_polygon_pad,
                        0,
                    ),
                ),
                Point(
                    door_polygon.get_bottom_right().x + door_polygon_pad,
                    door_polygon.get_bottom_right().y + door_polygon_pad,
                ),
                Point(
                    max(
                        door_polygon.get_bottom_left().x - door_polygon_pad,
                        0,
                    ),
                    door_polygon.get_bottom_left().y + door_polygon_pad,
                ),
            ]
        )

    def _legacy_postprocess_predictions(
        self,
        prediction: torch.tensor,
        track_id: int,
        door: DoorCameraConfig,
        pad_amount: int,
    ) -> Actor:
        """Legacy postprocess prediction
        Args:
            prediction (torch.tensor): test inference tensor
            door (DoorCameraConfig): door
            track_id (int): track id
            pad_amount (int): amount to pad polygon
        Returns:
            Actor: door actor
        """
        prediction = torch.softmax(prediction, dim=0).cpu()
        (
            closed_probability,
            open_probability,
            partially_open_probability,
        ) = list(prediction)

        vocabulary = {"closed": 0, "open": 1, "partially_open": 2}
        index = int(np.ndarray.argmax(prediction.detach().numpy()))

        door_state = DoorState.FULLY_CLOSED
        if index == vocabulary["open"]:
            door_state = DoorState.FULLY_OPEN
        elif index == vocabulary["partially_open"]:
            door_state = DoorState.PARTIALLY_OPEN

        door_orientation = (
            DoorOrientation.SIDE_DOOR
            if door.orientation == "SIDE_DOOR"
            else DoorOrientation.FRONT_DOOR
        )

        # There are some faulty door crossing events generated
        # due to the detected PIT being slightly outside the
        # door label. This results in multiple entering/exiting
        # events. Padding is a quick fix that should reduce our
        # piggyback false positives. In the future, we should
        # change the bounding box points for the camera, but
        # this would also affect the performance of the classifier.
        door_polygon = self._legacy_pad_polygon(door, pad_amount)

        return Actor(
            category=ActorCategory.DOOR,
            track_id=track_id,
            polygon=door_polygon,
            manual=False,
            confidence=float(open_probability)
            + float(partially_open_probability),
            door_state=door_state,
            door_orientation=door_orientation,
            door_state_probabilities=[
                float(open_probability),
                float(partially_open_probability),
                float(closed_probability),
            ],
        )

    def test_preprocess_image(self) -> None:
        """Test preprocess image step"""
        legacy_doors = self._legacy_get_doors()
        current_doors = extact_door_configs(
            self.camera_uuid, self.image_shape[0], self.image_shape[1]
        )
        self.assertEqual(len(legacy_doors), len(current_doors))
        for legacy_door, current_door in zip(legacy_doors, current_doors):
            legacy_cropped_door = self._legacy_crop_and_recolor_door(
                self._get_image(), legacy_door
            )
            cropped_door = crop_and_recolor_door(
                self._get_image(),
                current_door,
                self.preprocessing_transforms["bgr2rgb"],
            )
            self.assertTrue(np.array_equal(legacy_cropped_door, cropped_door))
            legacy_preprocessed_image = (
                self.training_transforms(Image.fromarray(legacy_cropped_door))
                .to(self.device)
                .unsqueeze(0)
            )
            preprocessed_image = extract_preprocessed_door_image(
                self._get_image(),
                current_door,
                self.preprocessing_transforms,
                self.training_transforms,
                self.device,
            )
            self.assertTrue(
                torch.equal(legacy_preprocessed_image, preprocessed_image)
            )

    def test_postprocess_result(self) -> None:
        """Test postprocess inference result"""
        legacy_doors = self._legacy_get_doors()
        current_doors = extact_door_configs(
            self.camera_uuid, self.image_shape[0], self.image_shape[1]
        )
        self.assertEqual(len(legacy_doors), len(current_doors))
        for track_id, (legacy_door, current_door) in enumerate(
            zip(legacy_doors, current_doors)
        ):
            legacy_padded_poly = self._legacy_pad_polygon(
                legacy_door, self.postprocessing_transforms["padding"]
            )
            padded_poly = pad_polygon(
                current_door, self.postprocessing_transforms["padding"]
            )
            self.assertTrue(legacy_padded_poly.iou(padded_poly) == 1.0)
            legacy_actor = self._legacy_postprocess_predictions(
                self._get_prediction(),
                track_id,
                legacy_door,
                self.postprocessing_transforms["padding"],
            )
            actor = post_process_prediction(
                self._get_prediction(),
                self.state2class,
                track_id,
                current_door,
                self.postprocessing_transforms,
                self.camera_uuid,
            )
            self.assertEqual(legacy_actor.category, actor.category)
            self.assertEqual(legacy_actor.track_id, actor.track_id)
            self.assertEqual(
                "3041c916-9b45-47f0-b264-5fee089a3fd0", actor.track_uuid
            )
            self.assertTrue(legacy_actor.polygon.iou(actor.polygon) == 1.0)
            self.assertEqual(legacy_actor.manual, actor.manual)
            self.assertEqual(legacy_actor.confidence, actor.confidence)
            self.assertEqual(legacy_actor.door_state, actor.door_state)
            self.assertEqual(
                legacy_actor.door_orientation, actor.door_orientation
            )
            self.assertEqual(
                legacy_actor.door_state_probabilities,
                actor.door_state_probabilities,
            )
