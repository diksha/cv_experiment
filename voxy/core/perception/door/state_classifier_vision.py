#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import copy
import pickle  # trunk-ignore(bandit/B403)
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from sklearn.neighbors import KNeighborsClassifier

from core.incidents.utils import CameraConfig
from core.structs.actor import (
    Actor,
    ActorCategory,
    DoorOrientation,
    DoorState,
    get_actor_id_from_actor_category_and_track_id,
    get_track_uuid,
)
from core.structs.attributes import Point, Polygon, RectangleXYWH
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.door_state.vanilla_resnet.factory import (
    InferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class DoorClassifierBase(ABC):
    DOOR_POLYGON_PAD = 30

    def __init__(
        self,
        camera_uuid: str,
        model_path: str,
        config: dict,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ):

        self.camera_uuid = camera_uuid
        self.doors = None
        self.config = config
        self.gpu_runtime = gpu_runtime
        self.triton_server_url = triton_server_url
        self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def predict(self, frame):
        raise NotImplementedError("Predict not implemented")

    def crop_and_recolor_door(self, frame, door):
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

    def pad_polygon(self, door) -> Polygon:
        """Padding the polygon with a value

        Args:
            door (dict): door dictionary

        Returns:
            Polygon: Padded polygon
        """
        door_polygon = copy.deepcopy(door.polygon)
        return Polygon(
            vertices=[
                Point(
                    max(
                        door_polygon.get_top_left().x - self.DOOR_POLYGON_PAD,
                        0,
                    ),
                    max(
                        door_polygon.get_top_left().y - self.DOOR_POLYGON_PAD,
                        0,
                    ),
                ),
                Point(
                    door_polygon.get_top_right().x + self.DOOR_POLYGON_PAD,
                    max(
                        door_polygon.get_top_right().y - self.DOOR_POLYGON_PAD,
                        0,
                    ),
                ),
                Point(
                    door_polygon.get_bottom_right().x + self.DOOR_POLYGON_PAD,
                    door_polygon.get_bottom_right().y + self.DOOR_POLYGON_PAD,
                ),
                Point(
                    max(
                        door_polygon.get_bottom_left().x
                        - self.DOOR_POLYGON_PAD,
                        0,
                    ),
                    door_polygon.get_bottom_left().y + self.DOOR_POLYGON_PAD,
                ),
            ]
        )


class ResnetClassifier(DoorClassifierBase):
    def load_model(self, model_path):
        self.model = torch.jit.load(model_path).eval().float().cuda()

    def predict(self, frame):
        actors = []
        if self.doors is None:
            camera_config = CameraConfig(
                self.camera_uuid, frame.shape[0], frame.shape[1]
            )
            self.doors = camera_config.doors

        for track_id, door in enumerate(self.doors):
            # Crop Door Polygon
            cropped_image_rgb = self.crop_and_recolor_door(frame, door)

            resized_image_rgb = cv2.resize(
                cropped_image_rgb, (224, 224)
            ).transpose(2, 0, 1)
            resized_image_rgb = np.ascontiguousarray(resized_image_rgb)
            resized_image_rgb = (
                torch.unsqueeze(torch.from_numpy(resized_image_rgb), 0)
                .float()
                .cuda()
            )

            resized_image_rgb /= 255.0
            # this is standard ImageNet normalization
            transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            normalized_image = transform(resized_image_rgb)

            prediction = torch.softmax(self.model(normalized_image)[0], dim=0)
            (
                closed_probability,
                open_probability,
                partially_open_probability,
            ) = list(prediction)

            vocabulary = {"closed": 0, "open": 1, "partially_open": 2}
            index = int(torch.argmax(prediction))

            open_door = index == vocabulary["open"]
            partially_open_door = index == vocabulary["partially_open"]

            door_state = DoorState.FULLY_CLOSED
            if open_door:
                door_state = DoorState.FULLY_OPEN
            elif partially_open_door:
                door_state = DoorState.PARTIALLY_OPEN

            # Bring it to cpu memory.
            open_probability = float(open_probability)
            partially_open_probability = float(partially_open_probability)
            closed_probability = float(closed_probability)

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
            door_polygon = self.pad_polygon(door)

            actor_id = get_actor_id_from_actor_category_and_track_id(
                track_id, ActorCategory.DOOR
            )

            actors.append(
                Actor(
                    category=ActorCategory.DOOR,
                    track_id=actor_id,
                    polygon=door_polygon,
                    manual=False,
                    confidence=open_probability + partially_open_probability,
                    door_state=door_state,
                    door_orientation=door_orientation,
                    door_state_probabilities=[
                        open_probability,
                        partially_open_probability,
                        closed_probability,
                    ],
                    track_uuid=get_track_uuid(
                        camera_uuid=self.camera_uuid,
                        unique_identifier=str(door.door_id),
                        category=ActorCategory.DOOR,
                    ),
                )
            )

        return actors


class VanillaResnetClassifier(DoorClassifierBase):
    """Resnet classifier for door trained using vanilla_resnet.py

    Args:
        DoorClassifierBase (_type_): Base door classifier for interfaces
    """

    def load_model(self, model_path) -> None:
        inference_backend_type = InferenceBackendType[
            self.config.get("inference_provider_type", "torchscript").upper()
        ]
        self.inference_provider = InferenceProviderFactory(
            local_inference_provider_type=inference_backend_type,
            gpu_runtime=self.gpu_runtime,
            triton_server_url=self.triton_server_url,
        ).get_inference_provider(
            model_path=model_path,
            camera_uuid=self.camera_uuid,
            config=self.config.get(
                "inference_config",
                {
                    "state2class": {
                        "closed": 0,
                        "open": 1,
                        "partially_open": 2,
                    },
                    "runtime_preprocessing_transforms": {
                        "bgr2rgb": True,
                    },
                    "postprocessing_transforms": {
                        "padding": self.DOOR_POLYGON_PAD,
                    },
                },
            ),
        )

    def predict(self, frame) -> List:
        """Predict state of doors in the frame

        Args:
            frame (_type_): Frame to do prediction on

        Returns:
            List: Actors of door with state
        """
        nhwc_input = torch.unsqueeze(torch.from_numpy(frame), 0)
        return self.inference_provider.process(nhwc_input)[0]


class KNNClassifier(DoorClassifierBase):
    NUM_NEIGHBORS = 5
    NUM_JOBS = -1
    DIST_THRESH = 2000
    RESOLUTION = 32

    def load_model(self, model_path):
        self.model = KNeighborsClassifier(
            n_neighbors=self.NUM_NEIGHBORS, n_jobs=self.NUM_JOBS
        )

        # Load the model
        with open(model_path, "rb") as model_file:
            self.model = pickle.load(model_file)  # trunk-ignore(bandit/B301)

    def image_to_feature_vector(self, image):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        sz = (self.RESOLUTION, self.RESOLUTION)
        return cv2.resize(image, sz).flatten()

    def predict(self, frame):
        actors = []
        if self.doors is None:
            camera_config = CameraConfig(
                self.camera_uuid, frame.shape[0], frame.shape[1]
            )
            self.doors = camera_config.doors

        for track_id, door in enumerate(self.doors):

            # Crop Door Polygon
            cropped_image_rgb = self.crop_and_recolor_door(frame, door)

            # Resize Image
            resized_image_rgb = self.image_to_feature_vector(cropped_image_rgb)

            # Predict Door State
            prediction = self.model.predict([resized_image_rgb])[0]

            open_door = prediction == "open"
            partially_open_door = prediction == "partially_open"

            door_state = DoorState.FULLY_CLOSED
            if open_door:
                door_state = DoorState.FULLY_OPEN
            elif partially_open_door:
                door_state = DoorState.PARTIALLY_OPEN

            # Check distance
            distance = self.model.kneighbors(
                [resized_image_rgb], n_neighbors=1
            )[0]
            logger.debug(
                f"Distance: {distance}\n Is too far: {distance[0] > self.DIST_THRESH}\n Door State: {door_state}"
            )

            # Set probabilities
            # TODO: update to using distance threshold
            open_probability = 0.0
            partially_open_probability = 0.0
            closed_probability = 0.0

            if door_state == DoorState.FULLY_OPEN:
                open_probability = 1.0
            elif door_state == DoorState.PARTIALLY_OPEN:
                partially_open_probability = 1.0
            else:
                closed_probability = 1.0

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
            door_polygon = self.pad_polygon(door)

            actors.append(
                Actor(
                    category=ActorCategory.DOOR,
                    track_id=track_id,
                    polygon=door_polygon,
                    manual=False,
                    confidence=open_probability + partially_open_probability,
                    door_state=door_state,
                    door_orientation=door_orientation,
                    door_state_probabilities=[
                        open_probability,
                        partially_open_probability,
                        closed_probability,
                    ],
                    track_uuid=get_track_uuid(
                        camera_uuid=self.camera_uuid,
                        unique_identifier=str(door.door_id),
                        category=ActorCategory.DOOR,
                    ),
                )
            )
        return actors


class DoorStateClassifier:
    def __init__(
        self,
        camera_uuid,
        model_path,
        model_type,
        config,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ):
        classifier_type_map = {
            "resnet": ResnetClassifier,
            "knn": KNNClassifier,
            "vanilla_resnet": VanillaResnetClassifier,
        }
        self.classifier = classifier_type_map[model_type](
            camera_uuid,
            model_path=model_path,
            config=config,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        )

    def __call__(self, frame):
        return self.classifier.predict(frame)
