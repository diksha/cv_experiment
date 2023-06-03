#
# Copyright 2020-2022 Voxel Labs, Inc.
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
import unittest

import numpy as np
import torch

from core.perception.detector_tracker.utils import letterbox
from core.structs.actor import ActorCategory
from lib.ml.inference.tasks.object_detection_2d.yolov5.utils import (
    get_inference_output_shape,
    post_process_prediction,
    preprocess_image,
    transform_detections,
)
from third_party.byte_track.utils import postprocess


class YOLOInferenceUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")
        self.image_shape = (32, 32)
        self.input_shape = (32, 32)  # Factors of 32
        if self.input_shape[0] % 32 != 0 or self.input_shape[1] % 32 != 0:
            raise RuntimeError("Input shape values must be a factor of 32!")
        rgb_range = 255
        n_color_channels = 3
        self.image = np.random.randint(
            rgb_range,
            size=(self.image_shape[0], self.image_shape[1], n_color_channels),
            dtype=np.uint8,
        )
        self.classes = {0: ActorCategory.PERSON, 1: ActorCategory.PIT}
        n_anchors, n_anchor_points = get_inference_output_shape(
            self.input_shape, len(self.classes)
        )
        self.detection = torch.rand(1, n_anchors, n_anchor_points)
        # Legacy value
        self.nms_thresh = 0.7
        self.confidence = 0.001
        self.device = torch.device("cpu")

    def _legacy_preprocess_image(
        self, image: np.ndarray, input_shape: tuple
    ) -> tuple:
        """Legacy YOLO inference framework image preprocessing step
        Args:
            image (np.ndarray): image to preprocess
            new_image_size (tuple): image size to resize to, normally factor of 32
        Returns:
            tuple: contains transformed image tensor, offset, and scale
        """
        legacy_image, scale, offset = letterbox(
            image, new_shape=input_shape, auto=False
        )
        # BGR to RGB
        legacy_image = legacy_image[:, :, ::-1].transpose(2, 0, 1)
        legacy_image = np.ascontiguousarray(legacy_image)
        legacy_image = (
            torch.unsqueeze(torch.from_numpy(legacy_image), 0).float().cpu()
        )
        legacy_image /= 255.0
        return legacy_image, offset, scale

    def _legacy_transform_detections(
        self, detections: torch.Tensor, offset: tuple, scale: tuple
    ) -> torch.Tensor:
        """transform_detections.

        this transforms the output of the detector into the resized image
        since inference was done on the resized image


        Args:
            detections (torch.Tensor): this is the raw detection output
            offset (tuple): the offset (x, y) in pixels of the detection
            scale (tuple): the scale of the resized image (x, y)

        Returns:
            torch.Tensor: the raw tensor output with the resized detections
        """
        # we just apply the transformation to the bounding box
        # transformations happen in place
        offset_x, offset_y = offset
        scale_x, scale_y = scale
        detections[:, :, 0] -= offset_x
        detections[:, :, 0] /= scale_x
        detections[:, :, 2] /= scale_x

        detections[:, :, 1] -= offset_y
        detections[:, :, 1] /= scale_y
        detections[:, :, 3] /= scale_y
        return detections

    def _legacy_post_process_prediction(
        self, prediction: torch.Tensor
    ) -> dict:
        """post_process_prediction

        from a raw input tensor, this produces the class indexed set of bounding box
        predictions with their confidences

        Args:
            prediction (torch.Tensor): raw output tensor of the yolo model

        Returns:
            dict: dictionary indexed by the class label
        """
        inference_dimension = 2
        bounding_box_dim = 4
        n_inference = prediction.size()[inference_dimension]
        n_classes = n_inference - bounding_box_dim - 1
        _, _, class_confidences = torch.split(
            prediction, [bounding_box_dim, 1, n_classes], 2
        )
        class_labels = torch.argmax(class_confidences, inference_dimension)
        output_observations = {}

        def class_is_predicted(index: int) -> bool:
            """Deterimine if class is predicted
            Args:
                index (int): index of prediction
            Returns:
                bool: if index prediction is an actual class
            """
            return index < n_classes

        for class_id, actor_category in self.classes.items():
            if not class_is_predicted(class_id):
                continue

            class_prediction = prediction[class_labels == class_id]
            # there is an issue with the way bytetrack uses it's detection probability
            # so we just dump the correct one in
            class_prediction[:, 5] = class_prediction[:, 5 + class_id]
            values = class_prediction[:, :6]
            post_processed = postprocess(
                values.view((1, values.size()[0], values.size()[1])),
                1,
                self.confidence,
                self.nms_thresh,
            )
            observation = post_processed[0]
            #  make the observation an empty tensor if no bounding boxes come through
            if observation is None:
                observation = torch.empty(0, n_inference)
            output_observations[actor_category] = observation
        return output_observations

    def test_preprocessing_and_postprocessing(self) -> None:
        """Test image preprocessing for YOLO inference"""
        (
            legacy_image_preprocess,
            legacy_offset,
            legacy_scale,
        ) = self._legacy_preprocess_image(
            copy.deepcopy(self.image),
            self.input_shape,
        )
        image_preprocess, offset, scale = preprocess_image(
            torch.unsqueeze(torch.tensor(copy.deepcopy(self.image)), 0),
            self.input_shape,
            self.device,
        )
        self.assertTrue(torch.equal(legacy_image_preprocess, image_preprocess))
        self.assertEqual(legacy_offset, offset)
        self.assertEqual(legacy_scale, scale)

        legacy_detection_transform = self._legacy_transform_detections(
            copy.deepcopy(self.detection), legacy_offset, legacy_scale
        )
        detection_transform = transform_detections(
            copy.deepcopy(self.detection), offset, scale
        )
        self.assertTrue(
            torch.equal(legacy_detection_transform, detection_transform)
        )

        legacy_observation = self._legacy_post_process_prediction(
            legacy_detection_transform
        )
        observation = post_process_prediction(
            detection_transform, self.classes, self.confidence, self.nms_thresh
        )[0]
        for legacy_observation_classes, observation_classes in zip(
            legacy_observation.keys(), observation.keys()
        ):
            self.assertEqual(legacy_observation_classes, observation_classes)
        for legacy_observation_pred, observation_pred in zip(
            legacy_observation.values(), observation.values()
        ):
            self.assertTrue(
                torch.equal(legacy_observation_pred, observation_pred)
            )
