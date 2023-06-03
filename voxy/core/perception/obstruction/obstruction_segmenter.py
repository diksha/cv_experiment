#
# Copyright 2021-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage.draw import polygon2mask

from core.execution.utils.graph_config_utils import (
    get_gpu_runtime_from_graph_config,
)
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.incidents.utils import CameraConfig
from core.perception.segmenter_tracker.tracker import SegmenterTracker
from core.perception.utils.segmentation_utils import (
    post_process_segment,
    preprocess_segmentation_input,
    update_segment_class_ids,
)
from core.structs.actor import Actor, ActorCategory
from core.structs.frame import FrameSegmentCategory
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.segmentation.unet_v1.factory import (
    UnetV1InferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class ObstructionSegmenter:
    """
    Obstruction segmenter class
    """

    def __init__(
        self,
        model_path: str,
        min_run_time_difference_ms: int,
        min_pixel_size: int,
        camera_uuid: str,
        frame_segments_category_to_class_id: dict,
        post_process_enabled: bool,
        ignore_actor_categories: list,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ):
        """Obstruction segmenter class constructor

        Args:
            model_path (str): the path to the model
            min_pixel_size (int): threshold remove small connected components
            camera_uuid (str): the camera uuid
            frame_segments_category_to_class_id (dict): mapping between FrameSegmentCategory
            post_process_enabled (bool): perform post process on the obstruction segment
            ignore_actor_categories (list): list of actor names to ignore from obstruction mask
            enum classes and the class ids the model was trained with
        """
        self.inference_provider = UnetV1InferenceProviderFactory(
            local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        ).get_inference_provider(model_path)
        self.min_run_time_difference_ms = min_run_time_difference_ms
        self._camera_uuid = camera_uuid
        self._no_obstruction_mask = None
        self.min_pixel_size = min_pixel_size
        self.frame_segments_category_to_class_id = (
            frame_segments_category_to_class_id
        )
        self.post_process_enabled = post_process_enabled
        self._tracker = SegmenterTracker(
            camera_uuid=self._camera_uuid,
            category="OBSTRUCTION",
            min_pixel_size=self.min_pixel_size,
        )
        self._last_run_timestamp_ms = 0
        self.ignore_actor_categories = [
            ActorCategory[category] for category in ignore_actor_categories
        ]

    @classmethod
    def from_config(cls, config: dict) -> object:
        """
        Create an ObstructionSegmenter class from config

        Args:
            config: a dictionary containing all relevant information to create a config

        Returns:
            An object of ObstructionSegmenter class initialized with values derived from the config
        """
        model_path = config["perception"]["obstruction_segmenter"][
            "model_path"
        ]
        min_run_time_difference_ms = config["perception"][
            "obstruction_segmenter"
        ].get("min_run_time_difference_ms", 1000)
        min_pixel_size = config["perception"]["obstruction_segmenter"].get(
            "min_pixel_size", 0
        )
        camera_uuid = config["camera_uuid"]
        # get the actor categories
        frame_segment_to_class = config["perception"]["obstruction_segmenter"][
            "frame_segment2class"
        ]
        post_process_enabled = (
            config["perception"]
            .get("obstruction_segmenter", {})
            .get("post_process_enabled")
        )
        ignore_actor_categories = (
            config["perception"]
            .get("obstruction_segmenter", {})
            .get("ignore_actor_categories")
        )

        frame_segments_category_to_class_id = {
            getattr(FrameSegmentCategory, category): class_id
            for category, class_id in frame_segment_to_class.items()
        }
        return ObstructionSegmenter(
            model_path,
            min_run_time_difference_ms,
            min_pixel_size,
            camera_uuid,
            frame_segments_category_to_class_id,
            post_process_enabled,
            ignore_actor_categories,
            gpu_runtime=get_gpu_runtime_from_graph_config(config),
            triton_server_url=PerceptionRunnerContext().triton_server_url,
        )

    def _set_obsturction_mask(self, image_shape: tuple):
        """Set the no obstruction mask from camera config

        Args:
            image_shape (tuple): shape of input image
        """
        camera_config = CameraConfig(
            self._camera_uuid, image_shape[0], image_shape[1]
        )
        obstruction_data = camera_config.no_obstruction_regions
        self._no_obstruction_mask = np.zeros((image_shape[0], image_shape[1]))
        for no_obstruction_zone in obstruction_data:
            no_obstruction_zone = [
                [a["x"], a["y"]]
                for a in no_obstruction_zone.to_dict()["vertices"]
            ]
            polygon_points = np.fliplr(np.array(no_obstruction_zone))
            self._no_obstruction_mask = (
                polygon2mask((image_shape[0], image_shape[1]), polygon_points)
                + self._no_obstruction_mask
            )

    def segment_image(self, frame: np.ndarray) -> np.ndarray:
        """Segment image

        Args:
            frame (np.ndarray): input image

        Returns:
            np.ndarray: segmented image
        """
        img_tensor = preprocess_segmentation_input(frame)
        segmented_image = self.inference_provider.process(img_tensor)
        segmented_image = segmented_image.sigmoid()
        segmented_image = (segmented_image > 0.5).float()
        segmented_image = np.squeeze(segmented_image[0].cpu().numpy(), axis=0)
        # convert model class to the Voxel's class ids
        segmented_image = update_segment_class_ids(
            segmented_image, self.frame_segments_category_to_class_id
        )
        # resize segmented image to size of input image
        segmented_image = cv2.resize(
            segmented_image,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return segmented_image

    def calculate_obstruction_segment(
        self, segmented_image: np.ndarray, actor_mask: np.ndarray
    ) -> np.ndarray:
        """Calculate obstruction segment from frame and segmented image

        Args:
            segmented_image (np.ndarray): segmented image
            actor_mask (np.ndarray): mask of actors to ignore
        Returns:
            np.ndarray: obstruction segment
        """
        actor_mask = (self._no_obstruction_mask * actor_mask).astype(
            np.float32
        )
        obstruction_segment = self._no_obstruction_mask.astype(np.float32) - (
            segmented_image * self._no_obstruction_mask
        ).astype(np.float32)
        obstruction_segment = obstruction_segment - actor_mask
        obstruction_segment[
            obstruction_segment == 1
        ] = FrameSegmentCategory.OBSTRUCTION.value

        return obstruction_segment

    def segment_image_at_time_interval(
        self,
        frame: np.ndarray,
        current_timestamp_ms: int,
        actor_mask: np.ndarray,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Segment image and get actors bounding boxes

        Args:
            frame (np.ndarray): input image
            current_timestamp_ms (int): current timestamp in milliseconds
            actor_mask (np.ndarray): mask of actors to ignore
        Returns:
            Tuple[np.ndarray, torch.Tensor]: segmented image, actors bounding boxes
        """

        if current_timestamp_ms == 0 or (
            current_timestamp_ms - self._last_run_timestamp_ms
            >= self.min_run_time_difference_ms
        ):
            segmented_image = self.segment_image(frame)
            obstruction_segment = self.calculate_obstruction_segment(
                segmented_image, actor_mask
            )
            obstruction_segment, actors_bbox = post_process_segment(
                obstruction_segment,
                FrameSegmentCategory.OBSTRUCTION,
                self.min_pixel_size,
            )
            segmented_image[
                obstruction_segment == FrameSegmentCategory.OBSTRUCTION.value
            ] = FrameSegmentCategory.OBSTRUCTION.value
            return segmented_image, actors_bbox
        return None, None

    def get_actor_masks(self, actors: Actor, frame_shape: tuple) -> np.ndarray:
        """Get binary mask using union of actors to ignore

        Args:
            actors (Actor): actors to ignore
            frame_shape (tuple): frame shape

        Returns:
            np.ndarray: binary mask using union of actors
        """
        actors_mask = np.zeros((frame_shape[0], frame_shape[1]))
        for actor in actors:
            if actor.category in self.ignore_actor_categories:
                actor_polygon = [
                    [a["x"], a["y"]]
                    for a in actor.polygon.to_dict()["vertices"]
                ]
                # the flip left right just permutes the x and y coodinates,
                # for the mask calculation
                actor_polygon = np.fliplr(np.array(actor_polygon))
                actors_mask = (
                    polygon2mask(
                        (frame_shape[0], frame_shape[1]), actor_polygon
                    )
                    + actors_mask
                )
        return actors_mask

    def __call__(
        self,
        frame: np.ndarray,
        current_timestamp_ms: int,
        actors: Actor,
    ) -> Tuple[List[Actor], Optional[np.ndarray]]:
        """
        Runs inference on segmented image and tracks the detected regions

        Args:
            frame (np.ndarray): the raw input image
            current_timestamp_ms (int): current timestamp in milliseconds
            actors (Actor): Actors from frame struct
        Returns:
            Tuple[List[Actor], Optional[np.ndarray]]: the list of tracked
                             actors and an the optional segmentation mask
        """
        actor_mask = self.get_actor_masks(actors, frame.shape)
        if self._no_obstruction_mask is None:
            self._set_obsturction_mask(frame.shape)
        segmentation_mask, actors_bbox = self.segment_image_at_time_interval(
            frame, current_timestamp_ms, actor_mask
        )
        actors = (
            self._tracker.track(False, actors_bbox, segmentation_mask.shape)
            if segmentation_mask is not None
            else self._tracker.track(True, actors_bbox, None)
        )
        return actors, segmentation_mask
