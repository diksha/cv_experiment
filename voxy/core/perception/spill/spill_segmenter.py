#
# Copyright 2021-2022 Voxel Labs, Inc.
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

from core.execution.utils.graph_config_utils import (
    get_gpu_runtime_from_graph_config,
)
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.perception.segmenter_tracker.tracker import SegmenterTracker
from core.perception.utils.segmentation_utils import (
    post_process_segment,
    preprocess_segmentation_input,
    update_segment_class_ids,
)
from core.structs.actor import Actor
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


class SpillSegmenter:
    """
    Spill segmenter class
    """

    def __init__(
        self,
        model_path: str,
        min_run_time_difference_ms: int,
        min_pixel_size: int,
        post_process_enabled: bool,
        frame_segments_category_to_class_id: dict,
        camera_uuid: str,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ):
        """Spill segmenter class constructor

        Args:
            model_path (str): the path to model state dictionary
            min_run_time_difference_ms (int): threshold to run the segmentation inference
            min_pixel_size (int): threshold remove small connected components
            post_process_enabled (bool): perform post process on the spill segment
            frame_segments_category_to_class_id (dict): mapping between FrameSegmentCategory
            enum classes and the class ids the model was trained with
        """
        self._camera_uuid = camera_uuid
        self.inference_provider = UnetV1InferenceProviderFactory(
            local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        ).get_inference_provider(model_path)
        self.min_run_time_difference_ms = min_run_time_difference_ms
        self.frame_segments_category_to_class_id = (
            frame_segments_category_to_class_id
        )
        self.min_pixel_size = min_pixel_size
        self.post_process_enabled = post_process_enabled
        self._tracker = SegmenterTracker(
            camera_uuid=self._camera_uuid,
            category="SPILL",
            min_pixel_size=self.min_pixel_size,
        )
        self._last_run_timestamp_ms = 0

    @classmethod
    def from_config(
        cls, config: dict, perception_runner_context: PerceptionRunnerContext
    ) -> object:
        """
        Create a SpillSegmenter class from config

        Args:
            config (dict): a dictionary containing all relevant information to create a config
            perception_runner_context (PerceptionRunnerContext): context object

        Returns:
            An object of SpillSegmenter class initialized with values derived from the config
        """
        model_path = config["perception"]["spill"]["model_path"]
        min_pixel_size = config["perception"]["spill"].get("min_pixel_size", 0)
        min_run_time_difference_ms = config["perception"]["spill"].get(
            "min_run_time_difference_ms", 0
        )
        post_process_enabled = (
            config["perception"].get("spill", {}).get("post_process_enabled")
        )

        # get the actor categories
        frame_segment_to_class = config["perception"]["spill"][
            "frame_segment2class"
        ]
        frame_segments_category_to_class_id = {
            getattr(FrameSegmentCategory, category): class_id
            for category, class_id in frame_segment_to_class.items()
        }

        camera_uuid = config["camera_uuid"]

        return SpillSegmenter(
            model_path,
            min_run_time_difference_ms,
            min_pixel_size,
            post_process_enabled,
            frame_segments_category_to_class_id,
            camera_uuid,
            gpu_runtime=get_gpu_runtime_from_graph_config(config),
            triton_server_url=perception_runner_context.triton_server_url,
        )

    def segment_image(self, frame: np.ndarray) -> np.ndarray:
        """Segment image and get spill actors

        Args:
            frame (np.ndarray): input image

        Returns:
            np.ndarray: A np.ndarray containing the segmentation result
        """
        img_tensor = preprocess_segmentation_input(frame)
        segment_index = self.inference_provider.process(img_tensor)
        segment_index = segment_index.sigmoid()
        segment_index = (segment_index > 0.5).float()
        segment_index = np.squeeze(segment_index[0].cpu().numpy(), axis=0)
        # convert model class to the Voxel's class ids
        segment_index = update_segment_class_ids(
            segment_index, self.frame_segments_category_to_class_id
        )
        segment_index = cv2.resize(
            segment_index,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return segment_index

    def segment_image_at_time_interval(
        self, frame: np.ndarray, current_timestamp_ms: int
    ) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
        """Segment image and get spill actors bounding boxes

        Args:
            frame (np.ndarray): input image
            current_timestamp_ms (int): current timestamp
        Returns:
            Tuple[np.ndarray, torch.Tensor]: A np.ndarray containing the segmentation result,
            spill ctors bounding boxes
            if and only if a time period equivalent to min_run_time_difference_ms has
            passed since the last run of the model
        """
        if current_timestamp_ms == 0 or (
            current_timestamp_ms - self._last_run_timestamp_ms
            >= self.min_run_time_difference_ms
        ):
            # set last run timestamp
            self._last_run_timestamp_ms = current_timestamp_ms

            # run inference
            segment_index = self.segment_image(frame)

            segment_index, actors_bbox = post_process_segment(
                segment_index,
                FrameSegmentCategory.SPILL,
                self.min_pixel_size,
            )
            return segment_index, actors_bbox
        return None, None

    def __call__(
        self, frame: np.ndarray, current_timestamp_ms: int
    ) -> Tuple[List[Actor], Optional[np.ndarray]]:
        """
        Runs inference on segmented image and tracks the detected regions

        Args:
            frame (np.ndarray): the raw input image
            current_timestamp_ms (int): the current timestamp of the frame

        Returns:
            Tuple[List[Actor], Optional[np.ndarray]]: the list of tracked
                             actors and an the optional segmentation mask
        """
        segmentation_mask, actors_bbox = self.segment_image_at_time_interval(
            frame, current_timestamp_ms
        )
        actors = (
            self._tracker.track(False, actors_bbox, segmentation_mask.shape)
            if segmentation_mask is not None
            else self._tracker.track(True, actors_bbox, None)
        )
        return actors, segmentation_mask
