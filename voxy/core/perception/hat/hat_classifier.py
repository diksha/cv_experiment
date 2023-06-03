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
from typing import Optional, Union

import numpy as np
import torch

from core.structs.actor import Actor, ActorCategory, HeadCoveringType
from core.structs.attributes import RectangleXYWH, RectangleXYXY
from core.structs.frame import Frame
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.ppe.vit_image_classification.factory import (
    VITImageClassificationInferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class HatClassifier:
    MIN_CLASSIFICATION_CONFIDENCE = 0.5

    def __init__(
        self,
        model_path: str,
        prediction_to_class: dict,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
        is_classification_by_detection: bool = False,
        min_actor_pixel_area: Union[int, None] = None,
        head_covering_type: Optional[
            HeadCoveringType
        ] = HeadCoveringType.HARD_HAT,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._min_actor_pixel_area = min_actor_pixel_area
        self.is_classification_by_detection = is_classification_by_detection

        if is_classification_by_detection:
            raise ValueError(
                "Currently only ViT hard hat classification is supported"
            )

        self.inference_provider = (
            VITImageClassificationInferenceProviderFactory(
                local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
                gpu_runtime=gpu_runtime,
                triton_server_url=triton_server_url,
            ).get_inference_provider(
                model_path=model_path,
                feature_extractor_weights="google/vit-base-patch16-224-in21k",
                preprocessing_padding=2,
            )
        )
        self.prediction_to_class = prediction_to_class
        self.head_covering_type = head_covering_type

    def __call__(self, frame_struct: Frame, frame: np.array) -> Frame:
        """Given frame struct filters the actor by checking the actor confidence and
        min_pixel_area. Processes the actors and frames which is then passed to the hat
        classifier to make predictions

        Args:
            frame_struct (Frame): frame struct contains all the info as actors present
            frame (np.array): Frame of the video sequence

        Raises:
            ValueError: If the model is not ViT

        Returns:
            Frame: frame_struct
        """
        actors_to_be_processed = [
            actor
            for actor in frame_struct.actors
            if not self.filter_actor(actor)
        ]

        if len(actors_to_be_processed) <= 0:
            return frame_struct

        actors_xyxy = [
            torch.tensor(RectangleXYXY.from_polygon(actor.polygon).to_list())
            for actor in actors_to_be_processed
        ]
        predictions = self.inference_provider.process(
            actors_xyxy, torch.from_numpy(frame)
        )
        is_hat_predictions = torch.logical_not(
            (
                torch.max(predictions, dim=1).indices
                == self.prediction_to_class["HAT"]
            )
            * (
                (torch.max(predictions, dim=1)).values
                > self.MIN_CLASSIFICATION_CONFIDENCE
            )
        )
        for idx, actor in enumerate(actors_to_be_processed):
            actor.is_wearing_hard_hat = is_hat_predictions[idx].item()
            actor.head_covering_type = (
                self.head_covering_type
                if actor.is_wearing_hard_hat
                else HeadCoveringType.BARE_HEAD
            )

        return frame_struct

    def filter_actor(self, actor: Actor) -> bool:
        """Filters actors which do not satisfy the minimum confidence scores
        and min_actor_pixel_area

        Args:
            actor (Actor): person detected in the frame

        Returns:
            bool: True if the actor regions satisfies all criteria
        """
        actor_bbox = RectangleXYWH.from_polygon(actor.polygon)
        return (
            actor.category != ActorCategory.PERSON
            or actor.confidence < 0.25
            or (
                self._min_actor_pixel_area is not None
                and actor_bbox.w * actor_bbox.h < self._min_actor_pixel_area
            )
        )
