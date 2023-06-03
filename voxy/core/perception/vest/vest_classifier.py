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
from typing import Union

import numpy as np
import torch

from core.structs.actor import Actor, ActorCategory
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


class VestClassifier:
    def __init__(
        self,
        model_path: str,
        prediction_to_class: dict,
        classification_model_type: str,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
        min_actor_pixel_area: Union[int, None] = None,
    ) -> None:
        """
        Constructor for VestClassifier

        Args:
            model_path (str): path to the model
            prediction_to_class (dict): the mapping from prediction, to the class index
            classification_model_type (str): the classification model type
            min_actor_pixel_area (Union[int, None], optional): The minimum
                           amount of pixels to classify.
                           Defaults to None.

        Raises:
            ValueError: If the model type is not recognized
        """
        self._min_actor_pixel_area = min_actor_pixel_area
        self.prediction_to_class = prediction_to_class

        self._model_type = classification_model_type
        if self._model_type == "Transformers":
            self.inference_provider = VITImageClassificationInferenceProviderFactory(
                local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
                gpu_runtime=gpu_runtime,
                triton_server_url=triton_server_url,
            ).get_inference_provider(
                model_path=model_path,
                feature_extractor_weights="google/vit-base-patch16-224-in21k",
                preprocessing_padding=0,
            )
        else:
            raise ValueError("Model Type is Not Recognized!")

    def __call__(self, frame_struct: Frame, frame: np.array) -> Frame:

        actors_to_be_processed = [
            actor
            for actor in frame_struct.actors
            if not self.filter_actor(actor)
        ]

        if len(actors_to_be_processed) > 0:
            if self._model_type == "Transformers":
                actors_xyxy = [
                    torch.tensor(
                        RectangleXYXY.from_polygon(actor.polygon).to_list()
                    )
                    for actor in actors_to_be_processed
                ]
                predictions = self.inference_provider.process(
                    actors_xyxy, torch.from_numpy(frame)
                )
                is_vest_predictions = self.postprocess_predictions(
                    predictions, self.prediction_to_class
                )
                for idx, actor in enumerate(actors_to_be_processed):
                    actor.is_wearing_safety_vest = is_vest_predictions[
                        idx
                    ].item()

            else:
                raise ValueError("Model Type is Not Recognized!")

        return frame_struct

    def filter_actor(self, actor: Actor) -> bool:
        actor_bbox = RectangleXYWH.from_polygon(actor.polygon)
        return (
            actor.category != ActorCategory.PERSON
            or actor.confidence < 0.25
            or (
                self._min_actor_pixel_area is not None
                and actor_bbox.w * actor_bbox.h < self._min_actor_pixel_area
            )
        )

    @classmethod
    def postprocess_predictions(
        cls, model_output: torch.Tensor, prediction_to_class: dict
    ) -> torch.Tensor:
        """Postprocesses batch of predicted outputs to return softmax output
        Args:
            model_output (torch.Tensor): output of ViT model
            prediction_to_class (dict): the mapping from prediction, to the class index
        Returns:
            torch.Tensor: boolean tensor if ViT model classifies image as positive
        """
        batched_predictions = torch.softmax(model_output, dim=1)
        return (
            torch.argmax(batched_predictions, dim=1)
            == prediction_to_class["VEST"]
        )
