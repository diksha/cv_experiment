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

import numpy as np
import torch
from sortedcontainers import SortedDict

from core.execution.utils.graph_config_utils import (
    get_gpu_runtime_from_graph_config,
)
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.structs.actor import ActorCategory
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.object_detection_2d.yolov5.factory import (
    InferenceProviderFactory as YOLOv5InferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class YoloDetector:
    def __init__(
        self,
        inference_provider_typed: InferenceBackendType,
        model_path: str,
        input_shape: tuple,
        classes: dict,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ) -> None:
        self.inference_provider = YOLOv5InferenceProviderFactory(
            local_inference_provider_type=inference_provider_typed,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        ).get_inference_provider(
            model_path=model_path,
            input_shape=input_shape,
            class_map=classes,
        )

    @classmethod
    def from_config(
        cls, config: dict, perception_runner_context: PerceptionRunnerContext
    ) -> "YoloDetector":
        """
        Generates yolo detector from config

        Args:
            config (dict): the config from graph config
            perception_runner_context (PerceptionRunnerContext): context object

        Returns:
            YoloDetector: _description_
        """
        height = config["perception"]["detector_tracker"]["height"]
        width = config["perception"]["detector_tracker"]["width"]
        input_shape = (height, width)
        model_path = config["perception"]["detector_tracker"]["model_path"]
        # get the actor categories
        actor_to_class = config["perception"]["detector_tracker"][
            "actor2class"
        ]
        classes = SortedDict(
            {
                class_id: getattr(ActorCategory, category)
                for category, class_id in actor_to_class.items()
            }
        )
        inference_provider_type = InferenceBackendType[
            config["perception"]["detector_tracker"]
            .get("inference_provider_type", "trt")
            .upper()
        ]
        detector = YoloDetector(
            inference_provider_type,
            model_path,
            input_shape,
            classes,
            gpu_runtime=get_gpu_runtime_from_graph_config(config),
            triton_server_url=perception_runner_context.triton_server_url,
        )
        return detector

    def get_input_shape(self):
        return tuple(self.inference_provider.input_shape[0].tolist())

    def get_actor_categories(self):
        return set(self.inference_provider.class_map.values())

    def predict(self, image: np.array) -> dict:
        """predict.
        TODO: Pass in NHWC input instead of image, return list of dictionary from predict

        Generates a set of predictions by observation class. Format is:
        {
            CategoryType: torch.Tensor,
            ...
        }
        The format of the tensor is [bbox predictions | bbox confidence | class confidences]
        (typical yolo format)

        Args:
            image (np.array): raw input image

        Returns:
            dict: set of predictions as indexed by their category
        """
        nhwc_input = torch.unsqueeze(torch.from_numpy(image), 0)
        return self.inference_provider.process(nhwc_input)[0]
