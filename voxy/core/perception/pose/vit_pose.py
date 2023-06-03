#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import typing

import numpy as np
import torch

from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Pose, RectangleXYXY
from core.structs.frame import Frame
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.human_keypoint_detection_2d.vitpose.factory import (
    ViTPoseInferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class ViTPoseModel:
    def __init__(
        self,
        model_path: str,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ) -> None:
        """
        Constructor for VestClassifier

        Args:
            model_path (str): path to the model
            prediction_to_class (dict): the mapping from prediction, to the class index

        Raises:
            ValueError: If the model type is not recognized
        """
        self.padding = 30
        inference_provider_factory = ViTPoseInferenceProviderFactory(
            local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        )
        self.inference_provider = (
            inference_provider_factory.get_inference_provider(
                model_path=model_path,
                padding=self.padding,
            )
        )

    def __call__(self, frame: np.array, frame_struct: Frame) -> Frame:
        actors_to_be_processed = [
            actor
            for actor in frame_struct.actors
            if not self.filter_actor(actor)
        ]

        if len(actors_to_be_processed) <= 0:
            return Frame
        actors_xyxy = [
            torch.tensor(RectangleXYXY.from_polygon(actor.polygon).to_list())
            for actor in actors_to_be_processed
        ]
        pose_keypoints, confidences = self.inference_provider.process(
            actors_xyxy, torch.from_numpy(frame)
        )
        poses = self.post_process_predictions(pose_keypoints, confidences)
        for actor, pose in zip(actors_to_be_processed, poses):
            actor.pose = pose
        return frame_struct

    def filter_actor(self, actor: Actor) -> bool:
        """
        Filters actors based on category

        Args:
            actor (Actor): the actor

        Returns:
            bool: True if actor should be filtered out
        """
        return actor.category != ActorCategory.PERSON

    def post_process_predictions(
        self,
        pose_keypoints: np.ndarray,
        confidences: np.ndarray,
    ) -> typing.List[Pose]:
        """
        Postprocesses batch of predicted outputs to return softmax output

        Args:
            pose_keypoints (dict): the mapping from prediction, to the class index
            confidences (np.ndarray): confidence scores for each class

        Returns:
            torch.Tensor: boolean tensor if ViT model classifies image as positive
        """
        return [
            Pose.from_inference_results(poses.tolist(), confidences.tolist())
            for poses, confidences in zip(pose_keypoints, confidences)
        ]
