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
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from core.perception.common.utils import reshape_polygon_crop_to_square
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import RectangleXYWH
from core.structs.frame import Frame
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.ergonomic_carry_object.res_net34.factory import (
    Resnet34InferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class CarryObjectClassifier:
    ACTOR_CONFIDENCE_THRESH = 0.25

    def __init__(
        self,
        model_path: str,
        prediction2class: dict,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
        score_cutoff: float = 0.8,
        min_actor_pixel_area: Union[int, None] = None,
    ) -> None:

        self.score_cutoff = score_cutoff
        self._min_actor_pixel_area = min_actor_pixel_area
        self._predicition2class = prediction2class

        # Load model
        factory = Resnet34InferenceProviderFactory(
            local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        )
        self.device = factory.get_device()
        self.inference_provider = factory.get_inference_provider(
            model_path=model_path
        )

    def __call__(self, frame_struct: Frame, frame_bgr: np.array) -> Frame:
        """Calls the CarryObject model with pre-processing

        Args:
            frame_struct (Frame): current perception frame struct
            frame_bgr (np.array): image to process

        Returns:
            Frame: updated perception frame struct
        """
        actors_to_be_processed = [
            actor
            for actor in frame_struct.actors
            if not self.filter_actor(actor)
        ]

        batched_preprocessed_input = self.preprocess_inputs(
            actors_to_be_processed, frame_bgr
        )

        if batched_preprocessed_input is not None:
            carrying_object_predictions = self._classify_carrying_object(
                batched_preprocessed_input
            )
            for idx, actor in enumerate(actors_to_be_processed):
                actor.is_carrying_object = (
                    carrying_object_predictions[idx].item()
                    == self._predicition2class["CARRYING"]
                )

        return frame_struct

    def _classify_carrying_object(
        self, preprocessed_input: torch.Tensor
    ) -> bool:
        """Run the classifier model

        Args:
            preprocessed_input (torch.Tensor): pre-processed image

        Returns:
            bool: classifies if the actor is carrying the object
        """

        logits = self.inference_provider.process(preprocessed_input)
        prob = torch.nn.functional.softmax(logits, dim=1)
        lift_scores, lift_labels = torch.topk(prob, 1)

        return self.postprocess_model_outputs(lift_scores, lift_labels)

    def postprocess_model_outputs(
        self, lift_scores: torch.tensor, lift_labels: torch.tensor
    ) -> list:
        """Postprocess model outputs to overwrite positive predictions with
        negative (CARRYING TO NOT_CARRYING) based on prediction score. If score is not provided for
        every prediction then no postprocessing is performed.

        Args:
            lift_scores (torch.tensor): confidence score for every classification
            lift_labels (torch.tensor): list of classifications

        Returns:
            list: list of postprocessed model outputs returned in the original order
        """
        if len(lift_scores) != len(lift_labels):
            return lift_labels

        # trunk-ignore(pylint/C0200)
        for i in range(len(lift_labels)):
            if lift_scores[i] < self.score_cutoff:
                lift_labels[i] = self._predicition2class["NOT_CARRYING"]

        return lift_labels

    def filter_actor(self, actor: Actor) -> bool:
        """Filters out actors with low confidence or low size

        Args:
            actor (Actor): the actor to evaluate filtering

        Returns:
            bool: true if actor should be filtered out
        """
        actor_bbox = RectangleXYWH.from_polygon(actor.polygon)
        return (
            actor.category != ActorCategory.PERSON
            or actor.confidence < self.ACTOR_CONFIDENCE_THRESH
            or (
                self._min_actor_pixel_area is not None
                and actor_bbox.w * actor_bbox.h < self._min_actor_pixel_area
            )
        )

    def _reshape_actor_box(
        self, actor: Actor, frame: np.array
    ) -> torch.Tensor:
        """Reshapes the actor box to be a square

        Args:
            actor (Actor): Actor to extract
            frame (np.array): Image to process

        Returns:
            Image: cropped actor image in a square shape
        """
        return reshape_polygon_crop_to_square(actor.polygon, frame)

    def preprocess_inputs(
        self, actors: List[Actor], frame: np.array
    ) -> Optional[torch.Tensor]:
        """Returns correctly shaped and transformed images of all actors to evaluate.

        Args:
            actors (List[Actor]): all actors to generate images for
            frame (np.array): image to process

        Returns:
            Optional[torch.Tensor]: a list of tensors of actor iamges.
        """
        cropped_images = []
        image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
        batch_image = torch.tensor(image.numpy()[np.newaxis, ...]).to(
            self.device
        )

        # converting back to PIL image in order to apply the exact same transform
        # used during training
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        for actor in actors:
            # reshape box to a square
            from_0, to_0, from_1, to_1 = self._reshape_actor_box(actor, frame)
            crop_bgr = frame[from_0:to_0, from_1:to_1, :]

            # converting back to PIL image in order to apply the exact same transform
            # used during training
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_img_rgb = Image.fromarray(np.uint8(crop_rgb))

            inp = transform(crop_img_rgb).to(self.device)

            # add to list
            cropped_images.append(inp)

        if len(cropped_images) == 0:
            return None

        batch_image = torch.stack(cropped_images)

        return batch_image
